"""
Stage 1: continuous masked autoencoding (CMAE) of the bidirectional Mamba-2 teacher.

The teacher is pretrained on the 34 non-held-out sessions of Makin and Flint without any
behavioral label: 60% of the timesteps of each 5 s segment are masked in contiguous blocks
of 10-50 steps and reconstructed in LFP space. The defaults are the settings of the paper,
which trained with 16 processes of batch 24 (effective batch 384):

    torchrun --nproc-per-node=<GPUs> scripts/pretrain.py --out output/teacher.pt

A single-process run (python scripts/pretrain.py) follows the same recipe with an effective
batch of --batch_size. Data are read from $REALM_DATA (default ./data).
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import REALMEncoder, TEACHER_REALM_KWARGS  # noqa: E402
from utils import MaskedLFPPretrainer  # noqa: E402
from utils.dataset import create_pretrain_dataloaders  # noqa: E402

# Fixed parts of the recipe
DATASETS = ['makin', 'flint']
SEGMENT_LENGTH = 500      # 5 s at 100 Hz
MASK_RATIO = 0.6
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

# torchrun sets these; a plain `python` run is a single process
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
RANK = int(os.environ.get('RANK', 0))
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
DDP_ENABLED = WORLD_SIZE > 1
IS_MAIN = RANK == 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 1: CMAE pretraining of the REALM teacher (defaults: the paper).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--out', type=Path, default=Path('output/teacher.pt'),
                   help="checkpoint path; the training history is written next to it")
    p.add_argument('--epochs', type=int, default=300, help="maximum number of epochs")
    p.add_argument('--batch_size', type=int, default=24, help="per process")
    p.add_argument('--lr', type=float, default=1.53e-3, help="peak learning rate")
    p.add_argument('--warmup_epochs', type=int, default=30,
                   help="linear warmup to the peak learning rate")
    p.add_argument('--warmup_start_factor', type=float, default=0.3,
                   help="learning rate of the first warmup epoch, as a fraction of the peak")
    p.add_argument('--lr_decay', type=float, default=0.995,
                   help="exponential decay per epoch after warmup")
    p.add_argument('--patience', type=int, default=30,
                   help="early stopping on the validation loss")
    p.add_argument('--stride', type=int, default=250,
                   help="stride between 500-sample training segments (250: 50%% overlap)")
    p.add_argument('--seed', type=int, default=42,
                   help="seed of the weights, masks, data order and train/validation split")
    p.add_argument('--num_workers', type=int, default=4, help="data-loading workers per process")
    p.add_argument('--save_every', type=int, default=0,
                   help="also save the encoder every N epochs (0: off)")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ddp_mean(x: float) -> float:
    """Average a scalar over processes, so every process takes the same decisions."""
    if not DDP_ENABLED:
        return x
    t = torch.tensor([x], device=DEVICE, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / WORLD_SIZE).item()


def log(msg):
    if IS_MAIN:
        print(msg, flush=True)


def save_checkpoint(path, state, encoder_kwargs, epoch, loss_key, loss, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'encoder_state_dict': state, 'encoder_kwargs': encoder_kwargs,
                'epoch': epoch, loss_key: loss, 'args': args}, path)


def main():
    args = parse_args()
    set_seed(args.seed)

    # ── Data: the 34 non-held-out sessions, split by session into train / validation ──
    train_loader, val_loader, n_sessions = create_pretrain_dataloaders(
        seed=args.seed, segment_length=SEGMENT_LENGTH, batch_size=args.batch_size,
        num_workers=args.num_workers, stride=args.stride, datasets=DATASETS,
        ddp=DDP_ENABLED, rank=RANK, world_size=WORLD_SIZE)

    # ── Model: bidirectional encoder + masked-reconstruction predictor ──
    # max_sessions sizes the (unused) session table; the released teacher has 34 entries.
    # The two fixed-value keys are stored as in the released teacher's configuration.
    encoder_kwargs = {**TEACHER_REALM_KWARGS, 'n_spatial_patches': 1,
                      'max_sessions': n_sessions, 'block_type': 'mamba'}
    encoder = REALMEncoder(**encoder_kwargs).to(DEVICE)
    model = MaskedLFPPretrainer(
        encoder, n_channels=96, n_bands=1, mask_ratio=MASK_RATIO,
        predictor_layers=1, predictor_expand=1, augment=True).to(DEVICE)
    if DDP_ENABLED:
        # The session table and the other tensors kept for checkpoint compatibility take no
        # gradient, hence find_unused_parameters.
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK,
                    find_unused_parameters=True)
    raw = model.module if DDP_ENABLED else model

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    n_params = sum(p.numel() for p in raw.encoder.parameters())
    log(f"Teacher encoder: {n_params:,} parameters; {n_sessions} sessions; "
        f"{WORLD_SIZE} process(es) x batch {args.batch_size}; device {DEVICE}")

    # ── Optimizer: AdamW, linear warmup from warmup_start_factor, then exponential decay ──
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            f0 = args.warmup_start_factor
            return f0 + (1.0 - f0) * (epoch + 1) / args.warmup_epochs
        return args.lr_decay ** (epoch - args.warmup_epochs)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    saved_args = {k: v for k, v in vars(args).items()
                  if k not in ('out', 'num_workers', 'save_every')}
    saved_args.update(datasets=DATASETS, segment_length=SEGMENT_LENGTH, mask_ratio=MASK_RATIO,
                      weight_decay=WEIGHT_DECAY, world_size=WORLD_SIZE)
    history_path = args.out.with_name(args.out.stem + '_history.json')

    # ── Training ──
    best_val, best_epoch, best_state, bad_epochs, history = float('inf'), 0, None, 0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if DDP_ENABLED:
            train_loader.ddp_sampler.set_epoch(epoch)
        lr = optimizer.param_groups[0]['lr']

        model.train()
        train_loss, n_batches, grad_norm = 0.0, 0, 0.0
        optimizer.zero_grad(set_to_none=True)
        for batch in train_loader:
            lfp = batch['lfp'].to(DEVICE, non_blocking=True)
            cmask = batch['channel_mask'].to(DEVICE, non_blocking=True)
            loss = model(lfp, channel_mask=cmask)['loss']
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            train_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_train = ddp_mean(train_loss / max(n_batches, 1))

        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                lfp = batch['lfp'].to(DEVICE, non_blocking=True)
                cmask = batch['channel_mask'].to(DEVICE, non_blocking=True)
                val_loss += model(lfp, channel_mask=cmask)['loss'].item()
                n_val += 1
        avg_val = ddp_mean(val_loss / max(n_val, 1))
        elapsed = time.time() - t0

        if not math.isfinite(avg_train):
            log(f"Epoch {epoch}: non-finite training loss, stopping")
            break

        is_best = avg_val < best_val
        log(f"Epoch {epoch:3d}/{args.epochs} | train {avg_train:.5f} | val {avg_val:.5f} | "
            f"lr {lr:.2e} | grad {float(grad_norm):.3f} | {elapsed:.1f}s"
            + (" | best" if is_best else ""))
        history.append({'epoch': epoch, 'train_loss': avg_train, 'val_loss': avg_val,
                        'lr': lr, 'time_s': round(elapsed, 1), 'is_best': is_best})

        if is_best:
            best_val, best_epoch, bad_epochs = avg_val, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in raw.encoder.state_dict().items()}
            if IS_MAIN:
                save_checkpoint(args.out, best_state, encoder_kwargs, epoch,
                                'best_val_loss', best_val, saved_args)
        else:
            bad_epochs += 1

        if IS_MAIN:
            if args.save_every and epoch % args.save_every == 0:
                state = {k: v.cpu().clone() for k, v in raw.encoder.state_dict().items()}
                save_checkpoint(args.out.with_name(f"{args.out.stem}_epoch{epoch:03d}.pt"),
                                state, encoder_kwargs, epoch, 'val_loss', avg_val, saved_args)
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(json.dumps(history, indent=2))

        if bad_epochs >= args.patience:
            log(f"Early stopping at epoch {epoch}")
            break

    if IS_MAIN and best_state is not None:
        save_checkpoint(args.out, best_state, encoder_kwargs, best_epoch,
                        'best_val_loss', best_val, saved_args)
    log(f"Best epoch {best_epoch}, validation loss {best_val:.5f}; saved to {args.out}")


if __name__ == '__main__':
    if DDP_ENABLED:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(LOCAL_RANK)
        DEVICE = torch.device(f'cuda:{LOCAL_RANK}')
    try:
        main()
    finally:
        if DDP_ENABLED and dist.is_initialized():
            dist.destroy_process_group()
