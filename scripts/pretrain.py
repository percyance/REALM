"""
Stage 1: MAE self-supervised pretraining of REALM BiMamba2.

Trains on Brochier (2) + DANDI:000070 (46) + DANDI:000121 (40) = 88 sessions.
No velocity labels needed -- only LFP reconstruction.

Usage:
    python -m REALM.scripts.pretrain [--epochs 200] [--seed 42]
"""

import argparse
import json
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.dataset import create_pretrain_dataloaders
from models.encoder import REALMEncoder
from models.configs import TEACHER_REALM_KWARGS
from utils.masked_pretraining import MaskedLFPPretrainerV2

# ── Config ────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_logging(output_dir):
    """Setup dual logging: console + file."""
    log_path = output_dir / "pretrain.log"
    logger = logging.getLogger("pretrain")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    fh = logging.FileHandler(log_path, mode='a')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def get_gpu_info():
    """Return GPU memory usage string."""
    if not torch.cuda.is_available():
        return ""
    alloc = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"VRAM: {alloc:.1f}/{peak:.1f}/{total:.0f} GB (alloc/peak/total)"


def main():
    parser = argparse.ArgumentParser(description="MAE Pretrain REALM BiMamba2")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=6.25e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=30,
                        help="Linear warmup epochs (CrossModalDistill: 30)")
    parser.add_argument('--lr_decay', type=float, default=0.995,
                        help="Exponential LR decay factor per epoch after warmup")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mask_ratio', type=float, default=0.6)
    parser.add_argument('--segment_length', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument('--no_compile', action='store_true',
                        help="Disable torch.compile")
    parser.add_argument('--stride', type=int, default=250,
                        help="Segment stride (default: 250 for 50%% overlap). "
                             "Use 500 for non-overlapping.")
    parser.add_argument('--spatial_patches', type=int, default=None,
                        help="Override n_spatial_patches (default: from teacher config)")
    parser.add_argument('--predictor_layers', type=int, default=1,
                        help="Number of predictor BiMamba2 layers")
    parser.add_argument('--predictor_expand', type=int, default=1,
                        help="Expand factor for predictor BiMamba2")
    parser.add_argument('--no_augment', action='store_true',
                        help="Disable data augmentation during pretraining")
    parser.add_argument('--m1_only', action='store_true',
                        help="Pretrain with M1 data only (exclude PMd/Array-B)")
    args = parser.parse_args()

    set_seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(OUTPUT_DIR)

    logger.info("=" * 70)
    logger.info("MAE Pretraining — REALM BiMamba2 (Teacher)")
    logger.info("=" * 70)
    logger.info(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"VRAM: {total_mem:.1f} GB")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Args: {vars(args)}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, n_sessions = create_pretrain_dataloaders(
        seed=args.seed,
        segment_length=args.segment_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stride=args.stride,
        m1_only=args.m1_only,
    )
    logger.info(f"Data: {n_sessions} sessions, "
                f"{len(train_loader)} train batches, "
                f"{len(val_loader)} val batches")

    # ── Model ─────────────────────────────────────────────────────────────
    encoder_kwargs = {**TEACHER_REALM_KWARGS, 'max_sessions': n_sessions}
    if args.spatial_patches is not None:
        encoder_kwargs['n_spatial_patches'] = args.spatial_patches
    encoder = REALMEncoder(**encoder_kwargs).to(DEVICE)
    sp = encoder_kwargs.get('n_spatial_patches', 1)
    bidir_str = "causal" if not encoder_kwargs.get('bidirectional', True) else "bidirectional"
    logger.info(f"Encoder: d_model={encoder_kwargs['d_model']}, "
                f"n_layers={encoder_kwargs['n_layers']}, expand={encoder_kwargs['expand']}, "
                f"spatial_patches={sp}, {bidir_str}")

    model = MaskedLFPPretrainerV2(
        encoder=encoder,
        n_channels=96, n_bands=1,
        mask_ratio=args.mask_ratio,
        mask_type='block',
        predictor_layers=args.predictor_layers,
        predictor_expand=args.predictor_expand,
        augment=not args.no_augment,
    ).to(DEVICE)
    logger.info(f"Objective: MAE (LFP reconstruction, block masking)")

    n_params = count_params(model)
    logger.info(f"Model parameters: {n_params:,}")

    # ── Performance optimizations ────────────────────────────────────────
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled TF32 + cuDNN benchmark")

    # Helper to get raw encoder (handles torch.compile wrapper)
    def get_encoder():
        m = model._orig_mod if hasattr(model, '_orig_mod') else model
        return m.encoder

    # ── Optimizer ─────────────────────────────────────────────────────────
    # Following CrossModalDistill (Erturk et al., NeurIPS 2025):
    # Linear warmup for warmup_epochs to peak lr, then exponential decay.
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        """Linear warmup then exponential decay (CrossModalDistill schedule)."""
        if epoch < args.warmup_epochs:
            # Linear warmup: 0 → 1 over warmup_epochs
            return (epoch + 1) / args.warmup_epochs
        else:
            # Exponential decay: lr_decay^(epoch - warmup_epochs)
            return args.lr_decay ** (epoch - args.warmup_epochs)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger.info(f"LR schedule: linear warmup {args.warmup_epochs} epochs "
                f"→ peak lr={args.lr} → exponential decay {args.lr_decay}/epoch")

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    patience_counter = 0
    history = []

    logger.info(f"\nStarting training: {args.epochs} epochs\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()

        # ── Train ─────────────────────────────────────────────────────
        model.train()
        train_loss, n_batches, grad_norm = 0.0, 0, 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.epochs} [train]",
                          leave=False, dynamic_ncols=True)
        for batch in train_pbar:
            lfp = batch['lfp'].to(DEVICE, non_blocking=True)
            sid = batch['session_id'].to(DEVICE, non_blocking=True)
            cmask = batch['channel_mask'].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(lfp, session_ids=sid, channel_mask=cmask)
            loss = out['loss']

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1
            train_pbar.set_postfix(
                loss=f"{train_loss / n_batches:.5f}",
                gnorm=f"{grad_norm:.2f}"
            )

            # Early NaN detection
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"  NaN/Inf detected at batch {n_batches}! "
                               f"grad_norm={grad_norm:.4f}")
                break

        scheduler.step()
        avg_train = train_loss / max(n_batches, 1)

        # ── Validate ──────────────────────────────────────────────────
        model.eval()
        val_loss, n_val = 0.0, 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:3d}/{args.epochs} [val]  ",
                        leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for batch in val_pbar:
                lfp = batch['lfp'].to(DEVICE, non_blocking=True)
                sid = batch['session_id'].to(DEVICE, non_blocking=True)
                cmask = batch['channel_mask'].to(DEVICE, non_blocking=True)
                out = model(lfp, session_ids=sid, channel_mask=cmask)
                val_loss += out['loss'].item()
                n_val += 1
                val_pbar.set_postfix(loss=f"{val_loss / n_val:.5f}")

        avg_val = val_loss / max(n_val, 1)
        elapsed = time.time() - t0
        peak_mem = torch.cuda.max_memory_allocated() / 1e9

        # ── Log ───────────────────────────────────────────────────────
        is_best = avg_val < best_val_loss
        best_marker = " ★ BEST" if is_best else ""
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={avg_train:.5f} | val_loss={avg_val:.5f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"grad_norm={grad_norm:.3f} | "
            f"peak_vram={peak_mem:.1f}GB | "
            f"time={elapsed:.1f}s{best_marker}"
        )

        history.append({
            'epoch': epoch,
            'train_loss': avg_train,
            'val_loss': avg_val,
            'lr': optimizer.param_groups[0]['lr'],
            'peak_vram_gb': round(peak_mem, 1),
            'time_s': round(elapsed, 1),
            'is_best': is_best,
        })

        # ── Checkpointing ─────────────────────────────────────────────
        ckpt_prefix = "pretrain_REALM_teacher"

        if is_best:
            best_val_loss = avg_val
            best_epoch = epoch
            best_state = {k: v.cpu().clone()
                          for k, v in get_encoder().state_dict().items()}
            patience_counter = 0

            # Save best checkpoint
            best_path = OUTPUT_DIR / f"{ckpt_prefix}_best.pt"
            torch.save({
                'encoder_state_dict': best_state,
                'encoder_kwargs': encoder_kwargs,
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'args': vars(args),
            }, best_path)
        else:
            patience_counter += 1

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = OUTPUT_DIR / f"{ckpt_prefix}_epoch{epoch:03d}.pt"
            torch.save({
                'encoder_state_dict': {k: v.cpu().clone()
                                       for k, v in get_encoder().state_dict().items()},
                'encoder_kwargs': encoder_kwargs,
                'epoch': epoch,
                'val_loss': avg_val,
                'args': vars(args),
            }, ckpt_path)
            logger.info(f"  → Saved periodic checkpoint: {ckpt_path.name}")

        # Save history after every epoch
        hist_path = OUTPUT_DIR / "pretrain_history.json"
        with open(hist_path, 'w') as f:
            json.dump(history, f, indent=2)

        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"\nEarly stopping at epoch {epoch} "
                        f"(best epoch: {best_epoch}, best val_loss: {best_val_loss:.5f})")
            break

    # ── Final save ────────────────────────────────────────────────────────
    final_prefix = "pretrain_REALM_teacher"
    final_path = OUTPUT_DIR / f"{final_prefix}.pt"
    torch.save({
        'encoder_state_dict': best_state,
        'encoder_kwargs': encoder_kwargs,
        'epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'args': vars(args),
    }, final_path)

    logger.info(f"\n{'='*70}")
    logger.info(f"Training complete!")
    logger.info(f"Best epoch: {best_epoch}, best val_loss: {best_val_loss:.5f}")
    logger.info(f"Final model saved to: {final_path}")
    logger.info(f"Training log saved to: {OUTPUT_DIR / 'pretrain.log'}")
    logger.info(f"History saved to: {OUTPUT_DIR / 'pretrain_history.json'}")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    main()
