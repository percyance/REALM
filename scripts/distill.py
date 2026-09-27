#!/usr/bin/env python3
"""
Stage 2: retrospective knowledge distillation (RKD), frozen teacher -> REALM or REALM-bi.

The BiMamba-2 teacher of stage 1 (scripts/pretrain.py) is frozen and distilled into a
student on the non-held-out sessions of one corpus:

    L = lambda_repr * L_repr + lambda_ae * L_ae + lambda_task * L_task

  L_repr  1 - cosine similarity between the student's and the teacher's last-layer
          representations, at every time step
  L_ae    reconstruction of the input LFP from the student's last-layer representation
  L_task  MSE of the student's velocity prediction (lambda_task = 0 in the paper, so no
          behavioural label enters distillation)

The student starts from a copy of the teacher's neural tokenizer (trained further); its
Mamba-2 layers start at random. The first 20% of the non-held-out sessions, in name order,
validate: they select the checkpoint and drive early stopping. The held-out sessions are
never read.

Usage:
    python scripts/distill.py --student realm --dataset makin --seed 42
    python scripts/distill.py --student realm_bi --dataset flint --seed 123 \
        --out output/realm_bi_flint_s123.pt
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from models import REALMDecoder, STUDENT_KWARGS                      # noqa: E402
from utils import RKDLoss                                            # noqa: E402
from utils.dataset import (                                          # noqa: E402
    discover_sessions, HELDOUT_MAKIN, HELDOUT_FLINT, MAX_CHANNELS, SEGMENT_LENGTH,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_RATIO = 0.8      # the remaining 20% of the non-held-out sessions validate
HELDOUT = {'makin': HELDOUT_MAKIN, 'flint': HELDOUT_FLINT}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve(path):
    """A relative path is taken from the working directory, else from the repository."""
    p = Path(path)
    return p if p.is_absolute() or p.exists() else REPO / p


def load_teacher(path):
    """The frozen teacher, built as a REALMDecoder around the pretrained encoder."""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    kwargs = dict(ckpt['encoder_kwargs'])
    # Built with a 200-entry session table, as in the paper's runs, so that the random-number
    # stream (and hence the student's initialisation) matches; the table is never used.
    kwargs['max_sessions'] = 200
    teacher = REALMDecoder(encoder_kwargs=kwargs, output_dim=2).to(DEVICE)
    state = {k: v for k, v in ckpt['encoder_state_dict'].items() if 'session_embed' not in k}
    missing, unexpected = teacher.encoder.load_state_dict(state, strict=False)
    assert not unexpected and set(missing) == {'session_embed.weight'}, (missing, unexpected)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    print(f"  Teacher: {sum(p.numel() for p in teacher.parameters()):,} params (frozen), "
          f"{kwargs['n_layers']} layers, bidirectional={kwargs['bidirectional']}")
    return teacher


def build_student(teacher, name):
    """Student of the given size, starting from a copy of the teacher's neural tokenizer."""
    kwargs = {**STUDENT_KWARGS[name], 'max_sessions': 200}
    student = REALMDecoder(encoder_kwargs=kwargs, output_dim=2).to(DEVICE)
    t_enc, s_enc = teacher.encoder, student.encoder
    for module in ('temporal_conv', 'eca_conv', 'spatial_norm', 'spatial_proj'):
        getattr(s_enc, module).load_state_dict(getattr(t_enc, module).state_dict())
    n = sum(p.numel() for p in student.parameters())
    print(f"  Student ({name}): {n:,} params, {kwargs['n_layers']} layers, "
          f"bidirectional={kwargs['bidirectional']}; tokenizer copied from the teacher")
    return student, kwargs


def load_session_segments(path, segment_length=SEGMENT_LENGTH):
    """One session cut into non-overlapping segments, channels zero-padded to 96."""
    data = np.load(path, allow_pickle=True)
    lfp = data['lfp_data'].astype(np.float32)      # (n_channels, 1, T)
    targets = data['targets'].astype(np.float32)   # (T, 2)
    n_ch = lfp.shape[0]
    if n_ch < MAX_CHANNELS:
        pad = np.zeros((MAX_CHANNELS - n_ch, lfp.shape[1], lfp.shape[2]), dtype=np.float32)
        lfp = np.concatenate([lfp, pad], axis=0)

    ch_mask = torch.zeros(MAX_CHANNELS)
    ch_mask[:n_ch] = 1.0
    lfp_segs, cmask_segs, tgt_segs = [], [], []
    for start in range(0, lfp.shape[2] - segment_length + 1, segment_length):
        end = start + segment_length
        lfp_segs.append(torch.from_numpy(lfp[:, :, start:end].copy()))
        tgt_segs.append(torch.from_numpy(targets[start:end].copy()))
        cmask_segs.append(ch_mask.clone())
    return torch.stack(lfp_segs), torch.stack(cmask_segs), torch.stack(tgt_segs)


def step_losses(teacher, student, criterion, lfp, cmask, tgt):
    """Distillation loss of one batch (teacher without gradients)."""
    with torch.no_grad():
        t_out = teacher(lfp, channel_mask=cmask, return_intermediates=True)
        teacher_repr = t_out['intermediates'][-1].detach()
    s_out = student(lfp, channel_mask=cmask, return_intermediates=True)
    return criterion(student_pred=s_out['prediction'], target=tgt,
                     student_repr=s_out['intermediates'][-1], teacher_repr=teacher_repr,
                     lfp_data=lfp)


@torch.no_grad()
def validation_loss(teacher, student, criterion, loader):
    student.eval()
    total, n = 0.0, 0
    for lfp, cmask, tgt in loader:
        out = step_losses(teacher, student, criterion,
                          lfp.to(DEVICE), cmask.to(DEVICE), tgt.to(DEVICE))
        total += out['loss'].item()
        n += 1
    return total / max(n, 1)


def main():
    p = argparse.ArgumentParser(
        description="Retrospective knowledge distillation: frozen teacher -> REALM / REALM-bi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--teacher', default='checkpoints/teacher.pt',
                   help="pretrained teacher (scripts/pretrain.py)")
    p.add_argument('--student', default='realm', choices=sorted(STUDENT_KWARGS),
                   help="realm: causal Mamba-2, 10 layers; realm_bi: BiMamba-2, 8 layers")
    p.add_argument('--dataset', default='makin', choices=['makin', 'flint'],
                   help="corpus whose non-held-out sessions are distilled on")
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default=None,
                   help="output checkpoint (default: output/<student>_<dataset>_s<seed>.pt)")
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--patience', type=int, default=30,
                   help="early stopping on the validation loss")
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=5e-4,
                   help="AdamW peak learning rate, cosine-annealed to lr/100")
    p.add_argument('--lambda_repr', type=float, default=1.0)
    p.add_argument('--lambda_ae', type=float, default=1.0)
    p.add_argument('--lambda_task', type=float, default=0.0,
                   help="velocity term; 0 keeps distillation label-free")
    p.add_argument('--num_workers', type=int, default=0)
    args = p.parse_args()

    out = (Path(args.out) if args.out
           else REPO / 'output' / f"{args.student}_{args.dataset}_s{args.seed}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"Note: {out} exists and will be overwritten")

    set_seed(args.seed)
    print(f"Device: {DEVICE} | student={args.student} dataset={args.dataset} seed={args.seed}")

    print("\n=== Teacher ===")
    teacher = load_teacher(resolve(args.teacher))
    print("\n=== Student ===")
    student, student_kwargs = build_student(teacher, args.student)

    # Non-held-out sessions of the corpus, in name order; the first 20% validate.
    print(f"\n=== Data ({args.dataset}) ===")
    heldout = set(HELDOUT[args.dataset])
    train, names = [], []
    for name, info in sorted(discover_sessions(datasets=[args.dataset]).items()):
        if name in heldout:
            print(f"  [held out] {name}")
            continue
        train.append(load_session_segments(info['path']))
        names.append(name)
        print(f"  [train]    {name:<25} N={len(train[-1][0])}")
    n_val = max(1, int(round((1.0 - TRAIN_RATIO) * len(train))))
    val, train = train[:n_val], train[n_val:]
    print(f"  validation sessions: {names[:n_val]}")

    def dataset(recs):
        return TensorDataset(*(torch.cat([r[i] for r in recs], dim=0) for i in range(3)))

    train_ds, val_ds = dataset(train), dataset(val)
    print(f"  train: {len(train_ds)} segments from {len(train)} sessions | "
          f"val: {len(val_ds)} segments from {len(val)} sessions")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            num_workers=args.num_workers)

    criterion = RKDLoss(lambda_task=args.lambda_task, lambda_repr=args.lambda_repr,
                        lambda_ae=args.lambda_ae, d_model=student_kwargs['d_model'],
                        n_channels=96, n_bands=1, ae_mask_ratio=0.3).to(DEVICE)
    # The reconstruction head is trained with the student.
    params = [q for q in list(student.parameters()) + list(criterion.parameters())
              if q.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                     eta_min=args.lr * 0.01)

    print(f"\n=== Distillation (lambda_repr={args.lambda_repr}, lambda_ae={args.lambda_ae}, "
          f"lambda_task={args.lambda_task}, last-layer alignment) ===")
    best_val, best_epoch, bad, t0 = float('inf'), 0, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        student.train()
        tot = {'loss': 0.0, 'l_repr': 0.0, 'l_ae': 0.0, 'l_task': 0.0}
        n = 0
        for lfp, cmask, tgt in train_loader:
            lfp, cmask, tgt = lfp.to(DEVICE), cmask.to(DEVICE), tgt.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            out_d = step_losses(teacher, student, criterion, lfp, cmask, tgt)
            out_d['loss'].backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            tot['loss'] += out_d['loss'].item()
            for k in ('l_repr', 'l_ae', 'l_task'):
                tot[k] += out_d[k]
            n += 1
        scheduler.step()
        avg = {k: v / max(n, 1) for k, v in tot.items()}

        val_loss = validation_loss(teacher, student, criterion, val_loader)
        improved = val_loss < best_val
        print(f"  Epoch {epoch:3d}/{args.epochs} | train={avg['loss']:.5f} val={val_loss:.5f} | "
              f"repr={avg['l_repr']:.4f} ae={avg['l_ae']:.4f} task={avg['l_task']:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | {time.time() - t0:.0f}s"
              f"{' *' if improved else ''}", flush=True)

        if improved:
            best_val, best_epoch, bad = val_loss, epoch, 0
            torch.save({
                'student_state_dict': {k: v.detach().cpu() for k, v in student.state_dict().items()},
                'encoder_kwargs': student_kwargs,
                'head_window': 1,
                'head_layers': 1,
                'head_dropout': 0.0,
                'epoch': epoch,
                'val_loss': best_val,
            }, out)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  Early stop at epoch {epoch}", flush=True)
                break

    print(f"\nBest epoch {best_epoch}, validation loss {best_val:.5f} -> {out} "
          f"({time.time() - t0:.0f}s)")


if __name__ == '__main__':
    main()
