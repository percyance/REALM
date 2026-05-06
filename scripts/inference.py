"""Minimal causal inference example for REALM students.

Loads a distilled checkpoint, optionally a per-session finetuned head,
runs prediction on a test split, and reports per-axis R².

Usage:
    # Single distilled student (no per-session finetune)
    python scripts/inference.py \
        --ckpt checkpoints/realm_makin.pt \
        --datasets makin --session indy_20160622_01

    # With per-session finetune (uses finetune.py logic)
    python scripts/inference.py \
        --ckpt checkpoints/realm_makin.pt \
        --datasets makin --session indy_20160622_01 \
        --finetune --epochs 50 --train_ratio 0.8
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.decoder import REALMDecoder
from utils.dataset import create_test_dataloaders, HELDOUT_MAKIN, HELDOUT_FLINT
from scripts.finetune import (load_model, finetune_session, set_seed, DEVICE)


def predict(model, lfp, cmask, tgt):
    model.eval()
    sid = torch.zeros(len(lfp), dtype=torch.long)
    ds = TensorDataset(lfp, cmask, tgt, sid)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    preds = []
    with torch.no_grad():
        for x, m, _, _ in loader:
            out = model(x.to(DEVICE), session_ids=None,
                        channel_mask=m.to(DEVICE))
            preds.append(out['prediction'].cpu())
    return torch.cat(preds, dim=0).numpy()


def r2_per_axis(pred, tgt):
    p = pred.reshape(-1, 2); t = tgt.reshape(-1, 2)
    ss_res = ((p - t) ** 2).sum(axis=0)
    ss_tot = ((t - t.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-10)


class _FTArgs:
    seed = 42; epochs = 50; lr = 5e-4; weight_decay = 1e-5
    patience = 15; batch_size = 32; use_session_embed = False
    early_stop_metric = 'val'; r2_mode = 'combined'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, type=str,
                    help='Path to distilled student ckpt')
    ap.add_argument('--datasets', nargs='+', default=['makin'],
                    choices=['makin', 'flint'])
    ap.add_argument('--session', type=str, default=None,
                    help='Specific heldout session (default: all heldout)')
    ap.add_argument('--finetune', action='store_true',
                    help='Per-session supervised finetune before predicting')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--train_ratio', type=float, default=0.8)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    print(f'Device: {DEVICE}')
    print(f'Loading: {args.ckpt}')

    test_loaders = create_test_dataloaders(segment_length=500, batch_size=256)
    target_sessions = set()
    if 'makin' in args.datasets: target_sessions.update(HELDOUT_MAKIN)
    if 'flint' in args.datasets: target_sessions.update(HELDOUT_FLINT)
    if args.session is not None:
        target_sessions = {args.session}

    for sess_name in sorted(target_sessions & set(test_loaders)):
        loader = test_loaders[sess_name]
        sess_hash = int(hashlib.md5(sess_name.encode()).hexdigest(), 16) % (2**31)
        set_seed(args.seed + sess_hash)

        all_lfp, all_cmask, all_tgt = [], [], []
        for b in loader:
            all_lfp.append(b['lfp'])
            all_cmask.append(b['channel_mask'])
            all_tgt.append(b['target'])
        all_lfp = torch.cat(all_lfp); all_cmask = torch.cat(all_cmask); all_tgt = torch.cat(all_tgt)

        n = len(all_lfp); split = int(args.train_ratio * n)
        indices = torch.randperm(n)
        train_idx, test_idx = indices[:split], indices[split:]
        train_lfp, test_lfp = all_lfp[train_idx], all_lfp[test_idx]
        train_cmask, test_cmask = all_cmask[train_idx], all_cmask[test_idx]
        train_tgt, test_tgt = all_tgt[train_idx], all_tgt[test_idx]

        t0 = time.time()
        model, _ = load_model(args.ckpt, freeze_layers=0)
        if args.finetune:
            ft_args = _FTArgs(); ft_args.epochs = args.epochs
            model = finetune_session(model, train_lfp, train_cmask, train_tgt, ft_args,
                                     test_lfp=test_lfp, test_cmask=test_cmask, test_tgt=test_tgt)
        preds = predict(model, test_lfp, test_cmask, test_tgt)
        r2 = r2_per_axis(preds, test_tgt.numpy())
        print(f'  {sess_name:<25} R² (vx, vy) = ({r2[0]:.4f}, {r2[1]:.4f})  '
              f'mean={r2.mean():.4f}  ({time.time()-t0:.1f}s)')


if __name__ == '__main__':
    main()
