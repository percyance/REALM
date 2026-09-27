"""
Per-session supervised fine-tuning of a distilled encoder on the held-out sessions.

Each held-out session is fine-tuned independently, starting from the same checkpoint:
  - its 5 s segments are split 72/8/20 into training, validation and test folds by the
    canonical split file (splits/canonical_splits_728020.json);
  - the checkpoint's encoder is loaded into REALMDecoder with a fresh linear velocity head;
  - the neural tokenizer (per-channel temporal convolution, ECA, projection, LayerNorm) stays
    frozen, and the raw-LFP skip is zeroed and frozen so that the prediction comes from the
    encoder alone; every Mamba-2 layer and the head are trained with an MSE loss on velocity;
  - training stops early on the per-axis R² of the validation fold (patience 20) and the
    best epoch's weights are scored once on the test fold (per-axis R²).

The encoder runs causally for a causal checkpoint (REALM) and bidirectionally for a
bidirectional one (REALM-bi). Flint sessions are read from the Tukey-filtered files written
by scripts/make_flint_tukey.py unless --no_tukey is given.

Usage:
    python scripts/finetune.py --ckpt checkpoints/realm_makin.pt --dataset makin --seed 42
    python scripts/finetune.py --ckpt checkpoints/realm_flint.pt --dataset flint --seed 42
"""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models import REALMDecoder
from utils.dataset import (
    create_test_dataloaders, compute_r2_per_axis, load_canonical_split,
    HELDOUT_MAKIN, HELDOUT_FLINT,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# The neural tokenizer, kept frozen during fine-tuning.
TOKENIZER = ('temporal_conv', 'eca_conv', 'spatial_proj', 'spatial_norm')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def session_seed(seed, session):
    """Seed of one session's run: the run seed offset by a hash of the session name."""
    return seed + int(hashlib.md5(session.encode()).hexdigest(), 16) % (2**31)


def resolve(path):
    """A path as given, or relative to the repository root if it does not exist as given."""
    p = Path(path)
    return p if p.exists() or p.is_absolute() else REPO_ROOT / p


def load_model(ckpt_path):
    """REALMDecoder with the checkpoint's encoder weights and a fresh velocity head.

    Accepts a distilled student (REALM, REALM-bi: 'student_state_dict') or the teacher
    ('encoder_state_dict'). The tokenizer and the unused session table are frozen, and the
    raw-LFP skip is zeroed and frozen.
    """
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    encoder_kwargs = dict(ckpt['encoder_kwargs'], max_sessions=200)
    model = REALMDecoder(encoder_kwargs=encoder_kwargs, output_dim=2).to(DEVICE)

    if 'student_state_dict' in ckpt:
        enc_state = {k[len('encoder.'):]: v for k, v in ckpt['student_state_dict'].items()
                     if k.startswith('encoder.')}
    elif 'encoder_state_dict' in ckpt:
        enc_state = ckpt['encoder_state_dict']
    else:
        raise ValueError(f"unknown checkpoint format: {list(ckpt)}")
    # The session table is not transferred: no session identity is used at any stage.
    msd = model.encoder.state_dict()
    state = {k: v for k, v in enc_state.items()
             if 'session_embed' not in k and k in msd and msd[k].shape == v.shape}
    missing = [k for k in msd if k not in state and 'session_embed' not in k]
    if missing:
        raise ValueError(f"{ckpt_path}: encoder weights missing from the checkpoint: {missing}")
    model.encoder.load_state_dict(state, strict=False)

    for name in TOKENIZER:
        getattr(model.encoder, name).requires_grad_(False)
    model.encoder.session_embed.requires_grad_(False)
    with torch.no_grad():
        model.skip.weight.zero_()
        model.skip.bias.zero_()
    model.skip.requires_grad_(False)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {'bidirectional' if encoder_kwargs['bidirectional'] else 'causal'} encoder, "
          f"{encoder_kwargs['n_layers']} layers; trainable {trainable:,} "
          f"({100 * trainable / total:.1f}%), frozen {total - trainable:,}", flush=True)
    return model, encoder_kwargs


@torch.no_grad()
def evaluate(model, lfp, cmask, tgt, return_pred=False):
    """Per-axis R² of the model's velocity on a set of segments."""
    model.eval()
    loader = DataLoader(TensorDataset(lfp, cmask, tgt), batch_size=64, shuffle=False)
    preds, targets = [], []
    for x, m, y in loader:
        preds.append(model(x.to(DEVICE), channel_mask=m.to(DEVICE))['prediction'].cpu())
        targets.append(y)
    preds = torch.cat(preds, dim=0).reshape(-1, 2)
    targets = torch.cat(targets, dim=0).reshape(-1, 2)
    r2 = compute_r2_per_axis(preds, targets)
    if return_pred:
        return r2, preds.numpy(), targets.numpy()
    return r2


def finetune_session(model, train, val, args):
    """Fine-tune on the training fold; keep the epoch with the best validation R²."""
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(TensorDataset(*train), batch_size=args.batch_size, shuffle=True,
                        drop_last=False, generator=g)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    criterion = nn.MSELoss()

    best_r2, best_state, best_epoch, bad = -float('inf'), None, 0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]['lr']
        model.train()
        total_loss, n = 0.0, 0
        for x, m, y in loader:
            x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x, channel_mask=m)['prediction'], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        scheduler.step()

        val_r2 = evaluate(model, *val)
        improved = val_r2 > best_r2
        if improved:
            best_r2, best_epoch, bad = val_r2, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"    epoch {epoch:3d}  train_loss {total_loss / max(n, 1):.5f}  "
              f"val_R2 {val_r2:.4f}  best {best_r2:.4f}  lr {lr:.2e}  "
              f"{time.time() - t0:.1f}s{'  *' if improved else ''}", flush=True)
        if bad >= args.patience:
            print(f"    early stop at epoch {epoch}; best epoch {best_epoch}", flush=True)
            break

    if best_state is not None:          # no epoch improved (e.g. NaN): keep the last weights
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model, best_r2, best_epoch


def main():
    parser = argparse.ArgumentParser(
        description="Per-session supervised fine-tuning on the held-out sessions. The neural "
                    "tokenizer stays frozen and the raw-LFP skip is zeroed; every Mamba-2 "
                    "layer and a fresh linear head are trained.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ckpt', required=True,
                        help="distilled student (REALM / REALM-bi) or teacher checkpoint")
    parser.add_argument('--dataset', required=True, choices=['makin', 'flint'],
                        help="corpus whose held-out sessions are fine-tuned and scored")
    parser.add_argument('--seed', type=int, default=42,
                        help="run seed; also selects the split of the split file")
    parser.add_argument('--splits_file', default='splits/canonical_splits_728020.json',
                        help="72/8/20 train/validation/test split of each held-out session")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20,
                        help="early-stopping patience on the validation R²")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4,
                        help="peak learning rate; cosine annealing to 1%% of it")
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--out', default=None,
                        help="result JSON (default: output/finetune_<ckpt>_<dataset>_s<seed>.json)")
    parser.add_argument('--dump_pred', default=None, metavar='DIR',
                        help="write each session's test-fold predictions and targets here")
    parser.add_argument('--save_dir', default=None, metavar='DIR',
                        help="write each session's fine-tuned weights here")
    parser.add_argument('--no_tukey', action='store_true',
                        help="read Flint from the unfiltered files instead of the Tukey-filtered ones")
    args = parser.parse_args()

    ckpt_path = resolve(args.ckpt)
    splits_file = resolve(args.splits_file)
    tukey = args.dataset == 'flint' and not args.no_tukey
    heldout = HELDOUT_MAKIN if args.dataset == 'makin' else HELDOUT_FLINT

    set_seed(args.seed)
    print(f"Device: {DEVICE}  checkpoint: {ckpt_path}  dataset: {args.dataset}  "
          f"seed: {args.seed}  tukey_flint: {tukey}", flush=True)
    loaders = create_test_dataloaders(batch_size=256, tukey_flint=tukey, datasets=[args.dataset])
    missing = sorted(set(heldout) - set(loaders))
    if missing:
        raise FileNotFoundError(f"held-out sessions not found under the data root: {missing}")
    print("  segments: " + "  ".join(f"{s}={len(loaders[s].dataset)}" for s in sorted(heldout)),
          flush=True)

    per_session, per_session_val, best_epochs = {}, {}, {}
    for sess in sorted(heldout):
        t0 = time.time()
        set_seed(session_seed(args.seed, sess))
        print(f"\n{sess}", flush=True)

        lfp, cmask, tgt = [], [], []
        for batch in loaders[sess]:
            lfp.append(batch['lfp'])
            cmask.append(batch['channel_mask'])
            tgt.append(batch['target'])
        lfp, cmask, tgt = torch.cat(lfp), torch.cat(cmask), torch.cat(tgt)

        tr, va, te = (torch.tensor(i) for i in load_canonical_split(
            splits_file, sess, args.seed, len(lfp), tukey_flint=tukey))
        print(f"  split: {len(tr)} train / {len(va)} val / {len(te)} test segments", flush=True)

        model, encoder_kwargs = load_model(ckpt_path)
        model, val_r2, best_epoch = finetune_session(
            model, (lfp[tr], cmask[tr], tgt[tr]), (lfp[va], cmask[va], tgt[va]), args)

        # The test fold is scored once, with the selected weights, in ascending segment order.
        te = te[torch.argsort(te)]
        r2, pred, target = evaluate(model, lfp[te], cmask[te], tgt[te], return_pred=True)
        per_session[sess], per_session_val[sess], best_epochs[sess] = r2, val_r2, best_epoch
        print(f"  {sess}: test R2 {r2:.4f}  (val R2 {val_r2:.4f}, best epoch {best_epoch}, "
              f"{time.time() - t0:.0f}s)", flush=True)

        if args.dump_pred:
            d = Path(args.dump_pred)
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / f"{sess}.npz", pred=pred, target=target, test_idx=te.numpy())
        if args.save_dir:
            d = Path(args.save_dir)
            d.mkdir(parents=True, exist_ok=True)
            torch.save({'model_state_dict': model.state_dict(), 'encoder_kwargs': encoder_kwargs,
                        'session': sess, 'seed': args.seed, 'test_r2': r2, 'val_r2': val_r2},
                       d / f"finetuned_{sess}_s{args.seed}.pt")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_r2 = float(np.mean(list(per_session.values())))
    print(f"\nmean test R2 over {len(per_session)} {args.dataset} sessions: {mean_r2:.4f}")
    for s, v in per_session.items():
        print(f"  {s:<20} {v:.4f}")

    result = {
        'checkpoint': str(args.ckpt), 'dataset': args.dataset, 'seed': args.seed,
        'splits_file': str(args.splits_file), 'tukey_flint': tukey,
        'per_session_r2': per_session, 'mean_r2': mean_r2,
        'per_session_val_r2': per_session_val, 'best_epoch': best_epochs,
        'settings': {k: getattr(args, k) for k in
                     ('epochs', 'patience', 'batch_size', 'lr', 'weight_decay')},
    }
    out = Path(args.out) if args.out else (
        REPO_ROOT / 'output' / f"finetune_{Path(args.ckpt).stem}_{args.dataset}_s{args.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"saved {out}")


if __name__ == '__main__':
    main()
