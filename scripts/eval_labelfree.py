#!/usr/bin/env python3
"""
Label-free evaluation: a ridge read-out on the frozen encoder, per held-out session.

The encoder is used exactly as distilled; velocity labels reach only the read-out. For each
held-out session of the chosen corpus:

  1. The session's non-overlapping 5 s segments are split 72/8/20 into training, validation
     and test folds (splits/canonical_splits_728020.json).
  2. Every segment is encoded by the frozen encoder -- causally for REALM, bidirectionally
     for REALM-bi, as recorded in the checkpoint. The feature of frame t is the encoder
     output after its final LayerNorm (d_model = 256).
  3. The features of frames t-10 ... t are stacked (past only; the first frames of a segment
     repeat frame 0), giving 11 x 256 = 2816 features per frame.
  4. The ridge penalty is chosen from {1, 10, ..., 1e6} by per-axis R2 on the validation
     fold, with the read-out fitted on the training fold.
  5. The read-out is refitted on training + validation and scored once on the test fold.

The reported metric is R2 per velocity axis, averaged over the two axes.

Usage:
    python scripts/eval_labelfree.py --ckpt checkpoints/realm_makin.pt --dataset makin
    python scripts/eval_labelfree.py --ckpt checkpoints/realm_bi_flint.pt --dataset flint
"""

import argparse
import hashlib
import json
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import LinAlgWarning
from sklearn.linear_model import Ridge

# Small penalties on 2816 strongly correlated features are ill-conditioned by design; the
# penalty is chosen on the validation fold, so scipy's warning is expected and not shown.
warnings.filterwarnings('ignore', category=LinAlgWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models import REALMEncoder
from utils.dataset import (
    DATA_ROOT, HELDOUT_FLINT, HELDOUT_MAKIN, compute_r2_per_axis,
    create_test_dataloaders, load_canonical_split,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HELDOUT = {'makin': HELDOUT_MAKIN, 'flint': HELDOUT_FLINT}
ALPHA_GRID = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6)
SEGMENT_LENGTH = 500   # 5 s at 100 Hz
BATCH_SIZE = 64        # segments per encoder forward pass


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_encoder(ckpt_path):
    """Frozen encoder from a released checkpoint.

    Accepts a distilled student (REALM / REALM-bi: 'student_state_dict', encoder weights
    under 'encoder.') or the teacher ('encoder_state_dict'). The session-embedding table is
    unused and not loaded (the teacher's is sized for its own sessions).
    """
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    kwargs = dict(ckpt['encoder_kwargs'], max_sessions=200)
    encoder = REALMEncoder(**kwargs).to(DEVICE)
    if 'student_state_dict' in ckpt:
        state = {k[len('encoder.'):]: v for k, v in ckpt['student_state_dict'].items()
                 if k.startswith('encoder.')}
    elif 'encoder_state_dict' in ckpt:
        state = ckpt['encoder_state_dict']
    else:
        raise ValueError(f"{ckpt_path}: unknown checkpoint format {list(ckpt)}")
    state = {k: v for k, v in state.items() if not k.startswith('session_embed')}
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith('session_embed') for k in missing):
        raise ValueError(f"{ckpt_path}: missing {missing}, unexpected {unexpected}")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder, kwargs


@torch.no_grad()
def encode(encoder, lfp, channel_mask, idx):
    """Encoder output of the segments `idx`: (len(idx), T, d_model) float32."""
    feats = []
    for i in range(0, len(idx), BATCH_SIZE):
        b = idx[i:i + BATCH_SIZE]
        out = encoder(lfp[b].to(DEVICE), channel_mask=channel_mask[b].to(DEVICE))
        feats.append(out.float().cpu())
    return torch.cat(feats, dim=0).numpy()


def past_context(X, k):
    """Stack frames t-k ... t within each segment: (N, T, d) -> (N * T, d * (k + 1)).

    The first k frames of a segment repeat its frame 0, so no frame looks ahead and no
    frame reaches into another segment.
    """
    N, T, d = X.shape
    if k == 0:
        return X.reshape(-1, d)
    padded = np.pad(X, ((0, 0), (k, 0), (0, 0)), mode='edge')
    stacked = np.concatenate([padded[:, i:i + T, :] for i in range(k + 1)], axis=-1)
    return stacked.reshape(-1, d * (k + 1))


def r2_axes(pred, target):
    """R2 of each velocity axis, computed as in compute_r2_per_axis."""
    p, t = torch.from_numpy(pred).float(), torch.from_numpy(target).float()
    out = []
    for i in range(t.shape[-1]):
        ss_res = torch.sum((t[:, i] - p[:, i]) ** 2)
        ss_tot = torch.sum((t[:, i] - t[:, i].mean()) ** 2)
        out.append((1 - ss_res / (ss_tot + 1e-8)).item())
    return out


def evaluate_session(encoder, loader, session, args, tukey):
    batches = list(loader)
    lfp = torch.cat([b['lfp'] for b in batches])
    cmask = torch.cat([b['channel_mask'] for b in batches])
    target = torch.cat([b['target'] for b in batches])

    train, val, test = load_canonical_split(args.splits_file, session, args.seed,
                                            len(lfp), tukey_flint=tukey)
    fit_idx = torch.tensor(sorted(train + val), dtype=torch.long)
    test_idx = torch.tensor(sorted(test), dtype=torch.long)

    X_fit = past_context(encode(encoder, lfp, cmask, fit_idx), args.context)
    y_fit = target[fit_idx].numpy().reshape(-1, 2)
    X_test = past_context(encode(encoder, lfp, cmask, test_idx), args.context)
    y_test = target[test_idx].numpy().reshape(-1, 2)

    # Penalty chosen on the validation fold with the read-out fitted on the training fold;
    # the test fold is never consulted.
    val_set = set(val)
    val_rows = np.repeat(np.array([i in val_set for i in fit_idx.tolist()]),
                         SEGMENT_LENGTH)
    scored = []
    for alpha in ALPHA_GRID:
        m = Ridge(alpha=alpha).fit(X_fit[~val_rows], y_fit[~val_rows])
        scored.append((float(compute_r2_per_axis(
            torch.from_numpy(m.predict(X_fit[val_rows])).float(),
            torch.from_numpy(y_fit[val_rows]).float())), alpha))
    val_r2, alpha = max(scored)

    pred = Ridge(alpha=alpha).fit(X_fit, y_fit).predict(X_test)
    r2_vx, r2_vy = r2_axes(pred, y_test)
    return {
        'r2': float(np.mean([r2_vx, r2_vy])), 'r2_vx': r2_vx, 'r2_vy': r2_vy,
        'alpha': float(alpha), 'val_r2': val_r2,
        'n_fit_segments': len(fit_idx), 'n_test_segments': len(test_idx),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Label-free evaluation: ridge read-out on the frozen encoder, "
                    "per held-out session.")
    parser.add_argument('--ckpt', type=str, required=True,
                        help="REALM / REALM-bi checkpoint (or the teacher)")
    parser.add_argument('--dataset', type=str, default='makin', choices=['makin', 'flint'],
                        help="corpus whose held-out sessions are scored (default: makin)")
    parser.add_argument('--seed', type=int, default=42,
                        help="selects the 72/8/20 split of each session: 42, 123 or 456 "
                             "(default: 42)")
    parser.add_argument('--splits_file', type=str,
                        default=str(REPO_ROOT / 'splits' / 'canonical_splits_728020.json'),
                        help="split file (default: splits/canonical_splits_728020.json)")
    parser.add_argument('--context', type=int, default=10,
                        help="past frames stacked with the current one (default: 10)")
    parser.add_argument('--no_tukey', action='store_true',
                        help="read Flint without the artifact-window removal "
                             "(the paper scores Flint on the filtered files)")
    parser.add_argument('--out', type=str, default=None,
                        help="result JSON (default: output/labelfree_<ckpt>_<dataset>"
                             "_s<seed>.json)")
    args = parser.parse_args()

    tukey = args.dataset == 'flint' and not args.no_tukey
    set_seed(args.seed)
    encoder, kwargs = load_encoder(args.ckpt)
    mode = 'bidirectional' if kwargs['bidirectional'] else 'causal'
    print(f"Checkpoint: {args.ckpt}  ({mode} encoder, {kwargs['n_layers']} layers)")
    print(f"Data: {DATA_ROOT}  corpus: {args.dataset}"
          f"{'  (Flint artifact windows removed)' if tukey else ''}")
    print(f"Read-out: ridge on frames t-{args.context}..t, penalty from "
          f"{{{', '.join(f'{a:g}' for a in ALPHA_GRID)}}} chosen on the validation fold; "
          f"split seed {args.seed}")

    loaders = create_test_dataloaders(segment_length=SEGMENT_LENGTH, batch_size=BATCH_SIZE,
                                      tukey_flint=tukey, datasets=[args.dataset])
    missing = [s for s in HELDOUT[args.dataset] if s not in loaders]
    if missing:
        raise FileNotFoundError(f"held-out sessions not found under {DATA_ROOT}: {missing}")

    per_session = {}
    for session in sorted(loaders):
        t0 = time.time()
        # Per-session seed, as in fine-tuning; the read-out itself has no randomness.
        set_seed(args.seed + int(hashlib.md5(session.encode()).hexdigest(), 16) % (2 ** 31))
        r = evaluate_session(encoder, loaders[session], session, args, tukey)
        per_session[session] = r
        print(f"  {session:<18} alpha={r['alpha']:<7g} R2={r['r2']:.4f}  "
              f"(vx {r['r2_vx']:.4f}, vy {r['r2_vy']:.4f})  "
              f"[{r['n_fit_segments']} fit / {r['n_test_segments']} test segments, "
              f"{time.time() - t0:.0f} s]", flush=True)

    mean_r2 = float(np.mean([r['r2'] for r in per_session.values()]))
    print(f"  {'mean':<18} R2={mean_r2:.4f}  over {len(per_session)} sessions")

    out = Path(args.out) if args.out else (
        Path('output') / f"labelfree_{Path(args.ckpt).stem}_{args.dataset}_s{args.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        'checkpoint': str(args.ckpt), 'encoder': mode, 'dataset': args.dataset,
        'seed': args.seed, 'splits_file': str(args.splits_file), 'flint_tukey': tukey,
        'context': args.context, 'alpha_grid': list(ALPHA_GRID),
        'per_session': per_session, 'mean_r2': mean_r2,
    }
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved to {out}")


if __name__ == '__main__':
    main()
