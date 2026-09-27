#!/usr/bin/env python3
"""Rebuild the canonical 72/8/20 split of the eight held-out sessions.

Each held-out session is cut into non-overlapping 5 s segments, exactly as the data loader
cuts it. For every seed (42, 123, 456) the segments are permuted by torch.randperm after
seeding with seed + md5(session) and the permutation is cut 72/8/20 into training, validation
and test folds. The held-out Flint sessions read with their artifact windows removed
(lfp_tukey, see make_flint_tukey.py) have fewer segments and get their own entries, drawn over
10 s pairs so that both halves of a 10 s window land in the same fold, as in
CrossModalDistill.

The default output is splits/canonical_splits_728020.json, which this script reproduces byte
for byte from the preprocessed data.

Usage:
  python scripts/make_splits.py [--out splits/canonical_splits_728020.json]
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from utils.dataset import (HELDOUT_FLINT, HELDOUT_MAKIN, SEGMENT_LENGTH,  # noqa: E402
                           discover_sessions)

SEEDS = (42, 123, 456)
TRAIN_FRAC, VAL_FRAC = 0.72, 0.08


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def n_segments(path):
    """Number of non-overlapping 5 s segments, counted as the loader counts them."""
    T = np.load(path, allow_pickle=True)["lfp_data"].shape[-1]
    return len(range(0, T - SEGMENT_LENGTH + 1, SEGMENT_LENGTH))


def draw(session, n, seed, paired):
    set_seed(seed + int(hashlib.md5(session.encode()).hexdigest(), 16) % (2 ** 31))
    if paired:
        n_pair = n // 2
        order = torch.randperm(n_pair).tolist()
        idx = [2 * j + h for j in order for h in (0, 1)]
        # Cut on pair boundaries. round() guards against 0.72 + 0.08 = 0.7999999999999999.
        n_tr = 2 * int(round(TRAIN_FRAC * n_pair, 9))
        n_va = 2 * int(round((TRAIN_FRAC + VAL_FRAC) * n_pair, 9))
    else:
        idx = torch.randperm(n).tolist()
        n_tr = int(round(TRAIN_FRAC * n, 9))
        n_va = int(round((TRAIN_FRAC + VAL_FRAC) * n, 9))
    return dict(n=n, paired_10s=bool(paired), train=idx[:n_tr], val=idx[n_tr:n_va],
                test=idx[n_va:])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(REPO / "splits" / "canonical_splits_728020.json"))
    args = ap.parse_args()

    held = sorted(HELDOUT_MAKIN) + sorted(HELDOUT_FLINT)
    plain = discover_sessions(tukey_flint=False)
    tukey = discover_sessions(datasets=["flint"], tukey_flint=True)
    out = {}
    for sub in ("lfp", "lfp_tukey"):
        for sess in held:
            if sub == "lfp":
                path = plain[sess]["path"]
            elif sess in tukey and Path(tukey[sess]["path"]).parent.name == "lfp_tukey":
                path = tukey[sess]["path"]
            else:
                continue                      # only the held-out Flint sessions are filtered
            n = n_segments(path)
            paired = sub == "lfp_tukey"
            for seed in SEEDS:
                out[f"{sub}|{sess}|{seed}"] = draw(sess, n, seed, paired)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}: {len(out)} entries")


if __name__ == "__main__":
    main()
