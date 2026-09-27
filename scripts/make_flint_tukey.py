#!/usr/bin/env python3
"""Remove the artifact windows of the three held-out Flint sessions, as CrossModalDistill does.

CrossModalDistill drops every 10 s window of a Flint session whose velocity standard deviation
lies above Tukey's upper fence (75th percentile + 1.5 x interquartile range); Makin has no
such rule. The rule looks at behaviour only, so it is applied to the held-out evaluation
sessions alone, and whole 10 s windows are dropped so that every 5 s segment stays
continuous in time.

Input:  $REALM_DATA/flint/preprocessed/lfp/<session>_rawlfp.npz
Output: $REALM_DATA/flint/preprocessed/lfp_tukey/<session>_rawlfp.npz, the same file with the
        rejected windows removed and the kept-window mask stored as `tukey_kept_windows`.
        Evaluation reads these files for Flint (--tukey_flint in the evaluation scripts).
"""
import argparse
import os
from pathlib import Path

import numpy as np

DATA_ROOT = Path(os.environ.get("REALM_DATA", Path(__file__).resolve().parent.parent / "data"))
WIN = 1000                                     # 10 s at 100 Hz
SESSIONS = ["Flint_e1_1", "Flint_e4_1", "Flint_e5_2"]


def keep_mask(vel, win=WIN):
    """True for every 10 s window whose velocity std is at or below Tukey's upper fence."""
    n = len(vel) // win
    s = vel[: n * win].reshape(n, win, -1).std(axis=(1, 2))
    p75, p25 = np.percentile(s, 75), np.percentile(s, 25)
    return s <= p75 + 1.5 * (p75 - p25), n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in_dir", default=str(DATA_ROOT / "flint" / "preprocessed" / "lfp"))
    ap.add_argument("--out_dir", default=str(DATA_ROOT / "flint" / "preprocessed" / "lfp_tukey"))
    ap.add_argument("--sessions", nargs="+", default=SESSIONS)
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sess in args.sessions:
        z = np.load(in_dir / f"{sess}_rawlfp.npz", allow_pickle=True)
        keep, n_win = keep_mask(z["targets"].astype(np.float32))
        idx = np.flatnonzero(np.repeat(keep, WIN))
        out = {k: z[k] for k in z.files}
        out["lfp_data"] = z["lfp_data"][..., idx]
        out["targets"] = z["targets"][idx]
        out["tukey_kept_windows"] = keep
        out["tukey_note"] = "velocity-std Tukey fence on 10 s windows, CMD flint_dataset.py:265"
        np.savez_compressed(out_dir / f"{sess}_rawlfp.npz", **out)
        print(f"{sess}: {n_win} windows, dropped {int((~keep).sum())}, "
              f"{len(idx)} samples kept -> {out_dir / (sess + '_rawlfp.npz')}")


if __name__ == "__main__":
    main()
