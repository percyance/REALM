#!/usr/bin/env python3
"""Flint (Monkey C): trial-wise recordings -> 100 Hz LFP and hand velocity.

The raw data are the DREAM files Flint_2012_e1.mat ... Flint_2012_e5.mat in
$REALM_DATA/flint/original/; every Subject of file e<k> is one session, Flint_e<k>_<subject>.
The chain follows CrossModalDistillation:
  1. concatenate the trials (LFP at 2 kHz, HandVel x/y)
  2. remove the mean, notch 60 Hz and harmonics, band-limit to 0.05-50 Hz (zero-phase)
  3. common average reference, FFT-resample to a uniform 100 Hz grid
  4. interpolate the velocity onto that grid, z-score it per axis
  5. z-score every LFP channel

Output: $REALM_DATA/flint/preprocessed/lfp/Flint_e<k>_<subject>_rawlfp.npz with
lfp_data (95, 1, T) and targets (T, 2).

Usage:
  python scripts/preprocess_flint.py                        # every session
  python scripts/preprocess_flint.py --session Flint_e4_1
"""
import argparse
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import resample

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing_utils import (DATA_ROOT, LFP_HP_CUTOFF, LFP_LP_CUTOFF, TARGET_FS,  # noqa: E402
                                 align_signal, common_average_reference, highpass_filter,
                                 lowpass_filter, notch_filter, z_score)

LFP_RAW_FS = 2000     # Hz


def process_file(mat_path, out_dir, only=None, force=False):
    """Process every subject (session) of one Flint_2012_e<k>.mat file."""
    day = mat_path.stem.split('_')[-1]                       # "e4"
    if only and not only.startswith(f"Flint_{day}_"):
        return                                               # the session is in another file
    print(f"  loading {mat_path.name}", flush=True)
    subjects = sio.loadmat(str(mat_path), squeeze_me=True)['Subject']
    if subjects.ndim == 0:                                   # a single subject
        subjects = [subjects[()]]

    for k, subj in enumerate(subjects):
        session = f"Flint_{day}_{k + 1}"
        if only and session != only:
            continue
        out_path = out_dir / f"{session}_rawlfp.npz"
        if out_path.exists() and not force:
            print(f"  skip (exists): {out_path}")
            continue
        print(f"  {session}", flush=True)

        trials = subj['Trial']
        if not hasattr(trials, '__len__'):
            trials = [trials]
        t0, t_end = trials[0]['Time'][0], trials[-1]['Time'][-1]
        num_steps = int((t_end - t0) / (1.0 / TARGET_FS))
        t_uniform = np.linspace(t0, t_end, num_steps, dtype=np.float64)

        lfps, vels, times = [], [], []
        for trial in trials:
            channels = []
            for ch in trial['Neuron']['LFP']:
                if hasattr(ch, '__len__') and len(ch) > 0 and ch.shape[-1] != 0:
                    channels.append((ch.flatten() if ch.ndim > 1 else ch).astype(np.float32))
            if not channels:
                continue
            n = min(len(ch) for ch in channels)
            lfps.append(np.stack([ch[:n] for ch in channels], axis=-1))
            vels.append(trial['HandVel'][:, :2].astype(np.float32))
            times.append(trial['Time'].astype(np.float64).reshape(-1))
        if not lfps:
            print("    no valid trials")
            continue

        lfp = np.concatenate(lfps, axis=0)                   # (samples at 2 kHz, channels)
        vel = np.concatenate(vels, axis=0)
        time = np.concatenate(times, axis=0)
        n_channels = lfp.shape[1]

        lfp -= lfp.mean(axis=0)
        lfp = notch_filter(lfp, LFP_RAW_FS, freq=60, quality_factor=30)
        lfp = highpass_filter(lfp, LFP_RAW_FS, cutoff=LFP_HP_CUTOFF)
        lfp = lowpass_filter(lfp, LFP_RAW_FS, cutoff=LFP_LP_CUTOFF)
        lfp = common_average_reference(lfp)
        lfp = resample(lfp, num_steps).astype(np.float32)   # onto the uniform 100 Hz grid

        vel = align_signal(vel, time, t_uniform)
        std = vel.std(axis=0)
        std[std == 0] = 1
        vel = ((vel - vel.mean(axis=0)) / std).astype(np.float32)
        lfp = z_score(lfp, axis=0)

        np.savez_compressed(
            out_path,
            lfp_data=lfp.T[:, np.newaxis, :],                # (channels, 1, T)
            targets=vel,                                     # (T, 2)
            bin_size_s=1.0 / TARGET_FS,
            n_channels=np.int32(n_channels),              # int32, as in the released files
            n_bands=np.int32(1),
            dataset='flint',
            monkey='MonkeyC',
            session=session,
        )
        print(f"    saved {out_path} lfp {lfp.T.shape}, targets {vel.shape}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--session", default=None, help="process one session, e.g. Flint_e4_1")
    ap.add_argument("--raw_dir", default=str(DATA_ROOT / "flint" / "original"),
                    help="directory with Flint_2012_e1.mat ... Flint_2012_e5.mat")
    ap.add_argument("--out_dir", default=str(DATA_ROOT / "flint" / "preprocessed" / "lfp"))
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args()

    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Only the five experiment files (the DREAM folder may hold other copies).
    files = sorted(p for p in raw_dir.glob("Flint_2012_e*.mat")
                   if re.fullmatch(r"Flint_2012_e\d+", p.stem))
    print(f"Flint: {len(files)} file(s), {raw_dir} -> {out_dir}")
    for f in files:
        try:
            process_file(f, out_dir, only=args.session, force=args.force)
        except Exception:
            print(f"  failed: {f.name}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
