#!/usr/bin/env python3
"""Makin (Monkey I): broadband recordings -> 100 Hz LFP and cursor velocity.

For each session the raw data are <session>.nwb (broadband, ~24.4 kHz) and <session>.mat
(cursor and target positions) in $REALM_DATA/makin/origin/. The chain follows
CrossModalDistillation:
  1. decimate the broadband signal 32x (4 x 4 x 2, zero-phase IIR) to ~763 Hz
  2. remove the mean, notch 60 Hz and harmonics, band-limit to 0.05-50 Hz (zero-phase)
  3. common average reference, FFT-resample to 100 Hz, interpolate onto the kinematic clock
  4. z-score every channel
  5. velocity = d(cursor position)/dt, FFT-resampled to 100 Hz, trimmed to the span between
     the first and last target change, z-scored per axis

Output: $REALM_DATA/makin/preprocessed/lfp/<session>_rawlfp.npz with lfp_data (96, 1, T) and
targets (T, 2).

Usage:
  python scripts/preprocess_makin.py                             # every session
  python scripts/preprocess_makin.py --session indy_20160622_01
"""
import argparse
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import decimate, resample

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing_utils import (DATA_ROOT, LFP_HP_CUTOFF, LFP_LP_CUTOFF, TARGET_FS,  # noqa: E402
                                 align_signal, common_average_reference, downsample_signal,
                                 highpass_filter, lowpass_filter, notch_filter, z_score)

READ_BLOCK = 8   # channels read from the NWB file at a time (the values are the same either way)


def velocity(mat_path):
    """Cursor velocity at 100 Hz, trimmed to the task span and z-scored; also its time vector."""
    with h5py.File(str(mat_path), 'r') as f:
        t_kin = f['t'][0]
        cursor_pos = f['cursor_pos'][:].T          # (n, 2)
        target_pos = f['target_pos'][:].T          # (n, 2)

    num_steps = int((t_kin[-1] - t_kin[0]) / (1.0 / TARGET_FS))
    vel = np.gradient(cursor_pos, t_kin, axis=0)
    vel, t = resample(vel, num_steps, t=t_kin)
    vel = vel.astype(np.float32)
    t = t.astype(np.float64)

    # Keep the span from the first to the last target change.
    trial_start = np.where(np.diff(target_pos, axis=0).sum(axis=1) != 0)[0] + 1
    if len(trial_start) >= 2:
        trial_start = (trial_start * (num_steps / target_pos.shape[0])).astype(np.int32)
        vel = vel[trial_start[0]:trial_start[-1]]
        t = t[trial_start[0]:trial_start[-1]]

    std = vel.std(axis=0)
    std[std == 0] = 1
    return (vel - vel.mean(axis=0)) / std, t


def broadband_to_lfp(nwb_path):
    """Decimated, filtered and re-referenced LFP at 100 Hz with its time vector."""
    with h5py.File(str(nwb_path), 'r') as f:
        ts = f['acquisition']['timeseries']['broadband']
        conversion = ts['data'].attrs.get('conversion', 1.0)
        shape = ts['data'].shape
        broadband_t = ts['timestamps'][:]

    transpose = shape[0] < shape[1]                 # stored as (channels, samples)?
    n_samples = shape[1] if transpose else shape[0]
    n_channels = shape[0] if transpose else shape[1]
    n1 = n_samples // 4
    n2 = n1 // 4
    n3 = n2 // 2

    # Decimate 32x one channel at a time, never holding the full broadband array.
    raw = np.zeros((n3, n_channels), dtype=np.float32)
    with h5py.File(str(nwb_path), 'r') as f:
        data = f['acquisition']['timeseries']['broadband']['data']
        for c0 in range(0, n_channels, READ_BLOCK):
            c1 = min(c0 + READ_BLOCK, n_channels)
            block = data[c0:c1, :].T if transpose else data[:, c0:c1]
            for j in range(c1 - c0):
                x = block[:, j].astype(np.float64) * conversion
                x = decimate(x[:n1 * 4], 4, ftype='iir', zero_phase=True)
                x = decimate(x[:n2 * 4], 4, ftype='iir', zero_phase=True)
                x = decimate(x[:n3 * 2], 2, ftype='iir', zero_phase=True)
                raw[:, c0 + j] = x.astype(np.float32)
            del block
            print(f"    decimated channels {c1}/{n_channels}", flush=True)

    t = np.linspace(broadband_t[0], broadband_t[-1], n3, dtype=np.float64)
    fs = n3 / (broadband_t[-1] - broadband_t[0])    # ~763 Hz

    raw -= raw.mean(axis=0)
    raw = notch_filter(raw, fs, freq=60, quality_factor=30)
    raw = highpass_filter(raw, fs, cutoff=LFP_HP_CUTOFF)
    raw = lowpass_filter(raw, fs, cutoff=LFP_LP_CUTOFF)
    raw = common_average_reference(raw)
    raw, t = downsample_signal(raw, fs, TARGET_FS, t=t)
    return raw, t, n_channels


def process_session(session, raw_dir, out_dir, force=False):
    out_path = out_dir / f"{session}_rawlfp.npz"
    if out_path.exists() and not force:
        print(f"  skip (exists): {out_path}")
        return
    print(f"  {session}", flush=True)
    vel, t = velocity(raw_dir / f"{session}.mat")
    lfp, lfp_t, n_channels = broadband_to_lfp(raw_dir / f"{session}.nwb")
    lfp = z_score(align_signal(lfp, lfp_t, t), axis=0)     # onto the kinematic clock

    np.savez_compressed(
        out_path,
        lfp_data=lfp.T[:, np.newaxis, :],                 # (channels, 1, T)
        targets=vel.astype(np.float32),                   # (T, 2)
        bin_size_s=1.0 / TARGET_FS,
        n_channels=np.int32(n_channels),                  # int32, as in the released files
        n_bands=np.int32(1),
        dataset='makin',
        monkey='Indy',
        session=session,
    )
    print(f"    saved {out_path} lfp {lfp.T.shape}, targets {vel.shape}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--session", default=None, help="process one session, e.g. indy_20160622_01")
    ap.add_argument("--raw_dir", default=str(DATA_ROOT / "makin" / "origin"),
                    help="directory with <session>.nwb and <session>.mat")
    ap.add_argument("--out_dir", default=str(DATA_ROOT / "makin" / "preprocessed" / "lfp"))
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args()

    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = [args.session] if args.session else sorted(p.stem for p in raw_dir.glob("*.nwb"))
    print(f"Makin: {len(sessions)} session(s), {raw_dir} -> {out_dir}")
    for s in sessions:
        if not (raw_dir / f"{s}.mat").exists():
            print(f"  skip {s}: no {s}.mat")
            continue
        try:
            process_session(s, raw_dir, out_dir, force=args.force)
        except Exception:
            print(f"  failed: {s}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
