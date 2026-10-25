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

--chain causal builds the copy used to score the held-out sessions under causal deployment:
the decimation and band-limiting of steps 1-3 read no later sample (causal IIR decimation,
causal notch, 0.05 Hz high-pass started from the first sample, minimum-phase FIR band-pass,
causal 8x decimation to 100 Hz). Only the interpolation onto the kinematic clock reads at
most the next 100 Hz sample. The z-score of step 4 is kept so that the file has the same
layout; evaluation re-standardises the signal with past samples only (utils/dataset.py).

Output: $REALM_DATA/makin/preprocessed/lfp/<session>_rawlfp.npz (lfp_causal/ for --chain
causal) with lfp_data (96, 1, T) and targets (T, 2).

Usage:
  python scripts/preprocess_makin.py                             # every session
  python scripts/preprocess_makin.py --session indy_20160622_01
  python scripts/preprocess_makin.py --chain causal --heldout    # held-out sessions, causal
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
                                 align_signal, causal_band_limit, causal_decimate,
                                 common_average_reference, downsample_signal,
                                 highpass_filter, lowpass_filter, notch_filter, z_score)

READ_BLOCK = 8   # channels read from the NWB file at a time (the values are the same either way)
HELDOUT = ["indy_20160622_01", "indy_20160630_01", "indy_20160915_01", "indy_20161013_03",
           "indy_20170124_01"]


def velocity(mat_path, causal=False):
    """Cursor velocity at 100 Hz, trimmed to the task span and z-scored; also its time vector.

    The causal chain z-scores in float64 and casts afterwards, as in the paper's runs.
    """
    with h5py.File(str(mat_path), 'r') as f:
        t_kin = f['t'][0]
        cursor_pos = f['cursor_pos'][:].T          # (n, 2)
        target_pos = f['target_pos'][:].T          # (n, 2)

    num_steps = int((t_kin[-1] - t_kin[0]) / (1.0 / TARGET_FS))
    vel = np.gradient(cursor_pos, t_kin, axis=0)
    vel, t = resample(vel, num_steps, t=t_kin)
    if not causal:
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
    return ((vel - vel.mean(axis=0)) / std).astype(np.float32), t


def broadband_to_lfp(nwb_path, causal=False):
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
                x = decimate(x[:n1 * 4], 4, ftype='iir', zero_phase=not causal)
                x = decimate(x[:n2 * 4], 4, ftype='iir', zero_phase=not causal)
                x = decimate(x[:n3 * 2], 2, ftype='iir', zero_phase=not causal)
                raw[:, c0 + j] = x.astype(np.float32)
            del block
            print(f"    decimated channels {c1}/{n_channels}", flush=True)

    if causal:
        broadband_fs = 1.0 / (broadband_t[1] - broadband_t[0])
        fs = n3 / (n_samples / broadband_fs)        # ~763 Hz
        raw = common_average_reference(causal_band_limit(raw, fs))
        raw = causal_decimate(raw, int(round(fs / TARGET_FS)))
        t = np.linspace(broadband_t[0], broadband_t[0] + n_samples / broadband_fs, len(raw))
        return raw, t, n_channels

    t = np.linspace(broadband_t[0], broadband_t[-1], n3, dtype=np.float64)
    fs = n3 / (broadband_t[-1] - broadband_t[0])    # ~763 Hz

    raw -= raw.mean(axis=0)
    raw = notch_filter(raw, fs, freq=60, quality_factor=30)
    raw = highpass_filter(raw, fs, cutoff=LFP_HP_CUTOFF)
    raw = lowpass_filter(raw, fs, cutoff=LFP_LP_CUTOFF)
    raw = common_average_reference(raw)
    raw, t = downsample_signal(raw, fs, TARGET_FS, t=t)
    return raw, t, n_channels


def process_session(session, raw_dir, out_dir, force=False, causal=False):
    out_path = out_dir / f"{session}_rawlfp.npz"
    if out_path.exists() and not force:
        print(f"  skip (exists): {out_path}")
        return
    print(f"  {session}", flush=True)
    vel, t = velocity(raw_dir / f"{session}.mat", causal)
    lfp, lfp_t, n_channels = broadband_to_lfp(raw_dir / f"{session}.nwb", causal=causal)
    if causal:
        # Keep the kinematic samples inside the LFP record; no extrapolation.
        keep = (t >= lfp_t[0]) & (t <= lfp_t[-1])
        vel = vel[keep]
        lfp = z_score(align_signal(lfp, lfp_t, t[keep], dtype=np.float64), axis=0)
    else:
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
        **({'chain': 'causal'} if causal else {}),        # zero-phase files keep their layout
    )
    print(f"    saved {out_path} lfp {lfp.T.shape}, targets {vel.shape}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--session", default=None, help="process one session, e.g. indy_20160622_01")
    ap.add_argument("--raw_dir", default=str(DATA_ROOT / "makin" / "origin"),
                    help="directory with <session>.nwb and <session>.mat")
    ap.add_argument("--chain", default="zero_phase", choices=["zero_phase", "causal"],
                    help="zero_phase: the training corpus; causal: the held-out sessions "
                         "under causal deployment")
    ap.add_argument("--heldout", action="store_true", help="only the five held-out sessions")
    ap.add_argument("--out_dir", default=None,
                    help="default: makin/preprocessed/lfp/ (lfp_causal/ for --chain causal)")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args()

    causal = args.chain == "causal"
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir or DATA_ROOT / "makin" / "preprocessed" /
                   ("lfp_causal" if causal else "lfp"))
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = ([args.session] if args.session else HELDOUT if args.heldout
                else sorted(p.stem for p in raw_dir.glob("*.nwb")))
    print(f"Makin: {len(sessions)} session(s), {raw_dir} -> {out_dir}")
    for s in sessions:
        if not (raw_dir / f"{s}.mat").exists():
            print(f"  skip {s}: no {s}.mat")
            continue
        try:
            process_session(s, raw_dir, out_dir, force=args.force, causal=causal)
        except Exception:
            print(f"  failed: {s}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
