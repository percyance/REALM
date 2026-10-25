"""Signal-processing helpers shared by the preprocessing scripts.

Two chains are provided. The zero-phase chain follows CrossModalDistillation (ShanechiLab),
so that REALM is trained on LFP prepared exactly as in that work; it filters forward and
backward and resamples in the Fourier domain, so it reads later samples. The causal chain
is the one used to score the held-out sessions under causal deployment: every filter and
the decimation read only past samples. All arrays are (n_samples, n_channels).
"""
import os
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import (butter, decimate, filtfilt, firwin, iirnotch, lfilter,
                          minimum_phase, resample, sosfilt, sosfilt_zi)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("REALM_DATA", REPO_ROOT / "data"))

TARGET_FS = 100        # Hz, one sample every 10 ms
LFP_HP_CUTOFF = 0.05   # Hz
LFP_LP_CUTOFF = 50     # Hz
CAUSAL_FIR_TAPS = 201  # minimum-phase FIR of the causal chain


def notch_filter(data, fs, freq=60, quality_factor=30):
    """Zero-phase notch at `freq` and every harmonic below Nyquist."""
    filtered = data.astype(np.float64)
    harmonic = freq
    while harmonic < fs / 2:
        b, a = iirnotch(harmonic, quality_factor, fs)
        filtered = filtfilt(b, a, filtered, axis=0)
        harmonic += freq
    return filtered.astype(np.float32)


def lowpass_filter(data, fs, cutoff, order=2):
    """Zero-phase Butterworth low-pass."""
    b, a = butter(order, cutoff / (fs / 2), btype='lowpass')
    return filtfilt(b, a, data.astype(np.float64), axis=0).astype(np.float32)


def highpass_filter(data, fs, cutoff, order=2):
    """Zero-phase Butterworth high-pass."""
    b, a = butter(order, cutoff / (fs / 2), btype='highpass')
    return filtfilt(b, a, data.astype(np.float64), axis=0).astype(np.float32)


def common_average_reference(data):
    """Subtract the mean across channels at every time point."""
    return (data - np.mean(data, axis=1, keepdims=True)).astype(np.float32)


def downsample_signal(data, original_fs, target_fs, t):
    """FFT resampling to `target_fs`; returns the resampled data and its time vector."""
    num_samples = int(len(data) * target_fs / original_fs)
    ds_data = resample(data, num_samples).astype(np.float32)
    ds_t = np.linspace(t[0], t[-1], num_samples).astype(np.float64)
    return ds_data, ds_t


def align_signal(data, t, align_t, dtype=np.float32):
    """Linearly interpolate each column of `data` from time vector `t` onto `align_t`."""
    if data.ndim == 1:
        data = data[:, None]
    aligned = np.zeros((len(align_t), data.shape[1]), dtype=dtype)
    for i in range(data.shape[1]):
        interp_fn = interp1d(t, data[:, i], kind='linear',
                             bounds_error=False, fill_value='extrapolate')
        aligned[:, i] = interp_fn(align_t)
    return aligned


def z_score(data, axis=0):
    """Z-score along `axis`; constant columns are left centred."""
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    std[std == 0] = 1
    return ((data - mean) / std).astype(np.float32)


# ── Causal chain ──────────────────────────────────────────────────────────

def causal_band_limit(data, fs, taps=CAUSAL_FIR_TAPS):
    """Causal counterpart of mean removal + notch + 0.05-50 Hz band-pass.

    Notch 60 Hz and every harmonic below Nyquist (lfilter), a second-order Butterworth
    high-pass at 0.05 Hz applied once and started from the steady state of the first
    sample (so the DC offset is rejected without a whole-record mean), then a
    minimum-phase FIR band-pass at 0.05-50 Hz. No step reads a later sample.
    """
    x = data.astype(np.float64)
    nyq = fs / 2
    harmonic = 60
    while harmonic < nyq:
        b, a = iirnotch(harmonic, 30, fs)
        x = lfilter(b, a, x, axis=0)
        harmonic += 60
    sos = butter(2, LFP_HP_CUTOFF / nyq, btype='highpass', output='sos')
    zi = sosfilt_zi(sos)[:, :, None] * x[0][None, None, :]
    x, _ = sosfilt(sos, x, axis=0, zi=zi)
    fir = minimum_phase(firwin(int(taps) | 1, [LFP_HP_CUTOFF, LFP_LP_CUTOFF],
                               pass_zero=False, fs=fs), method='hilbert')
    return lfilter(fir, 1.0, x, axis=0).astype(np.float32)


def causal_decimate(data, q):
    """Decimate by the integer factor `q` in stages of at most 10, with causal IIR filters."""
    stages = []
    for f in (10, 8, 6, 5, 4, 3, 2):
        while q % f == 0 and q > 1:
            stages.append(f)
            q //= f
    if q != 1:
        stages.append(q)
    for f in stages:
        data = decimate(data, f, ftype='iir', zero_phase=False, axis=0)
    return data.astype(np.float32)
