"""Signal-processing helpers shared by the preprocessing scripts.

The filter chain follows CrossModalDistillation (ShanechiLab), so that REALM is trained and
scored on LFP prepared exactly as in that work. All arrays are (n_samples, n_channels).
"""
import os
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirnotch, resample

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("REALM_DATA", REPO_ROOT / "data"))

TARGET_FS = 100        # Hz, one sample every 10 ms
LFP_HP_CUTOFF = 0.05   # Hz
LFP_LP_CUTOFF = 50     # Hz


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


def align_signal(data, t, align_t):
    """Linearly interpolate each column of `data` from time vector `t` onto `align_t`."""
    if data.ndim == 1:
        data = data[:, None]
    aligned = np.zeros((len(align_t), data.shape[1]), dtype=np.float32)
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
