"""
Unified multi-dataset loader for LFP foundation model pipeline.

Handles 5 dataset sources with varying channel counts:
  - Brochier (2 sessions, 96ch) -- pretrain only
  - DANDI:000070 (46 sessions, 96ch) -- pretrain only
  - DANDI:000121 (40 sessions, varies ch) -- pretrain only
  - Makin (30 sessions, 96ch, velocity targets) -- finetune/distill/eval
  - Flint (12 sessions, 95ch, velocity targets) -- finetune/distill/eval

All data is in _rawlfp.npz format:
  lfp_data:  (n_channels, 1, T)  float32
  targets:   (T, 2)             float32  (velocity x, y)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional


def _np_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert numpy array to torch tensor (bypasses broken numpy-torch bridge)."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    t = torch.frombuffer(bytearray(arr.data), dtype=torch.float32)
    return t.reshape(arr.shape).clone()

# ── Project root ──────────────────────────────────────────────────────────
# dataset.py is at <repo>/utils/dataset.py.
# Locate `data/` either as:
#   - sibling of utils/ (i.e. <repo>/data/) — recommended
#   - one level up      (i.e. <repo>/../data/) — server symlink layout
#   - via env var REALM_DATA_DIR — explicit override
import os as _os
_REPO = Path(__file__).resolve().parent.parent
if _os.environ.get('REALM_DATA_DIR'):
    ROOT = Path(_os.environ['REALM_DATA_DIR']).resolve().parent
else:
    _candidates = [_REPO, _REPO.parent, _REPO.parent.parent]
    # Require an actual processed subdir, not just any 'data/' (which may just hold a README).
    _has_data = lambda c: any(
        (c / 'data' / s / 'preprocessed').exists()
        for s in ['makin', 'flint', 'brochier', 'dandi_000070', 'dandi_000121'])
    ROOT = next((c for c in _candidates if _has_data(c)), _REPO)

# ── Constants ─────────────────────────────────────────────────────────────
MAX_CHANNELS = 96
SEGMENT_LENGTH = 500  # 5s @ 100Hz, matching CrossModalDistill
BATCH_SIZE = 64

HELDOUT_MAKIN = [
    "indy_20160622_01", "indy_20160630_01", "indy_20160915_01",
    "indy_20161005_06", "indy_20170124_01",
]
HELDOUT_FLINT = ["Flint_e1_1", "Flint_e4_1", "Flint_e5_2"]
HELDOUT_SESSIONS = set(HELDOUT_MAKIN + HELDOUT_FLINT)


# ── Session discovery ─────────────────────────────────────────────────────

def discover_sessions(datasets: Optional[List[str]] = None) -> OrderedDict:
    """Discover all preprocessed sessions across datasets.

    Returns:
        OrderedDict {session_stem: {'path': str, 'dataset': str}}
    """
    search_dirs = {
        'brochier': ROOT / "data" / "brochier" / "preprocessed",
        'makin': ROOT / "data" / "makin" / "preprocessed",
        'flint': ROOT / "data" / "flint" / "preprocessed",
        'dandi_000070': ROOT / "data" / "dandi_000070" / "preprocessed",
        'dandi_000121': ROOT / "data" / "dandi_000121" / "preprocessed",
    }
    if datasets is not None:
        search_dirs = {k: v for k, v in search_dirs.items() if k in datasets}

    sessions = OrderedDict()
    for dataset_name, data_dir in search_dirs.items():
        if not data_dir.exists():
            continue
        npz_files = sorted(data_dir.glob("*_rawlfp.npz"))
        for f in npz_files:
            stem = f.stem.replace("_rawlfp", "")
            sessions[stem] = {'path': str(f), 'dataset': dataset_name}
    return sessions


# ── Tukey's fences outlier filtering (for Flint) ─────────────────────────

def tukey_filter_segments(indices, targets, segment_length):
    """Remove outlier segments using Tukey's fences on velocity std."""
    stds = []
    for start in indices:
        seg = targets[start:start + segment_length]
        stds.append(np.std(seg))
    stds = np.array(stds)
    q25, q75 = np.percentile(stds, [25, 75])
    iqr = q75 - q25
    threshold = q75 + 1.5 * iqr
    return [idx for idx, s in zip(indices, stds) if s <= threshold]


# ── R² metric ────────────────────────────────────────────────────────────

def compute_r2(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute R² (coefficient of determination) — combined across all dims."""
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - targets.mean(dim=0, keepdim=True)) ** 2)
    return (1 - ss_res / (ss_tot + 1e-8)).item()


def compute_r2_per_axis(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute R² averaged across behavior dimensions (per-axis).
    Matches CrossModalDistill (Erturk et al., NeurIPS 2025) protocol."""
    r2s = []
    for i in range(targets.shape[-1]):
        ss_res = torch.sum((targets[..., i] - preds[..., i]) ** 2)
        ss_tot = torch.sum((targets[..., i] - targets[..., i].mean()) ** 2)
        r2s.append((1 - ss_res / (ss_tot + 1e-8)).item())
    return float(np.mean(r2s))


# ── Datasets ──────────────────────────────────────────────────────────────

class PretrainSegmentDataset(Dataset):
    """Multi-session pretraining dataset (LFP only, no targets).

    Each sample: {'lfp': (96, 1, T), 'session_id': int, 'channel_mask': (96,)}
    Channels zero-padded to MAX_CHANNELS.
    """

    def __init__(
        self,
        session_list: List[Tuple[str, dict]],
        segment_length: int = SEGMENT_LENGTH,
        stride: int = None,
    ):
        self.segment_length = segment_length
        self.stride = stride if stride is not None else segment_length
        self.segments = []  # (session_idx, start_idx)
        self.session_data = []  # list of (lfp_data, n_channels_real)
        self.session_names = []

        for sess_idx, (name, info) in enumerate(session_list):
            data = np.load(info['path'], allow_pickle=True)
            lfp = data['lfp_data'].astype(np.float32)  # (n_ch, 1, T)
            n_ch = lfp.shape[0]

            # Zero-pad to MAX_CHANNELS
            if n_ch < MAX_CHANNELS:
                pad = np.zeros((MAX_CHANNELS - n_ch, lfp.shape[1], lfp.shape[2]),
                               dtype=np.float32)
                lfp = np.concatenate([lfp, pad], axis=0)

            self.session_data.append((lfp, n_ch))
            self.session_names.append(name)

            # Segments with configurable stride (default: non-overlapping)
            T = lfp.shape[2]
            for start in range(0, T - segment_length + 1, self.stride):
                self.segments.append((sess_idx, start))

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        sess_idx, start = self.segments[idx]
        lfp, n_ch = self.session_data[sess_idx]
        end = start + self.segment_length

        lfp_seg = lfp[:, :, start:end].copy()  # (96, 1, seg_len)

        # Channel mask: True for valid channels
        channel_mask = np.zeros(MAX_CHANNELS, dtype=np.float32)
        channel_mask[:n_ch] = 1.0

        return {
            'lfp': _np_to_tensor(lfp_seg),
            'session_id': torch.tensor(sess_idx, dtype=torch.long),
            'channel_mask': _np_to_tensor(channel_mask),
        }


class SupervisedSegmentDataset(Dataset):
    """Multi-session supervised dataset (LFP + velocity targets).

    Each sample: {'lfp': (96, 1, T), 'target': (T, 2),
                  'session_id': int, 'channel_mask': (96,)}
    """

    def __init__(
        self,
        session_list: List[Tuple[str, dict]],
        segment_length: int = SEGMENT_LENGTH,
        filter_flint: bool = True,
    ):
        self.segment_length = segment_length
        self.segments = []
        self.session_data = []
        self.session_names = []

        for sess_idx, (name, info) in enumerate(session_list):
            data = np.load(info['path'], allow_pickle=True)
            lfp = data['lfp_data'].astype(np.float32)
            targets = data['targets'].astype(np.float32)
            n_ch = lfp.shape[0]

            # Zero-pad channels
            if n_ch < MAX_CHANNELS:
                pad = np.zeros((MAX_CHANNELS - n_ch, lfp.shape[1], lfp.shape[2]),
                               dtype=np.float32)
                lfp = np.concatenate([lfp, pad], axis=0)

            self.session_data.append((lfp, targets, n_ch))
            self.session_names.append(name)

            # Non-overlapping segment indices
            T = lfp.shape[2]
            indices = list(range(0, T - segment_length + 1, segment_length))

            # Tukey's fences for Flint
            if filter_flint and info['dataset'] == 'flint':
                indices = tukey_filter_segments(indices, targets, segment_length)

            for start in indices:
                self.segments.append((sess_idx, start))

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        sess_idx, start = self.segments[idx]
        lfp, targets, n_ch = self.session_data[sess_idx]
        end = start + self.segment_length

        lfp_seg = lfp[:, :, start:end].copy()
        tgt_seg = targets[start:end].copy()

        channel_mask = np.zeros(MAX_CHANNELS, dtype=np.float32)
        channel_mask[:n_ch] = 1.0

        return {
            'lfp': _np_to_tensor(lfp_seg),
            'target': _np_to_tensor(tgt_seg),
            'session_id': torch.tensor(sess_idx, dtype=torch.long),
            'channel_mask': _np_to_tensor(channel_mask),
        }


# ── Dataloader factories ─────────────────────────────────────────────────

def create_pretrain_dataloaders(
    seed: int = 42,
    segment_length: int = SEGMENT_LENGTH,
    batch_size: int = BATCH_SIZE,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    stride: int = None,
    m1_only: bool = False,
) -> Tuple[DataLoader, DataLoader, int]:
    """Create pretrain dataloaders.

    Includes: Brochier + DANDI:000070 + DANDI:000121 + Makin (non-heldout)
              + Flint (non-heldout). Held-out sessions are excluded and
              reserved for per-session MAE finetune + linear probe eval.

    Args:
        stride: Segment stride. None = non-overlapping (stride=segment_length).
                Use stride=250 for 50% overlap (doubles data).

    Returns:
        (train_loader, val_loader, n_sessions)
    """
    all_sessions = discover_sessions(
        datasets=['brochier', 'dandi_000070', 'dandi_000121'])
    # Pretrain uses only unlabeled data (no Makin/Flint velocity targets)
    session_list = list(all_sessions.items())

    # Filter by brain region if specified
    if m1_only:
        before = len(session_list)
        session_list = [(name, info) for name, info in session_list
                        if 'PMd' not in name and '_B' not in name]
        print(f"Pretrain (M1 only): filtered {before} → {len(session_list)} sessions "
              f"(removed PMd/Array-B)")
    else:
        print(f"Pretrain: {len(session_list)} sessions "
              f"(Brochier + DANDI:000070 + DANDI:000121, no Makin/Flint)")

    dataset = PretrainSegmentDataset(session_list, segment_length, stride=stride)
    print(f"Pretrain: {len(dataset)} segments total"
          f" (stride={dataset.stride})")

    # Split by SESSION (not segment) to prevent data leakage
    n_sess = len(session_list)
    rng = np.random.RandomState(seed)
    sess_order = np.arange(n_sess)
    rng.shuffle(sess_order)
    n_val_sess = max(1, int(n_sess * val_ratio))
    val_session_set = set(sess_order[:n_val_sess].tolist())

    val_indices = [i for i, (sess_idx, _) in enumerate(dataset.segments)
                   if sess_idx in val_session_set]
    train_indices = [i for i, (sess_idx, _) in enumerate(dataset.segments)
                     if sess_idx not in val_session_set]

    print(f"Pretrain split: {len(train_indices)} train segs "
          f"({n_sess - n_val_sess} sessions), "
          f"{len(val_indices)} val segs ({n_val_sess} sessions)")

    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
        pin_memory=True,
    )
    return train_loader, val_loader, n_sess


def create_finetune_dataloaders(
    seed: int = 42,
    segment_length: int = SEGMENT_LENGTH,
    batch_size: int = BATCH_SIZE,
    val_ratio: float = 0.1,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, int]:
    """Create finetune dataloaders from Makin + Flint (excluding held-out).

    Returns:
        (train_loader, val_loader, n_sessions)
    """
    sessions = discover_sessions(datasets=['makin', 'flint'])
    # Exclude held-out
    train_sessions = [(k, v) for k, v in sessions.items()
                      if k not in HELDOUT_SESSIONS]
    print(f"Finetune: {len(train_sessions)} sessions "
          f"(excluded {len(HELDOUT_SESSIONS)} held-out)")

    dataset = SupervisedSegmentDataset(train_sessions, segment_length)
    print(f"Finetune: {len(dataset)} segments total")

    # Split by SESSION (not segment) to prevent data leakage
    n_sess = len(train_sessions)

    if val_ratio <= 0.0:
        # No val split: use all data for training
        print(f"Finetune: {len(dataset)} train segs ({n_sess} sessions), no val split")
        train_loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers,
            shuffle=True, pin_memory=True,
        )
        return train_loader, None, n_sess

    rng = np.random.RandomState(seed)
    sess_order = np.arange(n_sess)
    rng.shuffle(sess_order)
    n_val_sess = max(1, int(n_sess * val_ratio))
    val_session_set = set(sess_order[:n_val_sess].tolist())

    val_indices = [i for i, (sess_idx, _) in enumerate(dataset.segments)
                   if sess_idx in val_session_set]
    train_indices = [i for i, (sess_idx, _) in enumerate(dataset.segments)
                     if sess_idx not in val_session_set]

    print(f"Finetune split: {len(train_indices)} train segs "
          f"({n_sess - n_val_sess} sessions), "
          f"{len(val_indices)} val segs ({n_val_sess} sessions)")

    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
        pin_memory=True,
    )
    return train_loader, val_loader, n_sess


def create_test_dataloaders(
    segment_length: int = SEGMENT_LENGTH,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Create per-session test dataloaders for held-out evaluation.

    Returns:
        {session_name: DataLoader}
    """
    sessions = discover_sessions(datasets=['makin', 'flint'])
    test_loaders = {}

    for name, info in sessions.items():
        if name not in HELDOUT_SESSIONS:
            continue
        data = np.load(info['path'], allow_pickle=True)
        lfp = data['lfp_data'].astype(np.float32)
        targets = data['targets'].astype(np.float32)
        n_ch = lfp.shape[0]

        if n_ch < MAX_CHANNELS:
            pad = np.zeros((MAX_CHANNELS - n_ch, lfp.shape[1], lfp.shape[2]),
                           dtype=np.float32)
            lfp = np.concatenate([lfp, pad], axis=0)

        # Non-overlapping segments covering whole session
        T = lfp.shape[2]
        segments = []
        for start in range(0, T - segment_length + 1, segment_length):
            end = start + segment_length
            lfp_seg = _np_to_tensor(lfp[:, :, start:end].copy())
            tgt_seg = _np_to_tensor(targets[start:end].copy())
            ch_mask = torch.zeros(MAX_CHANNELS)
            ch_mask[:n_ch] = 1.0
            segments.append({
                'lfp': lfp_seg, 'target': tgt_seg, 'channel_mask': ch_mask,
            })

        class _ListDataset(Dataset):
            def __init__(self, items):
                self.items = items
            def __len__(self):
                return len(self.items)
            def __getitem__(self, i):
                return self.items[i]

        test_loaders[name] = DataLoader(
            _ListDataset(segments), batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=True,
        )

    return test_loaders


# ── Spike Reconstruction Dataset ─────────────────────────────────────────

def discover_spike_sessions(
    datasets: Optional[List[str]] = None,
) -> OrderedDict:
    """Discover sessions with BOTH *_rawlfp.npz and *_spike.npz.

    Returns:
        OrderedDict {session_stem: {
            'lfp_path': str, 'spike_path': str, 'dataset': str
        }}
    """
    search_dirs = {
        'dandi_000070': ROOT / "data" / "dandi_000070" / "preprocessed",
        'dandi_000121': ROOT / "data" / "dandi_000121" / "preprocessed",
    }
    if datasets is not None:
        search_dirs = {k: v for k, v in search_dirs.items() if k in datasets}

    sessions = OrderedDict()
    for dataset_name, data_dir in search_dirs.items():
        if not data_dir.exists():
            continue
        spike_files = sorted(data_dir.glob("*_spike.npz"))
        for spike_path in spike_files:
            stem = spike_path.stem.replace("_spike", "")
            lfp_path = data_dir / f"{stem}_rawlfp.npz"
            if lfp_path.exists():
                sessions[stem] = {
                    'lfp_path': str(lfp_path),
                    'spike_path': str(spike_path),
                    'dataset': dataset_name,
                }
    return sessions


class SpikeReconSegmentDataset(Dataset):
    """Paired LFP + spike dataset for cross-modal reconstruction pretraining.

    Each sample:
        {'lfp':          (96, 1, T),     # raw LFP input
         'spike':        (96, T),         # spike counts target
         'spike_mask':   (96,),           # 1.0 for channels with units
         'session_id':   int,
         'channel_mask': (96,)}           # 1.0 for valid LFP channels
    """

    def __init__(
        self,
        session_list: List[Tuple[str, dict]],
        segment_length: int = SEGMENT_LENGTH,
        stride: int = None,
    ):
        self.segment_length = segment_length
        self.stride = stride if stride is not None else segment_length
        self.segments = []
        self.session_data = []
        self.session_names = []

        for sess_idx, (name, info) in enumerate(session_list):
            # Load LFP
            lfp_npz = np.load(info['lfp_path'], allow_pickle=True)
            lfp = lfp_npz['lfp_data'].astype(np.float32)  # (n_ch, 1, T)
            n_ch = lfp.shape[0]

            # Load spike data
            spike_npz = np.load(info['spike_path'], allow_pickle=True)
            spike_data = spike_npz['spike_data'].astype(np.float32)  # (96, T)
            spike_mask = spike_npz['spike_mask'].astype(np.float32)  # (96,)

            # Verify T alignment
            T_lfp = lfp.shape[2]
            T_spike = spike_data.shape[1]
            T = min(T_lfp, T_spike)
            lfp = lfp[:, :, :T]
            spike_data = spike_data[:, :T]

            # Zero-pad LFP to MAX_CHANNELS
            if n_ch < MAX_CHANNELS:
                pad = np.zeros((MAX_CHANNELS - n_ch, lfp.shape[1], T),
                               dtype=np.float32)
                lfp = np.concatenate([lfp, pad], axis=0)

            # Zero-pad spike to MAX_CHANNELS (should already be 96)
            if spike_data.shape[0] < MAX_CHANNELS:
                pad_d = np.zeros((MAX_CHANNELS - spike_data.shape[0], T),
                                 dtype=np.float32)
                spike_data = np.concatenate([spike_data, pad_d], axis=0)
                pad_m = np.zeros(MAX_CHANNELS - spike_mask.shape[0],
                                 dtype=np.float32)
                spike_mask = np.concatenate([spike_mask, pad_m])

            self.session_data.append((lfp, spike_data, spike_mask, n_ch))
            self.session_names.append(name)

            for start in range(0, T - segment_length + 1, self.stride):
                self.segments.append((sess_idx, start))

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        sess_idx, start = self.segments[idx]
        lfp, spike_data, spike_mask, n_ch = self.session_data[sess_idx]
        end = start + self.segment_length

        lfp_seg = lfp[:, :, start:end].copy()
        spike_seg = spike_data[:, start:end].copy()

        channel_mask = np.zeros(MAX_CHANNELS, dtype=np.float32)
        channel_mask[:n_ch] = 1.0

        return {
            'lfp': _np_to_tensor(lfp_seg),
            'spike': _np_to_tensor(spike_seg),
            'spike_mask': _np_to_tensor(spike_mask.copy()),
            'session_id': torch.tensor(sess_idx, dtype=torch.long),
            'channel_mask': _np_to_tensor(channel_mask),
        }


def create_spike_recon_dataloaders(
    seed: int = 42,
    segment_length: int = SEGMENT_LENGTH,
    batch_size: int = BATCH_SIZE,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    stride: int = None,
) -> Tuple[DataLoader, DataLoader, int]:
    """Create dataloaders for spike reconstruction pretraining.

    Returns:
        (train_loader, val_loader, n_sessions)
    """
    sessions = discover_spike_sessions()
    session_list = list(sessions.items())
    print(f"Spike Recon Pretrain: {len(session_list)} sessions discovered")

    if len(session_list) == 0:
        raise RuntimeError(
            "No spike sessions found! Run extract_spikes_dandi.py first."
        )

    dataset = SpikeReconSegmentDataset(session_list, segment_length, stride=stride)
    print(f"Spike Recon Pretrain: {len(dataset)} segments total "
          f"(stride={dataset.stride})")

    # Session-level split
    n_sess = len(session_list)
    rng = np.random.RandomState(seed)
    sess_order = np.arange(n_sess)
    rng.shuffle(sess_order)
    n_val_sess = max(1, int(n_sess * val_ratio))
    val_session_set = set(sess_order[:n_val_sess].tolist())

    val_indices = [i for i, (sess_idx, _) in enumerate(dataset.segments)
                   if sess_idx in val_session_set]
    train_indices = [i for i, (sess_idx, _) in enumerate(dataset.segments)
                     if sess_idx not in val_session_set]

    print(f"Spike Recon split: {len(train_indices)} train segs "
          f"({n_sess - n_val_sess} sessions), "
          f"{len(val_indices)} val segs ({n_val_sess} sessions)")

    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
        pin_memory=True,
    )
    return train_loader, val_loader, n_sess
