"""
Makin and Flint LFP loader.

Every session is one preprocessed file <session>_rawlfp.npz holding
  lfp_data: (n_channels, 1, T) float32   broadband LFP at 100 Hz, z-scored per channel
  targets:  (T, 2)             float32   velocity (vx, vy), z-scored per axis

under the data root, which defaults to ./data in the repository and can be overridden by
the environment variable REALM_DATA:

  <data root>/makin/preprocessed/lfp/indy_*_rawlfp.npz          30 sessions, Monkey I
  <data root>/flint/preprocessed/lfp/Flint_*_rawlfp.npz         12 sessions, Monkey C
  <data root>/flint/preprocessed/lfp_tukey/Flint_*_rawlfp.npz   the 3 held-out Flint
      sessions with outlier windows removed, read instead of lfp/ when tukey_flint=True

Files placed directly in <corpus>/preprocessed/ are found as well. Sessions are cut into
non-overlapping 5 s segments (500 samples); channels are zero-padded to 96 and marked in
a channel mask.
"""

import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _np_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert numpy array to torch tensor (bypasses broken numpy-torch bridge)."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    t = torch.frombuffer(bytearray(arr.data), dtype=torch.float32)
    return t.reshape(arr.shape).clone()


# ── Data root ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("REALM_DATA", REPO_ROOT / "data"))

# ── Constants ─────────────────────────────────────────────────────────────
MAX_CHANNELS = 96
SEGMENT_LENGTH = 500  # 5 s @ 100 Hz
BATCH_SIZE = 64

# Held-out sessions: never seen in pretraining or distillation, and each split 72/8/20
# for evaluation. They are the held-out sessions of CrossModalDistill (Erturk et al.,
# NeurIPS 2025).
HELDOUT_MAKIN = [
    "indy_20160622_01", "indy_20160630_01", "indy_20160915_01",
    "indy_20170124_01", "indy_20161013_03",
]
HELDOUT_FLINT = ["Flint_e1_1", "Flint_e4_1", "Flint_e5_2"]
HELDOUT_SESSIONS = set(HELDOUT_MAKIN + HELDOUT_FLINT)


# ── Session discovery ─────────────────────────────────────────────────────

def discover_sessions(datasets: Optional[List[str]] = None,
                      tukey_flint: bool = False) -> OrderedDict:
    """Discover all preprocessed sessions of the requested corpora ('makin', 'flint').

    Returns:
        OrderedDict {session_stem: {'path': str, 'dataset': str}}
    """
    search_dirs = {
        'makin': DATA_ROOT / "makin" / "preprocessed",
        'flint': DATA_ROOT / "flint" / "preprocessed",
    }
    if datasets is not None:
        search_dirs = {k: v for k, v in search_dirs.items() if k in datasets}

    pat, suffix = "*_rawlfp.npz", "_rawlfp"
    sessions = OrderedDict()
    for dataset_name, data_dir in search_dirs.items():
        if not data_dir.exists():
            continue
        # Look in the lfp/ subdirectory first, then directly in preprocessed/.
        npz_files = sorted((data_dir / "lfp").glob(pat)) or sorted(data_dir.glob(pat))
        if tukey_flint:
            # Per file: only the held-out Flint sessions have a filtered copy.
            tdir = data_dir / "lfp_tukey"
            npz_files = [tdir / f.name if (tdir / f.name).exists() else f
                         for f in npz_files]
        for f in npz_files:
            stem = f.stem.replace(suffix, "")
            sessions[stem] = {'path': str(f), 'dataset': dataset_name}
    return sessions


# ── Canonical 72/8/20 split ───────────────────────────────────────────────

def load_canonical_split(splits_file, session: str, seed: int, n_segments: int,
                         tukey_flint: bool = False) -> Tuple[List[int], List[int], List[int]]:
    """Train / validation / test segment indices of a held-out session.

    The split file (splits/canonical_splits_728020.json) holds one 72/8/20 partition per
    session and seed; Flint sessions read with tukey_flint=True have their own entries,
    since the filtered files have fewer segments.

    Returns:
        (train_idx, val_idx, test_idx), lists of segment indices
    """
    sub = "lfp_tukey" if (tukey_flint and session.startswith("Flint")) else "lfp"
    with open(splits_file) as f:
        rec = json.load(f)[f"{sub}|{session}|{seed}"]
    if rec["n"] != n_segments:
        raise ValueError(f"{session}: the split was built for {rec['n']} segments, "
                         f"the data has {n_segments}")
    return rec["train"], rec["val"], rec["test"]


# ── R² metric ────────────────────────────────────────────────────────────

def compute_r2(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute R² (coefficient of determination) -- combined across all dims."""
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - targets.mean(dim=0, keepdim=True)) ** 2)
    return (1 - ss_res / (ss_tot + 1e-8)).item()


def compute_r2_per_axis(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute R² averaged across behavior dimensions (per-axis), the metric of the paper."""
    r2s = []
    for i in range(targets.shape[-1]):
        ss_res = torch.sum((targets[..., i] - preds[..., i]) ** 2)
        ss_tot = torch.sum((targets[..., i] - targets[..., i].mean()) ** 2)
        r2s.append((1 - ss_res / (ss_tot + 1e-8)).item())
    return float(np.mean(r2s))


# ── Pretraining dataset ───────────────────────────────────────────────────

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
            if n_ch > MAX_CHANNELS:
                raise ValueError(f"{name}: {n_ch} channels exceeds MAX_CHANNELS={MAX_CHANNELS}")

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


# ── Dataloader factories ─────────────────────────────────────────────────

def create_pretrain_dataloaders(
    seed: int = 42,
    segment_length: int = SEGMENT_LENGTH,
    batch_size: int = BATCH_SIZE,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    stride: int = None,
    datasets: Optional[List[str]] = None,
    ddp: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader, int]:
    """Create pretraining dataloaders over the non-held-out sessions.

    Held-out sessions are excluded, so the teacher never sees an evaluation session.
    The validation set is a random 10% of the sessions (split by session, not by segment).

    Args:
        stride: Segment stride. None = non-overlapping (stride=segment_length).
                Use stride=250 for 50% overlap (doubles data).
        ddp: shard the training set across ranks with a DistributedSampler; the caller
             must call train_loader.ddp_sampler.set_epoch(epoch) every epoch.

    Returns:
        (train_loader, val_loader, n_sessions)
    """
    if datasets is None:
        datasets = ['makin', 'flint']
    all_sessions = discover_sessions(datasets=datasets)
    session_list = [(name, info) for name, info in all_sessions.items()
                    if name not in HELDOUT_SESSIONS]
    print(f"Pretrain: {len(session_list)} sessions "
          f"(datasets={datasets}, heldout excluded)")

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

    if ddp:
        # DistributedSampler shards a dataset, not an index list, so the training subset is
        # materialised first. `set_epoch` must be called each epoch by the caller or every
        # rank reshuffles identically and the shards never change.
        from torch.utils.data.distributed import DistributedSampler
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        train_sampler = DistributedSampler(
            train_subset, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=True)
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, num_workers=num_workers,
            sampler=train_sampler, pin_memory=True,
        )
        train_loader.ddp_sampler = train_sampler
    else:
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


class _ListDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def create_test_dataloaders(
    segment_length: int = SEGMENT_LENGTH,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    tukey_flint: bool = False,
    datasets: Optional[List[str]] = None,
) -> Dict[str, DataLoader]:
    """Per-session dataloaders over every segment of each held-out session, in time order.

    Each item: {'lfp': (96, 1, T), 'target': (T, 2), 'channel_mask': (96,)}

    Returns:
        {session_name: DataLoader}
    """
    if datasets is None:
        datasets = ['makin', 'flint']
    sessions = discover_sessions(datasets=datasets, tukey_flint=tukey_flint)
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

        test_loaders[name] = DataLoader(
            _ListDataset(segments), batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=True,
        )

    return test_loaders
