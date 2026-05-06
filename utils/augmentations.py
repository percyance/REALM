"""Data augmentation and regularization modules for LFP pretraining."""

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Stochastic depth for residual branches.

    During training, randomly drops entire residual branches with
    probability `drop_prob`. Identity at eval time.

    Ref: Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device) < keep
        return x * mask / keep

    def extra_repr(self):
        return f"drop_prob={self.drop_prob:.3f}"


class PretrainAugmentation(nn.Module):
    """Data augmentation for MAE pretraining.

    Applied to raw LFP input (B, C, n_bands, T) during training only.
    Augments input while reconstruction target uses original (unaugmented) LFP.

    Augmentations:
      - Channel dropout: zero-out random channels (simulates electrode failure)
      - Amplitude jitter: per-channel random scaling
      - Gaussian noise: additive noise
    """

    def __init__(
        self,
        channel_dropout_rate: float = 0.15,
        amplitude_range: tuple = (0.85, 1.15),
        noise_std: float = 0.05,
    ):
        super().__init__()
        self.channel_dropout_rate = channel_dropout_rate
        self.amp_lo, self.amp_hi = amplitude_range
        self.noise_std = noise_std

    def forward(self, lfp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lfp: (B, C, n_bands, T)
        Returns:
            augmented lfp, same shape
        """
        if not self.training:
            return lfp

        B, C, nb, T = lfp.shape

        # 1. Channel dropout
        ch_keep = (torch.rand(B, C, 1, 1, device=lfp.device)
                   > self.channel_dropout_rate).float()
        lfp = lfp * ch_keep

        # 2. Amplitude jitter (per-channel)
        scale = (torch.rand(B, C, 1, 1, device=lfp.device)
                 * (self.amp_hi - self.amp_lo) + self.amp_lo)
        lfp = lfp * scale

        # 3. Gaussian noise
        lfp = lfp + torch.randn_like(lfp) * self.noise_std

        return lfp

    def extra_repr(self):
        return (f"ch_dropout={self.channel_dropout_rate}, "
                f"amp=({self.amp_lo}, {self.amp_hi}), noise_std={self.noise_std}")
