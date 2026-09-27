"""Data augmentation for masked LFP pretraining."""

import torch
import torch.nn as nn


class PretrainAugmentation(nn.Module):
    """Data augmentation for masked-autoencoding pretraining.

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
