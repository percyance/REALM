"""
Continuous masked autoencoding (CMAE) of LFP with the bidirectional REALMEncoder.

  1. Data augmentation (training only): channel dropout, amplitude jitter, noise
  2. encoder.forward_spatial(lfp) -> (B, T, d_model)
  3. Mask 60% of the timesteps in contiguous blocks of 10-50 steps, each replaced by a
     learnable [MASK] token
  4. encoder.forward_temporal(masked) -> (B, T, d_model)
  5. Predictor: one BiMamba-2 layer + MLP head (lightweight, asymmetric)
  6. Loss: MSE at masked positions only (target = original, unaugmented LFP)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from models.encoder import REALMEncoder
from models.layers import BiMamba2Block
from .augmentations import PretrainAugmentation


class MaskedLFPPretrainer(nn.Module):
    """Masked-autoencoding pretraining of a REALMEncoder with a lightweight predictor."""

    def __init__(
        self,
        encoder: REALMEncoder,
        n_channels: int = 96,
        n_bands: int = 1,
        mask_ratio: float = 0.60,
        predictor_layers: int = 1,
        predictor_expand: int = 1,
        augment: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.mask_ratio = mask_ratio
        d_model = encoder.d_model

        # Target dimension per token: the raw LFP vector of one timestep
        self.target_dim = n_channels * n_bands

        # Data augmentation (training only, disabled at eval)
        self.augment = PretrainAugmentation() if augment else None

        # Learnable [MASK] token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        # Predictor: lightweight BiMamba2 (asymmetric -- much lighter than encoder)
        d_state = 64
        headdim = 64
        self.predictor_layers = nn.ModuleList()
        self.predictor_norms = nn.ModuleList()
        for _ in range(predictor_layers):
            self.predictor_layers.append(
                BiMamba2Block(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=4,
                    expand=predictor_expand,
                    headdim=headdim,
                )
            )
            self.predictor_norms.append(nn.LayerNorm(d_model))
        self.predictor_out_norm = nn.LayerNorm(d_model)

        # MLP head after the BiMamba2 predictor
        self.predictor_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, self.target_dim)

    def block_masking(self, x):
        """Block masking: mask contiguous temporal blocks.

        Places random-sized blocks (10-50 steps) until reaching mask_ratio.
        Forces encoder to learn long-range temporal dependencies instead of
        simple interpolation from nearby visible tokens.

        Args:
            x: (B, T, d_model)

        Returns:
            x_masked: (B, T, d_model) with masked positions replaced
            mask: (B, T) bool, True = masked
        """
        B, N, D = x.shape
        n_mask = max(1, int(N * self.mask_ratio))
        mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)

        for b in range(B):
            masked_count = 0
            attempts = 0
            while masked_count < n_mask and attempts < 200:
                block_size = torch.randint(10, 51, (1,)).item()
                block_size = min(block_size, n_mask - masked_count)
                start = torch.randint(0, max(1, N - block_size + 1), (1,)).item()
                mask[b, start:start + block_size] = True
                masked_count = mask[b].sum().item()
                attempts += 1

        mask_tokens = self.mask_token.expand(B, N, -1)
        x_masked = torch.where(mask.unsqueeze(-1), mask_tokens, x)
        return x_masked, mask

    def forward(
        self,
        lfp_data: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            lfp_data: (B, n_channels, n_bands, T)
            channel_mask: (B, n_channels) float

        Returns:
            dict with 'loss' (tensor), 'recon_loss', 'mask_ratio_actual' and 'mask'
        """
        B, C, nb, T = lfp_data.shape

        # 0. Save original for reconstruction target, then augment
        lfp_target = lfp_data
        if self.augment is not None:
            lfp_data = self.augment(lfp_data)

        # 1. Spatial encoding (neural tokenizer)
        spatial = self.encoder.forward_spatial(lfp_data, channel_mask)

        # 2. Mask timesteps
        spatial_masked, mask = self.block_masking(spatial)

        # 3. Temporal encoding through the BiMamba2 encoder
        encoded = self.encoder.forward_temporal(spatial_masked)

        # 4. Predictor: BiMamba2 + MLP head
        N = encoded.size(1)
        position_ids = torch.arange(N, device=encoded.device)

        pred_x = encoded
        for norm, layer in zip(self.predictor_norms, self.predictor_layers):
            pred_x = pred_x + layer(norm(pred_x), position_ids=position_ids)
        pred_x = self.predictor_out_norm(pred_x)
        pred_x = pred_x + self.predictor_mlp(pred_x)  # MLP head with residual

        # 5. Output projection
        pred = self.output_proj(pred_x)  # (B, T, target_dim)

        # 6. Reconstruction target: the ORIGINAL unaugmented LFP
        target = lfp_target.permute(0, 3, 1, 2).reshape(B, T, C * nb)

        # 7. MSE loss at masked positions only
        loss = F.mse_loss(pred[mask], target[mask])

        return {
            'loss': loss,
            'recon_loss': loss.item(),
            'mask_ratio_actual': mask.float().mean().item(),
            'mask': mask,         # (B, T) bool
        }
