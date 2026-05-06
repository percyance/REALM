"""
MAE-style masked pretraining using REALMEncoder (REALM BiMamba2).

Architecture:
  1. (optional) Data augmentation: channel dropout, amplitude jitter, noise
  2. encoder.forward_spatial(lfp) -> (B, T, d_model) or (B, T, S, d_model)
  3. + session_embed
  4. Mask 60% of spatiotemporal tokens with learnable [MASK] token
     - 'block' mode: contiguous temporal blocks (forces long-range learning)
     - 'random' mode: uniform random masking (original)
     - 'amplitude' mode: preferentially mask high-amplitude regions (NeurIPT-inspired)
  5. encoder.forward_temporal(masked) -> (B, N, d_model) where N=T or T*S
  6. Predictor: 1-layer BiMamba2 + MLP head (lightweight, asymmetric)
  7. Loss: MSE at masked positions only (target = original unaugmented LFP)

Changes from v2:
  - Block masking option (default) instead of purely random
  - Lighter predictor: 1 BiMamba2 + MLP instead of 2 BiMamba2
  - Data augmentation with clean reconstruction target
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from models.encoder import REALMEncoder
from models.layers import BiMamba2Block
from .augmentations import PretrainAugmentation


class MaskedLFPPretrainerV2(nn.Module):
    """MAE pretraining with REALM BiMamba2 encoder + lightweight predictor."""

    def __init__(
        self,
        encoder: REALMEncoder,
        n_channels: int = 96,
        n_bands: int = 1,
        mask_ratio: float = 0.60,
        mask_type: str = 'block',
        predictor_layers: int = 1,
        predictor_expand: int = 1,
        augment: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.n_spatial_patches = encoder.n_spatial_patches
        d_model = encoder.d_model

        # Target dimension per token
        if self.n_spatial_patches > 1:
            self.target_dim = (n_channels // self.n_spatial_patches) * n_bands
        else:
            self.target_dim = n_channels * n_bands

        # Data augmentation (training only, disabled at eval)
        self.augment = PretrainAugmentation() if augment else None

        # Learnable [MASK] token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        # Predictor: lightweight BiMamba2 (asymmetric — much lighter than encoder)
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

        # MLP head after BiMamba2 predictor (replaces removed layers)
        self.predictor_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, self.target_dim)

    def random_masking(self, x):
        """Vectorized random masking of tokens.

        Args:
            x: (B, N, d_model) where N = T or T*S

        Returns:
            x_masked: (B, N, d_model) with masked positions replaced
            mask: (B, N) bool, True = masked
        """
        B, N, D = x.shape
        n_mask = max(1, int(N * self.mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_sorted = torch.argsort(noise, dim=1, descending=True)
        mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)
        mask.scatter_(1, ids_sorted[:, :n_mask], True)

        mask_tokens = self.mask_token.expand(B, N, -1)
        x_masked = torch.where(mask.unsqueeze(-1), mask_tokens, x)
        return x_masked, mask

    def block_masking(self, x):
        """Block masking: mask contiguous temporal blocks.

        Places random-sized blocks (10-50 steps) until reaching mask_ratio.
        Forces encoder to learn long-range temporal dependencies instead of
        simple interpolation from nearby visible tokens.

        Args:
            x: (B, N, d_model) where N = T or T*S

        Returns:
            x_masked: (B, N, d_model) with masked positions replaced
            mask: (B, N) bool, True = masked
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

    def amplitude_aware_masking(self, x, lfp_amplitude):
        """Mask tokens with probability proportional to signal amplitude.

        High-amplitude regions (movement events) are more likely to be masked,
        forcing the encoder to predict important dynamics from context.
        Inspired by NeurIPT (NeurIPS 2025).

        Args:
            x: (B, N, d_model) spatial-encoded tokens
            lfp_amplitude: (B, N) per-token RMS amplitude

        Returns:
            x_masked: (B, N, d_model) with masked positions replaced
            mask: (B, N) bool, True = masked
        """
        B, N, D = x.shape
        n_mask = max(1, int(N * self.mask_ratio))

        # Softmax-temperature sampling: higher amplitude -> higher mask prob
        temperature = 0.5  # lower = more concentrated on high-amplitude
        probs = F.softmax(lfp_amplitude / temperature, dim=1)

        # Sample n_mask tokens without replacement
        indices = torch.multinomial(probs, n_mask, replacement=False)
        mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)
        mask.scatter_(1, indices, True)

        mask_tokens = self.mask_token.expand(B, N, -1)
        x_masked = torch.where(mask.unsqueeze(-1), mask_tokens, x)
        return x_masked, mask

    def forward(
        self,
        lfp_data: torch.Tensor,
        session_ids: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            lfp_data: (B, 96, n_bands, T)
            session_ids: (B,) int
            channel_mask: (B, 96) float

        Returns:
            dict with 'loss' and metrics
        """
        B, C, nb, T = lfp_data.shape
        S = self.n_spatial_patches

        # 0. Save original for reconstruction target, then augment
        lfp_target = lfp_data
        if self.augment is not None:
            lfp_data = self.augment(lfp_data)

        # 1. Spatial encoding
        spatial = self.encoder.forward_spatial(lfp_data, channel_mask)

        # 2. Add session embedding
        if session_ids is not None:
            sess_emb = self.encoder.session_embed(session_ids)
            if spatial.dim() == 4:
                spatial = spatial + sess_emb.unsqueeze(1).unsqueeze(2)
            else:
                spatial = spatial + sess_emb.unsqueeze(1)

        # 3. Flatten to token sequence for masking
        if spatial.dim() == 4:
            # (B, T, S, d_model) -> (B, T*S, d_model)
            spatial_flat = spatial.reshape(B, T * S, -1)
        else:
            spatial_flat = spatial

        # 4. Mask spatiotemporal tokens
        if self.mask_type == 'block':
            spatial_masked, mask = self.block_masking(spatial_flat)
        elif self.mask_type == 'amplitude':
            # Per-timestep RMS amplitude from original (unaugmented) LFP
            lfp_amplitude = lfp_target.pow(2).mean(dim=(1, 2)).sqrt()  # (B, T)
            if S > 1:
                # Repeat amplitude for each spatial patch
                lfp_amplitude = lfp_amplitude.unsqueeze(-1).expand(B, T, S).reshape(B, T * S)
            spatial_masked, mask = self.amplitude_aware_masking(spatial_flat, lfp_amplitude)
        else:
            spatial_masked, mask = self.random_masking(spatial_flat)

        # 5. Temporal encoding through BiMamba2 encoder
        if S > 1:
            # Reshape back to 4D for forward_temporal to build position_ids
            spatial_masked_4d = spatial_masked.reshape(B, T, S, -1)
            encoded = self.encoder.forward_temporal(spatial_masked_4d)
        else:
            encoded = self.encoder.forward_temporal(spatial_masked)

        # 6. Predictor: BiMamba2 + MLP head
        N = encoded.size(1)
        if S > 1:
            time_pos = torch.arange(T, device=encoded.device)
            position_ids = time_pos.unsqueeze(1).expand(T, S).reshape(T * S)
        else:
            position_ids = torch.arange(N, device=encoded.device)

        pred_x = encoded
        for norm, layer in zip(self.predictor_norms, self.predictor_layers):
            pred_x = pred_x + layer(norm(pred_x), position_ids=position_ids)
        pred_x = self.predictor_out_norm(pred_x)
        pred_x = pred_x + self.predictor_mlp(pred_x)  # MLP head with residual

        # 7. Output projection
        pred = self.output_proj(pred_x)  # (B, N, target_dim)

        # 8. Build reconstruction target (use ORIGINAL unaugmented LFP)
        if S > 1:
            ps = self.encoder.spatial_patch_size
            lfp_perm = lfp_target.permute(0, 3, 1, 2)  # (B, T, C, nb)
            target = lfp_perm.reshape(B, T, S, ps * nb).reshape(B, T * S, ps * nb)
        else:
            target = lfp_target.permute(0, 3, 1, 2).reshape(B, T, C * nb)

        # 9. MSE loss at masked positions only
        loss = F.mse_loss(pred[mask], target[mask])

        return {
            'loss': loss,
            'recon_loss': loss.item(),
            'mask_ratio_actual': mask.float().mean().item(),
        }
