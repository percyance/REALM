"""
REALMEncoder: Spatial-Temporal encoder for foundation model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .layers import DropPath, Mamba2Block, BiMamba2Block


class REALMEncoder(nn.Module):
    """Spatial-Temporal encoder for foundation model.

    Architecture (from SpatialTemporalEncoder pattern):
      Stage 1: Per-channel temporal Conv1d (shared weights)
      Stage 2: ECA channel attention
      Stage 3: Spatial projection (96*d_channel -> d_model)
      + Session embedding (before temporal)
      + Channel masking (zero-pad invalid channels)
      Stage 4: Temporal Mamba-2 layers (Bi or Causal)
    """

    def __init__(
        self,
        n_channels: int = 96,
        n_bands: int = 1,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        headdim: int = 64,
        d_channel: int = 8,
        eca_kernel: int = 5,
        disable_eca: bool = False,
        max_sessions: int = 200,
        n_spatial_patches: int = 1,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.d_model = d_model
        self.d_channel = d_channel
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.expand = expand
        self.d_inner = d_model * expand
        self.n_spatial_patches = n_spatial_patches
        self.disable_eca = disable_eca

        # Stage 1: Per-channel temporal embedding
        # Causal student: left-pad k-1 (no future leak).
        # Bidir student: symmetric padding (full context allowed).
        self.temporal_conv_kernel = 3
        if bidirectional:
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(n_bands, d_channel, kernel_size=3, padding=1),
                nn.GELU(),
            )
        else:
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(n_bands, d_channel, kernel_size=3, padding=0),
                nn.GELU(),
            )

        # Stage 2: ECA channel attention
        self.eca_conv = nn.Conv1d(
            1, 1, kernel_size=eca_kernel, padding=eca_kernel // 2, bias=False)

        # Stage 3: Spatial projection
        if n_spatial_patches > 1:
            assert n_channels % n_spatial_patches == 0
            self.spatial_patch_size = n_channels // n_spatial_patches
            self.spatial_patch_proj = nn.Linear(
                self.spatial_patch_size * d_channel, d_model)
            self.spatial_pos_embed = nn.Embedding(n_spatial_patches, d_model)
        else:
            self.spatial_proj = nn.Linear(n_channels * d_channel, d_model)
        self.spatial_norm = nn.LayerNorm(d_model)

        # Session embedding
        self.session_embed = nn.Embedding(max_sessions, d_model)

        # Stage 4: Temporal Mamba-2 layers (RoPE is inside each SSD block)
        BlockClass = BiMamba2Block if bidirectional else Mamba2Block
        block_kwargs = dict(d_model=d_model, d_state=d_state,
                            d_conv=d_conv, expand=expand, headdim=headdim)

        # Linearly increasing drop path rates (0 → drop_path_rate)
        dp_rates = [drop_path_rate * i / max(n_layers - 1, 1)
                    for i in range(n_layers)]

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.drop_paths = nn.ModuleList()
        self.layer_dropouts = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(BlockClass(**block_kwargs))
            self.layer_norms.append(nn.LayerNorm(d_model))
            self.drop_paths.append(DropPath(dp_rates[i]))
            self.layer_dropouts.append(nn.Dropout(dropout))

        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(d_model)

    def forward_spatial(self, lfp_data, channel_mask=None):
        """Stages 1-3: per-channel conv -> ECA -> spatial projection.

        Args:
            lfp_data: (B, 96, n_bands, T)
            channel_mask: (B, 96) float -- 1.0 for valid, 0.0 for padded

        Returns:
            (B, T, d_model) if n_spatial_patches == 1
            (B, T, S, d_model) if n_spatial_patches > 1
        """
        if channel_mask is not None:
            lfp_data = lfp_data * channel_mask.unsqueeze(2).unsqueeze(3)

        B, C, nb, T = lfp_data.shape

        # Stage 1: Per-channel temporal embedding
        x = lfp_data.reshape(B * C, nb, T)
        if not self.bidirectional:
            # Left-pad to make output[t] depend only on input[<=t]
            x = F.pad(x, (self.temporal_conv_kernel - 1, 0))
        x = self.temporal_conv(x)              # (B*C, d_ch, T)
        x = x.reshape(B, C, self.d_channel, T)

        # Stage 2: ECA channel attention
        if not self.disable_eca:
            if self.bidirectional:
                # Non-causal: global mean over d_channel and time
                energy = x.mean(dim=(2, 3))             # (B, C)
                attn = self.eca_conv(energy.unsqueeze(1))  # (B, 1, C)
                attn = torch.sigmoid(attn)
                x = x * attn.squeeze(1).unsqueeze(-1).unsqueeze(-1)
            else:
                # Causal: cumulative mean over time, per-timestep attention
                # x: (B, C, d_ch, T)
                energy_t = x.mean(dim=2)                # (B, C, T)
                cum_energy = energy_t.cumsum(dim=2)      # (B, C, T)
                counts = torch.arange(1, T + 1, device=x.device).float()
                cum_mean = cum_energy / counts            # (B, C, T)
                # Apply ECA conv along channel dim for each timestep
                cum_mean_p = cum_mean.permute(0, 2, 1)   # (B, T, C)
                cum_mean_p = cum_mean_p.reshape(B * T, 1, C)
                attn = self.eca_conv(cum_mean_p)          # (B*T, 1, C)
                attn = torch.sigmoid(attn)
                attn = attn.reshape(B, T, C).permute(0, 2, 1)  # (B, C, T)
                x = x * attn.unsqueeze(2)                 # (B, C, d_ch, T)

        # Stage 3: Spatial projection
        x = x.permute(0, 3, 1, 2)              # (B, T, C, d_ch)

        if self.n_spatial_patches > 1:
            S = self.n_spatial_patches
            ps = self.spatial_patch_size
            # Group channels into spatial patches: (B, T, S, ps*d_ch)
            x = x.reshape(B, T, S, ps * self.d_channel)
            x = self.spatial_patch_proj(x)       # (B, T, S, d_model)
            x = self.spatial_norm(x)
            # Add learnable spatial position embedding
            patch_ids = torch.arange(S, device=x.device)
            x = x + self.spatial_pos_embed(patch_ids)
            return x  # (B, T, S, d_model)
        else:
            x = x.reshape(B, T, C * self.d_channel)
            return self.spatial_norm(self.spatial_proj(x))  # (B, T, d_model)

    def forward_temporal(self, x, return_intermediates=False, position_ids=None):
        """Stage 4: temporal Mamba-2 layers with RoPE.

        Args:
            x: (B, T, d_model) or (B, T, S, d_model) if spatial patching
            return_intermediates: if True, return per-layer outputs
            position_ids: optional temporal position ids for RoPE

        Returns:
            (B, N, d_model) or ((B, N, d_model), list_of_intermediates)
            where N = T (no patching) or T*S (with spatial patching)
        """
        if x.dim() == 4:
            # Spatiotemporal token mode: (B, T, S, d_model)
            B, T, S, D = x.shape
            x = x.reshape(B, T * S, D)
            if position_ids is None:
                # Same temporal position for all S patches at same timestep
                time_pos = torch.arange(T, device=x.device)
                position_ids = time_pos.unsqueeze(1).expand(T, S).reshape(T * S)
        else:
            if position_ids is None:
                position_ids = torch.arange(x.size(1), device=x.device)

        x = self.dropout(x)

        intermediates = []
        for norm, layer, dp, layer_do in zip(
                self.layer_norms, self.layers,
                self.drop_paths, self.layer_dropouts):
            x = x + dp(layer_do(layer(norm(x), position_ids=position_ids)))
            if return_intermediates:
                intermediates.append(x)

        out = self.out_norm(x)
        if return_intermediates:
            return out, intermediates
        return out

    def forward(self, lfp_data, session_ids=None, channel_mask=None,
                return_intermediates=False):
        """Full forward: spatial -> (+ session embed) -> temporal.

        Args:
            lfp_data: (B, 96, n_bands, T)
            session_ids: (B,) int tensor
            channel_mask: (B, 96) float tensor
            return_intermediates: for distillation

        Returns:
            (B, N, d_model) or ((B, N, d_model), intermediates)
            where N = T (no patching) or T*S (with spatial patching)
        """
        spatial = self.forward_spatial(lfp_data, channel_mask)

        if session_ids is not None:
            sess_emb = self.session_embed(session_ids)
            if spatial.dim() == 4:
                # (B, T, S, d_model): broadcast session embed
                spatial = spatial + sess_emb.unsqueeze(1).unsqueeze(2)
            else:
                spatial = spatial + sess_emb.unsqueeze(1)

        return self.forward_temporal(spatial, return_intermediates)

    def enable_state_capture(self, enable=True):
        """Enable/disable SSM hidden state capture for all layers."""
        for layer in self.layers:
            layer.enable_state_capture(enable)

    def get_captured_states(self) -> List:
        """Return captured SSM states from all layers.

        For causal: list of (B, T, d_inner) tensors.
        For bidirectional: list of (fwd_states, bwd_states) tuples.
        """
        return [layer.get_captured_states() for layer in self.layers]
