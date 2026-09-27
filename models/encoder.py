"""
REALMEncoder: the neural tokenizer followed by a stack of Mamba-2 layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DropPath, Mamba2Block, BiMamba2Block


class REALMEncoder(nn.Module):
    """Neural tokenizer + temporal Mamba-2 encoder.

      Neural tokenizer
        Stage 1: per-channel temporal Conv1d (weights shared across channels)
        Stage 2: ECA channel attention
        Stage 3: spatial projection (n_channels * d_channel -> d_model) + LayerNorm
      Encoder
        Stage 4: temporal Mamba-2 layers, bidirectional (BiMamba-2) or causal

    With bidirectional=False every stage is past-only: the temporal convolution is
    left-padded, ECA uses a cumulative mean over time, and each layer is a causal scan.
    Padded channels (fewer than n_channels electrodes) are zeroed by channel_mask.
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
        max_sessions: int = 200,
        drop_path_rate: float = 0.0,
        n_spatial_patches: int = 1,
        block_type: str = 'mamba',
    ):
        super().__init__()
        # The last two arguments appear in the configurations stored with the released
        # checkpoints; only these values are supported.
        if n_spatial_patches != 1 or block_type != 'mamba':
            raise ValueError("REALMEncoder supports n_spatial_patches=1 and "
                             "block_type='mamba' only")
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.d_model = d_model
        self.d_channel = d_channel
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.expand = expand
        self.d_inner = d_model * expand

        # Stage 1: Per-channel temporal embedding
        # Causal: no built-in padding, the input is left-padded in forward_spatial.
        # Bidirectional: symmetric padding (full context allowed).
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
        self.spatial_proj = nn.Linear(n_channels * d_channel, d_model)
        self.spatial_norm = nn.LayerNorm(d_model)

        # Unused session table, kept so the released checkpoints load with strict=True and
        # the parameter initialisation order matches the runs reported in the paper.
        self.session_embed = nn.Embedding(max_sessions, d_model)

        # Stage 4: Temporal Mamba-2 layers
        BlockClass = BiMamba2Block if bidirectional else Mamba2Block
        block_kwargs = dict(d_model=d_model, d_state=d_state,
                            d_conv=d_conv, expand=expand, headdim=headdim)

        # Linearly increasing drop path rates (0 -> drop_path_rate)
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
        """Neural tokenizer (stages 1-3): per-channel conv -> ECA -> spatial projection.

        Args:
            lfp_data: (B, n_channels, n_bands, T)
            channel_mask: (B, n_channels) float -- 1.0 for valid, 0.0 for padded

        Returns:
            (B, T, d_model)
        """
        if channel_mask is not None:
            lfp_data = lfp_data * channel_mask.unsqueeze(2).unsqueeze(3)

        B, C, nb, T = lfp_data.shape

        # Stage 1: Per-channel temporal embedding
        x = lfp_data.reshape(B * C, nb, T)
        if not self.bidirectional:
            # Causal conv (padding=0): left-pad so output[t] uses input[<=t]
            x = F.pad(x, (self.temporal_conv_kernel - 1, 0))
            x = self.temporal_conv(x)          # (B*C, d_ch, T)
        else:
            x = self.temporal_conv(x)          # non-causal (symmetric padding=1)
        x = x.reshape(B, C, self.d_channel, T)

        # Stage 2: ECA channel attention
        if self.bidirectional:
            # Non-causal: global mean over d_channel and time
            energy = x.mean(dim=(2, 3))             # (B, C)
            attn = self.eca_conv(energy.unsqueeze(1))  # (B, 1, C)
            attn = torch.sigmoid(attn)
            x = x * attn.squeeze(1).unsqueeze(-1).unsqueeze(-1)
        else:
            # Causal: cumulative mean over time, per-timestep attention
            energy_t = x.mean(dim=2)                # (B, C, T)
            cum_energy = energy_t.cumsum(dim=2)      # (B, C, T)
            counts = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)
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
        x = x.reshape(B, T, C * self.d_channel)
        return self.spatial_norm(self.spatial_proj(x))  # (B, T, d_model)

    def forward_temporal(self, x, return_intermediates=False, position_ids=None):
        """Stage 4: temporal Mamba-2 layers with RoPE.

        Args:
            x: (B, T, d_model)
            return_intermediates: if True, also return the output of every layer
                (before the final LayerNorm)
            position_ids: optional temporal position ids for RoPE

        Returns:
            (B, T, d_model) or ((B, T, d_model), list_of_intermediates)
        """
        if position_ids is None:
            position_ids = torch.arange(x.size(1), device=x.device)

        x = self.dropout(x)

        intermediates = []
        for norm, layer, dp, layer_do in zip(
                self.layer_norms, self.layers, self.drop_paths, self.layer_dropouts):
            x = x + dp(layer_do(layer(norm(x), position_ids=position_ids)))
            if return_intermediates:
                intermediates.append(x)

        out = self.out_norm(x)
        if return_intermediates:
            return out, intermediates
        return out

    def forward(self, lfp_data, channel_mask=None, return_intermediates=False):
        """Full forward: neural tokenizer -> temporal layers.

        Args:
            lfp_data: (B, n_channels, n_bands, T)
            channel_mask: (B, n_channels) float tensor
            return_intermediates: if True, also return per-layer outputs (distillation)

        Returns:
            (B, T, d_model) or ((B, T, d_model), intermediates)
        """
        spatial = self.forward_spatial(lfp_data, channel_mask)
        return self.forward_temporal(spatial, return_intermediates)
