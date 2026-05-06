"""
REALMDecoder: encoder + velocity prediction head + skip connection.
"""

import torch
import torch.nn as nn

from .encoder import REALMEncoder


class REALMDecoder(nn.Module):
    """Full decoder: REALMEncoder + velocity head + skip connection."""

    def __init__(self, encoder_kwargs: dict, output_dim: int = 2,
                 head_layers: int = 1, head_hidden: int = None,
                 head_dropout: float = 0.0):
        super().__init__()
        self.encoder = REALMEncoder(**encoder_kwargs)
        d_model = encoder_kwargs['d_model']
        n_channels = encoder_kwargs.get('n_channels', 96)
        n_bands = encoder_kwargs.get('n_bands', 1)

        if head_layers <= 1:
            self.out_proj = nn.Linear(d_model, output_dim)
        else:
            hidden = head_hidden if head_hidden is not None else d_model
            modules = [nn.Linear(d_model, hidden), nn.GELU()]
            if head_dropout > 0:
                modules.append(nn.Dropout(head_dropout))
            for _ in range(head_layers - 2):
                modules.append(nn.Linear(hidden, hidden))
                modules.append(nn.GELU())
                if head_dropout > 0:
                    modules.append(nn.Dropout(head_dropout))
            modules.append(nn.Linear(hidden, output_dim))
            self.out_proj = nn.Sequential(*modules)
        self.skip = nn.Linear(n_channels * n_bands, output_dim)

    def forward(self, lfp_data, session_ids=None, channel_mask=None,
                return_intermediates=False):
        """
        Args:
            lfp_data: (B, 96, n_bands, T)
            session_ids: (B,)
            channel_mask: (B, 96)
            return_intermediates: if True, return per-layer encoder outputs

        Returns:
            dict with 'prediction' (B, T, output_dim) and optionally 'intermediates'
        """
        if return_intermediates:
            encoded, intermediates = self.encoder(
                lfp_data, session_ids=session_ids, channel_mask=channel_mask,
                return_intermediates=True)
        else:
            encoded = self.encoder(
                lfp_data, session_ids=session_ids, channel_mask=channel_mask)
            intermediates = None

        # Pool over spatial patches if needed: (B, T*S, d) -> (B, T, d)
        S = self.encoder.n_spatial_patches
        B, C, nb, T = lfp_data.shape
        if S > 1:
            encoded = encoded.reshape(B, T, S, -1).mean(dim=2)

        mamba_pred = self.out_proj(encoded)

        lfp_flat = lfp_data.permute(0, 3, 1, 2).reshape(B, T, C * nb)
        skip_pred = self.skip(lfp_flat)

        output = {'prediction': skip_pred + mamba_pred}
        if return_intermediates:
            output['intermediates'] = intermediates
        return output

    def enable_state_capture(self, enable=True):
        self.encoder.enable_state_capture(enable)

    def get_captured_states(self):
        return self.encoder.get_captured_states()
