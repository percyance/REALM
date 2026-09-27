"""
REALMDecoder: encoder + linear velocity head + linear skip from the raw LFP.
"""

import torch.nn as nn

from .encoder import REALMEncoder


class REALMDecoder(nn.Module):
    """REALMEncoder + pointwise velocity head + skip connection.

    The skip maps the current raw LFP sample linearly to velocity and is added to the
    head's output. Distillation is label-free (lambda_task = 0), so neither is used by the
    label-free read-out; per-session fine-tuning trains a fresh head and zeroes and freezes
    the skip, so that the prediction comes from the encoder alone.
    """

    def __init__(self, encoder_kwargs: dict, output_dim: int = 2):
        super().__init__()
        self.encoder = REALMEncoder(**encoder_kwargs)
        d_model = encoder_kwargs['d_model']
        n_channels = encoder_kwargs.get('n_channels', 96)
        n_bands = encoder_kwargs.get('n_bands', 1)

        self.out_proj = nn.Linear(d_model, output_dim)
        self.skip = nn.Linear(n_channels * n_bands, output_dim)

    def forward(self, lfp_data, channel_mask=None, return_intermediates=False):
        """
        Args:
            lfp_data: (B, n_channels, n_bands, T)
            channel_mask: (B, n_channels)
            return_intermediates: if True, also return per-layer encoder outputs

        Returns:
            dict with 'prediction' (B, T, output_dim) and optionally 'intermediates'
        """
        if return_intermediates:
            encoded, intermediates = self.encoder(
                lfp_data, channel_mask=channel_mask, return_intermediates=True)
        else:
            encoded = self.encoder(lfp_data, channel_mask=channel_mask)
            intermediates = None

        B, C, nb, T = lfp_data.shape
        mamba_pred = self.out_proj(encoded)

        lfp_flat = lfp_data.permute(0, 3, 1, 2).reshape(B, T, C * nb)
        skip_pred = self.skip(lfp_flat)

        output = {'prediction': skip_pred + mamba_pred}
        if return_intermediates:
            output['intermediates'] = intermediates
        return output
