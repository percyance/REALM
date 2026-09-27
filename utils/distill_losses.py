"""
Retrospective knowledge distillation (RKD) loss: bidirectional teacher -> student.

    L = lambda_task * L_task + lambda_repr * L_repr + lambda_ae * L_ae

  - L_repr: 1 - cosine similarity between the student's and the teacher's last-layer
    representations, per timestep (the teacher is frozen)
  - L_ae:   reconstruction of the input LFP from the student's last-layer representation
            by a two-layer head f_psi, scored on a random 30% of the timesteps
  - L_task: MSE between the student's velocity prediction and the recorded velocity

The paper sets lambda_task = 0 (no behavioural label enters distillation) and
lambda_repr = lambda_ae = 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class RKDLoss(nn.Module):
    """Representation alignment + reconstruction (+ optional task) distillation loss."""

    def __init__(
        self,
        lambda_task: float = 0.0,
        lambda_repr: float = 1.0,
        lambda_ae: float = 1.0,
        d_model: int = 256,
        n_channels: int = 96,
        n_bands: int = 1,
        ae_mask_ratio: float = 0.3,
    ):
        super().__init__()
        self.lambda_task = lambda_task
        self.lambda_repr = lambda_repr
        self.lambda_ae = lambda_ae
        self.ae_mask_ratio = ae_mask_ratio

        # Reconstruction head f_psi (lightweight)
        self.ae_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_channels * n_bands),
        )
        # Unused parameter, kept so the random-number stream (and hence the initialisation
        # of everything built after it) matches the runs reported in the paper.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    @staticmethod
    def _cosine_loss(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Per-timestep cosine dissimilarity: mean(1 - cos_sim)."""
        s_flat = s.reshape(-1, s.size(-1))
        t_flat = t.reshape(-1, t.size(-1))
        return (1 - F.cosine_similarity(s_flat, t_flat, dim=-1)).mean()

    def forward(
        self,
        student_pred: torch.Tensor,       # (B, T, 2)
        target: torch.Tensor,             # (B, T, 2)
        student_repr: torch.Tensor,       # (B, T, d_model), student's last layer
        teacher_repr: torch.Tensor,       # (B, T, d_model), teacher's last layer
        lfp_data: torch.Tensor,           # (B, n_channels, n_bands, T)
    ) -> Dict[str, torch.Tensor]:

        # L_task
        l_task = F.mse_loss(student_pred, target)

        # L_repr: last-layer alignment
        l_repr = self._cosine_loss(student_repr, teacher_repr)

        # L_ae: reconstruction of the input from the last-layer representation
        B, C, nb, T = lfp_data.shape
        ae_target = lfp_data.permute(0, 3, 1, 2).reshape(B, T, C * nb)
        ae_pred = self.ae_decoder(student_repr)

        n_mask = max(1, int(T * self.ae_mask_ratio))
        noise = torch.rand(B, T, device=student_repr.device)
        ids = torch.argsort(noise, dim=1, descending=True)
        ae_mask = torch.zeros(B, T, device=student_repr.device, dtype=torch.bool)
        ae_mask.scatter_(1, ids[:, :n_mask], True)
        l_ae = F.mse_loss(ae_pred[ae_mask], ae_target[ae_mask])

        loss = (self.lambda_task * l_task
                + self.lambda_repr * l_repr
                + self.lambda_ae * l_ae)

        return {
            'loss': loss,
            'l_task': l_task.item(),
            'l_repr': l_repr.item(),
            'l_ae': l_ae.item(),
        }
