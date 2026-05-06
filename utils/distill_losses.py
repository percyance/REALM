"""
Retrospective distillation losses: Teacher REALM BiMamba2 -> Student REALM Mamba2.

Two approaches:
  A) CrossModalDistill-style (Erturk et al., NeurIPS 2025):
     L = L_task + lambda_repr * L_cosine + lambda_ae * L_autoencoding

  B) Representation + Dynamics distillation (our novelty):
     L = alpha * L_task + beta * L_repr(CKA) + gamma * L_dynamics(SSM states)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Helper: Linear CKA (Centered Kernel Alignment)
# ---------------------------------------------------------------------------

def linear_CKA(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Linear CKA between two representation matrices.

    Args:
        X: (N, d_x) -- e.g., student representations
        Y: (N, d_y) -- e.g., teacher representations

    Returns:
        scalar CKA similarity in [0, 1], higher = more similar
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    XtX = X @ X.t()
    YtY = Y @ Y.t()

    hsic_xy = (XtX * YtY).sum()
    hsic_xx = (XtX * XtX).sum()
    hsic_yy = (YtY * YtY).sum()

    return hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-8)


# ---------------------------------------------------------------------------
# Approach A: CrossModalDistill-style
# ---------------------------------------------------------------------------

class CrossModalDistillLoss(nn.Module):
    """
    L = L_task + lambda_repr * L_cosine_repr + lambda_ae * L_autoencoding

    Following Erturk et al. (NeurIPS 2025):
      - L_task: MSE(student_pred, velocity_target)
      - L_cosine_repr: 1 - cosine_similarity(student_repr, teacher_repr)
        Supports single-layer (last only) or all-layer weighted alignment.
      - L_ae: student autoencoding regularization (reconstruct LFP)
    """

    def __init__(
        self,
        lambda_task: float = 1.0,
        lambda_repr: float = 1.0,
        lambda_ae: float = 0.1,
        d_model: int = 256,
        n_channels: int = 96,
        n_bands: int = 1,
        ae_mask_ratio: float = 0.3,
        align_layers: int = 1,
        align_layer_from_end: Optional[int] = None,
    ):
        super().__init__()
        self.lambda_task = lambda_task
        self.lambda_repr = lambda_repr
        self.lambda_ae = lambda_ae
        self.ae_mask_ratio = ae_mask_ratio
        self.align_layers = align_layers
        # If set (0=last, 1=2nd-to-last, 2=3rd-to-last, ...),
        # align ONLY that single layer instead of the last `align_layers`.
        self.align_layer_from_end = align_layer_from_end

        # Autoencoding decoder (lightweight)
        self.ae_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_channels * n_bands),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    def _cosine_loss(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Per-timestep cosine dissimilarity: mean(1 - cos_sim)."""
        s_flat = s.reshape(-1, s.size(-1))
        t_flat = t.reshape(-1, t.size(-1))
        return (1 - F.cosine_similarity(s_flat, t_flat, dim=-1)).mean()

    def forward(
        self,
        student_pred: torch.Tensor,       # (B, T, 2)
        target: torch.Tensor,             # (B, T, 2)
        student_repr: torch.Tensor,       # (B, T, d_model) or ignored if align_layers > 1
        teacher_repr: torch.Tensor,       # (B, T, d_model) or ignored if align_layers > 1
        lfp_data: torch.Tensor,           # (B, 96, 1, T)
        student_intermediates: Optional[List[torch.Tensor]] = None,
        teacher_intermediates: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:

        # L_task
        l_task = F.mse_loss(student_pred, target)

        # L_cosine_repr
        if (self.align_layer_from_end is not None
                and student_intermediates is not None
                and teacher_intermediates is not None):
            # Single-layer alignment at a specific depth from the end.
            # K=0 -> last layer, K=1 -> 2nd-to-last, K=2 -> 3rd-to-last, ...
            n_s = len(student_intermediates)
            n_t = len(teacher_intermediates)
            K = self.align_layer_from_end
            s_idx = n_s - 1 - K
            t_idx = n_t - 1 - K
            if s_idx < 0 or t_idx < 0:
                raise ValueError(
                    f"align_layer_from_end={K} exceeds available layers "
                    f"(student={n_s}, teacher={n_t})"
                )
            s_layer = student_intermediates[s_idx]
            t_layer = teacher_intermediates[t_idx].detach()
            l_repr = self._cosine_loss(s_layer, t_layer)
        elif (self.align_layers > 1 and student_intermediates is not None
                and teacher_intermediates is not None):
            # Align last N layers: weight_i = (i+1) / sum(1..N)
            n_s = len(student_intermediates)
            n_t = len(teacher_intermediates)
            n_pairs = min(self.align_layers, n_s, n_t)
            # Align from the end: e.g. align_layers=2 -> last 2 layers of each
            weights = [(i + 1) for i in range(n_pairs)]
            w_sum = sum(weights)

            l_repr = torch.tensor(0.0, device=student_pred.device)
            for i in range(n_pairs):
                s_layer = student_intermediates[n_s - n_pairs + i]
                t_layer = teacher_intermediates[n_t - n_pairs + i].detach()
                w = weights[i] / w_sum
                l_repr = l_repr + w * self._cosine_loss(s_layer, t_layer)
        else:
            # Last-layer only (default: align_layers=1)
            l_repr = self._cosine_loss(student_repr, teacher_repr)

        # L_ae: autoencoding with partial masking (use last-layer repr)
        final_repr = student_intermediates[-1] if student_intermediates else student_repr
        B, C, nb, T = lfp_data.shape
        ae_target = lfp_data.permute(0, 3, 1, 2).reshape(B, T, C * nb)
        ae_pred = self.ae_decoder(final_repr)

        n_mask = max(1, int(T * self.ae_mask_ratio))
        noise = torch.rand(B, T, device=final_repr.device)
        ids = torch.argsort(noise, dim=1, descending=True)
        ae_mask = torch.zeros(B, T, device=final_repr.device, dtype=torch.bool)
        ae_mask.scatter_(1, ids[:, :n_mask], True)
        l_ae = F.mse_loss(ae_pred[ae_mask], ae_target[ae_mask])

        loss = self.lambda_task * l_task + self.lambda_repr * l_repr + self.lambda_ae * l_ae

        return {
            'loss': loss,
            'l_task': l_task.item(),
            'l_repr': l_repr.item(),
            'l_ae': l_ae.item(),
        }


# ---------------------------------------------------------------------------
# Approach B: Representation + Dynamics Distillation
# ---------------------------------------------------------------------------

class ReprDynamicsDistillLoss(nn.Module):
    """
    L = alpha * L_task + beta * L_repr + gamma * L_dynamics

    Core novelty:
      - L_repr: CKA alignment between layer-wise representations
      - L_dynamics: MSE between SSM hidden state trajectories
        (unique to Mamba -- not possible in Transformers)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.1,
        n_layers: int = 4,
        student_d_inner: int = 256,   # d_model * expand_student
        teacher_d_inner: int = 512,   # d_model * expand_teacher
        repr_loss_type: str = 'cka',  # 'cka' or 'mse'
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_layers = n_layers
        self.repr_loss_type = repr_loss_type

        # Dynamics projectors: student d_inner -> teacher d_inner
        self.dynamics_projectors = nn.ModuleList([
            nn.Linear(student_d_inner, teacher_d_inner)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        student_pred: torch.Tensor,                 # (B, T, 2)
        target: torch.Tensor,                        # (B, T, 2)
        student_intermediates: List[torch.Tensor],   # list of (B, T, d_model)
        teacher_intermediates: List[torch.Tensor],   # list of (B, T, d_model)
        student_ssm_states: Optional[List] = None,   # list of (B, T, s_d_inner)
        teacher_ssm_states: Optional[List] = None,   # list of (fwd, bwd) tuples
    ) -> Dict[str, torch.Tensor]:

        device = student_pred.device

        # L_task
        l_task = F.mse_loss(student_pred, target)

        # L_repr: layer-wise representation alignment (1:1 mapping)
        l_repr = torch.tensor(0.0, device=device)
        n_pairs = min(len(student_intermediates), len(teacher_intermediates))
        for i in range(n_pairs):
            s_repr = student_intermediates[i]
            t_repr = teacher_intermediates[i].detach()

            if self.repr_loss_type == 'cka':
                s_flat = s_repr.reshape(-1, s_repr.size(-1))
                t_flat = t_repr.reshape(-1, t_repr.size(-1))
                l_repr = l_repr + (1 - linear_CKA(s_flat, t_flat))
            else:
                l_repr = l_repr + F.mse_loss(s_repr, t_repr)

        l_repr = l_repr / max(n_pairs, 1)

        # L_dynamics: SSM hidden state trajectory alignment
        l_dynamics = torch.tensor(0.0, device=device)
        if (student_ssm_states is not None and teacher_ssm_states is not None
                and len(student_ssm_states) > 0
                and len(teacher_ssm_states) > 0):

            n_dyn = min(len(student_ssm_states), len(teacher_ssm_states))
            for i in range(n_dyn):
                s_states = student_ssm_states[i]        # (B, T, s_d_inner)
                t_states_pair = teacher_ssm_states[i]    # (fwd, bwd) tuple

                if s_states is None:
                    continue

                # Match student (causal) to teacher forward direction only
                t_fwd_states = t_states_pair[0]          # (B, T, t_d_inner)
                if t_fwd_states is None:
                    continue

                s_proj = self.dynamics_projectors[i](s_states)
                l_dynamics = l_dynamics + F.mse_loss(s_proj, t_fwd_states.detach())

            l_dynamics = l_dynamics / max(n_dyn, 1)

        loss = self.alpha * l_task + self.beta * l_repr + self.gamma * l_dynamics

        return {
            'loss': loss,
            'l_task': l_task.item(),
            'l_repr': l_repr.item(),
            'l_dynamics': l_dynamics.item(),
        }
