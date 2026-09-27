"""
Low-level building blocks: DropPath, RoPE, SSD (the Mamba-2 core), Mamba2Block, BiMamba2Block.

Pure PyTorch: no `mamba_ssm` CUDA kernel is needed, so the same code runs on GPU, CPU and
edge devices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DropPath (Stochastic Depth)
# Ref: Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """Drop entire residual branches during training (stochastic depth).

    No learnable parameters -- does not affect state_dict or checkpoint compat.
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


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# Ref: Su et al., "RoFormer", 2021
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Precompute cos/sin cache for rotary position embedding."""

    def __init__(self, d_head, base=10000, max_pos=4096):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_pos, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # (max_pos, d_head/2)
        emb = torch.cat((freqs, freqs), dim=-1)         # (max_pos, d_head)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, position_ids):
        """position_ids: (T,) or (B, T) -> cos, sin of same shape + (..., d_head)."""
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]
        return cos, sin


def rotate_half(x):
    """Rotate half of the hidden dims: [x1, x2] -> [-x2, x1]."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_ssd(q, k, cos, sin):
    """Apply RoPE to SSD's Q/K. Shapes: Q,K=(B, nheads, T, d_state), cos,sin=(T, d_state)."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, d_state)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# SSD (State Space Duality) -- Mamba-2 core
# Ref: Dao & Gu, "Transformers are SSMs", ICML 2024
# ---------------------------------------------------------------------------

class SSD(nn.Module):
    """Pure-PyTorch State Space Duality (Mamba-2 core).

    Multi-head SSM with a scalar A per head, computed in its dual (attention-like) form,
    with rotary position embedding on B (keys) and C (queries).
    """

    def __init__(self, d_inner, headdim=64, d_state=64, ngroups=1):
        super().__init__()
        self.d_inner = d_inner
        self.headdim = headdim
        self.d_state = d_state
        self.nheads = d_inner // headdim
        self.ngroups = ngroups
        assert d_inner % headdim == 0
        assert self.nheads % ngroups == 0

        self.bc_proj = nn.Linear(d_inner, 2 * ngroups * d_state, bias=False)
        self.dt_proj = nn.Linear(d_inner, self.nheads, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, self.nheads + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.nheads))

        # RoPE for relative position encoding on Q(C) and K(B)
        self.rope = RotaryEmbedding(d_state, max_pos=4096)

    def forward(self, x, position_ids=None):
        """SSD dual form: attention-like O(T^2) matmul, fully parallel.

        From Dao & Gu (2024): the linear recurrence h[t] = decay[t]*h[t-1] + B[t]*x[t]
        with output y[t] = C[t]^T h[t] can be rewritten as:
            y = (L * (C @ B^T)) @ x
        where L[i,j] = prod_{k=j+1}^{i} decay[k] is the causal decay matrix.

        x: (B, T, d_inner) -> (B, T, d_inner)
        """
        batch, T, _ = x.shape

        bc = self.bc_proj(x)
        B_param, C_param = bc.chunk(2, dim=-1)
        B_param = B_param.view(batch, T, self.ngroups, self.d_state)
        C_param = C_param.view(batch, T, self.ngroups, self.d_state)

        dt = F.softplus(self.dt_proj(x))  # (B, T, nheads)
        A = -torch.exp(self.A_log)

        x_heads = x.view(batch, T, self.nheads, self.headdim)
        heads_per_group = self.nheads // self.ngroups

        B_exp = B_param.repeat_interleave(heads_per_group, dim=2)
        C_exp = C_param.repeat_interleave(heads_per_group, dim=2)

        # Build causal decay matrix L
        # L[i,j] = prod_{k=j+1}^{i} exp(A * dt[k]) for i >= j, 0 otherwise
        log_decay = A.unsqueeze(0) * dt  # (B, T, nheads), negative values
        cum_log = torch.cumsum(log_decay, dim=1)  # (B, T, nheads)
        cum_h = cum_log.permute(0, 2, 1)  # (B, nheads, T)
        # diff[i,j] = cum[i] - cum[j]; mask upper triangle to -inf before exp
        diff = cum_h.unsqueeze(-1) - cum_h.unsqueeze(-2)  # (B, nheads, T, T)
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        L = torch.exp(diff.masked_fill(~causal, float('-inf')))

        # QKV-style: y = (L * (C @ B^T)) @ x_heads
        Q = C_exp.permute(0, 2, 1, 3)  # (B, nheads, T, d_state)
        K = B_exp.permute(0, 2, 1, 3)  # (B, nheads, T, d_state)
        V = x_heads.permute(0, 2, 1, 3)  # (B, nheads, T, headdim)

        # Apply RoPE to Q(C) and K(B) for relative position encoding
        if position_ids is None:
            position_ids = torch.arange(T, device=x.device)
        cos, sin = self.rope(position_ids)  # (T, d_state)
        Q, K = apply_rotary_pos_emb_ssd(Q, K, cos, sin)

        # Force fp32 for score computation to prevent fp16 overflow
        # (QK^T can reach ~d_state, multiplied by T values in score@V -> overflow)
        score = L * torch.matmul(Q.float(), K.float().transpose(-1, -2))
        y = torch.matmul(score, V.float()).to(V.dtype)  # (B, nheads, T, headdim)

        # D skip connection
        y = y + self.D[None, :, None, None] * V

        return y.permute(0, 2, 1, 3).reshape(batch, T, self.d_inner)


# ---------------------------------------------------------------------------
# Mamba-2 Block (causal)
# ---------------------------------------------------------------------------

class Mamba2Block(nn.Module):
    """Causal Mamba-2 block: gated Conv1d (left-padded) + SSD."""

    def __init__(self, d_model=256, d_state=64, d_conv=4, expand=1, headdim=64):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner,
        )
        self.ssd = SSD(self.d_inner, headdim=headdim, d_state=d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x, position_ids=None):
        B, T, D = x.shape
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        x_conv = x_branch.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :T]
        x_conv = F.silu(x_conv.transpose(1, 2))

        y = self.ssd(x_conv, position_ids=position_ids)
        y = y * F.silu(z)
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# BiMamba-2 Block (bidirectional)
# ---------------------------------------------------------------------------

class BiMamba2Block(nn.Module):
    """Bidirectional Mamba-2: a forward and a time-reversed Mamba-2 block, merged."""

    def __init__(self, d_model=256, d_state=64, d_conv=4, expand=1, headdim=64):
        super().__init__()
        self.fwd = Mamba2Block(d_model, d_state, d_conv, expand, headdim)
        self.bwd = Mamba2Block(d_model, d_state, d_conv, expand, headdim)
        self.merge = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        # Unused projection, kept so the released checkpoints load with strict=True and
        # the parameter initialisation order matches the runs reported in the paper.
        self.causal_proj = nn.Linear(d_model, d_model)
        nn.init.eye_(self.causal_proj.weight)
        nn.init.zeros_(self.causal_proj.bias)

    def forward(self, x, position_ids=None):
        fwd_out = self.fwd(x, position_ids=position_ids)
        # Backward path: flip input, use default position_ids (arange(T))
        bwd_out = self.bwd(x.flip(1)).flip(1)
        merged = self.merge(torch.cat([fwd_out, bwd_out], dim=-1))
        return self.norm(merged)
