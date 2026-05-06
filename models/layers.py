"""
Low-level building blocks: DropPath, RoPE, parallel scan, SSD, Mamba2Block, BiMamba2Block.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DropPath (Stochastic Depth)
# Ref: Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """Drop entire residual branches during training (stochastic depth).

    No learnable parameters — does not affect state_dict or checkpoint compat.
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
# Ref: Su et al., "RoFormer", 2021; adapted from real_time_lfp/models/patch_transformer.py
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
# Parallel Scan (Blelloch algorithm) for linear recurrence
# Ref: Blelloch, "Prefix Sums and Their Applications", 1990
#      alxndrTL/mamba.py pscan implementation
# ---------------------------------------------------------------------------

def _pscan_noninplace(a, b):
    """Non-inplace recursive parallel scan — autograd safe.

    Computes y[0] = b[0], y[t] = a[t]*y[t-1] + b[t] for t >= 1.
    Returns a NEW tensor (never modifies a or b).

    Args:
        a: (B, T, D) — decay factors
        b: (B, T, D) — input values

    Returns:
        (B, T, D) — scan result
    """
    B, T, D = a.shape

    if T == 1:
        return b

    T_even = (T + 1) // 2   # ceil(T/2)
    T_odd = T // 2           # floor(T/2)

    a_even = a[:, 0::2]     # (B, T_even, D)
    b_even = b[:, 0::2]
    a_odd = a[:, 1::2]      # (B, T_odd, D)
    b_odd = b[:, 1::2]

    # Combine pairs
    b_odd_new = a_odd * b_even[:, :T_odd] + b_odd
    a_odd_new = a_odd * a_even[:, :T_odd]

    # Recurse on odd positions
    result_odd = _pscan_noninplace(a_odd_new, b_odd_new)

    # Even positions: y_even[0] = b_even[0], y_even[k] = a_even[k]*result_odd[k-1] + b_even[k]
    if T_even > 1:
        even_tail = a_even[:, 1:] * result_odd[:, :T_even - 1] + b_even[:, 1:]
        updated_even = torch.cat([b_even[:, :1], even_tail], dim=1)
    else:
        updated_even = b_even

    # Interleave even and odd (non-inplace via stack+reshape or cat)
    if T_even == T_odd:
        # T is even: simple interleave
        result = torch.stack([updated_even, result_odd], dim=2).reshape(B, T, D)
    else:
        # T is odd (T_even = T_odd + 1): interleave pairs, then append last even
        paired = torch.stack(
            [updated_even[:, :T_odd], result_odd], dim=2
        ).reshape(B, T_odd * 2, D)
        result = torch.cat([paired, updated_even[:, T_odd:]], dim=1)

    return result


def pscan(decays, values):
    """Parallel scan for linear recurrence: y[t] = decay[t]*y[t-1] + values[t].

    Args:
        decays: (B, T, D) — multiplicative factors (decay[0] is unused, y[0] = values[0])
        values: (B, T, D) — additive inputs

    Returns:
        (B, T, D) — output sequence
    """
    return _pscan_noninplace(decays, values)


# ---------------------------------------------------------------------------
# SSD (State Space Duality) — Mamba-2 core
# Ref: Dao & Gu, "Transformers are SSMs", ICML 2024
# ---------------------------------------------------------------------------

class SSD(nn.Module):
    """Pure-PyTorch State Space Duality (Mamba-2 core).

    Multi-head SSM with scalar A per head, supports optional state capture.
    Uses parallel scan for fast training; falls back to sequential for state capture.
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

        # State capture for dynamics distillation
        self._capture_states = False
        self._captured_states = None

    def enable_state_capture(self, enable=True):
        self._capture_states = enable

    def get_captured_states(self):
        states = self._captured_states
        self._captured_states = None
        return states

    def forward(self, x, position_ids=None):
        """x: (B, T, d_inner) -> (B, T, d_inner)"""
        if self._capture_states:
            return self._forward_pscan(x, position_ids)
        return self._forward_parallel(x, position_ids)

    def _forward_pscan(self, x, position_ids=None):
        """Parallel scan with state capture — O(T log T) vs O(T) sequential.

        Uses pscan to compute all hidden states h[t] at once, then
        vectorized output y[t] = C[t]^T h[t] + D * x[t].
        ~3-5x faster than _forward_sequential for T=500.
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

        # Apply RoPE to B(K) and C(Q) — approximate in recurrent form
        if position_ids is None:
            position_ids = torch.arange(T, device=x.device)
        cos, sin = self.rope(position_ids)  # (T, d_state)
        cos_bc = cos.unsqueeze(0).unsqueeze(2)  # (1, T, 1, d_state)
        sin_bc = sin.unsqueeze(0).unsqueeze(2)
        B_exp = B_exp * cos_bc + rotate_half(B_exp) * sin_bc
        C_exp = C_exp * cos_bc + rotate_half(C_exp) * sin_bc

        # Decay per head per timestep: scalar broadcast to (headdim, d_state)
        decay = torch.exp(A.unsqueeze(0) * dt)  # (B, T, nheads)

        BN = batch * self.nheads
        HD = self.headdim * self.d_state

        # Reshape for pscan: (B*nheads, T, headdim*d_state)
        a = decay.permute(0, 2, 1).reshape(BN, T, 1).expand(BN, T, HD)
        vals = x_heads.unsqueeze(-1) * B_exp.unsqueeze(-2)  # (B, T, nheads, headdim, d_state)
        b = vals.permute(0, 2, 1, 3, 4).reshape(BN, T, HD)

        # Parallel scan: compute all h[t] at once
        h_all = pscan(a, b)  # (BN, T, HD)
        h_all = h_all.reshape(batch, self.nheads, T, self.headdim, self.d_state)

        # Output: y[t] = (h[t] * C[t]).sum(d_state) + D * x[t]
        C_broad = C_exp.permute(0, 2, 1, 3).unsqueeze(3)  # (B, nheads, T, 1, d_state)
        y = (h_all * C_broad).sum(-1)  # (B, nheads, T, headdim)
        y = y + self.D[None, :, None, None] * x_heads.permute(0, 2, 1, 3)

        output = y.permute(0, 2, 1, 3).reshape(batch, T, self.d_inner)

        # Capture states: pool h over d_state dim -> (B, T, d_inner)
        # Clone to avoid inplace modification issues during backward
        self._captured_states = h_all.mean(dim=-1).permute(
            0, 2, 1, 3).reshape(batch, T, self.d_inner).clone()

        return output

    def _forward_parallel(self, x, position_ids=None):
        """SSD dual form: attention-like O(T^2) matmul, fully parallel.

        From Dao & Gu (2024): the linear recurrence h[t] = decay[t]*h[t-1] + B[t]*x[t]
        with output y[t] = C[t]^T h[t] can be rewritten as:
            y = (L * (C @ B^T)) @ x
        where L[i,j] = prod_{k=j+1}^{i} decay[k] is the causal decay matrix.

        O(T^2) compute but O(1) sequential depth — fast matmul on GPU.
        Memory: O(T^2) for the attention matrix (~512MB for B=128, T=500, fp16).
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
        # (QK^T can reach ~d_state, multiplied by T values in score@V → overflow)
        score = L * torch.matmul(Q.float(), K.float().transpose(-1, -2))
        y = torch.matmul(score, V.float()).to(V.dtype)  # (B, nheads, T, headdim)

        # D skip connection
        y = y + self.D[None, :, None, None] * V

        return y.permute(0, 2, 1, 3).reshape(batch, T, self.d_inner)

    def _forward_sequential(self, x, position_ids=None):
        """Sequential path with state capture (for distillation)."""
        batch, T, _ = x.shape

        bc = self.bc_proj(x)
        B_param, C_param = bc.chunk(2, dim=-1)
        B_param = B_param.view(batch, T, self.ngroups, self.d_state)
        C_param = C_param.view(batch, T, self.ngroups, self.d_state)

        dt = F.softplus(self.dt_proj(x))
        A = -torch.exp(self.A_log)

        x_heads = x.view(batch, T, self.nheads, self.headdim)
        heads_per_group = self.nheads // self.ngroups

        # Precompute RoPE for all timesteps
        if position_ids is None:
            position_ids = torch.arange(T, device=x.device)
        cos, sin = self.rope(position_ids)  # (T, d_state)

        h = torch.zeros(batch, self.nheads, self.headdim, self.d_state,
                        device=x.device, dtype=x.dtype)
        outputs = []
        h_states = []

        for t in range(T):
            decay = torch.exp(A * dt[:, t]).unsqueeze(-1).unsqueeze(-1)
            x_t = x_heads[:, t]
            B_t = B_param[:, t].repeat_interleave(heads_per_group, dim=1)
            C_t = C_param[:, t].repeat_interleave(heads_per_group, dim=1)

            # Apply RoPE to B_t and C_t
            cos_t = cos[t].unsqueeze(0).unsqueeze(0)  # (1, 1, d_state)
            sin_t = sin[t].unsqueeze(0).unsqueeze(0)
            B_t = B_t * cos_t + rotate_half(B_t) * sin_t
            C_t = C_t * cos_t + rotate_half(C_t) * sin_t

            h = decay * h + x_t.unsqueeze(-1) * B_t.unsqueeze(2)

            y_t = (h * C_t.unsqueeze(2)).sum(-1)
            y_t = y_t + self.D[None, :, None] * x_t
            outputs.append(y_t)

            h_pooled = h.mean(dim=-1).reshape(batch, -1)
            h_states.append(h_pooled)

        output = torch.stack(outputs, dim=1).reshape(batch, T, self.d_inner)
        # Clone to avoid inplace modification issues during backward
        self._captured_states = torch.stack(h_states, dim=1).clone()

        return output


# ---------------------------------------------------------------------------
# Mamba-2 Block (causal)
# ---------------------------------------------------------------------------

class Mamba2Block(nn.Module):
    """Mamba-2 block with Conv1d + SSD."""

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

    def enable_state_capture(self, enable=True):
        self.ssd.enable_state_capture(enable)

    def get_captured_states(self):
        return self.ssd.get_captured_states()


# ---------------------------------------------------------------------------
# BiMamba-2 Block (bidirectional)
# ---------------------------------------------------------------------------

class BiMamba2Block(nn.Module):
    """Bidirectional Mamba-2: forward + backward SSD merged."""

    def __init__(self, d_model=256, d_state=64, d_conv=4, expand=1, headdim=64):
        super().__init__()
        self.fwd = Mamba2Block(d_model, d_state, d_conv, expand, headdim)
        self.bwd = Mamba2Block(d_model, d_state, d_conv, expand, headdim)
        self.merge = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, position_ids=None):
        fwd_out = self.fwd(x, position_ids=position_ids)
        # Backward path: flip input, use default position_ids (arange(T))
        bwd_out = self.bwd(x.flip(1)).flip(1)
        merged = self.merge(torch.cat([fwd_out, bwd_out], dim=-1))
        return self.norm(merged)

    def enable_state_capture(self, enable=True):
        self.fwd.enable_state_capture(enable)
        self.bwd.enable_state_capture(enable)

    def get_captured_states(self):
        """Return (fwd_states, bwd_states), each (B, T, d_inner) or None."""
        return (self.fwd.get_captured_states(),
                self.bwd.get_captured_states())
