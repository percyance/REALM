"""
REALM model configurations and factory functions.
"""

from .encoder import REALMEncoder
from .decoder import REALMDecoder


# ── REALM configs ─────────────────────────────────────────────────────

TEACHER_REALM_KWARGS = dict(
    n_channels=96, n_bands=1, d_model=256, n_layers=10,
    d_state=64, d_conv=4, expand=2, dropout=0.1,
    bidirectional=True, headdim=64, d_channel=8, eca_kernel=5,
    n_spatial_patches=1, drop_path_rate=0.1,
)  # ~10.9M params (BiDir, d_model=256, n_layers=10, expand=2)

# ── New canonical naming ──
# REALM-S 2M, REALM 5M, REALM-L 10.5M (all causal); REALM-bi 5M, REALM-bi-L 10.9M (bidir)

STUDENT_REALM_S_KWARGS = dict(
    n_channels=96, n_bands=1, d_model=256, n_layers=8,
    d_state=64, d_conv=4, expand=1, dropout=0.1,
    bidirectional=False, headdim=64, d_channel=8, eca_kernel=5,
    drop_path_rate=0.0,
)  # ~2.1M params — REALM-S (Causal, n_layers=8, expand=1)

STUDENT_REALM_KWARGS = dict(
    n_channels=96, n_bands=1, d_model=256, n_layers=10,
    d_state=64, d_conv=4, expand=2, dropout=0.1,
    bidirectional=False, headdim=64, d_channel=8, eca_kernel=5,
    drop_path_rate=0.0,
)  # ~5M params — REALM (Causal, n_layers=10, expand=2)

STUDENT_REALM_L_KWARGS = dict(
    n_channels=96, n_bands=1, d_model=256, n_layers=22,
    d_state=64, d_conv=4, expand=2, dropout=0.1,
    bidirectional=False, headdim=64, d_channel=8, eca_kernel=5,
    drop_path_rate=0.0,
)  # ~10.5M params — REALM-L (Causal, n_layers=22, expand=2)

STUDENT_REALM_BI_KWARGS = dict(
    n_channels=96, n_bands=1, d_model=256, n_layers=8,
    d_state=64, d_conv=4, expand=1, dropout=0.1,
    bidirectional=True, headdim=64, d_channel=8, eca_kernel=5,
    n_spatial_patches=1, drop_path_rate=0.0,
)  # ~5M params — REALM-bi (BiDir, n_layers=8, expand=1)

STUDENT_REALM_BI_L_KWARGS = dict(
    n_channels=96, n_bands=1, d_model=256, n_layers=10,
    d_state=64, d_conv=4, expand=2, dropout=0.1,
    bidirectional=True, headdim=64, d_channel=8, eca_kernel=5,
    n_spatial_patches=1, drop_path_rate=0.0,
)  # ~10.9M params — REALM-bi-L (BiDir, n_layers=10, expand=2) — same size as teacher

# ── Legacy aliases (for backward compat with old scripts/ckpts) ──
STUDENT_REALM_XL_KWARGS  = STUDENT_REALM_KWARGS       # 5M causal (was XL)
STUDENT_REALM_UXL_KWARGS = STUDENT_REALM_L_KWARGS     # 10.5M causal (was UXL)
STUDENT_REALM_XLBI_KWARGS = STUDENT_REALM_BI_L_KWARGS # 10.9M bidir (was XLBI)
STUDENT_REALM_LBI_KWARGS = dict(                      # 8M bidir — kept for legacy ablation
    n_channels=96, n_bands=1, d_model=256, n_layers=8,
    d_state=64, d_conv=4, expand=2, dropout=0.1,
    bidirectional=True, headdim=64, d_channel=8, eca_kernel=5,
    n_spatial_patches=1, drop_path_rate=0.0,
)


# ── Factory functions ─────────────────────────────────────────────────

def create_teacher(max_sessions=200):
    """Create REALM teacher (~10.9M params, BiDir, d_model=256, n_layers=10, expand=2)."""
    kwargs = {**TEACHER_REALM_KWARGS, 'max_sessions': max_sessions}
    return REALMDecoder(encoder_kwargs=kwargs, output_dim=2)


def create_student_realm(max_sessions=200):
    """Create REALM student (~2.1M params, Causal, d_model=256, n_layers=8, expand=1)."""
    kwargs = {**STUDENT_REALM_KWARGS, 'max_sessions': max_sessions}
    return REALMDecoder(encoder_kwargs=kwargs, output_dim=2)

def create_student_realm_l(max_sessions=200):
    """Create REALM-L student (~4M params, Causal, d_model=256, n_layers=8, expand=2)."""
    kwargs = {**STUDENT_REALM_L_KWARGS, 'max_sessions': max_sessions}
    return REALMDecoder(encoder_kwargs=kwargs, output_dim=2)

def create_student_realm_xl(max_sessions=200):
    """Create REALM-XL student (~5M params, Causal, d_model=256, n_layers=10, expand=2)."""
    kwargs = {**STUDENT_REALM_XL_KWARGS, 'max_sessions': max_sessions}
    return REALMDecoder(encoder_kwargs=kwargs, output_dim=2)

def create_student_realm_uxl(max_sessions=200):
    """Create REALM-UXL student (~10.5M params, Causal, d_model=256, n_layers=22, expand=2)."""
    kwargs = {**STUDENT_REALM_UXL_KWARGS, 'max_sessions': max_sessions}
    return REALMDecoder(encoder_kwargs=kwargs, output_dim=2)

def create_student_realm_xlbi(max_sessions=200):
    """Create REALM-XLbi student (~10.9M params, BiDir, d_model=256, n_layers=10, expand=2) — same as teacher."""
    kwargs = {**STUDENT_REALM_XLBI_KWARGS, 'max_sessions': max_sessions}
    return REALMDecoder(encoder_kwargs=kwargs, output_dim=2)

def create_student_realm_bi(max_sessions=200):
    """Create REALM-BI student (~2.1M params, BiDir, d_model=256, n_layers=8, expand=1)."""
    kwargs = {**STUDENT_REALM_BI_KWARGS, 'max_sessions': max_sessions}
    return REALMDecoder(encoder_kwargs=kwargs, output_dim=2)

def create_student_realm_lbi(max_sessions=200):
    """Create REALM-LBI student (~8M params, BiDir, d_model=256, n_layers=8, expand=2)."""
    kwargs = {**STUDENT_REALM_LBI_KWARGS, 'max_sessions': max_sessions}
    return REALMDecoder(encoder_kwargs=kwargs, output_dim=2)
