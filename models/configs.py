"""
Configurations of the released models (REALMEncoder keyword arguments).

Parameter counts: the teacher encoder has 11,496,709 parameters with the session table of
its 34 pretraining sessions (max_sessions=34); REALM and REALM-bi, built as REALMDecoder
with the default max_sessions=200, have 4,909,273 and 5,542,057.
"""

# Teacher: bidirectional Mamba-2 (BiMamba-2), 10 layers, pretrained by continuous masked
# autoencoding.
TEACHER_REALM_KWARGS = dict(
    n_channels=96, n_bands=1, d_model=256, n_layers=10,
    d_state=64, d_conv=4, expand=2, dropout=0.1,
    bidirectional=True, headdim=64, d_channel=8, eca_kernel=5,
    drop_path_rate=0.1,
)

# REALM: causal Mamba-2 student, 10 layers.
STUDENT_REALM_KWARGS = dict(
    n_channels=96, n_bands=1, d_model=256, n_layers=10,
    d_state=64, d_conv=4, expand=2, dropout=0.1,
    bidirectional=False, headdim=64, d_channel=8, eca_kernel=5,
    drop_path_rate=0.0,
)

# REALM-bi: bidirectional student, 8 layers of BiMamba-2 at expand 1.
STUDENT_REALM_BI_KWARGS = dict(
    n_channels=96, n_bands=1, d_model=256, n_layers=8,
    d_state=64, d_conv=4, expand=1, dropout=0.1,
    bidirectional=True, headdim=64, d_channel=8, eca_kernel=5,
    drop_path_rate=0.0,
)

# Student configuration by name (the --student_size choices of the distillation script).
STUDENT_KWARGS = {
    'realm': STUDENT_REALM_KWARGS,
    'realm_bi': STUDENT_REALM_BI_KWARGS,
}
