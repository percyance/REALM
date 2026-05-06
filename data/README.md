# Data placement

Place pre-processed Makin / Flint LFP `.npz` files here:

```
data/
├── Makin/
│   ├── indy_20160622_01_rawlfp.npz
│   ├── indy_20160630_01_rawlfp.npz
│   └── ...   (29 sessions total)
└── Flint/
    ├── Flint_e1_1_rawlfp.npz
    └── ...   (11 sessions total)
```

Each `.npz` must contain at least:

| key | shape | description |
|---|---|---|
| `lfp_data` | `(96, n_bands, T)` | broadband-filtered LFP @ 100 Hz; pad to 96 channels with zeros if monkey has fewer |
| `targets`  | `(T, 2)` | vx, vy in standardized units |

See `utils/dataset.py` for the full I/O spec.

## Sources

- **Makin** (Indy 2016–2017): O'Doherty et al., zenodo public release
- **Flint** (DREAM 2012): Flint et al., open-source motor cortex dataset

The repo's checkpoints are trained on Makin only.
