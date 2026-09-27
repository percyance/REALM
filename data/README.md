# Data

The scripts read everything under a data root, `./data` by default or `$REALM_DATA` if set.

## Sources

- **Makin (Monkey I).** O'Doherty, Cardoso, Makin and Sabes, *Nonhuman Primate Reaching with
  Multichannel Sensorimotor Cortex Electrophysiology*, Zenodo, 2017,
  https://doi.org/10.5281/zenodo.583331. For each session, the `.mat` file (cursor kinematics) and
  the broadband `.nwb` recording.
- **Flint (Monkey C).** Flint, Lindberg, Jordan, Miller and Slutzky, *Accurate decoding of reaching
  movements from field potentials in the absence of spikes*, J. Neural Eng. 9, 046006 (2012); the
  `Flint_2012` data set of the DREAM database (https://crcns.org/data-sets/movements/dream), files
  `Flint_2012_e1.mat` ... `Flint_2012_e5.mat`.

## Layout

```
$REALM_DATA/
├── makin/
│   ├── origin/                  raw: <session>.mat and <session>.nwb (30 sessions)
│   └── preprocessed/lfp/        <session>_rawlfp.npz         (scripts/preprocess_makin.py)
└── flint/
    ├── original/                raw: Flint_2012_e1.mat ... Flint_2012_e5.mat
    └── preprocessed/
        ├── lfp/                 Flint_e<k>_<subject>_rawlfp.npz   (scripts/preprocess_flint.py)
        └── lfp_tukey/           held-out Flint sessions with artifact windows removed
                                 (scripts/make_flint_tukey.py)
```

## Processed files

Each `<session>_rawlfp.npz` holds

| key | shape | content |
|---|---|---|
| `lfp_data` | `(96, 1, T)` | LFP at 100 Hz, 0.05-50 Hz, common-average referenced, z-scored per channel (Flint: 95 channels) |
| `targets`  | `(T, 2)`     | 2D velocity at 100 Hz, z-scored per axis |

plus metadata (`bin_size_s`, `n_channels`, `dataset`, `monkey`, `session`). The files in
`lfp_tukey/` additionally store `tukey_kept_windows`.
