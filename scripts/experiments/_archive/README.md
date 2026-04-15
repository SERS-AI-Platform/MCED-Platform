# Archived Experiments

Completed one-off experiments and ablations. Code preserved for reproducibility; results live in `results/`.

Filename prefix convention: `YYYY-MM_{name}.py`.

| Script | Era | Purpose | Results location |
|---|---|---|---|
| `2026-03_step3_multichannel.py` | 2026-03 | Ablation: 3-channel ResNet input (raw/d1/d2) | `results/_archive/` or git history |
| `2026-03_step4_normalization.py` | 2026-03 | Ablation: SNV normalization effect | `results/_archive/` |
| `2026-03_step5_ensemble.py` | 2026-03 | LR + ResNet simple blend (no clinical) | `results/_archive/` |
| `2026-03_multimodal_early.py` | 2026-03 | Initial multimodal SERS+clinical exploration | `results/_archive/` |
| `2026-03_torch_hp_tune.py` | 2026-03 | PyTorch hyperparameter tuning | `results/tuning/` |
| `2026-04_pan_improvement.py` | 2026-04 | PAN Lasso + threshold optimization | `results/_archive/` |
| `2026-04_pancreatic_binary.py` | 2026-04 | CPAN+YPAN binary classification experiment | `results/_archive/` |
| `2026-04_train_val_test_split.py` | 2026-04 | Train/Val/Test holdout vs CV comparison | `results/_archive/` |

## Status

These scripts are **not on any active path**. They are not wired into CLI (`sers train`) or tests. Do not import from here into production code.

If you want to re-run one, check its imports — many still reference the legacy `from models.model import ...` path and may need `from sers.models._legacy.resnet_v1 import ...` after Phase 4 reorg.

## When to delete

Once the corresponding result artifacts are either (a) migrated to a paper/report or (b) deemed no longer useful, delete the script and rely on git history.
