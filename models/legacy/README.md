# Legacy Models

Last cleaned: 2026-05-19

This directory holds model code and artifacts that are not part of the active
uSERS-Net/STK-V2 workflow.

Active model files left at `models/` root:

- `build_production_stacking.py`
- `stacking_utils.py`
- `model.py` compatibility shim for old imports

Archived here:

- `scripts/`: historical ResNet, LR, train/test, and experiment entry points
- `architectures/`: FiLM, cross-attention, contrastive, transformer, and other experimental variants
- `artifacts/production*`: old production artifact copies; active artifacts live under `artifacts/`

Prefer new code under `src/sers/models/` or `scripts/training/`.
