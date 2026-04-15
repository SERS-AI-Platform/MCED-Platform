# cross_attention

**Status**: experimental (research) — not in production.
**Registry key**: `cross-attention`

## Purpose
TODO: Document the research hypothesis and what this model explores.

## Last known state
- Last modified: see `git log`
- Last experiment results: `results/_archive/` (if any)

## Known issues
- Imports may still reference legacy `from models.model import ...` — update to `from sers.models._legacy.resnet_v1 import ...` after Phase 4.

## How to run
Not currently wired into `sers train`. Run the training script directly:
```bash
python -m sers.models.experimental.cross_attention.train  # if train.py exists
```
