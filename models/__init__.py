"""Deprecated compatibility package.

This package was reorganized as part of Phase 4 (2026-04-15):
    - models.model           → sers.models._legacy.resnet_v1.model
    - models.stacking_utils  → sers.models.usersnet.stacking
    - models.clinical_utils  → sers.models.usersnet.clinical_fusion
    - models.train           → (script) scripts/training/_legacy/train_resnet.py
    - models.train_stacking  → (script) scripts/training/train_usersnet.py
    - models.contrastive     → sers.models.experimental.contrastive
    - models.cross_attention → sers.models.experimental.cross_attention
    - models.film            → sers.models.experimental.film
    - models.transformer     → sers.models.experimental.transformer
    - models.blc_mfds        → sers.models.experimental.blc_mfds

The `.train` and `.train_stacking` shims below exist so that legacy /
archived scripts that did `from models.train import load_processed_spectra, ...`
still resolve. New code must not rely on this compat package.
"""

import warnings
warnings.warn(
    "The top-level `models.*` package is deprecated; use `sers.models.*`. "
    "See docs/REORG_PLAN.md.",
    DeprecationWarning,
    stacklevel=2,
)
