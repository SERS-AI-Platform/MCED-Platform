"""Single source of truth for all SERS-AI models.

Consumed by:
    - src/sers/cli/train.py       (sers train --model <key>)
    - src/sers/cli/evaluate.py    (sers evaluate --model <key>)
    - src/sers/cli/compare.py     (sers compare --models a,b,c)
    - scripts/deployment/sers_predict.py  (production inference)

This file pairs with `docs/MODEL_STATUS.md` (human-readable status & rationale).
A pytest test enforces key-set parity between the two; do not drift.

When adding a new model:
    1. Add a ModelSpec entry below.
    2. Update `docs/MODEL_STATUS.md` with the same key + category + rationale.
    3. Create the module/script/artifact paths referenced.
    4. Run `pytest tests/test_model_registry.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Category = Literal["production", "baseline-legacy", "experimental", "archived"]


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a model known to the SERS-AI platform.

    Fields:
        key:               CLI-visible identifier, kebab-case (e.g. "usersnet", "sersnet-ensemble").
        display_name:      Human-readable name shown in CLI / reports.
        category:          Lifecycle tier — see DIRECTORY_CONVENTIONS.md.
        module:            Python import path for the model's library code.
        train_script:      Path (from repo root) to the training entrypoint. None if N/A.
        eval_script:       Path to the evaluation entrypoint. None if N/A.
        artifact_dir:      Path to the artifact directory (typically `artifacts/.../current`).
        paper_name:        Name used in publications / posters / figure legends.
        supports_clinical: Whether the model consumes clinical features (not only spectra).
        supports_compare:  Whether `sers compare` should include this by default.
        notes:             Optional free-form remarks.
    """

    key: str
    display_name: str
    category: Category
    module: str
    train_script: Optional[str]
    eval_script: Optional[str]
    artifact_dir: Optional[str]
    paper_name: str
    supports_clinical: bool = False
    supports_compare: bool = True
    notes: str = ""


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, ModelSpec] = {
    # --- 🟢 Production ---
    "usersnet": ModelSpec(
        key="usersnet",
        display_name="uSERS-Net",
        category="production",
        module="sers.models.usersnet",
        train_script="scripts/training/train_usersnet.py",
        eval_script="scripts/evaluation/eval_usersnet_holdout.py",
        artifact_dir="artifacts/usersnet/current",
        paper_name="uSERS-Net",
        supports_clinical=True,  # True from v1.1.0 (Late Fusion @ meta-learner)
        supports_compare=True,
        notes="Stacking V2 (10 base + ElasticNet meta). Clinical fusion integration skeleton in place; full integration in v1.1.0.",
    ),

    # --- 📘 Baseline-Legacy (retained for comparison) ---
    "resnet18": ModelSpec(
        key="resnet18",
        display_name="ResNet18-1D",
        category="baseline-legacy",
        module="sers.models._legacy.resnet_v1",
        train_script="scripts/training/_legacy/train_resnet.py",
        eval_script="scripts/evaluation/_legacy/eval_resnet.py",
        artifact_dir="artifacts/baselines/resnet18/v1.0.0",
        paper_name="ResNet18-1D (baseline)",
        supports_clinical=False,
        supports_compare=True,
        notes="Two-stage hierarchical (Binary + Cancer Type 8-class). Pre-stacking baseline.",
    ),
    "lr-fusion": ModelSpec(
        key="lr-fusion",
        display_name="LR-Fusion (SERS+Clinical)",
        category="baseline-legacy",
        module="sers.models._legacy.lr_fusion_v1",
        train_script="scripts/training/_legacy/build_lr_fusion.py",
        eval_script=None,
        artifact_dir="artifacts/baselines/lr-fusion/v1.0.0",
        paper_name="LR-Fusion (baseline)",
        supports_clinical=True,
        supports_compare=True,
        notes="Former production. LR on SERS + age/sex/BMI. Pre-stacking.",
    ),
    "sersnet-ensemble": ModelSpec(
        key="sersnet-ensemble",
        display_name="SERS-Net Ensemble",
        category="baseline-legacy",
        module="sers.models._legacy.sersnet_ensemble",
        train_script="scripts/training/_legacy/run_sersnet_ensemble.py",
        eval_script=None,
        artifact_dir="artifacts/baselines/sersnet-ensemble/v1.0.0",
        paper_name="SERS-Net Ensemble (LR+ResNet+Clinical)",
        supports_clinical=True,
        supports_compare=True,
        notes="0.8 * LR_fusion(SERS+clinical) + 0.2 * ResNet18(SERS). Sex constraint in Stage 2. Fair baseline for uSERS-Net v1.1.0.",
    ),

    # --- 🧪 Experimental (research) ---
    "contrastive": ModelSpec(
        key="contrastive",
        display_name="Contrastive",
        category="experimental",
        module="sers.models.experimental.contrastive",
        train_script=None,
        eval_script=None,
        artifact_dir=None,
        paper_name="Contrastive (experimental)",
        supports_clinical=False,
        supports_compare=False,
        notes="Contrastive pretraining. Status TBD — add README in the module.",
    ),
    "cross-attention": ModelSpec(
        key="cross-attention",
        display_name="Cross-Attention",
        category="experimental",
        module="sers.models.experimental.cross_attention",
        train_script=None,
        eval_script=None,
        artifact_dir=None,
        paper_name="Cross-Attention (experimental)",
        supports_clinical=True,
        supports_compare=False,
        notes="Cross-attention fusion. Paired with xai_fusion_analysis.py for SHAP.",
    ),
    "film": ModelSpec(
        key="film",
        display_name="FiLM",
        category="experimental",
        module="sers.models.experimental.film",
        train_script=None,
        eval_script=None,
        artifact_dir=None,
        paper_name="FiLM (experimental)",
        supports_clinical=True,
        supports_compare=False,
        notes="FiLM conditioning. Paired with xai_fusion_analysis.py for SHAP.",
    ),
    "transformer": ModelSpec(
        key="transformer",
        display_name="Transformer",
        category="experimental",
        module="sers.models.experimental.transformer",
        train_script=None,
        eval_script=None,
        artifact_dir=None,
        paper_name="Transformer (experimental)",
        supports_clinical=False,
        supports_compare=False,
        notes="Transformer encoder. Status TBD.",
    ),
    "blc-mfds": ModelSpec(
        key="blc-mfds",
        display_name="BLC-MFDS",
        category="experimental",
        module="sers.models.experimental.blc_mfds",
        train_script=None,
        eval_script=None,
        artifact_dir=None,
        paper_name="BLC-MFDS (experimental)",
        supports_clinical=False,
        supports_compare=False,
        notes="BLC MFDS-specific model. Status unknown — README required.",
    ),
}


# -----------------------------------------------------------------------------
# Accessors
# -----------------------------------------------------------------------------

def get_spec(key: str) -> ModelSpec:
    """Look up a model spec by key. Raises KeyError with the available keys listed."""
    if key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model key: {key!r}. Available: {available}")
    return MODEL_REGISTRY[key]


def list_by_category(category: Category) -> list[ModelSpec]:
    """Return all specs in a given category, sorted by key."""
    return sorted(
        (s for s in MODEL_REGISTRY.values() if s.category == category),
        key=lambda s: s.key,
    )


def list_all() -> list[ModelSpec]:
    """Return all specs, sorted by (category, key)."""
    category_order = {"production": 0, "baseline-legacy": 1, "experimental": 2, "archived": 3}
    return sorted(
        MODEL_REGISTRY.values(),
        key=lambda s: (category_order.get(s.category, 99), s.key),
    )
