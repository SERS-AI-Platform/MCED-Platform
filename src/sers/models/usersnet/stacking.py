"""Public STK-V2 helper API.

The active trainer lives in `scripts/training/train_usersnet.py`. These helpers
provide a stable import path for training, evaluation, and production builders.
"""

from __future__ import annotations

import numpy as np

from models.stacking_utils import (
    build_classifier,
    configure_runtime,
    load_processed_multichannel,
    load_raw_multichannel,
    preprocess_channel,
    train_base_model,
)


def infer_sex_from_groups(groups) -> np.ndarray:
    """Infer sex constraints from diagnosis groups when clinical sex is absent.

    Encoded convention: 1.0 = male, 0.0 = female, nan = unknown.
    Only PRO and OVA are deterministic from diagnosis labels.
    """
    sex = np.full(len(groups), np.nan, dtype=float)
    groups_arr = np.asarray(groups).astype(str)
    sex[groups_arr == "PRO"] = 1.0
    sex[groups_arr == "OVA"] = 0.0
    return sex


def apply_sex_constraint(type_probs, cancer_types, sex_arr) -> np.ndarray:
    """Mask biologically impossible cancer-type probabilities and renormalize."""
    constrained = np.asarray(type_probs, dtype=float).copy()
    cancer_types = list(cancer_types)
    sex_arr = np.asarray(sex_arr, dtype=float)

    pro_idx = cancer_types.index("PRO") if "PRO" in cancer_types else None
    ova_idx = cancer_types.index("OVA") if "OVA" in cancer_types else None

    for row_i in range(len(constrained)):
        masked = False
        if sex_arr[row_i] == 1.0 and ova_idx is not None:
            constrained[row_i, ova_idx] = 0.0
            masked = True
        elif sex_arr[row_i] == 0.0 and pro_idx is not None:
            constrained[row_i, pro_idx] = 0.0
            masked = True

        if masked:
            row_sum = constrained[row_i].sum()
            if row_sum > 0:
                constrained[row_i] /= row_sum

    return constrained


__all__ = [
    "apply_sex_constraint",
    "build_classifier",
    "configure_runtime",
    "infer_sex_from_groups",
    "load_processed_multichannel",
    "load_raw_multichannel",
    "preprocess_channel",
    "train_base_model",
]
