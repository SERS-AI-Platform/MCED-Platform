from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr


@dataclass(frozen=True, slots=True)
class PeakAgreement:
    shared_top_peaks: tuple[str, ...]
    jaccard: float
    spearman_rho: float


def peak_agreement(
    names: tuple[str, ...],
    legacy_importance: np.ndarray,
    powder_importance: np.ndarray,
    *,
    top_k: int,
) -> PeakAgreement:
    legacy_order = np.argsort(-legacy_importance, kind="stable")
    powder_order = np.argsort(-powder_importance, kind="stable")
    legacy_top = {names[index] for index in legacy_order[:top_k]}
    powder_top = {names[index] for index in powder_order[:top_k]}
    shared = tuple(names[index] for index in legacy_order[:top_k] if names[index] in powder_top)
    union = legacy_top | powder_top
    rho = float(spearmanr(legacy_importance, powder_importance).statistic)
    return PeakAgreement(
        shared_top_peaks=shared,
        jaccard=float(len(legacy_top & powder_top) / len(union)),
        spearman_rho=rho,
    )
