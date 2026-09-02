from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


@dataclass(frozen=True, slots=True)
class OrderEffect:
    rho: float
    p_value: float
    n: int


def _within_stratum_rank(values: np.ndarray, strata: np.ndarray) -> np.ndarray:
    ranked = np.zeros(len(values), dtype=float)
    for stratum in np.unique(strata):
        selected = strata == stratum
        local = rankdata(values[selected], method="average")
        ranked[selected] = local - float(np.mean(local))
    return ranked


def _stratified_rho(order: np.ndarray, values: np.ndarray, strata: np.ndarray) -> float:
    order_rank = _within_stratum_rank(order, strata)
    value_rank = _within_stratum_rank(values, strata)
    denominator = float(np.linalg.norm(order_rank) * np.linalg.norm(value_rank))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(order_rank, value_rank) / denominator)


def stratified_order_effect(
    order: np.ndarray,
    values: np.ndarray,
    strata: np.ndarray,
    *,
    seed: int,
    n_permutations: int,
) -> OrderEffect:
    observed = _stratified_rho(order, values, strata)
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(n_permutations):
        shuffled = values.copy()
        for stratum in np.unique(strata):
            selected = np.flatnonzero(strata == stratum)
            shuffled[selected] = rng.permutation(shuffled[selected])
        if abs(_stratified_rho(order, shuffled, strata)) >= abs(observed):
            exceedances += 1
    return OrderEffect(
        rho=observed,
        p_value=float((exceedances + 1) / (n_permutations + 1)),
        n=len(order),
    )
