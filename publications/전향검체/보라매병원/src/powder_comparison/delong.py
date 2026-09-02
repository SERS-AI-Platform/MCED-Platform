from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True, slots=True)
class DeLongResult:
    legacy_auc: float
    powder_auc: float
    delta: float
    z_score: float
    p_value: float


@dataclass(frozen=True, slots=True)
class DeLongInputError(ValueError):
    labels: tuple[int, ...]

    def __str__(self) -> str:
        return f"DeLong test requires both binary outcome classes; received {self.labels}"


@dataclass(frozen=True, slots=True)
class DeLongConsistencyError(RuntimeError):
    calculated: tuple[float, float]
    expected: tuple[float, float]

    def __str__(self) -> str:
        return (
            "Internal DeLong AUC calculation disagrees with sklearn: "
            f"calculated={self.calculated}, expected={self.expected}"
        )


def _midranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _covariance(scores: np.ndarray, positive_count: int) -> tuple[np.ndarray, np.ndarray]:
    model_count, sample_count = scores.shape
    negative_count = sample_count - positive_count
    positive_ranks = np.empty((model_count, positive_count), dtype=float)
    negative_ranks = np.empty((model_count, negative_count), dtype=float)
    combined_ranks = np.empty((model_count, sample_count), dtype=float)
    for model_index in range(model_count):
        positive_ranks[model_index] = _midranks(scores[model_index, :positive_count])
        negative_ranks[model_index] = _midranks(scores[model_index, positive_count:])
        combined_ranks[model_index] = _midranks(scores[model_index])
    aucs = combined_ranks[:, :positive_count].sum(axis=1) / positive_count / negative_count - (
        positive_count + 1.0
    ) / (2.0 * negative_count)
    positive_components = (combined_ranks[:, :positive_count] - positive_ranks) / negative_count
    negative_components = (
        1.0 - (combined_ranks[:, positive_count:] - negative_ranks) / positive_count
    )
    covariance = (
        np.atleast_2d(np.cov(positive_components, bias=False)) / positive_count
        + np.atleast_2d(np.cov(negative_components, bias=False)) / negative_count
    )
    return aucs, covariance


def correlated_auc_test(
    truth: np.ndarray, legacy_scores: np.ndarray, powder_scores: np.ndarray
) -> DeLongResult:
    labels = np.asarray(truth, dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise DeLongInputError(tuple(int(label) for label in np.unique(labels)))
    positive_first = np.argsort(-labels, kind="stable")
    scores = np.vstack((legacy_scores, powder_scores))[:, positive_first]
    aucs, covariance = _covariance(scores, int(np.sum(labels == 1)))
    contrast = np.array([-1.0, 1.0])
    variance = float(contrast @ covariance @ contrast)
    delta = float(aucs[1] - aucs[0])
    if variance <= np.finfo(float).eps:
        z_score = 0.0 if np.isclose(delta, 0.0) else float("inf")
        p_value = 1.0 if np.isclose(delta, 0.0) else 0.0
    else:
        z_score = float(abs(delta) / np.sqrt(variance))
        p_value = float(2.0 * norm.sf(z_score))
    expected = np.array(
        [roc_auc_score(labels, legacy_scores), roc_auc_score(labels, powder_scores)]
    )
    if not np.allclose(aucs, expected):
        raise DeLongConsistencyError(
            calculated=(float(aucs[0]), float(aucs[1])),
            expected=(float(expected[0]), float(expected[1])),
        )
    return DeLongResult(
        legacy_auc=float(aucs[0]),
        powder_auc=float(aucs[1]),
        delta=delta,
        z_score=z_score,
        p_value=p_value,
    )
