"""DeLong 검정 — 같은 표본에서 얻은 두 상관 ROC 곡선의 AUC 비교.

출처: `publications/전향검체/보라매병원/src/powder_comparison/delong.py`를
도메인 중립적으로 일반화해 옮긴 것. 원본 위치는 이 모듈을 재수출하는 얇은
shim으로 바뀌었으므로 구현은 한 곳뿐이다.

이 구현은 자체 검증을 포함한다 — 내부 계산 AUC가 sklearn과 어긋나면
`DeLongConsistencyError`를 던진다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True, slots=True)
class DeLongResult:
    """두 상관 ROC 곡선의 AUC 비교 결과. `delta`는 B − A."""

    auc_a: float
    auc_b: float
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
    truth: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray
) -> DeLongResult:
    """같은 표본에서 얻은 두 AUC를 DeLong 방법으로 비교한다.

    ⚠️ **관측치 독립을 가정한다.** 환자 1명당 스펙트럼이 여러 개인 데이터에
    스펙트럼 단위로 적용하면 분산이 과소평가돼 p값이 과도하게 유의해진다.
    반드시 **환자 단위로 집계한 뒤** 호출할 것. 군집 구조가 있는 채로 비교하려면
    `bootstrap_difference_ci`를 쓴다 (모든 지표에 적용 가능하고 군집도 처리).

    AUC 전용이다. 민감도·특이도·F1 차이는 이 검정으로 다룰 수 없다.
    """
    labels = np.asarray(truth, dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise DeLongInputError(tuple(int(label) for label in np.unique(labels)))
    positive_first = np.argsort(-labels, kind="stable")
    scores = np.vstack((scores_a, scores_b))[:, positive_first]
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
        [roc_auc_score(labels, scores_a), roc_auc_score(labels, scores_b)]
    )
    if not np.allclose(aucs, expected):
        raise DeLongConsistencyError(
            calculated=(float(aucs[0]), float(aucs[1])),
            expected=(float(expected[0]), float(expected[1])),
        )
    return DeLongResult(
        auc_a=float(aucs[0]),
        auc_b=float(aucs[1]),
        delta=delta,
        z_score=z_score,
        p_value=p_value,
    )
