from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Final, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray: TypeAlias = NDArray[np.float64]
GROUPS: Final = ("Control", "Biopsy-negative", "Cancer")
PAIR_INDICES: Final = tuple(combinations(range(3), 2))


@dataclass(frozen=True, slots=True)
class MetricData:
    name: str
    scale: str
    samples: tuple[FloatArray, FloatArray, FloatArray]


@dataclass(frozen=True, slots=True)
class TestResult:
    statistic: float
    df1: float
    df2: float
    p_value: float


@dataclass(frozen=True, slots=True)
class PairwiseResult:
    group_1: str
    group_2: str
    estimate: float
    ci_lower: float
    ci_upper: float
    statistic: float
    df: float
    p_value: float
    p_adjusted: float
    effect_size: float


@dataclass(frozen=True, slots=True)
class SmdResult:
    group_1: str
    group_2: str
    value: float


@dataclass(frozen=True, slots=True)
class MetricAnalysis:
    metric: MetricData
    welch: TestResult
    kruskal: TestResult
    games_howell: tuple[PairwiseResult, ...]
    dunn_holm: tuple[PairwiseResult, ...]
    smd: tuple[SmdResult, ...]


def holm_adjust(p_values: FloatArray) -> FloatArray:
    """Control family-wise error using Holm's step-down procedure."""
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * float(p_values[index])))
        adjusted[index] = running
    return adjusted


def welch_anova(samples: tuple[FloatArray, ...]) -> TestResult:
    """Compare group means without assuming equal variances."""
    counts = np.array([len(values) for values in samples], dtype=np.float64)
    means = np.array([np.mean(values) for values in samples])
    variances = np.array([np.var(values, ddof=1) for values in samples])
    weights = counts / variances
    weight_total = np.sum(weights)
    weighted_mean = np.sum(weights * means) / weight_total
    groups = len(samples)
    adjustment = np.sum(((1.0 - weights / weight_total) ** 2) / (counts - 1.0))
    numerator = np.sum(weights * (means - weighted_mean) ** 2) / (groups - 1)
    denominator = 1.0 + (2.0 * (groups - 2) / (groups**2 - 1)) * adjustment
    statistic = numerator / denominator
    df1 = float(groups - 1)
    df2 = float((groups**2 - 1) / (3.0 * adjustment))
    return TestResult(float(statistic), df1, df2, float(stats.f.sf(statistic, df1, df2)))


def games_howell(samples: tuple[FloatArray, ...]) -> tuple[PairwiseResult, ...]:
    """Compute family-wise pairwise mean comparisons for unequal variances."""
    results: list[PairwiseResult] = []
    critical_probability = 0.95
    for first, second in PAIR_INDICES:
        left, right = samples[first], samples[second]
        left_term = np.var(left, ddof=1) / len(left)
        right_term = np.var(right, ddof=1) / len(right)
        standard_error = float(np.sqrt(left_term + right_term))
        difference = float(np.mean(left) - np.mean(right))
        df = float(
            (left_term + right_term) ** 2
            / (left_term**2 / (len(left) - 1) + right_term**2 / (len(right) - 1))
        )
        q_statistic = float(np.sqrt(2.0) * abs(difference) / standard_error)
        p_value = max(
            float(stats.studentized_range.sf(q_statistic, len(samples), df)),
            float(np.finfo(np.float64).eps),
        )
        critical = float(stats.studentized_range.ppf(critical_probability, len(samples), df))
        half_width = critical * standard_error / np.sqrt(2.0)
        results.append(
            PairwiseResult(
                GROUPS[first],
                GROUPS[second],
                difference,
                difference - half_width,
                difference + half_width,
                q_statistic,
                df,
                p_value,
                p_value,
                float("nan"),
            )
        )
    return tuple(results)


def cliffs_delta(left: FloatArray, right: FloatArray) -> float:
    """Calculate the probability-of-superiority effect size."""
    differences = left[:, None] - right[None, :]
    return float((np.count_nonzero(differences > 0) - np.count_nonzero(differences < 0)) / differences.size)


def dunn_holm(samples: tuple[FloatArray, ...]) -> tuple[PairwiseResult, ...]:
    """Compute Dunn rank comparisons with tie correction and Holm adjustment."""
    combined = np.concatenate(samples)
    ranks = stats.rankdata(combined, method="average")
    _, tie_counts = np.unique(combined, return_counts=True)
    total = len(combined)
    variance = total * (total + 1) / 12.0 - np.sum(tie_counts**3 - tie_counts) / (
        12.0 * (total - 1)
    )
    boundaries = np.cumsum([0, *(len(values) for values in samples)])
    mean_ranks = [
        float(np.mean(ranks[boundaries[index] : boundaries[index + 1]]))
        for index in range(len(samples))
    ]
    raw_p_values: list[float] = []
    statistics: list[float] = []
    for first, second in PAIR_INDICES:
        denominator = np.sqrt(variance * (1 / len(samples[first]) + 1 / len(samples[second])))
        statistic = float((mean_ranks[first] - mean_ranks[second]) / denominator)
        statistics.append(statistic)
        raw_p_values.append(float(2.0 * stats.norm.sf(abs(statistic))))

    adjusted = holm_adjust(np.asarray(raw_p_values, dtype=np.float64))
    results: list[PairwiseResult] = []
    for index, (first, second) in enumerate(PAIR_INDICES):
        left, right = samples[first], samples[second]
        results.append(
            PairwiseResult(
                GROUPS[first],
                GROUPS[second],
                float(np.median(left) - np.median(right)),
                float("nan"),
                float("nan"),
                statistics[index],
                float("nan"),
                raw_p_values[index],
                float(adjusted[index]),
                cliffs_delta(left, right),
            )
        )
    return tuple(results)


def standardized_differences(samples: tuple[FloatArray, ...]) -> tuple[SmdResult, ...]:
    """Calculate pairwise standardized mean differences."""
    results: list[SmdResult] = []
    for first, second in PAIR_INDICES:
        left, right = samples[first], samples[second]
        denominator = np.sqrt((np.var(left, ddof=1) + np.var(right, ddof=1)) / 2.0)
        results.append(
            SmdResult(
                GROUPS[first],
                GROUPS[second],
                float((np.mean(left) - np.mean(right)) / denominator),
            )
        )
    return tuple(results)


def analyze_metric(metric: MetricData) -> MetricAnalysis:
    """Run omnibus, pairwise, rank, and balance analyses for one metric."""
    kruskal = stats.kruskal(*metric.samples)
    return MetricAnalysis(
        metric,
        welch_anova(metric.samples),
        TestResult(float(kruskal.statistic), 2.0, float("nan"), float(kruskal.pvalue)),
        games_howell(metric.samples),
        dunn_holm(metric.samples),
        standardized_differences(metric.samples),
    )
