"""환자(군집) 단위 부트스트랩 신뢰구간.

왜 군집 단위인가
----------------
이 데이터는 환자 1명당 스펙트럼이 최대 121개다. 스펙트럼을 독립적으로
재추출하면 13,552개의 독립 표본이 있는 것처럼 계산돼 **신뢰구간이 실제보다
훨씬 좁게** 나온다. 같은 환자의 스펙트럼은 강하게 상관돼 있어 독립이 아니다.
환자 단위로 재추출하면 그 상관구조가 보존되고, 실질 표본수가 스펙트럼 수가
아니라 **환자 수**라는 사실이 구간에 반영된다.

그래서 `groups`는 선택 인자가 아니라 **필수**다. 빠뜨리면 조용히 틀린(좁은)
구간이 나오는 종류의 실수라, 기본값을 두지 않고 예외를 던진다.

BCa를 기본으로 쓰는 이유
------------------------
AUC·민감도·특이도는 [0, 1]로 막혀 있어 1에 가까울수록 부트스트랩 분포가
한쪽으로 치우친다. 단순 백분위 구간은 이때 부정확하다. BCa는 편향(z0)과
왜도(a)를 보정한다. z0가 발산하는 퇴화 상황에서는 백분위로 자동 후퇴하고,
어떤 방식이 쓰였는지 결과에 남긴다.

출처: `scripts/analysis/yonsei_prospective_current_model.py:190`(부트스트랩
골격)과 `scripts/legacy/analysis/weekend_experiments/bootstrap_confidence_intervals.py`
(BCa `bca_ci:399`, 환자 단위 재추출 `resample_by_sample_id:340`)를 합쳐
설치 가능한 패키지로 옮긴 것. 원본의 jackknife는 leave-one-out마다 모델을
재학습했으나, 예측값이 이미 있는 상황에서는 지표만 다시 계산하면 되므로
그 부분은 바꿨다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from sers.evaluation.metrics import MetricFn

DEFAULT_N_BOOT = 2000
DEFAULT_ALPHA = 0.05
DEFAULT_SEED = 20260907


class BootstrapError(ValueError):
    """부트스트랩 입력이 유효하지 않을 때."""


@dataclass(frozen=True, slots=True)
class MetricCI:
    """지표 하나의 점추정과 신뢰구간.

    `n_units`가 실질 표본수(환자 수)이고 `n_rows`는 그 아래 행 수(스펙트럼
    수)다. 보고할 때는 반드시 `n_units`를 쓴다 — `n_rows`를 표본수로 제시하면
    실제보다 훨씬 강한 근거처럼 보인다.
    """

    metric: str
    value: float
    ci_low: float
    ci_high: float
    ci_method: str
    n_boot: int
    n_units: int
    n_rows: int

    @property
    def width(self) -> float:
        return self.ci_high - self.ci_low

    def excludes(self, null_value: float) -> bool:
        """구간이 기준값을 포함하지 않는가 (차이 검정에서 null_value=0)."""
        if not (np.isfinite(self.ci_low) and np.isfinite(self.ci_high)):
            return False
        return null_value < self.ci_low or null_value > self.ci_high


def _group_index(groups: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """고유 그룹과 각 그룹의 행 인덱스를 반환한다."""
    unique = np.unique(groups)
    order = np.argsort(groups, kind="stable")
    sorted_groups = np.asarray(groups)[order]
    boundaries = np.searchsorted(sorted_groups, unique, side="left")
    ends = np.searchsorted(sorted_groups, unique, side="right")
    return unique, [order[start:end] for start, end in zip(boundaries, ends)]


def _validate(y_true: np.ndarray, groups: np.ndarray | None, *scores: np.ndarray) -> None:
    if groups is None:
        raise BootstrapError(
            "groups is required: bootstrap must resample by patient, not by row. "
            "행 단위로 재추출하면 신뢰구간이 실제보다 좁아진다."
        )
    lengths = {len(y_true), len(groups), *(len(s) for s in scores)}
    if len(lengths) != 1:
        raise BootstrapError(f"length mismatch: {lengths}")
    if len(y_true) == 0:
        raise BootstrapError("empty input")


def _bca_interval(
    draws: np.ndarray, observed: float, jackknife: np.ndarray, alpha: float
) -> tuple[float, float, str]:
    """BCa 구간. 퇴화 시 백분위로 후퇴하고 어느 쪽이었는지 함께 반환한다."""
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return float("nan"), float("nan"), "undefined"

    def percentile() -> tuple[float, float, str]:
        return (
            float(np.percentile(finite, 100 * alpha / 2)),
            float(np.percentile(finite, 100 * (1 - alpha / 2))),
            "percentile",
        )

    proportion = float(np.mean(finite < observed))
    if proportion in (0.0, 1.0) or not np.isfinite(observed):
        return percentile()
    z0 = norm.ppf(proportion)

    jk = jackknife[np.isfinite(jackknife)]
    if jk.size < 2:
        return percentile()
    diffs = jk.mean() - jk
    denominator = 6.0 * float(np.sum(diffs**2)) ** 1.5
    acceleration = float(np.sum(diffs**3) / denominator) if denominator else 0.0

    def adjust(z: float) -> float:
        numerator = z0 + z
        return float(norm.cdf(z0 + numerator / (1 - acceleration * numerator)))

    low = np.clip(adjust(norm.ppf(alpha / 2)), 0.0, 1.0)
    high = np.clip(adjust(norm.ppf(1 - alpha / 2)), 0.0, 1.0)
    return (
        float(np.percentile(finite, 100 * low)),
        float(np.percentile(finite, 100 * high)),
        "bca",
    )


def _resample_rows(
    row_groups: list[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    """그룹을 복원추출하고 그 그룹에 속한 행을 통째로 모은다."""
    picked = rng.integers(0, len(row_groups), len(row_groups))
    return np.concatenate([row_groups[i] for i in picked])


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    metric_fn: MetricFn,
    *,
    name: str,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    method: str = "bca",
    seed: int = DEFAULT_SEED,
) -> MetricCI:
    """환자 단위 재추출로 지표 하나의 신뢰구간을 만든다.

    Parameters
    ----------
    y_true, y_score : np.ndarray
        정답과 예측 점수. 행 단위(스펙트럼 또는 환자).
    groups : np.ndarray
        각 행이 속한 재추출 단위(환자 ID). **필수** — 위 모듈 설명 참고.
    metric_fn : MetricFn
        `(y_true, y_score) -> float`.
    method : {"bca", "percentile"}
        기본 BCa. BCa가 퇴화하면 자동으로 백분위로 후퇴하며, 실제 사용된
        방식이 결과의 `ci_method`에 남는다.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    _validate(y_true, groups, y_score)
    groups = np.asarray(groups)

    unique, row_groups = _group_index(groups)
    observed = float(metric_fn(y_true, y_score))

    rng = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = _resample_rows(row_groups, rng)
        draws[i] = metric_fn(y_true[idx], y_score[idx])

    if method == "percentile":
        finite = draws[np.isfinite(draws)]
        low = float(np.percentile(finite, 100 * alpha / 2)) if finite.size else float("nan")
        high = float(np.percentile(finite, 100 * (1 - alpha / 2))) if finite.size else float("nan")
        used = "percentile" if finite.size else "undefined"
    else:
        # jackknife: 환자를 한 명씩 빼고 지표만 다시 계산 (모델 재학습 없음)
        jackknife = np.full(len(unique), np.nan)
        for i in range(len(unique)):
            keep = np.concatenate([row_groups[j] for j in range(len(unique)) if j != i])
            jackknife[i] = metric_fn(y_true[keep], y_score[keep])
        low, high, used = _bca_interval(draws, observed, jackknife, alpha)

    return MetricCI(
        metric=name,
        value=observed,
        ci_low=low,
        ci_high=high,
        ci_method=used,
        n_boot=n_boot,
        n_units=len(unique),
        n_rows=len(y_true),
    )


def bootstrap_difference_ci(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    groups: np.ndarray,
    metric_fn: MetricFn,
    *,
    name: str,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    method: str = "bca",
    seed: int = DEFAULT_SEED,
) -> MetricCI:
    """두 모델의 지표 차이(A − B)에 대한 짝지은 신뢰구간.

    같은 환자에 대한 두 예측을 **함께** 재추출하므로 모델 간 상관이 보존되어,
    각각의 구간을 따로 구해 겹치는지 보는 것보다 검정력이 높다. 어떤 지표에도
    쓸 수 있어 McNemar/DeLong이 없어도 비교가 가능하다 (DeLong은 AUC 전용).

    구간이 0을 포함하면 "이 표본수로는 차이를 판별할 수 없다"는 뜻이다.
    `MetricCI.excludes(0.0)`로 확인한다.
    """
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    _validate(y_true, groups, score_a, score_b)
    groups = np.asarray(groups)

    unique, row_groups = _group_index(groups)

    def difference(idx: np.ndarray) -> float:
        return float(metric_fn(y_true[idx], score_a[idx])) - float(
            metric_fn(y_true[idx], score_b[idx])
        )

    all_rows = np.arange(len(y_true))
    observed = difference(all_rows)

    rng = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    for i in range(n_boot):
        draws[i] = difference(_resample_rows(row_groups, rng))

    if method == "percentile":
        finite = draws[np.isfinite(draws)]
        low = float(np.percentile(finite, 100 * alpha / 2)) if finite.size else float("nan")
        high = float(np.percentile(finite, 100 * (1 - alpha / 2))) if finite.size else float("nan")
        used = "percentile" if finite.size else "undefined"
    else:
        jackknife = np.full(len(unique), np.nan)
        for i in range(len(unique)):
            keep = np.concatenate([row_groups[j] for j in range(len(unique)) if j != i])
            jackknife[i] = difference(keep)
        low, high, used = _bca_interval(draws, observed, jackknife, alpha)

    return MetricCI(
        metric=name,
        value=observed,
        ci_low=low,
        ci_high=high,
        ci_method=used,
        n_boot=n_boot,
        n_units=len(unique),
        n_rows=len(y_true),
    )
