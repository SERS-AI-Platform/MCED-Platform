from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from clinical_hypothesis_statistics import holm_adjust
from numpy.typing import NDArray
from scipy import stats

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AncovaData:
    log10_psa: FloatArray
    group_codes: FloatArray
    age: FloatArray
    bmi: FloatArray


@dataclass(frozen=True, slots=True)
class LinearFit:
    coefficients: FloatArray
    covariance: FloatArray
    df: float
    sample_size: int


@dataclass(frozen=True, slots=True)
class OmnibusResult:
    statistic: float
    df1: float
    df2: float
    p_value: float
    sample_size: int


@dataclass(frozen=True, slots=True)
class ContrastResult:
    comparison: str
    adjusted_ratio: float
    ci_lower: float
    ci_upper: float
    statistic: float
    df: float
    p_value: float
    p_adjusted: float
    sample_size: int


@dataclass(frozen=True, slots=True)
class SlopeCheck:
    check: str
    statistic: float
    df1: float
    df2: float
    p_value: float
    status: str


@dataclass(frozen=True, slots=True)
class AncovaAnalysis:
    omnibus: OmnibusResult
    contrasts: tuple[ContrastResult, ...]
    slope_checks: tuple[SlopeCheck, ...]


def fit_hc3(outcome: FloatArray, design: FloatArray) -> LinearFit:
    """Fit OLS and calculate heteroscedasticity-consistent HC3 covariance."""
    inverse = np.linalg.inv(design.T @ design)
    coefficients = inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ inverse) * design, axis=1)
    scaled = residuals / (1.0 - leverage)
    covariance = inverse @ (design.T @ ((scaled**2)[:, None] * design)) @ inverse
    return LinearFit(coefficients, covariance, float(len(outcome) - design.shape[1]), len(outcome))


def wald_f(fit: LinearFit, contrast: FloatArray) -> OmnibusResult:
    """Test a joint linear hypothesis with an HC3 Wald F statistic."""
    estimate = contrast @ fit.coefficients
    middle = np.linalg.inv(contrast @ fit.covariance @ contrast.T)
    df1 = float(contrast.shape[0])
    statistic = float(estimate.T @ middle @ estimate / df1)
    return OmnibusResult(
        statistic,
        df1,
        fit.df,
        float(stats.f.sf(statistic, df1, fit.df)),
        fit.sample_size,
    )


def base_design(data: AncovaData) -> FloatArray:
    """Construct the prespecified common-slope ANCOVA design."""
    return np.column_stack(
        (
            np.ones(len(data.log10_psa)),
            data.group_codes == 1,
            data.group_codes == 2,
            data.age - np.mean(data.age),
            data.bmi - np.mean(data.bmi),
        )
    ).astype(np.float64)


def interaction_design(data: AncovaData) -> FloatArray:
    """Add group-by-age and group-by-BMI terms for slope diagnostics."""
    base = base_design(data)
    group_1, group_2 = base[:, 1], base[:, 2]
    age, bmi = base[:, 3], base[:, 4]
    return np.column_stack(
        (base, group_1 * age, group_2 * age, group_1 * bmi, group_2 * bmi)
    )


def slope_checks(data: AncovaData) -> tuple[SlopeCheck, ...]:
    """Test whether group-specific covariate slopes are required."""
    fit = fit_hc3(data.log10_psa, interaction_design(data))
    contrasts = (
        ("common_slopes_joint", np.eye(9, dtype=np.float64)[5:9]),
        ("age_slopes", np.eye(9, dtype=np.float64)[5:7]),
        ("bmi_slopes", np.eye(9, dtype=np.float64)[7:9]),
    )
    results: list[SlopeCheck] = []
    for name, contrast in contrasts:
        result = wald_f(fit, contrast)
        status = "not_rejected" if result.p_value >= 0.05 else "interaction_signal"
        results.append(
            SlopeCheck(
                name,
                result.statistic,
                result.df1,
                result.df2,
                result.p_value,
                status,
            )
        )
    return tuple(results)


def adjusted_contrasts(fit: LinearFit) -> tuple[ContrastResult, ...]:
    """Estimate three planned adjusted group contrasts with Holm correction."""
    contrasts = (
        ("Biopsy-negative vs Control", np.array([0.0, 1.0, 0.0, 0.0, 0.0])),
        ("Cancer vs Control", np.array([0.0, 0.0, 1.0, 0.0, 0.0])),
        ("Biopsy-negative vs Cancer", np.array([0.0, 1.0, -1.0, 0.0, 0.0])),
    )
    estimates: list[tuple[str, float, float, float]] = []
    raw_p: list[float] = []
    critical = float(stats.t.ppf(0.975, fit.df))
    for comparison, contrast in contrasts:
        estimate = float(contrast @ fit.coefficients)
        standard_error = float(np.sqrt(contrast @ fit.covariance @ contrast))
        statistic = estimate / standard_error
        estimates.append((comparison, estimate, standard_error, statistic))
        raw_p.append(float(2.0 * stats.t.sf(abs(statistic), fit.df)))
    adjusted = holm_adjust(np.asarray(raw_p, dtype=np.float64))
    return tuple(
        ContrastResult(
            comparison,
            10**estimate,
            10 ** (estimate - critical * standard_error),
            10 ** (estimate + critical * standard_error),
            statistic,
            fit.df,
            raw_p[index],
            float(adjusted[index]),
            fit.sample_size,
        )
        for index, (comparison, estimate, standard_error, statistic) in enumerate(estimates)
    )


def analyze_ancova(data: AncovaData) -> AncovaAnalysis:
    """Test the prespecified adjusted PSA group hypothesis."""
    fit = fit_hc3(data.log10_psa, base_design(data))
    group_test = wald_f(fit, np.eye(5, dtype=np.float64)[1:3])
    return AncovaAnalysis(group_test, adjusted_contrasts(fit), slope_checks(data))
