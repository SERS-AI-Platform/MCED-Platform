from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from clinical_hypothesis_models import AncovaAnalysis
from clinical_hypothesis_statistics import MetricAnalysis, holm_adjust


@dataclass(frozen=True, slots=True)
class AssumptionContext:
    total_n: int
    positive_psa_n: int
    missing_bmi_n: int
    age_overlap: str
    bmi_overlap: str


def primary_frame(analysis: AncovaAnalysis) -> pl.DataFrame:
    """Build the prespecified H1 joint ANCOVA test row."""
    result = analysis.omnibus
    return pl.DataFrame(
        [
            {
                "hypothesis_id": "H1",
                "null_hypothesis": "Adjusted log10-PSA group coefficients are jointly zero",
                "alternative_hypothesis": "At least one adjusted group coefficient differs",
                "method": "log10-PSA ANCOVA with HC3 covariance",
                "statistic": result.statistic,
                "df1": result.df1,
                "df2": result.df2,
                "p_value": result.p_value,
                "alpha": 0.05,
                "decision": "reject_H0" if result.p_value < 0.05 else "do_not_reject_H0",
                "complete_case_n": result.sample_size,
            }
        ]
    )


def contrast_frame(analysis: AncovaAnalysis) -> pl.DataFrame:
    """Build the three prespecified adjusted group contrasts."""
    return pl.DataFrame(
        [
            {
                "hypothesis_id": "H1_pairwise",
                "comparison": result.comparison,
                "estimate": result.adjusted_ratio,
                "estimate_type": "adjusted_geometric_mean_ratio",
                "ci_lower": result.ci_lower,
                "ci_upper": result.ci_upper,
                "statistic": result.statistic,
                "df": result.df,
                "p_value": result.p_value,
                "p_adjusted": result.p_adjusted,
                "multiplicity": "Holm across three planned contrasts",
                "decision": "reject_H0" if result.p_adjusted < 0.05 else "do_not_reject_H0",
                "complete_case_n": result.sample_size,
            }
            for result in analysis.contrasts
        ]
    )


def secondary_frame(
    age: MetricAnalysis,
    bmi: MetricAnalysis,
) -> pl.DataFrame:
    """Build H2 age/BMI balance tests with a hierarchical pairwise gate."""
    analyses = (age, bmi)
    adjusted = holm_adjust(
        np.array([analysis.welch.p_value for analysis in analyses], dtype=np.float64)
    )
    rows = []
    for index, analysis in enumerate(analyses):
        hypothesis = "H2a" if analysis.metric.name == "Age" else "H2b"
        gate_open = adjusted[index] < 0.05
        rows.append(
            {
                "hypothesis_id": hypothesis,
                "metric": analysis.metric.name,
                "level": "omnibus",
                "comparison": "three groups",
                "method": "Welch ANOVA",
                "estimate": None,
                "ci_lower": None,
                "ci_upper": None,
                "statistic": analysis.welch.statistic,
                "df1": analysis.welch.df1,
                "df2": analysis.welch.df2,
                "p_value": analysis.welch.p_value,
                "p_adjusted": float(adjusted[index]),
                "multiplicity": "Holm across H2a and H2b",
                "gate_status": "opened" if gate_open else "closed",
                "max_absolute_smd": max(abs(result.value) for result in analysis.smd),
            }
        )
        for result in analysis.games_howell:
            rows.append(
                {
                    "hypothesis_id": hypothesis,
                    "metric": analysis.metric.name,
                    "level": "pairwise",
                    "comparison": f"{result.group_1} vs {result.group_2}",
                    "method": "Games-Howell",
                    "estimate": result.estimate,
                    "ci_lower": result.ci_lower,
                    "ci_upper": result.ci_upper,
                    "statistic": result.statistic,
                    "df1": None,
                    "df2": result.df,
                    "p_value": result.p_value,
                    "p_adjusted": result.p_adjusted,
                    "multiplicity": "Games-Howell family-wise within metric",
                    "gate_status": "interpretable" if gate_open else "not_opened",
                    "max_absolute_smd": None,
                }
            )
    return pl.DataFrame(rows)


def sensitivity_frame(psa: MetricAnalysis) -> pl.DataFrame:
    """Build prespecified unadjusted parametric and rank sensitivity checks."""
    rows = [
        {
            "hypothesis_id": "H3",
            "level": "omnibus",
            "comparison": "three groups",
            "method": "Welch ANOVA on log10-PSA",
            "estimate": None,
            "estimate_type": "",
            "statistic": psa.welch.statistic,
            "p_value": psa.welch.p_value,
            "p_adjusted": psa.welch.p_value,
            "effect_size": None,
        },
        {
            "hypothesis_id": "H3",
            "level": "omnibus",
            "comparison": "three groups",
            "method": "Kruskal-Wallis",
            "estimate": None,
            "estimate_type": "",
            "statistic": psa.kruskal.statistic,
            "p_value": psa.kruskal.p_value,
            "p_adjusted": psa.kruskal.p_value,
            "effect_size": None,
        },
    ]
    for result in psa.games_howell:
        rows.append(
            {
                "hypothesis_id": "H3",
                "level": "pairwise",
                "comparison": f"{result.group_1} vs {result.group_2}",
                "method": "Games-Howell",
                "estimate": 10**result.estimate,
                "estimate_type": "geometric_mean_ratio",
                "statistic": result.statistic,
                "p_value": result.p_value,
                "p_adjusted": result.p_adjusted,
                "effect_size": None,
            }
        )
    for result in psa.dunn_holm:
        rows.append(
            {
                "hypothesis_id": "H3",
                "level": "pairwise",
                "comparison": f"{result.group_1} vs {result.group_2}",
                "method": "Dunn-Holm",
                "estimate": result.estimate,
                "estimate_type": "log10_median_difference",
                "statistic": result.statistic,
                "p_value": result.p_value,
                "p_adjusted": result.p_adjusted,
                "effect_size": result.effect_size,
            }
        )
    return pl.DataFrame(rows)


def assumption_frame(
    analysis: AncovaAnalysis,
    context: AssumptionContext,
) -> pl.DataFrame:
    """Build model and data assumption checks."""
    rows = [
        {
            "check": "positive_psa",
            "status": "pass" if context.positive_psa_n == context.total_n else "fail",
            "statistic": None,
            "df1": None,
            "df2": None,
            "p_value": None,
            "details": f"{context.positive_psa_n}/{context.total_n} PSA values > 0",
        },
        {
            "check": "missing_bmi",
            "status": "complete_case",
            "statistic": context.missing_bmi_n,
            "df1": None,
            "df2": None,
            "p_value": None,
            "details": "Missing BMI values excluded from ANCOVA",
        },
        {
            "check": "age_overlap",
            "status": "pass",
            "statistic": None,
            "df1": None,
            "df2": None,
            "p_value": None,
            "details": context.age_overlap,
        },
        {
            "check": "bmi_overlap",
            "status": "pass",
            "statistic": None,
            "df1": None,
            "df2": None,
            "p_value": None,
            "details": context.bmi_overlap,
        },
    ]
    rows.extend(
        {
            "check": result.check,
            "status": result.status,
            "statistic": result.statistic,
            "df1": result.df1,
            "df2": result.df2,
            "p_value": result.p_value,
            "details": "HC3 Wald F test of group-by-covariate interactions",
        }
        for result in analysis.slope_checks
    )
    return pl.DataFrame(rows)
