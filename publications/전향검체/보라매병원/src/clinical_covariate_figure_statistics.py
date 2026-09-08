from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True, slots=True)
class FigureStatistics:
    primary_f: float
    primary_df1: float
    primary_df2: float
    primary_p: float
    biopsy_control_p: float
    cancer_control_p: float
    biopsy_cancer_p: float
    age_f: float
    age_p: float
    age_holm_p: float
    age_gate: str
    age_control_biopsy_p: float
    age_biopsy_cancer_p: float
    bmi_f: float
    bmi_p: float
    bmi_holm_p: float
    bmi_gate: str
    psa_welch_p: float
    psa_kruskal_p: float


def significance_symbol(p_value: float) -> str:
    """Map a p-value to the conventional publication significance symbol."""
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def _number(frame: pl.DataFrame, column: str) -> float:
    return float(frame.select(column).item())


def _comparison_p(frame: pl.DataFrame, comparison: str) -> float:
    return _number(frame.filter(pl.col("comparison") == comparison), "p_adjusted")


def load_figure_statistics(directory: Path) -> FigureStatistics:
    """Load the prespecified test results used as figure annotations."""
    primary = pl.read_csv(directory / "clinical_hypothesis_primary_ancova.csv")
    contrasts = pl.read_csv(directory / "clinical_hypothesis_pairwise_contrasts.csv")
    secondary = pl.read_csv(directory / "clinical_hypothesis_secondary_tests.csv").filter(
        pl.col("level") == "omnibus"
    )
    age = secondary.filter(pl.col("metric") == "Age")
    bmi = secondary.filter(pl.col("metric") == "BMI")
    age_pairwise = pl.read_csv(
        directory / "clinical_hypothesis_secondary_tests.csv"
    ).filter((pl.col("metric") == "Age") & (pl.col("level") == "pairwise"))
    sensitivity = pl.read_csv(
        directory / "clinical_hypothesis_sensitivity_tests.csv"
    ).filter(pl.col("level") == "omnibus")
    return FigureStatistics(
        primary_f=_number(primary, "statistic"),
        primary_df1=_number(primary, "df1"),
        primary_df2=_number(primary, "df2"),
        primary_p=_number(primary, "p_value"),
        biopsy_control_p=_comparison_p(contrasts, "Biopsy-negative vs Control"),
        cancer_control_p=_comparison_p(contrasts, "Cancer vs Control"),
        biopsy_cancer_p=_comparison_p(contrasts, "Biopsy-negative vs Cancer"),
        age_f=_number(age, "statistic"),
        age_p=_number(age, "p_value"),
        age_holm_p=_number(age, "p_adjusted"),
        age_gate=str(age.select("gate_status").item()),
        age_control_biopsy_p=_comparison_p(
            age_pairwise,
            "Control vs Biopsy-negative",
        ),
        age_biopsy_cancer_p=_comparison_p(
            age_pairwise,
            "Biopsy-negative vs Cancer",
        ),
        bmi_f=_number(bmi, "statistic"),
        bmi_p=_number(bmi, "p_value"),
        bmi_holm_p=_number(bmi, "p_adjusted"),
        bmi_gate=str(bmi.select("gate_status").item()),
        psa_welch_p=_number(
            sensitivity.filter(pl.col("method").str.starts_with("Welch")),
            "p_value",
        ),
        psa_kruskal_p=_number(
            sensitivity.filter(pl.col("method") == "Kruskal-Wallis"),
            "p_value",
        ),
    )


def annotation_frame(statistics: FigureStatistics) -> pl.DataFrame:
    """Create a machine-readable audit of the inferential figure labels."""
    rows = [
        {
            "panel": "PSA",
            "role": "primary_omnibus",
            "method": "HC3 ANCOVA Wald F",
            "p_value": statistics.primary_p,
            "p_adjusted": statistics.primary_p,
            "gate_status": "primary",
            "multiplicity": "none for primary omnibus",
            "rationale_code": "adjust_age_bmi_heteroscedasticity",
            "symbol": significance_symbol(statistics.primary_p),
        },
        {
            "panel": "Age",
            "role": "secondary_omnibus",
            "method": "Welch ANOVA",
            "p_value": statistics.age_p,
            "p_adjusted": statistics.age_holm_p,
            "gate_status": statistics.age_gate,
            "multiplicity": "Holm across Age and BMI",
            "rationale_code": "unequal_variance_sample_size",
            "symbol": significance_symbol(statistics.age_holm_p),
        },
        {
            "panel": "PSA",
            "role": "pairwise_biopsy_negative_vs_control",
            "method": "HC3 ANCOVA planned contrast",
            "p_value": statistics.biopsy_control_p,
            "p_adjusted": statistics.biopsy_control_p,
            "gate_status": "primary_follow_up",
            "multiplicity": "Holm across three planned contrasts",
            "rationale_code": "adjusted_pairwise_group_effect",
            "symbol": significance_symbol(statistics.biopsy_control_p),
        },
        {
            "panel": "PSA",
            "role": "pairwise_cancer_vs_control",
            "method": "HC3 ANCOVA planned contrast",
            "p_value": statistics.cancer_control_p,
            "p_adjusted": statistics.cancer_control_p,
            "gate_status": "primary_follow_up",
            "multiplicity": "Holm across three planned contrasts",
            "rationale_code": "adjusted_pairwise_group_effect",
            "symbol": significance_symbol(statistics.cancer_control_p),
        },
        {
            "panel": "PSA",
            "role": "pairwise_biopsy_negative_vs_cancer",
            "method": "HC3 ANCOVA planned contrast",
            "p_value": statistics.biopsy_cancer_p,
            "p_adjusted": statistics.biopsy_cancer_p,
            "gate_status": "primary_follow_up",
            "multiplicity": "Holm across three planned contrasts",
            "rationale_code": "adjusted_pairwise_group_effect",
            "symbol": significance_symbol(statistics.biopsy_cancer_p),
        },
        {
            "panel": "Age",
            "role": "pairwise_control_vs_biopsy_negative",
            "method": "Games-Howell",
            "p_value": statistics.age_control_biopsy_p,
            "p_adjusted": statistics.age_control_biopsy_p,
            "gate_status": "interpretable",
            "multiplicity": "Games-Howell family-wise within Age",
            "rationale_code": "unequal_variance_pairwise_mean",
            "symbol": significance_symbol(statistics.age_control_biopsy_p),
        },
        {
            "panel": "Age",
            "role": "pairwise_biopsy_negative_vs_cancer",
            "method": "Games-Howell",
            "p_value": statistics.age_biopsy_cancer_p,
            "p_adjusted": statistics.age_biopsy_cancer_p,
            "gate_status": "interpretable",
            "multiplicity": "Games-Howell family-wise within Age",
            "rationale_code": "unequal_variance_pairwise_mean",
            "symbol": significance_symbol(statistics.age_biopsy_cancer_p),
        },
        {
            "panel": "BMI",
            "role": "secondary_omnibus",
            "method": "Welch ANOVA",
            "p_value": statistics.bmi_p,
            "p_adjusted": statistics.bmi_holm_p,
            "gate_status": statistics.bmi_gate,
            "multiplicity": "Holm across Age and BMI",
            "rationale_code": "unequal_variance_sample_size",
            "symbol": significance_symbol(statistics.bmi_holm_p),
        },
        {
            "panel": "PSA",
            "role": "welch_sensitivity",
            "method": "Welch ANOVA",
            "p_value": statistics.psa_welch_p,
            "p_adjusted": statistics.psa_welch_p,
            "gate_status": "sensitivity",
            "multiplicity": "none for sensitivity omnibus",
            "rationale_code": "unadjusted_heteroscedastic_sensitivity",
            "symbol": significance_symbol(statistics.psa_welch_p),
        },
        {
            "panel": "PSA",
            "role": "rank_sensitivity",
            "method": "Kruskal-Wallis",
            "p_value": statistics.psa_kruskal_p,
            "p_adjusted": statistics.psa_kruskal_p,
            "gate_status": "sensitivity",
            "multiplicity": "none for sensitivity omnibus",
            "rationale_code": "rank_based_robustness",
            "symbol": significance_symbol(statistics.psa_kruskal_p),
        },
    ]
    return pl.DataFrame(rows)
