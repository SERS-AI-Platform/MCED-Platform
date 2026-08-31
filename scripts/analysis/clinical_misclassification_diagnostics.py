from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLINICAL = REPO_ROOT / "notebooks/aecd_api_model_baseline_outputs/clinical_df.xlsx"
DEFAULT_PREDICTIONS = REPO_ROOT / "notebooks/aecd_api_model_mean_spectrum_outputs/oof_predictions.csv"
DEFAULT_NOISE = REPO_ROOT / "notebooks/aecd_api_model_all_qc_spectra_outputs/subject_qc_noise_summary.csv"
DEFAULT_OUTPUT = REPO_ROOT / "notebooks/aecd_clinical_misclassification_outputs"
DEFAULT_THRESHOLD = 0.4


PALETTE = {
    "TP": "#2c7fb8",
    "FN": "#f0a202",
    "FP": "#d95f5f",
    "TN": "#7f8c8d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join OOF predictions with deidentified clinical exports and diagnose hard cases."
    )
    parser.add_argument("--clinical", type=Path, default=DEFAULT_CLINICAL)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--noise", type=Path, default=DEFAULT_NOISE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser.parse_args()


def load_joined_data(
    clinical_path: Path,
    predictions_path: Path,
    noise_path: Path,
    threshold: float,
) -> pd.DataFrame:
    clinical = pd.read_excel(clinical_path)
    predictions = pd.read_csv(predictions_path)
    noise = pd.read_csv(noise_path)

    required_prediction = {
        "subject_index_internal",
        "label",
        "y_true",
        "oof_probability",
        "prediction",
    }
    required_noise = {
        "subject_index_internal",
        "subject_key",
        "label",
        "repeats_total",
        "repeats_passed",
        "repeats_excluded",
        "qc_pass_rate",
        "noise_sigma_mean",
        "noise_sigma_p95",
    }
    missing_prediction = required_prediction.difference(predictions.columns)
    missing_noise = required_noise.difference(noise.columns)
    if missing_prediction:
        raise ValueError(f"Prediction export is missing columns: {sorted(missing_prediction)}")
    if missing_noise:
        raise ValueError(f"Noise export is missing columns: {sorted(missing_noise)}")
    if "subject_key" not in clinical.columns or "cohort_group" not in clinical.columns:
        raise ValueError("Clinical export must contain subject_key and cohort_group")

    index_map = noise[["subject_index_internal", "subject_key"]]
    joined = predictions.merge(
        index_map,
        on="subject_index_internal",
        how="left",
        validate="one_to_one",
    )
    joined = joined.merge(
        clinical,
        on="subject_key",
        how="left",
        validate="one_to_one",
        suffixes=("", "_clinical"),
    )
    joined = joined.merge(
        noise.drop(columns=["label", "subject_key"]),
        on="subject_index_internal",
        how="left",
        validate="one_to_one",
    )

    if joined["cohort_group"].isna().any():
        raise ValueError("Some prediction rows did not join to the clinical export")
    if not joined["cohort_group"].astype(str).equals(joined["label"].astype(str)):
        raise ValueError("Prediction labels and clinical cohort labels disagree")

    expected_prediction = (joined["oof_probability"] >= threshold).astype(int)
    if not joined["prediction"].eq(expected_prediction).all():
        raise ValueError("Prediction column does not match the configured threshold")

    joined["analysis_row"] = joined["subject_index_internal"].astype(int) + 1
    joined["y_true_binary"] = joined["cohort_group"].eq("prostate").astype(int)
    joined["margin_to_threshold"] = (joined["oof_probability"] - threshold).abs()
    joined["outcome"] = np.select(
        [
            (joined["y_true_binary"] == 1) & (joined["prediction"] == 1),
            (joined["y_true_binary"] == 1) & (joined["prediction"] == 0),
            (joined["y_true_binary"] == 0) & (joined["prediction"] == 1),
            (joined["y_true_binary"] == 0) & (joined["prediction"] == 0),
        ],
        ["TP", "FN", "FP", "TN"],
        default="unknown",
    )
    joined["collection_month"] = pd.to_datetime(
        joined["collection_date"], errors="coerce"
    ).dt.to_period("M").astype(str)
    return joined.sort_values("subject_index_internal").reset_index(drop=True)


def write_profile(joined: pd.DataFrame, output_dir: Path) -> None:
    features = [
        "cohort_group",
        "cancer_type",
        "gleason_score",
        "grade_group",
        "overall_stage",
        "psa_raw",
        "psa",
        "ua_ph",
        "ua_sg",
        "microscopy_wbc",
        "microscopy_rbc",
        "sex",
        "site_code",
        "collection_month",
    ]
    rows = []
    for feature in features:
        values = joined[feature]
        non_null = values.notna()
        rows.append(
            {
                "feature": feature,
                "dtype": str(values.dtype),
                "n_total": int(len(values)),
                "n_non_null": int(non_null.sum()),
                "missing_rate": float((~non_null).mean()),
                "n_unique": int(values.nunique(dropna=True)),
                "is_constant": bool(values.nunique(dropna=True) <= 1),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "clinical_data_profile.csv", index=False)


def write_patient_review(joined: pd.DataFrame, output_dir: Path) -> None:
    columns = [
        "analysis_row",
        "cohort_group",
        "oof_probability",
        "prediction",
        "outcome",
        "margin_to_threshold",
        "collection_month",
        "psa_raw",
        "ua_ph",
        "ua_sg",
        "microscopy_wbc",
        "microscopy_rbc",
        "gleason_score",
        "grade_group",
        "overall_stage",
        "repeats_total",
        "repeats_passed",
        "repeats_excluded",
        "qc_pass_rate",
        "qc_distance_max",
        "noise_sigma_mean",
        "noise_sigma_p95",
    ]
    review = joined[columns].rename(
        columns={
            "oof_probability": "prediction_probability",
            "prediction": "predicted_positive",
        }
    )
    review.to_csv(output_dir / "patient_error_review.csv", index=False)

    near_threshold = joined["margin_to_threshold"] <= 0.05
    misclassified = joined["outcome"].isin(["FP", "FN"])
    hard_cases = joined.loc[near_threshold | misclassified, columns].copy()
    reasons = []
    for _, row in hard_cases.iterrows():
        row_reasons = []
        if row["outcome"] in {"FP", "FN"}:
            row_reasons.append("misclassified")
        if row["margin_to_threshold"] <= 0.05:
            row_reasons.append("near_threshold")
        reasons.append(";".join(row_reasons))
    hard_cases.insert(1, "review_reason", reasons)
    hard_cases = hard_cases.rename(
        columns={
            "oof_probability": "prediction_probability",
            "prediction": "predicted_positive",
        }
    )
    hard_cases.sort_values(
        ["review_reason", "margin_to_threshold", "analysis_row"]
    ).to_csv(output_dir / "hard_case_review.csv", index=False)


def numeric_error_tables(joined: pd.DataFrame, output_dir: Path) -> None:
    feature_specs = [
        ("psa_raw", "PSA raw"),
        ("ua_ph", "Urine pH"),
        ("ua_sg", "Urine specific gravity"),
        ("grade_group", "Cancer grade group"),
        ("qc_pass_rate", "QC pass rate"),
        ("repeats_excluded", "QC-excluded repeats"),
        ("noise_sigma_mean", "Mean repeat noise SD"),
        ("noise_sigma_p95", "95th percentile repeat noise SD"),
        ("qc_distance_max", "Maximum QC distance"),
    ]
    scopes = [
        ("cancer_only", joined["y_true_binary"].eq(1), "FN", "TP"),
        ("non_cancer_only", joined["y_true_binary"].eq(0), "FP", "TN"),
    ]
    summary_rows = []
    test_rows = []
    for scope, scope_mask, error_label, correct_label in scopes:
        subset = joined.loc[scope_mask].copy()
        for feature, display_name in feature_specs:
            values = pd.to_numeric(subset[feature], errors="coerce")
            subset_values = pd.DataFrame({"value": values, "outcome": subset["outcome"]}).dropna()
            for group in [error_label, correct_label]:
                group_values = subset_values.loc[
                    subset_values["outcome"].eq(group), "value"
                ]
                if len(group_values) == 0:
                    continue
                summary_rows.append(
                    {
                        "scope": scope,
                        "feature": feature,
                        "feature_label": display_name,
                        "group": group,
                        "n": int(len(group_values)),
                        "median": float(group_values.median()),
                        "mean": float(group_values.mean()),
                        "q1": float(group_values.quantile(0.25)),
                        "q3": float(group_values.quantile(0.75)),
                        "min": float(group_values.min()),
                        "max": float(group_values.max()),
                    }
                )
            error_values = subset_values.loc[
                subset_values["outcome"].eq(error_label), "value"
            ].to_numpy()
            correct_values = subset_values.loc[
                subset_values["outcome"].eq(correct_label), "value"
            ].to_numpy()
            p_value = np.nan
            if len(error_values) >= 2 and len(correct_values) >= 2:
                p_value = float(
                    mannwhitneyu(
                        error_values,
                        correct_values,
                        alternative="two-sided",
                    ).pvalue
                )
            test_rows.append(
                {
                    "scope": scope,
                    "feature": feature,
                    "feature_label": display_name,
                    "error_group": error_label,
                    "correct_group": correct_label,
                    "n_error": int(len(error_values)),
                    "n_correct": int(len(correct_values)),
                    "mann_whitney_p": p_value,
                    "interpretation": "exploratory; no multiplicity correction",
                }
            )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "clinical_numeric_by_error_group.csv", index=False
    )
    pd.DataFrame(test_rows).to_csv(
        output_dir / "clinical_numeric_error_tests.csv", index=False
    )


def categorical_error_table(joined: pd.DataFrame, output_dir: Path) -> None:
    specifications = [
        ("cohort_group", "all", joined),
        ("grade_group", "cancer_only", joined.loc[joined["y_true_binary"].eq(1)]),
        ("microscopy_wbc", "all", joined),
        ("microscopy_rbc", "all", joined),
        ("collection_month", "all", joined),
    ]
    rows = []
    for feature, scope, subset in specifications:
        values = subset[feature].astype("string").fillna("<missing>")
        error_mask = subset["outcome"].isin(["FP", "FN"])
        for category, category_mask in values.groupby(values).groups.items():
            selected = subset.index.isin(category_mask)
            n = int(selected.sum())
            errors = int((selected & error_mask.to_numpy()).sum())
            rows.append(
                {
                    "scope": scope,
                    "feature": feature,
                    "category": str(category),
                    "n": n,
                    "errors": errors,
                    "error_rate": float(errors / n) if n else np.nan,
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "clinical_category_error_rates.csv", index=False)


def plot_probability_by_outcome(joined: pd.DataFrame, output_dir: Path) -> None:
    order = ["TN", "FP", "FN", "TP"]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    positions = np.arange(len(order))
    values = [joined.loc[joined["outcome"].eq(group), "oof_probability"] for group in order]
    ax.boxplot(
        values,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        boxprops={"facecolor": "#f7f7f7", "edgecolor": "#333333"},
        medianprops={"color": "#333333", "linewidth": 1.5},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
        flierprops={"marker": "", "markersize": 0},
    )
    rng = np.random.default_rng(42)
    for position, group in zip(positions, order, strict=True):
        group_values = joined.loc[joined["outcome"].eq(group), "oof_probability"].to_numpy()
        jitter = rng.uniform(-0.12, 0.12, size=len(group_values))
        ax.scatter(
            np.full(len(group_values), position) + jitter,
            group_values,
            s=22,
            color=PALETTE[group],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.4,
            label=group,
        )
    ax.axhline(DEFAULT_THRESHOLD, color="#333333", linestyle="--", linewidth=1.0)
    ax.text(
        len(order) - 0.45,
        DEFAULT_THRESHOLD + 0.015,
        f"threshold = {DEFAULT_THRESHOLD:.2f}",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#333333",
    )
    ax.set_xticks(positions, order)
    ax.set_ylim(0.1, 0.8)
    ax.set_ylabel("OOF cancer probability")
    ax.set_title("OOF probability by patient-level outcome")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_dir / "fig01_probability_by_outcome.png", dpi=180)
    plt.close(fig)


def plot_cancer_noise(joined: pd.DataFrame, output_dir: Path) -> None:
    subset = joined.loc[joined["y_true_binary"].eq(1)].copy()
    order = ["TP", "FN"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), sharex=True)
    specs = [
        ("noise_sigma_mean", "Mean repeat noise SD"),
        ("noise_sigma_p95", "95th percentile repeat noise SD"),
    ]
    rng = np.random.default_rng(43)
    for ax, (feature, label) in zip(axes, specs, strict=True):
        values = [subset.loc[subset["outcome"].eq(group), feature] for group in order]
        ax.boxplot(
            values,
            positions=np.arange(len(order)),
            widths=0.52,
            patch_artist=True,
            boxprops={"facecolor": "#f7f7f7", "edgecolor": "#333333"},
            medianprops={"color": "#333333", "linewidth": 1.5},
            whiskerprops={"color": "#555555"},
            capprops={"color": "#555555"},
            flierprops={"marker": "", "markersize": 0},
        )
        for position, group in enumerate(order):
            group_values = subset.loc[subset["outcome"].eq(group), feature].dropna().to_numpy()
            jitter = rng.uniform(-0.10, 0.10, size=len(group_values))
            ax.scatter(
                np.full(len(group_values), position) + jitter,
                group_values,
                s=24,
                color=PALETTE[group],
                alpha=0.8,
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_xticks(np.arange(len(order)), order)
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
    fig.suptitle("Repeat-level noise among cancer patients", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig02_cancer_noise_by_outcome.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_category_error_rates(joined: pd.DataFrame, output_dir: Path) -> None:
    frames = []
    for feature, subset in [
        ("cohort_group", joined),
        ("grade_group", joined.loc[joined["y_true_binary"].eq(1)]),
        ("microscopy_wbc", joined),
        ("microscopy_rbc", joined),
    ]:
        values = subset[feature].astype("string").fillna("<missing>")
        error_mask = subset["outcome"].isin(["FP", "FN"])
        for category, index in values.groupby(values).groups.items():
            selected = subset.index.isin(index)
            n = int(selected.sum())
            errors = int((selected & error_mask.to_numpy()).sum())
            frames.append(
                {
                    "feature": feature,
                    "category": str(category),
                    "n": n,
                    "error_rate": errors / n if n else np.nan,
                }
            )
    summary = pd.DataFrame(frames)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5))
    for ax, feature in zip(axes.ravel(), summary["feature"].drop_duplicates(), strict=True):
        plot_data = summary.loc[summary["feature"].eq(feature)].copy()
        plot_data = plot_data.sort_values("category")
        x = np.arange(len(plot_data))
        ax.bar(x, plot_data["error_rate"], color="#2c7fb8", alpha=0.86)
        ax.set_xticks(x, plot_data["category"], rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Misclassification rate")
        ax.set_title(feature)
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
        for xpos, rate, n in zip(x, plot_data["error_rate"], plot_data["n"], strict=True):
            ax.text(xpos, min(rate + 0.03, 0.97), f"{rate:.0%}\n(n={n})", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Clinical category error rates", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "fig03_clinical_category_error_rates.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_collection_month(joined: pd.DataFrame, output_dir: Path) -> None:
    grouped = (
        joined.assign(error=joined["outcome"].isin(["FP", "FN"]))
        .groupby("collection_month", dropna=False)
        .agg(n=("error", "size"), errors=("error", "sum"))
        .reset_index()
    )
    grouped["error_rate"] = grouped["errors"] / grouped["n"]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(grouped))
    ax.bar(x, grouped["error_rate"], color="#f0a202", alpha=0.9)
    ax.set_xticks(x, grouped["collection_month"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Misclassification rate")
    ax.set_title("Misclassification rate by collection month")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    for xpos, rate, n in zip(x, grouped["error_rate"], grouped["n"], strict=True):
        ax.text(xpos, min(rate + 0.03, 0.97), f"{rate:.0%}\n(n={n})", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "fig04_collection_month_error_rate.png", dpi=180)
    plt.close(fig)


def write_summary(joined: pd.DataFrame, output_dir: Path, threshold: float) -> None:
    cancer = joined.loc[joined["y_true_binary"].eq(1)]
    non_cancer = joined.loc[joined["y_true_binary"].eq(0)]
    fn = cancer.loc[cancer["outcome"].eq("FN")]
    tp = cancer.loc[cancer["outcome"].eq("TP")]
    fp = non_cancer.loc[non_cancer["outcome"].eq("FP")]
    tn = non_cancer.loc[non_cancer["outcome"].eq("TN")]

    def median(frame: pd.DataFrame, column: str) -> float | None:
        value = pd.to_numeric(frame[column], errors="coerce").median()
        return None if pd.isna(value) else float(value)

    outcome_counts = joined["outcome"].value_counts().to_dict()
    summary = {
        "source": {
            "clinical_export": str(DEFAULT_CLINICAL),
            "prediction_export": str(DEFAULT_PREDICTIONS),
            "noise_export": str(DEFAULT_NOISE),
            "database_status": "not refreshed; PGPASSWORD was unavailable",
        },
        "target": {
            "positive_label": "prostate",
            "negative_labels": ["control", "prostate disease control"],
            "threshold": threshold,
        },
        "data": {
            "subjects": int(len(joined)),
            "total_repeats": int(joined["repeats_total"].sum()),
            "all_subjects_have_121_repeats": bool(joined["repeats_total"].eq(121).all()),
            "qc_passed_repeats": int(joined["repeats_passed"].sum()),
            "qc_pass_rate": float(joined["repeats_passed"].sum() / joined["repeats_total"].sum()),
        },
        "outcome_counts": {str(key): int(value) for key, value in outcome_counts.items()},
        "hard_case_counts": {
            "cancer_false_negative": int(len(fn)),
            "cancer_false_negative_rate": float(len(fn) / len(cancer)),
            "non_cancer_false_positive": int(len(fp)),
            "non_cancer_false_positive_rate": float(len(fp) / len(non_cancer)),
            "near_threshold_within_0_05": int((joined["margin_to_threshold"] <= 0.05).sum()),
        },
        "observed_descriptive_findings": {
            "false_positive_prostate_disease_control": int((fp["cohort_group"] == "prostate disease control").sum()),
            "false_positive_control": int((fp["cohort_group"] == "control").sum()),
            "cancer_fn_noise_sigma_mean_median": median(fn, "noise_sigma_mean"),
            "cancer_tp_noise_sigma_mean_median": median(tp, "noise_sigma_mean"),
            "cancer_fn_psa_raw_median": median(fn, "psa_raw"),
            "cancer_tp_psa_raw_median": median(tp, "psa_raw"),
        },
        "missing_or_constant_clinical_fields": {
            "psa_numeric_all_missing": bool(joined["psa"].isna().all()),
            "overall_stage_all_missing": bool(joined["overall_stage"].isna().all()),
            "sex_constant": bool(joined["sex"].nunique(dropna=True) <= 1),
            "site_constant": bool(joined["site_code"].nunique(dropna=True) <= 1),
        },
        "interpretation": [
            "121 repeated spectra are repeated observations from 113 subjects, not 13,673 independent patients.",
            "The clinical comparisons are descriptive OOF error diagnostics, not causal or confirmatory inference.",
            "The AUC and error pattern may be affected by hospital, collection, or batch confounding.",
            "Numeric error-group tests are exploratory and have no multiplicity correction.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = "\n".join(
        [
            "# Clinical misclassification diagnostics",
            "",
            "## Technical summary",
            "",
            f"- The analysis contains **{len(joined)} subjects and {int(joined['repeats_total'].sum()):,} repeated spectra**. Every subject has 121 total acquisitions; the independent clinical unit remains the subject.",
            f"- At threshold **{threshold:.2f}**, the OOF prediction outcomes are TP={len(tp)}, FN={len(fn)}, FP={len(fp)}, TN={len(tn)}.",
            f"- Of the {len(fp)} false positives, {(fp['cohort_group'] == 'prostate disease control').sum()} are prostate disease controls and {(fp['cohort_group'] == 'control').sum()} are controls.",
            f"- Among cancer subjects, the median repeat noise SD is **{median(fn, 'noise_sigma_mean'):.1f} for FN vs {median(tp, 'noise_sigma_mean'):.1f} for TP**. This is an exploratory diagnostic signal, not a causal result.",
            "",
            "## Data and definitions",
            "",
            "Predictions are subject-level out-of-fold probabilities from the QC-passed mean-spectrum stacking run. The positive class is `prostate`; `control` and `prostate disease control` are negative. The review output uses an internal analysis row and omits original subject/sample identifiers.",
            "",
            "## Clinical fields available",
            "",
            "`psa_raw`, urine pH, urine specific gravity, microscopy WBC/RBC, Gleason/grade group, and collection month were available. Parsed PSA and overall stage were entirely missing; sex and site were constant in this export and therefore cannot explain within-cohort errors.",
            "",
            "## Interpretation",
            "",
            "The largest directly observed hard-negative pattern is the prostate disease control group producing 15 false positives. For cancer false negatives, the strongest exploratory signal in the available variables is higher repeat-level noise. QC pass rate itself is not meaningfully separated by outcome in this small sample. The collection-month pattern should be treated as an acquisition/batch sensitivity check, not as a clinical explanation.",
            "",
            "## Limitations and next step",
            "",
            "These are descriptive comparisons of OOF errors. They do not establish that noise causes misclassification, and the available clinical export lacks age, parsed PSA, and stage. The next model experiment should add clinical variables only after a subject-level, nested cross-validation design is fixed and should compare spectrum-only, clinical-only, and fused models on the same folds.",
            "",
        ]
    )
    (output_dir / "clinical_misclassification_report.md").write_text(report, encoding="utf-8")


def run(
    clinical_path: Path,
    predictions_path: Path,
    noise_path: Path,
    output_dir: Path,
    threshold: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joined = load_joined_data(clinical_path, predictions_path, noise_path, threshold)
    write_profile(joined, output_dir)
    write_patient_review(joined, output_dir)
    numeric_error_tables(joined, output_dir)
    categorical_error_table(joined, output_dir)
    plot_probability_by_outcome(joined, output_dir)
    plot_cancer_noise(joined, output_dir)
    plot_category_error_rates(joined, output_dir)
    plot_collection_month(joined, output_dir)
    write_summary(joined, output_dir, threshold)
    print(json.dumps(json.loads((output_dir / "summary.json").read_text()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    args = parse_args()
    run(args.clinical, args.predictions, args.noise, args.output_dir, args.threshold)
