#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "matplotlib", "openpyxl", "torch"]
# ///

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypeAlias

import numpy as np
from mapping_clinical_features import load_clinical_table
from run_mapping_multiscale_resnet import (
    CANDIDATE_N,
    aggregate_probabilities,
    fit_logistic,
    metric_row,
    minimum_repeat_rows,
    patient_training_means,
    summarize_repeat_rows,
    write_csv,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "results" / "mapping_clinical_spectrum_logistic_20260825_v1"
SOURCE_OUT = REPO / "results" / "mapping_multimodal_clinical_20260825_v1"
BASE_OUT = REPO / "results" / "mapping_multiscale_resnet_20260825_v1"
MODEL_NAME = "Clinical + Spectrum LR"
AGGREGATIONS = ("mean", "median", "majority", "trimmed_mean")
CsvValue: TypeAlias = str | int | float | bool | None
CsvRow: TypeAlias = dict[str, CsvValue]


class ClinicalMappingMismatchError(ValueError):
    """Raised when clinical rows and patient ordinals do not align."""


@dataclass(frozen=True, slots=True)
class RepeatPlan:
    patient_ids: np.ndarray
    labels: dict[str, np.ndarray]
    predictions: dict[str, np.ndarray]
    iterations: int
    seed: int


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def combined_features(spectrum: np.ndarray, clinical_values: np.ndarray, patient_ids: np.ndarray) -> np.ndarray:
    return np.hstack((spectrum, clinical_values[patient_ids - 1])).astype(np.float32)


def coefficient_rows(model: Pipeline, fold: int, task: str, spectrum_points: int, clinical_names: Sequence[str]) -> list[CsvRow]:
    pipeline = model
    logistic = pipeline.named_steps["logistic"]
    coefficients = np.asarray(logistic.coef_, dtype=float)
    feature_names = [f"spectrum_{index:04d}" for index in range(spectrum_points)] + list(clinical_names)
    rows: list[CsvRow] = []
    for class_index, values in enumerate(coefficients):
        for name, value in zip(feature_names, values, strict=True):
            rows.append(
                {
                    "fold": fold,
                    "task": task,
                    "class_index": class_index,
                    "feature": name,
                    "feature_group": "clinical" if name in clinical_names else "spectrum",
                    "coefficient": float(value),
                    "abs_coefficient": abs(float(value)),
                }
            )
    return rows


def repeat_evaluation(plan: RepeatPlan) -> tuple[list[CsvRow], list[CsvRow], list[CsvRow]]:
    rng = np.random.default_rng(plan.seed)
    patients = np.asarray(sorted(np.unique(plan.patient_ids)), dtype=int)
    rows_by_patient = {int(patient): np.flatnonzero(plan.patient_ids == patient) for patient in patients}
    candidates = tuple(n for n in CANDIDATE_N if n <= min(len(rows) for rows in rows_by_patient.values()))
    repeat_rows: list[CsvRow] = []
    stability_rows: list[CsvRow] = []
    for task, labels in plan.labels.items():
        probabilities = plan.predictions[task]
        for aggregation in AGGREGATIONS:
            full_selection = {int(patient): rows_by_patient[int(patient)] for patient in patients}
            _, _, baseline = aggregate_probabilities(probabilities, plan.patient_ids, labels, aggregation, full_selection)
            baseline_prediction = np.argmax(baseline, axis=1)
            for n in candidates:
                sampled: list[np.ndarray] = []
                for iteration in range(plan.iterations):
                    selected = {int(patient): rng.choice(rows_by_patient[int(patient)], size=n, replace=False) for patient in patients}
                    _, y_patient, probability = aggregate_probabilities(probabilities, plan.patient_ids, labels, aggregation, selected)
                    row = metric_row(task, MODEL_NAME, aggregation, y_patient, probability)
                    row.update({"n": n, "iteration": iteration + 1, "prediction_agreement": float(np.mean(np.argmax(probability, axis=1) == baseline_prediction))})
                    repeat_rows.append(row)
                    sampled.append(probability)
                stack = np.stack(sampled, axis=0)
                patient_sd = np.std(stack, axis=0, ddof=1)
                stability_rows.append(
                    {
                        "task": task,
                        "model": MODEL_NAME,
                        "aggregation": aggregation,
                        "n": n,
                        "probability_sd_mean": float(np.mean(patient_sd)),
                        "probability_sd_median": float(np.median(patient_sd)),
                    }
                )
    minimum: list[CsvRow] = []
    for task in plan.labels:
        for aggregation in AGGREGATIONS:
            minimum.extend(minimum_repeat_rows(task, MODEL_NAME, aggregation, repeat_rows))
    return repeat_rows, stability_rows, minimum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source-output", type=Path, default=SOURCE_OUT)
    parser.add_argument("--base-output", type=Path, default=BASE_OUT)
    parser.add_argument("--mc-iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    arrays = np.load(args.base_output.resolve() / "preprocessed_mapping_arrays.npz")
    spectrum = arrays["X"].astype(np.float32)
    patient_ids = arrays["patient_ids"].astype(int)
    labels = {"cancer_vs_non_cancer": arrays["y_binary"].astype(int), "three_class": arrays["y_three"].astype(int)}
    clinical = load_clinical_table(REPO / "data" / "mapping" / "clinical_df.xlsx")
    if len(clinical.values) != int(patient_ids.max()):
        raise ClinicalMappingMismatchError("clinical row count does not match patient ordinals")
    predictions = {task: np.full((len(spectrum), 2 if task == "cancer_vs_non_cancer" else 3), np.nan, dtype=float) for task in labels}
    coefficients: list[CsvRow] = []
    fold_rows: list[CsvRow] = []
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(splitter.split(spectrum, labels["cancer_vs_non_cancer"], groups=patient_ids), start=1):
        for task, y_values in labels.items():
            train_mean, train_labels, train_patients = patient_training_means(spectrum, y_values, patient_ids, train_index)
            train_features = np.hstack((train_mean, clinical.values[train_patients - 1])).astype(np.float32)
            model, best_c = fit_logistic(train_features, train_labels, train_patients, 500 + fold)
            test_features = combined_features(spectrum[test_index], clinical.values, patient_ids[test_index])
            predictions[task][test_index] = model.predict_proba(test_features)
            coefficients.extend(coefficient_rows(model, fold, task, spectrum.shape[1], clinical.feature_names))
            fold_rows.append({"fold": fold, "task": task, "best_c": best_c, "n_train_patients": len(train_patients), "n_test_patients": len(np.unique(patient_ids[test_index]))})
    metrics: list[CsvRow] = []
    oof_rows: list[CsvRow] = []
    for task, y_values in labels.items():
        for aggregation in AGGREGATIONS:
            ids, y_patient, probability = aggregate_probabilities(predictions[task], patient_ids, y_values, aggregation)
            metrics.append(metric_row(task, MODEL_NAME, aggregation, y_patient, probability))
            for index, patient in enumerate(ids):
                row: CsvRow = {"task": task, "model": MODEL_NAME, "aggregation": aggregation, "subject_ordinal": int(patient), "true_label": int(y_patient[index])}
                row.update({f"probability_class_{class_index}": float(probability[index, class_index]) for class_index in range(probability.shape[1])})
                oof_rows.append(row)
    plan = RepeatPlan(patient_ids, labels, predictions, args.mc_iterations, args.seed)
    repeat_rows, stability_rows, minimum_rows = repeat_evaluation(plan)
    repeat_summary = summarize_repeat_rows(repeat_rows)
    source = args.source_output.resolve()
    source_available = (source / "oof_metrics.csv").exists()
    source_oof = read_csv(source / "oof_metrics.csv") if source_available else []
    source_summary = read_csv(source / "repeat_metrics_summary.csv") if source_available else []
    source_minimum = read_csv(source / "minimum_repeat_count.csv") if source_available else []
    comparison_oof = source_oof + [{key: str(value) for key, value in row.items()} for row in metrics]
    comparison_summary = source_summary + [{key: str(value) for key, value in row.items()} for row in repeat_summary]
    comparison_minimum = source_minimum + [{key: str(value) for key, value in row.items()} for row in minimum_rows]
    write_csv(out / "fold_summary.csv", fold_rows)
    write_csv(out / "combined_lr_oof_metrics.csv", metrics)
    write_csv(out / "combined_lr_patient_oof_predictions.csv", oof_rows)
    np.savez_compressed(out / "combined_lr_oof_prediction_arrays.npz", **predictions)
    write_csv(out / "combined_lr_repeat_metrics_mc.csv", repeat_rows)
    write_csv(out / "combined_lr_repeat_metrics_summary.csv", repeat_summary)
    write_csv(out / "combined_lr_prediction_stability.csv", stability_rows)
    write_csv(out / "combined_lr_minimum_repeat_count.csv", minimum_rows)
    write_csv(out / "combined_lr_coefficients.csv", coefficients)
    write_csv(out / "comparison_oof_metrics.csv", comparison_oof)
    write_csv(out / "comparison_repeat_metrics_summary.csv", comparison_summary)
    write_csv(out / "comparison_minimum_repeat_count.csv", comparison_minimum)
    coefficient_summary: dict[str, CsvValue] = {}
    for group in ("spectrum", "clinical"):
        values = [float(row["abs_coefficient"]) for row in coefficients if row["feature_group"] == group]
        coefficient_summary[group] = float(np.mean(values)) if values else float("nan")
    metadata = {"subject_count": int(len(np.unique(patient_ids)),), "spectrum_rows": int(len(spectrum)), "grid_points": int(spectrum.shape[1]), "clinical_features": list(clinical.feature_names), "mc_iterations": args.mc_iterations, "seed": args.seed, "model": MODEL_NAME, "training_unit": "patient mean spectrum + clinical features", "test_unit": "each spectrum + patient clinical features", "source_oof_available": source_available}
    (out / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "coefficient_summary.json").write_text(json.dumps(coefficient_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "REPORT.md").write_text(
        "\n".join(
            [
                "# Clinical + Spectrum Logistic Regression",
                "",
                f"- Cohort: {metadata['subject_count']} patients, {metadata['spectrum_rows']} spectra, {metadata['grid_points']} spectral points.",
                "- Training input: patient-mean spectrum + five clinical features; test input: each spectrum + patient clinical features.",
                "- OOF metrics: `combined_lr_oof_metrics.csv`; repeat metrics: `combined_lr_repeat_metrics_summary.csv`.",
                "- Full model comparison: `comparison_oof_metrics.csv`, `comparison_repeat_metrics_summary.csv`, `comparison_minimum_repeat_count.csv`.",
                f"- Existing clinical fusion OOF source loaded: {source_available}; if false, comparison files contain only this combined LR.",
                "- Leakage fields excluded: cohort_group, cancer_type, gleason_score, grade_group, overall_stage.",
                "- Coefficients are exploratory and not causal attribution; Cancer Screening may contain cohort or measurement confounding.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(out), "subjects": metadata["subject_count"], "rows": metadata["spectrum_rows"], "combined_oof_rows": len(metrics), "combined_repeat_rows": len(repeat_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
