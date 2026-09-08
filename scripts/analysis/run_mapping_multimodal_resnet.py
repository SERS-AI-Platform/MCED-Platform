#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "scikit-learn", "matplotlib", "openpyxl", "joblib", "torch"]
# ///

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from mapping_clinical_features import load_clinical_table
from mapping_multimodal_models import (
    fit_clinical_scaler,
    fit_multimodal,
    predict_multimodal,
    transform_clinical,
)
from run_mapping_multiscale_resnet import (
    BLOCK_CHOICES,
    CANDIDATE_N,
    aggregate_probabilities,
    fit_logistic,
    metric_row,
    minimum_repeat_rows,
    write_csv,
)

DEFAULT_OUT = REPO / "results" / "mapping_multimodal_clinical_20260825_v1"
MODEL_NAMES = (
    "Spectrum LR-reference",
    "Spectrum-only CNN",
    "Clinical-only LR",
    "Clinical-input CNN",
)


def summarize_repeat_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task"]), str(row["model"]), str(row["aggregation"]), int(row["n"]))].append(row)
    output: list[dict[str, object]] = []
    metrics = ("roc_auc", "macro_roc_auc", "balanced_accuracy", "sensitivity", "specificity", "macro_f1", "prediction_agreement")
    for (task, model, aggregation, n), values in sorted(grouped.items()):
        summary: dict[str, object] = {"task": task, "model": model, "aggregation": aggregation, "n": n, "iterations": len(values)}
        for metric in metrics:
            numbers = np.asarray([float(value.get(metric, np.nan)) for value in values], dtype=float)
            finite = numbers[np.isfinite(numbers)]
            summary[f"{metric}_mean"] = float(finite.mean()) if len(finite) else float("nan")
            summary[f"{metric}_sd"] = float(finite.std(ddof=1)) if len(finite) > 1 else float("nan")
        output.append(summary)
    return output


def repeat_evaluation(
    patient_ids: np.ndarray,
    y_binary: np.ndarray,
    y_three: np.ndarray,
    predictions: dict[str, np.ndarray],
    iterations: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(seed)
    patients = np.asarray(sorted(np.unique(patient_ids)), dtype=int)
    rows_by_patient = {int(patient): np.flatnonzero(patient_ids == patient) for patient in patients}
    available = min(len(rows) for rows in rows_by_patient.values())
    candidates = tuple(n for n in CANDIDATE_N if n <= available)
    repeat_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    model_keys = {
        "Spectrum LR-reference": "spectrum_lr",
        "Spectrum-only CNN": "spectrum_cnn",
        "Clinical-only LR": "clinical_lr",
        "Clinical-input CNN": "clinical_cnn",
    }
    for task, y_rows, key in (("cancer_vs_non_cancer", y_binary, "binary"), ("three_class", y_three, "three")):
        for model_name in MODEL_NAMES:
            probability = predictions[f"{model_keys[model_name]}_{key}"]
            for aggregation in ("mean", "median", "majority", "trimmed_mean"):
                full_selection = {int(patient): rows_by_patient[int(patient)] for patient in patients}
                _, _, baseline_probability = aggregate_probabilities(probability, patient_ids, y_rows, aggregation, full_selection)
                baseline_prediction = np.argmax(baseline_probability, axis=1)
                for n in candidates:
                    samples: list[np.ndarray] = []
                    for iteration in range(iterations):
                        selected = {
                            int(patient): rng.choice(rows_by_patient[int(patient)], size=n, replace=False)
                            for patient in patients
                        }
                        _, y_patient, patient_probability = aggregate_probabilities(probability, patient_ids, y_rows, aggregation, selected)
                        row = metric_row(task, model_name, aggregation, y_patient, patient_probability)
                        row["n"] = n
                        row["iteration"] = iteration + 1
                        row["prediction_agreement"] = float(np.mean(np.argmax(patient_probability, axis=1) == baseline_prediction))
                        repeat_rows.append(row)
                        samples.append(patient_probability)
                    stack = np.stack(samples, axis=0)
                    patient_sd = np.std(stack, axis=0, ddof=1) if iterations > 1 else np.full(stack.shape[1:], np.nan)
                    stability_rows.append(
                        {
                            "task": task,
                            "model": model_name,
                            "aggregation": aggregation,
                            "n": n,
                            "probability_sd_mean": float(np.mean(patient_sd)),
                            "probability_sd_median": float(np.median(patient_sd)),
                        }
                    )
    minimum_rows: list[dict[str, object]] = []
    for task in ("cancer_vs_non_cancer", "three_class"):
        for model_name in MODEL_NAMES:
            for aggregation in ("mean", "median", "majority", "trimmed_mean"):
                minimum_rows.extend(minimum_repeat_rows(task, model_name, aggregation, repeat_rows))
    minimum_rows.append({"minimum_available_repeats": available, "evaluated_candidates": ",".join(map(str, candidates))})
    return repeat_rows, stability_rows, minimum_rows


def save_figures(out: Path, history: Sequence[dict[str, object]], repeat_summary: Sequence[dict[str, object]], stability: Sequence[dict[str, object]]) -> None:
    write_csv(out / "training_history.csv", list(history))
    for task in ("binary", "three"):
        rows = [row for row in history if row.get("task") == task]
        figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for fold in sorted({int(row["fold"]) for row in rows}):
            fold_rows = [row for row in rows if int(row["fold"]) == fold]
            axes[0].plot([int(row["epoch"]) for row in fold_rows], [float(row["train_loss"]) for row in fold_rows], alpha=0.6, label=f"fold {fold} train")
            axes[0].plot([int(row["epoch"]) for row in fold_rows], [float(row["validation_loss"]) for row in fold_rows], linestyle="--", alpha=0.8, label=f"fold {fold} val")
            axes[1].plot([int(row["epoch"]) for row in fold_rows], [float(row["train_auc"]) for row in fold_rows], alpha=0.6)
            axes[1].plot([int(row["epoch"]) for row in fold_rows], [float(row["validation_auc"]) for row in fold_rows], linestyle="--", alpha=0.8)
        axes[0].set(title=f"{task} loss", xlabel="epoch", ylabel="loss")
        axes[1].set(title=f"{task} AUC", xlabel="epoch", ylabel="AUC")
        axes[0].legend(fontsize=7, ncol=2)
        figure.savefig(out / f"figure_clinical_training_{task}.png", dpi=180)
        plt.close(figure)
    colors = {name: color for name, color in zip(MODEL_NAMES, ("#4C78A8", "#F58518", "#54A24B", "#E45756"), strict=True)}
    for metric, filename, title in (("roc_auc", "figure_clinical_repeat_auc.png", "Clinical fusion: repeat count vs AUC"), ("balanced_accuracy", "figure_clinical_repeat_balanced_accuracy.png", "Clinical fusion: repeat count vs balanced accuracy")):
        figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
        for model in MODEL_NAMES:
            rows = [row for row in repeat_summary if row["task"] == "cancer_vs_non_cancer" and row["model"] == model and row["aggregation"] == "mean"]
            axis.plot([int(row["n"]) for row in rows], [float(row[f"{metric}_mean"]) for row in rows], marker="o", label=model, color=colors[model])
        axis.set(xlabel="number of spectra", ylabel=metric, title=title)
        axis.legend(fontsize=8)
        figure.savefig(out / filename, dpi=180)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for model in MODEL_NAMES:
        rows = [row for row in stability if row["task"] == "cancer_vs_non_cancer" and row["model"] == model and row["aggregation"] == "mean"]
        axis.plot([int(row["n"]) for row in rows], [float(row["probability_sd_mean"]) for row in rows], marker="o", label=model, color=colors[model])
    axis.set(xlabel="number of spectra", ylabel="mean probability SD", title="Clinical fusion: prediction stability")
    axis.legend(fontsize=8)
    figure.savefig(out / "figure_clinical_prediction_stability.png", dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base-output", type=Path, default=REPO / "results" / "mapping_multiscale_resnet_20260825_v1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--mc-iterations", type=int, default=100)
    parser.add_argument("--n-blocks", type=int, choices=BLOCK_CHOICES, default=3)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    base = args.base_output.resolve()
    arrays = np.load(base / "preprocessed_mapping_arrays.npz")
    X = arrays["X"].astype(np.float32)
    patient_ids = arrays["patient_ids"].astype(int)
    y_binary = arrays["y_binary"].astype(int)
    y_three = arrays["y_three"].astype(int)
    clinical = load_clinical_table(REPO / "data" / "mapping" / "clinical_df.xlsx")
    if len(clinical.values) != int(patient_ids.max()):
        raise ValueError("clinical row count does not match mapping patient ordinals")
    write_csv(out / "clinical_feature_summary.csv", list(clinical.summary_rows))
    (out / "clinical_audit.json").write_text(json.dumps(clinical.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    base_predictions = np.load(base / "oof_prediction_arrays.npz")
    predictions = {
        "spectrum_lr_binary": base_predictions["lr_binary"],
        "spectrum_cnn_binary": base_predictions["resnet_binary"],
        "spectrum_lr_three": base_predictions["lr_three"],
        "spectrum_cnn_three": base_predictions["resnet_three"],
        "clinical_lr_binary": np.full((len(X), 2), np.nan, dtype=float),
        "clinical_cnn_binary": np.full((len(X), 2), np.nan, dtype=float),
        "clinical_lr_three": np.full((len(X), 3), np.nan, dtype=float),
        "clinical_cnn_three": np.full((len(X), 3), np.nan, dtype=float),
    }
    history: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(splitter.split(X, y_binary, groups=patient_ids), start=1):
        train_patients = np.asarray(sorted(np.unique(patient_ids[train_index])), dtype=int)
        inner_train, inner_val = next(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=100 + fold).split(X[train_index], y_binary[train_index], groups=patient_ids[train_index]))
        inner_patient_ids = patient_ids[train_index][inner_train]
        scaler = fit_clinical_scaler(clinical.values, inner_patient_ids)
        clinical_all_rows = transform_clinical(clinical.values, scaler)[patient_ids - 1]
        test_patients = np.asarray(sorted(np.unique(patient_ids[test_index])), dtype=int)
        for task_name, y_values, key, n_classes in (("binary", y_binary, "binary", 2), ("three", y_three, "three", 3)):
            train_patient_features = clinical.values[train_patients - 1]
            train_patient_labels = np.asarray([y_values[patient_ids == patient][0] for patient in train_patients], dtype=int)
            clinical_model, best_c = fit_logistic(train_patient_features, train_patient_labels, train_patients, 300 + fold)
            test_probability = clinical_model.predict_proba(clinical.values[test_patients - 1])
            test_lookup = {int(patient): test_probability[index] for index, patient in enumerate(test_patients)}
            predictions[f"clinical_lr_{key}"][test_index] = np.vstack([test_lookup[int(patient)] for patient in patient_ids[test_index]])
            fit = fit_multimodal(
                X[train_index][inner_train],
                clinical_all_rows[train_index][inner_train],
                y_values[train_index][inner_train],
                X[train_index][inner_val],
                clinical_all_rows[train_index][inner_val],
                y_values[train_index][inner_val],
                n_classes,
                len(clinical.feature_names),
                args.n_blocks,
                fold * 10 + n_classes + args.seed,
                args.epochs,
                device,
            )
            predictions[f"clinical_cnn_{key}"][test_index] = predict_multimodal(fit.model, X[test_index], clinical_all_rows[test_index], device)
            for row in fit.history:
                history.append({"fold": fold, "task": task_name, "model": "Clinical-input CNN", **row})
            fold_rows.append({"fold": fold, "task": task_name, "clinical_lr_best_c": best_c, "cnn_best_epoch": fit.best_epoch, "n_train_patients": len(train_patients), "n_test_patients": len(test_patients)})
            del fit
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    write_csv(out / "fold_summary.csv", fold_rows)
    np.savez_compressed(out / "oof_prediction_arrays.npz", **predictions)
    model_keys = {
        "Spectrum LR-reference": "spectrum_lr",
        "Spectrum-only CNN": "spectrum_cnn",
        "Clinical-only LR": "clinical_lr",
        "Clinical-input CNN": "clinical_cnn",
    }
    metrics: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []
    for task, y_rows, key in (("cancer_vs_non_cancer", y_binary, "binary"), ("three_class", y_three, "three")):
        for model_name in MODEL_NAMES:
            probability = predictions[f"{model_keys[model_name]}_{key}"]
            for aggregation in ("mean", "median", "majority", "trimmed_mean"):
                ids, y_patient, patient_probability = aggregate_probabilities(probability, patient_ids, y_rows, aggregation)
                metrics.append(metric_row(task, model_name, aggregation, y_patient, patient_probability))
                for index, patient in enumerate(ids):
                    row: dict[str, object] = {"task": task, "model": model_name, "aggregation": aggregation, "subject_ordinal": int(patient), "true_label": int(y_patient[index])}
                    if patient_probability.shape[1] == 2:
                        row["probability_class_1"] = float(patient_probability[index, 1])
                    else:
                        row.update({f"probability_class_{class_index}": float(patient_probability[index, class_index]) for class_index in range(patient_probability.shape[1])})
                    oof_rows.append(row)
    write_csv(out / "oof_metrics.csv", metrics)
    write_csv(out / "patient_oof_predictions.csv", oof_rows)
    repeat_rows, stability_rows, minimum_rows = repeat_evaluation(patient_ids, y_binary, y_three, predictions, args.mc_iterations, args.seed)
    repeat_summary = summarize_repeat_rows(repeat_rows)
    write_csv(out / "repeat_metrics_mc.csv", repeat_rows)
    write_csv(out / "repeat_metrics_summary.csv", repeat_summary)
    write_csv(out / "prediction_stability.csv", stability_rows)
    write_csv(out / "minimum_repeat_count.csv", [row for row in minimum_rows if "n" in row])
    save_figures(out, history, repeat_summary, stability_rows)
    metadata = {
        "subject_count": int(len(np.unique(patient_ids))),
        "spectrum_rows": int(len(X)),
        "grid_points": int(X.shape[1]),
        "n_blocks": args.n_blocks,
        "device": device,
        "epochs": args.epochs,
        "mc_iterations": args.mc_iterations,
        "seed": args.seed,
        "clinical_features": clinical.metadata,
        "clinical_input_method": "8-dimensional clinical embedding broadcast as input channels to Conv1D",
        "excluded_leakage_fields": clinical.metadata["leakage_fields_excluded"],
        "spectrum_control_source": str(base),
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    mean_metrics = [row for row in metrics if row["aggregation"] == "mean"]
    lines = [
        "# mapping clinical-input CNN evaluation",
        "",
        f"- Cohort: {len(np.unique(patient_ids))} patients, {len(X)} preprocessed spectra, grid {X.shape[1]} points.",
        f"- Clinical input: {', '.join(clinical.feature_names)}.",
        "- Excluded leakage/target-derived fields: cohort_group, cancer_type, gleason_score, grade_group, overall_stage.",
        "- site_code, sex, and collection_date were not used as primary features because site and sex are constant and date can encode acquisition batch.",
        "- CNN clinical fusion: 5 clinical features → 8-dimensional MLP embedding → broadcast input channels concatenated with the spectrum before Conv1D.",
        "",
        "## Patient mean OOF metrics",
        "",
        "| Task | Model | ROC-AUC | Macro ROC-AUC | Balanced accuracy | Sensitivity | Specificity | Macro F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in mean_metrics:
        def fmt(name: str) -> str:
            value = float(row.get(name, np.nan))
            return "-" if not math.isfinite(value) else f"{value:.3f}"
        lines.append(f"| {row['task']} | {row['model']} | {fmt('roc_auc')} | {fmt('macro_roc_auc')} | {fmt('balanced_accuracy')} | {fmt('sensitivity')} | {fmt('specificity')} | {fmt('macro_f1')} |")
    lines.extend(
        [
            "",
            "## Clinical data interpretation",
            "",
            "- PSA_raw is a strong clinical discriminator in this cohort and is not a Raman-only signal. Any gain from clinical fusion must therefore be reported as multimodal performance.",
            "- Gleason/grade/cancer_type are populated only for prostate cancer and were excluded from primary modeling because they reveal the target or post-diagnosis state.",
            "- Because site_code and sex are constant, they cannot explain within-cohort prediction differences and were not passed to the network.",
            "",
            "## Caution",
            "",
            "- Spectrum-only control predictions were taken from the locked 3-block patient-level OOF run in the base output directory; clinical-only and clinical-input CNN were fit on the identical outer folds.",
            "- Clinical fusion performance is not evidence that the CNN learned Raman peaks. It quantifies the combined value of Raman plus clinical variables.",
            "- Cancer Screening remains susceptible to cohort and measurement confounding.",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "subjects": int(len(np.unique(patient_ids))), "rows": int(len(X)), "device": device}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
