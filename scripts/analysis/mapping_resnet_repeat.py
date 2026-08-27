from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
from mapping_resnet_evaluation import aggregate_probabilities, metric_row

CANDIDATE_N = (1, 3, 5, 9, 16, 25, 36, 49, 64, 81, 100, 121)


def repeat_evaluation(
    patient_ids: np.ndarray,
    y_binary: np.ndarray,
    y_three: np.ndarray,
    predictions: dict[str, np.ndarray],
    iterations: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(seed)
    patient_order = np.array(sorted(np.unique(patient_ids)), dtype=int)
    row_by_patient = {int(patient): np.flatnonzero(patient_ids == patient) for patient in patient_order}
    available = min(len(indices) for indices in row_by_patient.values())
    candidates = tuple(n for n in CANDIDATE_N if n <= available)
    repeat_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    min_rows: list[dict[str, object]] = []
    for task, y_rows, key in (("cancer_vs_non_cancer", y_binary, "binary"), ("three_class", y_three, "three")):
        models = {
            "LR-reference": predictions[f"lr_{key}"],
            "Multi-scale ResNet": predictions[f"resnet_{key}"],
            "LR+ResNet fixed blend": 0.8 * predictions[f"lr_{key}"] + 0.2 * predictions[f"resnet_{key}"],
        }
        for model_name, probability in models.items():
            baseline_indices = {int(patient): row_by_patient[int(patient)] for patient in patient_order}
            for aggregation in ("mean", "median", "majority", "trimmed_mean"):
                _, _baseline_y, baseline_prob = aggregate_probabilities(probability, patient_ids, y_rows, aggregation, baseline_indices)
                baseline_prediction = np.argmax(baseline_prob, axis=1)
                for n in candidates:
                    sampled_probabilities: list[np.ndarray] = []
                    for iteration in range(iterations):
                        selected = {
                            int(patient): rng.choice(row_by_patient[int(patient)], size=n, replace=False)
                            for patient in patient_order
                        }
                        _, y_patient, p_patient = aggregate_probabilities(probability, patient_ids, y_rows, aggregation, selected)
                        row = metric_row(task, model_name, aggregation, y_patient, p_patient)
                        row["n"] = n
                        row["iteration"] = iteration + 1
                        row["prediction_agreement"] = float(np.mean(np.argmax(p_patient, axis=1) == baseline_prediction))
                        repeat_rows.append(row)
                        sampled_probabilities.append(p_patient)
                    stack = np.stack(sampled_probabilities, axis=0)
                    stability_rows.append(
                        {
                            "task": task,
                            "model": model_name,
                            "aggregation": aggregation,
                            "n": n,
                            "probability_sd_mean": float(np.mean(np.std(stack, axis=0, ddof=1))) if iterations > 1 else float("nan"),
                            "probability_sd_median": float(np.median(np.std(stack, axis=0, ddof=1))) if iterations > 1 else float("nan"),
                        }
                    )
    for task, _y_rows, _key in (("cancer_vs_non_cancer", y_binary, "binary"), ("three_class", y_three, "three")):
        for model_name in ("LR-reference", "Multi-scale ResNet", "LR+ResNet fixed blend"):
            for aggregation in ("mean", "median", "majority", "trimmed_mean"):
                min_rows.extend(minimum_repeat_rows(task, model_name, aggregation, repeat_rows))
    min_rows.append({"minimum_available_repeats": available, "evaluated_candidates": ",".join(map(str, candidates))})
    return repeat_rows, stability_rows, min_rows


def minimum_repeat_rows(task: str, model_name: str, aggregation: str, rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    selected = [row for row in rows if row.get("task") == task and row.get("model") == model_name and row.get("aggregation") == aggregation]
    if not selected:
        return []
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in selected:
        grouped[int(row["n"])].append(row)
    ns = sorted(grouped)
    reference_n = max(ns)

    def mean_metric(name: str, n: int) -> float:
        values = np.asarray([float(row.get(name, np.nan)) for row in grouped[n]], dtype=float)
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if len(finite) else float("nan")

    ref_auc_name = "roc_auc" if task == "cancer_vs_non_cancer" else "macro_roc_auc"
    ref_auc = mean_metric(ref_auc_name, reference_n)
    ref_ba = mean_metric("balanced_accuracy", reference_n)
    ref_sens = mean_metric("sensitivity", reference_n) if task == "cancer_vs_non_cancer" else float("nan")
    candidates: list[dict[str, object]] = []
    for index, n in enumerate(ns):
        auc = mean_metric(ref_auc_name, n)
        ba = mean_metric("balanced_accuracy", n)
        sens = mean_metric("sensitivity", n) if task == "cancer_vs_non_cancer" else float("nan")
        agreement = mean_metric("prediction_agreement", n)
        next_auc = mean_metric(ref_auc_name, ns[index + 1]) if index + 1 < len(ns) else float("nan")
        gain = next_auc - auc if np.isfinite(next_auc) else 0.0
        meets = auc >= ref_auc - 0.01 and ba >= ref_ba - 0.02 and agreement >= 0.95
        if task == "cancer_vs_non_cancer":
            meets = meets and sens >= ref_sens - 0.02
        candidates.append(
            {
                "task": task,
                "model": model_name,
                "aggregation": aggregation,
                "n": n,
                "reference_n": reference_n,
                "reference_auc": ref_auc,
                "reference_balanced_accuracy": ref_ba,
                "reference_sensitivity": ref_sens,
                "auc": auc,
                "balanced_accuracy": ba,
                "sensitivity": sens,
                "prediction_agreement": agreement,
                "auc_gain_to_next_n": gain,
                "plateau_flag_gain_le_0.005": bool(abs(gain) <= 0.005),
                "criteria_met": bool(meets),
            }
        )
    selected_n = next((int(row["n"]) for row in candidates if row["criteria_met"] and row["plateau_flag_gain_le_0.005"]), None)
    for row in candidates:
        row["minimum_n"] = selected_n if selected_n is not None else "not_met"
    return candidates


def summarize_repeat_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task"]), str(row["model"]), str(row["aggregation"]), int(row["n"]))].append(row)
    output: list[dict[str, object]] = []
    for (task, model, aggregation, n), values in sorted(grouped.items()):
        result: dict[str, object] = {"task": task, "model": model, "aggregation": aggregation, "n": n, "iterations": len(values)}
        for metric in ("roc_auc", "macro_roc_auc", "balanced_accuracy", "sensitivity", "specificity", "macro_f1", "prediction_agreement"):
            numbers = np.asarray([float(value.get(metric, np.nan)) for value in values], dtype=float)
            finite = numbers[np.isfinite(numbers)]
            result[f"{metric}_mean"] = float(finite.mean()) if len(finite) else float("nan")
            result[f"{metric}_sd"] = float(finite.std(ddof=1)) if len(finite) > 1 else float("nan")
        output.append(result)
    return output
