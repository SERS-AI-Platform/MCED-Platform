from __future__ import annotations

import numpy as np
from mapping_resnet_data import CLASS_NAMES
from mapping_resnet_model import _auc
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def aggregate_probabilities(
    probabilities: np.ndarray,
    patient_ids: np.ndarray,
    true_labels: np.ndarray,
    mode: str,
    selected_indices: dict[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_patients = np.array(sorted(np.unique(patient_ids)), dtype=int)
    out_prob: list[np.ndarray] = []
    out_true: list[int] = []
    for patient in unique_patients:
        row_indices = np.flatnonzero(patient_ids == patient)
        if selected_indices is not None:
            row_indices = selected_indices[int(patient)]
        values = probabilities[row_indices]
        if mode == "mean":
            summary = values.mean(axis=0)
        elif mode == "median":
            summary = np.median(values, axis=0)
        elif mode == "trimmed_mean":
            trim = int(len(values) * 0.10)
            if trim and len(values) > 2 * trim:
                ordered = np.sort(values, axis=0)
                summary = ordered[trim : len(values) - trim].mean(axis=0)
            else:
                summary = values.mean(axis=0)
        elif mode == "majority":
            votes = np.argmax(values, axis=1)
            summary = np.bincount(votes, minlength=values.shape[1]).astype(float) / len(votes)
        else:
            raise ValueError(f"unknown aggregation mode: {mode}")
        total = float(np.sum(summary))
        if total > 0:
            summary = summary / total
        out_prob.append(summary)
        out_true.append(int(true_labels[row_indices[0]]))
    return unique_patients, np.asarray(out_true, dtype=int), np.vstack(out_prob)


def metric_row(task: str, model_name: str, aggregation: str, y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    prediction = np.argmax(probabilities, axis=1)
    row: dict[str, object] = {
        "task": task,
        "model": model_name,
        "aggregation": aggregation,
        "n_patients": int(len(y_true)),
        "roc_auc": float("nan"),
        "macro_roc_auc": float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "sensitivity": float("nan"),
        "specificity": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "f1": float("nan"),
        "macro_f1": float("nan"),
        "npv": float("nan"),
        "ppv": float("nan"),
        "brier_score": float("nan"),
    }
    if probabilities.shape[1] == 2:
        row["roc_auc"] = _auc(y_true, probabilities)
        row["sensitivity"] = float(recall_score(y_true, prediction, zero_division=0))
        row["specificity"] = float(recall_score(1 - y_true, 1 - prediction, zero_division=0))
        row["precision"] = float(precision_score(y_true, prediction, zero_division=0))
        row["recall"] = row["sensitivity"]
        row["f1"] = float(f1_score(y_true, prediction, zero_division=0))
        row["ppv"] = row["precision"]
        tn, fp, fn, _tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
        row["npv"] = float(tn / max(tn + fn, 1))
        row["brier_score"] = float(brier_score_loss(y_true, probabilities[:, 1]))
    else:
        row["macro_roc_auc"] = _auc(y_true, probabilities)
        row["macro_f1"] = float(f1_score(y_true, prediction, average="macro", zero_division=0))
        for class_index, class_name in enumerate(CLASS_NAMES):
            actual = (y_true == class_index).astype(int)
            predicted = (prediction == class_index).astype(int)
            row[f"sensitivity_{class_name}"] = float(recall_score(actual, predicted, zero_division=0))
            row[f"specificity_{class_name}"] = float(recall_score(1 - actual, 1 - predicted, zero_division=0))
    return row


def patient_oof_rows(
    patient_ids: np.ndarray,
    y_binary: np.ndarray,
    y_three: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> tuple[list[dict[str, object]], dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    output: list[dict[str, object]] = []
    aggregates: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for task, y_true_rows, key in (("cancer_vs_non_cancer", y_binary, "binary"), ("three_class", y_three, "three")):
        for model_name in ("LR-reference", "Multi-scale ResNet", "LR+ResNet fixed blend"):
            if model_name == "LR-reference":
                probability = predictions[f"lr_{key}"]
            elif model_name == "Multi-scale ResNet":
                probability = predictions[f"resnet_{key}"]
            else:
                probability = 0.8 * predictions[f"lr_{key}"] + 0.2 * predictions[f"resnet_{key}"]
            for aggregation in ("mean", "median", "majority", "trimmed_mean"):
                ids, y_patient, p_patient = aggregate_probabilities(probability, patient_ids, y_true_rows, aggregation)
                aggregates[(task, model_name, aggregation)] = (ids, y_patient, p_patient)
                output.append(metric_row(task, model_name, aggregation, y_patient, p_patient))
    return output, aggregates
