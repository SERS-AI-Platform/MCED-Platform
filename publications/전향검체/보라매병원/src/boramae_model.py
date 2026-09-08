from __future__ import annotations

import csv

import numpy as np
from boramae_data import LABELS, TABLE_DIR, SubjectSpectrum
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_three_group(
    subjects: list[SubjectSpectrum],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    x = np.vstack([sub.mean_spectrum for sub in subjects])
    y = [sub.sample.group for sub in subjects]
    idx = np.arange(len(subjects))
    train_idx, test_idx = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")
    )
    model.fit(x[train_idx], [y[i] for i in train_idx])
    pred_train = model.predict(x[train_idx])
    pred_test = model.predict(x[test_idx])
    train_y = [y[i] for i in train_idx]
    test_y = [y[i] for i in test_idx]
    metrics = {
        "train_accuracy": accuracy_score(train_y, pred_train),
        "train_balanced_accuracy": balanced_accuracy_score(train_y, pred_train),
        "train_macro_f1": f1_score(train_y, pred_train, average="macro"),
        "test_accuracy": accuracy_score(test_y, pred_test),
        "test_balanced_accuracy": balanced_accuracy_score(test_y, pred_test),
        "test_macro_f1": f1_score(test_y, pred_test, average="macro"),
    }
    cm_train = confusion_matrix(train_y, pred_train, labels=LABELS)
    cm_test = confusion_matrix(test_y, pred_test, labels=LABELS)
    write_split_table(subjects, train_idx, test_idx)
    write_classification_report("training", train_y, pred_train)
    write_classification_report("test", test_y, pred_test)
    write_confusion_matrix_table(cm_train, cm_test)
    write_metrics_table(metrics)
    return cm_train, cm_test, metrics


def write_split_table(
    subjects: list[SubjectSpectrum], train_idx: np.ndarray, test_idx: np.ndarray
) -> None:
    split = {int(i): "train" for i in train_idx} | {int(i): "test" for i in test_idx}
    with (TABLE_DIR / "boramae_train_test_split.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(["split", "case_id", "clinical_group", "grade_group"])
        for i, subject in enumerate(subjects):
            writer.writerow(
                [
                    split[i],
                    f"BRM-{i + 1:03d}",
                    subject.sample.group,
                    subject.sample.grade_group or "",
                ]
            )


def write_classification_report(name: str, y_true: list[str], y_pred: np.ndarray) -> None:
    rows = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    with (TABLE_DIR / f"fig03_{name}_classification_report.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(["label", "precision", "recall", "f1_score", "support"])
        for label, values in rows.items():
            if isinstance(values, dict):
                writer.writerow(
                    [
                        label,
                        values.get("precision", ""),
                        values.get("recall", ""),
                        values.get("f1-score", ""),
                        values.get("support", ""),
                    ]
                )


def write_confusion_matrix_table(cm_train: np.ndarray, cm_test: np.ndarray) -> None:
    with (TABLE_DIR / "fig03_confusion_matrices.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(["split", "true_label", *[f"pred_{label}" for label in LABELS]])
        for split, matrix in [("training", cm_train), ("test", cm_test)]:
            for label, row in zip(LABELS, matrix, strict=True):
                writer.writerow([split, label, *row.tolist()])


def write_metrics_table(metrics: dict[str, float]) -> None:
    with (TABLE_DIR / "fig03_metrics_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])
