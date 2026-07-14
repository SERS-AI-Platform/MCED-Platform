from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, assert_never

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TaskName = Literal["three_group", "strict_binary", "screening_binary"]
BOOTSTRAP_ITERATIONS = 2000


@dataclass(frozen=True, slots=True)
class OofResult:
    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray
    folds: np.ndarray
    metrics: dict[str, float]
    intervals: dict[str, tuple[float, float]]
    confusion: np.ndarray


def build_task(
    x: np.ndarray, labels: np.ndarray, task: TaskName
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    match task:
        case "three_group":
            mapping = {"Control": 0, "Biopsy-negative": 1, "Prostate cancer": 2}
            return x, np.array([mapping[str(label)] for label in labels]), indices
        case "strict_binary":
            keep = labels != "Biopsy-negative"
            return x[keep], (labels[keep] == "Prostate cancer").astype(int), indices[keep]
        case "screening_binary":
            return x, (labels == "Prostate cancer").astype(int), indices
        case unreachable:
            assert_never(unreachable)


def metric_values(
    y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    values = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    if probabilities.shape[1] == 2:
        positive = probabilities[:, 1]
        values |= {
            "roc_auc": float(roc_auc_score(y_true, positive)),
            "pr_auc": float(average_precision_score(y_true, positive)),
            "sensitivity": float(recall_score(y_true, y_pred, pos_label=1)),
            "specificity": float(recall_score(y_true, y_pred, pos_label=0)),
        }
    else:
        values["macro_ovr_roc_auc"] = float(
            roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
        )
    return values


def bootstrap_intervals(
    y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(20260710)
    samples: dict[str, list[float]] = {}
    for _ in range(BOOTSTRAP_ITERATIONS):
        index = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[index])) != probabilities.shape[1]:
            continue
        for name, value in metric_values(
            y_true[index], y_pred[index], probabilities[index]
        ).items():
            samples.setdefault(name, []).append(value)
    return {
        name: (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
        for name, values in samples.items()
    }


def nested_oof(x: np.ndarray, y: np.ndarray) -> OofResult:
    classes = np.unique(y)
    probabilities = np.zeros((len(y), len(classes)), dtype=float)
    folds = np.zeros(len(y), dtype=int)
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train, test) in enumerate(outer.split(x, y), start=1):
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs"),
        )
        inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=100 + fold)
        search = GridSearchCV(
            estimator,
            {"logisticregression__C": [0.001, 0.01, 0.1, 1.0, 10.0]},
            scoring="balanced_accuracy",
            cv=inner,
            n_jobs=-1,
        )
        search.fit(x[train], y[train])
        probabilities[test] = search.predict_proba(x[test])
        folds[test] = fold
    y_pred = probabilities.argmax(axis=1)
    metrics = metric_values(y, y_pred, probabilities)
    return OofResult(
        y_true=y,
        y_pred=y_pred,
        probabilities=probabilities,
        folds=folds,
        metrics=metrics,
        intervals=bootstrap_intervals(y, y_pred, probabilities),
        confusion=confusion_matrix(y, y_pred, labels=classes),
    )
