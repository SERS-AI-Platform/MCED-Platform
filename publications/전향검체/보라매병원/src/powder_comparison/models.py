from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, assert_never

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TaskName = Literal["screening_binary", "three_group"]


@dataclass(frozen=True, slots=True)
class BinaryThresholdCrossfit:
    objective: Literal["youden_j"]
    legacy_thresholds: np.ndarray
    native_thresholds: np.ndarray
    legacy_predictions: np.ndarray
    transfer_predictions: np.ndarray
    native_predictions: np.ndarray


@dataclass(frozen=True, slots=True)
class PairedCrossfitResult:
    task: TaskName
    class_names: tuple[str, ...]
    y_true: np.ndarray
    folds: np.ndarray
    legacy_probabilities: np.ndarray
    transfer_probabilities: np.ndarray
    native_probabilities: np.ndarray
    binary_thresholds: BinaryThresholdCrossfit | None


def _encode_labels(labels: np.ndarray, task: TaskName) -> tuple[np.ndarray, tuple[str, ...]]:
    match task:
        case "screening_binary":
            return (labels == "Cancer").astype(int), ("Non-cancer", "Cancer")
        case "three_group":
            names = ("Control", "Biopsy-negative", "Cancer")
            mapping = {name: index for index, name in enumerate(names)}
            return np.array([mapping[str(label)] for label in labels], dtype=int), names
        case unreachable:
            assert_never(unreachable)


def _search_model(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    inner_splits: int,
    seed: int,
    c_values: tuple[float, ...],
) -> GridSearchCV:
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs"),
    )
    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    search = GridSearchCV(
        estimator,
        {"logisticregression__C": c_values},
        scoring="balanced_accuracy",
        cv=inner,
        n_jobs=-1,
    )
    return search.fit(features, labels)


def _youden_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, probabilities)
    finite = np.isfinite(thresholds)
    candidates = thresholds[finite]
    scores = (true_positive_rate - false_positive_rate)[finite]
    best = candidates[np.isclose(scores, np.max(scores))]
    return float(best[np.argmin(np.abs(best - 0.5))])


def _cross_validated_threshold(
    search: GridSearchCV,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    inner_splits: int,
    seed: int,
) -> float:
    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    probabilities = cross_val_predict(
        clone(search.best_estimator_),
        features,
        labels,
        cv=inner,
        method="predict_proba",
        n_jobs=-1,
    )
    return _youden_threshold(labels, np.asarray(probabilities)[:, 1])


def run_paired_crossfit(
    legacy: np.ndarray,
    powder: np.ndarray,
    labels: np.ndarray,
    *,
    task: TaskName,
    outer_splits: int = 5,
    inner_splits: int = 4,
    seed: int = 42,
    c_values: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0),
) -> PairedCrossfitResult:
    encoded, class_names = _encode_labels(labels, task)
    class_count = len(class_names)
    legacy_probabilities = np.zeros((len(labels), class_count), dtype=float)
    transfer_probabilities = np.zeros_like(legacy_probabilities)
    native_probabilities = np.zeros_like(legacy_probabilities)
    folds = np.zeros(len(labels), dtype=int)
    legacy_thresholds = np.zeros(len(labels), dtype=float)
    native_thresholds = np.zeros(len(labels), dtype=float)
    legacy_predictions = np.zeros(len(labels), dtype=int)
    transfer_predictions = np.zeros(len(labels), dtype=int)
    native_predictions = np.zeros(len(labels), dtype=int)
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(outer.split(legacy, encoded), start=1):
        legacy_model = _search_model(
            legacy[train],
            encoded[train],
            inner_splits=inner_splits,
            seed=100 + fold,
            c_values=c_values,
        )
        powder_model = _search_model(
            powder[train],
            encoded[train],
            inner_splits=inner_splits,
            seed=200 + fold,
            c_values=c_values,
        )
        legacy_probabilities[test] = legacy_model.predict_proba(legacy[test])
        transfer_probabilities[test] = legacy_model.predict_proba(powder[test])
        native_probabilities[test] = powder_model.predict_proba(powder[test])
        folds[test] = fold
        match task:
            case "screening_binary":
                legacy_threshold = _cross_validated_threshold(
                    legacy_model,
                    legacy[train],
                    encoded[train],
                    inner_splits=inner_splits,
                    seed=300 + fold,
                )
                native_threshold = _cross_validated_threshold(
                    powder_model,
                    powder[train],
                    encoded[train],
                    inner_splits=inner_splits,
                    seed=400 + fold,
                )
                legacy_thresholds[test] = legacy_threshold
                native_thresholds[test] = native_threshold
                legacy_predictions[test] = (
                    legacy_probabilities[test, 1] >= legacy_threshold
                ).astype(int)
                transfer_predictions[test] = (
                    transfer_probabilities[test, 1] >= legacy_threshold
                ).astype(int)
                native_predictions[test] = (
                    native_probabilities[test, 1] >= native_threshold
                ).astype(int)
            case "three_group":
                pass
            case unreachable:
                assert_never(unreachable)
    match task:
        case "screening_binary":
            binary_thresholds = BinaryThresholdCrossfit(
                objective="youden_j",
                legacy_thresholds=legacy_thresholds,
                native_thresholds=native_thresholds,
                legacy_predictions=legacy_predictions,
                transfer_predictions=transfer_predictions,
                native_predictions=native_predictions,
            )
        case "three_group":
            binary_thresholds = None
        case unreachable:
            assert_never(unreachable)
    return PairedCrossfitResult(
        task=task,
        class_names=class_names,
        y_true=encoded,
        folds=folds,
        legacy_probabilities=legacy_probabilities,
        transfer_probabilities=transfer_probabilities,
        native_probabilities=native_probabilities,
        binary_thresholds=binary_thresholds,
    )
