from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from typing_extensions import assert_never

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .models import TaskName

FeatureSetName = Literal[
    "full_spectrum",
    "common_stable",
    "powder_only",
    "powder_only_excluded",
]


@dataclass(frozen=True, slots=True)
class FeatureMasks:
    full: np.ndarray
    common_stable: np.ndarray
    powder_only: np.ndarray
    powder_only_excluded: np.ndarray


@dataclass(frozen=True, slots=True)
class AblationMetrics:
    roc_auc: float
    balanced_accuracy: float
    sensitivity: float
    specificity: float


@dataclass(frozen=True, slots=True)
class AblationResult:
    task: TaskName
    feature_set: FeatureSetName
    class_names: tuple[str, ...]
    y_true: np.ndarray
    probabilities: np.ndarray
    folds: np.ndarray
    feature_counts: np.ndarray
    confusion: np.ndarray
    metrics: AblationMetrics


def build_feature_masks(
    legacy_presence: np.ndarray,
    powder_presence: np.ndarray,
    *,
    minimum_subject_rate: float = 0.30,
) -> FeatureMasks:
    """Build label-free peak-region masks from training-subject consensus rates."""
    legacy_stable = np.mean(legacy_presence, axis=0) >= minimum_subject_rate
    powder_stable = np.mean(powder_presence, axis=0) >= minimum_subject_rate
    common = legacy_stable & powder_stable
    powder_only = powder_stable & ~legacy_stable
    return FeatureMasks(
        full=np.ones(legacy_presence.shape[1], dtype=bool),
        common_stable=common,
        powder_only=powder_only,
        powder_only_excluded=~powder_only,
    )


def _encoded(labels: np.ndarray, task: TaskName) -> tuple[np.ndarray, tuple[str, ...]]:
    match task:
        case "screening_binary":
            return (labels == "Cancer").astype(int), ("Non-cancer", "Cancer")
        case "three_group":
            names = ("Control", "Biopsy-negative", "Cancer")
            mapping = {name: index for index, name in enumerate(names)}
            return np.array([mapping[str(label)] for label in labels], dtype=int), names
        case unreachable:
            assert_never(unreachable)


def _selected_mask(masks: FeatureMasks, name: FeatureSetName) -> np.ndarray:
    match name:
        case "full_spectrum":
            return masks.full
        case "common_stable":
            return masks.common_stable
        case "powder_only":
            return masks.powder_only
        case "powder_only_excluded":
            return masks.powder_only_excluded
        case unreachable:
            assert_never(unreachable)


def _predict_fold(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    *,
    class_count: int,
    inner_splits: int,
    seed: int,
    c_values: tuple[float, ...],
) -> np.ndarray:
    if train_features.shape[1] == 0:
        counts = np.bincount(train_labels, minlength=class_count).astype(float)
        return np.tile(counts / np.sum(counts), (len(test_features), 1))
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs"),
    )
    search = GridSearchCV(
        estimator,
        {"logisticregression__C": c_values},
        scoring="balanced_accuracy",
        cv=StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed),
        n_jobs=-1,
    )
    search.fit(train_features, train_labels)
    return search.predict_proba(test_features)


def _metrics(labels: np.ndarray, probabilities: np.ndarray, task: TaskName) -> AblationMetrics:
    predictions = probabilities.argmax(axis=1)
    balanced = float(balanced_accuracy_score(labels, predictions))
    match task:
        case "screening_binary":
            matrix = confusion_matrix(labels, predictions, labels=(0, 1))
            true_negative, false_positive, false_negative, true_positive = matrix.ravel()
            return AblationMetrics(
                roc_auc=float(roc_auc_score(labels, probabilities[:, 1])),
                balanced_accuracy=balanced,
                sensitivity=true_positive / (true_positive + false_negative),
                specificity=true_negative / (true_negative + false_positive),
            )
        case "three_group":
            return AblationMetrics(
                roc_auc=float(
                    roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
                ),
                balanced_accuracy=balanced,
                sensitivity=float("nan"),
                specificity=float("nan"),
            )
        case unreachable:
            assert_never(unreachable)


def run_peak_ablation(
    powder_spectra: np.ndarray,
    labels: np.ndarray,
    legacy_presence: np.ndarray,
    powder_presence: np.ndarray,
    *,
    task: TaskName,
    outer_splits: int = 5,
    inner_splits: int = 4,
    seed: int = 42,
    c_values: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0),
) -> tuple[AblationResult, ...]:
    """Evaluate powder-native LR feature ablations without outcome or mask leakage."""
    encoded, class_names = _encoded(labels, task)
    feature_sets: tuple[FeatureSetName, ...] = (
        "full_spectrum",
        "common_stable",
        "powder_only",
        "powder_only_excluded",
    )
    probabilities = {
        name: np.zeros((len(labels), len(class_names)), dtype=float) for name in feature_sets
    }
    feature_counts = {name: np.zeros(outer_splits, dtype=int) for name in feature_sets}
    folds = np.zeros(len(labels), dtype=int)
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(outer.split(powder_spectra, encoded), start=1):
        masks = build_feature_masks(legacy_presence[train], powder_presence[train])
        folds[test] = fold
        for name in feature_sets:
            mask = _selected_mask(masks, name)
            feature_counts[name][fold - 1] = int(np.sum(mask))
            probabilities[name][test] = _predict_fold(
                powder_spectra[train][:, mask],
                encoded[train],
                powder_spectra[test][:, mask],
                class_count=len(class_names),
                inner_splits=inner_splits,
                seed=200 + fold,
                c_values=c_values,
            )
    return tuple(
        AblationResult(
            task=task,
            feature_set=name,
            class_names=class_names,
            y_true=encoded,
            probabilities=probabilities[name],
            folds=folds.copy(),
            feature_counts=feature_counts[name],
            confusion=confusion_matrix(
                encoded,
                probabilities[name].argmax(axis=1),
                labels=np.arange(len(class_names)),
            ),
            metrics=_metrics(encoded, probabilities[name], task),
        )
        for name in feature_sets
    )
