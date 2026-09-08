from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from typing_extensions import assert_never

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)

MetricTask = Literal["screening_binary", "three_group"]


@dataclass(frozen=True, slots=True)
class ReclassificationSummary:
    stable_correct: int
    improved: int
    worsened: int
    stable_incorrect: int
    net_improvement_rate: float


@dataclass(frozen=True, slots=True)
class AgreementSummary:
    exact_agreement: float
    kappa: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True, slots=True)
class MetricValue:
    name: str
    value: float


@dataclass(frozen=True, slots=True)
class MetricComparison:
    name: str
    legacy_value: float
    powder_value: float
    delta: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True, slots=True)
class McNemarResult:
    legacy_only: int
    powder_only: int
    p_value: float


def reclassification_summary(
    truth: np.ndarray, legacy_prediction: np.ndarray, powder_prediction: np.ndarray
) -> ReclassificationSummary:
    legacy_correct = legacy_prediction == truth
    powder_correct = powder_prediction == truth
    improved = int(np.sum(~legacy_correct & powder_correct))
    worsened = int(np.sum(legacy_correct & ~powder_correct))
    return ReclassificationSummary(
        stable_correct=int(np.sum(legacy_correct & powder_correct)),
        improved=improved,
        worsened=worsened,
        stable_incorrect=int(np.sum(~legacy_correct & ~powder_correct)),
        net_improvement_rate=float((improved - worsened) / len(truth)),
    )


def _kappa(first: np.ndarray, second: np.ndarray) -> float:
    labels = np.unique(np.concatenate((first, second)))
    observed = float(np.mean(first == second))
    expected = 0.0
    for label in labels:
        expected += float(np.mean(first == label) * np.mean(second == label))
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def agreement_summary(
    legacy_prediction: np.ndarray,
    powder_prediction: np.ndarray,
    *,
    seed: int,
    n_resamples: int,
) -> AgreementSummary:
    rng = np.random.default_rng(seed)
    bootstrap: list[float] = []
    for _ in range(n_resamples):
        indices = rng.integers(0, len(legacy_prediction), len(legacy_prediction))
        value = _kappa(legacy_prediction[indices], powder_prediction[indices])
        if np.isfinite(value):
            bootstrap.append(value)
    return AgreementSummary(
        exact_agreement=float(np.mean(legacy_prediction == powder_prediction)),
        kappa=_kappa(legacy_prediction, powder_prediction),
        ci_low=float(np.percentile(bootstrap, 2.5)),
        ci_high=float(np.percentile(bootstrap, 97.5)),
    )


def metric_values(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    task: MetricTask,
    predictions: np.ndarray | None = None,
) -> tuple[MetricValue, ...]:
    predicted = probabilities.argmax(axis=1) if predictions is None else predictions
    base = [
        MetricValue("accuracy", float(accuracy_score(y_true, predicted))),
        MetricValue("balanced_accuracy", float(balanced_accuracy_score(y_true, predicted))),
        MetricValue("macro_f1", float(f1_score(y_true, predicted, average="macro"))),
    ]
    match task:
        case "screening_binary":
            positive = probabilities[:, 1]
            base.extend(
                (
                    MetricValue("roc_auc", float(roc_auc_score(y_true, positive))),
                    MetricValue("sensitivity", float(recall_score(y_true, predicted, pos_label=1))),
                    MetricValue("specificity", float(recall_score(y_true, predicted, pos_label=0))),
                    MetricValue("brier_score", float(brier_score_loss(y_true, positive))),
                )
            )
        case "three_group":
            base.extend(
                (
                    MetricValue(
                        "macro_ovr_roc_auc",
                        float(
                            roc_auc_score(
                                y_true,
                                probabilities,
                                multi_class="ovr",
                                average="macro",
                            )
                        ),
                    ),
                    MetricValue("log_loss", float(log_loss(y_true, probabilities))),
                )
            )
            for class_index, class_name in enumerate(("control", "biopsy_negative", "cancer")):
                class_truth = (y_true == class_index).astype(int)
                class_prediction = (predicted == class_index).astype(int)
                base.extend(
                    (
                        MetricValue(
                            f"{class_name}_ovr_roc_auc",
                            float(roc_auc_score(class_truth, probabilities[:, class_index])),
                        ),
                        MetricValue(
                            f"{class_name}_sensitivity",
                            float(recall_score(class_truth, class_prediction, pos_label=1)),
                        ),
                        MetricValue(
                            f"{class_name}_specificity",
                            float(recall_score(class_truth, class_prediction, pos_label=0)),
                        ),
                    )
                )
        case unreachable:
            assert_never(unreachable)
    return tuple(base)


def _stratified_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sampled = [
        rng.choice(np.flatnonzero(y_true == label), size=int(np.sum(y_true == label)), replace=True)
        for label in np.unique(y_true)
    ]
    return np.concatenate(sampled)


def paired_metric_comparisons(
    y_true: np.ndarray,
    legacy_probabilities: np.ndarray,
    powder_probabilities: np.ndarray,
    task: MetricTask,
    *,
    legacy_predictions: np.ndarray | None = None,
    powder_predictions: np.ndarray | None = None,
    seed: int,
    n_resamples: int,
) -> tuple[MetricComparison, ...]:
    legacy = metric_values(y_true, legacy_probabilities, task, legacy_predictions)
    powder = metric_values(y_true, powder_probabilities, task, powder_predictions)
    names = tuple(item.name for item in legacy)
    bootstrap = {name: [] for name in names}
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        indices = _stratified_indices(y_true, rng)
        legacy_sample = metric_values(
            y_true[indices],
            legacy_probabilities[indices],
            task,
            None if legacy_predictions is None else legacy_predictions[indices],
        )
        powder_sample = metric_values(
            y_true[indices],
            powder_probabilities[indices],
            task,
            None if powder_predictions is None else powder_predictions[indices],
        )
        for old, new in zip(legacy_sample, powder_sample, strict=True):
            bootstrap[old.name].append(new.value - old.value)
    rows: list[MetricComparison] = []
    for old, new in zip(legacy, powder, strict=True):
        differences = bootstrap[old.name]
        rows.append(
            MetricComparison(
                name=old.name,
                legacy_value=old.value,
                powder_value=new.value,
                delta=new.value - old.value,
                ci_low=float(np.percentile(differences, 2.5)),
                ci_high=float(np.percentile(differences, 97.5)),
            )
        )
    return tuple(rows)


def exact_mcnemar(legacy_outcome: np.ndarray, powder_outcome: np.ndarray) -> McNemarResult:
    legacy_only = int(np.sum(legacy_outcome & ~powder_outcome))
    powder_only = int(np.sum(~legacy_outcome & powder_outcome))
    discordant = legacy_only + powder_only
    p_value = (
        1.0
        if discordant == 0
        else float(binomtest(legacy_only, discordant, p=0.5, alternative="two-sided").pvalue)
    )
    return McNemarResult(legacy_only=legacy_only, powder_only=powder_only, p_value=p_value)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, float((len(p_values) - rank) * p_values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted
