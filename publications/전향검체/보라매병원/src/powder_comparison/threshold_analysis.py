from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from powder_comparison.runner import AnalysisResult


@dataclass(frozen=True, slots=True)
class ThresholdStrategy:
    name: str
    modality: str
    method: str
    source: str
    probabilities: np.ndarray
    thresholds: np.ndarray
    predictions: np.ndarray


def threshold_strategies(result: AnalysisResult) -> tuple[ThresholdStrategy, ...]:
    screening = result.screening
    optimized = screening.binary_thresholds
    assert optimized is not None
    fixed = np.full(len(screening.y_true), 0.5)
    return (
        ThresholdStrategy(
            "legacy_fixed_0.5",
            "legacy_liquid",
            "fixed_0.5",
            "predefined",
            screening.legacy_probabilities,
            fixed,
            screening.legacy_probabilities.argmax(axis=1),
        ),
        ThresholdStrategy(
            "legacy_crossfit_youden",
            "legacy_liquid",
            "crossfit_youden_j",
            "liquid_outer_train",
            screening.legacy_probabilities,
            optimized.legacy_thresholds,
            optimized.legacy_predictions,
        ),
        ThresholdStrategy(
            "powder_transfer_fixed_0.5",
            "powder_transfer",
            "fixed_0.5",
            "predefined",
            screening.transfer_probabilities,
            fixed,
            screening.transfer_probabilities.argmax(axis=1),
        ),
        ThresholdStrategy(
            "powder_transfer_crossfit_youden",
            "powder_transfer",
            "crossfit_youden_j",
            "liquid_outer_train",
            screening.transfer_probabilities,
            optimized.legacy_thresholds,
            optimized.transfer_predictions,
        ),
        ThresholdStrategy(
            "powder_native_fixed_0.5",
            "powder_native",
            "fixed_0.5",
            "predefined",
            screening.native_probabilities,
            fixed,
            screening.native_probabilities.argmax(axis=1),
        ),
        ThresholdStrategy(
            "powder_native_crossfit_youden",
            "powder_native",
            "crossfit_youden_j",
            "powder_outer_train",
            screening.native_probabilities,
            optimized.native_thresholds,
            optimized.native_predictions,
        ),
    )


def classification_status(truth: int, fixed: int, optimized: int) -> str:
    fixed_correct = fixed == truth
    optimized_correct = optimized == truth
    if fixed_correct and optimized_correct:
        return "stable_correct"
    if not fixed_correct and optimized_correct:
        return "improved"
    if fixed_correct and not optimized_correct:
        return "worsened"
    return "stable_incorrect"
