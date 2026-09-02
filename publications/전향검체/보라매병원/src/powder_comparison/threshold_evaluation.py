from __future__ import annotations

from powder_comparison.evaluation import ComparisonBlock
from powder_comparison.models import PairedCrossfitResult
from powder_comparison.statistics import (
    agreement_summary,
    paired_metric_comparisons,
    reclassification_summary,
)


def compare_binary_thresholds(
    result: PairedCrossfitResult, *, seed: int, n_resamples: int
) -> tuple[ComparisonBlock, ...]:
    optimized = result.binary_thresholds
    assert optimized is not None
    specifications = (
        (
            "screening_binary_liquid_threshold_optimized_vs_0.5",
            result.legacy_probabilities,
            result.legacy_probabilities.argmax(axis=1),
            optimized.legacy_predictions,
        ),
        (
            "screening_binary_transfer_threshold_optimized_vs_0.5",
            result.transfer_probabilities,
            result.transfer_probabilities.argmax(axis=1),
            optimized.transfer_predictions,
        ),
        (
            "screening_binary_native_threshold_optimized_vs_0.5",
            result.native_probabilities,
            result.native_probabilities.argmax(axis=1),
            optimized.native_predictions,
        ),
    )
    rows: list[ComparisonBlock] = []
    for index, (name, probabilities, fixed, tuned) in enumerate(specifications):
        rows.append(
            ComparisonBlock(
                name=name,
                metrics=paired_metric_comparisons(
                    result.y_true,
                    probabilities,
                    probabilities,
                    result.task,
                    legacy_predictions=fixed,
                    powder_predictions=tuned,
                    seed=seed + index * 3,
                    n_resamples=n_resamples,
                ),
                reclassification=reclassification_summary(result.y_true, fixed, tuned),
                agreement=agreement_summary(
                    fixed,
                    tuned,
                    seed=seed + index * 3 + 1,
                    n_resamples=n_resamples,
                ),
            )
        )
    return tuple(rows)
