from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from powder_comparison.data import PairedSpectra
from powder_comparison.lr_attribution import LrPeakAttributions, PeakContributionBatch
from powder_comparison.models import PairedCrossfitResult
from powder_comparison.order_effect import OrderEffect, stratified_order_effect
from powder_comparison.peaks import PeakAgreement, peak_agreement
from powder_comparison.statistics import (
    AgreementSummary,
    McNemarResult,
    MetricComparison,
    ReclassificationSummary,
    agreement_summary,
    exact_mcnemar,
    paired_metric_comparisons,
    reclassification_summary,
)


@dataclass(frozen=True, slots=True)
class ComparisonBlock:
    name: str
    metrics: tuple[MetricComparison, ...]
    reclassification: ReclassificationSummary
    agreement: AgreementSummary


@dataclass(frozen=True, slots=True)
class McNemarRow:
    scope: str
    result: McNemarResult


@dataclass(frozen=True, slots=True)
class OrderEffectRow:
    scope: str
    endpoint: str
    result: OrderEffect


@dataclass(frozen=True, slots=True)
class TertileRow:
    period: str
    n: int
    probability_mean: float
    error_rate: float
    pc1_mean: float
    raw_intensity_mean: float
    replicate_correlation_mean: float


@dataclass(frozen=True, slots=True)
class PeakBlock:
    agreement: PeakAgreement
    legacy: PeakContributionBatch
    powder: PeakContributionBatch
    transfer: PeakContributionBatch


def compare_crossfit(
    result: PairedCrossfitResult, *, seed: int, n_resamples: int
) -> tuple[ComparisonBlock, ComparisonBlock]:
    legacy_prediction = result.legacy_probabilities.argmax(axis=1)
    transfer_prediction = result.transfer_probabilities.argmax(axis=1)
    native_prediction = result.native_probabilities.argmax(axis=1)
    transfer = ComparisonBlock(
        name=f"{result.task}_liquid_to_powder",
        metrics=paired_metric_comparisons(
            result.y_true,
            result.legacy_probabilities,
            result.transfer_probabilities,
            result.task,
            seed=seed,
            n_resamples=n_resamples,
        ),
        reclassification=reclassification_summary(
            result.y_true, legacy_prediction, transfer_prediction
        ),
        agreement=agreement_summary(
            legacy_prediction,
            transfer_prediction,
            seed=seed + 1,
            n_resamples=n_resamples,
        ),
    )
    native = ComparisonBlock(
        name=f"{result.task}_powder_native_vs_transfer",
        metrics=paired_metric_comparisons(
            result.y_true,
            result.transfer_probabilities,
            result.native_probabilities,
            result.task,
            seed=seed + 2,
            n_resamples=n_resamples,
        ),
        reclassification=reclassification_summary(
            result.y_true, transfer_prediction, native_prediction
        ),
        agreement=agreement_summary(
            transfer_prediction,
            native_prediction,
            seed=seed + 3,
            n_resamples=n_resamples,
        ),
    )
    return transfer, native


def crossfit_mcnemar_rows(result: PairedCrossfitResult) -> tuple[McNemarRow, ...]:
    legacy_prediction = result.legacy_probabilities.argmax(axis=1)
    powder_prediction = result.transfer_probabilities.argmax(axis=1)
    rows: list[McNemarRow] = []
    for scope, selected in (
        ("sensitivity", result.y_true == 1),
        ("specificity", result.y_true == 0),
    ):
        rows.append(
            McNemarRow(
                scope=scope,
                result=exact_mcnemar(
                    legacy_prediction[selected] == result.y_true[selected],
                    powder_prediction[selected] == result.y_true[selected],
                ),
            )
        )
    return tuple(rows)


def order_effect_rows(
    spectra: PairedSpectra,
    screening: PairedCrossfitResult,
    *,
    seed: int,
    n_permutations: int,
) -> tuple[tuple[OrderEffectRow, ...], np.ndarray]:
    pc1 = PCA(n_components=1).fit_transform(spectra.powder.spectra)[:, 0]
    transfer_probability = screening.transfer_probabilities[:, 1]
    transfer_truth = screening.y_true.astype(float)
    transfer_correct = (screening.transfer_probabilities.argmax(axis=1) == screening.y_true).astype(
        float
    )
    endpoints = (
        ("lr_transfer_probability", transfer_probability),
        (
            "lr_transfer_absolute_residual",
            np.abs(transfer_truth - transfer_probability),
        ),
        ("lr_transfer_correct", transfer_correct),
        ("spectral_pc1", pc1),
        ("raw_mean_intensity", spectra.powder.raw_mean_intensity),
        ("replicate_correlation", spectra.powder.replicate_correlation),
    )
    scopes = [("BPRO_label_adjusted", spectra.prefixes == "BPRO")]
    scopes.extend(
        (group, spectra.clinical_groups == group)
        for group in ("Control", "Biopsy-negative", "Cancer")
    )
    rows: list[OrderEffectRow] = []
    for scope_index, (scope, selected) in enumerate(scopes):
        for endpoint_index, (endpoint, values) in enumerate(endpoints):
            usable = selected & np.isfinite(values)
            rows.append(
                OrderEffectRow(
                    scope=scope,
                    endpoint=endpoint,
                    result=stratified_order_effect(
                        spectra.powder_order[usable].astype(float),
                        values[usable],
                        spectra.clinical_groups[usable],
                        seed=seed + scope_index * 20 + endpoint_index,
                        n_permutations=n_permutations,
                    ),
                )
            )
    return tuple(rows), pc1


def bpro_tertiles(
    spectra: PairedSpectra,
    screening: PairedCrossfitResult,
    pc1: np.ndarray,
) -> tuple[TertileRow, ...]:
    selected = np.flatnonzero(spectra.prefixes == "BPRO")
    ordered = selected[np.argsort(spectra.powder_order[selected])]
    probability = screening.transfer_probabilities[:, 1]
    prediction = screening.transfer_probabilities.argmax(axis=1)
    rows: list[TertileRow] = []
    for period, indices in zip(
        ("early", "middle", "late"), np.array_split(ordered, 3), strict=True
    ):
        rows.append(
            TertileRow(
                period=period,
                n=len(indices),
                probability_mean=float(np.mean(probability[indices])),
                error_rate=float(np.mean(prediction[indices] != screening.y_true[indices])),
                pc1_mean=float(np.mean(pc1[indices])),
                raw_intensity_mean=float(np.mean(spectra.powder.raw_mean_intensity[indices])),
                replicate_correlation_mean=float(
                    np.mean(spectra.powder.replicate_correlation[indices])
                ),
            )
        )
    return tuple(rows)


def compare_peaks(attributions: LrPeakAttributions) -> PeakBlock:
    return PeakBlock(
        agreement=peak_agreement(
            attributions.legacy.names,
            attributions.legacy.global_importance,
            attributions.native.global_importance,
            top_k=5,
        ),
        legacy=attributions.legacy,
        powder=attributions.native,
        transfer=attributions.transfer,
    )
