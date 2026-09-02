from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .data import PairedSpectra
from .group_separation import GroupSeparationDiagnostics, evaluate_group_separation
from .models import PairedCrossfitResult
from .signal_noise_ablation import AblationResult, run_peak_ablation
from .signal_noise_lot import VarianceComponents, load_lot_replicates, variance_components
from .signal_noise_peak import PeakRepeatabilityDiagnostics, evaluate_peak_repeatability
from .signal_noise_types import (
    EffectRetentionSummary,
    ImportantPeakRetention,
    PatientNoiseLink,
    PeakLandscape,
    SignalNoiseDiagnostics,
)


class MissingReplicateStagesError(RuntimeError):
    """Raised when signal/noise diagnostics receive subject averages only."""


def summarize_effect_retention(
    legacy_effect: np.ndarray,
    powder_effect: np.ndarray,
    *,
    minimum_abs_legacy_effect: float = 0.10,
    group_pair: str = "unspecified",
) -> EffectRetentionSummary:
    """Summarize preservation of liquid subgroup effects in powder spectra."""
    relevant = np.abs(legacy_effect) >= minimum_abs_legacy_effect
    legacy = legacy_effect[relevant]
    powder = powder_effect[relevant]
    if len(legacy) < 2:
        return EffectRetentionSummary(group_pair, len(legacy), np.nan, np.nan, np.nan)
    correlation = spearmanr(legacy, powder).statistic
    retention = np.abs(powder) / np.abs(legacy)
    return EffectRetentionSummary(
        group_pair=group_pair,
        relevant_feature_count=len(legacy),
        spearman_rho=float(correlation),
        sign_agreement=float(np.mean(np.sign(legacy) == np.sign(powder))),
        median_absolute_retention=float(np.median(retention)),
    )


def _category(legacy_rate: float, powder_rate: float) -> str:
    legacy_stable = legacy_rate >= 0.30
    powder_stable = powder_rate >= 0.30
    if legacy_stable and powder_stable:
        return "common_stable"
    if powder_stable:
        return "powder_only"
    if legacy_stable:
        return "liquid_only"
    return "neither_stable"


def _landscape(
    repeatability: PeakRepeatabilityDiagnostics,
    separation: GroupSeparationDiagnostics,
) -> PeakLandscape:
    legacy_rate = np.mean(repeatability.legacy_presence, axis=0)
    powder_rate = np.mean(repeatability.powder_presence, axis=0)
    legacy_effect = np.max(
        np.abs(np.vstack([row.hedges_g for row in separation.legacy_pairs])), axis=0
    )
    powder_effect = np.max(
        np.abs(np.vstack([row.hedges_g for row in separation.powder_pairs])), axis=0
    )
    return PeakLandscape(
        legacy_presence_rate=legacy_rate,
        powder_presence_rate=powder_rate,
        legacy_max_abs_effect=legacy_effect,
        powder_max_abs_effect=powder_effect,
        categories=np.array(
            [_category(old, new) for old, new in zip(legacy_rate, powder_rate, strict=True)]
        ),
    )


def _important_peak_rows(
    spectra: PairedSpectra,
    separation: GroupSeparationDiagnostics,
    landscape: PeakLandscape,
    names: tuple[str, ...],
    importance: np.ndarray,
) -> tuple[ImportantPeakRetention, ...]:
    legacy_effect = separation.legacy_pairs[0].hedges_g
    powder_effect = separation.powder_pairs[0].hedges_g
    rows: list[ImportantPeakRetention] = []
    for name, weight in zip(names, importance, strict=True):
        shift = float(name.split()[0])
        index = int(np.argmin(np.abs(spectra.grid - shift)))
        denominator = abs(float(legacy_effect[index]))
        rows.append(
            ImportantPeakRetention(
                peak_name=name,
                shift_cm_1=float(spectra.grid[index]),
                legacy_importance=float(weight),
                legacy_effect=float(legacy_effect[index]),
                powder_effect=float(powder_effect[index]),
                absolute_effect_retention=(
                    abs(float(powder_effect[index])) / denominator if denominator > 0.0 else np.nan
                ),
                direction_agreement=bool(
                    np.sign(legacy_effect[index]) == np.sign(powder_effect[index])
                ),
                legacy_presence_rate=float(landscape.legacy_presence_rate[index]),
                powder_presence_rate=float(landscape.powder_presence_rate[index]),
                category=str(landscape.categories[index]),
            )
        )
    return tuple(rows)


def _patient_links(
    spectra: PairedSpectra,
    patient_ids: np.ndarray,
    screening: PairedCrossfitResult,
    repeatability: PeakRepeatabilityDiagnostics,
) -> tuple[PatientNoiseLink, ...]:
    snv_rows = {
        row.sample_id: row
        for row in repeatability.stage_rows
        if row.modality == "powder" and row.stage == "snv"
    }
    predictions = screening.native_probabilities.argmax(axis=1)
    return tuple(
        PatientNoiseLink(
            patient_id=str(patient_ids[index]),
            sample_id=int(sample_id),
            clinical_group=str(spectra.clinical_groups[index]),
            powder_replicate_correlation=float(spectra.powder.replicate_correlation[index]),
            powder_consensus_fraction=snv_rows[int(sample_id)].consensus_fraction,
            native_cancer_probability=float(screening.native_probabilities[index, 1]),
            native_screening_correct=bool(predictions[index] == screening.y_true[index]),
        )
        for index, sample_id in enumerate(spectra.sample_ids)
    )


def _conclusions(
    repeatability: PeakRepeatabilityDiagnostics,
    summaries: tuple[EffectRetentionSummary, ...],
    lot_variance: VarianceComponents,
    ablation: tuple[AblationResult, ...],
) -> tuple[str, ...]:
    legacy = [
        row
        for row in repeatability.stage_rows
        if row.modality == "legacy_liquid" and row.stage == "snv"
    ]
    powder = [
        row for row in repeatability.stage_rows if row.modality == "powder" and row.stage == "snv"
    ]
    full = next(row for row in ablation if row.feature_set == "full_spectrum")
    excluded = next(row for row in ablation if row.feature_set == "powder_only_excluded")
    technical_fraction = float(
        np.median(lot_variance.lot_fraction + lot_variance.residual_fraction)
    )
    return (
        f"Median apparent peaks (SNV): liquid {np.median([r.median_peak_count for r in legacy]):.1f}, powder {np.median([r.median_peak_count for r in powder]):.1f}",
        f"Median 4/5 consensus fraction: liquid {np.median([r.consensus_fraction for r in legacy]):.3f}, powder {np.median([r.consensus_fraction for r in powder]):.3f}",
        f"Cancer-Control effect sign agreement: {summaries[0].sign_agreement:.3f}; median |effect| retention {summaries[0].median_absolute_retention:.3f}",
        f"Five-lot median technical variance fraction: {technical_fraction:.3f}",
        f"Binary OOF AUROC full {full.metrics.roc_auc:.3f}, powder-only excluded {excluded.metrics.roc_auc:.3f}",
    )


def evaluate_signal_noise_diagnostics(
    spectra: PairedSpectra,
    patient_ids: np.ndarray,
    screening: PairedCrossfitResult,
    three_group: PairedCrossfitResult,
    peak_names: tuple[str, ...],
    peak_importance: np.ndarray,
    lot_root: Path,
) -> SignalNoiseDiagnostics:
    """Run repeatability, biological-retention, lot, and leakage-free ablation analyses."""
    legacy_replicates = spectra.legacy.replicate_stages
    powder_replicates = spectra.powder.replicate_stages
    if legacy_replicates is None or powder_replicates is None:
        raise MissingReplicateStagesError("Replicate-level preprocessing stages are required")
    repeatability = evaluate_peak_repeatability(
        spectra.grid, legacy_replicates, powder_replicates, spectra.sample_ids
    )
    separation = evaluate_group_separation(spectra)
    summaries = tuple(
        summarize_effect_retention(
            old.hedges_g,
            new.hedges_g,
            group_pair=old.label,
        )
        for old, new in zip(separation.legacy_pairs, separation.powder_pairs, strict=True)
    )
    landscape = _landscape(repeatability, separation)
    lot_variance = variance_components(load_lot_replicates(lot_root, spectra.grid).spectra)
    screening_ablation = run_peak_ablation(
        spectra.powder.spectra,
        spectra.clinical_groups,
        repeatability.legacy_presence,
        repeatability.powder_presence,
        task="screening_binary",
    )
    three_ablation = run_peak_ablation(
        spectra.powder.spectra,
        spectra.clinical_groups,
        repeatability.legacy_presence,
        repeatability.powder_presence,
        task="three_group",
    )
    return SignalNoiseDiagnostics(
        repeatability=repeatability,
        separation=separation,
        effect_summaries=summaries,
        important_peaks=_important_peak_rows(
            spectra, separation, landscape, peak_names, peak_importance
        ),
        landscape=landscape,
        lot_variance=lot_variance,
        screening_ablation=screening_ablation,
        three_group_ablation=three_ablation,
        patient_links=_patient_links(spectra, patient_ids, screening, repeatability),
        conclusions=_conclusions(repeatability, summaries, lot_variance, screening_ablation),
    )
