from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from powder_comparison.cohort import PairedSubject, collect_subject_files, pair_subjects
from powder_comparison.data import (
    ClinicalCovariates,
    PairedSpectra,
    build_paired_spectra,
    load_clinical_covariates,
    load_clinical_labels,
)
from powder_comparison.evaluation import (
    ComparisonBlock,
    McNemarRow,
    OrderEffectRow,
    PeakBlock,
    TertileRow,
    bpro_tertiles,
    compare_crossfit,
    compare_peaks,
    crossfit_mcnemar_rows,
    order_effect_rows,
)
from powder_comparison.lr_attribution import fit_lr_peak_attributions
from powder_comparison.models import PairedCrossfitResult, run_paired_crossfit
from powder_comparison.preprocessing_diagnostics import (
    PreprocessingDiagnostics,
    evaluate_preprocessing_shift,
)
from powder_comparison.signal_noise_diagnostics import evaluate_signal_noise_diagnostics
from powder_comparison.signal_noise_types import SignalNoiseDiagnostics
from powder_comparison.threshold_evaluation import compare_binary_thresholds


@dataclass(frozen=True, slots=True)
class SourcePaths:
    legacy_root: Path
    powder_root: Path
    powder_lot_root: Path
    clinical_path: Path
    artifact_dir: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    sources: SourcePaths
    pairs: tuple[PairedSubject, ...]
    spectra: PairedSpectra
    clinical: ClinicalCovariates
    screening: PairedCrossfitResult
    three_group: PairedCrossfitResult
    comparisons: tuple[ComparisonBlock, ComparisonBlock, ComparisonBlock, ComparisonBlock]
    threshold_comparisons: tuple[ComparisonBlock, ...]
    mcnemar: tuple[McNemarRow, ...]
    order_effects: tuple[OrderEffectRow, ...]
    order_pc1: np.ndarray
    tertiles: tuple[TertileRow, ...]
    peaks: PeakBlock
    preprocessing: PreprocessingDiagnostics
    signal_noise: SignalNoiseDiagnostics
    decision: str


def default_sources(repo: Path) -> SourcePaths:
    return SourcePaths(
        legacy_root=repo / "data" / "raw_data" / "20260709_BPRO,BNOR_1mW_0.05s_Ave100",
        powder_root=repo / "data" / "20260716_Urine test (Powder_BNOR, BPRO)",
        powder_lot_root=repo / "data" / "20260715_Powder_Reproducibility test",
        clinical_path=repo / "data" / "clinical_data" / "보라매 병원 임상정보.xlsx",
        artifact_dir=repo / "artifacts" / "usersnet" / "v1.0.0",
        output_dir=repo / "publications" / "전향검체" / "보라매병원",
    )


def _metric(block: ComparisonBlock, name: str):
    return next(row for row in block.metrics if row.name == name)


def _development_decision(comparisons: tuple[ComparisonBlock, ...]) -> str:
    screening_transfer, screening_native, three_transfer, three_native = comparisons
    discrimination_loss = (
        _metric(screening_transfer, "roc_auc").ci_high < 0.0
        or _metric(three_transfer, "macro_ovr_roc_auc").ci_high < 0.0
    )
    native_gain = (
        _metric(screening_native, "roc_auc").ci_low > 0.0
        or _metric(three_native, "macro_ovr_roc_auc").ci_low > 0.0
    )
    if discrimination_loss and native_gain:
        return "신규 powder 전용 모델 개발 필요"
    if discrimination_loss:
        return "성능 저하 원인 규명 후 모델 개발 재평가"
    return "현 모델 유지 가능하나 추가 lot·무작위 순서 검증 필요"


def run_analysis(
    sources: SourcePaths,
    *,
    bootstrap_resamples: int = 10_000,
    order_permutations: int = 9_999,
) -> AnalysisResult:
    labels = load_clinical_labels(sources.clinical_path)
    pairs = pair_subjects(
        labels,
        collect_subject_files(sources.legacy_root),
        collect_subject_files(sources.powder_root),
    )
    grid = np.load(sources.artifact_dir / "common_grid.npy")
    spectra = build_paired_spectra(pairs, grid)
    clinical = load_clinical_covariates(sources.clinical_path, spectra.sample_ids)
    screening = run_paired_crossfit(
        spectra.legacy.spectra,
        spectra.powder.spectra,
        spectra.clinical_groups,
        task="screening_binary",
    )
    three_group = run_paired_crossfit(
        spectra.legacy.spectra,
        spectra.powder.spectra,
        spectra.clinical_groups,
        task="three_group",
    )
    screening_transfer, screening_native = compare_crossfit(
        screening, seed=20260721, n_resamples=bootstrap_resamples
    )
    three_transfer, three_native = compare_crossfit(
        three_group, seed=20260731, n_resamples=bootstrap_resamples
    )
    comparisons = (
        screening_transfer,
        screening_native,
        three_transfer,
        three_native,
    )
    threshold_comparisons = compare_binary_thresholds(
        screening,
        seed=20260741,
        n_resamples=bootstrap_resamples,
    )
    order_effects, order_pc1 = order_effect_rows(
        spectra,
        screening,
        seed=20260751,
        n_permutations=order_permutations,
    )
    peak_attributions = fit_lr_peak_attributions(
        grid,
        spectra.legacy.spectra,
        spectra.powder.spectra,
        spectra.clinical_groups,
    )
    peaks = compare_peaks(peak_attributions)
    preprocessing = evaluate_preprocessing_shift(
        grid,
        spectra.legacy.stages,
        spectra.powder.stages,
        peak_names=peaks.legacy.names,
        peak_importance=peaks.legacy.global_importance,
        legacy_probability=screening.legacy_probabilities[:, 1],
        transfer_probability=screening.transfer_probabilities[:, 1],
    )
    signal_noise = evaluate_signal_noise_diagnostics(
        spectra,
        clinical.patient_ids,
        screening,
        three_group,
        peaks.legacy.names,
        peaks.legacy.global_importance,
        sources.powder_lot_root,
    )
    return AnalysisResult(
        sources=sources,
        pairs=pairs,
        spectra=spectra,
        clinical=clinical,
        screening=screening,
        three_group=three_group,
        comparisons=comparisons,
        threshold_comparisons=threshold_comparisons,
        mcnemar=crossfit_mcnemar_rows(screening),
        order_effects=order_effects,
        order_pc1=order_pc1,
        tertiles=bpro_tertiles(spectra, screening, order_pc1),
        peaks=peaks,
        preprocessing=preprocessing,
        signal_noise=signal_noise,
        decision=_development_decision(comparisons),
    )
