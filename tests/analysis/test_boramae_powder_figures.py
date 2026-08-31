from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2] / "publications" / "전향검체" / "보라매병원" / "src"
sys.path.insert(0, str(SRC))

import powder_comparison
from powder_comparison import figures as powder_figures
from powder_comparison import (
    inverted_band_figure,
    preprocessing_diagnostics,
    preprocessing_figure,
    signal_noise_figures_a,
)
from powder_comparison import summary_figure as summary_figures
from powder_comparison.data import ModalitySpectra, PairedSpectra, PreprocessingStageMatrices
from powder_comparison.evaluation import ComparisonBlock
from powder_comparison.statistics import (
    AgreementSummary,
    MetricComparison,
    ReclassificationSummary,
)


def _paired_spectra() -> PairedSpectra:
    grid = np.linspace(400.0, 800.0, 9)
    groups = np.repeat(np.array(["Control", "Biopsy-negative", "Cancer"]), 2)
    base = np.sin(grid / 90.0)
    legacy_values = np.vstack([base + offset for offset in (0.0, 0.1, 0.3, 0.4, 0.7, 0.8)])
    powder_values = np.vstack(
        [base + offset for offset in (0.0, 0.08, 0.10, 0.18, 0.21, 0.29)]
    ) + np.linspace(-0.2, 0.3, len(grid))

    def modality(values: np.ndarray) -> ModalitySpectra:
        stages = PreprocessingStageMatrices(
            raw=values + 5.0,
            smoothed=values + 4.0,
            baseline_corrected=values + 2.0,
            snv=values,
        )
        return ModalitySpectra(
            spectra=values,
            raw_mean_intensity=np.mean(values, axis=1),
            raw_max_intensity=np.max(values, axis=1),
            replicate_correlation=np.ones(len(values)),
            stages=stages,
        )

    return PairedSpectra(
        grid=grid,
        sample_ids=np.arange(1, 7),
        clinical_groups=groups,
        prefixes=np.repeat(np.array(["BNOR", "BPRO", "BPRO"]), 2),
        powder_order=np.arange(1, 7),
        legacy=modality(legacy_values),
        powder=modality(powder_values),
    )


def test_powder_summary_separates_transfer_and_overall_analysis(tmp_path: Path) -> None:
    # Given: transfer and powder-native comparisons for both classification tasks.
    agreement = AgreementSummary(0.5, 0.2, 0.0, 0.4)
    reclassification = ReclassificationSummary(53, 14, 25, 17, -11 / 109)

    def block(name: str, values: tuple[tuple[str, float, float], ...]) -> ComparisonBlock:
        return ComparisonBlock(
            name=name,
            metrics=tuple(
                MetricComparison(metric, legacy, powder, powder - legacy, -0.2, 0.2)
                for metric, legacy, powder in values
            ),
            reclassification=reclassification,
            agreement=agreement,
        )

    data = summary_figures.SummaryFigureData(
        comparisons=(
            block(
                "screening_transfer",
                (
                    ("roc_auc", 0.714, 0.529),
                    ("sensitivity", 0.659, 0.098),
                    ("specificity", 0.750, 0.926),
                    ("balanced_accuracy", 0.704, 0.512),
                ),
            ),
            block(
                "screening_native",
                (
                    ("roc_auc", 0.529, 0.583),
                    ("sensitivity", 0.098, 0.585),
                    ("specificity", 0.926, 0.559),
                    ("balanced_accuracy", 0.512, 0.572),
                ),
            ),
            block(
                "three_transfer",
                (
                    ("macro_ovr_roc_auc", 0.811, 0.509),
                    ("control_ovr_roc_auc", 0.926, 0.519),
                    ("biopsy_negative_ovr_roc_auc", 0.747, 0.538),
                    ("cancer_ovr_roc_auc", 0.759, 0.468),
                ),
            ),
            block(
                "three_native",
                (
                    ("macro_ovr_roc_auc", 0.509, 0.616),
                    ("control_ovr_roc_auc", 0.519, 0.808),
                    ("biopsy_negative_ovr_roc_auc", 0.538, 0.496),
                    ("cancer_ovr_roc_auc", 0.468, 0.543),
                ),
            ),
        ),
        n_subjects=109,
    )

    # When: publication summary figures are written.
    summary_figures.write_summary_figures(data, tmp_path)

    # Then: transfer and overall analysis are separate PNG/PDF artifacts.
    for name in (
        "fig07a_powder_transfer_summary",
        "fig07b_powder_overall_analysis_summary",
        "fig07_powder_comparison_summary",
    ):
        assert (tmp_path / f"{name}.png").is_file()
        assert (tmp_path / f"{name}.pdf").is_file()


def test_preprocessing_panel_traces_stage_shift_to_lr_probability(tmp_path: Path) -> None:
    # Given: paired stage spectra with a powder shift concentrated at a legacy LR peak.
    grid = np.array([400.0, 410.0, 420.0, 430.0, 440.0])
    legacy_snv = np.array([[0.0, -1.0, 0.0, 1.0, 0.0], [1.0, 0.0, -1.0, 0.0, 1.0]])
    powder_snv = legacy_snv.copy()
    powder_snv[:, 2] += np.array([2.0, 3.0])
    legacy = PreprocessingStageMatrices(
        raw=legacy_snv + 5.0,
        smoothed=legacy_snv + 4.0,
        baseline_corrected=legacy_snv + 2.0,
        snv=legacy_snv,
    )
    powder = PreprocessingStageMatrices(
        raw=(legacy_snv + 5.0) * 2.0,
        smoothed=(legacy_snv + 4.0) * 1.8,
        baseline_corrected=legacy_snv + 2.0,
        snv=powder_snv,
    )

    # When: stage shift and its link to prediction movement are evaluated and plotted.
    diagnostics = preprocessing_diagnostics.evaluate_preprocessing_shift(
        grid,
        legacy,
        powder,
        peak_names=("420.0 cm^-1", "440.0 cm^-1"),
        peak_importance=np.array([0.8, 0.2]),
        legacy_probability=np.array([0.2, 0.7]),
        transfer_probability=np.array([0.5, 0.1]),
    )
    preprocessing_figure.write_preprocessing_diagnostics_figure(
        preprocessing_figure.PreprocessingFigureData(
            grid=grid,
            legacy=legacy,
            powder=powder,
            diagnostics=diagnostics,
            clinical_groups=np.array(["Control", "Cancer"]),
            patient_ids=np.array(["P01", "P02"]),
        ),
        tmp_path,
    )

    # Then: raw scale shift, peak-localized SNV shift, and the publication artifact are traceable.
    assert diagnostics.rows[0].median_log2_rms_ratio == pytest.approx(1.0)
    assert diagnostics.peak_shift_enrichment > 1.0
    assert (tmp_path / "fig16_preprocessing_domain_shift_diagnostics.png").is_file()
    assert (tmp_path / "fig16_preprocessing_domain_shift_diagnostics.pdf").is_file()


def test_clinical_group_spectra_figure_shows_paired_modalities(tmp_path: Path) -> None:
    # Given: paired final-SNV spectra for all three clinical subgroups.
    spectra = _paired_spectra()

    # When: the subgroup spectrum comparison is exported.
    powder_figures.write_clinical_group_spectra_figure(spectra, tmp_path)

    # Then: both publication image formats are available for visual inspection.
    assert (tmp_path / "fig17_clinical_group_spectra_comparison.png").is_file()
    assert (tmp_path / "fig17_clinical_group_spectra_comparison.pdf").is_file()


def test_inverted_band_figure_stratifies_clinical_and_gleason_groups(
    tmp_path: Path,
) -> None:
    # Given: paired spectra spanning the inverted 936 cm^-1 band and Cancer grade labels.
    spectra = _paired_spectra()
    spectra = PairedSpectra(
        grid=np.linspace(900.0, 970.0, len(spectra.grid)),
        sample_ids=spectra.sample_ids,
        clinical_groups=spectra.clinical_groups,
        prefixes=spectra.prefixes,
        powder_order=spectra.powder_order,
        legacy=spectra.legacy,
        powder=spectra.powder,
    )
    grade_bands = np.array(["Unknown", "Unknown", "Unknown", "Unknown", "GG1-2", "GG3-5"])

    # When: the inverted-band diagnostic is exported.
    inverted_band_figure.write_inverted_band_figure(spectra, grade_bands, tmp_path)

    # Then: PNG and PDF artifacts are available for slide use.
    assert (tmp_path / "fig26_inverted_band_group_spectra.png").is_file()
    assert (tmp_path / "fig26_inverted_band_group_spectra.pdf").is_file()
    assert (tmp_path / "fig26_inverted_band_group_spectra_mobile.png").is_file()


def test_group_separation_figure_compares_signal_with_within_group_spread(
    tmp_path: Path,
) -> None:
    # Given: liquid spectra with wider subgroup gaps than the paired powder spectra.
    spectra = _paired_spectra()

    # When: standardized subgroup-separation diagnostics are exported.
    powder_figures.write_clinical_group_separation_figure(spectra, tmp_path)

    # Then: the visual comparison is available in both publication formats.
    assert (tmp_path / "fig18_clinical_group_separation_diagnostics.png").is_file()
    assert (tmp_path / "fig18_clinical_group_separation_diagnostics.pdf").is_file()


def test_group_separation_ratio_detects_collapsed_powder_subgroups() -> None:
    # Given: the powder subgroup centroids are closer than their liquid counterparts.
    spectra = _paired_spectra()

    # When: subgroup signal is standardized by within-group spectral spread.
    diagnostics = powder_comparison.evaluate_group_separation(spectra)

    # Then: every pair has less relative separation in powder than in liquid.
    legacy_ratios = np.array([row.separation_to_spread for row in diagnostics.legacy_pairs])
    powder_ratios = np.array([row.separation_to_spread for row in diagnostics.powder_pairs])
    assert np.all(legacy_ratios > powder_ratios)


def test_labeled_boxplot_uses_current_matplotlib_tick_label_contract() -> None:
    # Given: two distributions that need reader-facing modality labels.
    figure, axis = plt.subplots()

    # When: the shared labeled-boxplot adapter renders them.
    signal_noise_figures_a.add_labeled_boxplot(
        axis,
        (np.array([0.9, 0.95]), np.array([0.7, 0.8])),
        ("Liquid", "Powder"),
    )

    # Then: the active Matplotlib API receives and displays both tick labels.
    assert [tick.get_text() for tick in axis.get_xticklabels()] == ["Liquid", "Powder"]
    plt.close(figure)
