from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2] / "publications" / "전향검체" / "보라매병원" / "src"
sys.path.insert(0, str(SRC))

from powder_comparison import data as powder_data
from powder_comparison import figures as powder_figures
from powder_comparison.cohort import collect_subject_files, pair_subjects
from powder_comparison.data import build_paired_spectra, load_clinical_labels
from powder_comparison.delong import correlated_auc_test
from powder_comparison.models import PairedCrossfitResult, run_paired_crossfit
from powder_comparison.order_effect import stratified_order_effect
from powder_comparison.peaks import peak_agreement
from powder_comparison.runner import default_sources
from powder_comparison.statistics import (
    agreement_summary,
    reclassification_summary,
)
from sklearn.metrics import roc_auc_score


def test_correlated_delong_matches_auc_and_handles_identical_scores() -> None:
    # Given: identical paired scores for a binary outcome.
    truth = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])

    # When: the correlated AUC difference is tested.
    result = correlated_auc_test(truth, scores, scores.copy())

    # Then: both AUCs match and the null receives a unit p-value.
    assert result.legacy_auc == 1.0
    assert result.powder_auc == 1.0
    assert result.delta == 0.0
    assert result.p_value == 1.0


def test_default_outputs_share_boramae_publication_layout() -> None:
    # Given: the repository root and the Boramae publication convention.
    repo = Path(__file__).resolve().parents[2]
    publication_dir = repo / "publications" / "전향검체" / "보라매병원"

    # When: default powder-comparison paths are resolved.
    sources = default_sources(repo)

    # Then: figures and tables are rooted beside the existing Boramae outputs.
    assert sources.output_dir == publication_dir


def _write_replicates(root: Path, prefix: str, sample_id: int) -> None:
    for replicate in range(1, 6):
        (root / f"{prefix} {sample_id}_{replicate}.CSV").write_text(
            "400,1\n401,2\n", encoding="utf-8"
        )
    (root / f"{prefix} {sample_id}_ave.CSV").write_text("400,1\n401,2\n", encoding="utf-8")


def test_pair_subjects_uses_clinical_labels_and_five_replicates(tmp_path: Path) -> None:
    # Given: two modalities with the same control and cancer subjects.
    legacy = tmp_path / "legacy"
    powder = tmp_path / "powder"
    legacy.mkdir()
    powder.mkdir()
    for root in (legacy, powder):
        _write_replicates(root, "BNOR", 1)
        _write_replicates(root, "BPRO", 3)

    # When: subject files are paired against clinical labels.
    pairs = pair_subjects(
        {1: "Control", 3: "Cancer"},
        collect_subject_files(legacy),
        collect_subject_files(powder),
    )

    # Then: one row per patient is returned and averaged files are excluded.
    assert [(row.sample_id, row.clinical_group) for row in pairs] == [
        (1, "Control"),
        (3, "Cancer"),
    ]
    assert all(len(row.legacy.replicates) == 5 for row in pairs)
    assert all("_ave" not in path.stem for row in pairs for path in row.powder.replicates)


def test_reclassification_summary_separates_improvement_and_worsening() -> None:
    # Given: paired predictions with one improvement and two worsenings.
    truth = np.array([0, 0, 1, 1, 1])
    legacy = np.array([0, 1, 0, 1, 1])
    powder = np.array([0, 0, 1, 0, 0])

    # When: reclassification is summarized.
    result = reclassification_summary(truth, legacy, powder)

    # Then: all four directions are counted independently.
    assert result.stable_correct == 1
    assert result.improved == 2
    assert result.worsened == 2
    assert result.stable_incorrect == 0
    assert result.net_improvement_rate == 0.0


def test_agreement_summary_reports_unweighted_kappa() -> None:
    # Given: two three-class prediction vectors with partial agreement.
    legacy = np.array([0, 0, 1, 1, 2, 2])
    powder = np.array([0, 1, 1, 1, 2, 0])

    # When: agreement is calculated with deterministic bootstrap.
    result = agreement_summary(legacy, powder, seed=7, n_resamples=200)

    # Then: exact agreement and a bounded kappa interval are returned.
    assert result.exact_agreement == 4 / 6
    assert 0.0 < result.kappa < 1.0
    assert -1.0 <= result.ci_low <= result.ci_high <= 1.0


def test_stratified_order_effect_detects_within_group_drift() -> None:
    # Given: both clinical strata increase monotonically with acquisition order.
    order = np.arange(12, dtype=float)
    strata = np.array(["A"] * 6 + ["B"] * 6)
    values = np.array([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15], dtype=float)

    # When: order association is tested by shuffling only within strata.
    result = stratified_order_effect(order, values, strata, seed=11, n_permutations=999)

    # Then: a strong positive within-stratum effect is detected.
    assert result.rho > 0.95
    assert result.p_value < 0.01


def test_peak_agreement_uses_top_k_overlap_and_rank_correlation() -> None:
    # Given: two importance profiles with two shared top-three peaks.
    names = ("p1", "p2", "p3", "p4", "p5")
    legacy = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    powder = np.array([5.0, 1.0, 4.0, 3.0, 2.0])

    # When: peak agreement is calculated.
    result = peak_agreement(names, legacy, powder, top_k=3)

    # Then: overlap and full-profile rank agreement are both retained.
    assert result.shared_top_peaks == ("p1", "p3")
    assert result.jaccard == 0.5
    assert 0.0 < result.spearman_rho < 1.0


def test_powder_native_figures_match_binary_and_three_group_layout(tmp_path: Path) -> None:
    # Given: unbiased powder-native probabilities for binary and three-group tasks.
    binary_probabilities = np.array(
        [[0.9, 0.1], [0.7, 0.3], [0.4, 0.6], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]]
    )
    three_probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.6, 0.3, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
            [0.2, 0.1, 0.7],
        ]
    )
    binary = PairedCrossfitResult(
        task="screening_binary",
        class_names=("Non-cancer", "Cancer"),
        y_true=np.array([0, 0, 1, 1, 0, 1]),
        folds=np.array([1, 2, 1, 2, 1, 2]),
        legacy_probabilities=binary_probabilities,
        transfer_probabilities=binary_probabilities,
        native_probabilities=binary_probabilities,
        binary_thresholds=None,
    )
    three_group = PairedCrossfitResult(
        task="three_group",
        class_names=("Control", "Biopsy-negative", "Cancer"),
        y_true=np.array([0, 0, 1, 1, 2, 2]),
        folds=np.array([1, 2, 1, 2, 1, 2]),
        legacy_probabilities=three_probabilities,
        transfer_probabilities=three_probabilities,
        native_probabilities=three_probabilities,
        binary_thresholds=None,
    )

    # When: the powder-native publication figures are written.
    powder_figures.write_powder_native_performance_figures(binary, three_group, tmp_path)

    # Then: matching binary and three-group PNG/PDF outputs are available.
    for name in (
        "fig14_powder_native_screening_auc_confusion_matrix",
        "fig15_powder_native_three_group_auc_confusion_matrix",
    ):
        assert (tmp_path / f"{name}.png").is_file()
        assert (tmp_path / f"{name}.pdf").is_file()


def test_actual_powder_inventory_has_109_paired_subjects() -> None:
    # Given: the reviewed liquid, powder, and Boramae clinical sources.
    repo = Path(__file__).resolve().parents[2]
    legacy_root = repo / "data" / "raw_data" / "20260709_BPRO,BNOR_1mW_0.05s_Ave100"
    powder_root = repo / "data" / "20260716_Urine test (Powder_BNOR, BPRO)"
    clinical_path = repo / "data" / "clinical_data" / "보라매 병원 임상정보.xlsx"

    # When: the eligible paired cohort is assembled.
    pairs = pair_subjects(
        load_clinical_labels(clinical_path),
        collect_subject_files(legacy_root),
        collect_subject_files(powder_root),
    )

    # Then: all planned subjects and clinical strata are present.
    assert len(pairs) == 109
    assert sum(row.clinical_group == "Control" for row in pairs) == 21
    assert sum(row.clinical_group == "Biopsy-negative" for row in pairs) == 47
    assert sum(row.clinical_group == "Cancer" for row in pairs) == 41


def test_clinical_covariates_preserve_psa_missingness_by_group() -> None:
    # Given: the Boramae clinical workbook and paired subject order.
    repo = Path(__file__).resolve().parents[2]
    sources = default_sources(repo)
    pairs = pair_subjects(
        load_clinical_labels(sources.clinical_path),
        collect_subject_files(sources.legacy_root),
        collect_subject_files(sources.powder_root),
    )

    # When: PSA is aligned to the paired analysis cohort.
    covariates = powder_data.load_clinical_covariates(
        sources.clinical_path,
        np.array([pair.sample_id for pair in pairs]),
    )

    # Then: patient IDs and PSA values remain aligned to every paired sample.
    groups = np.array([pair.clinical_group for pair in pairs])
    assert len(covariates.patient_ids) == len(pairs)
    assert all(str(patient_id).isdigit() for patient_id in covariates.patient_ids)
    assert np.sum(np.isfinite(covariates.psa[groups == "Biopsy-negative"])) == 47
    assert np.sum(np.isfinite(covariates.psa[groups == "Cancer"])) == 0
    assert set(covariates.psa_bands[groups == "Biopsy-negative"]) == {"<4", "4-<10", ">=10"}


def test_paired_crossfit_shares_folds_and_keeps_modalities_separate() -> None:
    # Given: paired measurements with two balanced clinical classes.
    rng = np.random.default_rng(13)
    labels = np.array(["Control"] * 8 + ["Cancer"] * 8)
    base = np.concatenate((np.zeros((8, 4)), np.ones((8, 4))), axis=0)
    legacy = base + rng.normal(0.0, 0.1, base.shape)
    powder = base + 0.2 + rng.normal(0.0, 0.1, base.shape)

    # When: liquid-transfer and powder-native models use the same outer folds.
    result = run_paired_crossfit(
        legacy,
        powder,
        labels,
        task="screening_binary",
        outer_splits=4,
        inner_splits=2,
        seed=17,
        c_values=(0.1, 1.0),
    )

    # Then: every subject receives all three unbiased probability vectors.
    assert set(result.folds.tolist()) == {1, 2, 3, 4}
    assert result.legacy_probabilities.shape == (16, 2)
    assert result.transfer_probabilities.shape == (16, 2)
    assert result.native_probabilities.shape == (16, 2)
    assert np.allclose(result.legacy_probabilities.sum(axis=1), 1.0)
    assert np.allclose(result.transfer_probabilities.sum(axis=1), 1.0)


def test_binary_crossfit_optimizes_threshold_inside_each_training_fold() -> None:
    # Given: paired binary measurements with separable but shifted probabilities.
    rng = np.random.default_rng(23)
    labels = np.array(["Control"] * 12 + ["Cancer"] * 12)
    base = np.concatenate((np.zeros((12, 5)), np.ones((12, 5))), axis=0)
    legacy = base + rng.normal(0.0, 0.35, base.shape)
    powder = base + 0.6 + rng.normal(0.0, 0.35, base.shape)

    # When: paired LR cross-fitting includes training-fold Youden optimization.
    result = run_paired_crossfit(
        legacy,
        powder,
        labels,
        task="screening_binary",
        outer_splits=4,
        inner_splits=3,
        seed=19,
        c_values=(0.1, 1.0),
    )

    # Then: each held-out fold receives one threshold learned without its outcomes.
    optimized = result.binary_thresholds
    assert optimized is not None
    assert np.all((optimized.legacy_thresholds >= 0.0) & (optimized.legacy_thresholds <= 1.0))
    assert np.all((optimized.native_thresholds >= 0.0) & (optimized.native_thresholds <= 1.0))
    for fold in np.unique(result.folds):
        selected = result.folds == fold
        assert np.unique(optimized.legacy_thresholds[selected]).size == 1
        assert np.unique(optimized.native_thresholds[selected]).size == 1
    assert np.array_equal(
        optimized.transfer_predictions,
        result.transfer_probabilities[:, 1] >= optimized.legacy_thresholds,
    )


def test_lr_crossfit_reproduces_historical_liquid_auc() -> None:
    # Given: the paired Boramae cohort and the historical model grid.
    repo = Path(__file__).resolve().parents[2]
    sources = default_sources(repo)
    pairs = pair_subjects(
        load_clinical_labels(sources.clinical_path),
        collect_subject_files(sources.legacy_root),
        collect_subject_files(sources.powder_root),
    )
    spectra = build_paired_spectra(pairs, np.load(sources.artifact_dir / "common_grid.npy"))

    # When: the binary LR is evaluated with the locked historical nested-CV protocol.
    result = run_paired_crossfit(
        spectra.legacy.spectra,
        spectra.powder.spectra,
        spectra.clinical_groups,
        task="screening_binary",
    )

    # Then: the liquid score reproduces 0.714 and the same LR transfers to powder.
    legacy_auc = roc_auc_score(result.y_true, result.legacy_probabilities[:, 1])
    powder_auc = roc_auc_score(result.y_true, result.transfer_probabilities[:, 1])
    assert legacy_auc == 0.7137733142037302
    assert powder_auc == 0.5290530846484935
