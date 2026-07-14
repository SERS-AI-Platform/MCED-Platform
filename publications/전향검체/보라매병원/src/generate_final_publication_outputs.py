#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.7",
#   "numpy>=1.24",
#   "openpyxl>=3.1",
#   "pandas>=2.0",
#   "pyyaml>=6.0",
#   "scikit-learn>=1.3",
#   "scipy>=1.10",
# ]
# ///
# How to run:
#   uv run publications/전향검체/보라매병원/src/generate_final_publication_outputs.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from boramae_data import (
    FIG_DIR,
    MODEL_GRID,
    OUT,
    TABLE_DIR,
    build_boramae_subjects,
    load_clinical_samples,
)
from boramae_peak_modes import PeakModeResult, generate_peak_mode_outputs
from prostate_comparison_model import OofResult, TaskName, build_task, nested_oof
from prostate_comparison_outputs import (
    ResultOutput,
    validate_report_snapshot,
    write_combined_metrics,
    write_result,
)
from prostate_comparison_plots import plot_screening_performance, plot_three_group_performance

REPORT: Final = OUT / "PROSTATE_COMPARISON.md"
TASK_CLASSES: Final[dict[TaskName, tuple[str, ...]]] = {
    "screening_binary": ("Non-cancer", "Cancer"),
    "three_group": ("Control", "Biopsy-negative", "Cancer"),
}
OBSOLETE_FIGURES: Final = (
    "fig03_three_group_train_test_performance",
    "fig04_subject_peak_count_distribution",
    "fig06_clean_pro_vs_prospective_shift",
    "fig07_three_group_and_binary_oof_performance",
    "fig07_legacy_vs_prospective_peak_alignment",
    "fig08_legacy_vs_prospective_peak_alignment",
    "fig04_group_peak_sets_common_differential",
)
OBSOLETE_TABLE_PATTERNS: Final = (
    "aligned_*",
    "strict_binary_*",
    "cohort_source_*",
    "prostate_peak_alignment_*",
    "prostate_spectral_shift_*",
    "prostate_urea_calibration_shifts.csv",
    "prostate_cohort_clinical_comparison.csv",
    "fig03_*",
    "fig04_subject_peak_counts.csv",
    "fig04_group_peak_count_summary.csv",
    "fig04_peak_detection_criteria.csv",
    "fig04_group_peak_sets.csv",
    "fig04_common_and_differential_peaks.csv",
    "boramae_train_test_split.csv",
)


@dataclass(frozen=True, slots=True)
class CohortCounts:
    control: int
    psa_bx_negative: int
    prostate_cancer: int

    @property
    def total(self) -> int:
        return self.control + self.psa_bx_negative + self.prostate_cancer


@dataclass(frozen=True, slots=True)
class AnalysisResults:
    screening_binary: OofResult
    three_group: OofResult


def remove_obsolete_outputs() -> None:
    for name in OBSOLETE_FIGURES:
        for suffix in ("png", "pdf"):
            (FIG_DIR / f"{name}.{suffix}").unlink(missing_ok=True)
    for pattern in OBSOLETE_TABLE_PATTERNS:
        for path in TABLE_DIR.glob(pattern):
            path.unlink()
    (OUT / "LEGACY_SINGLE_SPLIT_SUMMARY.md").unlink(missing_ok=True)


def metric_ci(result: OofResult, name: str) -> str:
    low, high = result.intervals[name]
    return f"{result.metrics[name]:.3f} ({low:.3f}-{high:.3f})"


def count_cohort(labels: np.ndarray) -> CohortCounts:
    return CohortCounts(
        control=int(np.sum(labels == "Control")),
        psa_bx_negative=int(np.sum(labels == "Biopsy-negative")),
        prostate_cancer=int(np.sum(labels == "Prostate cancer")),
    )


def run_task(task: TaskName, x: np.ndarray, labels: np.ndarray) -> OofResult:
    task_x, task_y, _ = build_task(x, labels, task)
    result = nested_oof(task_x, task_y)
    write_result(ResultOutput(task, result, list(TASK_CLASSES[task])))
    return result


def write_report(results: AnalysisResults, counts: CohortCounts) -> None:
    with REPORT.open("w", encoding="utf-8") as handle:
        handle.write("# 보라매 전향검체 Screening 및 3군 성능\n\n")
        handle.write("## 분석 대상\n\n")
        handle.write(
            f"- 총 {counts.total}명: Control {counts.control}명, "
            f"Biopsy-negative (PSA 상승·전립선 생검 음성) {counts.psa_bx_negative}명, "
            f"전립선암 {counts.prostate_cancer}명\n"
        )
        handle.write("- `_ave` 파일은 제외하고 `_1.._5` replicate만 사용, subject mean으로 집계\n")
        handle.write("- 모든 성능은 subject-level nested 5-fold OOF 예측으로 계산\n")
        handle.write(
            "- 내부 4-fold에서 Logistic Regression `C`를 선택하고 외부 fold는 모델 선택에 사용하지 않음\n"
        )
        handle.write("- 95% CI는 OOF 예측의 2,000회 bootstrap\n\n")
        handle.write("## Screening 성능\n\n")
        handle.write("Biopsy-negative 47명은 non-cancer 쪽에 포함했다.\n\n")
        handle.write("| 정의 | N | ROC-AUC | Balanced accuracy | Sensitivity | Specificity |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        handle.write(
            f"| Screening: Control+Biopsy-negative vs Cancer | {len(results.screening_binary.y_true)} | "
            f"{metric_ci(results.screening_binary, 'roc_auc')} | "
            f"{metric_ci(results.screening_binary, 'balanced_accuracy')} | "
            f"{metric_ci(results.screening_binary, 'sensitivity')} | "
            f"{metric_ci(results.screening_binary, 'specificity')} |\n\n"
        )
        handle.write(
            "- Figure: `figures/fig03_screening_binary_auc_confusion_matrix.png` / `.pdf`\n"
        )
        handle.write("- Confusion matrix: `tables/screening_binary_confusion_matrix.csv`\n\n")
        handle.write("## 3-group 성능\n\n")
        handle.write("| 정의 | N | Macro OVR ROC-AUC | Balanced accuracy | Macro F1 |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        handle.write(
            f"| 3-group: Control/Biopsy-negative/Cancer | {len(results.three_group.y_true)} | "
            f"{metric_ci(results.three_group, 'macro_ovr_roc_auc')} | "
            f"{metric_ci(results.three_group, 'balanced_accuracy')} | "
            f"{metric_ci(results.three_group, 'macro_f1')} |\n\n"
        )
        handle.write("- Figure: `figures/fig06_three_group_auc_confusion_matrix.png` / `.pdf`\n")
        handle.write("- Confusion matrix: `tables/three_group_confusion_matrix.csv`\n\n")
        handle.write("## 해석 주의\n\n")
        handle.write(
            "- 현재 결과는 표본 수가 작아 validation split 없이 nested OOF로만 산출한 exploratory 성능이다.\n"
        )
        handle.write(
            "- Cancer Screening AUC는 외부검증 성능이나 cross-hospital 일반화 근거로 사용하면 안 된다.\n"
        )


def write_summary(
    results: AnalysisResults,
    counts: CohortCounts,
    screening_peaks: PeakModeResult,
    three_group_peaks: PeakModeResult,
) -> None:
    with (OUT / "SUMMARY.md").open("w", encoding="utf-8") as handle:
        handle.write("# 보라매병원 전향검체 SERS 분석 산출물\n\n")
        handle.write(
            f"- 분석 포함: {counts.total}명 — Control {counts.control}, Biopsy-negative {counts.psa_bx_negative}, Prostate {counts.prostate_cancer}\n"
        )
        handle.write(
            "- Figures: Fig01 preprocessing, Fig02 Clean PRO comparison, Fig03 Screening AUC/CM, "
            "Fig04a Screening peaks, Fig04b 3-group peaks, Fig05 Grade Group, Fig06 3-group AUC/CM\n"
        )
        handle.write(
            f"- Screening ROC-AUC: {results.screening_binary.metrics['roc_auc']:.3f}, balanced accuracy: {results.screening_binary.metrics['balanced_accuracy']:.3f}\n"
        )
        handle.write(
            f"- 3-group macro OVR ROC-AUC: {results.three_group.metrics['macro_ovr_roc_auc']:.3f}, balanced accuracy: {results.three_group.metrics['balanced_accuracy']:.3f}\n"
        )
        handle.write(
            "- Peak criteria: group mean preprocessed SNV spectra, local maxima, "
            "prominence `max(0.08, 10% of group range)`, min distance 18 cm^-1, "
            "common if all groups are within +/-12 cm^-1\n"
        )
        handle.write(
            f"- Peak clusters: Fig04a Screening common {screening_peaks.common_count}, "
            f"differential {screening_peaks.differential_count}; "
            f"Fig04b 3-group common {three_group_peaks.common_count}, "
            f"differential {three_group_peaks.differential_count}\n"
        )
        handle.write(
            f"- Shaded regions: top {screening_peaks.shaded_count} differential peak clusters "
            f"for Fig04a and top {three_group_peaks.shaded_count} for Fig04b\n"
        )
        handle.write("- Peak tables: `tables/fig04a_screening_*`, `tables/fig04b_three_group_*`\n")
        handle.write("- 상세 성능과 95% CI: `PROSTATE_COMPARISON.md`\n")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    remove_obsolete_outputs()
    grid = np.load(MODEL_GRID)
    subjects = build_boramae_subjects(load_clinical_samples(), grid)
    screening_peaks, three_group_peaks = generate_peak_mode_outputs(subjects, grid)
    x = np.vstack([subject.mean_spectrum for subject in subjects])
    labels = np.array([subject.sample.group for subject in subjects])
    results = AnalysisResults(
        screening_binary=run_task("screening_binary", x, labels),
        three_group=run_task("three_group", x, labels),
    )
    write_combined_metrics(list(TASK_CLASSES))
    plot_screening_performance(results.screening_binary, FIG_DIR)
    plot_three_group_performance(results.three_group, FIG_DIR)
    counts = count_cohort(labels)
    write_report(results, counts)
    write_summary(results, counts, screening_peaks, three_group_peaks)
    validate_report_snapshot(REPORT)
    print(f"Wrote final Boramae publication outputs to {OUT}")


if __name__ == "__main__":
    main()
