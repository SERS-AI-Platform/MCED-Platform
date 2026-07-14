#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "click>=8.1",
#   "matplotlib>=3.7",
#   "numpy>=1.24",
#   "openpyxl>=3.1",
#   "pandas>=2.0",
#   "pyyaml>=6.0",
#   "scikit-learn>=1.3",
#   "scipy>=1.10",
#   "tqdm>=4.65",
# ]
# ///
from __future__ import annotations

from typing import Final

import numpy as np
from boramae_data import (
    MODEL_GRID,
    OUT,
    build_aligned_boramae_subjects,
    build_boramae_subjects,
    build_clean_pro_alignment,
    load_clinical_samples,
)
from prostate_comparison_model import OofResult, TaskName, build_task, nested_oof
from prostate_comparison_outputs import (
    AlignmentOutput,
    ResultOutput,
    validate_report_snapshot,
    write_alignment,
    write_combined_metrics,
    write_result,
)
from prostate_comparison_plots import (
    AlignmentPlotData,
    plot_peak_alignment,
    plot_screening_performance,
    plot_three_group_performance,
)
from prostate_comparison_report import (
    AlignmentResults,
    AnalysisResults,
    count_cohort,
    write_report,
    write_summary,
)
from prostate_shift_alignment import detect_mean_peaks, summarize_peaks

TASK_CLASSES: Final[dict[TaskName, tuple[str, ...]]] = {
    "screening_binary": ("Non-cancer", "Cancer"),
    "strict_binary": ("Control", "Cancer"),
    "three_group": ("Control", "Biopsy-negative", "Cancer"),
}
ANALYSIS_OUT: Final = OUT / "clean_vs_prospective"
FIG_DIR: Final = ANALYSIS_OUT / "figures"
TABLE_DIR: Final = ANALYSIS_OUT / "tables"
REPORT: Final = ANALYSIS_OUT / "PROSTATE_COMPARISON.md"
SUMMARY: Final = ANALYSIS_OUT / "SUMMARY.md"


def run_task(task: TaskName, output_name: str, x: np.ndarray, labels: np.ndarray) -> OofResult:
    task_x, task_y, _ = build_task(x, labels, task)
    result = nested_oof(task_x, task_y)
    write_result(ResultOutput(output_name, result, list(TASK_CLASSES[task])), TABLE_DIR)
    return result


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    grid = np.load(MODEL_GRID)
    clinical_samples = load_clinical_samples()
    subjects = build_boramae_subjects(clinical_samples, grid)
    aligned_subjects, prospective_shifts = build_aligned_boramae_subjects(clinical_samples, grid)
    clean = build_clean_pro_alignment(grid)
    x = np.vstack([subject.mean_spectrum for subject in subjects])
    aligned_x = np.vstack([subject.mean_spectrum for subject in aligned_subjects])
    labels = np.array([subject.sample.group for subject in subjects])

    results = AnalysisResults(
        screening_binary=run_task("screening_binary", "screening_binary", x, labels),
        aligned_screening_binary=run_task(
            "screening_binary", "aligned_screening_binary", aligned_x, labels
        ),
        strict_binary=run_task("strict_binary", "strict_binary", x, labels),
        aligned_strict_binary=run_task("strict_binary", "aligned_strict_binary", aligned_x, labels),
        three_group=run_task("three_group", "three_group", x, labels),
        aligned_three_group=run_task("three_group", "aligned_three_group", aligned_x, labels),
    )

    cancer = x[labels == "Prostate cancer"]
    aligned_cancer = aligned_x[labels == "Prostate cancer"]
    cancer_shifts = prospective_shifts[labels == "Prostate cancer"].ravel()
    source_y = np.r_[np.zeros(len(clean.spectra), dtype=int), np.ones(len(cancer), dtype=int)]
    source = nested_oof(np.vstack([clean.spectra, cancer]), source_y)
    aligned_source = nested_oof(np.vstack([clean.aligned_spectra, aligned_cancer]), source_y)
    write_result(
        ResultOutput("cohort_source", source, ["Clean PRO", "Prospective cancer"]), TABLE_DIR
    )
    write_result(
        ResultOutput("aligned_cohort_source", aligned_source, ["Clean PRO", "Prospective cancer"]),
        TABLE_DIR,
    )

    before = summarize_peaks(
        "Common grid only",
        detect_mean_peaks(grid, clean.spectra),
        detect_mean_peaks(grid, cancer),
    )
    after = summarize_peaks(
        "Urea aligned",
        detect_mean_peaks(grid, clean.aligned_spectra),
        detect_mean_peaks(grid, aligned_cancer),
    )
    write_alignment(
        AlignmentOutput(
            (before[0], after[0]),
            (before[1], after[1]),
            (clean.shifts, cancer_shifts),
        ),
        TABLE_DIR,
    )
    alignment = AlignmentResults(
        common_grid=before[0],
        urea_aligned=after[0],
        source_auc=source.metrics["roc_auc"],
        aligned_source_auc=aligned_source.metrics["roc_auc"],
        mean_spectrum_correlation=float(
            np.corrcoef(clean.spectra.mean(axis=0), cancer.mean(axis=0))[0, 1]
        ),
        clean_anchor_count=int(np.count_nonzero(clean.shifts)),
        prospective_anchor_count=int(np.count_nonzero(cancer_shifts)),
    )

    tasks = [
        "screening_binary",
        "aligned_screening_binary",
        "strict_binary",
        "aligned_strict_binary",
        "three_group",
        "aligned_three_group",
        "cohort_source",
        "aligned_cohort_source",
    ]
    write_combined_metrics(tasks, TABLE_DIR)
    plot_screening_performance(results.screening_binary, FIG_DIR)
    plot_three_group_performance(results.three_group, FIG_DIR)
    plot_peak_alignment(
        AlignmentPlotData(
            grid,
            clean.spectra,
            cancer,
            clean.aligned_spectra,
            aligned_cancer,
            (before[1], after[1]),
            (clean.shifts, cancer_shifts),
        ),
        FIG_DIR,
    )
    counts = count_cohort(labels)
    write_report(results, counts, alignment, REPORT)
    write_summary(results, counts, alignment, SUMMARY)
    validate_report_snapshot(REPORT, TABLE_DIR)
    print(f"Wrote clean-vs-prospective outputs to {REPORT.parent}")


if __name__ == "__main__":
    main()
