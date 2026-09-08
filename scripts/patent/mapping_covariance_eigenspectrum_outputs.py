from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from mapping_covariance_eigenspectrum_core import ExperimentResult


def _write_csv(path: Path, header: list[str], rows: list[list[str | float | int]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_tables(out: Path, result: ExperimentResult) -> None:
    """Write eigenvalues, wavenumber profiles, modes, and subject summaries."""
    within, raw, corrected = result.within, result.between_raw, result.between_corrected
    eigen_rows = [
        [index + 1, within.eigenvalues[index], within.explained[index], within.cumulative[index], raw.eigenvalues[index], raw.explained[index], raw.cumulative[index], corrected.eigenvalues[index], corrected.explained[index], corrected.cumulative[index]]
        for index in range(len(result.grid))
    ]
    _write_csv(out / "eigen_spectrum.csv", ["component", "within_eigenvalue", "within_explained", "within_cumulative", "between_raw_eigenvalue", "between_raw_explained", "between_raw_cumulative", "between_corrected_eigenvalue", "between_corrected_explained", "between_corrected_cumulative"], eigen_rows)
    within_variance = within.diagonal_sd**2
    mean_noise_variance = within_variance * result.mean_inverse_repeats
    corrected_variance = np.maximum(np.diag(result.corrected_between_covariance), 0.0)
    raw_variance = np.maximum(np.diag(result.raw_mean_covariance), 0.0)
    profile_rows = [
        [float(wavenumber), within.diagonal_sd[index], np.sqrt(mean_noise_variance[index]), np.sqrt(raw_variance[index]), np.sqrt(corrected_variance[index]), corrected_variance[index] / max(corrected_variance[index] + mean_noise_variance[index], 1e-12)]
        for index, wavenumber in enumerate(result.grid)
    ]
    _write_csv(out / "wavenumber_covariance_profile.csv", ["wavenumber_cm-1", "within_repeat_noise_sd", "mean_spectrum_noise_sd", "between_raw_sd", "between_corrected_sd", "mean_reliability_ratio"], profile_rows)
    mode_rows = [
        [component + 1, float(wavenumber), within.eigenvectors[index, component], corrected.eigenvectors[index, component]]
        for component in range(min(10, len(result.grid)))
        for index, wavenumber in enumerate(result.grid)
    ]
    _write_csv(out / "top_covariance_modes.csv", ["component", "wavenumber_cm-1", "within_loading", "between_corrected_loading"], mode_rows)
    _write_csv(out / "subject_covariance_summary.csv", ["subject_ordinal", "group", "retained_repeats", "noise_floor_median", "noise_p95", "residual_rms"], [[row.ordinal, row.group, row.retained_repeats, row.noise_floor_median, row.noise_p95, row.residual_rms] for row in result.subject_rows])
    group_rows = [
        [summary.name, summary.n_subjects, summary.degrees_of_freedom, summary.trace, summary.effective_rank, summary.components_95, float(np.median(summary.diagonal_sd)), float(np.percentile(summary.diagonal_sd, 95))]
        for summary in result.group_within
    ]
    _write_csv(out / "group_within_covariance_summary.csv", ["group", "subjects", "degrees_of_freedom", "trace", "effective_rank", "components_for_95pct", "median_noise_sd", "p95_noise_sd"], group_rows)


def write_metadata(out: Path, result: ExperimentResult, raw_axis: Mapping[str, float | int | list[int]]) -> None:
    """Write a machine-readable description of the covariance experiment."""
    metadata = {
        "analysis_name": "Mapping within-repeat and between-subject covariance eigen-spectrum",
        "subject_count": len(result.subject_rows),
        "repeat_count_total": int(result.repeat_counts.sum()),
        "grid_points": len(result.grid),
        "grid_min_cm-1": float(result.grid[0]),
        "grid_max_cm-1": float(result.grid[-1]),
        "raw_axis": dict(raw_axis),
        "mean_inverse_repeats": result.mean_inverse_repeats,
        "retained_repeats_min": int(result.repeat_counts.min()),
        "retained_repeats_median": float(np.median(result.repeat_counts)),
        "retained_repeats_max": int(result.repeat_counts.max()),
        "within_effective_rank": result.within.effective_rank,
        "within_components_for_95pct": result.within.components_95,
        "between_corrected_effective_rank": result.between_corrected.effective_rank,
        "between_corrected_components_for_95pct": result.between_corrected.components_95,
        "corrected_between_min_eigenvalue": result.corrected_between_min_eigenvalue,
        "corrected_between_negative_variance_fraction": result.corrected_between_negative_variance_fraction,
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(out: Path, result: ExperimentResult, raw_axis: Mapping[str, float | int | list[int]]) -> None:
    """Write the interpretation boundary and the main covariance findings."""
    within, corrected = result.within, result.between_corrected
    report = "\n".join([
        "# Mapping covariance eigen-spectrum experiment", "", "## Design", "",
        f"- Cohort: {len(result.subject_rows)} subjects and {int(result.repeat_counts.sum())} QC-retained finite repeats.",
        f"- Retained repeats per subject: {int(result.repeat_counts.min())}–{int(result.repeat_counts.max())} (median {float(np.median(result.repeat_counts)):.0f}).",
        f"- Grid: {result.grid[0]:.1f}–{result.grid[-1]:.1f} cm⁻¹, {len(result.grid)} points.",
        f"- Raw axis: {dict(raw_axis)}.",
        "- Replicates were aligned without baseline correction or SNV. The existing repeat-residual robust MAD QC mask was applied before covariance estimation.",
        "- Within-repeat covariance was pooled from residuals around each subject mean spectrum.",
        f"- Between-subject mean covariance used a method-of-moments correction subtracting mean(1/n) × within covariance; mean(1/n)={result.mean_inverse_repeats:.6f}.", "",
        "## Results", "",
        f"- Within-repeat covariance effective rank: {within.effective_rank:.1f}; components for 95% variance: {within.components_95}.",
        f"- Corrected between-subject covariance effective rank: {corrected.effective_rank:.1f}; components for 95% variance: {corrected.components_95}.",
        f"- Raw between-subject minimum eigenvalue: {result.raw_between_min_eigenvalue:.6g}.",
        f"- Corrected between-subject minimum eigenvalue before clipping: {result.corrected_between_min_eigenvalue:.6g}.",
        f"- Negative eigenvalue absolute-mass fraction before clipping: {result.corrected_between_negative_variance_fraction:.4%}.",
        f"- Median within-repeat noise SD across wavenumbers: {float(np.median(within.diagonal_sd)):.6g}.",
        f"- Median noise SD of the subject mean: {float(np.median(within.diagonal_sd * np.sqrt(result.mean_inverse_repeats))):.6g}.", "",
        f"- Median corrected between-subject SD across wavenumbers: {float(np.median(corrected.diagonal_sd)):.6g}.",
        f"- Median subject-mean noise / corrected between-subject SD ratio: {float(np.median((within.diagonal_sd * np.sqrt(result.mean_inverse_repeats)) / np.maximum(corrected.diagonal_sd, 1e-12))):.4%}.", "",
        "## Interpretation boundary", "",
        "- Large eigenvalues of within-repeat covariance describe structured repeat-to-repeat variability. They are not automatically biological signal.",
        "- Corrected between-subject covariance contains biological variation plus subject-level batch, collection, substrate, and other confounding variation.",
        "- Corrected covariance eigenvalues were clipped at zero only for explained-variance plots and mode export. A material negative-eigenvalue fraction indicates that the simple covariance decomposition is unstable or incomplete.",
        "- The covariance profile should be compared with model occlusion maps and reproducible peak support before assigning a mode to a biochemical feature.", "",
        "## Outputs", "", "- `eigen_spectrum.csv`", "- `wavenumber_covariance_profile.csv`", "- `top_covariance_modes.csv`", "- `subject_covariance_summary.csv`", "- `group_within_covariance_summary.csv`", "- `01`–`04` PNG figures",
    ])
    (out / "REPORT.md").write_text(report, encoding="utf-8")
