from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from boramae_repeat_average_utils import stats, write_csv
from mapping_repeat_average_core import GROUP_ORDER, SubjectResult
from mapping_repeat_average_peaks import (
    Peak,
    detect_peaks,
    high_frequency_noise,
    reproducible_peak_count,
)

CsvValue = str | int | float | bool
CsvRow = dict[str, CsvValue]


@dataclass(frozen=True, slots=True)
class OutputTables:
    subject_rows: tuple[CsvRow, ...]
    summary_rows: tuple[CsvRow, ...]
    spectrum_rows: tuple[CsvRow, ...]
    peak_rows: tuple[CsvRow, ...]
    partial_rows: tuple[CsvRow, ...]


def _peak_snr(peaks: tuple[Peak, ...]) -> tuple[float, float]:
    values = [float(peak.snr) for peak in peaks]
    return (
        (float(np.median(values)), float(np.max(values)))
        if values
        else (float("nan"), float("nan"))
    )


def subject_rows(results: list[SubjectResult]) -> list[CsvRow]:
    rows = []
    for result in results:
        existing_snr = _peak_snr(result.existing_peaks)
        patent_snr = _peak_snr(result.patent_peaks)
        rows.append(
            {
                "subject_ordinal": result.subject.ordinal,
                "group": result.subject.group,
                "input_repeats": len(result.subject.replicate_paths),
                "qc_passed_repeats": int(result.keep.sum()),
                "qc_rejection_rate": float(1 - result.keep.mean()),
                "repeat_noise_floor": result.repeat_noise_floor,
                "expected_mean_noise": result.expected_mean_noise,
                "single_repeat_hf_noise": result.single_hf_noise,
                "existing_ave_hf_noise": result.existing_hf_noise,
                "patent_qc_average_hf_noise": result.patent_hf_noise,
                "existing_noise_reduction_vs_single_pct": 100
                * (1 - result.existing_hf_noise / result.single_hf_noise),
                "patent_noise_reduction_vs_single_pct": 100
                * (1 - result.patent_hf_noise / result.single_hf_noise),
                "patent_extra_reduction_vs_existing_pct": 100
                * (1 - result.patent_hf_noise / result.existing_hf_noise),
                "instrument_average_normalized_rmse": result.average_normalized_rmse,
                "instrument_average_correlation": result.average_correlation,
                "patent_existing_normalized_rmse": result.method_difference_normalized_rmse,
                "patent_existing_correlation": result.method_difference_correlation,
                "baseline_aligned_normalized_rmse": result.baseline_aligned_normalized_rmse,
                "baseline_aligned_correlation": result.baseline_aligned_correlation,
                "raw_peak_count_median": result.raw_peak_count_median,
                "existing_peak_count": len(result.existing_peaks),
                "patent_peak_count": len(result.patent_peaks),
                "existing_reproducible_peak_count": reproducible_peak_count(result.existing_peaks),
                "patent_reproducible_peak_count": reproducible_peak_count(result.patent_peaks),
                "existing_median_peak_snr": existing_snr[0],
                "patent_median_peak_snr": patent_snr[0],
                "existing_max_peak_snr": existing_snr[1],
                "patent_max_peak_snr": patent_snr[1],
            }
        )
    return rows


def summary_rows(rows: list[CsvRow]) -> list[CsvRow]:
    metrics = tuple(
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and key != "subject_ordinal"
    )
    output = []
    for group in (*GROUP_ORDER, "All included"):
        selected = (
            rows if group == "All included" else [row for row in rows if row["group"] == group]
        )
        for metric in metrics:
            values = stats([row[metric] for row in selected])
            output.append({"group": group, "metric": metric, **values})
    return output


def spectrum_rows(grid: np.ndarray, results: list[SubjectResult]) -> list[CsvRow]:
    output = []
    for group in (*GROUP_ORDER, "All included"):
        selected = (
            results if group == "All included" else [r for r in results if r.subject.group == group]
        )
        existing = np.vstack([r.subject.existing_average for r in selected])
        patent = np.vstack([r.patent_average for r in selected])
        for index, wavenumber in enumerate(grid):
            output.append(
                {
                    "group": group,
                    "wavenumber_cm-1": float(wavenumber),
                    "existing_ave_mean": float(existing[:, index].mean()),
                    "patent_qc_mean": float(patent[:, index].mean()),
                    "patent_minus_existing": float((patent[:, index] - existing[:, index]).mean()),
                    "existing_between_subject_sd": float(existing[:, index].std(ddof=1)),
                    "patent_between_subject_sd": float(patent[:, index].std(ddof=1)),
                }
            )
    return output


def peak_catalog_rows(grid: np.ndarray, results: list[SubjectResult]) -> list[CsvRow]:
    output = []
    for group in (*GROUP_ORDER, "All included"):
        selected = (
            results if group == "All included" else [r for r in results if r.subject.group == group]
        )
        for method in ("existing", "patent"):
            spectra = np.vstack(
                [
                    r.subject.existing_average if method == "existing" else r.patent_average
                    for r in selected
                ]
            )
            mean_spectrum = spectra.mean(axis=0)
            noise = high_frequency_noise(mean_spectrum)
            peaks = detect_peaks(mean_spectrum, grid, noise)
            subject_peaks = [
                r.existing_peaks if method == "existing" else r.patent_peaks for r in selected
            ]
            for peak_number, peak in enumerate(peaks, start=1):
                support = sum(
                    any(abs(candidate.index - peak.index) <= 4 for candidate in peaks_for_subject)
                    for peaks_for_subject in subject_peaks
                ) / len(subject_peaks)
                output.append(
                    {
                        "group": group,
                        "method": method,
                        "peak_number": peak_number,
                        "wavenumber_cm-1": peak.wavenumber,
                        "prominence": peak.prominence,
                        "snr": peak.snr,
                        "subject_support_fraction": float(support),
                        "support_ge_50pct": bool(support >= 0.5),
                    }
                )
    return output


def partial_summary_rows(rows: list[dict[str, CsvValue]]) -> list[CsvRow]:
    output = []
    for group in (*GROUP_ORDER, "All included"):
        selected = (
            rows if group == "All included" else [row for row in rows if row["group"] == group]
        )
        for n in sorted({int(row["n"]) for row in selected}):
            values = [row for row in selected if row["n"] == n]
            empirical = [row["empirical_noise_scale_median"] for row in values]
            finite_empirical = [float(value) for value in empirical if np.isfinite(float(value))]
            output.append(
                {
                    "group": group,
                    "n": n,
                    "subjects_available": len(values),
                    "subset_count_median": float(
                        np.median([row["subset_count"] for row in values])
                    ),
                    "empirical_noise_scale_median": float(np.median(finite_empirical))
                    if finite_empirical
                    else float("nan"),
                    "expected_noise_scale_median": float(
                        np.median([row["expected_noise_scale"] for row in values])
                    ),
                    "expected_variance_ratio_1_over_n": 1 / n,
                }
            )
    return output


def write_tables(
    out: Path,
    grid: np.ndarray,
    results: list[SubjectResult],
    partial_rows: list[dict[str, CsvValue]],
) -> OutputTables:
    subjects = tuple(subject_rows(results))
    summaries = tuple(summary_rows(list(subjects)))
    spectra = tuple(spectrum_rows(grid, results))
    peaks = tuple(peak_catalog_rows(grid, results))
    partial = tuple(partial_summary_rows(partial_rows))
    write_csv(out / "subject_mapping_repeatability_peak_metrics.csv", list(subjects))
    write_csv(out / "mapping_method_comparison_summary.csv", list(summaries))
    write_csv(out / "mapping_group_method_spectra.csv", list(spectra))
    write_csv(out / "mapping_peak_catalog.csv", list(peaks))
    write_csv(out / "mapping_partial_average_summary.csv", list(partial))
    return OutputTables(subjects, summaries, spectra, peaks, partial)
