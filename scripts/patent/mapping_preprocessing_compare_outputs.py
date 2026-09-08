from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from boramae_repeat_average_utils import stats, write_csv
from mapping_preprocessing_compare_core import PreprocessingComparison
from mapping_repeat_average_peaks import detect_peaks, high_frequency_noise

METHODS = (
    "changed_raw_qc_average",
    "changed_average_after_previous_transform",
    "previous_production_average",
)
PLOT_METHODS = (METHODS[0], METHODS[2])


@dataclass(frozen=True, slots=True)
class ComparisonTables:
    subject_rows: tuple[dict[str, str | int | float], ...]
    summary_rows: tuple[dict[str, str | int | float], ...]
    spectrum_rows: tuple[dict[str, str | int | float], ...]


def _subject_row(result: PreprocessingComparison) -> dict[str, str | int | float]:
    raw_repro = sum(peak.support_fraction >= 0.5 for peak in result.raw_peaks)
    legacy_repro = sum(peak.support_fraction >= 0.5 for peak in result.legacy_peaks)
    raw_snr = [peak.snr for peak in result.raw_peaks]
    legacy_snr = [peak.snr for peak in result.legacy_peaks]
    return {
        "subject_ordinal": result.subject.ordinal,
        "group": result.subject.group,
        "qc_passed_repeats": int(result.keep.sum()),
        "calibration_shift_median_cm-1": float(np.median(result.calibration_shifts)),
        "calibration_shift_abs_max_cm-1": float(np.max(np.abs(result.calibration_shifts))),
        "changed_average_hf_noise": high_frequency_noise(result.raw_average),
        "changed_after_previous_transform_hf_noise": high_frequency_noise(
            result.changed_after_previous_transform
        ),
        "previous_average_hf_noise": high_frequency_noise(result.legacy_average),
        "changed_repeat_hf_noise": result.raw_noise,
        "previous_repeat_hf_noise": result.legacy_noise,
        "changed_average_std": float(result.raw_average.std()),
        "previous_average_std": float(result.legacy_average.std()),
        "changed_peak_count": len(result.raw_peaks),
        "previous_peak_count": len(result.legacy_peaks),
        "changed_reproducible_peak_count": raw_repro,
        "previous_reproducible_peak_count": legacy_repro,
        "changed_median_peak_snr": float(np.median(raw_snr)) if raw_snr else float("nan"),
        "previous_median_peak_snr": (
            float(np.median(legacy_snr)) if legacy_snr else float("nan")
        ),
        "changed_max_peak_snr": float(np.max(raw_snr)) if raw_snr else float("nan"),
        "previous_max_peak_snr": float(np.max(legacy_snr)) if legacy_snr else float("nan"),
        "changed_previous_shape_correlation": result.raw_legacy_correlation,
        "changed_after_previous_transform_peak_count": len(
            detect_peaks(
                result.changed_after_previous_transform,
                np.arange(len(result.changed_after_previous_transform), dtype=float),
                high_frequency_noise(result.changed_after_previous_transform),
            )
        ),
        "changed_after_previous_transform_shape_correlation": result.transformed_shape_correlation,
        "changed_average_min": float(result.raw_average.min()),
        "changed_average_max": float(result.raw_average.max()),
        "previous_average_min": float(result.legacy_average.min()),
        "previous_average_max": float(result.legacy_average.max()),
    }


def subject_rows(
    results: list[PreprocessingComparison],
) -> list[dict[str, str | int | float]]:
    return [_subject_row(result) for result in results]


def summary_rows(
    rows: list[dict[str, str | int | float]], groups: tuple[str, ...]
) -> list[dict[str, str | int | float]]:
    metrics = tuple(
        key
        for key, value in rows[0].items()
        if key not in {"subject_ordinal", "group"}
        and isinstance(value, (int, float))
    )
    output: list[dict[str, str | int | float]] = []
    for group in (*groups, "All included"):
        selected = rows if group == "All included" else [r for r in rows if r["group"] == group]
        for metric in metrics:
            output.append({"group": group, "metric": metric, **stats([r[metric] for r in selected])})
    return output


def spectrum_rows(
    grid: np.ndarray,
    results: list[PreprocessingComparison],
    groups: tuple[str, ...],
) -> list[dict[str, str | int | float]]:
    output: list[dict[str, str | int | float]] = []
    for group in (*groups, "All included"):
        selected = results if group == "All included" else [r for r in results if r.subject.group == group]
        for method in METHODS:
            values = np.vstack(
                [
                    r.raw_average
                    if method == METHODS[0]
                    else r.changed_after_previous_transform
                    if method == METHODS[1]
                    else r.legacy_average
                    for r in selected
                ]
            )
            for index, wavenumber in enumerate(grid):
                output.append(
                    {
                        "group": group,
                        "method": method,
                        "wavenumber_cm-1": float(wavenumber),
                        "mean_intensity": float(values[:, index].mean()),
                        "between_subject_sd": float(values[:, index].std(ddof=1)),
                    }
                )
    return output


def write_tables(
    out: Path,
    grid: np.ndarray,
    results: list[PreprocessingComparison],
    groups: tuple[str, ...],
) -> ComparisonTables:
    subjects = tuple(subject_rows(results))
    summaries = tuple(summary_rows(list(subjects), groups))
    spectra = tuple(spectrum_rows(grid, results, groups))
    write_csv(out / "subject_preprocessing_comparison.csv", list(subjects))
    write_csv(out / "preprocessing_comparison_summary.csv", list(summaries))
    write_csv(out / "preprocessing_group_spectra.csv", list(spectra))
    return ComparisonTables(subjects, summaries, spectra)
