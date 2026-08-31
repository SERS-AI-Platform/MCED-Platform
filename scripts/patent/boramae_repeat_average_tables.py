from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from boramae_repeat_average_core import (
    GROUP_ORDER,
    LAGS,
    SUBJECT_FIELDS,
    subject_metric_rows,
)
from boramae_repeat_average_utils import stats, write_csv


def cohort_rows(
    subjects: list[dict[str, object]], audits: list[dict[str, object]]
) -> list[dict[str, object]]:
    rows = []
    for group in (*GROUP_ORDER, "All included"):
        selected = (
            subjects if group == "All included" else [s for s in subjects if s["group"] == group]
        )
        group_audits = (
            audits if group == "All included" else [r for r in audits if r["group"] == group]
        )
        counts = [int(s["keep"].sum()) for s in selected]
        rows.append(
            {
                "group": group,
                "subjects": len(selected),
                "input_replicates": len(selected) * 5,
                "qc_passed_replicates": sum(counts),
                "qc_subjects_with_all_5": sum(c == 5 for c in counts),
                "qc_min_per_subject": min(counts),
                "qc_median_per_subject": float(np.median(counts)),
                "qc_max_per_subject": max(counts),
                "average_files_audited": len(group_audits),
                "average_files_passing_normalized_rmse_le_1e-4": sum(
                    bool(r["target_usable"]) for r in group_audits
                ),
            }
        )
    return rows


def metric_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for group in (*GROUP_ORDER, "All included"):
        selected = rows if group == "All included" else [r for r in rows if r["group"] == group]
        for metric in SUBJECT_FIELDS[5:]:
            output.append(
                {"group": group, "metric": metric, **stats([row[metric] for row in selected])}
            )
    return output


def spectrum_rows(
    grid: np.ndarray, subjects: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    spectra, noise = [], []
    for group in GROUP_ORDER:
        selected = [s for s in subjects if s["group"] == group]
        means = np.vstack([s["mean"] for s in selected])
        sds = np.vstack([s["sd"] for s in selected])
        for index, wn in enumerate(grid):
            between_sd = float(means[:, index].std(ddof=1))
            spectra.append(
                {
                    "group": group,
                    "wavenumber_cm-1": float(wn),
                    "mean_representative_spectrum": float(means[:, index].mean()),
                    "between_subject_sd": between_sd,
                    "between_subject_sem": between_sd / math.sqrt(len(means)),
                }
            )
            noise.append(
                {
                    "group": group,
                    "wavenumber_cm-1": float(wn),
                    "median_within_subject_sd": float(np.median(sds[:, index])),
                    "p25_within_subject_sd": float(np.percentile(sds[:, index], 25)),
                    "p75_within_subject_sd": float(np.percentile(sds[:, index], 75)),
                }
            )
    return spectra, noise


def subject_spectrum_rows(
    grid: np.ndarray, subjects: list[dict[str, object]]
) -> list[dict[str, object]]:
    return [
        {
            "subject_ordinal": subject["ordinal"],
            "group": subject["group"],
            "qc_replicates": int(subject["keep"].sum()),
            "wavenumber_cm-1": float(wn),
            "average_representative_spectrum": float(value),
        }
        for subject in subjects
        for wn, value in zip(grid, subject["mean"], strict=True)
    ]


def covariance_rows(
    subjects: list[dict[str, object]], grid: np.ndarray
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows, pairs = [], []
    for group in (*GROUP_ORDER, "All included"):
        selected = (
            subjects if group == "All included" else [s for s in subjects if s["group"] == group]
        )
        for lag in LAGS:
            rows.append(
                {
                    "group": group,
                    "lag_points": lag,
                    "lag_cm-1": float(lag * np.median(np.diff(grid))),
                    "median_autocovariance": stats([s["acov"][lag] for s in selected])["median"],
                    "median_autocorrelation": stats([s["acorr"][lag] for s in selected])["median"],
                }
            )
        pairs.append(
            {
                "group": group,
                "median_off_diagonal_residual_covariance": stats(
                    [s["mean_pair_covariance"] for s in selected]
                )["median"],
                "median_off_diagonal_residual_correlation": stats(
                    [s["mean_pair_correlation"] for s in selected]
                )["median"],
            }
        )
    return rows, pairs


def partial_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for group in (*GROUP_ORDER, "All included"):
        selected = rows if group == "All included" else [r for r in rows if r["group"] == group]
        for n in sorted({int(r["n"]) for r in selected}):
            n_rows = [r for r in selected if r["n"] == n]
            output.append(
                {
                    "group": group,
                    "n": n,
                    "subjects_available": len(n_rows),
                    "empirical_subset_noise_scale_median": stats(
                        [r["empirical_noise_scale"] for r in n_rows]
                    )["median"],
                    "expected_noise_scale_median": stats(
                        [r["expected_noise_scale"] for r in n_rows]
                    )["median"],
                    "expected_variance_ratio_1_over_n": 1 / n,
                    "median_subset_rmse_to_full": stats(
                        [r["median_subset_rmse_to_full"] for r in n_rows]
                    )["median"],
                    "subset_variance_observed": int(
                        any(np.isfinite(r["empirical_noise_scale"]) for r in n_rows)
                    ),
                }
            )
    return output


def write_tables(
    out: Path,
    grid: np.ndarray,
    subjects: list[dict[str, object]],
    audits: list[dict[str, object]],
    partial_rows: list[dict[str, object]],
) -> dict[str, object]:
    subject_rows = subject_metric_rows(subjects)
    subject_spectra = subject_spectrum_rows(grid, subjects)
    spectra, noise = spectrum_rows(grid, subjects)
    covariances, pairs = covariance_rows(subjects, grid)
    partial = partial_summary(partial_rows)
    write_csv(out / "subject_repeatability_metrics.csv", subject_rows)
    write_csv(out / "subject_average_representative_spectra.csv", subject_spectra)
    write_csv(out / "average_file_audit_deidentified.csv", audits)
    write_csv(out / "cohort_summary.csv", cohort_rows(subjects, audits))
    write_csv(out / "repeatability_metric_summary.csv", metric_summary(subject_rows))
    write_csv(out / "group_mean_representative_spectra.csv", spectra)
    write_csv(out / "within_subject_noise_by_wavenumber.csv", noise)
    write_csv(out / "residual_autocovariance_summary.csv", covariances)
    write_csv(out / "inter_replicate_residual_covariance_summary.csv", pairs)
    write_csv(out / "partial_average_summary.csv", partial)
    return {
        "subject_rows": subject_rows,
        "partial_summary": partial,
        "covariance_rows": covariances,
    }
