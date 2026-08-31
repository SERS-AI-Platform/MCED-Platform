from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "notebooks/aecd_api_subject_peak_outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
PUBLICATION_FIGURE = (
    REPO_ROOT
    / "publications/전향검체/보라매병원/figures/mean_spectrum/fig10_subject_peak_dispersion.png"
)
LOCAL_VALIDATION_FIGURE = FIGURE_DIR / "local_neighbor_snr_validation.png"
PUBLICATION_LOCAL_VALIDATION_FIGURE = (
    REPO_ROOT
    / "publications/전향검체/보라매병원/figures/mean_spectrum/fig11_local_neighbor_snr_validation.png"
)
SOURCE_SCRIPT = (
    REPO_ROOT / "scripts/analysis/aecd_api_model_mean_spectrum_clinical_performance.py"
)
PEAK_DISPERSION_TOLERANCE_CM1 = 5.0
LOCAL_NOISE_HALF_WINDOW_CM1 = 12.0
LOCAL_NOISE_EXCLUSION_HALF_WIDTH_CM1 = 5.0
SNR_FACTORS = (2.0, 3.0, 5.0)
REPRODUCIBLE_REPEAT_MINIMUM = 2

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
PUBLICATION_FIGURE.parent.mkdir(parents=True, exist_ok=True)


def load_pipeline():
    spec = importlib.util.spec_from_file_location("mean_spectrum_pipeline", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the mean-spectrum pipeline module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def estimate_local_noise(
    spectrum: np.ndarray,
    common_grid: np.ndarray,
    peak_index: int,
) -> float:
    center = float(common_grid[peak_index])
    distance = np.abs(common_grid - center)
    nearby = (distance <= LOCAL_NOISE_HALF_WINDOW_CM1) & (
        distance >= LOCAL_NOISE_EXCLUSION_HALF_WIDTH_CM1
    )
    nearby_indices = np.flatnonzero(nearby)
    left_indices = nearby_indices[common_grid[nearby_indices] < center]
    right_indices = nearby_indices[common_grid[nearby_indices] > center]
    difference_values = []
    for indices in (left_indices, right_indices):
        if len(indices) >= 2:
            difference_values.append(np.diff(spectrum[indices]))
    if not difference_values:
        difference_values.append(np.diff(spectrum))
    differences = np.concatenate(difference_values)
    centered = differences - np.median(differences)
    return max(
        1.4826 * float(np.median(np.abs(centered))) / np.sqrt(2.0),
        np.finfo(float).eps,
    )


def detect_candidates(
    spectrum: np.ndarray,
    common_grid: np.ndarray,
    snr_factor: float,
) -> list[dict]:
    grid_step = float(np.median(np.diff(common_grid)))
    local_wlen_points = max(
        3,
        2 * int(np.ceil(LOCAL_NOISE_HALF_WINDOW_CM1 / grid_step)) + 1,
    )
    candidate_indices, properties = find_peaks(
        np.asarray(spectrum, dtype=np.float64),
        prominence=0.0,
        wlen=local_wlen_points,
    )
    if len(candidate_indices) == 0:
        return []
    prominence = np.asarray(properties["prominences"], dtype=np.float64)
    rows = []
    for position, index in enumerate(candidate_indices):
        local_noise_sigma = estimate_local_noise(
            spectrum,
            common_grid,
            int(index),
        )
        local_snr = float(prominence[position] / local_noise_sigma)
        if local_snr < snr_factor:
            continue
        rows.append(
            {
                "index": int(index),
                "center_cm1": float(common_grid[index]),
                "intensity": float(spectrum[index]),
                "prominence": float(prominence[position]),
                "local_noise_sigma": float(local_noise_sigma),
                "local_snr": local_snr,
            }
        )
    return rows


def match_repeat_candidates(
    mean_candidates: list[dict],
    repeat_candidates: list[dict],
) -> list[dict]:
    pairs = []
    for mean_index, mean_candidate in enumerate(mean_candidates):
        for repeat_index, repeat_candidate in enumerate(repeat_candidates):
            distance = abs(
                mean_candidate["center_cm1"]
                - repeat_candidate["center_cm1"]
            )
            if distance <= PEAK_DISPERSION_TOLERANCE_CM1:
                pairs.append(
                    (
                        distance,
                        -repeat_candidate["local_snr"],
                        mean_index,
                        repeat_index,
                    )
                )
    pairs.sort()
    used_mean_indices = set()
    used_repeat_indices = set()
    matches = []
    for _distance, _negative_snr, mean_index, repeat_index in pairs:
        if mean_index in used_mean_indices or repeat_index in used_repeat_indices:
            continue
        used_mean_indices.add(mean_index)
        used_repeat_indices.add(repeat_index)
        matches.append(
            {
                "mean_index": int(mean_index),
                "repeat_candidate_index": int(repeat_index),
                **repeat_candidates[repeat_index],
            }
        )
    return matches


def build_subject_registry(
    pipeline,
    common_grid: np.ndarray,
    subject_keys: list[str],
    labels: list[str],
    replicate_matrices: list[np.ndarray],
    snr_factor: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry_rows: list[dict] = []
    summary_rows: list[dict] = []
    for subject_index, (subject_key, label, replicates) in enumerate(
        zip(subject_keys, labels, replicate_matrices, strict=True)
    ):
        keep, distances, limit = pipeline.qc_repeats(replicates)
        passed = replicates[keep]
        representative = passed.mean(axis=0)
        residuals = passed - representative
        noise_sigma = np.maximum(
            np.std(residuals, axis=0, ddof=1),
            np.finfo(float).eps,
        )
        mean_candidates_before_dispersion = detect_candidates(
            representative,
            common_grid,
            snr_factor,
        )
        mean_candidates = mean_candidates_before_dispersion
        repeat_candidates = [
            detect_candidates(
                spectrum,
                common_grid,
                snr_factor,
            )
            for spectrum in passed
        ]
        matched_repeat_detection_count = int(
            sum(len(rows) for rows in repeat_candidates)
        )
        reproducible_counts = {2: 0, 50: 0, 80: 0}
        matched_rows_by_mean = {index: [] for index in range(len(mean_candidates))}
        for repeat_index, repeat_rows in enumerate(repeat_candidates):
            for match in match_repeat_candidates(mean_candidates, repeat_rows):
                matched_rows_by_mean[match["mean_index"]].append(
                    {
                        **match,
                        "repeat_index": int(repeat_index),
                    }
                )
        for mean_index, candidate in enumerate(mean_candidates):
            peak_index = mean_index + 1
            matched_rows = matched_rows_by_mean[mean_index]
            matched_count = len(matched_rows)
            presence_fraction = matched_count / len(passed)
            if matched_count >= REPRODUCIBLE_REPEAT_MINIMUM:
                reproducible_counts[2] += 1
            if presence_fraction >= 0.5:
                reproducible_counts[50] += 1
            if presence_fraction >= 0.8:
                reproducible_counts[80] += 1
            matched_centers = [row["center_cm1"] for row in matched_rows]
            registry_rows.append(
                {
                    "subject_index_internal": int(subject_index),
                    "label": str(label),
                    "peak_index_within_subject": int(peak_index),
                    "peak_center_cm1": float(candidate["center_cm1"]),
                    "mean_peak_intensity": float(candidate["intensity"]),
                    "mean_peak_prominence": float(candidate["prominence"]),
                    "mean_peak_local_noise_sigma": float(
                        candidate["local_noise_sigma"]
                    ),
                    "mean_peak_local_snr": float(candidate["local_snr"]),
                    "mean_peak_repeat_variability_noise_sigma": float(
                        noise_sigma[int(candidate["index"])]
                    ),
                    "mean_peak_repeat_variability_snr": float(
                        candidate["prominence"]
                        / noise_sigma[int(candidate["index"])]
                    ),
                    "repeat_detection_count": int(matched_count),
                    "qc_repeat_count": int(len(passed)),
                    "repeat_presence_fraction": float(presence_fraction),
                    "repeat_center_mean_cm1": float(np.mean(matched_centers))
                    if matched_centers
                    else float("nan"),
                    "repeat_center_sd_cm1": float(np.std(matched_centers, ddof=1))
                    if len(matched_centers) > 1
                    else 0.0,
                    "repeat_center_range_cm1": float(
                        np.ptp(matched_centers)
                    )
                    if matched_centers
                    else float("nan"),
                    "dispersion_tolerance_cm1": PEAK_DISPERSION_TOLERANCE_CM1,
                    "snr_factor": float(snr_factor),
                    "reproducible_in_2plus_repeats": bool(
                        matched_count >= REPRODUCIBLE_REPEAT_MINIMUM
                    ),
                    "reproducible_in_50pct_repeats": bool(
                        presence_fraction >= 0.5
                    ),
                    "reproducible_in_80pct_repeats": bool(
                        presence_fraction >= 0.8
                    ),
                }
            )
        summary_rows.append(
            {
                "subject_index_internal": int(subject_index),
                "label": str(label),
                "qc_repeat_count": int(len(passed)),
                "mean_candidate_count": int(len(mean_candidates)),
                "subject_mean_peak_count": int(len(mean_candidates)),
                "repeat_reproducible_2plus_count": int(reproducible_counts[2]),
                "repeat_reproducible_50pct_count": int(reproducible_counts[50]),
                "repeat_reproducible_80pct_count": int(reproducible_counts[80]),
                "repeat_detection_count_total": matched_repeat_detection_count,
                "qc_distance_limit": float(limit),
                "qc_distance_median": float(np.median(distances)),
                "snr_factor": float(snr_factor),
                "dispersion_tolerance_cm1": PEAK_DISPERSION_TOLERANCE_CM1,
            }
        )
    return pd.DataFrame(registry_rows), pd.DataFrame(summary_rows)


def plot_subject_peak_summary(
    common_grid: np.ndarray,
    means: np.ndarray,
    labels: list[str],
    noise_vectors: np.ndarray,
    registry: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
) -> int:
    colors = {
        "control": "#457b9d",
        "prostate disease control": "#f4a261",
        "prostate": "#9d0208",
    }
    selected_row = summary.iloc[
        int(
            np.argmin(
                np.abs(
                    summary["subject_mean_peak_count"].to_numpy()
                    - summary["subject_mean_peak_count"].median()
                )
            )
        )
    ]
    subject_index = int(selected_row["subject_index_internal"])
    spectrum = means[subject_index]
    noise = noise_vectors[subject_index]
    selected_registry = registry[
        registry["subject_index_internal"] == subject_index
    ].sort_values("peak_center_cm1")
    fig, axes = plt.subplots(3, 1, figsize=(13, 12))
    color = colors[str(selected_row["label"])]
    axes[0].fill_between(
        common_grid,
        spectrum - noise,
        spectrum + noise,
        color="#6c757d",
        alpha=0.25,
        label="± repeat residual SD",
    )
    axes[0].plot(
        common_grid,
        spectrum,
        color=color,
        linewidth=1.2,
        label=f"subject {subject_index} mean spectrum",
    )
    for _, row in selected_registry.iterrows():
        fraction = float(row["repeat_presence_fraction"])
        line_color = "#2a9d8f" if fraction >= 0.5 else "#e76f51"
        axes[0].axvline(
            float(row["peak_center_cm1"]),
            color=line_color,
            alpha=0.65,
            linewidth=0.8,
        )
    axes[0].set_title(
        f"Subject-level mean spectrum: {int(selected_row['subject_mean_peak_count'])} peaks "
        f"(5 cm matching only across repeats)"
    )
    axes[0].set_ylabel("API intensity")
    axes[0].legend(frameon=False)

    for label, color in colors.items():
        subset = summary[summary["label"] == label]
        if subset.empty:
            continue
        axes[1].hist(
            subset["subject_mean_peak_count"],
            bins=np.arange(
                subset["subject_mean_peak_count"].min() - 0.5,
                subset["subject_mean_peak_count"].max() + 1.5,
                1,
            ),
            alpha=0.45,
            color=color,
            label=label,
        )
    axes[1].set_title("Subject-level peak count distribution")
    axes[1].set_xlabel("Peak count per subject mean spectrum")
    axes[1].set_ylabel("Subjects")
    axes[1].legend(frameon=False)

    axes[2].scatter(
        summary["subject_mean_peak_count"],
        summary["repeat_reproducible_50pct_count"],
        c=[colors[label] for label in summary["label"]],
        alpha=0.8,
        s=32,
    )
    axes[2].plot(
        [0, summary["subject_mean_peak_count"].max()],
        [0, summary["subject_mean_peak_count"].max()],
        color="#777777",
        linestyle="--",
        linewidth=0.9,
    )
    axes[2].set_title("All detected peaks vs peaks present in at least 50% of repeats")
    axes[2].set_xlabel("All subject-mean peaks")
    axes[2].set_ylabel("Repeat-reproducible peaks")
    for axis in axes:
        axis.grid(axis="y", alpha=0.18)
    fig.suptitle(
        "Subject-level peak detection; local-neighbor SNR, no cohort median or peak-count cap",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return subject_index


def plot_local_neighbor_validation(
    common_grid: np.ndarray,
    spectrum: np.ndarray,
    registry: pd.DataFrame,
    subject_index: int,
    path: Path,
) -> float:
    selected_registry = registry[
        registry["subject_index_internal"] == subject_index
    ].copy()
    target_row = selected_registry.iloc[
        int(np.argmin(np.abs(selected_registry["peak_center_cm1"].to_numpy() - 1000.0)))
    ]
    center = float(target_row["peak_center_cm1"])
    local_mask = (common_grid >= center - 24.0) & (common_grid <= center + 24.0)
    local_grid = common_grid[local_mask]
    local_spectrum = spectrum[local_mask]
    local_noise_sigma = float(target_row["mean_peak_local_noise_sigma"])
    prominence = float(target_row["mean_peak_prominence"])
    local_snr = float(target_row["mean_peak_local_snr"])
    difference_grid = (common_grid[:-1] + common_grid[1:]) / 2.0
    differences = np.diff(spectrum)
    difference_mask = (difference_grid >= center - 16.0) & (
        difference_grid <= center + 16.0
    )
    difference_noise_mask = (
        ((difference_grid >= center - LOCAL_NOISE_HALF_WINDOW_CM1) &
         (difference_grid <= center - LOCAL_NOISE_EXCLUSION_HALF_WIDTH_CM1))
        | ((difference_grid >= center + LOCAL_NOISE_EXCLUSION_HALF_WIDTH_CM1) &
           (difference_grid <= center + LOCAL_NOISE_HALF_WINDOW_CM1))
    )
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    axes[0].axvspan(
        center - LOCAL_NOISE_HALF_WINDOW_CM1,
        center - LOCAL_NOISE_EXCLUSION_HALF_WIDTH_CM1,
        color="#6c757d",
        alpha=0.20,
        label="nearby noise window",
    )
    axes[0].axvspan(
        center + LOCAL_NOISE_EXCLUSION_HALF_WIDTH_CM1,
        center + LOCAL_NOISE_HALF_WINDOW_CM1,
        color="#6c757d",
        alpha=0.20,
    )
    axes[0].axvspan(
        center - LOCAL_NOISE_EXCLUSION_HALF_WIDTH_CM1,
        center + LOCAL_NOISE_EXCLUSION_HALF_WIDTH_CM1,
        color="#2a9d8f",
        alpha=0.14,
        label="candidate peak window",
    )
    axes[0].fill_between(
        local_grid,
        local_spectrum - local_noise_sigma,
        local_spectrum + local_noise_sigma,
        color="#457b9d",
        alpha=0.14,
        label="± local noise σ",
    )
    axes[0].plot(
        local_grid,
        local_spectrum,
        color="#e76f51",
        linewidth=1.8,
        label="subject mean spectrum",
    )
    axes[0].axvline(center, color="#264653", linewidth=1.2)
    peak_index = int(np.argmin(np.abs(common_grid - center)))
    axes[0].scatter(
        [center],
        [spectrum[peak_index]],
        color="#264653",
        s=32,
        zorder=5,
    )
    axes[0].annotate(
        f"prominence={prominence:.1f}\nlocal noise σ={local_noise_sigma:.1f}\nlocal SNR={local_snr:.2f}",
        xy=(center, spectrum[peak_index]),
        xytext=(0.98, 0.94),
        textcoords="axes fraction",
        ha="right",
        va="top",
        arrowprops={"arrowstyle": "->", "color": "#264653"},
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    axes[0].set_title(
        f"Subject {subject_index}: local signal/noise windows around {center:.1f} cm$^{{-1}}$"
    )
    axes[0].set_ylabel("API intensity")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(alpha=0.18)
    axes[1].axhspan(
        -np.sqrt(2.0) * local_noise_sigma,
        np.sqrt(2.0) * local_noise_sigma,
        color="#6c757d",
        alpha=0.20,
        label="local noise band from first differences",
    )
    axes[1].scatter(
        difference_grid[difference_mask],
        differences[difference_mask],
        color="#adb5bd",
        s=20,
        label="nearby first differences",
    )
    axes[1].scatter(
        difference_grid[difference_noise_mask],
        differences[difference_noise_mask],
        color="#457b9d",
        s=24,
        label="differences used for noise",
    )
    axes[1].axvline(center, color="#264653", linewidth=1.2)
    axes[1].axhline(0.0, color="#777777", linewidth=0.8)
    axes[1].set_title("Noise is estimated from nearby first differences outside the candidate window")
    axes[1].set_xlabel("Raman shift (cm$^{-1}$)")
    axes[1].set_ylabel("Adjacent intensity difference")
    axes[1].legend(frameon=False, ncol=2)
    axes[1].grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return center


def main() -> None:
    pipeline = load_pipeline()
    items, _cohorts, _api_metadata = pipeline.load_api_spectra()
    common_grid, observed_step, common_points = pipeline.construct_common_grid(items)
    subject_keys, labels, replicate_matrices, alignment_metadata = (
        pipeline.group_and_align_subjects(items, common_grid)
    )
    all_registry: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []
    for snr_factor in SNR_FACTORS:
        registry, summary = build_subject_registry(
            pipeline,
            common_grid,
            subject_keys,
            labels,
            replicate_matrices,
            snr_factor,
        )
        registry.to_csv(
            OUTPUT_DIR / f"subject_peak_registry_snr{snr_factor:g}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        summary.to_csv(
            OUTPUT_DIR / f"subject_peak_count_summary_snr{snr_factor:g}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        all_registry.append(registry)
        all_summary.append(summary)
    registry_frame = pd.concat(all_registry, ignore_index=True)
    summary_frame = pd.concat(all_summary, ignore_index=True)
    registry_frame.to_csv(
        OUTPUT_DIR / "subject_peak_registry_all_snr_factors.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_frame.to_csv(
        OUTPUT_DIR / "subject_peak_count_summary_all_snr_factors.csv",
        index=False,
        encoding="utf-8-sig",
    )
    primary_registry = registry_frame[registry_frame["snr_factor"] == 3.0]
    primary_summary = summary_frame[summary_frame["snr_factor"] == 3.0]
    means = []
    noise_vectors = []
    for replicates in replicate_matrices:
        keep, _distances, _limit = pipeline.qc_repeats(replicates)
        passed = replicates[keep]
        representative = passed.mean(axis=0)
        means.append(representative)
        noise_vectors.append(
            np.maximum(
                np.std(passed - representative, axis=0, ddof=1),
                np.finfo(float).eps,
            )
        )
    mean_matrix = np.vstack(means)
    noise_matrix = np.vstack(noise_vectors)
    figure_path = FIGURE_DIR / "subject_peak_dispersion_summary.png"
    selected_subject = plot_subject_peak_summary(
        common_grid,
        mean_matrix,
        labels,
        noise_matrix,
        primary_registry,
        primary_summary,
        figure_path,
    )
    shutil.copy2(figure_path, PUBLICATION_FIGURE)
    local_validation_center = plot_local_neighbor_validation(
        common_grid,
        mean_matrix[selected_subject],
        primary_registry,
        selected_subject,
        LOCAL_VALIDATION_FIGURE,
    )
    shutil.copy2(LOCAL_VALIDATION_FIGURE, PUBLICATION_LOCAL_VALIDATION_FIGURE)
    count_rows = []
    for factor, summary in summary_frame.groupby("snr_factor"):
        count_rows.append(
            {
                "snr_factor": float(factor),
                "subjects": int(len(summary)),
                "total_subject_mean_peaks": int(summary["subject_mean_peak_count"].sum()),
                "mean_subject_peak_count": float(summary["subject_mean_peak_count"].mean()),
                "median_subject_peak_count": float(summary["subject_mean_peak_count"].median()),
                "min_subject_peak_count": int(summary["subject_mean_peak_count"].min()),
                "max_subject_peak_count": int(summary["subject_mean_peak_count"].max()),
                "total_reproducible_2plus": int(
                    summary["repeat_reproducible_2plus_count"].sum()
                ),
                "total_reproducible_50pct": int(
                    summary["repeat_reproducible_50pct_count"].sum()
                ),
                "total_reproducible_80pct": int(
                    summary["repeat_reproducible_80pct_count"].sum()
                ),
                "total_repeat_detections": int(
                    summary["repeat_detection_count_total"].sum()
                ),
            }
        )
    count_summary = pd.DataFrame(count_rows)
    count_summary.to_csv(
        OUTPUT_DIR / "subject_peak_total_count_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "data_source": "AECD REST API",
        "patent_pipeline": {
            "reference_alignment": True,
            "within_subject_qc_before_mean": True,
            "subject_mean_spectrum_used_for_peak_candidates": True,
            "repeat_residual_sd_used_for_post_detection_snr": True,
            "local_neighbor_noise_used_for_candidate_detection": True,
            "local_noise_window_cm1": LOCAL_NOISE_HALF_WINDOW_CM1,
            "local_noise_exclusion_half_width_cm1": LOCAL_NOISE_EXCLUSION_HALF_WIDTH_CM1,
            "cohort_median_used": False,
            "rolling_baseline_used": False,
            "fixed_intensity_prominence_used": False,
            "peak_count_cap_used": False,
        },
        "grid": {
            "range_cm1": list(pipeline.SPECTRAL_RANGE_CM1),
            "observed_step_cm1": observed_step,
            "common_grid_points": common_points,
        },
        "dispersion_tolerance_cm1": PEAK_DISPERSION_TOLERANCE_CM1,
        "snr_factors": list(SNR_FACTORS),
        "primary_snr_factor": 3.0,
        "primary_reproducibility_definitions": {
            "2plus_repeats": REPRODUCIBLE_REPEAT_MINIMUM,
            "50pct_repeats": 0.5,
            "80pct_repeats": 0.8,
        },
        "subjects": int(len(subject_keys)),
        "selected_visual_subject_index": selected_subject,
        "selected_local_validation_center_cm1": local_validation_center,
        "count_summary": count_summary.to_dict(orient="records"),
        "alignment_summary": alignment_metadata["wavenumber_calibration"],
    }
    (OUTPUT_DIR / "subject_peak_dispersion_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(count_summary.to_string(index=False))
    print(f"figure={figure_path}")
    print(f"publication_figure={PUBLICATION_FIGURE}")
    print(f"local_validation_figure={LOCAL_VALIDATION_FIGURE}")
    print(f"publication_local_validation_figure={PUBLICATION_LOCAL_VALIDATION_FIGURE}")


if __name__ == "__main__":
    main()
