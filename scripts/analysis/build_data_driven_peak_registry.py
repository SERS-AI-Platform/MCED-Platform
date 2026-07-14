#!/usr/bin/env python3
"""Build a data-driven SERS peak registry with explicit window criteria."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.stats import ttest_ind

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COHORT = PROJECT_ROOT / "results" / "kao_20260610_updated_cohort" / "processed_spectra.csv"
DEFAULT_OUT = PROJECT_ROOT / "results" / "model_revision" / "peak_registry_v1"

GROUP_ALIASES = {"CPAN": "PAN", "YPAN": "PAN", "YNOR": "NOR"}
CANCER_GROUPS = {"PRO", "BRE", "OVA", "LUN", "CRC", "BLC", "PAN"}
GROUP_ORDER = ("CONTROL", "BLC", "CRC", "LUN", "PAN", "PRO", "OVA", "BRE")
COLORS = {
    "CONTROL": "#666666",
    "PRO": "#8B4513",
    "BRE": "#D4527A",
    "OVA": "#6A5ACD",
    "LUN": "#2E8B57",
    "CRC": "#4682B4",
    "PAN": "#CD5C5C",
    "BLC": "#E8960C",
}


def _feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if str(c).startswith("x_")]
    if not cols:
        raise ValueError("No x_* spectral columns found")
    return sorted(cols, key=lambda c: float(str(c)[2:]))


def _load_subject_spectra(path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = pd.read_csv(path)
    feature_cols = _feature_columns(df)
    wavenumbers = np.asarray([float(c[2:]) for c in feature_cols], dtype=float)
    df["source_group"] = df["group"].astype(str).str.upper()
    df["sample_id_str"] = df["sample_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["subject_id"] = df["source_group"] + "_" + df["sample_id_str"]
    df["model_group"] = df["source_group"].replace(GROUP_ALIASES)
    df["display_group"] = np.where(
        df["model_group"].isin(CANCER_GROUPS), df["model_group"], "CONTROL"
    )
    meta_cols = ["subject_id", "source_group", "sample_id_str", "model_group", "display_group"]
    subject = df.groupby(meta_cols, dropna=False)[feature_cols].mean().reset_index()
    return subject, wavenumbers, feature_cols


def _odd_window(n: int, target: int) -> int:
    target = min(target, n if n % 2 == 1 else n - 1)
    target = max(target, 5)
    if target % 2 == 0:
        target -= 1
    return target


def _smooth(y: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    win = _odd_window(len(y), window)
    poly = min(polyorder, win - 2)
    return savgol_filter(y, win, poly)


def _robust_sigma(residual: np.ndarray) -> float:
    residual = np.asarray(residual, dtype=float)
    med = np.median(residual)
    mad = np.median(np.abs(residual - med))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(residual))
    return float(max(sigma, 1e-8))


def _bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    q = np.full_like(pvalues, np.nan, dtype=float)
    valid = np.isfinite(pvalues)
    if not valid.any():
        return q
    p = pvalues[valid]
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = adj
    q[valid] = out
    return q


def _window_feature(
    X: np.ndarray, wavenumbers: np.ndarray, left: float, right: float
) -> np.ndarray:
    mask = (wavenumbers >= left) & (wavenumbers <= right)
    if mask.sum() == 0:
        idx = int(np.argmin(np.abs(wavenumbers - (left + right) / 2.0)))
        mask[idx] = True
    return X[:, mask].max(axis=1)


def _subject_peak_centers(
    X: np.ndarray,
    wavenumbers: np.ndarray,
    *,
    smooth_window: int,
    polyorder: int,
    snr_threshold: float,
    min_peak_distance_cm1: float,
    min_width_cm1: float,
    max_width_cm1: float,
) -> list[np.ndarray]:
    step = float(np.median(np.diff(wavenumbers)))
    distance_pts = max(1, int(round(min_peak_distance_cm1 / step)))
    centers: list[np.ndarray] = []
    for row in X:
        ys = _smooth(row, smooth_window, polyorder)
        sigma = _robust_sigma(row - ys)
        peaks, props = find_peaks(ys, prominence=snr_threshold * sigma, distance=distance_pts)
        if len(peaks) == 0:
            centers.append(np.asarray([], dtype=float))
            continue
        widths = peak_widths(ys, peaks, rel_height=0.5)
        left = np.interp(widths[2], np.arange(len(wavenumbers)), wavenumbers)
        right = np.interp(widths[3], np.arange(len(wavenumbers)), wavenumbers)
        width_cm = right - left
        keep = (width_cm >= min_width_cm1) & (width_cm <= max_width_cm1)
        centers.append(wavenumbers[peaks[keep]])
    return centers


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / (len(a) + len(b) - 2)
    )
    if pooled <= 0 or not np.isfinite(pooled):
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def build_registry(
    cohort_csv: Path,
    out_dir: Path,
    *,
    smooth_window: int = 21,
    polyorder: int = 3,
    snr_threshold: float = 3.0,
    min_reproducibility: float = 0.30,
    min_peak_distance_cm1: float = 35.0,
    min_width_cm1: float = 6.0,
    max_width_cm1: float = 90.0,
    min_window_cm1: float = 16.0,
    max_window_cm1: float = 90.0,
    window_expansion_fraction: float = 0.20,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    subject, wavenumbers, feature_cols = _load_subject_spectra(cohort_csv)
    X_all = subject[feature_cols].to_numpy(dtype=float)
    control_mask = subject["display_group"].to_numpy() == "CONTROL"
    X_control = X_all[control_mask]

    criteria = {
        "smoothing": {
            "method": "Savitzky-Golay",
            "window_points": smooth_window,
            "polyorder": polyorder,
        },
        "noise": "robust MAD sigma from spectrum minus smoothed spectrum",
        "subject_peak_exists_if": {
            "local_maximum": True,
            "prominence_snr_min": snr_threshold,
            "min_width_cm-1": min_width_cm1,
            "max_width_cm-1": max_width_cm1,
        },
        "group_peak_accepted_if": {
            "group_mean_prominence_snr_min": snr_threshold,
            "subject_reproducibility_min": min_reproducibility,
            "min_peak_distance_cm-1": min_peak_distance_cm1,
        },
        "window_rule": {
            "base": "group mean half-prominence width",
            "expansion_fraction_each_side": window_expansion_fraction,
            "min_window_cm-1": min_window_cm1,
            "max_window_cm-1": max_window_cm1,
        },
        "statistics": {
            "window_feature": "maximum processed intensity within accepted window",
            "test": "Welch t-test versus CONTROL for cancer groups",
            "multiplicity": "Benjamini-Hochberg over all tested cancer-group windows",
        },
    }

    rows: list[dict[str, object]] = []
    all_presence_rows: list[dict[str, object]] = []
    step = float(np.median(np.diff(wavenumbers)))
    distance_pts = max(1, int(round(min_peak_distance_cm1 / step)))

    for group in GROUP_ORDER:
        group_mask = subject["display_group"].to_numpy() == group
        Xg = X_all[group_mask]
        if len(Xg) < 2:
            continue

        group_mean = Xg.mean(axis=0)
        group_smooth = _smooth(group_mean, smooth_window, polyorder)
        subject_smoothed = np.vstack([_smooth(row, smooth_window, polyorder) for row in Xg])
        subject_sigmas = np.asarray(
            [_robust_sigma(row - sm) for row, sm in zip(Xg, subject_smoothed)]
        )
        group_sigma = float(np.median(subject_sigmas) / np.sqrt(len(Xg)))
        prominence_threshold = snr_threshold * group_sigma

        peaks, props = find_peaks(
            group_smooth, prominence=prominence_threshold, distance=distance_pts
        )
        if len(peaks) == 0:
            continue
        widths = peak_widths(group_smooth, peaks, rel_height=0.5)
        left_base = np.interp(widths[2], np.arange(len(wavenumbers)), wavenumbers)
        right_base = np.interp(widths[3], np.arange(len(wavenumbers)), wavenumbers)
        subject_centers = _subject_peak_centers(
            Xg,
            wavenumbers,
            smooth_window=smooth_window,
            polyorder=polyorder,
            snr_threshold=snr_threshold,
            min_peak_distance_cm1=min_peak_distance_cm1,
            min_width_cm1=min_width_cm1,
            max_width_cm1=max_width_cm1,
        )

        for local_rank, peak_idx in enumerate(peaks, start=1):
            center = float(wavenumbers[peak_idx])
            width = float(right_base[local_rank - 1] - left_base[local_rank - 1])
            if width < min_width_cm1 or width > max_width_cm1:
                continue
            expand = max(step, width * window_expansion_fraction)
            left = float(left_base[local_rank - 1] - expand)
            right = float(right_base[local_rank - 1] + expand)
            window_width = right - left
            if window_width < min_window_cm1:
                left = center - min_window_cm1 / 2.0
                right = center + min_window_cm1 / 2.0
            elif window_width > max_window_cm1:
                left = center - max_window_cm1 / 2.0
                right = center + max_window_cm1 / 2.0
            left = max(float(wavenumbers.min()), left)
            right = min(float(wavenumbers.max()), right)

            present = np.asarray(
                [np.any((centers >= left) & (centers <= right)) for centers in subject_centers]
            )
            reproducibility = float(present.mean())
            group_snr = float(props["prominences"][local_rank - 1] / group_sigma)
            accepted = group_snr >= snr_threshold and reproducibility >= min_reproducibility

            values_group = _window_feature(Xg, wavenumbers, left, right)
            if group == "CONTROL":
                pvalue = float("nan")
                effect = float("nan")
                control_mean = float(np.mean(values_group))
            else:
                values_control = _window_feature(X_control, wavenumbers, left, right)
                pvalue = float(
                    ttest_ind(
                        values_group, values_control, equal_var=False, nan_policy="omit"
                    ).pvalue
                )
                effect = _cohen_d(values_group, values_control)
                control_mean = float(np.mean(values_control))

            peak_id = f"{group}_P{len(rows) + 1:03d}"
            rows.append(
                {
                    "peak_id": peak_id,
                    "display_group": group,
                    "center_cm-1": center,
                    "window_left_cm-1": float(left),
                    "window_right_cm-1": float(right),
                    "half_prominence_width_cm-1": width,
                    "group_mean_prominence": float(props["prominences"][local_rank - 1]),
                    "group_noise_sigma": group_sigma,
                    "group_mean_snr": group_snr,
                    "n_subjects_group": int(len(Xg)),
                    "subject_reproducibility": reproducibility,
                    "accepted_peak": bool(accepted),
                    "window_feature_mean_group": float(np.mean(values_group)),
                    "window_feature_mean_control": control_mean,
                    "welch_p_vs_control": pvalue,
                    "cohen_d_vs_control": effect,
                }
            )
            group_subjects = subject.loc[group_mask, ["subject_id", "display_group"]].reset_index(
                drop=True
            )
            for subj, is_present in zip(group_subjects["subject_id"], present):
                all_presence_rows.append(
                    {
                        "peak_id": peak_id,
                        "subject_id": subj,
                        "display_group": group,
                        "peak_present": bool(is_present),
                    }
                )

    registry = pd.DataFrame(rows)
    if registry.empty:
        raise RuntimeError("No peaks discovered with the current criteria")
    registry["bh_q_vs_control"] = _bh_qvalues(registry["welch_p_vs_control"].to_numpy(dtype=float))
    registry["disease_discriminating_peak"] = (
        registry["accepted_peak"]
        & registry["display_group"].ne("CONTROL")
        & registry["bh_q_vs_control"].lt(0.05)
    )
    registry = registry.sort_values(["display_group", "center_cm-1"]).reset_index(drop=True)
    registry.to_csv(out_dir / "peak_registry.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_presence_rows).to_csv(
        out_dir / "peak_presence_by_subject.csv", index=False, encoding="utf-8-sig"
    )
    (out_dir / "peak_registry_criteria.json").write_text(
        json.dumps(criteria, indent=2), encoding="utf-8"
    )

    plot_peak_registry(subject, feature_cols, wavenumbers, registry, out_dir)
    print(f"Output: {out_dir}")
    print(registry.groupby("display_group")["accepted_peak"].sum().to_string())


def plot_peak_registry(
    subject: pd.DataFrame,
    feature_cols: list[str],
    wavenumbers: np.ndarray,
    registry: pd.DataFrame,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(11.4, 10.2), sharex=True)
    axes_flat = axes.ravel()
    for ax, group in zip(axes_flat, GROUP_ORDER):
        sub = subject[subject["display_group"] == group]
        if sub.empty:
            ax.axis("off")
            continue
        X = sub[feature_cols].to_numpy(dtype=float)
        mean = X.mean(axis=0)
        sem = X.std(axis=0, ddof=1) / np.sqrt(len(X)) if len(X) > 1 else np.zeros_like(mean)
        color = COLORS.get(group, "#333333")
        ax.fill_between(
            wavenumbers, mean - 1.96 * sem, mean + 1.96 * sem, color=color, alpha=0.14, linewidth=0
        )
        ax.plot(wavenumbers, mean, color=color, lw=1.1)
        peaks = registry[(registry["display_group"] == group) & (registry["accepted_peak"])]
        for _, row in peaks.iterrows():
            left = float(row["window_left_cm-1"])
            right = float(row["window_right_cm-1"])
            center = float(row["center_cm-1"])
            ax.axvspan(left, right, color=color, alpha=0.09, linewidth=0)
            ax.text(
                center,
                ax.get_ylim()[0],
                f"{center:.0f}",
                rotation=90,
                va="bottom",
                ha="center",
                fontsize=6,
                color=color,
            )
        ax.set_title(
            f"{group} accepted peak windows (n={len(sub)})",
            loc="left",
            fontsize=9,
            fontweight="bold",
            color=color,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, color="#eeeeee", linewidth=0.45)
        ax.tick_params(labelsize=7)
        ax.set_ylabel("Intensity (a.u.)", fontsize=8)
    for ax in axes_flat[len(GROUP_ORDER) :]:
        ax.axis("off")
    for ax in axes_flat[-2:]:
        ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=8)
    fig.suptitle("Data-driven peak registry windows", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"peak_registry_windows.{ext}", dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-csv", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smooth-window", type=int, default=21)
    parser.add_argument("--polyorder", type=int, default=3)
    parser.add_argument("--snr-threshold", type=float, default=3.0)
    parser.add_argument("--min-reproducibility", type=float, default=0.30)
    parser.add_argument("--min-peak-distance-cm1", type=float, default=35.0)
    parser.add_argument("--min-width-cm1", type=float, default=6.0)
    parser.add_argument("--max-width-cm1", type=float, default=90.0)
    parser.add_argument("--min-window-cm1", type=float, default=16.0)
    parser.add_argument("--max-window-cm1", type=float, default=90.0)
    args = parser.parse_args()
    build_registry(
        args.cohort_csv.resolve(),
        args.out_dir.resolve(),
        smooth_window=args.smooth_window,
        polyorder=args.polyorder,
        snr_threshold=args.snr_threshold,
        min_reproducibility=args.min_reproducibility,
        min_peak_distance_cm1=args.min_peak_distance_cm1,
        min_width_cm1=args.min_width_cm1,
        max_width_cm1=args.max_width_cm1,
        min_window_cm1=args.min_window_cm1,
        max_window_cm1=args.max_window_cm1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
