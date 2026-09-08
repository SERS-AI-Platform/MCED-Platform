#!/usr/bin/env python3
"""Plot sample replicates with trimming only.

This does exactly one transformation: keep rows whose x value is within the
requested range. It does not interpolate, average, normalize, smooth, baseline
correct, reference-correct, blank-correct, or compute centroids.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.trim_only_raw_utils import (  # noqa: E402
    DEFAULT_CLINICAL_ROOT,
    DEFAULT_PRIMARY_ROOT,
    date_key_from_path,
    parse_sample,
    read_trimmed_xy,
)

DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "raw_data_vs_clinical_reference"
    / "figures"
    / "trim_only_sample_reports"
)


def collect_files(root: Path, group: str, sample_id: str, date_key: str | None) -> list[Path]:
    paths = sorted(list(root.rglob("*.[Cc][Ss][Vv]")) + list(root.rglob("*.txt")))
    out: list[Path] = []
    for path in paths:
        key = parse_sample(path)
        if key is None:
            continue
        if key.group != group or key.sample_id != sample_id:
            continue
        if date_key is not None and date_key_from_path(path) != date_key:
            continue
        out.append(path)
    return out


def read_trimmed(
    paths: list[Path], x_min: float, x_max: float
) -> list[tuple[Path, np.ndarray, np.ndarray]]:
    rows = []
    for path in paths:
        x, y = read_trimmed_xy(path, x_min, x_max)
        rows.append((path, x, y))
    return rows


def plot_trim_only(
    primary_rows: list[tuple[Path, np.ndarray, np.ndarray]],
    clinical_rows: list[tuple[Path, np.ndarray, np.ndarray]],
    out_dir: Path,
    label: str,
    x_min: float,
    x_max: float,
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    panels = [
        (axes[0], primary_rows, "primary raw_data: CSV values after trim only"),
        (axes[1], clinical_rows, "clinical reacquired: CSV values after trim only"),
    ]
    for ax, rows, title in panels:
        for path, x, y in rows:
            ax.plot(x, y, linewidth=0.9, alpha=0.85, label=path.name)
            neg = y < 0
            if np.any(neg):
                ax.scatter(x[neg], y[neg], s=5, alpha=0.35)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel("CSV intensity")
        ax.legend(fontsize=7, ncol=3)
    axes[1].set_xlabel("CSV Raman shift")

    fig.suptitle(
        f"{label} | TRIM ONLY x=[{x_min:g}, {x_max:g}] | no centroid, no interpolation, no preprocessing",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{label}_trim_only_replicates.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_trim_only_same_axis(
    primary_rows: list[tuple[Path, np.ndarray, np.ndarray]],
    clinical_rows: list[tuple[Path, np.ndarray, np.ndarray]],
    out_dir: Path,
    label: str,
    x_min: float,
    x_max: float,
) -> Path:
    all_y = np.concatenate([y for _, _, y in primary_rows + clinical_rows if len(y)])
    y_min = float(np.min(all_y))
    y_max = float(np.max(all_y))
    pad = max((y_max - y_min) * 0.05, 1.0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    panels = [
        (axes[0], primary_rows, "primary raw_data: same y-axis"),
        (axes[1], clinical_rows, "clinical reacquired: same y-axis"),
    ]
    for ax, rows, title in panels:
        for path, x, y in rows:
            ax.plot(x, y, linewidth=0.9, alpha=0.85, label=path.name)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_title(title)
        ax.set_ylabel("CSV intensity")
        ax.legend(fontsize=7, ncol=3)
    axes[1].set_xlabel("CSV Raman shift")
    fig.suptitle(
        f"{label} | TRIM ONLY same y-axis | x=[{x_min:g}, {x_max:g}] | no preprocessing",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{label}_trim_only_same_y_axis.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def write_file_stats(
    primary_rows: list[tuple[Path, np.ndarray, np.ndarray]],
    clinical_rows: list[tuple[Path, np.ndarray, np.ndarray]],
    out_dir: Path,
    label: str,
    x_min: float,
    x_max: float,
) -> Path:
    rows = []
    for dataset, spectra in [
        ("primary_raw_data", primary_rows),
        ("clinical_reacquired", clinical_rows),
    ]:
        for path, x, y in spectra:
            n_negative = int(np.sum(y < 0)) if len(y) else 0
            rows.append(
                {
                    "dataset": dataset,
                    "path": str(path),
                    "trim_x_min_requested": x_min,
                    "trim_x_max_requested": x_max,
                    "n_points_after_trim": len(x),
                    "n_negative_after_trim": n_negative,
                    "negative_fraction_after_trim": float(n_negative / len(y))
                    if len(y)
                    else np.nan,
                    "has_negative_after_trim": bool(n_negative > 0),
                    "x_min_after_trim": float(np.min(x)) if len(x) else np.nan,
                    "x_max_after_trim": float(np.max(x)) if len(x) else np.nan,
                    "y_min_after_trim": float(np.min(y)) if len(y) else np.nan,
                    "y_max_after_trim": float(np.max(y)) if len(y) else np.nan,
                    "y_mean_after_trim": float(np.mean(y)) if len(y) else np.nan,
                    "y_median_after_trim": float(np.median(y)) if len(y) else np.nan,
                }
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{label}_trim_only_file_stats.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, default=DEFAULT_PRIMARY_ROOT)
    parser.add_argument("--clinical-root", type=Path, default=DEFAULT_CLINICAL_ROOT)
    parser.add_argument("--group", default="CRC")
    parser.add_argument("--sample-id", default="286")
    parser.add_argument("--date-key", default="20260518")
    parser.add_argument("--trim-min", type=float, default=400.0)
    parser.add_argument("--trim-max", type=float, default=1800.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    group = args.group.upper()
    sample_id = str(int(args.sample_id))
    label = f"{group}_{sample_id}_{args.date_key}"

    primary_files = collect_files(args.primary_root, group, sample_id, None)
    clinical_files = collect_files(args.clinical_root, group, sample_id, args.date_key)
    if not primary_files:
        raise RuntimeError(f"No primary files found for {group} {sample_id}")
    if not clinical_files:
        raise RuntimeError(f"No clinical files found for {group} {sample_id} {args.date_key}")

    primary_rows = read_trimmed(primary_files, args.trim_min, args.trim_max)
    clinical_rows = read_trimmed(clinical_files, args.trim_min, args.trim_max)
    fig_path = plot_trim_only(
        primary_rows, clinical_rows, args.output_dir, label, args.trim_min, args.trim_max
    )
    same_axis_fig_path = plot_trim_only_same_axis(
        primary_rows,
        clinical_rows,
        args.output_dir,
        label,
        args.trim_min,
        args.trim_max,
    )
    stats_path = write_file_stats(
        primary_rows, clinical_rows, args.output_dir, label, args.trim_min, args.trim_max
    )

    print(f"Figure: {fig_path}")
    print(f"Same-axis figure: {same_axis_fig_path}")
    print(f"Stats: {stats_path}")
    stats = pd.read_csv(stats_path)
    print(
        stats[
            [
                "dataset",
                "path",
                "n_points_after_trim",
                "n_negative_after_trim",
                "negative_fraction_after_trim",
                "y_min_after_trim",
                "y_max_after_trim",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
