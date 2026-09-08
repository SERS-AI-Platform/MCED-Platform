#!/usr/bin/env python3
"""Create trim-only native Nanoscope raw visualizations.

Reads two-column txt files directly, keeps only rows in the requested x range,
and plots the values without sorting, interpolation, smoothing, baseline
correction, normalization, averaging, or background subtraction.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NANOSCOPE_ROOT = PROJECT_ROOT / "data" / "equipment_test_data" / "nanoscope" / "1. NOR"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "equipment_test_data" / "nanoscope_trim_only"
FILENAME_RE = re.compile(r"(?i)^NOR\s+(?P<subject>\d+)_(?P<point>\d+)\.txt$")


def parse_nanoscope_file(path: Path) -> tuple[int, int] | None:
    if "_ave" in path.name.lower():
        return None
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    return int(match.group("subject")), int(match.group("point"))


def collect_files(root: Path) -> dict[int, list[tuple[int, Path]]]:
    by_subject: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for path in sorted(root.glob("*.txt")):
        parsed = parse_nanoscope_file(path)
        if parsed is None:
            continue
        subject_id, point_id = parsed
        by_subject[subject_id].append((point_id, path))
    return {subject_id: sorted(rows) for subject_id, rows in sorted(by_subject.items())}


def read_trimmed_xy(path: Path, x_min: float, x_max: float) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(
        path, sep=r"\s+", header=None, usecols=[0, 1], names=["x", "y"], engine="python"
    )
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    mask = (x >= x_min) & (x <= x_max)
    return x[mask], y[mask]


def read_rows(
    files: list[tuple[int, Path]], x_min: float, x_max: float
) -> list[tuple[int, Path, np.ndarray, np.ndarray]]:
    rows = []
    for point_id, path in files:
        x, y = read_trimmed_xy(path, x_min, x_max)
        rows.append((point_id, path, x, y))
    return rows


def concat_y(rows: list[tuple[int, Path, np.ndarray, np.ndarray]]) -> np.ndarray:
    ys = [y for _, _, _, y in rows if len(y)]
    return np.concatenate(ys) if ys else np.array([])


def stat_block(prefix: str, rows: list[tuple[int, Path, np.ndarray, np.ndarray]]) -> dict:
    y = concat_y(rows)
    return {
        f"{prefix}_n_files": len(rows),
        f"{prefix}_n_points_after_trim": int(sum(len(row_y) for _, _, _, row_y in rows)),
        f"{prefix}_negative_fraction_after_trim": float(np.mean(y < 0)) if len(y) else np.nan,
        f"{prefix}_y_min_after_trim": float(np.min(y)) if len(y) else np.nan,
        f"{prefix}_y_max_after_trim": float(np.max(y)) if len(y) else np.nan,
        f"{prefix}_abs_y_p95_after_trim": float(np.percentile(np.abs(y), 95)) if len(y) else np.nan,
    }


def file_stats(
    dataset: str, subject_id: int, rows: list[tuple[int, Path, np.ndarray, np.ndarray]]
) -> list[dict]:
    out = []
    for point_id, path, x, y in rows:
        n_neg = int(np.sum(y < 0)) if len(y) else 0
        out.append(
            {
                "dataset": dataset,
                "subject_id": subject_id,
                "point_id": point_id,
                "path": str(path),
                "n_points_after_trim": int(len(y)),
                "n_negative_after_trim": n_neg,
                "negative_fraction_after_trim": float(n_neg / len(y)) if len(y) else np.nan,
                "has_negative_after_trim": bool(n_neg > 0),
                "x_min_after_trim": float(np.min(x)) if len(x) else np.nan,
                "x_max_after_trim": float(np.max(x)) if len(x) else np.nan,
                "y_min_after_trim": float(np.min(y)) if len(y) else np.nan,
                "y_max_after_trim": float(np.max(y)) if len(y) else np.nan,
                "abs_y_p95_after_trim": float(np.percentile(np.abs(y), 95)) if len(y) else np.nan,
            }
        )
    return out


def summarize_subject(
    subject_id: int,
    sample_rows: list[tuple[int, Path, np.ndarray, np.ndarray]],
    background_rows: list[tuple[int, Path, np.ndarray, np.ndarray]],
) -> dict:
    row = {"subject_id": subject_id}
    row.update(stat_block("sample", sample_rows))
    row.update(stat_block("background", background_rows))
    sample_p95 = row["sample_abs_y_p95_after_trim"]
    background_p95 = row["background_abs_y_p95_after_trim"]
    ratio = background_p95 / sample_p95 if np.isfinite(sample_p95) and sample_p95 > 0 else np.nan
    row["background_to_sample_abs_y_p95_ratio_after_trim"] = ratio
    row["log10_background_to_sample_abs_y_p95_ratio_after_trim"] = (
        float(np.log10(ratio)) if np.isfinite(ratio) and ratio > 0 else np.nan
    )
    return row


def plot_subject_report(
    subject_id: int,
    sample_rows: list[tuple[int, Path, np.ndarray, np.ndarray]],
    background_rows: list[tuple[int, Path, np.ndarray, np.ndarray]],
    out_dir: Path,
    x_min: float,
    x_max: float,
    mark_negatives: bool,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"NOR_{subject_id:03d}_nanoscope_trim_only.png"
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    panels = [
        (axes[0], sample_rows, "nanoscope sample txt: own y-axis", "#436f9f"),
        (axes[1], background_rows, "nanoscope background txt: own y-axis", "#9a6b3f"),
    ]
    for ax, rows, title, color in panels:
        if not rows:
            ax.text(0.5, 0.5, "no files", transform=ax.transAxes, ha="center", va="center")
        for point_id, path, x, y in rows:
            ax.plot(x, y, linewidth=0.7, alpha=0.70, color=color, label=f"{point_id}")
            if mark_negatives and len(y):
                neg = y < 0
                if np.any(neg):
                    ax.scatter(x[neg], y[neg], s=3, alpha=0.35, color="#b33a3a", linewidths=0)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("txt intensity")

    ax = axes[2]
    for _, _, x, y in sample_rows:
        ax.plot(x, y, linewidth=0.65, alpha=0.60, color="#436f9f")
    for _, _, x, y in background_rows:
        ax.plot(x, y, linewidth=0.65, alpha=0.45, color="#9a6b3f")
    all_y = [y for _, _, _, y in sample_rows + background_rows if len(y)]
    if all_y:
        y_all = np.concatenate(all_y)
        y_min = float(np.min(y_all))
        y_max = float(np.max(y_all))
        pad = max((y_max - y_min) * 0.05, 1.0)
        ax.set_ylim(y_min - pad, y_max + pad)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title("sample + background overlaid: same y-axis", fontsize=10)
    ax.set_ylabel("txt intensity")
    ax.set_xlabel("txt Raman shift")
    fig.suptitle(
        f"Nanoscope NOR {subject_id} | TRIM ONLY x=[{x_min:g}, {x_max:g}] | native txt values",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_contact_sheet(
    items: list[dict], out_path: Path, x_min: float, x_max: float, n_cols: int
) -> None:
    n_rows = math.ceil(len(items) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3.0 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, item in zip(axes.ravel(), items):
        ax.axis("on")
        for _, _, x, y in item["sample_rows"]:
            ax.plot(x, y, linewidth=0.45, alpha=0.58, color="#436f9f")
        for _, _, x, y in item["background_rows"]:
            ax.plot(x, y, linewidth=0.45, alpha=0.35, color="#9a6b3f")
        ax.axhline(0, color="black", linewidth=0.45)
        ax.set_xlim(x_min, x_max)
        ax.set_title(f"NOR {item['subject_id']}", fontsize=8)
        ax.tick_params(axis="both", labelsize=6, length=2)
    fig.suptitle(
        f"Nanoscope TRIM ONLY contact sheet | x=[{x_min:g}, {x_max:g}] | blue=sample, brown=background | no preprocessing",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_summary(subject_summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = subject_summary.sort_values("subject_id")
    x = summary["subject_id"].to_numpy()
    paths = []

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    axes[0].scatter(
        x, summary["log10_background_to_sample_abs_y_p95_ratio_after_trim"], s=22, color="#5f5f5f"
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("log10 background / sample\nabs intensity p95")
    axes[0].set_title("Nanoscope trim-only sample/background intensity scale")

    axes[1].scatter(
        x,
        summary["sample_negative_fraction_after_trim"],
        s=18,
        color="#436f9f",
        alpha=0.70,
        label="sample",
    )
    axes[1].scatter(
        x,
        summary["background_negative_fraction_after_trim"],
        s=18,
        color="#9a6b3f",
        alpha=0.70,
        label="background",
    )
    axes[1].set_ylabel("negative fraction")
    axes[1].legend(fontsize=8)

    axes[2].scatter(
        x, summary["sample_abs_y_p95_after_trim"], s=18, color="#436f9f", alpha=0.70, label="sample"
    )
    axes[2].scatter(
        x,
        summary["background_abs_y_p95_after_trim"],
        s=18,
        color="#9a6b3f",
        alpha=0.70,
        label="background",
    )
    axes[2].set_yscale("log")
    axes[2].set_ylabel("abs intensity p95")
    axes[2].set_xlabel("NOR subject")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "nanoscope_trim_only_metric_overview.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    values = summary["log10_background_to_sample_abs_y_p95_ratio_after_trim"].dropna().to_numpy()
    ax.boxplot([values], tick_labels=["Nanoscope"])
    ax.scatter(
        np.ones(len(values)) + np.linspace(-0.12, 0.12, len(values)),
        values,
        s=18,
        alpha=0.45,
        color="#5f5f5f",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("log10 background / sample abs intensity p95")
    ax.set_title("Nanoscope trim-only background-to-sample intensity ratio")
    fig.tight_layout()
    path = out_dir / "nanoscope_trim_only_background_ratio_boxplot.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nanoscope-root", type=Path, default=DEFAULT_NANOSCOPE_ROOT)
    parser.add_argument("--background-root", type=Path, default=None)
    parser.add_argument("--trim-min", type=float, default=400.0)
    parser.add_argument("--trim-max", type=float, default=1800.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-reports", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--contact-sheets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--contact-sheet-cols", type=int, default=3)
    parser.add_argument("--contact-sheet-page-size", type=int, default=12)
    parser.add_argument("--mark-negatives", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    background_root = args.background_root or args.nanoscope_root / "Background"
    sample_files = collect_files(args.nanoscope_root)
    background_files = collect_files(background_root) if background_root.exists() else {}
    if not sample_files:
        raise RuntimeError(f"No Nanoscope sample txt files found in {args.nanoscope_root}")

    sample_report_dir = args.output_dir / "sample_reports"
    contact_sheet_dir = args.output_dir / "contact_sheets"
    summary_dir = args.output_dir / "summary"
    subject_rows = []
    file_rows = []
    contact_items = []
    sample_report_paths = []

    for idx, subject_id in enumerate(sorted(sample_files), start=1):
        sample_rows = read_rows(sample_files[subject_id], args.trim_min, args.trim_max)
        background_rows = read_rows(
            background_files.get(subject_id, []), args.trim_min, args.trim_max
        )
        subject_rows.append(summarize_subject(subject_id, sample_rows, background_rows))
        file_rows.extend(file_stats("nanoscope_sample", subject_id, sample_rows))
        file_rows.extend(file_stats("nanoscope_background", subject_id, background_rows))
        if args.sample_reports:
            sample_report_paths.append(
                plot_subject_report(
                    subject_id,
                    sample_rows,
                    background_rows,
                    sample_report_dir,
                    args.trim_min,
                    args.trim_max,
                    args.mark_negatives,
                )
            )
        if args.contact_sheets:
            contact_items.append(
                {
                    "subject_id": subject_id,
                    "sample_rows": sample_rows,
                    "background_rows": background_rows,
                }
            )
        if idx % 25 == 0:
            print(f"Processed {idx}/{len(sample_files)} Nanoscope subjects")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subject_summary_path = args.output_dir / "nanoscope_trim_only_subject_summary.csv"
    file_stats_path = args.output_dir / "nanoscope_trim_only_file_stats.csv"
    subject_summary = pd.DataFrame(subject_rows)
    subject_summary.to_csv(subject_summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(file_rows).to_csv(file_stats_path, index=False, encoding="utf-8-sig")

    summary_paths = plot_summary(subject_summary, summary_dir)
    contact_paths = []
    if args.contact_sheets:
        page_size = args.contact_sheet_page_size
        for page_idx in range(0, len(contact_items), page_size):
            page = contact_items[page_idx : page_idx + page_size]
            out_path = (
                contact_sheet_dir
                / f"nanoscope_trim_only_contact_sheet_{page_idx // page_size + 1:03d}.png"
            )
            plot_contact_sheet(
                page, out_path, args.trim_min, args.trim_max, args.contact_sheet_cols
            )
            contact_paths.append(out_path)

    manifest_path = args.output_dir / "nanoscope_trim_only_outputs_manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as fh:
        fh.write(f"nanoscope_root={args.nanoscope_root}\n")
        fh.write(f"background_root={background_root}\n")
        fh.write(f"subjects={len(sample_files)}\n")
        fh.write(f"sample_reports={len(sample_report_paths)}\n")
        fh.write(f"contact_sheets={len(contact_paths)}\n")
        fh.write(f"subject_summary={subject_summary_path}\n")
        fh.write(f"file_stats={file_stats_path}\n")
        for path in summary_paths:
            fh.write(f"summary_figure={path}\n")
        for path in contact_paths:
            fh.write(f"contact_sheet={path}\n")
        for path in sample_report_paths:
            fh.write(f"sample_report={path}\n")

    print(f"Nanoscope subjects: {len(sample_files)}")
    print(f"Sample reports: {len(sample_report_paths)}")
    print(f"Contact sheets: {len(contact_paths)}")
    print(f"Subject summary: {subject_summary_path}")
    print(f"File stats: {file_stats_path}")
    print(f"Manifest: {manifest_path}")
    for path in summary_paths:
        print(f"Summary figure: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
