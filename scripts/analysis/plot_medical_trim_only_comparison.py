#!/usr/bin/env python3
"""Compare raw_data_medical vs Medical using strict trim-only native txt values.

This is the Medical-instrument counterpart of the Thermo raw-vs-reacquired
trim-only comparison. It reads two-column txt/csv files directly and applies
only an x-range trim. It does not sort, interpolate, smooth, baseline-correct,
normalize, average, reference-correct, blank-correct, or background-subtract.
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
DEFAULT_PRIMARY_ROOT = PROJECT_ROOT / "data" / "raw_data_medical"
DEFAULT_MEDICAL_ROOT = PROJECT_ROOT / "data" / "Medical"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "medical_vs_raw_data_medical" / "trim_only"

SAMPLE_RE = re.compile(
    r"(?i)^(?P<group>BNOR|YNOR|YPAN|CPAN|SPAN|BLC|BRE|CRC|DIA|H\.?\s?D\.?|HBP|LUN|NOR|OVA|PRO|PAN|HD)"
    r"\s+(?P<num>\d+)_(?P<rep>\d+|ave)(?:\(\d+\))?(?:_Sample_.*)?\.(?:csv|txt)$"
)
GROUP_ORDER = [
    "NOR",
    "BNOR",
    "YNOR",
    "DIA",
    "HBP",
    "H.D.",
    "CPAN",
    "SPAN",
    "YPAN",
    "PRO",
    "OVA",
    "BRE",
    "BLC",
    "CRC",
    "LUN",
]
GROUP_COLORS = {
    "NOR": "#2f6f4e",
    "BNOR": "#76a85b",
    "YNOR": "#9ab85a",
    "DIA": "#b07d2c",
    "HBP": "#8e5ea2",
    "H.D.": "#7c5f45",
    "CPAN": "#d17c4a",
    "SPAN": "#cc8c3d",
    "YPAN": "#e0a33a",
    "PRO": "#486fb5",
    "OVA": "#be5c8a",
    "BRE": "#c75f55",
    "BLC": "#4f8f9f",
    "CRC": "#c04537",
    "LUN": "#344f7a",
}


def normalize_group(group: str) -> str:
    compact = re.sub(r"\s+", "", group.upper())
    if compact in {"H.D", "H.D.", "HD"}:
        return "H.D."
    if compact == "PAN":
        return "CPAN"
    return compact


def parse_sample(path: Path) -> tuple[str, str, str] | None:
    lower_parts = [p.lower() for p in path.parts]
    if any(
        token in part
        for part in lower_parts
        for token in ("background", "reference", "blank", "ref ps", "ref si")
    ):
        return None
    if any(
        part in {"0. ps", "0. si", "0. ref ps", "0. ref si", "0. blank"} for part in lower_parts
    ):
        return None
    if "_ave" in path.stem.lower():
        return None
    if re.search(r"(?i)(^|[\s_/.-])NF($|[\s_.-])", path.name):
        return None
    if re.search(r"(?i)(^|[\s_/.-])PO\.?\s+", path.name):
        return None
    match = SAMPLE_RE.match(path.name)
    if not match:
        return None
    rep = match.group("rep").lower()
    if rep == "ave":
        return None
    return normalize_group(match.group("group")), str(int(match.group("num"))), str(int(rep))


def date_key_from_path(path: Path) -> str:
    for part in path.parts:
        if re.match(r"^20\d{6}$", part):
            return part
        if re.match(r"^20\d{6}", part):
            return part.split("_", 1)[0]
    return "primary"


def sample_files(root: Path) -> list[Path]:
    files = list(root.rglob("*.[Tt][Xx][Tt]")) + list(root.rglob("*.[Cc][Ss][Vv]"))
    return sorted(p for p in files if parse_sample(p) is not None)


def collect_matched_samples(primary_root: Path, medical_root: Path) -> list[dict]:
    primary_by_subject: dict[tuple[str, str], list[Path]] = defaultdict(list)
    medical_by_subject_date: dict[tuple[str, str, str], list[Path]] = defaultdict(list)

    for path in sample_files(primary_root):
        parsed = parse_sample(path)
        if parsed is None:
            continue
        group, sample_id, _ = parsed
        primary_by_subject[(group, sample_id)].append(path)

    for path in sample_files(medical_root):
        parsed = parse_sample(path)
        if parsed is None:
            continue
        group, sample_id, _ = parsed
        medical_by_subject_date[(group, sample_id, date_key_from_path(path))].append(path)

    matched = []
    for (group, sample_id, date_key), medical_paths in medical_by_subject_date.items():
        primary_paths = primary_by_subject.get((group, sample_id), [])
        if not primary_paths:
            continue
        matched.append(
            {
                "group": group,
                "sample_id": sample_id,
                "date_key": date_key,
                "primary_paths": sorted(primary_paths),
                "medical_paths": sorted(medical_paths),
            }
        )
    return sorted(matched, key=sort_key)


def sort_key(item: dict) -> tuple[int, str, int, str]:
    group = item["group"]
    group_rank = GROUP_ORDER.index(group) if group in GROUP_ORDER else len(GROUP_ORDER)
    return group_rank, item["date_key"], int(item["sample_id"]), group


def safe_label(group: str, sample_id: str, date_key: str) -> str:
    return f"{group.replace('.', 'D').replace(' ', '')}_{sample_id}_{date_key}"


def read_native_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, header=None, usecols=[0, 1], names=["x", "y"])
    else:
        df = pd.read_csv(
            path, sep=r"\s+", header=None, usecols=[0, 1], names=["x", "y"], engine="python"
        )
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df["x"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float)


def read_trimmed_xy(path: Path, x_min: float, x_max: float) -> tuple[np.ndarray, np.ndarray]:
    x, y = read_native_xy(path)
    mask = (x >= x_min) & (x <= x_max)
    return x[mask], y[mask]


def read_rows(
    paths: list[Path], x_min: float, x_max: float
) -> list[tuple[Path, np.ndarray, np.ndarray]]:
    return [(path, *read_trimmed_xy(path, x_min, x_max)) for path in paths]


def concat_y(rows: list[tuple[Path, np.ndarray, np.ndarray]]) -> np.ndarray:
    ys = [y for _, _, y in rows if len(y)]
    return np.concatenate(ys) if ys else np.array([])


def file_stats(
    dataset: str,
    group: str,
    sample_id: str,
    date_key: str,
    rows: list[tuple[Path, np.ndarray, np.ndarray]],
) -> list[dict]:
    out = []
    for path, x, y in rows:
        parsed = parse_sample(path)
        replicate = parsed[2] if parsed is not None else ""
        n_neg = int(np.sum(y < 0)) if len(y) else 0
        out.append(
            {
                "dataset": dataset,
                "group": group,
                "sample_id": sample_id,
                "date_key": date_key,
                "replicate": replicate,
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


def summarize_sample(
    item: dict,
    primary_rows: list[tuple[Path, np.ndarray, np.ndarray]],
    medical_rows: list[tuple[Path, np.ndarray, np.ndarray]],
) -> dict:
    primary_y = concat_y(primary_rows)
    medical_y = concat_y(medical_rows)

    def p95_abs(y: np.ndarray) -> float:
        return float(np.percentile(np.abs(y), 95)) if len(y) else np.nan

    def neg_frac(y: np.ndarray) -> float:
        return float(np.mean(y < 0)) if len(y) else np.nan

    primary_p95 = p95_abs(primary_y)
    medical_p95 = p95_abs(medical_y)
    ratio = medical_p95 / primary_p95 if np.isfinite(primary_p95) and primary_p95 > 0 else np.nan
    return {
        "group": item["group"],
        "sample_id": item["sample_id"],
        "date_key": item["date_key"],
        "n_primary_files": len(primary_rows),
        "n_medical_files": len(medical_rows),
        "n_primary_points_after_trim": int(sum(len(y) for _, _, y in primary_rows)),
        "n_medical_points_after_trim": int(sum(len(y) for _, _, y in medical_rows)),
        "primary_y_min_after_trim": float(np.min(primary_y)) if len(primary_y) else np.nan,
        "primary_y_max_after_trim": float(np.max(primary_y)) if len(primary_y) else np.nan,
        "medical_y_min_after_trim": float(np.min(medical_y)) if len(medical_y) else np.nan,
        "medical_y_max_after_trim": float(np.max(medical_y)) if len(medical_y) else np.nan,
        "primary_negative_fraction_after_trim": neg_frac(primary_y),
        "medical_negative_fraction_after_trim": neg_frac(medical_y),
        "primary_abs_y_p95_after_trim": primary_p95,
        "medical_abs_y_p95_after_trim": medical_p95,
        "medical_to_primary_abs_y_p95_ratio_after_trim": ratio,
        "log10_medical_to_primary_abs_y_p95_ratio_after_trim": float(np.log10(ratio))
        if np.isfinite(ratio) and ratio > 0
        else np.nan,
    }


def plot_sample_report(
    item: dict,
    primary_rows: list[tuple[Path, np.ndarray, np.ndarray]],
    medical_rows: list[tuple[Path, np.ndarray, np.ndarray]],
    out_dir: Path,
    x_min: float,
    x_max: float,
    mark_negatives: bool,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        out_dir / f"{safe_label(item['group'], item['sample_id'], item['date_key'])}_trim_only.png"
    )
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    panels = [
        (axes[0], primary_rows, "raw_data_medical: own y-axis", "#606060"),
        (axes[1], medical_rows, "Medical date-acquired: own y-axis", "#1f77b4"),
    ]
    for ax, rows, title, color in panels:
        for path, x, y in rows:
            ax.plot(x, y, linewidth=0.75, alpha=0.70, color=color)
            if mark_negatives and len(y):
                neg = y < 0
                if np.any(neg):
                    ax.scatter(x[neg], y[neg], s=3, alpha=0.30, color="#b33a3a", linewidths=0)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("txt intensity")

    ax = axes[2]
    for _, x, y in primary_rows:
        ax.plot(x, y, linewidth=0.65, alpha=0.58, color="#606060")
    for _, x, y in medical_rows:
        ax.plot(x, y, linewidth=0.65, alpha=0.58, color="#1f77b4")
    all_y = [y for _, _, y in primary_rows + medical_rows if len(y)]
    if all_y:
        y_all = np.concatenate(all_y)
        y_min = float(np.min(y_all))
        y_max = float(np.max(y_all))
        pad = max((y_max - y_min) * 0.05, 1.0)
        ax.set_ylim(y_min - pad, y_max + pad)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title("raw_data_medical + Medical overlaid: same y-axis", fontsize=10)
    ax.set_ylabel("txt intensity")
    ax.set_xlabel("txt Raman shift")
    fig.suptitle(
        f"{item['group']} {item['sample_id']} | {item['date_key']} | TRIM ONLY x=[{x_min:g}, {x_max:g}] | native txt values",
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
        for _, x, y in item["primary_rows"]:
            ax.plot(x, y, linewidth=0.45, alpha=0.55, color="#606060")
        for _, x, y in item["medical_rows"]:
            ax.plot(x, y, linewidth=0.45, alpha=0.65, color="#1f77b4")
        ax.axhline(0, color="black", linewidth=0.45)
        ax.set_xlim(x_min, x_max)
        ax.set_title(f"{item['group']} {item['sample_id']} | {item['date_key']}", fontsize=8)
        ax.tick_params(axis="both", labelsize=6, length=2)
    fig.suptitle(
        f"Medical TRIM ONLY contact sheet | x=[{x_min:g}, {x_max:g}] | gray=raw_data_medical, blue=Medical | no preprocessing",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_summary(sample_summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = sample_summary.copy()
    summary = summary.sort_values(
        ["group", "date_key", "sample_id"],
        key=lambda s: s.map(lambda v: GROUP_ORDER.index(v) if v in GROUP_ORDER else v)
        if s.name == "group"
        else s,
    )
    x = np.arange(len(summary))
    colors = [GROUP_COLORS.get(g, "#555555") for g in summary["group"]]
    paths = []

    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    axes[0].scatter(
        x,
        summary["log10_medical_to_primary_abs_y_p95_ratio_after_trim"],
        s=14,
        c=colors,
        alpha=0.85,
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("log10 Medical / raw\nabs intensity p95")
    axes[0].set_title("Medical trim-only intensity scale shift by sample")

    axes[1].scatter(
        x,
        summary["primary_negative_fraction_after_trim"],
        s=10,
        c="#606060",
        alpha=0.55,
        label="raw_data_medical",
    )
    axes[1].scatter(
        x,
        summary["medical_negative_fraction_after_trim"],
        s=10,
        c="#1f77b4",
        alpha=0.55,
        label="Medical",
    )
    axes[1].set_ylabel("negative fraction")
    axes[1].legend(fontsize=8)

    axes[2].scatter(
        x,
        summary["primary_abs_y_p95_after_trim"],
        s=10,
        c="#606060",
        alpha=0.45,
        label="raw_data_medical",
    )
    axes[2].scatter(
        x, summary["medical_abs_y_p95_after_trim"], s=10, c="#1f77b4", alpha=0.45, label="Medical"
    )
    axes[2].set_yscale("log")
    axes[2].set_ylabel("abs intensity p95")
    axes[2].set_xlabel("matched sample-date index")
    axes[2].legend(fontsize=8)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=GROUP_COLORS.get(g, "#555555"),
            label=g,
            markersize=6,
        )
        for g in GROUP_ORDER
        if g in set(summary["group"])
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=min(len(handles), 15),
        fontsize=8,
        bbox_to_anchor=(0.5, 0.985),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "medical_trim_only_all_samples_metric_overview.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    grouped = [
        summary.loc[summary["group"] == g, "log10_medical_to_primary_abs_y_p95_ratio_after_trim"]
        .dropna()
        .to_numpy()
        for g in GROUP_ORDER
        if g in set(summary["group"])
    ]
    labels = [g for g in GROUP_ORDER if g in set(summary["group"])]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(grouped, tick_labels=labels, showfliers=False)
    for i, values in enumerate(grouped, start=1):
        jitter = np.linspace(-0.18, 0.18, len(values)) if len(values) else []
        ax.scatter(
            np.full(len(values), i) + jitter,
            values,
            s=10,
            alpha=0.35,
            color=GROUP_COLORS.get(labels[i - 1], "#555555"),
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("log10 Medical / raw_data_medical abs intensity p95")
    ax.set_title("Medical trim-only intensity scale shift by group")
    fig.tight_layout()
    path = out_dir / "medical_trim_only_group_intensity_ratio_boxplot.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, default=DEFAULT_PRIMARY_ROOT)
    parser.add_argument("--medical-root", type=Path, default=DEFAULT_MEDICAL_ROOT)
    parser.add_argument("--trim-min", type=float, default=400.0)
    parser.add_argument("--trim-max", type=float, default=1800.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-reports", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--contact-sheets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--contact-sheet-cols", type=int, default=3)
    parser.add_argument("--contact-sheet-page-size", type=int, default=12)
    parser.add_argument("--mark-negatives", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    matched = collect_matched_samples(args.primary_root, args.medical_root)
    if args.max_samples is not None:
        matched = matched[: args.max_samples]
    if not matched:
        raise RuntimeError(
            f"No matched sample-date pairs found between {args.primary_root} and {args.medical_root}"
        )

    sample_report_dir = args.output_dir / "sample_reports"
    contact_sheet_dir = args.output_dir / "contact_sheets"
    summary_dir = args.output_dir / "summary"
    file_rows = []
    sample_rows = []
    contact_items = []
    report_paths = []

    for idx, item in enumerate(matched, start=1):
        primary_rows = read_rows(item["primary_paths"], args.trim_min, args.trim_max)
        medical_rows = read_rows(item["medical_paths"], args.trim_min, args.trim_max)
        file_rows.extend(
            file_stats(
                "raw_data_medical", item["group"], item["sample_id"], item["date_key"], primary_rows
            )
        )
        file_rows.extend(
            file_stats("Medical", item["group"], item["sample_id"], item["date_key"], medical_rows)
        )
        sample_rows.append(summarize_sample(item, primary_rows, medical_rows))

        if args.sample_reports:
            path = plot_sample_report(
                item,
                primary_rows,
                medical_rows,
                sample_report_dir,
                args.trim_min,
                args.trim_max,
                args.mark_negatives,
            )
            numbered = sample_report_dir / f"{idx:04d}_{path.name}"
            path.rename(numbered)
            report_paths.append(numbered)
        if args.contact_sheets:
            contact_item = dict(item)
            contact_item["primary_rows"] = primary_rows
            contact_item["medical_rows"] = medical_rows
            contact_items.append(contact_item)
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(matched)} matched sample-date pairs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    file_stats_path = args.output_dir / "medical_trim_only_file_stats_all_samples.csv"
    sample_summary_path = args.output_dir / "medical_trim_only_sample_summary_all_samples.csv"
    pd.DataFrame(file_rows).to_csv(file_stats_path, index=False, encoding="utf-8-sig")
    sample_summary = pd.DataFrame(sample_rows)
    sample_summary.to_csv(sample_summary_path, index=False, encoding="utf-8-sig")
    summary_paths = plot_summary(sample_summary, summary_dir)

    contact_paths = []
    if args.contact_sheets:
        page_size = args.contact_sheet_page_size
        for page_idx in range(0, len(contact_items), page_size):
            page = contact_items[page_idx : page_idx + page_size]
            out_path = (
                contact_sheet_dir
                / f"medical_trim_only_contact_sheet_{page_idx // page_size + 1:03d}.png"
            )
            plot_contact_sheet(
                page, out_path, args.trim_min, args.trim_max, args.contact_sheet_cols
            )
            contact_paths.append(out_path)

    manifest_path = args.output_dir / "medical_trim_only_outputs_manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as fh:
        fh.write(f"primary_root={args.primary_root}\n")
        fh.write(f"medical_root={args.medical_root}\n")
        fh.write(f"matched_sample_date_pairs={len(matched)}\n")
        fh.write(f"sample_reports={len(report_paths)}\n")
        fh.write(f"contact_sheets={len(contact_paths)}\n")
        fh.write(f"file_stats={file_stats_path}\n")
        fh.write(f"sample_summary={sample_summary_path}\n")
        for path in summary_paths:
            fh.write(f"summary_figure={path}\n")
        for path in contact_paths:
            fh.write(f"contact_sheet={path}\n")
        for path in report_paths:
            fh.write(f"sample_report={path}\n")

    print(f"Matched sample-date pairs: {len(matched)}")
    print(f"Sample reports: {len(report_paths)}")
    print(f"Contact sheets: {len(contact_paths)}")
    print(f"File stats: {file_stats_path}")
    print(f"Sample summary: {sample_summary_path}")
    print(f"Manifest: {manifest_path}")
    for path in summary_paths:
        print(f"Summary figure: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
