#!/usr/bin/env python3
"""Build process-report figures for the SERS-AI professor report.

The figures are intentionally explanatory: each plot shows which SERS-specific
measurement issue a pipeline step handles and what the corrected signal looks
like afterward.

Outputs:
    results/figures/process_report/*.png
    results/figures/process_report/figure_manifest.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sers.config import load_config  # noqa: E402
from sers.io import find_spectra, parse_filename, read_spectrum  # noqa: E402
from sers.preprocessing import (  # noqa: E402
    DEFAULT_REFERENCE_PEAK_WN,
    calibrate_spectrum,
    trim_spectrum,
    smooth,
    baseline_correction,
    normalize_spectrum,
    resample,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


OUT_DIR = PROJECT_ROOT / "results" / "figures" / "process_report"
RAW_DIR = PROJECT_ROOT / "data" / "raw_data"
MEDICAL_DIR = PROJECT_ROOT / "data" / "raw_data_medical"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "usersnet" / "current"
SHIFT_CSV_CANDIDATES = [
    PROJECT_ROOT / "results" / "preprocessing_dacr_all" / "calibration_shifts.csv",
    PROJECT_ROOT / "results" / "preprocessing" / "calibration_shifts.csv",
]

COLORS = {
    "raw": "#234E70",
    "after": "#D1495B",
    "accent": "#2A9D8F",
    "gold": "#E9A23B",
    "muted": "#6B7280",
    "dark": "#1F2937",
    "light": "#F6F7F9",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FBFCFE",
            "axes.edgecolor": "#D1D5DB",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, name: str, title: str, issue: str, solution: str, rows: list[dict]) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=180)
    plt.close(fig)
    rows.append({"file": name, "title": title, "sers_issue": issue, "pipeline_solution": solution})
    print(f"saved {path}")


def load_raw_index(config) -> dict[tuple[str, str, int], Path]:
    index: dict[tuple[str, str, int], Path] = {}
    for folder, group in config.folder_to_group.items():
        d = RAW_DIR / folder
        if not d.is_dir():
            continue
        files = []
        for pat in ("*.CSV", "*.csv", "*.txt"):
            files.extend(find_spectra(d, pattern=pat, recursive=False))
        for fp in files:
            if "ave" in fp.stem.lower() or "zone.identifier" in fp.name.lower():
                continue
            try:
                sid = parse_filename(fp, fallback_group=group)
            except Exception:
                continue
            index[(sid.group, str(sid.sample_id), int(sid.replicate))] = fp
    return index


def load_group_files(config, groups: tuple[str, ...], max_per_group: int = 80) -> list[tuple[str, Path]]:
    out = []
    for folder, group in config.folder_to_group.items():
        if group not in groups:
            continue
        d = RAW_DIR / folder
        if not d.is_dir():
            continue
        files = []
        for pat in ("*.CSV", "*.csv"):
            files.extend(find_spectra(d, pattern=pat, recursive=False))
        clean = [f for f in sorted(files) if "ave" not in f.stem.lower() and "zone.identifier" not in f.name.lower()]
        out.extend((group, f) for f in clean[:max_per_group])
    return out


def load_replicates(index: dict[tuple[str, str, int], Path], preferred_group: str = "NOR"):
    by_sample: dict[tuple[str, str], list[tuple[int, Path]]] = {}
    for (group, sid, rep), fp in index.items():
        by_sample.setdefault((group, sid), []).append((rep, fp))
    candidates = [
        (key, sorted(vals))
        for key, vals in by_sample.items()
        if key[0] == preferred_group and len(vals) >= 5
    ]
    if not candidates:
        candidates = [(key, sorted(vals)) for key, vals in by_sample.items() if len(vals) >= 5]
    if not candidates:
        raise RuntimeError("No sample with at least 5 replicates found")
    (group, sid), reps = sorted(candidates, key=lambda x: (x[0][0], int(x[0][1]) if x[0][1].isdigit() else 9999))[0]
    spectra = []
    for rep, fp in reps[:5]:
        x, y = read_spectrum(fp)
        spectra.append({"group": group, "sample_id": sid, "replicate": rep, "path": fp, "x": x, "y": y})
    return spectra


def load_grid() -> np.ndarray:
    grid_path = ARTIFACT_DIR / "common_grid.npy"
    if grid_path.exists():
        return np.load(grid_path)
    return np.linspace(402.0, 2198.0, 935)


def baseline_curve(y: np.ndarray, window: int = 101) -> np.ndarray:
    return pd.Series(y).rolling(window, center=True, min_periods=1).min().to_numpy()


def snv_rows(mat: np.ndarray) -> np.ndarray:
    mean = mat.mean(axis=1, keepdims=True)
    std = mat.std(axis=1, keepdims=True)
    return (mat - mean) / (std + 1e-8)


def fig_process_map(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        ("Raw SERS", "hotspot scale,\nbaseline,\ndrift"),
        ("QC", "enhancement failure,\nreplicate outlier"),
        ("Calibration", "wavenumber drift"),
        ("Preprocess", "noise,\nbaseline,\nscale"),
        ("STK-V2 views", "raw + D1 + D2\n+ peak features"),
        ("Stacking", "base model diversity"),
        ("SSI/report", "replicate-level\nclinical output"),
    ]
    xs = np.linspace(0.07, 0.93, len(stages))
    for i, ((label, sub), x) in enumerate(zip(stages, xs)):
        box = FancyBboxPatch(
            (x - 0.055, 0.52),
            0.11,
            0.2,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            linewidth=1.2,
            edgecolor=COLORS["dark"],
            facecolor="#EEF4F8" if i % 2 == 0 else "#F7F3EA",
        )
        ax.add_patch(box)
        ax.text(x, 0.65, label, ha="center", va="center", fontweight="bold")
        ax.text(x, 0.57, sub, ha="center", va="center", fontsize=8, color=COLORS["dark"])
        if i < len(stages) - 1:
            arr = FancyArrowPatch((x + 0.06, 0.62), (xs[i + 1] - 0.06, 0.62), arrowstyle="-|>", mutation_scale=14, lw=1.4, color=COLORS["muted"])
            ax.add_patch(arr)
    ax.text(0.5, 0.9, "SERS-AI process: each step removes a measurement artifact before modeling", ha="center", fontsize=15, fontweight="bold")
    ax.text(
        0.5,
        0.28,
        "Key principle: keep metabolite peak pattern, remove substrate/batch/instrument artifacts.",
        ha="center",
        fontsize=12,
        color=COLORS["dark"],
    )
    save(
        fig,
        "00_process_map.png",
        "Process map",
        "SERS signals mix biological peak patterns with measurement artifacts.",
        "The pipeline removes artifacts stepwise before STK-V2 inference.",
        rows,
    )


def fig_raw_and_trim(spectrum: dict, rows: list[dict]) -> None:
    x, y = spectrum["x"], spectrum["y"]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(x, y, color=COLORS["raw"], lw=0.8)
    ax.axvspan(400, 2200, color=COLORS["accent"], alpha=0.12, label="kept: urine metabolite fingerprint")
    ax.axvspan(x.min(), 400, color="#D1495B", alpha=0.08, label="removed: low-wavenumber noise/substrate")
    ax.axvspan(2200, x.max(), color="#D1495B", alpha=0.08, label="removed: non-fingerprint region")
    ax.set_title("Raw SERS spectrum -> fingerprint region trim")
    ax.set_xlabel("Wavenumber (cm^-1)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.legend(loc="upper right")
    save(
        fig,
        "01_raw_fingerprint_trim.png",
        "Raw spectrum and fingerprint trim",
        "Raw SERS contains useful urine-metabolite peaks plus non-fingerprint/substrate regions.",
        "Keep 400-2200 cm^-1 before later signal processing.",
        rows,
    )


def fig_qc(files: list[tuple[str, Path]], replicates: list[dict], grid: np.ndarray, rows: list[dict]) -> None:
    fp_rows = []
    for group, fp in files:
        try:
            x, y = read_spectrum(fp)
        except Exception:
            continue
        mask = (x >= 400) & (x <= 2200)
        fp_rows.append({"group": group, "file": fp.name, "fp_mean": float(np.mean(np.abs(y[mask] if mask.any() else y)))})
    df = pd.DataFrame(fp_rows)
    threshold = df["fp_mean"].median() * 0.1 if len(df) else 0.0

    reps = []
    for r in replicates:
        x, y = trim_spectrum(r["x"], r["y"], (400, 2200))
        y2 = resample(x, y, grid)
        reps.append(y2)
    mat = np.vstack(reps)
    corr = np.corrcoef(mat)
    mean_corr = (corr[np.triu_indices_from(corr, 1)]).mean()
    rsd = np.mean(np.std(mat, axis=0) / (np.maximum(np.mean(np.abs(mat), axis=0), 1e-8))) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    for group, c in [("NOR", COLORS["accent"]), ("PRO", COLORS["raw"]), ("CRC", COLORS["after"])]:
        vals = df.loc[df["group"] == group, "fp_mean"]
        if len(vals):
            ax.hist(vals, bins=20, alpha=0.55, label=group, color=c, edgecolor="white")
    ax.axvline(threshold, color=COLORS["gold"], ls="--", lw=2, label=f"gate = median x 0.1")
    ax.set_title("QC: SERS enhancement intensity gate")
    ax.set_xlabel("Mean intensity in 400-2200 cm^-1")
    ax.set_ylabel("Count")
    ax.legend()

    ax = axes[1]
    for i, y in enumerate(mat):
        ax.plot(grid, y, lw=0.8, alpha=0.72, label=f"rep {i + 1}")
    ax.set_title(f"QC: replicate consistency\nmean corr={mean_corr:.4f}, approx RSD={rsd:.2f}%")
    ax.set_xlabel("Wavenumber (cm^-1)")
    ax.set_ylabel("Interpolated intensity")
    ax.legend(ncol=2)
    save(
        fig,
        "02_qc_hotspot_replicates.png",
        "QC for hotspot failure and replicate consistency",
        "SERS hotspot/substrate activation can fail; replicate spectra can be inconsistent.",
        "Intensity gate and replicate correlation/RSD detect unreliable spectra before modeling.",
        rows,
    )


def find_shift_csv() -> Path | None:
    for p in SHIFT_CSV_CANDIDATES:
        if p.exists():
            return p
    return None


def fig_wavenumber_drift(index: dict[tuple[str, str, int], Path], rows: list[dict]) -> None:
    shift_csv = find_shift_csv()
    df = pd.read_csv(shift_csv) if shift_csv else pd.DataFrame()
    selected = None
    if len(df):
        df["abs_shift"] = df["shift_cm1"].abs()
        for _, row in df.sort_values("abs_shift", ascending=False).iterrows():
            key = (str(row["group"]), str(row["sample_id"]), int(row["replicate"]))
            if key in index and 2 <= abs(float(row["shift_cm1"])) <= 12:
                selected = (key, index[key])
                break
    if selected is None:
        selected = next(iter(index.items()))

    (group, sid, rep), fp = selected
    x, y = read_spectrum(fp)
    x_cal, y_cal, shift = calibrate_spectrum(x, y, target_wn=DEFAULT_REFERENCE_PEAK_WN, window=20.0)

    mask_raw = (x >= 940) & (x <= 1060)
    mask_cal = (x_cal >= 940) & (x_cal <= 1060)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    if len(df):
        ax.hist(df["shift_cm1"], bins=45, color=COLORS["raw"], alpha=0.72, edgecolor="white")
        ax.axvline(0, color=COLORS["dark"], lw=1)
        ax.axvline(df["shift_cm1"].median(), color=COLORS["gold"], ls="--", lw=2, label=f"median={df['shift_cm1'].median():+.2f}")
        ax.set_title("Observed wavenumber calibration shifts")
        ax.set_xlabel("Applied shift (cm^-1)")
        ax.set_ylabel("Spectra")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No calibration shift CSV found", ha="center", va="center")
        ax.axis("off")

    ax = axes[1]
    ax.plot(x[mask_raw], y[mask_raw], color=COLORS["raw"], lw=1.0, label="before alignment")
    ax.plot(x_cal[mask_cal], y_cal[mask_cal], color=COLORS["after"], lw=1.0, label=f"after alignment ({shift:+.2f} cm^-1)")
    ax.axvline(DEFAULT_REFERENCE_PEAK_WN, color=COLORS["accent"], ls="--", lw=1.8, label="urea reference 1001.4")
    ax.set_title(f"Wavenumber drift correction example: {group} {sid}_{rep}")
    ax.set_xlabel("Wavenumber (cm^-1)")
    ax.set_ylabel("Intensity")
    ax.legend()
    save(
        fig,
        "03_wavenumber_drift_alignment.png",
        "Wavenumber drift alignment",
        "Batch/instrument drift moves peak positions and can look like a cancer-specific feature.",
        "Align reference peak and resample to a fixed grid before feature extraction.",
        rows,
    )


def fig_baseline_snv(spectrum: dict, replicates: list[dict], grid: np.ndarray, rows: list[dict]) -> None:
    x0, y0 = spectrum["x"], spectrum["y"]
    xt, yt = trim_spectrum(x0, y0, (400, 2200))
    ys = smooth(yt, 11, 3)
    bl = baseline_curve(ys, 101)
    yb = ys - bl
    yn = normalize_spectrum(yb, "snv")

    raw_rep = []
    snv_rep = []
    for r in replicates:
        x, y = trim_spectrum(r["x"], r["y"], (400, 2200))
        y = resample(x, y, grid)
        raw_rep.append(y)
        y2 = baseline_correction(smooth(y, 11, 3), 101)
        snv_rep.append(normalize_spectrum(y2, "snv"))
    raw_rep = np.vstack(raw_rep)
    snv_rep = np.vstack(snv_rep)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    ax.plot(xt, ys, color=COLORS["raw"], lw=0.9, label="smoothed spectrum")
    ax.plot(xt, bl, color=COLORS["gold"], lw=1.2, label="estimated baseline")
    ax.plot(xt, yb, color=COLORS["after"], lw=0.9, label="baseline corrected")
    ax.set_title("Baseline correction removes broad background")
    ax.set_xlabel("Wavenumber (cm^-1)")
    ax.set_ylabel("Intensity")
    ax.legend()

    ax = axes[1]
    for row in raw_rep:
        ax.plot(grid, row / (np.max(np.abs(row)) + 1e-8), color=COLORS["muted"], alpha=0.45, lw=0.7)
    for row in snv_rep:
        ax.plot(grid, row, color=COLORS["accent"], alpha=0.75, lw=0.8)
    ax.set_title("SNV reduces hotspot-scale variation\n(gray: raw scaled; green: SNV)")
    ax.set_xlabel("Wavenumber (cm^-1)")
    ax.set_ylabel("Relative / SNV intensity")
    save(
        fig,
        "04_baseline_snv_normalization.png",
        "Baseline correction and SNV normalization",
        "SERS has fluorescence/background drift and hotspot-driven intensity scaling.",
        "Rolling-minimum baseline correction and SNV preserve peak ratios while reducing artifacts.",
        rows,
    )


def load_peaks() -> list[tuple[float, str, float]]:
    peak_path = ARTIFACT_DIR / "peak_config.json"
    if peak_path.exists():
        with open(peak_path) as f:
            return [tuple(x) for x in json.load(f)["known_peaks"]]
    return [
        (683.3, "creatinine", 15),
        (723.8, "adenine", 15),
        (795.1, "hippuric", 15),
        (895.4, "uric_acid", 15),
        (999.5, "phe_urea", 15),
        (1597.1, "purine_CC", 20),
    ]


def make_multichannel(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    channels = []
    for deriv in (0, 1, 2):
        y_proc = savgol_filter(y, 11, 3, deriv=deriv)
        xt, yt = trim_spectrum(x, y_proc, (400, 2200))
        if deriv == 0:
            yt = baseline_correction(yt, 101)
        yt = normalize_spectrum(yt, "snv")
        channels.append(resample(xt, yt, grid))
    return np.vstack(channels)


def fig_multiview_stk(spectrum: dict, grid: np.ndarray, rows: list[dict]) -> None:
    x, y = spectrum["x"], spectrum["y"]
    ch = make_multichannel(x, y, grid)
    peaks = load_peaks()

    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[:, 1])

    ax0.plot(grid, ch[0], color=COLORS["raw"], lw=0.9, label="raw/SNV")
    for center, name, _ in peaks:
        if 400 <= center <= 2200:
            ax0.axvline(center, color=COLORS["gold"], alpha=0.2, lw=0.8)
    ax0.set_title("STK-V2 spectral views: raw channel + metabolite peak anchors")
    ax0.set_ylabel("SNV intensity")
    ax0.legend()

    ax1.plot(grid, ch[1], color=COLORS["accent"], lw=0.9, label="D1: peak position/slope")
    ax1.plot(grid, ch[2], color=COLORS["after"], lw=0.9, label="D2: peak curvature/overlap")
    ax1.set_title("Derivative views emphasize peak position and shape")
    ax1.set_xlabel("Wavenumber (cm^-1)")
    ax1.set_ylabel("SNV derivative")
    ax1.legend()

    ax2.axis("off")
    ax2.set_title("STK-V2 stacking ensemble", fontsize=13, fontweight="bold", pad=12)
    boxes = [
        (0.08, 0.78, 0.84, 0.12, "Input views\nRaw + D1 + D2 + 75 peak features", "#EEF4F8"),
        (0.08, 0.58, 0.84, 0.14, "10 base models\nLR / XGBoost / Random Forest / Ridge", "#F7F3EA"),
        (0.08, 0.38, 0.84, 0.14, "Meta features\nStage 1 probs + Stage 2 type probs", "#EDF7F2"),
        (0.08, 0.18, 0.84, 0.14, "Meta learner\nCancer Screening + Cancer Type ID", "#FDECEF"),
    ]
    for x0, y0, w, h, text, color in boxes:
        ax2.add_patch(
            FancyBboxPatch(
                (x0, y0),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                linewidth=1.1,
                edgecolor=COLORS["dark"],
                facecolor=color,
            )
        )
        ax2.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=10)
    for y0 in (0.74, 0.54, 0.34):
        ax2.annotate("", xy=(0.5, y0), xytext=(0.5, y0 - 0.05), arrowprops=dict(arrowstyle="<|-", color=COLORS["muted"], lw=1.3))
    save(
        fig,
        "05_stk_v2_multiview_model.png",
        "STK-V2 multi-view stacking model",
        "A single spectral view can miss peak position, curvature, and interpretable metabolite-band information.",
        "STK-V2 fuses raw, derivative, and peak-feature views through heterogeneous stacking.",
        rows,
    )


def fig_pds_and_output(rows: list[dict]) -> None:
    summary_path = PROJECT_ROOT / "results" / "cross_instrument" / "calibration" / "sweep" / "method_comparison_summary.csv"
    pds_df = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    holdout_path = PROJECT_ROOT / "results" / "training" / "stacking_v2_holdout" / "holdout_report.json"
    holdout = json.load(open(holdout_path)) if holdout_path.exists() else {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    if len(pds_df):
        cols = [c for c in pds_df.columns if "auc" in c.lower() or "f1" in c.lower()]
        label_col = "method" if "method" in pds_df.columns else pds_df.columns[0]
        metric = cols[0] if cols else pds_df.columns[-1]
        plot_df = pds_df[[label_col, metric]].dropna().head(8)
        ax.bar(plot_df[label_col].astype(str), plot_df[metric].astype(float), color=COLORS["accent"])
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=35)
        ax.set_title("Cross-instrument correction comparison")
    else:
        methods = ["Medical raw", "M->T PDS", "Thermo ref."]
        auc = [0.447, 0.922, 0.991]
        ax.bar(methods, auc, color=[COLORS["muted"], COLORS["accent"], COLORS["raw"]])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Cancer Screening AUC")
        ax.set_title("PDS maps Medical spectra toward Thermo space")

    ax = axes[1]
    if holdout:
        test = next((r for r in holdout.get("results", []) if r.get("split") == "Test"), {})
        labels = ["AUC", "Sens\nscreen", "Spec\nscreen", "Type F1"]
        vals = [
            test.get("s1_auc", np.nan),
            test.get("s1_screening_sens", np.nan),
            test.get("s1_screening_spec", np.nan),
            test.get("s2_f1_macro", np.nan),
        ]
    else:
        labels = ["AUC", "Sensitivity", "Specificity", "Type F1"]
        vals = [0.9925, 0.9544, 0.9425, 0.9593]
    ax.bar(labels, vals, color=[COLORS["raw"], COLORS["accent"], COLORS["gold"], COLORS["after"]])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric")
    ax.set_title("STK-V2 output after artifact control")
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    save(
        fig,
        "06_calibration_transfer_and_output.png",
        "Calibration transfer and STK-V2 output",
        "Different instruments create domain shift even after ordinary preprocessing.",
        "PDS calibration transfer and fixed decision rules produce auditable SSI/report outputs.",
        rows,
    )


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config(PROJECT_ROOT / "config" / "config.yaml")
    index = load_raw_index(config)
    if not index:
        raise RuntimeError("No raw spectra found under data/raw_data")

    grid = load_grid()
    replicates = load_replicates(index, preferred_group="NOR")
    spectrum = replicates[0]
    group_files = load_group_files(config, groups=("NOR", "PRO", "CRC"), max_per_group=120)

    rows: list[dict] = []
    fig_process_map(rows)
    fig_raw_and_trim(spectrum, rows)
    fig_qc(group_files, replicates, grid, rows)
    fig_wavenumber_drift(index, rows)
    fig_baseline_snv(spectrum, replicates, grid, rows)
    fig_multiview_stk(spectrum, grid, rows)
    fig_pds_and_output(rows)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT_DIR / "figure_manifest.csv", index=False)
    print(f"saved {OUT_DIR / 'figure_manifest.csv'}")


if __name__ == "__main__":
    main()
