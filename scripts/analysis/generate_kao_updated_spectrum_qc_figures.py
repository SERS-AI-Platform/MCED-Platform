#!/usr/bin/env python3
"""Generate updated cohort mean-spectrum CI and PCA figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COHORT = PROJECT_ROOT / "results" / "kao_20260610_updated_cohort" / "processed_spectra.csv"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "results" / "kao_20260610_updated_cohort" / "cohort_subject_manifest.csv"
)
DEFAULT_OUT = PROJECT_ROOT / "publications" / "대한암학회_20260610_updated"

GROUP_ALIASES = {"CPAN": "PAN", "YPAN": "PAN", "YNOR": "NOR"}
CANCER_GROUPS = {"PRO", "BRE", "OVA", "LUN", "CRC", "BLC", "PAN"}
DISPLAY_ORDER = ("CONTROL", "PRO", "BRE", "OVA", "LUN", "CRC", "BLC", "PAN")
SOURCE_ORDER = (
    "NOR",
    "YNOR",
    "DIA",
    "HBP",
    "H.D.",
    "PRO",
    "BRE",
    "OVA",
    "LUN",
    "CRC",
    "BLC",
    "CPAN",
    "YPAN",
)
COLORS = {
    "CONTROL": "#666666",
    "NOR": "#6E6E6E",
    "YNOR": "#9E9E9E",
    "DIA": "#7F7F7F",
    "HBP": "#4D4D4D",
    "H.D.": "#A9A9A9",
    "PRO": "#8B4513",
    "BRE": "#D4527A",
    "OVA": "#6A5ACD",
    "LUN": "#2E8B57",
    "CRC": "#4682B4",
    "PAN": "#CD5C5C",
    "CPAN": "#B94A48",
    "YPAN": "#E07A5F",
    "BLC": "#E8960C",
}

STAGE_ORDER = ("1-2", "3-4")
STAGE_LABELS = {"1-2": "Stage I-II", "3-4": "Stage III-IV"}
CANCER_DISPLAY_ORDER = ("BLC", "CRC", "LUN", "PAN", "PRO", "OVA", "BRE")


def _feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if str(c).startswith("x_")]
    if not cols:
        raise ValueError("No x_* columns found")
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


def _mean_ci_table(
    subject: pd.DataFrame, wavenumbers: np.ndarray, feature_cols: list[str]
) -> pd.DataFrame:
    rows = []
    for group in DISPLAY_ORDER:
        sub = subject[subject["display_group"] == group]
        if sub.empty:
            continue
        X = sub[feature_cols].to_numpy(dtype=float)
        mean = X.mean(axis=0)
        sem = X.std(axis=0, ddof=1) / np.sqrt(len(X)) if len(X) > 1 else np.zeros(X.shape[1])
        ci = 1.96 * sem
        rows.append(
            pd.DataFrame(
                {
                    "display_group": group,
                    "n_subjects": len(X),
                    "wavenumber_cm-1": wavenumbers,
                    "mean": mean,
                    "sem": sem,
                    "ci95_low": mean - ci,
                    "ci95_high": mean + ci,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.6, labelsize=7)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=8)
    ax.set_ylabel("Intensity (a.u.)", fontsize=8)


def _lighten_color(color: str, amount: float = 0.55) -> tuple[float, float, float]:
    import matplotlib.colors as mcolors

    rgb = np.asarray(mcolors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * amount)


def _stage_mean_ci_table(
    subject: pd.DataFrame,
    manifest_csv: Path,
    wavenumbers: np.ndarray,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(manifest_csv)
    clinical_cols = [
        "subject_id",
        "stage_1_2_or_3_4",
        "stage_source",
        "clinical_match_status",
    ]
    missing = [c for c in clinical_cols if c not in manifest.columns]
    if missing:
        raise ValueError(f"Clinical manifest is missing required columns: {missing}")

    staged = subject.merge(manifest[clinical_cols], on="subject_id", how="left")
    staged["stage_1_2_or_3_4"] = staged["stage_1_2_or_3_4"].fillna("missing")
    rows = []
    for cancer in CANCER_DISPLAY_ORDER:
        for stage_group in STAGE_ORDER:
            sub = staged[
                (staged["model_group"] == cancer) & (staged["stage_1_2_or_3_4"] == stage_group)
            ]
            if sub.empty:
                continue
            X = sub[feature_cols].to_numpy(dtype=float)
            mean = X.mean(axis=0)
            if len(X) > 1:
                sem = X.std(axis=0, ddof=1) / np.sqrt(len(X))
            else:
                sem = np.zeros(X.shape[1])
            ci = 1.96 * sem
            rows.append(
                pd.DataFrame(
                    {
                        "model_group": cancer,
                        "stage_group": stage_group,
                        "stage_label": STAGE_LABELS[stage_group],
                        "n_subjects": len(X),
                        "wavenumber_cm-1": wavenumbers,
                        "mean": mean,
                        "sem": sem,
                        "ci95_low": mean - ci,
                        "ci95_high": mean + ci,
                    }
                )
            )

    if rows:
        table = pd.concat(rows, ignore_index=True)
    else:
        table = pd.DataFrame(
            columns=[
                "model_group",
                "stage_group",
                "stage_label",
                "n_subjects",
                "wavenumber_cm-1",
                "mean",
                "sem",
                "ci95_low",
                "ci95_high",
            ]
        )

    stage_counts = (
        staged[staged["model_group"].isin(CANCER_DISPLAY_ORDER)]
        .groupby(["model_group", "stage_1_2_or_3_4"], dropna=False)
        .size()
        .reset_index(name="n_subjects")
        .sort_values(["model_group", "stage_1_2_or_3_4"])
    )
    return table, stage_counts


def _top_stage_delta_peaks(
    table: pd.DataFrame,
    *,
    n_peaks: int = 3,
    min_separation: float = 80.0,
) -> pd.DataFrame:
    rows = []
    for cancer in CANCER_DISPLAY_ORDER:
        sub = table[table["model_group"] == cancer]
        if set(sub["stage_group"]) != set(STAGE_ORDER):
            continue
        early = sub[sub["stage_group"] == "1-2"].sort_values("wavenumber_cm-1")
        late = sub[sub["stage_group"] == "3-4"].sort_values("wavenumber_cm-1")
        x = early["wavenumber_cm-1"].to_numpy(dtype=float)
        delta = late["mean"].to_numpy(dtype=float) - early["mean"].to_numpy(dtype=float)
        order = np.argsort(np.abs(delta))[::-1]
        selected: list[int] = []
        for idx in order:
            if all(abs(x[idx] - x[prev]) >= min_separation for prev in selected):
                selected.append(int(idx))
            if len(selected) >= n_peaks:
                break
        for rank, idx in enumerate(selected, start=1):
            rows.append(
                {
                    "model_group": cancer,
                    "rank": rank,
                    "wavenumber_cm-1": float(x[idx]),
                    "delta_stage_3_4_minus_1_2": float(delta[idx]),
                    "abs_delta": float(abs(delta[idx])),
                }
            )
    return pd.DataFrame(rows)


def plot_stage_spectra(table: pd.DataFrame, peak_table: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(11.4, 10.2), sharex=True, sharey=False)
    axes_flat = axes.ravel()
    for ax, cancer in zip(axes_flat, CANCER_DISPLAY_ORDER):
        sub = table[table["model_group"] == cancer]
        color = COLORS.get(cancer, "#333333")
        early_color = _lighten_color(color, 0.50)
        late_color = color

        for stage_group, line_color, line_style, zorder in (
            ("1-2", early_color, "-", 2),
            ("3-4", late_color, "-", 3),
        ):
            st = sub[sub["stage_group"] == stage_group].sort_values("wavenumber_cm-1")
            if st.empty:
                continue
            x = st["wavenumber_cm-1"].to_numpy(dtype=float)
            mean = st["mean"].to_numpy(dtype=float)
            lo = st["ci95_low"].to_numpy(dtype=float)
            hi = st["ci95_high"].to_numpy(dtype=float)
            n = int(st["n_subjects"].iloc[0])
            ax.fill_between(x, lo, hi, color=line_color, alpha=0.16, linewidth=0, zorder=zorder - 1)
            ax.plot(
                x,
                mean,
                color=line_color,
                lw=1.35 if stage_group == "3-4" else 1.15,
                ls=line_style,
                label=f"{STAGE_LABELS[stage_group]} (n={n})",
                zorder=zorder,
            )

        peaks = peak_table[peak_table["model_group"] == cancer]
        y_min, y_max = ax.get_ylim()
        y_label = y_min + 0.06 * (y_max - y_min)
        for _, row in peaks.iterrows():
            wn = float(row["wavenumber_cm-1"])
            ax.axvspan(wn - 12, wn + 12, color=color, alpha=0.08, lw=0)
            ax.text(
                wn,
                y_label,
                f"{wn:.0f}",
                color=color,
                fontsize=6,
                fontweight="bold",
                ha="center",
                va="bottom",
                rotation=90,
            )

        ax.set_title(cancer, loc="left", fontsize=10, fontweight="bold", color=color)
        _style_ax(ax)
        ax.grid(True, axis="both", color="#EAEAEA", linewidth=0.45)
        ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2)

        present = set(sub["stage_group"])
        missing = [STAGE_LABELS[s] for s in STAGE_ORDER if s not in present]
        if missing:
            ax.text(
                0.985,
                0.10,
                "No " + ", ".join(missing),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                color="#777777",
            )

    for ax in axes_flat[len(CANCER_DISPLAY_ORDER) :]:
        ax.axis("off")

    if not table.empty:
        xmin = float(table["wavenumber_cm-1"].min())
        xmax = float(table["wavenumber_cm-1"].max())
        for ax in axes_flat[: len(CANCER_DISPLAY_ORDER)]:
            ax.set_xlim(xmin, xmax)

    fig.suptitle(
        "Stage-specific mean SERS spectra by cancer type", fontsize=12, fontweight="bold", y=0.995
    )
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.055, top=0.955, hspace=0.34, wspace=0.12)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"stage_spectra_95ci_by_cancer.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mean_spectra(table: pd.DataFrame, out_dir: Path) -> None:
    groups = [g for g in DISPLAY_ORDER if g in set(table["display_group"])]
    fig, axes = plt.subplots(len(groups), 1, figsize=(7.2, 1.05 * len(groups)), sharex=True)
    if len(groups) == 1:
        axes = [axes]
    for ax, group in zip(axes, groups):
        sub = table[table["display_group"] == group]
        color = COLORS.get(group, "#333333")
        x = sub["wavenumber_cm-1"].to_numpy(dtype=float)
        mean = sub["mean"].to_numpy(dtype=float)
        lo = sub["ci95_low"].to_numpy(dtype=float)
        hi = sub["ci95_high"].to_numpy(dtype=float)
        n = int(sub["n_subjects"].iloc[0])
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, mean, color=color, lw=1.25)
        ax.text(
            0.985,
            0.78,
            f"{group} (n={n})",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=color,
        )
        _style_ax(ax)
    axes[-1].set_xlim(float(table["wavenumber_cm-1"].min()), float(table["wavenumber_cm-1"].max()))
    fig.suptitle("Mean SERS spectra with 95% CI", fontsize=10, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.06, top=0.95, hspace=0.16)
    for ext in ("png", "pdf"):
        fig.savefig(
            out_dir / f"mean_spectra_95ci_by_model_group.{ext}", dpi=300, bbox_inches="tight"
        )
    plt.close(fig)


def _pca_scores(
    subject: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = subject[feature_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=5, random_state=42)
    scores = pca.fit_transform(X_scaled)
    score_df = subject[
        ["subject_id", "source_group", "sample_id_str", "model_group", "display_group"]
    ].copy()
    for i in range(scores.shape[1]):
        score_df[f"PC{i + 1}"] = scores[:, i]
    variance = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "explained_variance_percent": pca.explained_variance_ratio_ * 100,
        }
    )
    return score_df, variance


def _scatter_pca(
    scores: pd.DataFrame,
    variance: pd.DataFrame,
    *,
    group_col: str,
    order: tuple[str, ...],
    filename: str,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    present = [g for g in order if g in set(scores[group_col])]
    for group in present:
        sub = scores[scores[group_col] == group]
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            s=18,
            alpha=0.72,
            color=COLORS.get(group, "#333333"),
            edgecolor="white",
            linewidth=0.25,
            label=f"{group} (n={len(sub)})",
        )
    pc1 = float(variance.loc[variance["component"] == "PC1", "explained_variance_percent"].iloc[0])
    pc2 = float(variance.loc[variance["component"] == "PC2", "explained_variance_percent"].iloc[0])
    ax.set_xlabel(f"PC1 ({pc1:.1f}%)", fontsize=9)
    ax.set_ylabel(f"PC2 ({pc2:.1f}%)", fontsize=9)
    ax.set_title(f"PCA by {group_col}", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.6, labelsize=8)
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{filename}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build(cohort_csv: Path, out_root: Path, manifest_csv: Path = DEFAULT_MANIFEST) -> None:
    fig_dir = out_root / "figures"
    csv_dir = out_root / "csv"
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    subject, wavenumbers, feature_cols = _load_subject_spectra(cohort_csv)
    mean_table = _mean_ci_table(subject, wavenumbers, feature_cols)
    stage_table, stage_counts = _stage_mean_ci_table(
        subject, manifest_csv, wavenumbers, feature_cols
    )
    stage_peak_table = _top_stage_delta_peaks(stage_table)
    scores, variance = _pca_scores(subject, feature_cols)

    subject[["subject_id", "source_group", "sample_id_str", "model_group", "display_group"]].to_csv(
        csv_dir / "spectrum_subject_manifest.csv", index=False, encoding="utf-8-sig"
    )
    mean_table.to_csv(
        csv_dir / "mean_spectra_95ci_by_model_group.csv", index=False, encoding="utf-8-sig"
    )
    stage_table.to_csv(
        csv_dir / "stage_spectra_95ci_by_cancer.csv", index=False, encoding="utf-8-sig"
    )
    stage_counts.to_csv(
        csv_dir / "stage_spectrum_subject_counts.csv", index=False, encoding="utf-8-sig"
    )
    stage_peak_table.to_csv(
        csv_dir / "stage_spectra_top_delta_peaks.csv", index=False, encoding="utf-8-sig"
    )
    scores.to_csv(csv_dir / "pca_subject_scores.csv", index=False, encoding="utf-8-sig")
    variance.to_csv(csv_dir / "pca_explained_variance.csv", index=False, encoding="utf-8-sig")

    plot_mean_spectra(mean_table, fig_dir)
    plot_stage_spectra(stage_table, stage_peak_table, fig_dir)
    _scatter_pca(
        scores,
        variance,
        group_col="display_group",
        order=DISPLAY_ORDER,
        filename="pca_by_model_group",
        out_dir=fig_dir,
    )
    _scatter_pca(
        scores,
        variance,
        group_col="source_group",
        order=SOURCE_ORDER,
        filename="pca_by_source_group",
        out_dir=fig_dir,
    )

    print(f"Output figures: {fig_dir}")
    print(f"Output CSV: {csv_dir}")
    print(variance.head(2).to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-csv", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(args.cohort_csv, args.out_root, args.manifest_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
