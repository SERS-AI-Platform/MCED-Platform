#!/usr/bin/env python3
"""Create manuscript-ready 3D SERS spectral maps for Cancers figures.

The spectral map uses one subject-mean spectrum per patient:
5 replicate spectra -> mean spectrum -> one 3D line.

Outputs are written next to this script in publications/cancers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

PROCESSED_SPECTRA = ROOT / "results" / "processed_spectra.csv"
COHORT_MANIFEST = (
    ROOT
    / "publications"
    / "대한암학회_20260610_pan_combined_split"
    / "csv"
    / "cohort_subject_manifest.csv"
)
PCA_SCORES = (
    ROOT
    / "publications"
    / "대한암학회_20260610_pan_combined_split"
    / "csv"
    / "pca_subject_scores.csv"
)

GROUP_ORDER = ["CONTROL", "PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "BLC"]
GROUP_LABELS = {
    "CONTROL": "Control",
    "PRO": "Prostate",
    "BRE": "Breast",
    "OVA": "Ovarian",
    "LUN": "Lung",
    "CRC": "Colorectal",
    "PAN": "Pancreatic",
    "BLC": "Bladder",
}
GROUP_COLORS = {
    "CONTROL": "#6b7280",
    "PRO": "#d55e00",
    "BRE": "#cc79a7",
    "OVA": "#e69f00",
    "LUN": "#009e73",
    "CRC": "#0072b2",
    "PAN": "#8f6f00",
    "BLC": "#7e57c2",
}


def configure_3d_axes(ax: plt.Axes) -> None:
    ax.grid(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0.78, 0.78, 0.78, 0.75)
        axis._axinfo["grid"]["linewidth"] = 0.55
        axis._axinfo["axisline"]["color"] = (0.30, 0.30, 0.30, 1.0)
    ax.xaxis.pane.set_facecolor((0.96, 0.96, 0.96, 0.55))
    ax.yaxis.pane.set_facecolor((0.96, 0.96, 0.96, 0.55))
    ax.zaxis.pane.set_facecolor((0.96, 0.96, 0.96, 0.55))
    ax.tick_params(axis="both", which="major", labelsize=8, pad=0)


def save_figure(fig: plt.Figure, stem: str) -> None:
    png_path = OUT_DIR / f"{stem}.png"
    tiff_path = OUT_DIR / f"{stem}.tiff"
    pdf_path = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"[saved] {png_path}")
    print(f"[saved] {tiff_path}")
    print(f"[saved] {pdf_path}")


def load_subject_mean_spectra() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    spectra = pd.read_csv(PROCESSED_SPECTRA)
    cohort = pd.read_csv(COHORT_MANIFEST, encoding="utf-8-sig")
    cohort["display_group"] = np.where(
        cohort["analysis_group"].eq("Control"), "CONTROL", cohort["model_group"]
    )
    cohort = cohort[["source_group", "sample_id", "subject_id", "display_group"]]

    joined = spectra.merge(
        cohort,
        left_on=["group", "sample_id"],
        right_on=["source_group", "sample_id"],
        how="inner",
    )
    joined = joined[joined["display_group"].isin(GROUP_ORDER)].copy()

    wn_cols = [col for col in joined.columns if col.startswith("x_")]
    wavenumbers = np.array([float(col.replace("x_", "")) for col in wn_cols])

    subject_mean = (
        joined.groupby(["subject_id", "display_group"], sort=False)[wn_cols].mean().reset_index()
    )
    subject_mean["display_group"] = pd.Categorical(
        subject_mean["display_group"], categories=GROUP_ORDER, ordered=True
    )
    subject_mean = subject_mean.sort_values(["display_group", "subject_id"]).reset_index(drop=True)

    matrix = subject_mean[wn_cols].to_numpy(dtype=float)
    return subject_mean, wavenumbers, matrix


def normalize_rows(matrix: np.ndarray, z_max: float = 0.30) -> np.ndarray:
    row_min = np.nanmin(matrix, axis=1, keepdims=True)
    row_max = np.nanmax(matrix, axis=1, keepdims=True)
    denom = np.maximum(row_max - row_min, np.finfo(float).eps)
    return ((matrix - row_min) / denom) * z_max


def assign_subject_indices(subject_mean: pd.DataFrame, gap: int = 22) -> pd.DataFrame:
    rows = []
    current = 0
    for group in GROUP_ORDER:
        group_idx = subject_mean.index[subject_mean["display_group"].astype(str).eq(group)].tolist()
        for idx in group_idx:
            rows.append((idx, current))
            current += 1
        current += gap

    y_by_index = pd.Series({idx: y for idx, y in rows})
    subject_mean = subject_mean.copy()
    subject_mean["sample_index"] = subject_mean.index.map(y_by_index).astype(float)
    return subject_mean


def legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=GROUP_COLORS[group], lw=2.0, label=GROUP_LABELS[group])
        for group in GROUP_ORDER
    ]


def add_spectral_lines(
    ax: plt.Axes,
    subject_mean: pd.DataFrame,
    wavenumbers: np.ndarray,
    z_matrix: np.ndarray,
    *,
    linewidth: float = 0.28,
    alpha: float = 0.42,
    stride: int = 1,
) -> None:
    x = wavenumbers[::stride]
    for group in GROUP_ORDER:
        mask = subject_mean["display_group"].astype(str).eq(group).to_numpy()
        group_rows = np.where(mask)[0]
        if len(group_rows) == 0:
            continue

        segments = []
        for row_idx in group_rows:
            y = np.full_like(x, subject_mean.loc[row_idx, "sample_index"], dtype=float)
            segments.append(np.column_stack([x, y, z_matrix[row_idx, ::stride]]))

        collection = Line3DCollection(
            segments,
            colors=GROUP_COLORS[group],
            linewidths=linewidth,
            alpha=alpha,
        )
        ax.add_collection3d(collection)


def plot_spectral_map() -> None:
    subject_mean, wavenumbers, matrix = load_subject_mean_spectra()
    subject_mean = assign_subject_indices(subject_mean)
    z_matrix = normalize_rows(matrix, z_max=0.30)

    counts = (
        subject_mean.groupby("display_group", observed=True)
        .size()
        .rename("n_subjects")
        .reset_index()
    )
    counts["group_label"] = counts["display_group"].astype(str).map(GROUP_LABELS)
    counts.to_csv(
        OUT_DIR / "fig_cancers_3d_subject_mean_counts.csv", index=False, encoding="utf-8-sig"
    )

    fig = plt.figure(figsize=(9.2, 7.4))
    ax = fig.add_subplot(111, projection="3d")
    add_spectral_lines(ax, subject_mean, wavenumbers, z_matrix)

    ax.set_title("3D Spectral Map of Subject-Mean SERS Spectra", fontsize=12, pad=14)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=9, labelpad=7)
    ax.set_ylabel("Sample index", fontsize=9, labelpad=7)
    ax.set_zlabel("Normalized intensity", fontsize=9, labelpad=7)

    ax.set_xlim(float(wavenumbers.min()), float(wavenumbers.max()))
    ax.set_ylim(
        float(subject_mean["sample_index"].min()), float(subject_mean["sample_index"].max())
    )
    ax.set_zlim(0, 0.30)
    ax.view_init(elev=27, azim=-135)
    ax.set_box_aspect((1.55, 1.00, 0.52))
    configure_3d_axes(ax)

    ax.legend(
        handles=legend_handles(),
        loc="upper right",
        bbox_to_anchor=(1.13, 0.98),
        fontsize=7,
        frameon=False,
    )
    fig.tight_layout()
    save_figure(fig, "fig_cancers_3d_subject_mean_spectral_map")
    plt.close(fig)


def plot_spectral_map_stacked_view() -> None:
    subject_mean, wavenumbers, matrix = load_subject_mean_spectra()
    subject_mean = assign_subject_indices(subject_mean, gap=45)
    z_matrix = normalize_rows(matrix, z_max=0.30)

    fig = plt.figure(figsize=(10.2, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    add_spectral_lines(
        ax,
        subject_mean,
        wavenumbers,
        z_matrix,
        linewidth=0.22,
        alpha=0.34,
        stride=2,
    )

    ax.set_title("3D Stacked Spectral Map of Subject-Mean SERS Spectra", fontsize=12, pad=14)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=9, labelpad=7)
    ax.set_ylabel("Sample index", fontsize=9, labelpad=7)
    ax.set_zlabel("Normalized intensity", fontsize=9, labelpad=7)

    ax.set_xlim(float(wavenumbers.min()), float(wavenumbers.max()))
    ax.set_ylim(
        float(subject_mean["sample_index"].min()), float(subject_mean["sample_index"].max())
    )
    ax.set_zlim(0, 0.30)
    ax.view_init(elev=26, azim=-62)
    ax.set_box_aspect((1.70, 1.35, 0.55))
    configure_3d_axes(ax)

    ax.legend(
        handles=legend_handles(),
        loc="upper right",
        bbox_to_anchor=(1.13, 0.98),
        fontsize=7,
        frameon=False,
    )
    fig.tight_layout()
    save_figure(fig, "fig_cancers_3d_subject_mean_spectral_map_stacked_view")
    plt.close(fig)


def plot_pca_3d() -> None:
    pca = pd.read_csv(PCA_SCORES, encoding="utf-8-sig")
    pca = pca[pca["display_group"].isin(GROUP_ORDER)].copy()
    pca["display_group"] = pd.Categorical(
        pca["display_group"], categories=GROUP_ORDER, ordered=True
    )

    fig = plt.figure(figsize=(8.0, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    for group in GROUP_ORDER:
        group_df = pca[pca["display_group"].astype(str).eq(group)]
        if group_df.empty:
            continue
        ax.scatter(
            group_df["PC1"],
            group_df["PC2"],
            group_df["PC3"],
            s=15,
            alpha=0.72,
            color=GROUP_COLORS[group],
            edgecolor="none",
            label=GROUP_LABELS[group],
            depthshade=True,
        )

    ax.set_title("PCA: First Three Principal Components", fontsize=12, pad=14)
    ax.set_xlabel("Principal Component 1", fontsize=9, labelpad=7)
    ax.set_ylabel("Principal Component 2", fontsize=9, labelpad=7)
    ax.set_zlabel("Principal Component 3", fontsize=9, labelpad=7)
    ax.view_init(elev=27, azim=-130)
    ax.set_box_aspect((1.2, 1.0, 0.9))
    configure_3d_axes(ax)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 0.98), fontsize=7, frameon=False)

    fig.tight_layout()
    save_figure(fig, "fig_cancers_3d_pca_subject_scores")
    plt.close(fig)


def plot_combined() -> None:
    subject_mean, wavenumbers, matrix = load_subject_mean_spectra()
    subject_mean = assign_subject_indices(subject_mean)
    z_matrix = normalize_rows(matrix, z_max=0.30)
    pca = pd.read_csv(PCA_SCORES, encoding="utf-8-sig")
    pca = pca[pca["display_group"].isin(GROUP_ORDER)].copy()

    fig = plt.figure(figsize=(15.2, 7.2))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    add_spectral_lines(ax1, subject_mean, wavenumbers, z_matrix)
    ax1.set_title("A. 3D Spectral Map", fontsize=12, pad=12)
    ax1.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=9, labelpad=7)
    ax1.set_ylabel("Sample index", fontsize=9, labelpad=7)
    ax1.set_zlabel("Normalized intensity", fontsize=9, labelpad=7)
    ax1.set_xlim(float(wavenumbers.min()), float(wavenumbers.max()))
    ax1.set_ylim(
        float(subject_mean["sample_index"].min()), float(subject_mean["sample_index"].max())
    )
    ax1.set_zlim(0, 0.30)
    ax1.view_init(elev=27, azim=-135)
    ax1.set_box_aspect((1.55, 1.00, 0.52))
    configure_3d_axes(ax1)

    for group in GROUP_ORDER:
        group_df = pca[pca["display_group"].astype(str).eq(group)]
        if group_df.empty:
            continue
        ax2.scatter(
            group_df["PC1"],
            group_df["PC2"],
            group_df["PC3"],
            s=13,
            alpha=0.72,
            color=GROUP_COLORS[group],
            edgecolor="none",
            label=GROUP_LABELS[group],
            depthshade=True,
        )
    ax2.set_title("B. PCA: PC1-PC3", fontsize=12, pad=12)
    ax2.set_xlabel("Principal Component 1", fontsize=9, labelpad=7)
    ax2.set_ylabel("Principal Component 2", fontsize=9, labelpad=7)
    ax2.set_zlabel("Principal Component 3", fontsize=9, labelpad=7)
    ax2.view_init(elev=27, azim=-130)
    ax2.set_box_aspect((1.2, 1.0, 0.9))
    configure_3d_axes(ax2)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.18, 0.98), fontsize=7, frameon=False)

    fig.tight_layout()
    save_figure(fig, "fig_cancers_3d_spectral_map_and_pca")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_spectral_map()
    plot_spectral_map_stacked_view()
    plot_pca_3d()
    plot_combined()


if __name__ == "__main__":
    main()
