from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from boramae_repeat_average_core import GROUP_ORDER

COLORS = {"Control": "#2C7FB8", "Biopsy-negative": "#7A5195", "Prostate cancer": "#D95F02"}


def save_figure(fig: plt.Figure, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_repeatability(out: Path, grid: np.ndarray, subjects: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for axis, group in zip(axes, GROUP_ORDER, strict=True):
        group_subjects = [s for s in subjects if s["group"] == group]
        median_noise = np.median([s["noise_floor"] for s in group_subjects])
        selected = min(group_subjects, key=lambda s: abs(s["noise_floor"] - median_noise))[
            "selected"
        ]
        for row in selected:
            axis.plot(grid, row, color=COLORS[group], alpha=0.22, lw=0.55)
        mean = selected.mean(axis=0)
        sd = selected.std(axis=0, ddof=1)
        axis.plot(
            grid, mean, color=COLORS[group], lw=1.6, label=f"{group}: QC mean (n={len(selected)})"
        )
        axis.fill_between(grid, mean - sd, mean + sd, color=COLORS[group], alpha=0.14, linewidth=0)
        axis.set_ylabel("relative intensity")
        axis.legend(frameon=False)
        axis.grid(alpha=0.18)
    axes[-1].set_xlabel("Raman shift (cm$^{-1}$)")
    fig.suptitle(
        "Boramae prospective repeat spectra and average representative spectrum",
        x=0.02,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, out, "fig01_repeatability_examples")


def plot_group_means(out: Path, grid: np.ndarray, subjects: list[dict[str, object]]) -> None:
    fig, axis = plt.subplots(figsize=(12, 5.8))
    for group in GROUP_ORDER:
        selected = [s for s in subjects if s["group"] == group]
        means = np.vstack([s["mean"] for s in selected])
        mean = means.mean(axis=0)
        sem = means.std(axis=0, ddof=1) / math.sqrt(len(means))
        axis.plot(grid, mean, color=COLORS[group], lw=1.5, label=f"{group} (n={len(means)})")
        axis.fill_between(
            grid, mean - 1.96 * sem, mean + 1.96 * sem, color=COLORS[group], alpha=0.13, linewidth=0
        )
    axis.set(
        xlabel="Raman shift (cm$^{-1}$)",
        ylabel="relative intensity",
        title="Group-level mean of subject-level average representative spectra",
    )
    axis.legend(frameon=False, ncol=3)
    axis.grid(alpha=0.18)
    fig.tight_layout()
    save_figure(fig, out, "fig02_group_mean_representative_spectra")


def plot_partial_average(out: Path, partial: list[dict[str, object]]) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 5.5))
    for group in (*GROUP_ORDER, "All included"):
        rows = [r for r in partial if r["group"] == group]
        ns = [r["n"] for r in rows]
        expected = [r["expected_noise_scale_median"] for r in rows]
        observed = np.asarray([r["empirical_subset_noise_scale_median"] for r in rows], dtype=float)
        axis.plot(ns, expected, marker="o", lw=1.6, label=group)
        if np.isfinite(observed).any():
            axis.plot(ns, observed, marker="x", ls="--", lw=0.9, alpha=0.5)
    axis.set(
        xticks=range(1, 6),
        xlabel="number of QC-passed repeats averaged (n)",
        ylabel="median noise scale / relative intensity",
        title="Repeat averaging effect: expected noise scale follows 1/√n",
    )
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    save_figure(fig, out, "fig03_partial_average_noise_reduction")


def plot_autocorrelation(out: Path, covariance_rows: list[dict[str, object]]) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 5.5))
    for group in GROUP_ORDER:
        rows = [r for r in covariance_rows if r["group"] == group]
        axis.plot(
            [r["lag_cm-1"] for r in rows],
            [r["median_autocorrelation"] for r in rows],
            marker="o",
            color=COLORS[group],
            label=group,
        )
    axis.axhline(0, color="#777", lw=0.8)
    axis.set(
        xlabel="lag along common wavenumber grid (cm$^{-1}$)",
        ylabel="median residual autocorrelation",
        title="Residual autocorrelation by clinical group",
    )
    axis.grid(alpha=0.18)
    axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out, "fig04_residual_autocorrelation")


def write_plots(
    out: Path,
    grid: np.ndarray,
    subjects: list[dict[str, object]],
    partial: list[dict[str, object]],
    covariance_rows: list[dict[str, object]],
) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plot_repeatability(out, grid, subjects)
    plot_group_means(out, grid, subjects)
    plot_partial_average(out, partial)
    plot_autocorrelation(out, covariance_rows)
