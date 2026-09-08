from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mapping_repeat_average_core import GROUP_ORDER, SubjectResult
from mapping_repeat_average_outputs import OutputTables

COLORS = {"Control": "#2C7FB8", "Prostate disease control": "#7A5195", "Prostate cancer": "#D95F02"}


def save_figure(fig: plt.Figure, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_spectra(out: Path, grid: np.ndarray, results: list[SubjectResult]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for axis, group in zip(axes, GROUP_ORDER, strict=True):
        selected = [r for r in results if r.subject.group == group]
        existing = np.vstack([r.subject.existing_average for r in selected]).mean(axis=0)
        patent = np.vstack([r.patent_average for r in selected]).mean(axis=0)
        axis.plot(grid, existing, color="#777777", lw=1.0, label="Existing _ave")
        axis.plot(grid, patent, color=COLORS[group], lw=1.35, label="Patent QC average")
        axis.plot(
            grid, patent - existing, color="#111111", lw=0.7, alpha=0.65, label="Patent − existing"
        )
        axis.set_ylabel("relative intensity")
        axis.set_title(f"{group} (n={len(selected)})", loc="left", fontsize=11)
        axis.grid(alpha=0.18)
        axis.legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel("Raman shift (cm$^{-1}$)")
    fig.suptitle(
        "Existing instrument average versus patent QC average",
        x=0.02,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, out, "fig01_mapping_existing_vs_patent_spectra")


def plot_noise(out: Path, results: list[SubjectResult]) -> None:
    fig, axis = plt.subplots(figsize=(10, 5.8))
    positions = np.arange(len(GROUP_ORDER))
    width = 0.22
    for offset, label, attr, color in (
        (-width, "Single repeat", "single_hf_noise", "#999999"),
        (0, "Existing _ave", "existing_hf_noise", "#555555"),
        (width, "Patent QC average", "patent_hf_noise", "#D95F02"),
    ):
        data = [
            [getattr(r, attr) for r in results if r.subject.group == group] for group in GROUP_ORDER
        ]
        axis.boxplot(
            data,
            positions=positions + offset,
            widths=width * 0.8,
            patch_artist=True,
            boxprops={"facecolor": color, "alpha": 0.7},
            medianprops={"color": "black"},
            showfliers=False,
        )
    axis.set_xticks(positions, GROUP_ORDER)
    axis.set_ylabel("high-frequency noise scale")
    axis.set_title("Noise scale before and after repeat averaging")
    axis.legend(
        [plt.Line2D([0], [0], color=c, lw=7) for c in ("#999999", "#555555", "#D95F02")],
        ["Single repeat", "Existing _ave", "Patent QC average"],
        frameon=False,
    )
    axis.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    save_figure(fig, out, "fig02_mapping_noise_reduction")


def plot_peaks(out: Path, results: list[SubjectResult]) -> None:
    fig, axis = plt.subplots(figsize=(10, 5.8))
    positions = np.arange(len(GROUP_ORDER))
    width = 0.18
    metrics = (
        ("Raw median", [r.raw_peak_count_median for r in results], "#999999"),
        ("Existing total", [len(r.existing_peaks) for r in results], "#555555"),
        (
            "Existing reproducible",
            [sum(p.support_fraction >= 0.5 for p in r.existing_peaks) for r in results],
            "#777777",
        ),
        ("Patent total", [len(r.patent_peaks) for r in results], "#D95F02"),
        (
            "Patent reproducible",
            [sum(p.support_fraction >= 0.5 for p in r.patent_peaks) for r in results],
            "#F39C12",
        ),
    )
    for index, (label, values, color) in enumerate(metrics):
        medians = [
            np.median(
                [
                    value
                    for value, r in zip(values, results, strict=True)
                    if r.subject.group == group
                ]
            )
            for group in GROUP_ORDER
        ]
        axis.bar(positions + (index - 2) * width, medians, width=width, label=label, color=color)
    axis.set_xticks(positions, GROUP_ORDER)
    axis.set_ylabel("median number of detected peaks")
    axis.set_title("Peak detectability and repeat support")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    axis.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    save_figure(fig, out, "fig03_mapping_peak_detectability")


def plot_partial_average(out: Path, tables: OutputTables) -> None:
    rows = [row for row in tables.partial_rows if row["group"] == "All included"]
    ns = [int(row["n"]) for row in rows]
    empirical = [float(row["empirical_noise_scale_median"]) for row in rows]
    expected = [float(row["expected_noise_scale_median"]) for row in rows]
    fig, axis = plt.subplots(figsize=(8.5, 5.5))
    axis.plot(ns, expected, marker="o", lw=1.6, label="Expected 1/√n")
    axis.plot(ns, empirical, marker="x", ls="--", lw=1.0, label="Empirical subset noise")
    axis.set(
        xlabel="number of QC-passed repeats averaged (n)",
        ylabel="median noise scale",
        title="Partial averaging effect across mapping subjects",
    )
    axis.set_xscale("log")
    axis.grid(alpha=0.18)
    axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out, "fig04_mapping_partial_average_effect")


def write_plots(
    out: Path, grid: np.ndarray, results: list[SubjectResult], tables: OutputTables
) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    plot_spectra(out, grid, results)
    plot_noise(out, results)
    plot_peaks(out, results)
    plot_partial_average(out, tables)
