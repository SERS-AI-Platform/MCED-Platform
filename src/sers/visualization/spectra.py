from __future__ import annotations

"""Spectrum visualization: raw, preprocessed, mean overlay."""

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._common import (
    PAPER_LEGEND_SIZE,
    PAPER_LINEWIDTH,
    PAPER_TICK_SIZE,
    apply_publication_style,
    feature_axis_from_names,
    save_figure,
)

# ---------------------------------------------------------------------------
# Mean-spectrum profile builder (used by multiple plots)
# ---------------------------------------------------------------------------

def build_mean_spectrum_profile(
    spectra: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Build a full-spectrum summary with mean and std along the spectral axis."""
    spectra = np.asarray(spectra, dtype=float)
    if spectra.ndim != 2 or spectra.shape[0] == 0:
        raise ValueError("Expected a 2D array with at least one sample")

    n_features = spectra.shape[1]
    if feature_names is None:
        feature_names = [f"x_{i}" for i in range(n_features)]

    axis = feature_axis_from_names(feature_names)
    return pd.DataFrame({
        "feature": list(feature_names),
        "wavenumber": axis,
        "mean_intensity": spectra.mean(axis=0),
        "std_intensity": spectra.std(axis=0),
    }).sort_values("wavenumber", ascending=True, ignore_index=True)


# ---------------------------------------------------------------------------
# Raw spectra — grouped by replicate
# ---------------------------------------------------------------------------

def visualize_raw_spectra(
    raw_spectra: Dict,
    output_dir: Path,
    group: Optional[str] = None,
) -> None:
    """
    Visualize raw spectra grouped by replicate number.

    Saves one PNG per replicate to *output_dir*.
    """
    replicate_data: Dict = {}
    for key in raw_spectra:
        g, sid, replicate = key
        if group and g != group:
            continue
        rep_key = (g, replicate) if group else replicate
        replicate_data.setdefault(rep_key, []).append((key, raw_spectra[key]))

    for rep_key, spectra_list in sorted(replicate_data.items()):
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = plt.cm.viridis(np.linspace(0, 1, len(spectra_list)))

        for j, (key, y_raw) in enumerate(sorted(spectra_list)):
            g, sid, rep = key
            ax.plot(y_raw, color=colors[j], linewidth=1, alpha=0.6,
                    label=f"S{sid} Rep {rep}")

        if isinstance(rep_key, tuple):
            g, rep = rep_key
            title = f"{g} - Replicate {rep} - Raw Spectra"
            filename = f"raw_spectra_{g}_rep{rep}.png"
        else:
            rep = rep_key
            title = f"All Groups - Replicate {rep} - Raw Spectra"
            filename = f"raw_spectra_allgroups_rep{rep}.png"

        apply_publication_style(
            ax,
            xlabel="Wavenumber (cm\u207b\u00b9)",
            ylabel="Intensity (a.u.)",
            title=title,
        )
        ax.legend(loc="upper right", fontsize=PAPER_LEGEND_SIZE)
        ax.grid(True, alpha=0.3)
        save_figure(fig, Path(output_dir) / filename)


# ---------------------------------------------------------------------------
# Preprocessed spectra — grouped by replicate
# ---------------------------------------------------------------------------

def visualize_preprocessed_spectra_by_replicate(
    processed_spectra: Dict,
    grid: np.ndarray,
    output_dir: Path,
    group: Optional[str] = None,
) -> None:
    """
    Visualize preprocessed spectra grouped by replicate number.

    Saves one PNG per replicate to *output_dir*.
    """
    replicates_data: Dict = {}
    for key in processed_spectra:
        g, sid, replicate = key
        if group and g != group:
            continue
        rep_key = (g, replicate) if group else replicate
        replicates_data.setdefault(rep_key, []).append((key, processed_spectra[key]))

    for rep_key, spectra_list in sorted(replicates_data.items()):
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = plt.cm.viridis(np.linspace(0, 1, len(spectra_list)))

        for j, (key, y_proc) in enumerate(spectra_list):
            g, sid, rep = key
            ax.plot(grid, y_proc, color=colors[j], linewidth=1.5, alpha=0.6,
                    label=f"{g} S{sid}")

        if isinstance(rep_key, tuple):
            g, rep = rep_key
            title = f"{g} - All Samples (Replicate {rep})"
            filename = f"preprocessed_spectra_{g}_replicate_{rep}.png"
        else:
            rep = rep_key
            title = f"All Groups - Replicate {rep}"
            filename = f"preprocessed_spectra_all_replicate_{rep}.png"

        apply_publication_style(
            ax,
            xlabel="Wavenumber (cm\u207b\u00b9)",
            ylabel="Intensity (a.u.)",
            title=title,
        )
        ncol = 3 if len(spectra_list) > 10 else 1
        ax.legend(loc="upper right", fontsize=PAPER_TICK_SIZE, ncol=ncol)
        ax.grid(True, alpha=0.3)
        save_figure(fig, Path(output_dir) / filename)


# ---------------------------------------------------------------------------
# Raw spectra — one sample per file, all replicates together
# ---------------------------------------------------------------------------

def visualize_raw_spectra_by_sample(
    raw_spectra: Dict,
    output_dir: Path,
    group: Optional[str] = None,
    n_examples: int = 5,
) -> None:
    """
    Visualize raw spectra per sample with all replicates overlaid.

    Saves up to *n_examples* PNGs to *output_dir*.
    """
    samples: Dict = {}
    for key in raw_spectra:
        g, sid, replicate = key
        if group and g != group:
            continue
        samples.setdefault((g, sid), []).append(key)

    for (g, sid), replicate_keys in list(samples.items())[:n_examples]:
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = plt.cm.viridis(np.linspace(0, 1, len(replicate_keys)))

        for j, key in enumerate(sorted(replicate_keys)):
            x_raw, y_raw = raw_spectra[key]
            rep = key[2]
            ax.plot(x_raw, y_raw, color=colors[j], linewidth=2, alpha=0.7,
                    label=f"Replicate {rep}")

        apply_publication_style(
            ax,
            xlabel="Raman Shift (cm\u207b\u00b9)",
            ylabel="Intensity (a.u.)",
            title=f"{g} Sample {sid} - Raw Spectra",
        )
        ax.legend(loc="upper right", fontsize=PAPER_LEGEND_SIZE)
        ax.grid(True, alpha=0.3)
        save_figure(fig, Path(output_dir) / f"raw_spectra_{g}_{sid}.png")


# ---------------------------------------------------------------------------
# Mean spectrum — single class
# ---------------------------------------------------------------------------

def plot_mean_spectrum(
    spectra: np.ndarray,
    output_path: Path,
    feature_names: Optional[Sequence[str]] = None,
    title: str = "Mean Spectrum",
    color: str = "#c0392b",
) -> pd.DataFrame:
    """Plot mean spectrum with +/-1 std band."""
    profile = build_mean_spectrum_profile(spectra, feature_names)

    x = profile["wavenumber"].to_numpy()
    mean_y = profile["mean_intensity"].to_numpy()
    std_y = profile["std_intensity"].to_numpy()

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(x, mean_y, color=color, linewidth=PAPER_LINEWIDTH)
    ax.fill_between(x, mean_y - std_y, mean_y + std_y, color=color, alpha=0.16)
    apply_publication_style(
        ax,
        xlabel="Raman Shift (cm\u207b\u00b9)",
        ylabel="Mean Intensity (a.u.)",
        title=title,
    )
    ax.grid(True, alpha=0.25)
    save_figure(fig, output_path)
    return profile


# ---------------------------------------------------------------------------
# Mean spectra overlay — multiple classes
# ---------------------------------------------------------------------------

def plot_mean_spectra_overlay(
    spectra_by_class: Sequence[np.ndarray],
    class_names: Sequence[str],
    output_path: Path,
    feature_names: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    title: str = "Mean Spectra by Diagnosis",
) -> pd.DataFrame:
    """Plot one mean spectrum per class on a shared axis."""
    rows: List[pd.DataFrame] = []
    fig, ax = plt.subplots(figsize=(13, 5.5))

    for idx, (spectra, class_name) in enumerate(zip(spectra_by_class, class_names)):
        spectra = np.asarray(spectra, dtype=float)
        if spectra.ndim != 2 or spectra.shape[0] == 0:
            continue
        profile = build_mean_spectrum_profile(spectra, feature_names)
        color = colors[idx] if colors is not None and idx < len(colors) else None
        ax.plot(
            profile["wavenumber"],
            profile["mean_intensity"],
            linewidth=PAPER_LINEWIDTH - 0.2,
            color=color,
            label=f"{class_name} (n={spectra.shape[0]})",
        )
        profile.insert(0, "diagnosis", class_name)
        profile.insert(1, "n_samples", int(spectra.shape[0]))
        rows.append(profile)

    if not rows:
        raise ValueError("No class spectra available for overlay plot")

    apply_publication_style(
        ax,
        xlabel="Raman Shift (cm\u207b\u00b9)",
        ylabel="Mean Intensity (a.u.)",
        title=title,
    )
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=PAPER_LEGEND_SIZE, ncol=2, frameon=True)
    save_figure(fig, output_path)
    return pd.concat(rows, ignore_index=True)
