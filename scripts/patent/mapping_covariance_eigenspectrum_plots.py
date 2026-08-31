from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mapping_covariance_eigenspectrum_core import ExperimentResult


def write_plots(out: Path, result: ExperimentResult) -> None:
    """Write eigen-spectrum, cumulative, profile, and eigenmode figures."""
    within, corrected = result.within, result.between_corrected
    components = np.arange(1, len(result.grid) + 1)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.semilogy(components, np.maximum(within.eigenvalues, 1e-18), label="Within-repeat")
    axis.semilogy(components, np.maximum(corrected.eigenvalues, 1e-18), label="Between-subject corrected")
    axis.set(xlabel="Component", ylabel="Eigenvalue (log scale)", title="Covariance eigen-spectrum")
    axis.legend()
    fig.tight_layout()
    fig.savefig(out / "01_covariance_eigenspectrum.png", dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(components, within.cumulative, label="Within-repeat")
    axis.plot(components, corrected.cumulative, label="Between-subject corrected")
    axis.axhline(0.95, color="black", linestyle="--", linewidth=0.8)
    axis.set(xlabel="Component", ylabel="Cumulative explained variance", ylim=(0, 1.02), title="Cumulative covariance variance")
    axis.legend()
    fig.tight_layout()
    fig.savefig(out / "02_covariance_cumulative_variance.png", dpi=180)
    plt.close(fig)
    mean_noise_sd = within.diagonal_sd * np.sqrt(result.mean_inverse_repeats)
    corrected_sd = np.sqrt(np.maximum(np.diag(result.corrected_between_covariance), 0.0))
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(result.grid, mean_noise_sd, label="Noise SD of subject mean")
    axis.plot(result.grid, corrected_sd, label="Corrected between-subject SD")
    axis.set(xlabel="Wavenumber (cm⁻¹)", ylabel="SD", title="Wavenumber-wise covariance profile")
    axis.legend()
    fig.tight_layout()
    fig.savefig(out / "03_wavenumber_covariance_profile.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for component in range(5):
        axes[0].plot(result.grid, within.eigenvectors[:, component], label=f"PC{component + 1}")
        axes[1].plot(result.grid, corrected.eigenvectors[:, component], label=f"PC{component + 1}")
    axes[0].set_title("Top within-repeat covariance modes")
    axes[1].set_title("Top corrected between-subject covariance modes")
    axes[1].set_xlabel("Wavenumber (cm⁻¹)")
    for axis in axes:
        axis.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "04_top_covariance_modes.png", dpi=180)
    plt.close(fig)
