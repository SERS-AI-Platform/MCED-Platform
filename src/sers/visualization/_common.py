from __future__ import annotations

"""Shared helpers, constants, and styling utilities for visualization."""

from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CANCER_GROUPS = ("PRO", "OVA", "LUN", "CRC", "PAN", "BLC", "BRE")

# Publication-quality font sizes
PAPER_TITLE_SIZE = 18
PAPER_LABEL_SIZE = 15
PAPER_TICK_SIZE = 12
PAPER_LEGEND_SIZE = 11
PAPER_ANNOTATION_SIZE = 10
PAPER_LINEWIDTH = 2.2

# Uniform DPI for all figures
DEFAULT_DPI = 300


# ---------------------------------------------------------------------------
# File / path helpers
# ---------------------------------------------------------------------------

def ensure_output_dir(output_path: Path) -> None:
    """Create parent directories for *output_path* if they don't exist."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Feature-column helpers
# ---------------------------------------------------------------------------

def extract_feature_columns(df: pd.DataFrame, prefix: str = "x_") -> List[str]:
    """Return column names that start with *prefix*."""
    return [c for c in df.columns if str(c).startswith(prefix)]


def feature_axis_from_names(feature_names: Sequence[str]) -> np.ndarray:
    """Convert feature column names like ``x_401.81`` to a float wavenumber array."""
    axis = []
    for i, name in enumerate(feature_names):
        try:
            axis.append(float(str(name).split("_", 1)[1]))
        except (IndexError, ValueError):
            axis.append(float(i))
    return np.asarray(axis, dtype=float)


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

def apply_publication_style(
    ax: plt.Axes,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """Apply consistent publication-quality styling to an axes."""
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=PAPER_LABEL_SIZE, labelpad=8)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=PAPER_LABEL_SIZE, labelpad=8)
    if title is not None:
        ax.set_title(title, fontsize=PAPER_TITLE_SIZE, fontweight="bold", pad=12)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=PAPER_TICK_SIZE,
        width=1.5,
        length=5,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_alpha(0.85)


def save_figure(
    fig: plt.Figure,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Save *fig* to *output_path* with tight layout, then close."""
    ensure_output_dir(output_path)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {Path(output_path).name}")


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def annotate_peak_labels(
    ax: plt.Axes,
    peak_df: pd.DataFrame,
    value_col: str,
    color: str = "#111111",
) -> None:
    """Add wavenumber annotations above peaks in *peak_df*."""
    for idx, row in enumerate(peak_df.itertuples(index=False)):
        y_offset = 16 + (idx % 2) * 12
        ax.annotate(
            f"{row.wavenumber:.0f}",
            xy=(row.wavenumber, getattr(row, value_col)),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=PAPER_ANNOTATION_SIZE,
            color=color,
            bbox={
                "boxstyle": "round,pad=0.2",
                "fc": "white",
                "ec": "none",
                "alpha": 0.75,
            },
        )


# ---------------------------------------------------------------------------
# SHAP value helpers
# ---------------------------------------------------------------------------

def normalize_shap_values(shap_values: Any) -> List[np.ndarray]:
    """Normalize heterogeneous SHAP value formats to ``List[np.ndarray]``."""
    if isinstance(shap_values, list):
        return [np.asarray(v) for v in shap_values]

    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 2:
        return [shap_values]
    if shap_values.ndim == 3:
        return [shap_values[:, :, c] for c in range(shap_values.shape[-1])]
    raise ValueError(f"Unsupported SHAP value shape: {shap_values.shape}")
