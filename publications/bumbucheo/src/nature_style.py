"""
Figure Style — Single Source of Truth for colors, labels, and styling.

범부처 과제 figure scripts import from this module to ensure visual consistency.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Paths ──
BUMBU_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERS_ROOT = os.path.dirname(os.path.dirname(BUMBU_DIR))
FIGURE_DIR = os.path.join(BUMBU_DIR, "figures")
DATA_DIR = os.path.join(BUMBU_DIR, "data")
BENCHMARK_DIR = os.path.join(
    SERS_ROOT, "models", "results", "01_benchmarks", "main_5models"
)

# ── Internal cancer codes (data-level) ──
CANCER_COLORS = {
    "PRO": "#8B4513",   # warm brown
    "BRE": "#D4527A",   # rose pink
    "OVA": "#6A5ACD",   # slate blue-purple
    "LUN": "#2E8B57",   # sea green
    "CRC": "#4682B4",   # steel blue
    "PAN": "#CD5C5C",   # indian red
    "BLC": "#E8960C",   # amber orange
}

NON_CANCER_COLOR = "#808080"
NON_CANCER_GROUPS = {"NOR", "DIA", "HBP", "H.D."}

# ── AACR display labels (internal code -> poster abbreviation) ──
AACR_LABELS = {
    "PRO": "PRC",
    "BRE": "BRC",
    "OVA": "OVC",
    "LUN": "LC",
    "CRC": "CRC",
    "PAN": "PAC",
    "BLC": "BLC",
}

NON_CANCER_LABEL = "Non-Cancer"

# AACR 5-cancer order (poster)
AACR_CANCER_ORDER = ["PRC", "OVC", "LC", "PAC", "CRC"]
AACR_INTERNAL_ORDER = ["PRO", "OVA", "LUN", "PAN", "CRC"]

# Colors keyed by AACR display labels
AACR_COLORS = {AACR_LABELS[k]: v for k, v in CANCER_COLORS.items() if k in AACR_LABELS}
AACR_COLORS[NON_CANCER_LABEL] = NON_CANCER_COLOR

# Group code (from data) -> AACR display label
GROUP_TO_AACR = {
    "PRO": "PRC", "OVA": "OVC", "LUN": "LC", "CRC": "CRC",
    "PAN": "PAC", "CPAN": "PAC",
    "NOR": NON_CANCER_LABEL, "DIA": NON_CANCER_LABEL,
    "HBP": NON_CANCER_LABEL, "H.D.": NON_CANCER_LABEL,
    "YNOR": NON_CANCER_LABEL,
}

# ── Model colors ──
# Our model name
OUR_MODEL = "uSERS-Net"

MODEL_COLORS = {
    "Logistic Regression": "#2C3E50",
    "Random Forest":       "#8E44AD",
    "XGBoost":             "#2980B9",
    "CNN1D":               "#E67E22",
    "ResNet18":            "#27AE60",
    OUR_MODEL:             "#C0392B",
}
MODEL_ORDER = list(MODEL_COLORS.keys())

# ── Typography ──
FONT_SIZE = {
    "title":       8,
    "axis_label":  7,
    "tick":        6,
    "legend":      6,
    "annotation":  5.5,
    "panel_label": 9,
}

LINE_WIDTH = {
    "spectrum": 1.0,
    "thin":     0.6,
    "axis":     0.5,
}

# Nature figure widths (mm -> inches)
SINGLE_COL = 89 / 25.4   # ~3.50 in
DOUBLE_COL = 183 / 25.4   # ~7.20 in

# ── Spectral grid ──
FULL_GRID = np.linspace(402.0, 2198.0, 935)
MODEL_GRID = FULL_GRID[1:-1]  # 933 features, matches model input

# ── SHAP peak regions (wavenumber ranges per cancer) ──
PEAK_REGIONS = [
    ("PRO", 2088, 2118, "2100"),
    ("PAN", 710,  745,  "725"),
    ("OVA", 1588, 1620, "1603"),
    ("LUN", 975,  1005, "990"),
    ("CRC", 1330, 1365, "1350"),
]


# ═══════════════════════════════════════════
# Style functions
# ═══════════════════════════════════════════

def nature_rcparams():
    """Return rcParams dict for Nature-style figures."""
    return {
        "font.family": "DejaVu Sans",
        "font.size": FONT_SIZE["tick"],
        "axes.labelsize": FONT_SIZE["axis_label"],
        "axes.titlesize": FONT_SIZE["title"],
        "axes.linewidth": LINE_WIDTH["axis"],
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": LINE_WIDTH["spectrum"],
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "legend.frameon": False,
        "legend.fontsize": FONT_SIZE["legend"],
    }


def apply_style():
    """Apply Nature rcParams globally."""
    mpl.rcParams.update(nature_rcparams())


def apply_nature_style(ax):
    """Clean up a single axes: remove top/right spines, thin ticks."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(LINE_WIDTH["axis"])
    ax.spines["bottom"].set_linewidth(LINE_WIDTH["axis"])
    ax.tick_params(axis="both", which="major",
                   labelsize=FONT_SIZE["tick"], width=0.4, length=3)


def add_panel_label(ax, label, x=-0.08, y=1.08):
    """Add bold panel label (a, b, c, ...) to an axes."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=FONT_SIZE["panel_label"], fontweight="bold",
            va="top", ha="left")


def save_figure(fig, name, formats=("png", "pdf")):
    """Save figure to AACR/figures/ in multiple formats."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    for fmt in formats:
        path = os.path.join(FIGURE_DIR, f"{name}.{fmt}")
        fig.savefig(path, dpi=300, facecolor="none", edgecolor="none", transparent=True)
    print(f"Saved: {name} ({', '.join(formats)})")


# ═══════════════════════════════════════════
# Data loading utilities
# ═══════════════════════════════════════════

def load_spectra():
    """
    Load preprocessed spectra from benchmark fold_predictions.npz.

    Returns:
        spec: DataFrame with columns [group, sample_id, x_402.00, x_403.92, ...]
        wavenumbers: ndarray of full grid (935 points)
        wn_cols: list of wavenumber column names
    """
    npz_path = os.path.join(
        BENCHMARK_DIR, "logistic_regression", "v004", "fold_predictions.npz"
    )
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]            # (6200, 933) — model grid
    groups = data["groups"]   # (6200,)
    sample_ids = data["sample_ids"]  # (6200,)

    # Pad to full grid: duplicate edge features for 935-point grid
    X_full = np.column_stack([X[:, :1], X, X[:, -1:]])  # (6200, 935)

    wn_cols = [f"x_{wn:.2f}" for wn in FULL_GRID]
    spec = pd.DataFrame(X_full, columns=wn_cols)
    spec["group"] = groups.astype(str)
    spec["sample_id"] = sample_ids.astype(int)

    return spec, FULL_GRID, wn_cols


def compute_group_means(spec, wn_cols, groups_map):
    """
    Compute per-group mean and SEM spectra.

    Args:
        spec: DataFrame from load_spectra()
        wn_cols: wavenumber column names
        groups_map: dict mapping raw group -> display group

    Returns:
        dict of {display_group: {"mean": ndarray, "sem": ndarray, "n": int}}
    """
    spec = spec.copy()
    spec["display_group"] = spec["group"].map(groups_map)
    spec = spec.dropna(subset=["display_group"])

    # Take medoid (first per sample) to avoid replicate bias
    medoid = spec.groupby(["display_group", "sample_id"]).first().reset_index()

    stats = {}
    for grp, sub in medoid.groupby("display_group"):
        vals = sub[wn_cols].values
        stats[grp] = {
            "mean": vals.mean(axis=0),
            "sem": vals.std(axis=0) / np.sqrt(len(vals)),
            "n": len(sub),
        }
    return stats


# ═══════════════════════════════════════════
# JSON config export (for R scripts)
# ═══════════════════════════════════════════

def export_config_json(path=None):
    """Export color/label config as JSON for R scripts."""
    if path is None:
        path = os.path.join(DATA_DIR, "figure_config.json")

    config = {
        "cancer_colors": CANCER_COLORS,
        "aacr_labels": AACR_LABELS,
        "aacr_colors": {k: v for k, v in AACR_COLORS.items()
                        if k != NON_CANCER_LABEL},
        "non_cancer_color": NON_CANCER_COLOR,
        "non_cancer_label": NON_CANCER_LABEL,
        "aacr_cancer_order": AACR_CANCER_ORDER,
        "aacr_internal_order": AACR_INTERNAL_ORDER,
        "group_to_aacr": GROUP_TO_AACR,
        "model_colors": MODEL_COLORS,
        "model_order": MODEL_ORDER,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config exported: {path}")
    return path


if __name__ == "__main__":
    export_config_json()
    print("nature_style.py: config export OK")
