"""
Shared Nature-style figure configuration for SERS-AI paper.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

# ── Paths ──
AACR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(AACR_DIR))  # SERS-AI project root (publications/aacr → publications → SERS-AI)
OUTPUT_DIR = os.path.join(AACR_DIR, "figures")

# ── Color palettes (muted, Nature-style) ──
CANCER_COLORS = {
    "PRO": "#8B4513",   # warm brown
    "BRE": "#D4527A",   # rose pink
    "OVA": "#6A5ACD",   # slate blue-purple
    "LUN": "#2E8B57",   # sea green
    "CRC": "#4682B4",   # steel blue
    "PAN": "#CD5C5C",   # indian red
    "BLC": "#E8960C",   # amber orange
}

CANCER_LABELS = {
    "PRO": "Prostate",
    "BRE": "Breast",
    "OVA": "Ovarian",
    "LUN": "Lung",
    "CRC": "Colorectal",
    "PAN": "Pancreatic",
    "BLC": "Bladder",
}

# Fold prediction group → display name mapping
FOLD_GROUP_TO_DISPLAY = {
    "PRO": "PRO", "LUN": "LUN", "CRC": "CRC", "CPAN": "PAN", "OVA": "OVA",
}

NON_CANCER_COLOR = "#808080"
NON_CANCER_GROUPS = {"NOR", "DIA", "HBP", "H.D."}

MODEL_COLORS = {
    "Logistic Regression": "#2C3E50",
    "Random Forest":       "#7F8C8D",
    "XGBoost":             "#8E44AD",
    "CNN1D":               "#2980B9",
    "ResNet18":            "#C0392B",
    "Ensemble":            "#D4A017",
}

MODEL_ORDER = [
    "Ensemble", "Logistic Regression", "XGBoost",
    "ResNet18", "Random Forest", "CNN1D",
]

# ── Cancer-specific discriminative peak regions ──
PEAK_REGIONS = [
    ("PRO", 2088, 2118, "2100"),
    ("PAN", 710,  745,  "725"),
    ("OVA", 1588, 1620, "1603"),
    ("LUN", 975,  1005, "990"),
    ("CRC", 1330, 1365, "1350"),
]

# ── Typography (Nature conventions) ──
MM_TO_INCH = 1 / 25.4
SINGLE_COL = 89 * MM_TO_INCH    # ~3.5 in
DOUBLE_COL = 183 * MM_TO_INCH   # ~7.2 in

FONT_SIZE = {
    "panel_label": 12,
    "title":       9,
    "axis_label":  8,
    "tick":        7,
    "legend":      7,
    "annotation":  6.5,
}

LINE_WIDTH = {
    "spectrum": 1.0,
    "thin":     0.5,
    "spine":    0.6,
    "roc":      1.2,
}


def nature_rcparams():
    """Return rcParams dict for Nature-style figures."""
    return {
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":         FONT_SIZE["tick"],
        "axes.titlesize":    FONT_SIZE["title"],
        "axes.labelsize":    FONT_SIZE["axis_label"],
        "xtick.labelsize":   FONT_SIZE["tick"],
        "ytick.labelsize":   FONT_SIZE["tick"],
        "legend.fontsize":   FONT_SIZE["legend"],
        "axes.linewidth":    LINE_WIDTH["spine"],
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size":  3,
        "ytick.major.size":  3,
        "xtick.minor.size":  1.5,
        "ytick.minor.size":  1.5,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         False,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype":      42,   # TrueType for editability
        "ps.fonttype":       42,
    }


def apply_style():
    """Apply Nature rcParams globally."""
    mpl.rcParams.update(nature_rcparams())


def apply_nature_style(ax, xlabel=None, ylabel=None):
    """Style a single axes for Nature formatting."""
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(direction="out", length=3, width=0.5)


def add_panel_label(ax, label, x=-0.12, y=1.06):
    """Add bold lowercase panel label (a, b, c ...) at top-left."""
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=FONT_SIZE["panel_label"],
        fontweight="bold",
        va="top", ha="left",
    )


def save_figure(fig, name, formats=("pdf", "png")):
    """Save figure to output directory in specified formats."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for fmt in formats:
        path = os.path.join(OUTPUT_DIR, f"{name}.{fmt}")
        fig.savefig(path, format=fmt, facecolor="white", edgecolor="none")
        print(f"  Saved: {path}")


def load_spectra():
    """Load processed spectra and return (DataFrame, wavenumber_array, feature_columns)."""
    import pandas as pd
    spec = pd.read_csv(os.path.join(ROOT, "results", "processed_spectra.csv"))
    wn_cols = [c for c in spec.columns if c.startswith("x_")]
    wavenumbers = np.array([float(c.replace("x_", "")) for c in wn_cols])
    return spec, wavenumbers, wn_cols


def compute_group_means(spec, wn_cols, groups_map=None):
    """Compute per-group mean and SEM spectra.

    groups_map: dict mapping raw group -> display group.
        Defaults to mapping non-cancer to 'Non-Cancer' and CPAN to PAN.

    Returns dict of {group: {'mean': array, 'sem': array, 'n': int}}
    """
    if groups_map is None:
        groups_map = {}
        for g in NON_CANCER_GROUPS:
            groups_map[g] = "Non-Cancer"
        groups_map["CPAN"] = "PAN"
        # Identity for cancer groups
        for g in ["PRO", "OVA", "LUN", "CRC"]:
            groups_map[g] = g

    spec = spec.copy()
    spec["display_group"] = spec["group"].map(groups_map)
    spec = spec.dropna(subset=["display_group"])

    # Per-patient medoid (first replicate)
    spec_med = spec.groupby(["display_group", "sample_id"]).first().reset_index()

    result = {}
    for g in spec_med["display_group"].unique():
        sub = spec_med[spec_med["display_group"] == g][wn_cols].values
        result[g] = {
            "mean": sub.mean(axis=0),
            "sem": sub.std(axis=0) / np.sqrt(sub.shape[0]),
            "n": sub.shape[0],
        }
    return result
