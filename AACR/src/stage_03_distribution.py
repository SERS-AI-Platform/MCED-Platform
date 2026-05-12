"""Stage analysis - Step 3: Patient-count distribution per cancer x stage group.

Output: AACR/figures/fig_stage_distribution.{png,pdf}
Stacked horizontal bar chart; numbers annotated inside each segment.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path("/home/user/SERS-AI")
DATA = ROOT / "AACR/data/stage_clinical_unique.csv"
OUT_DIR = ROOT / "AACR/figures"

CANCERS = ["CRC", "CPAN", "PRO", "OVA", "LUN"]
CANCER_LABEL = {
    "CRC": "Colorectal (CRC)",
    "CPAN": "Pancreatic (PAC)",
    "PRO": "Prostate (PRC)",
    "OVA": "Ovarian (OVC)",
    "LUN": "Lung (LC)",
}
STAGES = ["early", "advanced", "unknown"]
STAGE_LABEL = {
    "early": "Early (I–II)",
    "advanced": "Advanced (III–IV)",
    "unknown": "Unknown / X",
}
STAGE_COLOR = {"early": "#2563eb", "advanced": "#dc2626", "unknown": "#9ca3af"}

mpl.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 22,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "axes.linewidth": 1.4,
})


def main() -> None:
    cli = pd.read_csv(DATA)
    # cross-tab
    ct = cli.groupby(["group", "cancer_stage_group"]).size().unstack(fill_value=0)
    ct = ct.reindex(CANCERS).fillna(0).astype(int)
    for s in STAGES:
        if s not in ct.columns:
            ct[s] = 0
    ct = ct[STAGES]
    ct["total"] = ct.sum(axis=1)

    fig, ax = plt.subplots(figsize=(16, 8))
    ys = np.arange(len(CANCERS))[::-1]
    left = np.zeros(len(CANCERS))
    for stage in STAGES:
        vals = ct[stage].values
        bars = ax.barh(ys, vals, left=left,
                       color=STAGE_COLOR[stage], edgecolor="white",
                       linewidth=1.2, label=STAGE_LABEL[stage])
        for y, v, l in zip(ys, vals, left):
            if v > 0:
                ax.text(l + v / 2, y, str(int(v)),
                        va="center", ha="center",
                        color="white", fontsize=16, fontweight="bold")
        left = left + vals

    # total label on the right
    for y, t in zip(ys, ct["total"].values):
        ax.text(t + 4, y, f"n = {int(t)}",
                va="center", ha="left",
                color="#111827", fontsize=15, fontweight="bold")

    ax.set_yticks(ys)
    ax.set_yticklabels([CANCER_LABEL[c] for c in CANCERS])
    ax.set_xlabel("Number of patients (unique)")
    ax.set_title("Patient Distribution by Cancer Type & Stage Group",
                 fontweight="bold", loc="left")
    ax.set_xlim(0, ct["total"].max() * 1.15)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    png_path = OUT_DIR / "fig_stage_distribution.png"
    pdf_path = OUT_DIR / "fig_stage_distribution.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")
    print("\n[crosstab]")
    print(ct)


if __name__ == "__main__":
    main()
