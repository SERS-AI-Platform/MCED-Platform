"""Stage analysis - Step 2: Mean SERS spectra by cancer x stage group.

Figure: 5 cancers (CRC, CPAN, PRO, OVA, LUN) as subplots.
  In each subplot: mean spectrum +/- 1SD shading for Early vs Advanced.
  CRC has no advanced in current data (shows Early only).
  Unknown stage excluded.

Output: AACR/figures/fig_stage_mean_spectra.{png,pdf}
Font & linewidth made large for poster-style readability.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path("/home/user/SERS-AI")
DATA = ROOT / "AACR/data/stage_merged.csv"
OUT_DIR = ROOT / "AACR/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANCERS = ["CRC", "CPAN", "PRO", "OVA", "LUN"]
CANCER_LABEL = {
    "CRC": "Colorectal (CRC)",
    "CPAN": "Pancreatic (PAC)",
    "PRO": "Prostate (PRC)",
    "OVA": "Ovarian (OVC)",
    "LUN": "Lung (LC)",
}
STAGE_ORDER = ["early", "advanced"]
STAGE_LABEL = {"early": "Early (I–II)", "advanced": "Advanced (III–IV)"}
STAGE_COLOR = {"early": "#2563eb", "advanced": "#dc2626"}

# ---------- Poster-style large font ----------
mpl.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 15,
    "figure.titlesize": 24,
    "axes.linewidth": 1.4,
    "lines.linewidth": 2.2,
})


def main() -> None:
    df = pd.read_csv(DATA)
    wv_cols = [c for c in df.columns if c.startswith("x_")]
    wv_nums = np.array([float(c[2:]) for c in wv_cols])
    # Focus on fingerprint region 400-1800 cm-1
    fp_mask = (wv_nums >= 400) & (wv_nums <= 1800)
    wv = wv_nums[fp_mask]
    wv_cols_fp = [c for c, m in zip(wv_cols, fp_mask) if m]

    fig, axes = plt.subplots(5, 1, figsize=(18, 22), sharex=True)
    fig.suptitle("Mean SERS Spectra by Cancer Type & Stage Group",
                 fontweight="bold", y=0.995)

    legend_made = False
    for i, cancer in enumerate(CANCERS):
        ax = axes[i]
        sub = df[df["group"] == cancer].copy()
        for stage in STAGE_ORDER:
            s = sub[sub["cancer_stage_group"] == stage]
            if len(s) == 0:
                continue
            X = s[wv_cols_fp].to_numpy(dtype=float)
            mean = np.nanmean(X, axis=0)
            sd = np.nanstd(X, axis=0)
            n_pat = s["sid_num"].nunique()
            ax.plot(wv, mean,
                    color=STAGE_COLOR[stage],
                    label=f"{STAGE_LABEL[stage]} (n={n_pat})")
            ax.fill_between(wv, mean - sd, mean + sd,
                            color=STAGE_COLOR[stage], alpha=0.18)
        ax.set_title(f"{CANCER_LABEL[cancer]}", loc="left", fontweight="bold")
        ax.set_ylabel("Intensity (a.u.)")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)
        ax.set_xlim(wv.min(), wv.max())

    axes[-1].set_xlabel("Raman shift (cm$^{-1}$)")

    fig.tight_layout()

    png_path = OUT_DIR / "fig_stage_mean_spectra.png"
    pdf_path = OUT_DIR / "fig_stage_mean_spectra.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")


if __name__ == "__main__":
    main()
