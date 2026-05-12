"""Stage analysis - Step 6: Non-cancer vs Early vs Advanced mean spectra.

5-cancer subplot x 3 curves (non-cancer, early, advanced).
Non-cancer baseline = NOR + DIA + HBP + H.D. x 100 each (= 400 total, shared across subplots).
Patient-level filter uses master_clinical.csv cancer_stage_group
(post stage_05 update, including CRC 271-300 override).

Sampling: sample_id 1..100 per non-cancer group (deterministic).

Output: AACR/figures/fig_stage_vs_noncancer.{png,pdf}
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path("/home/user/SERS-AI")
MASTER = ROOT / "data/clinical_data/master_clinical.csv"
SPECTRA = ROOT / "results/processed_spectra.csv"
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
NONCANCER_GROUPS = ["NOR", "DIA", "HBP", "H.D."]
SAMPLES_PER_NONCANCER = 100

COLOR_NONCANCER = "#374151"   # dark gray
COLOR_EARLY = "#2563eb"       # blue
COLOR_ADV = "#dc2626"         # red

mpl.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 24,
    "axes.linewidth": 1.4,
    "lines.linewidth": 2.2,
})


def load_stage_labels() -> pd.DataFrame:
    m = pd.read_csv(MASTER, low_memory=False)
    m["spectral_sample_id"] = pd.to_numeric(m["spectral_sample_id"], errors="coerce")
    keep = m[["spectral_group", "spectral_sample_id", "cancer_stage_group"]].copy()
    keep = keep.drop_duplicates(subset=["spectral_group", "spectral_sample_id"])
    return keep


def pick_noncancer_ids() -> pd.DataFrame:
    """Return (group, sid_num) selections: first 100 per non-cancer group."""
    rows = []
    for g in NONCANCER_GROUPS:
        for sid in range(1, SAMPLES_PER_NONCANCER + 1):
            rows.append({"group": g, "sid_num": sid})
    return pd.DataFrame(rows)


def main() -> None:
    labels = load_stage_labels()

    sp = pd.read_csv(SPECTRA)
    sp["sid_num"] = pd.to_numeric(sp["sample_id"], errors="coerce").astype("Int64")

    wv_cols = [c for c in sp.columns if c.startswith("x_")]
    wv_nums = np.array([float(c[2:]) for c in wv_cols])
    fp = (wv_nums >= 400) & (wv_nums <= 1800)
    wv = wv_nums[fp]
    wv_cols_fp = [c for c, m in zip(wv_cols, fp) if m]

    # Non-cancer pool
    nc_ids = pick_noncancer_ids()
    nc_sp = sp.merge(nc_ids, on=["group", "sid_num"], how="inner")
    nc_X = nc_sp[wv_cols_fp].to_numpy(dtype=float)
    nc_mean = np.nanmean(nc_X, axis=0)
    nc_sd = np.nanstd(nc_X, axis=0)
    nc_n_pt = nc_sp.groupby(["group", "sid_num"]).ngroups
    print(f"[non-cancer] pooled patients: {nc_n_pt}, spectra: {len(nc_sp)}")

    # Plot
    fig, axes = plt.subplots(5, 1, figsize=(18, 22), sharex=True)
    fig.suptitle("Mean SERS Spectra: Non-Cancer vs Early vs Advanced",
                 fontweight="bold", y=0.995)

    for i, cancer in enumerate(CANCERS):
        ax = axes[i]
        # Non-cancer (same in every subplot)
        ax.plot(wv, nc_mean, color=COLOR_NONCANCER,
                label=f"Non-cancer (n={nc_n_pt})", linestyle="--")
        ax.fill_between(wv, nc_mean - nc_sd, nc_mean + nc_sd,
                        color=COLOR_NONCANCER, alpha=0.12)

        # Cancer stage groups
        sub_lbl = labels[(labels["spectral_group"] == cancer)]
        for stage, color, txt in [("early", COLOR_EARLY, "Early (I–II)"),
                                   ("advanced", COLOR_ADV, "Advanced (III–IV)")]:
            pats = sub_lbl[sub_lbl["cancer_stage_group"] == stage]
            if len(pats) == 0:
                continue
            sp_sub = sp[(sp["group"] == cancer)
                        & (sp["sid_num"].isin(pats["spectral_sample_id"]))]
            if len(sp_sub) == 0:
                continue
            X = sp_sub[wv_cols_fp].to_numpy(dtype=float)
            mean = np.nanmean(X, axis=0)
            sd = np.nanstd(X, axis=0)
            n_pt = sp_sub["sid_num"].nunique()
            ax.plot(wv, mean, color=color, label=f"{txt} (n={n_pt})")
            ax.fill_between(wv, mean - sd, mean + sd, color=color, alpha=0.18)

        ax.set_title(CANCER_LABEL[cancer], loc="left", fontweight="bold")
        ax.set_ylabel("Intensity (a.u.)")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)
        ax.set_xlim(wv.min(), wv.max())

    axes[-1].set_xlabel("Raman shift (cm$^{-1}$)")
    fig.tight_layout()

    png = OUT_DIR / "fig_stage_vs_noncancer.png"
    pdf = OUT_DIR / "fig_stage_vs_noncancer.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {png}")
    print(f"saved: {pdf}")


if __name__ == "__main__":
    main()
