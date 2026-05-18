"""Figure 3 — Cancer type-specific SERS spectral fingerprints identified by uSERS-Net.

(A) Discriminative peaks per cancer type with shaded regions where uSERS-Net attribution
    distinguishes cancer from normal (NOR baseline). Mean cancer spectrum overlaid on
    mean NOR spectrum; shaded bands mark wavenumbers with |combined_attr| above threshold.
(B) Heatmap of attribution intensity at top peaks × 7 cancers, showing shared yet
    cancer-specific intensity patterns.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "AACR" / "src"))
from nature_style import CANCER_COLORS as AACR_CC, AACR_LABELS

CANCER_COLORS = dict(AACR_CC)
CANCER_COLORS.update({"BRE": "#D4527A", "BLC": "#E8960C"})
TARGET_CANCERS = ["CRC", "LUN", "BLC", "PRO", "PAN", "OVA", "BRE"]
CANCER_INDEX = {"PRO": 0, "BRE": 1, "OVA": 2, "LUN": 3, "CRC": 4, "PAN": 5, "BLC": 6}

RUN_DIR = PROJECT_ROOT / "results" / "runs" / "2026-05-11_usersnet_alpha0.8_ovr_roc"


def main():
    d = np.load(RUN_DIR / "combined_attribution.npz", allow_pickle=True)
    combined = d["combined"]      # (n, 7, L)
    X = d["X"]; groups_arr = d["groups_arr"]; ctl = d["ctl"]
    wn = d["wavenumbers"]; baseline = d["baseline"]
    nor_mask = (groups_arr == "NOR")
    nor_mean = X[nor_mask].mean(axis=0)
    nor_std = X[nor_mask].std(axis=0)

    n_cancers = len(TARGET_CANCERS)

    # (A) per-cancer spectrum with discriminative shaded regions
    fig = plt.figure(figsize=(15, 18))
    gs = GridSpec(n_cancers + 2, 1, height_ratios=[1] * n_cancers + [1.2, 0.2], hspace=0.45)

    top_peak_summary = {}   # cancer -> list of (wn, attr_signed)

    for ax_idx, cancer in enumerate(TARGET_CANCERS):
        ci = CANCER_INDEX[cancer]
        cmask = (groups_arr == cancer)
        if cmask.sum() == 0:
            continue
        attr = combined[cmask, ci]        # (k, L)
        mean_abs = np.mean(np.abs(attr), axis=0)
        mean_signed = np.mean(attr, axis=0)
        cancer_mean = X[cmask].mean(axis=0)
        cancer_std = X[cmask].std(axis=0)

        # threshold: top decile of attribution magnitude
        threshold = np.percentile(mean_abs, 90)
        discrim_mask = mean_abs >= threshold

        # Top-8 peaks within discriminative regions for labelling
        from scipy.signal import find_peaks
        peaks_idx, _ = find_peaks(mean_abs, distance=10, prominence=mean_abs.std() * 0.4)
        peaks_idx = peaks_idx[mean_abs[peaks_idx] >= threshold]
        top_peaks = peaks_idx[np.argsort(-mean_abs[peaks_idx])][:8]
        top_peak_summary[cancer] = [(float(wn[p]), float(mean_signed[p]), float(mean_abs[p])) for p in top_peaks]

        ax = fig.add_subplot(gs[ax_idx])
        color = CANCER_COLORS[cancer]
        # shaded NOR ±std band (light gray)
        ax.fill_between(wn, nor_mean - nor_std, nor_mean + nor_std, color="#bbb", alpha=0.3,
                        label="NOR ±1σ")
        ax.plot(wn, nor_mean, color="#666", lw=1.0, ls="--", label="NOR mean")
        # cancer spectrum
        ax.fill_between(wn, cancer_mean - cancer_std, cancer_mean + cancer_std,
                        color=color, alpha=0.18)
        ax.plot(wn, cancer_mean, color=color, lw=1.5, label=f"{cancer} mean")
        # shaded discriminative regions: highlight where |combined_attr| is in top decile
        ymin, ymax = ax.get_ylim()
        ax.fill_between(wn, ymin, ymax, where=discrim_mask, color=color, alpha=0.10,
                        step="mid", label="Discriminative (top 10%)")
        ax.set_ylim(ymin, ymax)
        # top peak labels
        for p in top_peaks:
            ax.axvline(wn[p], color=color, alpha=0.5, lw=0.7)
            ax.text(wn[p], ymax * 0.92, f"{wn[p]:.0f}", color=color, fontsize=7,
                    rotation=90, va="top", ha="center", fontweight="bold")
        ax.set_ylabel(f"{AACR_LABELS.get(cancer, cancer)}\n(n={int(cmask.sum())})\nIntensity (a.u.)",
                      color=color, fontsize=10)
        ax.grid(alpha=0.2)
        if ax_idx == 0:
            ax.legend(loc="upper right", fontsize=7.5, ncol=4)
        if ax_idx == n_cancers - 1:
            ax.set_xlabel("Wavenumber (cm$^{-1}$)")

    # (B) heatmap of attribution across cancers at top peaks
    # Build unique top-peak set (union of top-N across cancers)
    union_peaks = sorted({int(np.argmin(np.abs(wn - w))) for c in top_peak_summary for w, *_ in top_peak_summary[c]})
    heat = np.zeros((len(TARGET_CANCERS), len(union_peaks)))
    sign_heat = np.zeros_like(heat)
    for ri, cancer in enumerate(TARGET_CANCERS):
        ci = CANCER_INDEX[cancer]
        cmask = (groups_arr == cancer)
        if cmask.sum() == 0:
            continue
        attr = combined[cmask, ci]
        mean_signed = np.mean(attr, axis=0)
        mean_abs = np.mean(np.abs(attr), axis=0)
        for cj, pi in enumerate(union_peaks):
            heat[ri, cj] = mean_abs[pi]
            sign_heat[ri, cj] = mean_signed[pi]

    ax_h = fig.add_subplot(gs[n_cancers])
    # row-normalize for visibility (each cancer's intensity scale)
    heat_norm = heat / (heat.max(axis=1, keepdims=True) + 1e-9)
    vmax = float(np.max(np.abs(sign_heat)))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    im = ax_h.imshow(sign_heat, aspect="auto", cmap="RdBu_r", norm=norm)
    ax_h.set_xticks(range(len(union_peaks)))
    ax_h.set_xticklabels([f"{wn[p]:.0f}" for p in union_peaks], rotation=45, ha="right", fontsize=7)
    ax_h.set_yticks(range(len(TARGET_CANCERS)))
    ax_h.set_yticklabels([AACR_LABELS.get(c, c) for c in TARGET_CANCERS], fontsize=9)
    ax_h.set_xlabel("Wavenumber (cm$^{-1}$)")
    plt.colorbar(im, ax=ax_h, fraction=0.025, pad=0.02, label="mean signed attribution")
    plt.tight_layout()
    out_png = RUN_DIR / "fig3_uSERSnet_fingerprints.png"
    plt.savefig(out_png, dpi=250, bbox_inches="tight"); plt.close()

    # Export top peaks summary
    rows = []
    for cancer, lst in top_peak_summary.items():
        for rank, (w, sgn, mag) in enumerate(lst, 1):
            rows.append({"cancer": cancer, "rank": rank, "wavenumber": w,
                         "mean_signed_attr": sgn, "mean_abs_attr": mag})
    pd.DataFrame(rows).to_csv(RUN_DIR / "fig3_top_peaks.csv", index=False, encoding="utf-8-sig")
    # Heatmap CSV
    pd.DataFrame(sign_heat, index=TARGET_CANCERS,
                 columns=[f"{wn[p]:.1f}" for p in union_peaks]).to_csv(
                     RUN_DIR / "fig3_heatmap_signed_attr.csv", encoding="utf-8-sig")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
