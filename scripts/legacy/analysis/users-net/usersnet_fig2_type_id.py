"""Figure 2 — uSERS-Net cancer type identification.

Left: per-cancer sensitivity bar (7 bars, each in cancer-specific color, with 95% CI).
Right: One-vs-Rest precision-recall curves for all 7 cancers, each in its cancer color.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from sklearn.metrics import auc as _auc, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "AACR" / "src"))
from nature_style import CANCER_COLORS as AACR_CC, AACR_LABELS

# Combined 7-cancer palette (AACR 5-cancer + bumbucheo BLC/BRE)
CANCER_COLORS = dict(AACR_CC)
CANCER_COLORS.update({"BRE": "#D4527A", "BLC": "#E8960C"})
TARGET_CANCERS = ["CRC", "LUN", "BLC", "PRO", "PAN", "OVA", "BRE"]
CANCER_INDEX = {"PRO": 0, "BRE": 1, "OVA": 2, "LUN": 3, "CRC": 4, "PAN": 5, "BLC": 6}

RUN_DIR = PROJECT_ROOT / "results" / "runs" / "2026-05-11_usersnet_alpha0.8_ovr_roc"


def boot_sens_ci(y_true, y_pred, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return (np.nan, np.nan, np.nan)
    idx_pos = np.where(y_true == 1)[0]
    senss = []
    for _ in range(n):
        i = rng.choice(idx_pos, n_pos, replace=True)
        senss.append((y_pred[i] == 1).mean())
    point = (y_pred[idx_pos] == 1).mean()
    return point, float(np.percentile(senss, 2.5)), float(np.percentile(senss, 97.5))


def main():
    d = np.load(RUN_DIR / "all_models_oof.npz", allow_pickle=True)
    bl = d["bl"].astype(int); ctl = d["ctl"]
    bp = d["uSERS-Net_bp"]; cl = d["uSERS-Net_cl"]  # (n,7)

    # Stage 1 binary detection mask at Youden
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(bl, bp)
    yo = thr[np.argmax(tpr - fpr)]

    # For each cancer, sensitivity = of true cancer-i samples, how many are
    # both screened positive (bp >= yo) AND classified as type i.
    sens_rows = []
    for cancer in TARGET_CANCERS:
        ci = CANCER_INDEX[cancer]
        mask = (ctl == ci)
        if mask.sum() == 0:
            continue
        passed_s1 = (bp[mask] >= yo)
        pred_type = np.argmax(cl[mask], axis=1)
        correct = passed_s1 & (pred_type == ci)
        y_correct = correct.astype(int)
        y_true_pos = np.ones(mask.sum(), dtype=int)
        point, lo, hi = boot_sens_ci(y_true_pos, y_correct)
        sens_rows.append({"cancer": cancer, "n": int(mask.sum()),
                          "sensitivity": float(point), "ci_lo": lo, "ci_hi": hi})

    df_sens = pd.DataFrame(sens_rows)

    fig = plt.figure(figsize=(13, 5.5))
    gs = GridSpec(1, 2, width_ratios=[1, 1.1], wspace=0.28)
    ax_bar = fig.add_subplot(gs[0])
    ax_pr = fig.add_subplot(gs[1])

    xs = np.arange(len(df_sens))
    colors = [CANCER_COLORS[c] for c in df_sens.cancer.values]
    bars = ax_bar.bar(xs, df_sens.sensitivity, color=colors,
                      yerr=[df_sens.sensitivity - df_sens.ci_lo,
                            df_sens.ci_hi - df_sens.sensitivity],
                      capsize=4, edgecolor="black", linewidth=0.5)
    for x, (s, n) in enumerate(zip(df_sens.sensitivity, df_sens.n)):
        ax_bar.text(x, s + 0.02, f"{s:.2f}\n(n={n})", ha="center", fontsize=8.5)
    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels([AACR_LABELS.get(c, c) for c in df_sens.cancer])
    ax_bar.set_ylim(0, 1.1); ax_bar.set_ylabel("Sensitivity")
    ax_bar.grid(alpha=0.25, axis="y")
    ax_bar.set_axisbelow(True)

    # OvR ROC curve for each cancer
    pr_rows = []
    for cancer in TARGET_CANCERS:
        ci = CANCER_INDEX[cancer]
        y_true = (ctl == ci).astype(int)
        y_score = cl[:, ci]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val = _auc(fpr, tpr)
        c = CANCER_COLORS[cancer]
        ax_pr.plot(fpr, tpr, color=c, lw=2.2,
                   label=f"{AACR_LABELS.get(cancer, cancer)}  AUC={auc_val:.3f}  (n={int(y_true.sum())})")
        pr_rows.append({"cancer": cancer, "auc": auc_val,
                        "n": int(y_true.sum())})
    ax_pr.plot([0, 1], [0, 1], "k--", lw=0.6, alpha=0.4)
    ax_pr.set_xlabel("1 − Specificity"); ax_pr.set_ylabel("Sensitivity")
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1.02)
    ax_pr.legend(loc="lower right", fontsize=8.5)
    ax_pr.grid(alpha=0.3)
    plt.tight_layout()
    out_png = RUN_DIR / "fig2_type_id_sens_pr.png"
    plt.savefig(out_png, dpi=250, bbox_inches="tight"); plt.close()

    df_sens.to_csv(RUN_DIR / "fig2_sensitivity_per_cancer.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pr_rows).to_csv(RUN_DIR / "fig2_ovr_auc.csv", index=False, encoding="utf-8-sig")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
