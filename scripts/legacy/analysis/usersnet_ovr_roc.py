"""uSERS-Net (LR_fusion + ResNet18 α-blend, sex constraint) per-cancer One-vs-Rest ROC.

Compares uSERS-Net (α=0.8 blend) against LR_fusion-alone and ResNet18-alone on the
same 5-fold OOF predictions stored in AACR/data/early_fusion/fold_predictions_7cancer.npz.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import auc, roc_curve

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "AACR" / "data" / "early_fusion" / "fold_predictions_7cancer.npz"
OUT_DIR = ROOT / "results" / "runs" / "2026-05-11_usersnet_alpha0.8_ovr_roc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.8
N_BOOT = 1000
RNG_SEED = 42
TARGET_CANCERS = ["BLC", "BRE", "CRC", "LUN", "OVA", "PAN", "PRO"]  # user-requested order


def apply_sex_constraint(probs: np.ndarray, sex: np.ndarray, cancer_types: list[str]) -> np.ndarray:
    out = probs.copy()
    pro_idx = cancer_types.index("PRO")
    ova_idx = cancer_types.index("OVA")
    out[sex == 0, pro_idx] = 0.0  # female: no PRO
    out[sex == 1, ova_idx] = 0.0  # male: no OVA
    row_sums = out.sum(axis=1, keepdims=True)
    nonzero = row_sums.squeeze() > 0
    out[nonzero] = out[nonzero] / row_sums[nonzero]
    return out


def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = N_BOOT, seed: int = RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if y_true.sum() < 2 or (n - y_true.sum()) < 2:
        return (np.nan, np.nan, np.nan)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.sum() < 1 or yt.sum() == len(yt):
            continue
        ys = y_score[idx]
        fpr, tpr, _ = roc_curve(yt, ys)
        aucs.append(auc(fpr, tpr))
    aucs = np.asarray(aucs)
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)), float(np.std(aucs)))


def main():
    d = np.load(NPZ, allow_pickle=True)
    cancer_types = [str(c) for c in d["cancer_types"].tolist()]
    sex = d["sex_for_constraint"].astype(int)
    ctl = d["cancer_type_labels"]
    bp_lr = d["val_bp_lr"]; bp_rn = d["val_bp_rn"]
    cl_lr = d["val_cl_lr"]  # already softmax + sex-constrained
    cl_rn_logits = d["val_cl_rn"]

    rn_probs = softmax(cl_rn_logits, axis=1)
    rn_probs = apply_sex_constraint(rn_probs, sex, cancer_types)

    cl_ens = ALPHA * cl_lr + (1 - ALPHA) * rn_probs  # uSERS-Net α=0.8

    models = {
        "uSERS-Net (α=0.8)": cl_ens,
        "LR_fusion alone": cl_lr,
        "ResNet18 alone": rn_probs,
    }

    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    colors = {"uSERS-Net (α=0.8)": "#d62728", "LR_fusion alone": "#1f77b4", "ResNet18 alone": "#2ca02c"}

    for ax_idx, cancer in enumerate(TARGET_CANCERS):
        ci = cancer_types.index(cancer)
        y_true = (ctl == ci).astype(int)
        ax = axes[ax_idx]
        for name, prob_mat in models.items():
            y_score = prob_mat[:, ci]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            ci_lo, ci_hi, std = bootstrap_auc_ci(y_true, y_score)
            rows.append({"cancer": cancer, "model": name, "auc": roc_auc,
                         "ci_lo": ci_lo, "ci_hi": ci_hi, "std": std,
                         "n_pos": int(y_true.sum()), "n_neg": int(len(y_true) - y_true.sum())})
            ax.plot(fpr, tpr, color=colors[name], lw=2,
                    label=f"{name} AUC={roc_auc:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
        ax.set_title(f"{cancer} (n_pos={int((ctl==ci).sum())})", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)

    # 8th subplot: macro-average across the 7 cancers
    ax = axes[7]
    macro_rows = []
    for name, prob_mat in models.items():
        all_fpr = np.linspace(0, 1, 200)
        mean_tpr = np.zeros_like(all_fpr)
        aucs_per = []
        for cancer in TARGET_CANCERS:
            ci = cancer_types.index(cancer)
            y_true = (ctl == ci).astype(int)
            y_score = prob_mat[:, ci]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            mean_tpr += np.interp(all_fpr, fpr, tpr)
            aucs_per.append(auc(fpr, tpr))
        mean_tpr /= len(TARGET_CANCERS)
        macro_auc = auc(all_fpr, mean_tpr)
        macro_rows.append({"model": name, "macro_auc": macro_auc, "mean_per_cancer_auc": float(np.mean(aucs_per))})
        ax.plot(all_fpr, mean_tpr, color=colors[name], lw=2.2,
                label=f"{name} macro AUC={macro_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title("Macro-average (7 cancers)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Per-cancer One-vs-Rest ROC — uSERS-Net (α=0.8, sex constraint) vs base models\n"
                 f"5-fold OOF predictions, n={len(ctl)} (incl. non-cancer)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = OUT_DIR / "ovr_roc_7cancer.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "ovr_auc_by_cancer.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(macro_rows).to_csv(OUT_DIR / "ovr_macro_auc.csv", index=False, encoding="utf-8-sig")

    print(f"\nSaved: {out_png}")
    print(f"Saved: {OUT_DIR/'ovr_auc_by_cancer.csv'}")
    print(f"Saved: {OUT_DIR/'ovr_macro_auc.csv'}")
    print("\nSummary (AUC by cancer):")
    pivot = df.pivot(index="cancer", columns="model", values="auc")
    print(pivot.round(3).to_string())
    print("\nMacro-average AUC:")
    print(pd.DataFrame(macro_rows).to_string(index=False))


if __name__ == "__main__":
    main()
