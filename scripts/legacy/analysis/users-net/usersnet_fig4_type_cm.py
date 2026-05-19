"""Figure 4 — uSERS-Net cancer-type confusion matrix + per-cancer F1.

Uses existing OOF predictions (all_models_oof.npz). No retraining.
Cancer samples only (ctl >= 0). True = ctl, Pred = argmax of uSERS-Net 7-class probs.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "AACR" / "src"))
from nature_style import AACR_LABELS

ORDER = ["CRC", "LUN", "BLC", "PRO", "PAN", "OVA", "BRE"]
INTERNAL_IDX = {"PRO": 0, "BRE": 1, "OVA": 2, "LUN": 3, "CRC": 4, "PAN": 5, "BLC": 6}
RUN_DIR = PROJECT_ROOT / "results" / "runs" / "2026-05-11_usersnet_alpha0.8_ovr_roc"


def main():
    d = np.load(RUN_DIR / "all_models_oof.npz", allow_pickle=True)
    ctl = d["ctl"]; cl = d["uSERS-Net_cl"]

    cancer_mask = ctl >= 0
    y_true_int = ctl[cancer_mask]
    y_pred_int = np.argmax(cl[cancer_mask], axis=1)

    order_idx = [INTERNAL_IDX[c] for c in ORDER]
    cm = confusion_matrix(y_true_int, y_pred_int, labels=order_idx)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    per_f1 = f1_score(y_true_int, y_pred_int, labels=order_idx, average=None, zero_division=0)
    per_prec = precision_score(y_true_int, y_pred_int, labels=order_idx, average=None, zero_division=0)
    per_rec = recall_score(y_true_int, y_pred_int, labels=order_idx, average=None, zero_division=0)
    macro_f1 = f1_score(y_true_int, y_pred_int, labels=order_idx, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_int, y_pred_int, labels=order_idx, average="weighted", zero_division=0)

    labels_disp = [AACR_LABELS.get(c, c) for c in ORDER]

    fig = plt.figure(figsize=(11, 5.6))
    gs = GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.3)
    ax_cm = fig.add_subplot(gs[0])
    ax_f1 = fig.add_subplot(gs[1])

    im = ax_cm.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(cm):
        norm_v = cm_norm[i, j]
        txt = f"{v}\n({norm_v:.2f})"
        ax_cm.text(j, i, txt, ha="center", va="center",
                   color="white" if norm_v > 0.5 else "black",
                   fontsize=9, fontweight="bold" if i == j else "normal")
    ax_cm.set_xticks(range(len(ORDER))); ax_cm.set_yticks(range(len(ORDER)))
    ax_cm.set_xticklabels(labels_disp, rotation=0)
    ax_cm.set_yticklabels(labels_disp)
    ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("True")
    plt.colorbar(im, ax=ax_cm, fraction=0.045, pad=0.04, label="Row-normalized")

    sys.path.insert(0, str(PROJECT_ROOT / "AACR" / "src"))
    from nature_style import CANCER_COLORS as AACR_CC
    CANCER_COLORS = dict(AACR_CC); CANCER_COLORS.update({"BRE": "#D4527A", "BLC": "#E8960C"})

    xs = np.arange(len(ORDER))
    colors = [CANCER_COLORS[c] for c in ORDER]
    bars = ax_f1.bar(xs, per_f1, color=colors, edgecolor="black", linewidth=0.5)
    for x, f, n in zip(xs, per_f1, cm.sum(axis=1)):
        ax_f1.text(x, f + 0.02, f"{f:.3f}\n(n={int(n)})", ha="center", fontsize=8.5)
    ax_f1.axhline(macro_f1, color="#C0392B", lw=1.5, ls="--",
                  label=f"Macro F1 = {macro_f1:.3f}")
    ax_f1.axhline(weighted_f1, color="#444", lw=1.2, ls=":",
                  label=f"Weighted F1 = {weighted_f1:.3f}")
    ax_f1.set_xticks(xs); ax_f1.set_xticklabels(labels_disp)
    ax_f1.set_ylim(0, 1.1)
    ax_f1.set_ylabel("F1 score")
    leg = ax_f1.legend(loc="lower right", fontsize=8.5)
    ax_f1.grid(alpha=0.25, axis="y")
    ax_f1.set_axisbelow(True)

    plt.tight_layout()
    out_png = RUN_DIR / "fig4_type_cm_f1.png"
    plt.savefig(out_png, dpi=250, bbox_inches="tight"); plt.close()

    pd.DataFrame(cm, index=labels_disp, columns=labels_disp).to_csv(
        RUN_DIR / "fig4_type_cm_counts.csv", encoding="utf-8-sig")
    pd.DataFrame(cm_norm, index=labels_disp, columns=labels_disp).to_csv(
        RUN_DIR / "fig4_type_cm_normalized.csv", encoding="utf-8-sig")
    pd.DataFrame({"cancer": ORDER, "n": cm.sum(axis=1),
                  "precision": per_prec, "recall": per_rec, "f1": per_f1}).to_csv(
        RUN_DIR / "fig4_type_metrics.csv", index=False, encoding="utf-8-sig")
    with open(RUN_DIR / "fig4_macro_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"macro_f1={macro_f1:.4f}\nweighted_f1={weighted_f1:.4f}\nn_cancer={cancer_mask.sum()}\n")

    print(f"Saved {out_png}")
    print(f"macro F1 = {macro_f1:.4f}, weighted F1 = {weighted_f1:.4f}")
    print(pd.DataFrame({"cancer": ORDER, "n": cm.sum(axis=1), "f1": per_f1}).to_string(index=False))


if __name__ == "__main__":
    main()
