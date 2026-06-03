"""Figure 5 — Cancer-type confusion matrix only, cells colored per cancer-row color.

Each row uses the true cancer's representative color (alpha = row-normalized value).
Diagonal values bold. Standalone figure, no F1 panel.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "AACR" / "src"))
from nature_style import AACR_LABELS, CANCER_COLORS as AACR_CC

CANCER_COLORS = dict(AACR_CC)
CANCER_COLORS.update({"BRE": "#D4527A", "BLC": "#E8960C"})

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

    labels_disp = [AACR_LABELS.get(c, c) for c in ORDER]
    n = len(ORDER)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    rgba = np.ones((n, n, 4))
    for i, cancer in enumerate(ORDER):
        base = np.array(to_rgba(CANCER_COLORS[cancer]))
        for j in range(n):
            v = cm_norm[i, j]
            rgba[i, j] = (1 - v) * np.array([1, 1, 1, 1]) + v * base
            rgba[i, j, 3] = 1.0
    ax.imshow(rgba, aspect="equal")

    for (i, j), v in np.ndenumerate(cm):
        norm_v = cm_norm[i, j]
        txt = f"{v}\n({norm_v:.2f})"
        col = "black" if norm_v < 0.55 else "white"
        ax.text(j, i, txt, ha="center", va="center",
                color=col, fontsize=10,
                fontweight="bold" if i == j else "normal")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels_disp, fontsize=10)
    ax.set_yticklabels(labels_disp, fontsize=10)
    for tick, cancer in zip(ax.get_yticklabels(), ORDER):
        tick.set_color(CANCER_COLORS[cancer]); tick.set_fontweight("bold")
    for tick, cancer in zip(ax.get_xticklabels(), ORDER):
        tick.set_color(CANCER_COLORS[cancer]); tick.set_fontweight("bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)

    plt.tight_layout()
    out_png = RUN_DIR / "fig5_type_cm_colored.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close()

    pd.DataFrame(cm, index=labels_disp, columns=labels_disp).to_csv(
        RUN_DIR / "fig5_type_cm_counts.csv", encoding="utf-8-sig")
    pd.DataFrame(cm_norm, index=labels_disp, columns=labels_disp).to_csv(
        RUN_DIR / "fig5_type_cm_normalized.csv", encoding="utf-8-sig")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
