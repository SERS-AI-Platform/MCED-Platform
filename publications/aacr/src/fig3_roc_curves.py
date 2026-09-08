"""Figure 3: Non-YPAN internal test ROC curves with bootstrap 95% CI."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nature_style import (
    CANCER_COLORS,
    DOUBLE_COL,
    FONT_SIZE,
    LINE_WIDTH,
    OUTPUT_DIR,
    add_panel_label,
    apply_nature_style,
    apply_style,
    save_figure,
)
from stk_v2_fixed_non_ypan import (
    DISPLAY_CANCERS,
    bootstrap_roc_ci,
    load_cancer_types,
    load_test_predictions,
)

apply_style()


def _auc_label(name: str, stats: dict[str, np.ndarray | float]) -> str:
    return (
        f"{name} AUC={float(stats['auc']):.3f} "
        f"(95% CI {float(stats['auc_ci_low']):.3f}-{float(stats['auc_ci_high']):.3f})"
    )


def main() -> None:
    d = load_test_predictions()
    cancer_types = load_cancer_types()
    y_bin = d["y_bin"].astype(int)
    y_type = d["y_type"].astype(int)
    s1_prob = d["s1_prob"].astype(float)
    s2_prob = d["s2_prob"].astype(float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.43))
    fig.subplots_adjust(wspace=0.35)

    add_panel_label(ax1, "a")
    s1 = bootstrap_roc_ci(y_bin, s1_prob, n_bootstrap=1000, seed=42)
    ax1.plot(
        s1["fpr"],
        s1["tpr"],
        color="#2C3E50",
        lw=LINE_WIDTH["roc"] + 0.4,
        label=_auc_label("", s1).strip(),
    )
    ax1.fill_between(
        s1["fpr_grid"],
        s1["tpr_ci_low"],
        s1["tpr_ci_high"],
        color="#2C3E50",
        alpha=0.14,
        linewidth=0,
    )
    ax1.plot([0, 1], [0, 1], "--", color="#BBBBBB", lw=LINE_WIDTH["thin"])
    ax1.set_xlabel("1 - Specificity")
    ax1.set_ylabel("Sensitivity")
    ax1.set_title("Test Set Cancer Detection", fontsize=FONT_SIZE["title"], pad=8)
    ax1.legend(
        loc="lower right",
        frameon=True,
        edgecolor="#DDDDDD",
        fancybox=False,
        framealpha=0.9,
        fontsize=5.8,
    )
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_aspect("equal")
    apply_nature_style(ax1)

    add_panel_label(ax2, "b")
    cancer_mask = y_bin == 1
    type_index = {ct: idx for idx, ct in enumerate(cancer_types)}
    rows = [
        {
            "task": "Cancer vs non-cancer",
            "auc": float(s1["auc"]),
            "auc_ci_low": float(s1["auc_ci_low"]),
            "auc_ci_high": float(s1["auc_ci_high"]),
            "n_positive": int(y_bin.sum()),
            "n_negative": int((y_bin == 0).sum()),
        }
    ]
    for ct in DISPLAY_CANCERS:
        if ct not in type_index:
            continue
        idx = type_index[ct]
        yy = (y_type[cancer_mask] == idx).astype(int)
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        stats = bootstrap_roc_ci(yy, s2_prob[cancer_mask, idx], n_bootstrap=1000, seed=100 + idx)
        color = CANCER_COLORS.get(ct, "#333333")
        label_name = ct
        ax2.plot(
            stats["fpr"],
            stats["tpr"],
            color=color,
            lw=LINE_WIDTH["roc"],
            label=_auc_label(label_name, stats),
        )
        ax2.fill_between(
            stats["fpr_grid"],
            stats["tpr_ci_low"],
            stats["tpr_ci_high"],
            color=color,
            alpha=0.06,
            linewidth=0,
        )
        rows.append(
            {
                "task": ct,
                "auc": float(stats["auc"]),
                "auc_ci_low": float(stats["auc_ci_low"]),
                "auc_ci_high": float(stats["auc_ci_high"]),
                "n_positive": int(yy.sum()),
                "n_negative": int(len(yy) - yy.sum()),
            }
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "fig3_roc_ci.csv"), index=False, encoding="utf-8-sig"
    )
    ax2.plot([0, 1], [0, 1], "--", color="#BBBBBB", lw=LINE_WIDTH["thin"])
    ax2.set_xlabel("1 - Specificity")
    ax2.set_ylabel("Sensitivity")
    ax2.set_title("Test Set Cancer Type ROC", fontsize=FONT_SIZE["title"], pad=8)
    ax2.legend(
        loc="lower right",
        frameon=True,
        edgecolor="#DDDDDD",
        fancybox=False,
        framealpha=0.9,
        fontsize=5.7,
    )
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_aspect("equal")
    apply_nature_style(ax2)

    save_figure(fig, "fig3_roc_curves")
    plt.close(fig)
    print("Figure 3 complete.")


if __name__ == "__main__":
    main()
