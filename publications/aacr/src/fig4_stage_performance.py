"""Figure 4: Non-YPAN internal test performance with 95% CI."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix

from nature_style import (
    apply_style,
    apply_nature_style,
    add_panel_label,
    save_figure,
    CANCER_COLORS,
    CANCER_LABELS,
    DOUBLE_COL,
    FONT_SIZE,
    ROOT,
)
from stk_v2_fixed_non_ypan import (
    DISPLAY_CANCERS,
    binary_metric_table,
    cancer_type_sensitivity_table,
    load_cancer_types,
    load_test_predictions,
)


apply_style()


def _blend_with_white(hex_color: str, amount: float) -> tuple[float, float, float]:
    rgb = np.asarray(mcolors.to_rgb(hex_color), dtype=float)
    amount = float(np.clip(amount, 0.0, 1.0))
    return tuple((1.0 - amount) * np.ones(3) + amount * rgb)


def _relative_luminance(rgb: tuple[float, float, float]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def main() -> None:
    d = load_test_predictions()
    labels = load_cancer_types()
    y_bin = d["y_bin"].astype(int)
    y_type = d["y_type"].astype(int)
    s2_prob = d["s2_prob"].astype(float)
    cancer = y_bin == 1
    pred_type = s2_prob[cancer].argmax(axis=1)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    present_raw = {
        labels[i]
        for i in (set(y_type[cancer].tolist()) | set(pred_type.tolist()))
        if 0 <= i < len(labels)
    }
    present_labels = [g for g in DISPLAY_CANCERS if g in present_raw and g in label_to_idx]
    present_idx = [label_to_idx[g] for g in present_labels]

    binary_df = binary_metric_table()
    type_df = cancer_type_sensitivity_table()
    type_df["label"] = type_df["cancer_type"].map(lambda x: CANCER_LABELS.get(x, x))

    out_dir = os.path.join(ROOT, "publications", "aacr", "figures")
    os.makedirs(out_dir, exist_ok=True)
    binary_df.to_csv(os.path.join(out_dir, "fig4_binary_metrics_ci.csv"), index=False)
    type_df.to_csv(os.path.join(out_dir, "fig4_cancer_type_sensitivity_ci.csv"), index=False)

    fig = plt.figure(figsize=(DOUBLE_COL, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.16, 1.0, 1.22], wspace=0.52)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    add_panel_label(ax1, "a")
    ordered_metrics = ["AUROC", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1"]
    sub = binary_df.set_index("metric").loc[ordered_metrics].reset_index()
    x = np.arange(len(sub))
    y = sub["value"].values
    err = np.vstack([
        np.maximum(0, y - sub["ci_low"].values),
        np.maximum(0, sub["ci_high"].values - y),
    ])
    ax1.errorbar(
        x, y, yerr=err, fmt="o", color="#2C3E50",
        ecolor="#2C3E50", elinewidth=0.9, capsize=3,
        markerfacecolor="#2C3E50", markeredgecolor="white", markeredgewidth=0.6,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(sub["metric"], rotation=40, ha="right")
    ax1.set_ylim(0.78, 1.02)
    ax1.set_ylabel("Score")
    ax1.set_title("Cancer Detection Metrics", fontsize=FONT_SIZE["title"], pad=8)
    ax1.axhline(0.9, color="#DDDDDD", lw=0.6, ls="--", zorder=0)
    apply_nature_style(ax1)

    add_panel_label(ax2, "b")
    type_order = [g for g in DISPLAY_CANCERS if g in set(type_df["cancer_type"])]
    type_sub = type_df.set_index("cancer_type").loc[type_order].reset_index()
    x2 = np.arange(len(type_sub))
    y2 = type_sub["sensitivity"].values
    err2 = np.vstack([
        np.maximum(0, y2 - type_sub["ci_low"].values),
        np.maximum(0, type_sub["ci_high"].values - y2),
    ])
    colors = [CANCER_COLORS.get(g, "#333333") for g in type_sub["cancer_type"]]
    for xi, yi, lohi, color, row in zip(x2, y2, err2.T, colors, type_sub.to_dict("records")):
        ax2.errorbar(
            xi, yi, yerr=np.array([[lohi[0]], [lohi[1]]]), fmt="o",
            color=color, ecolor=color, elinewidth=0.9, capsize=3,
            markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.6,
        )
        ax2.text(xi, min(1.02, yi + lohi[1] + 0.035), f"n={int(row['n'])}",
                 ha="center", fontsize=FONT_SIZE["annotation"], color=color)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(type_sub["cancer_type"], rotation=0, ha="center")
    ax2.set_ylim(0.43, 1.08)
    ax2.set_ylabel("Sensitivity")
    ax2.set_title("Cancer Type Sensitivity", fontsize=FONT_SIZE["title"], pad=8)
    ax2.axhline(0.9, color="#DDDDDD", lw=0.6, ls="--", zorder=0)
    apply_nature_style(ax2)

    add_panel_label(ax3, "c")
    cm = confusion_matrix(y_type[cancer], pred_type, labels=present_idx)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum > 0)
    pd.DataFrame(cm, index=present_labels, columns=present_labels).to_csv(
        os.path.join(out_dir, "fig4_confusion_matrix_counts.csv")
    )
    pd.DataFrame(cm_norm, index=present_labels, columns=present_labels).to_csv(
        os.path.join(out_dir, "fig4_confusion_matrix_row_normalized.csv")
    )

    n_cm = len(present_labels)
    for (i, j), value in np.ndenumerate(cm):
        norm_value = cm_norm[i, j]
        pred_label = present_labels[j]
        base_color = CANCER_COLORS.get(pred_label, "#808080")
        if value == 0:
            facecolor = "#FAFAFA"
        else:
            strength = 0.18 + (0.72 if i == j else 0.56) * float(norm_value)
            facecolor = _blend_with_white(base_color, strength)
        ax3.add_patch(
            plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                facecolor=facecolor,
                edgecolor="white",
                linewidth=0.8,
            )
        )
        text_color = "white" if value > 0 and _relative_luminance(mcolors.to_rgb(facecolor)) < 0.55 else "#222222"
        ax3.text(
            j, i, f"{value}\n{norm_value:.2f}",
            ha="center", va="center",
            fontsize=5.4,
            color=text_color,
        )
    ax3.set_xlim(-0.5, n_cm - 0.5)
    ax3.set_ylim(n_cm - 0.5, -0.5)
    ax3.set_xticks(np.arange(len(present_labels)))
    ax3.set_yticks(np.arange(len(present_labels)))
    ax3.set_xticklabels(present_labels)
    ax3.set_yticklabels(present_labels)
    for tick, label in zip(ax3.get_xticklabels(), present_labels):
        tick.set_color(CANCER_COLORS.get(label, "#333333"))
        tick.set_fontweight("bold")
    for tick, label in zip(ax3.get_yticklabels(), present_labels):
        tick.set_color(CANCER_COLORS.get(label, "#333333"))
        tick.set_fontweight("bold")
    ax3.set_xlabel("Predicted Cancer Type")
    ax3.set_ylabel("True Cancer Type")
    ax3.set_title("Cancer Type Confusion Matrix", fontsize=FONT_SIZE["title"], pad=8)
    apply_nature_style(ax3)

    fig.subplots_adjust(left=0.07, right=0.965, bottom=0.24, top=0.82)
    save_figure(fig, "fig4_stage_performance")
    plt.close(fig)
    print("Figure 4 complete.")


if __name__ == "__main__":
    main()
