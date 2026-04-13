"""
Figure 5: Six-model performance comparison.
(a) Stage 1 metrics (AUC, Sensitivity, Specificity)
(b) Stage 2 metrics (F1-macro, AUC)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nature_style import (
    apply_style, apply_nature_style, add_panel_label, save_figure,
    MODEL_COLORS, MODEL_ORDER, DOUBLE_COL, FONT_SIZE, LINE_WIDTH, ROOT,
)

apply_style()

# ── Load benchmark data ──
bench_path = os.path.join(ROOT, "models", "results", "01_benchmarks",
                          "main_5models", "benchmark_summary.csv")
bench = pd.read_csv(bench_path)

# Rename models for display
model_name_map = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "cnn1d": "CNN1D",
    "resnet18": "ResNet18",
}
bench["display_name"] = bench["model_name"].map(model_name_map)

# ── Load ensemble data ──
blend_path = os.path.join(ROOT, "results", "training", "step5_ensemble", "blend_results.csv")
blend = pd.read_csv(blend_path)
ens = blend[blend["alpha"] == 0.8].iloc[0]

# Add ensemble row
ens_row = {
    "display_name": "Ensemble",
    "val_s1_auc": ens["s1_auc"],
    "val_s1_sensitivity": ens["s1_sensitivity"],
    "val_s1_specificity": ens["s1_specificity"],
    "val_s2_f1_macro": ens["s2_f1_macro"],
    "val_s2_auc": ens["s2_auc"],
}
bench = pd.concat([bench, pd.DataFrame([ens_row])], ignore_index=True)

# Sort by MODEL_ORDER
bench["sort_key"] = bench["display_name"].map({m: i for i, m in enumerate(MODEL_ORDER)})
bench = bench.sort_values("sort_key").reset_index(drop=True)

# ── Figure ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.38))
fig.subplots_adjust(wspace=0.4)

# ── Helper: grouped horizontal bar chart ──
def plot_grouped_bars(ax, data, metrics, metric_labels, panel_label, title):
    add_panel_label(ax, panel_label)

    n_models = len(data)
    n_metrics = len(metrics)
    bar_height = 0.7 / n_metrics
    y_positions = np.arange(n_models)

    metric_colors = ["#2C3E50", "#7F8C8D", "#B0BEC5"][:n_metrics]

    for j, (metric, mlabel, mcolor) in enumerate(zip(metrics, metric_labels, metric_colors)):
        offsets = y_positions + (j - (n_metrics - 1) / 2) * bar_height
        values = data[metric].values

        bars = ax.barh(offsets, values, height=bar_height * 0.9,
                       color=mcolor, alpha=0.85, label=mlabel,
                       edgecolor="white", linewidth=0.3)

        # Value labels
        for bar, val in zip(bars, values):
            ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=FONT_SIZE["annotation"],
                    color="#333333")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(data["display_name"].values)

    # Highlight ensemble row
    ens_idx = data[data["display_name"] == "Ensemble"].index
    if len(ens_idx) > 0:
        idx = list(data.index).index(ens_idx[0])
        ax.get_yticklabels()[idx].set_fontweight("bold")

    ax.set_title(title, fontsize=FONT_SIZE["title"], pad=8)
    ax.legend(loc="lower left", frameon=True, edgecolor="#DDDDDD",
              fancybox=False, framealpha=0.9)
    apply_nature_style(ax)

    # Set x limits to show differences, leave room for labels
    all_vals = data[metrics].values.flatten()
    x_min = max(0, all_vals.min() - 0.05)
    x_max = all_vals.max() + 0.06
    ax.set_xlim(x_min, x_max)

# ── Panel (a): Stage 1 ──
plot_grouped_bars(
    ax1, bench,
    metrics=["val_s1_auc", "val_s1_sensitivity", "val_s1_specificity"],
    metric_labels=["AUC", "Sensitivity", "Specificity"],
    panel_label="a",
    title="Stage 1: Cancer Screening",
)

# ── Panel (b): Stage 2 ──
plot_grouped_bars(
    ax2, bench,
    metrics=["val_s2_f1_macro", "val_s2_auc"],
    metric_labels=["F1-macro", "AUC"],
    panel_label="b",
    title="Stage 2: Cancer Type ID",
)

save_figure(fig, "fig5_model_comparison")
plt.close()
print("Figure 5 complete.")
