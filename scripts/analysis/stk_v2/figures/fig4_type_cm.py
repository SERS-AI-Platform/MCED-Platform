#!/usr/bin/env python3
"""STK-V2 Figure 4 — Cancer-type confusion matrix + per-cancer F1.

Refactored from usersnet_fig4_type_cm.py to use reconstructed stacking OOF predictions.

Outputs:
- stk_v2_fig4_type_cm_f1.png
- stk_v2_fig4_type_cm_counts.csv
- stk_v2_fig4_type_cm_normalized.csv
- stk_v2_fig4_type_metrics.csv
- stk_v2_fig4_macro_metrics.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

try:
    from .utils import (
        NATURE_BLUE,
        NATURE_GOLD,
        NATURE_MUTED,
        NATURE_RED,
        NATURE_TEAL,
        NATURE_TEXT,
        add_panel_label,
        apply_nature_style,
        get_palette_and_labels,
        light_colormap,
        reconstruct_stacking_oof,
        save_nature_figure,
        style_axis,
        style_matrix_axis,
    )
except ImportError:
    from utils import (  # type: ignore
        NATURE_BLUE,
        NATURE_GOLD,
        NATURE_MUTED,
        NATURE_RED,
        NATURE_TEAL,
        NATURE_TEXT,
        add_panel_label,
        apply_nature_style,
        get_palette_and_labels,
        light_colormap,
        reconstruct_stacking_oof,
        save_nature_figure,
        style_axis,
        style_matrix_axis,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=str, default=None, help='STK-V2 output dir (default: results/training/stacking_v2)')
    ap.add_argument('--fig-dir', type=str, default=None, help='Figure output dir (default: results/figures/training/stacking_v2)')
    ap.add_argument('--dpi', type=int, default=300)
    args = ap.parse_args()
    apply_nature_style()

    run_dir = Path(args.run_dir) if args.run_dir else Path('results') / 'training' / 'stacking_v2'
    fig_dir = Path(args.fig_dir) if args.fig_dir else Path('results') / 'figures' / 'training' / 'stacking_v2'
    fig_dir.mkdir(parents=True, exist_ok=True)

    order, internal_idx, colors, labels_disp = get_palette_and_labels()
    n_classes = len(order)

    bl, ctl, s1_prob, s2_prob, _ = reconstruct_stacking_oof(run_dir, n_classes=n_classes)

    cancer_mask = ctl >= 0
    y_true_int = ctl[cancer_mask]
    y_pred_int = np.argmax(s2_prob[cancer_mask], axis=1)

    order_idx = [internal_idx[c] for c in order]

    cm = confusion_matrix(y_true_int, y_pred_int, labels=order_idx)
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)

    per_f1 = f1_score(y_true_int, y_pred_int, labels=order_idx, average=None, zero_division=0)
    per_prec = precision_score(y_true_int, y_pred_int, labels=order_idx, average=None, zero_division=0)
    per_rec = recall_score(y_true_int, y_pred_int, labels=order_idx, average=None, zero_division=0)
    macro_f1 = f1_score(y_true_int, y_pred_int, labels=order_idx, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true_int, y_pred_int, labels=order_idx, average='weighted', zero_division=0)

    # Compute sensitivity (recall) and specificity for each cancer type
    per_sensitivity = np.zeros(n_classes)
    per_specificity = np.zeros(n_classes)
    for idx, i in enumerate(order_idx):
        tp = cm[idx, idx]
        fn = cm[idx, :].sum() - tp
        fp = cm[:, idx].sum() - tp
        tn = cm.sum() - tp - fp - fn
        per_sensitivity[idx] = tp / (tp + fn) if (tp + fn) > 0 else 0
        per_specificity[idx] = tn / (tn + fp) if (tn + fp) > 0 else 0

    fig = plt.figure(figsize=(7.35, 3.75))
    gs = GridSpec(1, 2, width_ratios=[1.25, 1.0], wspace=0.32)
    ax_cm = fig.add_subplot(gs[0])
    ax_f1 = fig.add_subplot(gs[1])

    im = ax_cm.imshow(cm_norm, cmap=light_colormap('fig4_confusion_blue', NATURE_BLUE), vmin=0, vmax=1, aspect='equal')
    for (i, j), v in np.ndenumerate(cm):
        norm_v = cm_norm[i, j]
        if int(v) == 0 and norm_v < 0.005:
            continue
        txt = f"{v}\n{norm_v:.2f}"
        ax_cm.text(j, i, txt,
                   ha='center', va='center',
                   color='white' if norm_v > 0.62 else NATURE_TEXT,
                   fontsize=6.8,
                   fontweight='bold' if i == j else 'normal')

    ax_cm.set_xticks(range(n_classes)); ax_cm.set_yticks(range(n_classes))
    ax_cm.set_xticklabels(labels_disp, rotation=0)
    ax_cm.set_yticklabels(labels_disp)
    ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('True')
    ax_cm.set_title('Confusion matrix')
    style_matrix_axis(ax_cm, n_classes, n_classes)
    add_panel_label(ax_cm, 'a', x=-0.18, y=1.08)

    xs = np.arange(n_classes)
    bar_colors = [colors[c] for c in order]
    ax_f1.bar(xs, per_f1, color=bar_colors, edgecolor='white', linewidth=0.5, width=0.72)
    for x, f, n in zip(xs, per_f1, cm.sum(axis=1)):
        ax_f1.text(x, min(f + 0.025, 1.085), f"{f:.2f}", ha='center', fontsize=6.6, color=NATURE_MUTED)

    ax_f1.axhline(macro_f1, color=NATURE_RED, lw=1.0, ls='--', label=f"Macro F1 {macro_f1:.2f}")
    ax_f1.axhline(weighted_f1, color=NATURE_MUTED, lw=0.95, ls=':', label=f"Weighted F1 {weighted_f1:.2f}")
    ax_f1.set_xticks(xs); ax_f1.set_xticklabels(labels_disp)
    ax_f1.set_ylim(0, 1.12)
    ax_f1.set_ylabel('F1 score')
    ax_f1.set_title('Per-cancer F1')
    style_axis(ax_f1, grid='y')
    ax_f1.legend(loc='upper left', bbox_to_anchor=(0.00, -0.18), ncol=2, columnspacing=1.0, handlelength=1.7)
    add_panel_label(ax_f1, 'b', x=-0.18, y=1.08)

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.24, top=0.86, wspace=0.30)
    out_png = fig_dir / 'stk_v2_fig4_type_cm_f1.png'
    save_nature_figure(fig, out_png, dpi=args.dpi)
    plt.close(fig)

    # ── Figure 2: Sensitivity/Specificity 바 차트 ──
    fig, ax = plt.subplots(figsize=(5.35, 3.1))
    xs = np.arange(n_classes)
    width = 0.35

    ax.bar(xs - width/2, per_sensitivity, width, label='Sensitivity', color=NATURE_BLUE, edgecolor='white', linewidth=0.5)
    ax.bar(xs + width/2, per_specificity, width, label='Specificity', color=NATURE_RED, edgecolor='white', linewidth=0.5)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels_disp)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Sensitivity and specificity')
    ax.legend(loc='lower right')
    style_axis(ax, grid='y')

    plt.tight_layout()
    out_png2 = fig_dir / 'stk_v2_fig4_type_sens_spec.png'
    save_nature_figure(fig, out_png2, dpi=args.dpi)
    plt.close(fig)

    # ── Figure 3: Precision/Recall 바 차트 ──
    fig, ax = plt.subplots(figsize=(5.35, 3.1))
    xs = np.arange(n_classes)
    width = 0.35

    ax.bar(xs - width/2, per_prec, width, label='Precision', color=NATURE_GOLD, edgecolor='white', linewidth=0.5)
    ax.bar(xs + width/2, per_rec, width, label='Recall', color=NATURE_RED, edgecolor='white', linewidth=0.5)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels_disp)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Precision and recall')
    ax.legend(loc='lower right')
    style_axis(ax, grid='y')

    plt.tight_layout()
    out_png3 = fig_dir / 'stk_v2_fig4_type_prec_rec.png'
    save_nature_figure(fig, out_png3, dpi=args.dpi)
    plt.close(fig)

    # ── Figure 4: 모든 메트릭 Heatmap (암종 × 메트릭) ──
    metrics_data = np.column_stack([
        per_sensitivity,
        per_specificity,
        per_prec,
        per_rec,
        per_f1,
    ])
    metric_names = ['Sensitivity', 'Specificity', 'Precision', 'Recall', 'F1']

    fig, ax = plt.subplots(figsize=(5.2, 3.25))
    im = ax.imshow(metrics_data.T, cmap=light_colormap('fig4_metric_teal', NATURE_TEAL), vmin=0, vmax=1, aspect='auto')

    # 각 셀에 값 표시
    for i in range(len(labels_disp)):
        for j, metric_name in enumerate(metric_names):
            val = metrics_data[i, j]
            ax.text(i, j, f'{val:.3f}',
                   ha='center', va='center',
                   color='white' if val > 0.68 else NATURE_TEXT,
                   fontsize=6.9, fontweight='bold' if val > 0.90 else 'normal')

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(labels_disp, rotation=0)
    ax.set_yticks(range(len(metric_names)))
    ax.set_yticklabels(metric_names)
    ax.set_xlabel('Cancer type')
    ax.set_ylabel('Metric')
    ax.set_title('Performance metrics')
    style_matrix_axis(ax, len(metric_names), n_classes)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label('Score', fontsize=7.2)
    cbar.ax.tick_params(labelsize=6.8, length=2, width=0.5)
    plt.tight_layout()
    out_png4 = fig_dir / 'stk_v2_fig4_type_metrics_heatmap.png'
    save_nature_figure(fig, out_png4, dpi=args.dpi)
    plt.close(fig)

    # Save CSVs
    pd.DataFrame(cm, index=labels_disp, columns=labels_disp).to_csv(fig_dir / 'stk_v2_fig4_type_cm_counts.csv', encoding='utf-8-sig')
    pd.DataFrame(cm_norm, index=labels_disp, columns=labels_disp).to_csv(fig_dir / 'stk_v2_fig4_type_cm_normalized.csv', encoding='utf-8-sig')
    pd.DataFrame({
        'cancer': order,
        'n': cm.sum(axis=1),
        'sensitivity': per_sensitivity,
        'specificity': per_specificity,
        'precision': per_prec,
        'recall': per_rec,
        'f1': per_f1
    }).to_csv(fig_dir / 'stk_v2_fig4_type_metrics.csv', index=False, encoding='utf-8-sig')
    (fig_dir / 'stk_v2_fig4_macro_metrics.txt').write_text(
        f"""macro_f1={macro_f1:.4f}
weighted_f1={weighted_f1:.4f}
mean_sensitivity={per_sensitivity.mean():.4f}
mean_specificity={per_specificity.mean():.4f}
n_cancer={int(cancer_mask.sum())}
""",
        encoding='utf-8'
    )

    print(f"Saved figures:")
    print(f"  {out_png}")
    print(f"  {out_png2}")
    print(f"  {out_png3}")
    print(f"  {out_png4}")


if __name__ == '__main__':
    main()
