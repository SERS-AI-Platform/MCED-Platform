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
    from .utils import reconstruct_stacking_oof, get_palette_and_labels
except ImportError:
    from utils import reconstruct_stacking_oof, get_palette_and_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=str, default=None, help='STK-V2 output dir (default: results/training/stacking_v2)')
    ap.add_argument('--fig-dir', type=str, default=None, help='Figure output dir (default: results/figures/training/stacking_v2)')
    ap.add_argument('--dpi', type=int, default=300)
    args = ap.parse_args()

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

    fig = plt.figure(figsize=(11.4, 5.8))
    gs = GridSpec(1, 2, width_ratios=[1.25, 1.0], wspace=0.32)
    ax_cm = fig.add_subplot(gs[0])
    ax_f1 = fig.add_subplot(gs[1])

    im = ax_cm.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    for (i, j), v in np.ndenumerate(cm):
        norm_v = cm_norm[i, j]
        txt = f"{v}({norm_v:.2f})"
        ax_cm.text(j, i, txt,
                   ha='center', va='center',
                   color='white' if norm_v > 0.5 else 'black',
                   fontsize=9,
                   fontweight='bold' if i == j else 'normal')

    ax_cm.set_xticks(range(n_classes)); ax_cm.set_yticks(range(n_classes))
    ax_cm.set_xticklabels(labels_disp, rotation=0)
    ax_cm.set_yticklabels(labels_disp)
    ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('True')
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04, label='Row-normalized')

    xs = np.arange(n_classes)
    bar_colors = [colors[c] for c in order]
    ax_f1.bar(xs, per_f1, color=bar_colors, edgecolor='black', linewidth=0.6)
    for x, f, n in zip(xs, per_f1, cm.sum(axis=1)):
        ax_f1.text(x, f + 0.02, f"{f:.3f}\n(n={int(n)})", ha='center', fontsize=8.5)

    ax_f1.axhline(macro_f1, color='#C0392B', lw=1.5, ls='--', label=f"Macro F1 = {macro_f1:.3f}")
    ax_f1.axhline(weighted_f1, color='#444', lw=1.2, ls=':', label=f"Weighted F1 = {weighted_f1:.3f}")
    ax_f1.set_xticks(xs); ax_f1.set_xticklabels(labels_disp)
    ax_f1.set_ylim(0, 1.1)
    ax_f1.set_ylabel('F1 score')
    ax_f1.grid(alpha=0.25, axis='y'); ax_f1.set_axisbelow(True)
    ax_f1.legend(loc='lower right', fontsize=8.5)

    plt.tight_layout()
    out_png = fig_dir / 'stk_v2_fig4_type_cm_f1.png'
    plt.savefig(out_png, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    # ── Figure 2: Sensitivity/Specificity 바 차트 ──
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(n_classes)
    width = 0.35

    ax.bar(xs - width/2, per_sensitivity, width, label='Sensitivity', color='#2E86AB', edgecolor='black', linewidth=0.6)
    ax.bar(xs + width/2, per_specificity, width, label='Specificity', color='#A23B72', edgecolor='black', linewidth=0.6)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels_disp)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Stage 2: Sensitivity vs Specificity by Cancer Type', fontweight='bold', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.25, axis='y')
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_png2 = fig_dir / 'stk_v2_fig4_type_sens_spec.png'
    plt.savefig(out_png2, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    # ── Figure 3: Precision/Recall 바 차트 ──
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(n_classes)
    width = 0.35

    ax.bar(xs - width/2, per_prec, width, label='Precision', color='#F18F01', edgecolor='black', linewidth=0.6)
    ax.bar(xs + width/2, per_rec, width, label='Recall', color='#C73E1D', edgecolor='black', linewidth=0.6)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels_disp)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Stage 2: Precision vs Recall by Cancer Type', fontweight='bold', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.25, axis='y')
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_png3 = fig_dir / 'stk_v2_fig4_type_prec_rec.png'
    plt.savefig(out_png3, dpi=args.dpi, bbox_inches='tight')
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

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(metrics_data.T, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    # 각 셀에 값 표시
    for i in range(len(labels_disp)):
        for j, metric_name in enumerate(metric_names):
            val = metrics_data[i, j]
            ax.text(i, j, f'{val:.3f}',
                   ha='center', va='center',
                   color='white' if val < 0.5 else 'black',
                   fontsize=9, fontweight='bold')

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(labels_disp, rotation=45, ha='right')
    ax.set_yticks(range(len(metric_names)))
    ax.set_yticklabels(metric_names)
    ax.set_xlabel('Cancer Type', fontweight='bold')
    ax.set_ylabel('Metric', fontweight='bold')
    ax.set_title('Stage 2: Performance Metrics Heatmap', fontweight='bold', fontsize=12)

    plt.colorbar(im, ax=ax, label='Score', fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_png4 = fig_dir / 'stk_v2_fig4_type_metrics_heatmap.png'
    plt.savefig(out_png4, dpi=args.dpi, bbox_inches='tight')
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
