#!/usr/bin/env python3
"""STK-V2 Figure 2 (Type ID) — refactored from usersnet_fig2_type_id.py.

Left: per-cancer sensitivity (recall) bar plot with bootstrap 95% CI.
Right: One-vs-Rest ROC curves for 7 cancer types using reconstructed stacking OOF probs.

Inputs (default):
- results/training/stacking_v2/oof_predictions.npz
- results/training/stacking_v2/checkpoints/oof_<model>_outerfold<k>.npz
- results/training/stacking_v2/best_ensemble_config.json

Outputs (default):
- results/figures/training/stacking_v2/stk_v2_fig2_type_id.png
- .../stk_v2_fig2_type_id_metrics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, auc as _auc

try:
    from .utils import (
        NATURE_MUTED,
        add_panel_label,
        apply_nature_style,
        get_palette_and_labels,
        reconstruct_stacking_oof,
        save_nature_figure,
        style_axis,
    )
except ImportError:
    from utils import (  # type: ignore
        NATURE_MUTED,
        add_panel_label,
        apply_nature_style,
        get_palette_and_labels,
        reconstruct_stacking_oof,
        save_nature_figure,
        style_axis,
    )


def boot_ci_recall(y_true: np.ndarray, y_pred: np.ndarray, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y_true == 1)[0]
    if len(idx_pos) == 0:
        return np.nan, np.nan, np.nan
    point = (y_pred[idx_pos] == 1).mean()
    vals = []
    for _ in range(n):
        samp = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        vals.append((y_pred[samp] == 1).mean())
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=str, default=None, help='STK-V2 output dir (default: results/training/stacking_v2)')
    ap.add_argument('--fig-dir', type=str, default=None, help='Figure output dir (default: results/figures/training/stacking_v2)')
    ap.add_argument('--dpi', type=int, default=300)
    args = ap.parse_args()
    apply_nature_style()

    # Defaults consistent with repo config.py (RESULTS_DIR/FIG_DIR)
    project_root = Path(__file__).resolve().parents[0]
    # If script is placed under repo, adjust project_root accordingly if needed

    run_dir = Path(args.run_dir) if args.run_dir else Path('results') / 'training' / 'stacking_v2'
    fig_dir = Path(args.fig_dir) if args.fig_dir else Path('results') / 'figures' / 'training' / 'stacking_v2'
    fig_dir.mkdir(parents=True, exist_ok=True)

    order, internal_idx, colors, labels_disp = get_palette_and_labels()
    n_classes = len(order)

    bl, ctl, s1_prob, s2_prob, _ = reconstruct_stacking_oof(run_dir, n_classes=n_classes)

    # Cancer-only subset for type evaluation
    cancer_mask = ctl >= 0
    y_true_int = ctl[cancer_mask]
    y_pred_int = np.argmax(s2_prob[cancer_mask], axis=1)

    order_idx = [internal_idx[c] for c in order]

    # Per-cancer recall + bootstrap CI
    metrics_rows = []
    recalls, ci_lo, ci_hi = [], [], []
    for c in order:
        ci = internal_idx[c]
        y_true_bin = (y_true_int == ci).astype(int)
        y_pred_bin = (y_pred_int == ci).astype(int)
        r, lo, hi = boot_ci_recall(y_true_bin, y_pred_bin)
        recalls.append(r)
        ci_lo.append(lo)
        ci_hi.append(hi)
        metrics_rows.append({'cancer': c, 'n_pos': int(y_true_bin.sum()), 'recall': r, 'ci_lo': lo, 'ci_hi': hi})

    # OVR ROC for each class using probabilities
    fig = plt.figure(figsize=(7.35, 3.55))
    gs = GridSpec(1, 2, width_ratios=[1, 1.22], wspace=0.30)

    # Left: bar plot
    ax0 = fig.add_subplot(gs[0])
    xs = np.arange(n_classes)
    bar_colors = [colors[c] for c in order]
    ax0.bar(xs, recalls, color=bar_colors, edgecolor='white', linewidth=0.5, width=0.72)
    ax0.errorbar(xs, recalls, yerr=[np.array(recalls) - np.array(ci_lo), np.array(ci_hi) - np.array(recalls)],
                fmt='none', ecolor=NATURE_MUTED, elinewidth=0.75, capsize=2.3, capthick=0.75)
    for x, r in zip(xs, recalls):
        ax0.text(x, min(r + 0.025, 1.085), f"{r:.2f}", ha='center', fontsize=6.8, color=NATURE_MUTED)
    ax0.set_xticks(xs)
    ax0.set_xticklabels(labels_disp, rotation=0)
    ax0.set_ylim(0, 1.10)
    ax0.set_ylabel('Sensitivity')
    ax0.set_title('Per-cancer sensitivity')
    style_axis(ax0, grid='y')
    add_panel_label(ax0, 'a', x=-0.18, y=1.08)

    # Right: OVR ROC curves
    ax1 = fig.add_subplot(gs[1])
    for c in order:
        ci = internal_idx[c]
        y_true_bin = (y_true_int == ci).astype(int)
        # Use s2_prob on cancer-only subset
        p = s2_prob[cancer_mask, ci]
        fpr, tpr, _ = roc_curve(y_true_bin, p)
        roc_auc = _auc(fpr, tpr)
        ax1.plot(fpr, tpr, lw=1.15, color=colors[c], label=f"{c} {roc_auc:.2f}")
    ax1.plot([0, 1], [0, 1], color=NATURE_MUTED, lw=0.8, ls=(0, (3, 3)))
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel('False positive rate')
    ax1.set_ylabel('True positive rate')
    ax1.set_title('One-vs-rest ROC')
    style_axis(ax1, grid='both')
    ax1.legend(fontsize=6.3, ncol=1, loc='lower right', title='AUROC', title_fontsize=6.5)
    add_panel_label(ax1, 'b', x=-0.16, y=1.08)

    out_png = fig_dir / 'stk_v2_fig2_type_id.png'
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.16, top=0.86, wspace=0.32)
    save_nature_figure(fig, out_png, dpi=args.dpi)
    plt.close(fig)

    pd.DataFrame(metrics_rows).to_csv(fig_dir / 'stk_v2_fig2_type_id_metrics.csv', index=False, encoding='utf-8-sig')
    print(f"Saved: {out_png}")


if __name__ == '__main__':
    main()
