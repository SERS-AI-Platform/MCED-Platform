#!/usr/bin/env python3
"""STK-V2 Figure 5 — Colored confusion matrix (row color = true cancer).

Refactored from usersnet_fig5_type_cm_colored.py to use reconstructed stacking OOF predictions.

Outputs:
- stk_v2_fig5_type_cm_colored.png
- stk_v2_fig5_type_cm_counts.csv
- stk_v2_fig5_type_cm_normalized.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from sklearn.metrics import confusion_matrix

from stk_v2_fig_utils import reconstruct_stacking_oof, get_palette_and_labels


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

    n = n_classes
    rgba = np.ones((n, n, 4))
    for i, cancer in enumerate(order):
        base = np.array(to_rgba(colors[cancer]))
        for j in range(n):
            v = cm_norm[i, j]
            rgba[i, j] = (1 - v) * np.array([1, 1, 1, 1]) + v * base
            rgba[i, j, 3] = 1.0

    fig, ax = plt.subplots(figsize=(7.7, 6.6))
    ax.imshow(rgba, aspect='equal')

    for (i, j), v in np.ndenumerate(cm):
        norm_v = cm_norm[i, j]
        txt = f"{v}\n({norm_v:.2f})"
        col = 'black' if norm_v < 0.55 else 'white'
        ax.text(j, i, txt, ha='center', va='center', color=col, fontsize=10,
                fontweight='bold' if i == j else 'normal')

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels_disp, fontsize=10)
    ax.set_yticklabels(labels_disp, fontsize=10)

    for tick, cancer in zip(ax.get_yticklabels(), order):
        tick.set_color(colors[cancer]); tick.set_fontweight('bold')
    for tick, cancer in zip(ax.get_xticklabels(), order):
        tick.set_color(colors[cancer]); tick.set_fontweight('bold')

    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.2)
    ax.tick_params(which='minor', length=0)

    plt.tight_layout()
    out_png = fig_dir / 'stk_v2_fig5_type_cm_colored.png'
    plt.savefig(out_png, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    pd.DataFrame(cm, index=labels_disp, columns=labels_disp).to_csv(fig_dir / 'stk_v2_fig5_type_cm_counts.csv', encoding='utf-8-sig')
    pd.DataFrame(cm_norm, index=labels_disp, columns=labels_disp).to_csv(fig_dir / 'stk_v2_fig5_type_cm_normalized.csv', encoding='utf-8-sig')
    print(f"Saved: {out_png}")


if __name__ == '__main__':
    main()
