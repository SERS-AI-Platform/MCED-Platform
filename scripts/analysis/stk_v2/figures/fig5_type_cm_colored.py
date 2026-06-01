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

try:
    from .utils import (
        NATURE_TEXT,
        apply_nature_style,
        get_palette_and_labels,
        reconstruct_stacking_oof,
        save_nature_figure,
        softened_color,
        style_matrix_axis,
    )
except ImportError:
    from utils import (  # type: ignore
        NATURE_TEXT,
        apply_nature_style,
        get_palette_and_labels,
        reconstruct_stacking_oof,
        save_nature_figure,
        softened_color,
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

    n = n_classes
    rgba = np.ones((n, n, 4))
    for i, cancer in enumerate(order):
        base = np.array(to_rgba(colors[cancer]))
        base[:3] = softened_color(colors[cancer], amount=0.08)
        for j in range(n):
            v = cm_norm[i, j]
            rgba[i, j] = (1 - v) * np.array([1, 1, 1, 1]) + v * base
            rgba[i, j, 3] = 1.0

    fig, ax = plt.subplots(figsize=(4.85, 4.45))
    ax.imshow(rgba, aspect='equal')

    for (i, j), v in np.ndenumerate(cm):
        norm_v = cm_norm[i, j]
        if int(v) == 0 and norm_v < 0.005:
            continue
        txt = f"{v}\n({norm_v:.2f})"
        col = NATURE_TEXT if norm_v < 0.55 else 'white'
        ax.text(j, i, txt, ha='center', va='center', color=col, fontsize=7.4,
                fontweight='bold' if i == j else 'normal')

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels_disp)
    ax.set_yticklabels(labels_disp)

    for tick, cancer in zip(ax.get_yticklabels(), order):
        tick.set_color(colors[cancer]); tick.set_fontweight('bold')
    for tick, cancer in zip(ax.get_xticklabels(), order):
        tick.set_color(colors[cancer]); tick.set_fontweight('bold')

    ax.set_xlabel('Predicted class'); ax.set_ylabel('True class')
    ax.set_title('Cancer-type confusion matrix')
    style_matrix_axis(ax, n, n)

    plt.tight_layout()
    out_png = fig_dir / 'stk_v2_fig5_type_cm_colored.png'
    save_nature_figure(fig, out_png, dpi=args.dpi)
    plt.close(fig)

    pd.DataFrame(cm, index=labels_disp, columns=labels_disp).to_csv(fig_dir / 'stk_v2_fig5_type_cm_counts.csv', encoding='utf-8-sig')
    pd.DataFrame(cm_norm, index=labels_disp, columns=labels_disp).to_csv(fig_dir / 'stk_v2_fig5_type_cm_normalized.csv', encoding='utf-8-sig')
    print(f"Saved: {out_png}")


if __name__ == '__main__':
    main()
