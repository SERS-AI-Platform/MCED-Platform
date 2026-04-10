#!/usr/bin/env python3
"""
AACR Poster — Fig 7: Model Architecture Diagram
LR + ResNet18-1D Ensemble with Clinical Fusion
5-cancer (Phase Q) results
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# === Color palette (consistent with existing AACR figures) ===
PAL = dict(
    lr_fill='#e3f2fd',    lr_border='#1565c0',   lr_text='#0d47a1',
    rn_fill='#e8f5e9',    rn_border='#2e7d32',   rn_text='#1b5e20',
    ens_fill='#fce4ec',   ens_border='#c62828',  ens_text='#b71c1c',
    fus_fill='#f3e5f5',   fus_border='#6a1b9a',  fus_text='#4a148c',
    inp_fill='#37474f',   inp_text='#ffffff',
    out_fill='#263238',   out_text='#ffffff',
    s1='#1565c0',         s2='#e65100',
    arrow='#78909c',      gray='#9e9e9e',
    dark='#1a1a2e',       mid='#616161',
    layer_fills=['#c8e6c9', '#a5d6a7', '#81c784', '#66bb6a', '#4caf50', '#388e3c'],
)

# Cancer type colors
CC = dict(PRO='#1976d2', OVA='#7b1fa2', LUN='#388e3c', PAN='#ef6c00', CRC='#c62828')


# === Helper functions ===
def draw_box(ax, x, y, w, h, fill, border=None, lw=1.2, radius=0.02, zorder=2):
    """Draw a rounded rectangle and return its center for arrow connections."""
    border = border or fill
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=border, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(box)
    return box


def txt(ax, x, y, s, size=9, color='black', weight='bold', ha='center', va='center', **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=10, **kw)


def draw_arrow(ax, x1, y1, x2, y2, color=None, lw=1.3, style='->', headw=6, headl=8,
               linestyle='-', zorder=3):
    color = color or PAL['arrow']
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle=f'{style},head_width=0.15,head_length=0.1',
                    color=color, lw=lw, linestyle=linestyle,
                    shrinkA=0, shrinkB=0,
                ), zorder=zorder)


def draw_curved_arrow(ax, x1, y1, x2, y2, color=None, lw=1.3, rad=0.25, linestyle='-'):
    color = color or PAL['arrow']
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='->,head_width=0.15,head_length=0.1',
                    color=color, lw=lw, linestyle=linestyle,
                    connectionstyle=f'arc3,rad={rad}',
                    shrinkA=0, shrinkB=0,
                ), zorder=3)


# === Create figure ===
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')
fig.patch.set_facecolor('white')

# === Title ===
txt(ax, 8, 8.55, 'Model Architecture: LR + ResNet18-1D Ensemble with Clinical Fusion',
    size=14, color=PAL['dark'])
txt(ax, 8, 8.15, 'AECD Platform  |  SERS-Based Urine Multi-Cancer Screening  |  5 Cancer Types',
    size=9, color=PAL['mid'], weight='normal')

# ─────────────────────────────────────────────────────────────────────────────
# Column 1: INPUT (x ≈ 1.3)
# ─────────────────────────────────────────────────────────────────────────────
# SERS spectrum input
draw_box(ax, 1.3, 5.5, 2.0, 1.6, PAL['inp_fill'], lw=1.5)
txt(ax, 1.3, 5.9, 'SERS Spectrum', size=10, color=PAL['inp_text'])
txt(ax, 1.3, 5.5, '935 features', size=8.5, color='#b0bec5', weight='normal')
txt(ax, 1.3, 5.1, '(400–2200 cm⁻¹)', size=7, color='#78909c', weight='normal')

# Clinical features input
draw_box(ax, 1.3, 2.3, 1.8, 1.0, '#f3e5f5', PAL['fus_border'], lw=1.0)
txt(ax, 1.3, 2.6, 'Clinical', size=9, color=PAL['fus_text'])
txt(ax, 1.3, 2.25, 'Age, Sex, BMI', size=7.5, color=PAL['fus_text'], weight='normal')
txt(ax, 1.3, 1.95, '(3 features)', size=6.5, color=PAL['mid'], weight='normal')

# ─────────────────────────────────────────────────────────────────────────────
# Column 2: TWO MODEL PATHS (x ≈ 4.5–5.5)
# ─────────────────────────────────────────────────────────────────────────────

# --- Logistic Regression (upper path, y ≈ 6.8) ---
lr_x, lr_y = 4.8, 6.8
draw_box(ax, lr_x, lr_y, 3.2, 2.0, PAL['lr_fill'], PAL['lr_border'], lw=1.5)
txt(ax, lr_x, 7.45, 'Logistic Regression', size=10.5, color=PAL['lr_text'])
# Divider line
ax.plot([lr_x - 1.4, lr_x + 1.4], [7.15, 7.15], color=PAL['lr_border'], lw=0.5, alpha=0.4, zorder=5)
txt(ax, lr_x, 6.9, 'L2 (C=1.0), SAGA', size=7.5, color=PAL['mid'], weight='normal')
txt(ax, lr_x, 6.55, 'Balanced class weights', size=7.5, color=PAL['mid'], weight='normal')
txt(ax, lr_x, 6.2, 'S1: Binary → P(cancer)', size=7, color=PAL['s1'], weight='normal')
txt(ax, lr_x, 5.95, 'S2: OvR → 5-class probs', size=7, color=PAL['s2'], weight='normal')

# --- ResNet18-1D (lower path, y ≈ 4.2) ---
rn_x, rn_y = 4.8, 3.8
rn_w, rn_h = 3.2, 3.6
draw_box(ax, rn_x, rn_y, rn_w, rn_h, PAL['rn_fill'], PAL['rn_border'], lw=1.5)
txt(ax, rn_x, 5.35, 'ResNet18-1D', size=10.5, color=PAL['rn_text'])
# Divider
ax.plot([rn_x - 1.4, rn_x + 1.4], [5.1, 5.1], color=PAL['rn_border'], lw=0.5, alpha=0.4, zorder=5)

# Layer stack inside ResNet box
layers = [
    ('Conv1D(1→32, K=7, S=2)', '#c8e6c9'),
    ('BN + ReLU + MaxPool',     '#c8e6c9'),
    ('2× Block(32→32)',         '#a5d6a7'),
    ('2× Block(32→64, S=2)',    '#81c784'),
    ('2× Block(64→128, S=2)',   '#66bb6a'),
    ('2× Block(128→256, S=2)',  '#4caf50'),
    ('GlobalAvgPool → 256-d',   '#388e3c'),
]
layer_h = 0.3
layer_w = 2.6
ly_start = 4.85
for i, (lbl, col) in enumerate(layers):
    ly = ly_start - i * (layer_h + 0.06)
    draw_box(ax, rn_x, ly, layer_w, layer_h, col, border='#2e7d32', lw=0.6, radius=0.01, zorder=5)
    text_col = 'white' if i >= 5 else PAL['rn_text']
    txt(ax, rn_x, ly, lbl, size=6.5, color=text_col, weight='normal')
    # Small arrow between layers
    if i < len(layers) - 1:
        arr_y = ly - layer_h/2 - 0.01
        ax.annotate('', xy=(rn_x, arr_y - 0.04), xytext=(rn_x, arr_y + 0.01),
                    arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=0.6), zorder=5)

# Heads label
txt(ax, rn_x, 2.3, 'S1 Head: 256→64→1', size=6.5, color=PAL['s1'], weight='normal')
txt(ax, rn_x, 2.1, 'S2 Head: 256→64→5', size=6.5, color=PAL['s2'], weight='normal')

# --- Arrows from Input to models ---
draw_arrow(ax, 2.3, 5.7, 3.2, 6.5, PAL['lr_border'])     # Input → LR (up)
draw_arrow(ax, 2.3, 5.3, 3.2, 4.5, PAL['rn_border'])     # Input → ResNet (down)

# ─────────────────────────────────────────────────────────────────────────────
# Column 3: ENSEMBLE BLENDING (x ≈ 8.3)
# ─────────────────────────────────────────────────────────────────────────────
ens_x, ens_y = 8.3, 5.5
draw_box(ax, ens_x, ens_y, 2.6, 2.0, PAL['ens_fill'], PAL['ens_border'], lw=1.5)
txt(ax, ens_x, 6.15, 'Ensemble', size=10.5, color=PAL['ens_text'])
txt(ax, ens_x, 5.8, 'Blend', size=10.5, color=PAL['ens_text'])
# Divider
ax.plot([ens_x - 1.1, ens_x + 1.1], [5.5, 5.5], color=PAL['ens_border'], lw=0.5, alpha=0.4, zorder=5)
# Formula
txt(ax, ens_x, 5.2, 'P = α·P_LR + (1-α)·P_RN', size=7, color=PAL['ens_text'], weight='normal',
    fontstyle='italic')
txt(ax, ens_x, 4.85, 'α = 0.8 (optimal)', size=7.5, color=PAL['mid'], weight='normal')

# Arrows from models → ensemble
draw_arrow(ax, 6.4, 6.6, 7.0, 6.0, PAL['lr_border'], lw=1.5)
txt(ax, 6.5, 6.65, 'α=0.8', size=7, color=PAL['lr_text'], weight='normal')

draw_arrow(ax, 6.4, 4.2, 7.0, 5.0, PAL['rn_border'], lw=1.5)
txt(ax, 6.3, 4.0, '1-α=0.2', size=7, color=PAL['rn_text'], weight='normal')

# ─────────────────────────────────────────────────────────────────────────────
# Column 4: CLINICAL FUSION (x ≈ 11.5)
# ─────────────────────────────────────────────────────────────────────────────
fus_x, fus_y = 11.5, 5.5
draw_box(ax, fus_x, fus_y, 2.6, 2.2, PAL['fus_fill'], PAL['fus_border'], lw=1.5)
txt(ax, fus_x, 6.25, 'Clinical', size=10.5, color=PAL['fus_text'])
txt(ax, fus_x, 5.85, 'Fusion', size=10.5, color=PAL['fus_text'])
# Divider
ax.plot([fus_x - 1.1, fus_x + 1.1], [5.55, 5.55], color=PAL['fus_border'], lw=0.5, alpha=0.4, zorder=5)
txt(ax, fus_x, 5.25, '935 + 3 → 938 features', size=7.5, color=PAL['fus_text'], weight='normal')
txt(ax, fus_x, 4.9, 'Feature concatenation', size=7, color=PAL['mid'], weight='normal')
# Sex constraint
draw_box(ax, fus_x, 4.45, 2.2, 0.4, '#ede7f6', PAL['fus_border'], lw=0.7, radius=0.01)
txt(ax, fus_x, 4.45, 'Sex constraint', size=7, color=PAL['fus_text'], weight='normal')

# Arrow: Ensemble → Fusion
draw_arrow(ax, 9.6, 5.5, 10.2, 5.5, PAL['arrow'], lw=1.5)

# Dashed arrow: Clinical features → Fusion (bottom route with right-angle)
# Draw as two manual segments: right along bottom, then up to fusion
ax.annotate('', xy=(fus_x, 4.25), xytext=(fus_x, 1.5),
            arrowprops=dict(arrowstyle='->,head_width=0.15,head_length=0.1',
                            color=PAL['fus_border'], lw=1.2, linestyle='--'), zorder=3)
ax.plot([2.2, fus_x], [1.5, 1.5], color=PAL['fus_border'], lw=1.2, linestyle='--', zorder=3)
ax.plot([2.2, 2.2], [1.8, 1.5], color=PAL['fus_border'], lw=1.2, linestyle='--', zorder=3)

# ─────────────────────────────────────────────────────────────────────────────
# Column 5: OUTPUT (x ≈ 14.5)
# ─────────────────────────────────────────────────────────────────────────────

# --- Stage 1 output (top) ---
s1_x, s1_y = 14.5, 7.0
draw_box(ax, s1_x, s1_y, 2.6, 1.6, '#e3f2fd', PAL['s1'], lw=1.5)
txt(ax, s1_x, 7.5, 'STAGE 1', size=9, color=PAL['s1'])
txt(ax, s1_x, 7.15, 'Cancer Screening', size=8, color=PAL['s1'], weight='normal')
# Cancer / Non-Cancer pills
draw_box(ax, 14.0, 6.65, 1.0, 0.3, '#c62828', lw=0, radius=0.01)
txt(ax, 14.0, 6.65, 'Cancer', size=6.5, color='white')
draw_box(ax, 15.1, 6.65, 1.2, 0.3, '#2e7d32', lw=0, radius=0.01)
txt(ax, 15.1, 6.65, 'Non-Cancer', size=6.5, color='white')
txt(ax, s1_x, 6.3, 'AUC 0.984', size=8, color=PAL['s1'], weight='normal', fontstyle='italic')

# --- Stage 2 output (bottom) ---
s2_x, s2_y = 14.5, 4.5
draw_box(ax, s2_x, s2_y, 2.6, 2.4, '#fff3e0', PAL['s2'], lw=1.5)
txt(ax, s2_x, 5.4, 'STAGE 2', size=9, color=PAL['s2'])
txt(ax, s2_x, 5.05, 'Cancer Type ID', size=8, color=PAL['s2'], weight='normal')

# 5 cancer type pills
cc_list = [('PRO', CC['PRO']), ('OVA', CC['OVA']), ('LUN', CC['LUN']),
           ('PAN', CC['PAN']), ('CRC', CC['CRC'])]
pill_y = [4.65, 4.3, 3.95, 3.6, 3.6]  # Stack vertically then wrap
pill_positions = [(13.7, 4.65), (15.3, 4.65), (13.7, 4.3), (15.3, 4.3), (14.5, 3.95)]

for (name, col), (px, py) in zip(cc_list, pill_positions):
    draw_box(ax, px, py, 1.0, 0.28, col, lw=0, radius=0.01)
    txt(ax, px, py, name, size=7, color='white')

txt(ax, s2_x, 3.55, 'F1 macro 0.875', size=8, color=PAL['s2'], weight='normal', fontstyle='italic')

# Arrows: Fusion → Outputs
draw_arrow(ax, 12.8, 6.0, 13.2, 6.8, PAL['s1'], lw=1.5)
draw_arrow(ax, 12.8, 5.2, 13.2, 5.0, PAL['s2'], lw=1.5)

# ─────────────────────────────────────────────────────────────────────────────
# Bottom: Performance comparison bar
# ─────────────────────────────────────────────────────────────────────────────
draw_box(ax, 8, 0.7, 14.5, 0.7, '#fafafa', '#e0e0e0', lw=0.8, radius=0.01)

metrics_text = (
    'SERS-only LR:  S1 AUC 0.981  |  S2 F1 0.871        '
    'Ensemble (α=0.8):  S1 AUC 0.984  |  S2 F1 0.875        '
    'Fusion + Sex constraint:  S1 AUC 0.977  |  S2 F1 0.892'
)
txt(ax, 8, 0.7, metrics_text, size=7.5, color=PAL['mid'], weight='normal')

# No legend dots — metrics bar text is self-explanatory

# ─────────────────────────────────────────────────────────────────────────────
# Optional labels / annotations
# ─────────────────────────────────────────────────────────────────────────────

# Path labels (horizontal, above arrows)
txt(ax, 2.7, 6.7, 'Primary', size=7, color=PAL['lr_border'],
    weight='normal', fontstyle='italic')
txt(ax, 2.7, 4.6, 'Secondary', size=7, color=PAL['rn_border'],
    weight='normal', fontstyle='italic')

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
out_dir = '/home/user/SERS-AI/AACR'
fig.savefig(f'{out_dir}/fig7_model_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(f'{out_dir}/fig7_model_architecture.pdf', bbox_inches='tight', facecolor='white')
print(f"Saved: {out_dir}/fig7_model_architecture.png")
print(f"Saved: {out_dir}/fig7_model_architecture.pdf")
plt.close()
