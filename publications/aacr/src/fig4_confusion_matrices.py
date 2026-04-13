"""
AACR Poster — Confusion Matrices
CM1: Cancer Screening (2x2) — Cancer vs Non-Cancer
CM2: Cancer Type ID (5x5) — PRO, OVA, LUN, PAN, CRC
Colors matched to poster bar chart palette.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Color palette (from bar chart) ──
colors = {
    'Non-Cancer': '#1b3a5c',
    'Cancer':     '#1b3a5c',
    'PRO': '#e8733a',
    'PAN': '#a01020',
    'OVA': '#7b2d8e',
    'CRC': '#2980b9',
    'LUN': '#d4a088',
}

# ── Stage 1: Cancer Screening CM (2x2) ──
# From R_BLC_added evaluation
s1_counts = np.array([
    [364, 36],    # Non-cancer → [Non-cancer, Cancer]
    [104, 1137],  # Cancer → [Non-cancer, Cancer]
])
s1_labels = ['Non-Cancer', 'Cancer']
s1_totals = s1_counts.sum(axis=1, keepdims=True)
s1_pct = s1_counts / s1_totals * 100

# ── Stage 2: Cancer Type ID CM (5x5) ──
cancer_labels = ['PRC', 'OVC', 'LC', 'PAC', 'CRC']
cancer_codes = ['PRO', 'OVA', 'LUN', 'CPAN', 'CRC']
color_keys = ['PRO', 'OVA', 'LUN', 'PAN', 'CRC']
totals = [500, 350, 1500, 350, 1500]

# Off-diagonal errors (from confusion_pairs_exp002.csv)
errors = {
    ('PRO', 'OVA'): 29, ('PRO', 'LUN'): 10, ('PRO', 'CPAN'): 1, ('PRO', 'CRC'): 3,
    ('OVA', 'PRO'): 25, ('OVA', 'LUN'): 16, ('OVA', 'CPAN'): 5, ('OVA', 'CRC'): 8,
    ('LUN', 'PRO'): 44, ('LUN', 'OVA'): 34, ('LUN', 'CPAN'): 19, ('LUN', 'CRC'): 37,
    ('CPAN', 'PRO'): 1, ('CPAN', 'OVA'): 6, ('CPAN', 'LUN'): 5, ('CPAN', 'CRC'): 41,
    ('CRC', 'PRO'): 21, ('CRC', 'OVA'): 12, ('CRC', 'LUN'): 38, ('CRC', 'CPAN'): 58,
}

s2_counts = np.zeros((5, 5), dtype=int)
for i, tc in enumerate(cancer_codes):
    total_wrong = sum(errors.get((tc, pc), 0) for pc in cancer_codes if pc != tc)
    s2_counts[i, i] = totals[i] - total_wrong
    for j, pc in enumerate(cancer_codes):
        if i != j:
            s2_counts[i, j] = errors.get((tc, pc), 0)

s2_totals = np.array(totals).reshape(-1, 1)
s2_pct = s2_counts / s2_totals * 100


def lighten(hex_color, factor=0.85):
    """Lighten a hex color towards white."""
    rgb = mcolors.hex2color(hex_color)
    return tuple(c + (1 - c) * factor for c in rgb)


def draw_cm(ax, pct_matrix, labels, diag_colors, title, fontsize=28):
    n = len(labels)

    for i in range(n):
        for j in range(n):
            pct = pct_matrix[i, j]
            if i == j:
                # Diagonal: class color
                fc = diag_colors[i]
                tc = 'white'
            else:
                # Off-diagonal: light gray, darker if higher error
                intensity = min(pct / 15.0, 1.0)  # cap at 15%
                gray = 1.0 - intensity * 0.25  # range 0.75-1.0
                fc = (gray, gray, gray)
                tc = '#333333' if pct > 0 else '#aaaaaa'

            rect = plt.Rectangle((j, n - 1 - i), 1, 1, facecolor=fc,
                                  edgecolor='white', linewidth=3)
            ax.add_patch(rect)

            # Text
            if pct >= 0.5:
                label = f"{pct:.1f}%"
            elif pct > 0:
                label = f"{pct:.1f}%"
            else:
                label = "0%"

            ax.text(j + 0.5, n - 1 - i + 0.5, label,
                    ha='center', va='center', fontsize=fontsize,
                    fontweight='bold', color=tc)

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect('equal')

    # Axis labels
    ax.set_xticks([i + 0.5 for i in range(n)])
    ax.set_xticklabels(labels, fontsize=fontsize * 0.75, fontweight='bold',
                        rotation=30, ha='right')
    ax.set_yticks([i + 0.5 for i in range(n)])
    ax.set_yticklabels(list(reversed(labels)), fontsize=fontsize * 0.75,
                        fontweight='bold')

    ax.set_xlabel('Predicted', fontsize=fontsize * 0.85, fontweight='bold',
                   labelpad=12)
    ax.set_ylabel('Actual', fontsize=fontsize * 0.85, fontweight='bold',
                   labelpad=12)
    ax.set_title(title, fontsize=fontsize * 1.1, fontweight='bold', pad=20)

    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ── Figure: Two CMs side by side ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(44, 16))
fig.patch.set_facecolor('white')

# CM1: Cancer Screening
s1_diag_colors = [colors['Non-Cancer'], '#2c6e9e']  # navy, slightly lighter blue for cancer
draw_cm(ax1, s1_pct, s1_labels, s1_diag_colors,
        'A. Cancer Screening', fontsize=48)

# CM2: Cancer Type ID
s2_diag_colors = [colors[k] for k in color_keys]
draw_cm(ax2, s2_pct, cancer_labels, s2_diag_colors,
        'B. Cancer Type Identification', fontsize=40)

plt.tight_layout(pad=4)
plt.savefig('/home/user/SERS-AI/AACR/fig4_confusion_matrices.png', dpi=150,
            bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.3)
print("Saved fig4_confusion_matrices.png")
plt.close()
