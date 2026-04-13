"""
AACR Poster — Mean SERS Spectra with cancer-specific peak regions shaded.
Each cancer type gets its own highlighted band at its most discriminative wavenumber region.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Colors (matched to bar chart palette) ──
colors = {
    'Non-Cancer': '#1b3a5c',
    'PRC': '#e8733a',
    'OVC': '#7b2d8e',
    'LC':  '#d4a088',
    'PAC': '#a01020',
    'CRC': '#2980b9',
}

# ── Load spectra ──
spec = pd.read_csv('/home/user/SERS-AI/results/processed_spectra.csv')

# Map groups
spec['cancer_group'] = spec['group'].map({
    'NOR': 'Non-Cancer', 'DIA': 'Non-Cancer', 'HBP': 'Non-Cancer', 'H.D.': 'Non-Cancer',
    'PRO': 'PRC', 'OVA': 'OVC', 'LUN': 'LC', 'CPAN': 'PAC', 'CRC': 'CRC',
})
spec = spec.dropna(subset=['cancer_group'])

# Get wavenumber columns
wn_cols = [c for c in spec.columns if c.startswith('x_')]
wavenumbers = np.array([float(c.replace('x_', '')) for c in wn_cols])

# Medoid per patient
spec_med = spec.groupby(['cancer_group', 'sample_id']).first().reset_index()

# Compute mean per group
group_order = ['Non-Cancer', 'PRC', 'OVC', 'LC', 'PAC', 'CRC']
means = {}
sems = {}
for g in group_order:
    sub = spec_med[spec_med['cancer_group'] == g][wn_cols].values
    means[g] = sub.mean(axis=0)
    sems[g] = sub.std(axis=0) / np.sqrt(sub.shape[0])

# ── Cancer-specific peak regions (non-overlapping, from discriminative peak analysis) ──
# Each: (label, lo_cm1, hi_cm1, annotation_text)
peak_regions = [
    ('PRC', 2088, 2118, '~2100 cm⁻¹'),
    ('PAC', 710, 745,   '~725 cm⁻¹'),
    ('OVC', 1588, 1620, '~1603 cm⁻¹'),
    ('LC',  975, 1005,  '~990 cm⁻¹'),
    ('CRC', 1330, 1365, '~1350 cm⁻¹'),
]

# ── Plot ──
fig, ax = plt.subplots(figsize=(24, 12))
fig.patch.set_facecolor('white')

# Downsample wavenumbers for cleaner lines (every 3rd)
mask = np.arange(len(wavenumbers)) % 3 == 0
wn_plot = wavenumbers[mask]

# Draw shaded peak regions FIRST (behind spectra)
for cancer, lo, hi, anno_text in peak_regions:
    color = colors[cancer]
    ax.axvspan(lo, hi, alpha=0.15, color=color, zorder=0)
    # Add label at top of shaded region
    mid = (lo + hi) / 2
    ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 6.5, '',
            ha='center', va='bottom', fontsize=18, fontweight='bold',
            color=color)

# Plot spectra
for g in group_order:
    m = means[g][mask]
    s = sems[g][mask]
    ax.fill_between(wn_plot, m - s, m + s, alpha=0.1, color=colors[g])
    ax.plot(wn_plot, m, color=colors[g], linewidth=1.5, alpha=0.9, label=g)

# Now add peak region labels at fixed y position
y_max = max(means[g][mask].max() for g in group_order)
label_y = y_max * 1.05

for cancer, lo, hi, anno_text in peak_regions:
    color = colors[cancer]
    mid = (lo + hi) / 2
    ax.annotate(
        f'{cancer}\n{anno_text}',
        xy=(mid, label_y * 0.95),
        ha='center', va='bottom',
        fontsize=20, fontweight='bold', color=color,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color,
                  alpha=0.9, linewidth=2),
    )
    # Dashed vertical lines at region boundaries
    ax.axvline(lo, color=color, linewidth=1, linestyle='--', alpha=0.4, zorder=0)
    ax.axvline(hi, color=color, linewidth=1, linestyle='--', alpha=0.4, zorder=0)

# Formatting
ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=24, fontweight='bold')
ax.set_ylabel('Intensity (a.u.)', fontsize=24, fontweight='bold')
ax.set_title('Mean SERS Spectra with Cancer-Specific Discriminative Peaks',
             fontsize=28, fontweight='bold', pad=20)
ax.tick_params(axis='both', labelsize=20)
ax.set_xlim(wavenumbers.min(), wavenumbers.max())

# Legend
ax.legend(fontsize=18, loc='upper left', framealpha=0.9, edgecolor='#cccccc',
          ncol=3, handlelength=2.5)

ax.set_ylim(bottom=-0.5, top=label_y * 1.18)
ax.grid(axis='y', alpha=0.2)

plt.tight_layout()
plt.savefig('/home/user/SERS-AI/AACR/fig2b_spectra_peaks.png', dpi=150,
            bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.3)
print("Saved fig2b_spectra_peaks.png")
plt.close()
