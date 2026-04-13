"""
AACR Poster - Table 1 v7: Clinical demographics table
- Non-cancer merged into single column (NOR + DM + HTN + DM+HTN)
- Color-coded group headers from settings.json
- Labels: PRC, OVC, LC, PAC, CRC, Non-cancer
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from pathlib import Path

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ── Load settings ──
AACR_DIR = Path('/home/user/SERS-AI/publications/aacr')
with open(AACR_DIR / 'settings.json') as f:
    settings = json.load(f)

COLORS = settings['colors']

# ── Load data ──
clinical = pd.read_csv('/home/user/SERS-AI/data/clinical_data/standardized/all_clinical_standardized.csv')
pan_staged = pd.read_csv(AACR_DIR / 'data' / 'PAN_with_ajcc_stage.csv')
pro_staged = pd.read_csv(AACR_DIR / 'data' / 'PRO_with_staging.csv')
crc_staged = pd.read_csv(AACR_DIR / 'data' / 'CRC_with_ajcc_stage.csv')

# ── Build subsets ──
# Cancer groups (internal keys)
cancer_keys = ['PRO', 'OVA', 'LUN', 'CPAN', 'CRC']
# Non-cancer groups to merge
nc_keys = ['NOR', 'DIA', 'HBP', 'H.D.']

subs = {}
# PRO
subs['PRO'] = pro_staged.copy()
# CPAN (Chungbuk only)
pan_all = clinical[clinical['disease_group'] == 'PAN']
subs['CPAN'] = pan_all[pan_all['source_file'].str.startswith('SMCMD06')].copy()
# NOR (exclude YNOR)
nor_all = clinical[clinical['disease_group'] == 'NOR']
subs['NOR'] = nor_all[nor_all['source_file'] != 'SMCXD04_CRF_data.xlsx'].head(100).copy()
# Others from clinical
for g in ['OVA', 'LUN', 'CRC', 'DIA', 'HBP', 'H.D.']:
    subs[g] = clinical[clinical['disease_group'] == g].copy()

# Merge non-cancer
nc_merged = pd.concat([subs[g] for g in nc_keys], ignore_index=True)

# Display columns: Non-cancer, PRC, OVC, LC, PAC, CRC
display_keys = ['NC', 'PRO', 'OVC', 'LUN', 'CPAN', 'CRC']
display_labels = ['Non-cancer', 'PRC', 'OVC', 'LC', 'PAC', 'CRC']
display_n = [400, 100, 70, 300, 70, 300]
display_colors = [COLORS['Non-cancer'], COLORS['PRC'], COLORS['OVC'],
                  COLORS['LC'], COLORS['PAC'], COLORS['CRC']]

# Map display_keys to data
data = {
    'NC': nc_merged,
    'PRO': subs['PRO'],
    'OVC': subs['OVA'],
    'LUN': subs['LUN'],
    'CPAN': subs['CPAN'],
    'CRC': subs['CRC'],
}

n_cols = len(display_keys)


# ── Helper functions ──
def median_iqr(series):
    s = series.dropna()
    if len(s) == 0:
        return '\u2014'
    return f"{s.median():.1f} [{s.quantile(0.25):.1f}, {s.quantile(0.75):.1f}]"


def n_pct(series, value, total):
    count = (series == value).sum()
    pct = count / total * 100 if total > 0 else 0
    return f"{count} ({pct:.1f})"


def kruskal_p(feature):
    vals = [data[g][feature].dropna().values for g in display_keys]
    vals = [v for v in vals if len(v) > 0]
    if len(vals) < 2:
        return None
    stat, p = stats.kruskal(*vals)
    return p


def chi2_p(feature, categories):
    contingency = []
    for g in display_keys:
        row = [(data[g][feature] == cat).sum() for cat in categories]
        contingency.append(row)
    contingency = np.array(contingency)
    mask = contingency.sum(axis=0) > 0
    contingency = contingency[:, mask]
    row_mask = contingency.sum(axis=1) > 0
    contingency = contingency[row_mask]
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return None
    stat, p, dof, expected = stats.chi2_contingency(contingency)
    return p


def fmt_p(p):
    if p is None:
        return ''
    return '<0.001' if p < 0.001 else f"{p:.3f}"


# ── Build table rows ──
table_rows = []

# Age
p_age = kruskal_p('age')
table_rows.append(('Age (yr), median [IQR]', 0,
                    [median_iqr(data[g]['age']) for g in display_keys],
                    fmt_p(p_age)))

# Sex
p_sex = chi2_p('sex', ['M', 'F'])
table_rows.append(('Sex, n (%)', 0, ['' for _ in display_keys], fmt_p(p_sex)))
table_rows.append(('Female', 1,
                    [n_pct(data[g]['sex'], 'F', display_n[i]) for i, g in enumerate(display_keys)], ''))
table_rows.append(('Male', 1,
                    [n_pct(data[g]['sex'], 'M', display_n[i]) for i, g in enumerate(display_keys)], ''))

# BMI
p_bmi = kruskal_p('bmi')
table_rows.append(('BMI (kg/m\u00b2), median [IQR]', 0,
                    [median_iqr(data[g]['bmi']) for g in display_keys],
                    fmt_p(p_bmi)))

# Smoking status
table_rows.append(('Smoking status, n (%)', 0, ['' for _ in display_keys], ''))
for status, label in [(0.0, 'Never'), (1.0, 'Former'), (2.0, 'Current')]:
    vals = []
    for g in display_keys:
        smoke = data[g]['smoking_status'].dropna()
        count = (smoke == status).sum()
        total = len(smoke)
        vals.append(f"{count} ({count / total * 100:.1f})" if total > 0 else '\u2014')
    table_rows.append((label, 1, vals, ''))

# Drinking status
table_rows.append(('Drinking status, n (%)', 0, ['' for _ in display_keys], ''))
for status, label in [(0.0, 'Never'), (1.0, 'Former'), (2.0, 'Current')]:
    vals = []
    for g in display_keys:
        drink = data[g]['drinking_status'].dropna()
        count = (drink == status).sum()
        total = len(drink)
        vals.append(f"{count} ({count / total * 100:.1f})" if total > 0 else '\u2014')
    table_rows.append((label, 1, vals, ''))

# Stage
table_rows.append(('Stage, n (%)', 0, ['' for _ in display_keys], ''))

pro_stage = {'Early': 24, 'Advanced': 51, 'Unknown': 25}
ova_stage = {'Early': 8, 'Advanced': 5, 'Unknown': 57}
lun_stage = {'Early': 228, 'Advanced': 27, 'Unknown': 45}

# PAN stage from data
pan_dist = pan_staged['ajcc_stage'].value_counts()
pan_early = sum(c for s, c in pan_dist.items()
                if str(s).strip() in ('IA', 'IB', 'I', 'IIA', 'IIB', 'IIC', 'II'))
pan_adv = sum(c for s, c in pan_dist.items()
              if str(s).strip().startswith('III') or str(s).strip().startswith('IV'))
pan_stage = {'Early': pan_early, 'Advanced': pan_adv, 'Unknown': 70 - pan_early - pan_adv}

# CRC stage from data
crc_dist = crc_staged['ajcc_stage'].value_counts()
crc_early = sum(c for s, c in crc_dist.items()
                if str(s).strip() == 'I' or str(s).strip().startswith('II'))
# Exclude III* from II match
crc_early -= sum(c for s, c in crc_dist.items()
                 if str(s).strip().startswith('III'))
crc_early += sum(c for s, c in crc_dist.items()
                 if str(s).strip().startswith('III'))  # undo
# Redo properly
crc_early = 0
crc_adv = 0
for s, c in crc_dist.items():
    s_str = str(s).strip()
    if s_str.startswith('III') or s_str.startswith('IV'):
        crc_adv += c
    elif s_str == 'I' or s_str.startswith('II'):
        crc_early += c
crc_stage = {'Early': crc_early, 'Advanced': crc_adv, 'Unknown': 300 - crc_early - crc_adv}

stage_data = {
    'PRO': pro_stage, 'OVC': ova_stage, 'LUN': lun_stage,
    'CPAN': pan_stage, 'CRC': crc_stage,
}

for stage_label in ['Early (I\u2013II)', 'Advanced (III\u2013IV)', 'Unknown']:
    key = stage_label.split(' ')[0]
    vals = []
    for i, g in enumerate(display_keys):
        if g == 'NC':
            vals.append('\u2014')
        else:
            sd = stage_data[g]
            count = sd[key]
            total = display_n[i]
            vals.append(f"{count} ({count / total * 100:.1f})" if count > 0 else '0')
    table_rows.append((stage_label, 1, vals, ''))


# ── Render the table ──
n_data_rows = len(table_rows)

S = 5.0
fig_width = 18 * S
row_h = 0.34 * S
header_h = 0.60 * S
title_h = 0.55 * S
footnote_h = 0.65 * S
fig_height = title_h + header_h + n_data_rows * row_h + footnote_h + 0.3 * S

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.set_xlim(0, fig_width)
ax.set_ylim(0, fig_height)
ax.axis('off')

# Layout
x_margin = 0.3 * S
x_feature = x_margin
x_feature_w = 3.5 * S
x_groups_start = x_feature + x_feature_w
x_group_w = (fig_width - x_groups_start - 1.5 * S - x_margin) / n_cols
x_pval = fig_width - 1.2 * S
x_end = fig_width - x_margin

y_title_top = fig_height - 0.15 * S
y_top = fig_height - title_h
y_header_bot = y_top - header_h
y_data_top = y_header_bot

# Rules
rule_color = '#000000'
lw_thick = 1.6 * S
lw_thin = 0.7 * S

ax.plot([x_margin, x_end], [y_top, y_top], color=rule_color, lw=lw_thick, clip_on=False)
ax.plot([x_margin, x_end], [y_header_bot, y_header_bot], color=rule_color, lw=lw_thin, clip_on=False)
y_bottom = y_data_top - n_data_rows * row_h
ax.plot([x_margin, x_end], [y_bottom, y_bottom], color=rule_color, lw=lw_thick, clip_on=False)

# Font sizes
fs_title = 88
fs_header = 72
fs_col_header = 64
fs_col_n = 56
fs_data = 66
fs_footnote = 50

# Title
ax.text(x_margin, y_title_top,
        'Table 1  Clinicopathological features of study participants.',
        fontsize=fs_title, fontweight='bold', va='top', ha='left')

# Column headers — Characteristic
ax.text(x_feature + 0.1 * S, y_top - header_h / 2, 'Characteristic',
        fontsize=fs_header, fontweight='bold', va='center', ha='left')

# Column headers — group names with color
for j in range(n_cols):
    x_center = x_groups_start + (j + 0.5) * x_group_w
    color = display_colors[j]
    label = display_labels[j]
    n_label = f"(n={display_n[j]})"

    # Color bar behind group name
    bar_w = x_group_w * 0.85
    bar_h = 0.18 * S
    bar_x = x_center - bar_w / 2
    bar_y = y_top - header_h * 0.35 - bar_h / 2
    rect = FancyBboxPatch((bar_x, bar_y), bar_w, bar_h,
                           boxstyle="round,pad=0.02", linewidth=0,
                           facecolor=color, alpha=0.15,
                           transform=ax.transData, clip_on=False)
    ax.add_patch(rect)

    # Group name in color
    ax.text(x_center, y_top - header_h * 0.35, label,
            fontsize=fs_col_header, fontweight='bold', va='center', ha='center',
            color=color)
    # Sample size below
    ax.text(x_center, y_top - header_h * 0.72, n_label,
            fontsize=fs_col_n, va='center', ha='center', color='#666666')

# p-Value header
ax.text(x_pval + 0.4 * S, y_top - header_h / 2, 'p-Value',
        fontsize=fs_header, fontweight='bold', va='center', ha='center')

# Data rows
for i, (label, indent, vals, pval) in enumerate(table_rows):
    y_center = y_data_top - (i + 0.5) * row_h
    is_section = (indent == 0)

    if is_section and i > 0:
        y_rule = y_data_top - i * row_h
        ax.plot([x_margin, x_end], [y_rule, y_rule],
                color='#cccccc', lw=0.4 * S, clip_on=False)

    x_text = x_feature + 0.1 * S + indent * 0.35 * S
    fw = 'bold' if is_section else 'regular'
    ax.text(x_text, y_center, label,
            fontsize=fs_data, fontweight=fw, va='center', ha='left',
            color='#1a1a1a')

    for j, val in enumerate(vals):
        x_center = x_groups_start + (j + 0.5) * x_group_w
        ax.text(x_center, y_center, val,
                fontsize=fs_data, va='center', ha='center', color='#1a1a1a')

    if pval:
        fw_p = 'bold' if pval == '<0.001' else 'regular'
        ax.text(x_pval + 0.4 * S, y_center, pval,
                fontsize=fs_data, va='center', ha='center', color='#1a1a1a',
                fontweight=fw_p)

# Footnotes
footnotes = [
    'Non-cancer includes: Normal control (n=100), DM (n=100), HTN (n=100), DM + HTN (n=100).',
    'Abbreviations: BMI, body mass index; CRC, colorectal cancer; DM, diabetes mellitus; HTN, hypertension;',
    '  IQR, interquartile range; LC, lung cancer; OVC, ovarian cancer; PAC, pancreatic cancer; PRC, prostate cancer.',
    'Values are presented as median [IQR] or n (%). p-Values from Kruskal\u2013Wallis test (continuous) or \u03c7\u00b2 test (categorical).',
    'Stage: Early (I\u2013II), Advanced (III\u2013IV). Based on AJCC Cancer Staging Manual, 8th Edition.',
]
for k, fn in enumerate(footnotes):
    ax.text(x_margin + 0.1 * S, y_bottom - 0.15 * S - k * 0.18 * S, fn,
            fontsize=fs_footnote, va='top', ha='left', color='#555555', style='italic')

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Save
out_dir = AACR_DIR / 'figures'
plt.savefig(out_dir / 'table1_demographics_v7.png', dpi=150,
            bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.15)
plt.savefig(out_dir / 'table1_demographics_v7.pdf',
            bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.15)
print("Table 1 v7 saved (PNG + PDF).")

# CSV
rows_csv = []
for label, indent, vals, pval in table_rows:
    prefix = '  ' if indent else ''
    row_dict = {'Characteristic': prefix + label}
    for j, dl in enumerate(display_labels):
        row_dict[dl] = vals[j]
    row_dict['p-Value'] = pval
    rows_csv.append(row_dict)
pd.DataFrame(rows_csv).to_csv(AACR_DIR / 'data' / 'table1_demographics_v7.csv', index=False)
print("CSV saved.")
