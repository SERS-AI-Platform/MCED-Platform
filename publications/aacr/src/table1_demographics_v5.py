"""
AACR Poster - Table 1 v5: Clinical demographics table
Items: Age, Sex, BMI, Smoking status, Drinking status, Stage (Early/Advanced/Unknown)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# Load data
clinical = pd.read_csv('/home/user/SERS-AI/data/clinical_data/standardized/all_clinical_standardized.csv')
pan_staged = pd.read_csv('/home/user/SERS-AI/AACR/PAN_with_ajcc_stage.csv')
pro_staged = pd.read_csv('/home/user/SERS-AI/AACR/PRO_with_staging.csv')
crc_staged = pd.read_csv('/home/user/SERS-AI/AACR/CRC_with_ajcc_stage.csv')

groups = ['NOR', 'PRO', 'OVA', 'LUN', 'PAN', 'CRC']
group_totals = {'NOR': 100, 'PRO': 100, 'OVA': 70, 'LUN': 300, 'PAN': 70, 'CRC': 300}
col_labels = ['NOR\n(n=100)', 'PRC\n(n=100)', 'OVC\n(n=70)',
              'LC\n(n=300)', 'PAC\n(n=70)', 'CRC\n(n=300)']

# Get subsets — PRO from pro_staged, PAN from pan_staged, others from clinical
subs = {}
for g in groups:
    if g == 'PRO':
        subs[g] = pro_staged.copy()
    elif g == 'PAN':
        subs[g] = pan_staged.copy()
    else:
        sub = clinical[clinical['disease_group'] == g].copy()
        if g == 'NOR' and len(sub) > 100:
            sub = sub.head(100)
        subs[g] = sub


def median_iqr(series):
    s = series.dropna()
    if len(s) == 0:
        return '\u2014'
    med = s.median()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    return f"{med:.1f} [{q1:.1f}, {q3:.1f}]"


def n_pct(series, value, total):
    count = (series == value).sum()
    pct = count / total * 100 if total > 0 else 0
    return f"{count} ({pct:.1f})"


def kruskal_p(feature):
    vals = [subs[g][feature].dropna().values for g in groups]
    vals = [v for v in vals if len(v) > 0]
    if len(vals) < 2:
        return None
    stat, p = stats.kruskal(*vals)
    return p


def chi2_p(feature, categories):
    contingency = []
    for g in groups:
        row = []
        for cat in categories:
            row.append((subs[g][feature] == cat).sum())
        contingency.append(row)
    contingency = np.array(contingency)
    mask = contingency.sum(axis=0) > 0
    contingency = contingency[:, mask]
    if contingency.shape[1] < 2:
        return None
    stat, p, dof, expected = stats.chi2_contingency(contingency)
    return p


def fmt_p(p):
    if p is None:
        return ''
    if p < 0.001:
        return '<0.001'
    return f"{p:.3f}"


# ── Build table rows ──
table_rows = []

# Age
p_age = kruskal_p('age')
table_rows.append(('Age (yr), median [IQR]', 0,
                    [median_iqr(subs[g]['age']) for g in groups],
                    fmt_p(p_age)))

# Sex
p_sex = chi2_p('sex', ['M', 'F'])
table_rows.append(('Sex, n (%)', 0,
                    ['' for _ in groups], fmt_p(p_sex)))
table_rows.append(('Female', 1,
                    [n_pct(subs[g]['sex'], 'F', len(subs[g])) for g in groups], ''))
table_rows.append(('Male', 1,
                    [n_pct(subs[g]['sex'], 'M', len(subs[g])) for g in groups], ''))

# BMI
p_bmi = kruskal_p('bmi')
table_rows.append(('BMI (kg/m\u00b2), median [IQR]', 0,
                    [median_iqr(subs[g]['bmi']) for g in groups],
                    fmt_p(p_bmi)))

# Smoking status
table_rows.append(('Smoking status, n (%)', 0,
                    ['' for _ in groups], ''))
for status, label in [(0.0, 'Never'), (1.0, 'Former'), (2.0, 'Current')]:
    vals = []
    for g in groups:
        smoke = subs[g]['smoking_status'].dropna()
        count = (smoke == status).sum()
        total = len(smoke)
        if total > 0:
            vals.append(f"{count} ({count / total * 100:.1f})")
        else:
            vals.append('\u2014')
    table_rows.append((label, 1, vals, ''))

# Drinking status
table_rows.append(('Drinking status, n (%)', 0,
                    ['' for _ in groups], ''))
for status, label in [(0.0, 'Never'), (1.0, 'Former'), (2.0, 'Current')]:
    vals = []
    for g in groups:
        drink = subs[g]['drinking_status'].dropna()
        count = (drink == status).sum()
        total = len(drink)
        if total > 0:
            vals.append(f"{count} ({count / total * 100:.1f})")
        else:
            vals.append('\u2014')
    table_rows.append((label, 1, vals, ''))

# ── Stage: Early (I+II) / Advanced (III+IV) / Unknown ──
table_rows.append(('Stage, n (%)', 0,
                    ['' for _ in groups], ''))

# Hardcoded stage data (user-provided)
# PRO: I=4, II=20, III=20, IV=31, UNK=25
# OVA: I=8, II=0, III=4, IV=1, UNK=57
# LUN: Early=228, Advanced=27, UNK=45
pro_stage = {'Early': 4 + 20, 'Advanced': 20 + 31, 'Unknown': 25}
ova_stage = {'Early': 8 + 0, 'Advanced': 4 + 1, 'Unknown': 57}
lun_stage = {'Early': 228, 'Advanced': 27, 'Unknown': 45}

# Compute PAN: Early/Advanced from ajcc_stage
pan_dist = pan_staged['ajcc_stage'].value_counts()
pan_early = 0
pan_adv = 0
for s, c in pan_dist.items():
    s_str = str(s).strip()
    if s_str.startswith('III') or s_str.startswith('IV'):
        pan_adv += c
    elif s_str in ('IA', 'IB', 'I', 'IIA', 'IIB', 'IIC', 'II'):
        pan_early += c
pan_stage = {'Early': pan_early, 'Advanced': pan_adv,
             'Unknown': group_totals['PAN'] - pan_early - pan_adv}

# Compute CRC: Early/Advanced from ajcc_stage
crc_dist = crc_staged['ajcc_stage'].value_counts()
crc_early = 0
crc_adv = 0
for s, c in crc_dist.items():
    s_str = str(s).strip()
    # Check III/IV first to avoid 'IIIA'.startswith('II') false match
    if s_str.startswith('III') or s_str.startswith('IV'):
        crc_adv += c
    elif s_str == 'I' or s_str.startswith('II'):
        crc_early += c
crc_stage = {'Early': crc_early, 'Advanced': crc_adv,
             'Unknown': group_totals['CRC'] - crc_early - crc_adv}

stage_data = {
    'PRO': pro_stage,
    'OVA': ova_stage,
    'LUN': lun_stage,
    'PAN': pan_stage,
    'CRC': crc_stage,
}

for stage_label in ['Early (I\u2013II)', 'Advanced (III\u2013IV)', 'Unknown']:
    key = stage_label.split(' ')[0]  # 'Early', 'Advanced', 'Unknown'
    vals = []
    for g in groups:
        if g == 'NOR':
            vals.append('\u2014')
        else:
            sd = stage_data[g]
            count = sd[key]
            total = group_totals[g]
            if count > 0:
                vals.append(f"{count} ({count / total * 100:.1f})")
            else:
                vals.append('0')
    table_rows.append((stage_label, 1, vals, ''))


# ── Render the table (4x font) ──
n_data_rows = len(table_rows)
n_group_cols = 6

S = 5.0  # scale factor for dimensions

fig_width = 18 * S
row_h = 0.34 * S
header_h = 0.55 * S
title_h = 0.55 * S
footnote_h = 0.5 * S
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
x_group_w = (fig_width - x_groups_start - 1.5 * S - x_margin) / n_group_cols
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

# Font sizes (8x original)
fs_title = 88
fs_header = 72
fs_col_header = 68
fs_data = 66
fs_footnote = 56

# Title
ax.text(x_margin, y_title_top,
        'Table 1  Clinicopathological features of study participants.',
        fontsize=fs_title, fontweight='bold', va='top', ha='left')

# Column headers
ax.text(x_feature + 0.1 * S, y_top - header_h / 2, 'Characteristic',
        fontsize=fs_header, fontweight='bold', va='center', ha='left')

for j, label in enumerate(col_labels):
    x_center = x_groups_start + (j + 0.5) * x_group_w
    ax.text(x_center, y_top - header_h / 2, label,
            fontsize=fs_col_header, fontweight='bold', va='center', ha='center',
            linespacing=1.3)

ax.text(x_pval + 0.4 * S, y_top - header_h / 2, 'p-Value',
        fontsize=fs_header, fontweight='bold', va='center', ha='center')

# Data rows
for i, (label, indent, vals, pval) in enumerate(table_rows):
    y_center = y_data_top - (i + 0.5) * row_h
    is_header = (indent == 0 and all(v == '' for v in vals))
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
    'Abbreviations: BMI, body mass index; IQR, interquartile range; yr, years.',
    'Values are presented as median [IQR] or n (%). p-Values from Kruskal\u2013Wallis test (continuous) or \u03c7\u00b2 test (categorical).',
    'Stage: Early (I\u2013II), Advanced (III\u2013IV). Based on AJCC Cancer Staging Manual, 8th Edition.',
]
for k, fn in enumerate(footnotes):
    ax.text(x_margin + 0.1 * S, y_bottom - 0.18 * S - k * 0.20 * S, fn,
            fontsize=fs_footnote, va='top', ha='left', color='#555555', style='italic')

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('/home/user/SERS-AI/AACR/table1_demographics_v5.png', dpi=150,
            bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.15)
print("Table 1 v5 saved (PNG).")

# Also save CSV
rows_csv = []
for label, indent, vals, pval in table_rows:
    prefix = '  ' if indent else ''
    row_dict = {'Characteristic': prefix + label}
    for j, g_label in enumerate(['NOR', 'PRC', 'OVC', 'LC', 'PAC', 'CRC']):
        row_dict[g_label] = vals[j]
    row_dict['p-Value'] = pval
    rows_csv.append(row_dict)
pd.DataFrame(rows_csv).to_csv('/home/user/SERS-AI/AACR/table1_demographics_v5.csv', index=False)
print("CSV saved.")
