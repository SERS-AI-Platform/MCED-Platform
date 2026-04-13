"""
AACR Poster - Table 1 v6: Clinical demographics table
Groups: NOR, DIA, HBP, H.D., PRC, OVC, LC, PAC, CRC
Labels loaded from settings.json
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from pathlib import Path

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ── Load settings ──
AACR_DIR = Path('/home/user/SERS-AI/publications/aacr')
with open(AACR_DIR / 'settings.json') as f:
    settings = json.load(f)

LABEL_MAP = settings['label_map']
DISPLAY_ORDER = settings['display_order']
GROUP_TOTALS = settings['group_totals']

# Internal group keys (data column values)
internal_groups = ['NOR', 'DIA', 'HBP', 'H.D.', 'PRO', 'OVA', 'LUN', 'CPAN', 'CRC']

# ── Load data ──
clinical = pd.read_csv('/home/user/SERS-AI/data/clinical_data/standardized/all_clinical_standardized.csv')
pan_staged = pd.read_csv(AACR_DIR / 'data' / 'PAN_with_ajcc_stage.csv')
pro_staged = pd.read_csv(AACR_DIR / 'data' / 'PRO_with_staging.csv')
crc_staged = pd.read_csv(AACR_DIR / 'data' / 'CRC_with_ajcc_stage.csv')

# ── Build subsets ──
subs = {}
for g in internal_groups:
    if g == 'PRO':
        subs[g] = pro_staged.copy()
    elif g == 'CPAN':
        # CPAN = PAN from Chungbuk only (SMCMD06 source)
        pan_all = clinical[clinical['disease_group'] == 'PAN']
        subs[g] = pan_all[pan_all['source_file'].str.startswith('SMCMD06')].copy()
    elif g == 'NOR':
        # NOR excluding YNOR (SMCXD04 = Yonsei)
        nor_all = clinical[clinical['disease_group'] == 'NOR']
        subs[g] = nor_all[nor_all['source_file'] != 'SMCXD04_CRF_data.xlsx'].head(100).copy()
    else:
        subs[g] = clinical[clinical['disease_group'] == g].copy()

# Display labels and column headers
col_labels = []
for g in internal_groups:
    display = LABEL_MAP.get(g, g)
    n = GROUP_TOTALS.get(g, len(subs[g]))
    col_labels.append(f"{display}\n(n={n})")

n_group_cols = len(internal_groups)


# ── Helper functions ──
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
    vals = [subs[g][feature].dropna().values for g in internal_groups]
    vals = [v for v in vals if len(v) > 0]
    if len(vals) < 2:
        return None
    stat, p = stats.kruskal(*vals)
    return p


def chi2_p(feature, categories):
    contingency = []
    for g in internal_groups:
        row = []
        for cat in categories:
            row.append((subs[g][feature] == cat).sum())
        contingency.append(row)
    contingency = np.array(contingency)
    mask = contingency.sum(axis=0) > 0
    contingency = contingency[:, mask]
    if contingency.shape[1] < 2:
        return None
    # Remove rows/cols with all zeros to avoid zero expected frequencies
    row_mask = contingency.sum(axis=1) > 0
    contingency = contingency[row_mask]
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
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
                    [median_iqr(subs[g]['age']) for g in internal_groups],
                    fmt_p(p_age)))

# Sex
p_sex = chi2_p('sex', ['M', 'F'])
table_rows.append(('Sex, n (%)', 0,
                    ['' for _ in internal_groups], fmt_p(p_sex)))
table_rows.append(('Female', 1,
                    [n_pct(subs[g]['sex'], 'F', len(subs[g])) for g in internal_groups], ''))
table_rows.append(('Male', 1,
                    [n_pct(subs[g]['sex'], 'M', len(subs[g])) for g in internal_groups], ''))

# BMI
p_bmi = kruskal_p('bmi')
table_rows.append(('BMI (kg/m\u00b2), median [IQR]', 0,
                    [median_iqr(subs[g]['bmi']) for g in internal_groups],
                    fmt_p(p_bmi)))

# Smoking status
table_rows.append(('Smoking status, n (%)', 0,
                    ['' for _ in internal_groups], ''))
for status, label in [(0.0, 'Never'), (1.0, 'Former'), (2.0, 'Current')]:
    vals = []
    for g in internal_groups:
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
                    ['' for _ in internal_groups], ''))
for status, label in [(0.0, 'Never'), (1.0, 'Former'), (2.0, 'Current')]:
    vals = []
    for g in internal_groups:
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
                    ['' for _ in internal_groups], ''))

# Non-cancer groups have no stage
non_cancer = {'NOR', 'DIA', 'HBP', 'H.D.'}

# Hardcoded stage data
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
             'Unknown': GROUP_TOTALS['CPAN'] - pan_early - pan_adv}

# Compute CRC: Early/Advanced from ajcc_stage
crc_dist = crc_staged['ajcc_stage'].value_counts()
crc_early = 0
crc_adv = 0
for s, c in crc_dist.items():
    s_str = str(s).strip()
    if s_str.startswith('III') or s_str.startswith('IV'):
        crc_adv += c
    elif s_str == 'I' or s_str.startswith('II'):
        crc_early += c
crc_stage = {'Early': crc_early, 'Advanced': crc_adv,
             'Unknown': GROUP_TOTALS['CRC'] - crc_early - crc_adv}

stage_data = {
    'PRO': pro_stage,
    'OVA': ova_stage,
    'LUN': lun_stage,
    'CPAN': pan_stage,
    'CRC': crc_stage,
}

for stage_label in ['Early (I\u2013II)', 'Advanced (III\u2013IV)', 'Unknown']:
    key = stage_label.split(' ')[0]  # 'Early', 'Advanced', 'Unknown'
    vals = []
    for g in internal_groups:
        if g in non_cancer:
            vals.append('\u2014')
        else:
            sd = stage_data[g]
            count = sd[key]
            total = GROUP_TOTALS.get(g, len(subs[g]))
            if count > 0:
                vals.append(f"{count} ({count / total * 100:.1f})")
            else:
                vals.append('0')
    table_rows.append((stage_label, 1, vals, ''))


# ── Render the table ──
n_data_rows = len(table_rows)

S = 5.0  # scale factor

fig_width = 24 * S  # wider to accommodate 9 columns
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

# Separator line between non-cancer and cancer columns
x_sep = x_groups_start + 4 * x_group_w  # after NOR, DIA, HBP, H.D.
ax.plot([x_sep, x_sep], [y_top, y_bottom], color='#cccccc', lw=0.5 * S,
        linestyle='--', clip_on=False)

# Font sizes
fs_title = 88
fs_header = 72
fs_col_header = 60
fs_data = 58
fs_footnote = 50

# Title
ax.text(x_margin, y_title_top,
        'Table 1  Clinicopathological features of study participants.',
        fontsize=fs_title, fontweight='bold', va='top', ha='left')

# Sub-headers: "Non-cancer" and "Cancer"
x_nc_center = x_groups_start + 2 * x_group_w
x_c_center = x_groups_start + 6.5 * x_group_w
ax.text(x_nc_center, y_top + 0.12 * S, 'Non-cancer',
        fontsize=fs_col_header, fontweight='bold', va='bottom', ha='center',
        color='#555555', style='italic')
ax.text(x_c_center, y_top + 0.12 * S, 'Cancer',
        fontsize=fs_col_header, fontweight='bold', va='bottom', ha='center',
        color='#555555', style='italic')

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
    'Abbreviations: BMI, body mass index; DIA, diabetes; H.D., hypertension + diabetes; HBP, hypertension; IQR, interquartile range; yr, years.',
    'Values are presented as median [IQR] or n (%). p-Values from Kruskal\u2013Wallis test (continuous) or \u03c7\u00b2 test (categorical).',
    'Stage: Early (I\u2013II), Advanced (III\u2013IV). Based on AJCC Cancer Staging Manual, 8th Edition.',
]
for k, fn in enumerate(footnotes):
    ax.text(x_margin + 0.1 * S, y_bottom - 0.18 * S - k * 0.22 * S, fn,
            fontsize=fs_footnote, va='top', ha='left', color='#555555', style='italic')

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Save outputs
out_dir = AACR_DIR / 'figures'
plt.savefig(out_dir / 'table1_demographics_v6.png', dpi=150,
            bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.15)
plt.savefig(out_dir / 'table1_demographics_v6.pdf',
            bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.15)
print("Table 1 v6 saved (PNG + PDF).")

# CSV with display labels
display_labels = [LABEL_MAP.get(g, g) for g in internal_groups]
rows_csv = []
for label, indent, vals, pval in table_rows:
    prefix = '  ' if indent else ''
    row_dict = {'Characteristic': prefix + label}
    for j, dl in enumerate(display_labels):
        row_dict[dl] = vals[j]
    row_dict['p-Value'] = pval
    rows_csv.append(row_dict)
pd.DataFrame(rows_csv).to_csv(AACR_DIR / 'data' / 'table1_demographics_v6.csv', index=False)
print("CSV saved.")
