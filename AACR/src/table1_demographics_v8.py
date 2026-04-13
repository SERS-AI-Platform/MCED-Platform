"""
AACR Poster - Table 1 v8: Clinical demographics (Portrait)
- No title, Sex merged to F/M single column
- Footnote: 1x4 cards, left-aligned, full width
- Clean modern design
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
AACR_DIR = Path('/home/user/SERS-AI/AACR')
with open(AACR_DIR / 'settings.json') as f:
    settings = json.load(f)
COLORS = settings['colors']
STYLE = settings['table_style']

# ── Load data ──
clinical = pd.read_csv('/home/user/SERS-AI/data/clinical_data/standardized/all_clinical_standardized.csv')

# ── Build subsets ──
nc_keys = ['NOR', 'DIA', 'HBP', 'H.D.']
subs = {}
subs['PRO'] = clinical[clinical['disease_group'] == 'PRO'].copy()
pan_all = clinical[clinical['disease_group'] == 'PAN']
subs['CPAN'] = pan_all[pan_all['source_file'].str.startswith('SMCMD06')].copy()
nor_all = clinical[clinical['disease_group'] == 'NOR']
subs['NOR'] = nor_all[nor_all['source_file'] != 'SMCXD04_CRF_data.xlsx'].head(100).copy()
for g in ['OVA', 'LUN', 'CRC', 'DIA', 'HBP', 'H.D.']:
    subs[g] = clinical[clinical['disease_group'] == g].copy()
nc_merged = pd.concat([subs[g] for g in nc_keys], ignore_index=True)

display_keys = ['NC', 'PRO', 'OVC', 'LUN', 'CPAN', 'CRC']
display_labels = ['Non-cancer', 'PRC', 'OVC', 'LC', 'PAC', 'CRC']
display_n = [400, 100, 70, 300, 70, 300]
display_colors = [COLORS['Non-cancer'], COLORS['PRC'], COLORS['OVC'],
                  COLORS['LC'], COLORS['PAC'], COLORS['CRC']]

data_dict = {
    'NC': nc_merged, 'PRO': subs['PRO'], 'OVC': subs['OVA'],
    'LUN': subs['LUN'], 'CPAN': subs['CPAN'], 'CRC': subs['CRC'],
}
n_groups = len(display_keys)

# ── Helpers ──
def median_iqr(series):
    s = series.dropna()
    if len(s) == 0:
        return '\u2014'
    return f"{s.median():.0f} [{s.quantile(0.25):.0f}\u2013{s.quantile(0.75):.0f}]"

def sex_ratio(df, total):
    f = (df['sex'] == 'F').sum()
    m = (df['sex'] == 'M').sum()
    return f"{f}:{m}\n({f/total*100:.0f}/{m/total*100:.0f}%)"

def habit_dist(df, col):
    """0=never, 1=former, 2=current, NaN=unknown → 'nv/fmr/cur/unk %'"""
    s = df[col]
    n = len(s)
    if n == 0:
        return '\u2014'
    nv = (s == 0).sum()
    fmr = (s == 1).sum()
    cur = (s == 2).sum()
    unk = s.isna().sum()
    return f"{nv/n*100:.0f}/{fmr/n*100:.0f}/\n{cur/n*100:.0f}/{unk/n*100:.0f}%"

# ── Build row data ──
# Columns: Group | n | Age | Sex (F:M) | BMI | Smoking | Drinking
rows_data = []
for i, (gk, gl, gn, gc) in enumerate(zip(display_keys, display_labels, display_n, display_colors)):
    df = data_dict[gk]
    age_val = median_iqr(df['age'])
    sex_val = sex_ratio(df, gn)
    bmi_val = median_iqr(df['bmi'])
    rows_data.append({
        'label': gl, 'n': str(gn), 'color': gc,
        'vals': [age_val, sex_val, bmi_val]
    })

# p-values
p_age = stats.kruskal(*[data_dict[g]['age'].dropna().values for g in display_keys
                         if len(data_dict[g]['age'].dropna()) > 0])
p_sex_cont = np.array([[(data_dict[g]['sex'] == c).sum() for c in ['M', 'F']] for g in display_keys])
p_sex_r = stats.chi2_contingency(p_sex_cont)
p_bmi = stats.kruskal(*[data_dict[g]['bmi'].dropna().values for g in display_keys
                          if len(data_dict[g]['bmi'].dropna()) > 0])


def fmt_p(p):
    return '<0.001' if p < 0.001 else f"{p:.3f}"

# ── Layout ──
S = 20.0
n_table_cols = 5  # Group, n, Age, Sex, BMI
row_h = 1.50 * S
header_h = 0.80 * S
pval_row_h = 0.60 * S
footnote_h = 1.80 * S
top_pad = 0.15 * S

fig_width = 18 * S
fig_height = top_pad + header_h + n_groups * row_h + pval_row_h + footnote_h + 0.2 * S

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.set_xlim(0, fig_width)
ax.set_ylim(0, fig_height)
ax.axis('off')

x_margin = 0.25 * S
x_end = fig_width - x_margin

# Column widths: Group, n, Age, Sex(F:M), BMI
col_w_rel = [1.6, 0.7, 1.8, 1.6, 1.8]
total_rel = sum(col_w_rel)
avail_w = x_end - x_margin
col_widths = [w / total_rel * avail_w for w in col_w_rel]
col_x_left = [x_margin]
for w in col_widths[:-1]:
    col_x_left.append(col_x_left[-1] + w)
col_x_center = [xl + w / 2 for xl, w in zip(col_x_left, col_widths)]

# No shaded columns
stage_col_indices = []

# Y positions (no title)
y_top = fig_height - top_pad
y_header_bot = y_top - header_h
y_data_top = y_header_bot
y_data_bot = y_data_top - n_groups * row_h
y_pval_bot = y_data_bot - pval_row_h

# ── Font sizes ──
fs_header = 540
fs_data = 500
fs_group = 560
fs_pval_label = 440

# ── Colors (from config) ──
stage_shade = STYLE['stage_shade']
alt_row_color = STYLE['alt_row_color']
accent_line = STYLE['accent_line']

# ── Stage shading (full height) ──
for ci in stage_col_indices:
    rect = plt.Rectangle(
        (col_x_left[ci], y_pval_bot), col_widths[ci], y_top - y_pval_bot,
        facecolor=stage_shade, edgecolor='none', zorder=0)
    ax.add_patch(rect)

# ── Rules ──
lw_thick = 1.5 * S
lw_thin = 0.4 * S

# Top thick rule
ax.plot([x_margin, x_end], [y_top, y_top], color=accent_line, lw=lw_thick, clip_on=False)
# Under header
ax.plot([x_margin, x_end], [y_header_bot, y_header_bot], color=accent_line, lw=lw_thin, clip_on=False)
# Above p-value
ax.plot([x_margin, x_end], [y_data_bot, y_data_bot], color='#CCCCCC', lw=lw_thin, clip_on=False)
# Bottom thick rule
ax.plot([x_margin, x_end], [y_pval_bot, y_pval_bot], color=accent_line, lw=lw_thick, clip_on=False)

# ── Header row ──
col_labels = ['Group', 'n', 'Age', 'Sex (F:M)', 'BMI']
for ci, lbl in enumerate(col_labels):
    color = accent_line
    ax.text(col_x_center[ci], y_top - header_h * 0.5, lbl,
            fontsize=fs_header, fontweight='bold', va='center', ha='center',
            color=color)

# ── Data rows ──
for i, rd in enumerate(rows_data):
    y_center = y_data_top - (i + 0.5) * row_h

    # Alternating row bg (non-stage columns)
    if i % 2 == 1:
        for ci in range(len(col_widths)):
            if ci not in stage_col_indices:
                rect = plt.Rectangle(
                    (col_x_left[ci], y_data_top - (i + 1) * row_h),
                    col_widths[ci], row_h,
                    facecolor=alt_row_color, edgecolor='none', zorder=0)
                ax.add_patch(rect)

    # Subtle row divider
    if i > 0:
        y_div = y_data_top - i * row_h
        ax.plot([x_margin, x_end], [y_div, y_div],
                color=STYLE['divider'], lw=0.2 * S, clip_on=False)

    # Group name
    ax.text(col_x_center[0], y_center, rd['label'],
            fontsize=fs_group, fontweight='bold', va='center', ha='center',
            color=rd['color'])
    # n
    ax.text(col_x_center[1], y_center, rd['n'],
            fontsize=fs_data, va='center', ha='center', color=STYLE['n_text'])
    # Values (Age=0, Sex=1, BMI=2, Early=3, Adv=4, Unk=5)
    for j, val in enumerate(rd['vals']):
        ax.text(col_x_center[j + 2], y_center, val,
                fontsize=fs_data, va='center', ha='center',
                color=STYLE['data_text'], linespacing=1.0)

# ── p-Value row ──
y_pval_center = y_data_bot - pval_row_h * 0.5
ax.text(col_x_center[0], y_pval_center, 'p',
        fontsize=fs_pval_label, fontweight='bold', va='center', ha='center',
        color=STYLE['pval_text'], style='italic')
pval_map = {2: fmt_p(p_age.pvalue), 3: fmt_p(p_sex_r[1]), 4: fmt_p(p_bmi.pvalue)}
for ci, pv in pval_map.items():
    fw = 'bold' if pv == '<0.001' else 'regular'
    ax.text(col_x_center[ci], y_pval_center, pv,
            fontsize=fs_data, fontweight=fw, va='center', ha='center', color=STYLE['data_text'])

# ── Footnote: 1 row x 4 cards ──
fs_fn_title = 280
fs_fn_body = 220
fn_gap = 0.08 * S
fn_top = y_pval_bot - fn_gap
fn_bot = 0.1 * S
fn_h = fn_top - fn_bot
fn_card_gap = 0.06 * S
fn_total_w = x_end - x_margin - 3 * fn_card_gap
fn_card_w = fn_total_w / 4
fn_inner_pad = 0.15 * S

card_accents = STYLE['card_accents']

fn_data = [
    ('Study Design',
     'Non-cancer: Normal, DM,\n'
     'HTN, Heart Dz (n=100 each)\n'
     'Values: median [IQR] or %\n'
     'p: Kruskal\u2013Wallis / \u03c7\u00b2 test'),
    ('Smoking',
     'Coded as 0/1/2 \u2192\n'
     'never / former / current\n'
     'unk = missing in source CRF\n'
     'Computed from clinical CSV'),
    ('Drinking',
     'Coded as 0/1/2 \u2192\n'
     'never / former / current\n'
     'unk = missing in source CRF\n'
     'Computed from clinical CSV'),
    ('Notes',
     'PAC, CRC have no smoking/\n'
     'drinking records (\u2192 100% unk)\n'
     'OVC, LC partially recorded\n'
     '\u03c7\u00b2 includes unk category'),
]

for i, (title, body) in enumerate(fn_data):
    x0 = x_margin + i * (fn_card_w + fn_card_gap)

    # Card background with rounded corners
    card = FancyBboxPatch(
        (x0, fn_bot), fn_card_w, fn_h,
        boxstyle="round,pad=0.02",
        facecolor=STYLE['footnote_bg'], edgecolor=STYLE['footnote_border'],
        linewidth=0.15 * S, zorder=1)
    ax.add_patch(card)

    # Accent top bar
    bar_h = 0.06 * S
    ax.add_patch(plt.Rectangle(
        (x0 + 0.01 * S, fn_bot + fn_h - bar_h - 0.01 * S),
        fn_card_w - 0.02 * S, bar_h,
        facecolor=card_accents[i], edgecolor='none', zorder=2))

    # Title (left-aligned)
    ax.text(x0 + fn_inner_pad, fn_bot + fn_h - bar_h - 0.08 * S, title,
            fontsize=fs_fn_title, fontweight='bold', va='top', ha='left',
            color=card_accents[i], zorder=3)

    # Body (left-aligned)
    ax.text(x0 + fn_inner_pad, fn_bot + fn_h - bar_h - 0.38 * S, body,
            fontsize=fs_fn_body, va='top', ha='left',
            linespacing=1.2, zorder=3, color=STYLE['footnote_body'])

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Save
plt.savefig(AACR_DIR / 'figures' / 'table1_demographics_v8.png', dpi=36,
            bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.15)
plt.savefig(AACR_DIR / 'figures' / 'table1_demographics_v8.pdf',
            bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.15)
print("Table 1 v8 saved (PNG + PDF).")

# CSV
rows_csv = []
for rd in rows_data:
    row_dict = {'Group': rd['label'], 'n': rd['n']}
    csv_cols = ['Age (yr)', 'Sex (F:M)', 'BMI (kg/m²)']
    for j, cc in enumerate(csv_cols):
        row_dict[cc] = rd['vals'][j].replace('\n', ' ')
    rows_csv.append(row_dict)
pd.DataFrame(rows_csv).to_csv(AACR_DIR / 'data' / 'table1_demographics_v8.csv', index=False)
print("CSV saved.")
