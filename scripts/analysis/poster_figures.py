"""
Poster Figure Generation Script
================================
1. Cohort demography (1,628 subjects) — Table + visualizations
2. SPAN demography (122 subjects) — Separate comparison
3. Preprocessed spectra: individual + mean by cancer type
4. CPAN vs YPAN vs SPAN spectrum comparison

Output: results/poster_figures/
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "results" / "poster_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

# Group display config
GROUP_ORDER_CANCER = ["LUN", "CRC", "BLC", "PRO", "OVA", "CPAN", "YPAN", "BRE"]
GROUP_ORDER_NONCANCER = ["NOR", "YNOR", "DIA", "HBP", "H.D."]
GROUP_ORDER_ALL = GROUP_ORDER_CANCER + GROUP_ORDER_NONCANCER

GROUP_LABELS = {
    "LUN": "Lung",
    "CRC": "Colorectal",
    "BLC": "Bladder",
    "PRO": "Prostate",
    "OVA": "Ovarian",
    "CPAN": "Pancreatic\n(CBNU)",
    "YPAN": "Pancreatic\n(Severance)",
    "BRE": "Breast",
    "NOR": "Normal",
    "YNOR": "Normal\n(Severance)",
    "DIA": "Diabetes",
    "HBP": "Hypertension",
    "H.D.": "HTN+DM",
    "SPAN": "Pancreatic\n(Samsung,\npost-op)",
}

CANCER_COLOR = "#E74C3C"
NONCANCER_COLOR = "#3498DB"
SPAN_COLOR = "#F39C12"

GROUP_COLORS = {g: CANCER_COLOR for g in GROUP_ORDER_CANCER}
GROUP_COLORS.update({g: NONCANCER_COLOR for g in GROUP_ORDER_NONCANCER})
GROUP_COLORS["SPAN"] = SPAN_COLOR

# Spectrum colors per pancreatic subgroup
PAN_COLORS = {"CPAN": "#2ECC71", "YPAN": "#3498DB", "SPAN": "#E74C3C"}

SPEC_COLORS = {
    "LUN": "#E74C3C", "CRC": "#E67E22", "BLC": "#F1C40F",
    "PRO": "#2ECC71", "OVA": "#1ABC9C", "CPAN": "#3498DB",
    "YPAN": "#9B59B6", "BRE": "#E91E63",
    "NOR": "#95A5A6", "YNOR": "#BDC3C7",
    "DIA": "#7F8C8D", "HBP": "#34495E", "H.D.": "#2C3E50",
    "SPAN": "#F39C12",
}


# ──────────────────────────────────────────────
# 1. Load Data
# ──────────────────────────────────────────────
def load_cohort_clinical():
    """Load clinical data for the 1,628 cohort + SPAN.

    Strategy: use individual standardized clinical files directly for each group.
    Some groups (BRE, OVA, BLC) have hospital-number patient_ids that don't
    match spectral sample_ids, so we use the clinical files as-is for
    aggregate demography (no per-sample merge needed).
    """
    std_dir = PROJECT_ROOT / "data" / "clinical_data" / "standardized"

    # Map: spectral group -> clinical file + disease_group name
    group_file_map = {
        "LUN": ("LUN_clinical_standardized.csv", "LUN"),
        "CRC": ("CRC_clinical_standardized.csv", "CRC"),
        "BLC": ("BLA_clinical_standardized.csv", "BLC"),   # BLA -> BLC
        "PRO": ("PRO_clinical_standardized.csv", "PRO"),
        "OVA": ("OVA_clinical_standardized.csv", "OVA"),
        "CPAN": ("PAN_clinical_standardized.csv", "CPAN"),  # PAN -> CPAN
        "YPAN": ("YPAN_clinical_standardized.csv", "YPAN"),
        "BRE": ("BRE_clinical_standardized.csv", "BRE"),
        "NOR": ("NOR_clinical_standardized.csv", "NOR"),
        "YNOR": ("YNOR_clinical_standardized.csv", "YNOR"),
        "DIA": ("DIA_clinical_standardized.csv", "DIA"),
        "HBP": ("HBP_clinical_standardized.csv", "HBP"),
        "H.D.": ("H.D._clinical_standardized.csv", "H.D."),
    }

    # Spectral subject counts (ground truth)
    spec = pd.read_csv(
        PROJECT_ROOT / "results" / "processed_spectra.csv",
        usecols=["group", "sample_id"],
    )
    spec_n = spec.groupby("group")["sample_id"].nunique().to_dict()

    parts = []
    for grp, (fname, label) in group_file_map.items():
        fpath = std_dir / fname
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found, skipping {grp}")
            continue
        df = pd.read_csv(fpath)
        df["disease_group"] = label
        # Limit to the number of subjects that actually have spectra
        n_spec = spec_n.get(grp, len(df))
        if len(df) > n_spec:
            df = df.head(n_spec)
        parts.append(df)

    cohort_clin = pd.concat(parts, ignore_index=True)

    # Clean obvious BMI outliers (data entry errors: height in cm < 100, BMI > 60)
    cohort_clin.loc[cohort_clin["bmi"] > 60, "bmi"] = np.nan

    # SPAN clinical (separate)
    span_clin = pd.read_csv(std_dir / "SPAN_clinical_standardized.csv")
    span_clin["disease_group"] = "SPAN"

    return cohort_clin, span_clin


def load_spectra():
    """Load processed spectra for main cohort."""
    df = pd.read_csv(PROJECT_ROOT / "results" / "processed_spectra.csv")
    wn_cols = [c for c in df.columns if c.startswith("x_")]
    wavenumbers = np.array([float(c.replace("x_", "")) for c in wn_cols])
    return df, wavenumbers, wn_cols


def process_span_spectra():
    """Process SPAN raw spectra through the same pipeline."""
    from src.sers.config import load_config
    from src.sers.io import find_spectra, read_spectrum, parse_filename, make_common_grid
    from src.sers.preprocessing import preprocess_spectra

    config = load_config()
    span_dir = PROJECT_ROOT / "data" / "raw_data" / "10-2. S-Pancreatic cancer (72개)"

    files = list(find_spectra(span_dir, pattern="*.CSV"))
    print(f"  SPAN: Found {len(files)} raw files")

    raw_spectra = {}
    for fp in files:
        try:
            spec_id = parse_filename(fp, fallback_group="SPAN")
            x, y = read_spectrum(fp)
            raw_spectra[(spec_id.group, spec_id.sample_id, spec_id.replicate)] = (x, y)
        except Exception:
            continue

    print(f"  SPAN: Loaded {len(raw_spectra)} spectra")

    # Use same common grid as main cohort
    main_df = pd.read_csv(PROJECT_ROOT / "results" / "processed_spectra.csv", nrows=1)
    wn_cols = [c for c in main_df.columns if c.startswith("x_")]
    common_grid = np.array([float(c.replace("x_", "")) for c in wn_cols])

    processed, _, proc_grid = preprocess_spectra(raw_spectra, common_grid, config)
    print(f"  SPAN: Preprocessed {len(processed)} spectra")

    # Convert to DataFrame
    rows = []
    for (g, sid, rep), y in processed.items():
        row = {"group": g, "sample_id": sid, "replicate": rep}
        for i, wn in enumerate(proc_grid):
            row[f"x_{wn:.2f}"] = y[i]
        rows.append(row)

    span_df = pd.DataFrame(rows)
    return span_df, proc_grid


# ──────────────────────────────────────────────
# 2. Demography Figures
# ──────────────────────────────────────────────
def make_demography_table(clin, groups, label="cohort"):
    """Create demography summary table."""
    rows = []
    for g in groups:
        sub = clin[clin["disease_group"] == g]
        n = len(sub)
        if n == 0:
            continue
        age_mean = sub["age"].mean()
        age_std = sub["age"].std()
        n_male = (sub["sex"] == "M").sum()
        n_female = (sub["sex"] == "F").sum()
        bmi_mean = sub["bmi"].mean()
        bmi_std = sub["bmi"].std()
        rows.append({
            "Group": GROUP_LABELS.get(g, g).replace("\n", " "),
            "Code": g,
            "N": n,
            "Age (mean±SD)": f"{age_mean:.1f}±{age_std:.1f}" if pd.notna(age_mean) else "N/A",
            "Sex (M/F)": f"{n_male}/{n_female}",
            "BMI (mean±SD)": f"{bmi_mean:.1f}±{bmi_std:.1f}" if pd.notna(bmi_mean) else "N/A",
        })

    # Total row
    total = clin[clin["disease_group"].isin(groups)]
    rows.append({
        "Group": "Total",
        "Code": "",
        "N": len(total),
        "Age (mean±SD)": f"{total['age'].mean():.1f}±{total['age'].std():.1f}",
        "Sex (M/F)": f"{(total['sex']=='M').sum()}/{(total['sex']=='F').sum()}",
        "BMI (mean±SD)": f"{total['bmi'].mean():.1f}±{total['bmi'].std():.1f}" if total['bmi'].notna().sum() > 0 else "N/A",
    })

    df = pd.DataFrame(rows)
    return df


def plot_demography_overview(cohort_clin, span_clin):
    """Create comprehensive demography figure."""
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ── Panel A: Sample count bar chart ──
    ax_bar = fig.add_subplot(gs[0, 0])
    groups_in_data = [g for g in GROUP_ORDER_ALL if g in cohort_clin["disease_group"].values]
    counts = [len(cohort_clin[cohort_clin["disease_group"] == g]) for g in groups_in_data]
    colors = [GROUP_COLORS.get(g, "#999") for g in groups_in_data]
    labels = [GROUP_LABELS.get(g, g).replace("\n", " ") for g in groups_in_data]

    bars = ax_bar.barh(range(len(groups_in_data)), counts, color=colors, edgecolor="white", height=0.7)
    ax_bar.set_yticks(range(len(groups_in_data)))
    ax_bar.set_yticklabels(labels, fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Number of Subjects")
    ax_bar.set_title("A. Cohort Composition (N=1,628)")
    for bar, c in zip(bars, counts):
        ax_bar.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2, str(c),
                    va="center", fontsize=8)
    ax_bar.spines[["top", "right"]].set_visible(False)

    # ── Panel B: Age distribution by group (box plot) ──
    ax_age = fig.add_subplot(gs[0, 1])
    age_data = []
    age_labels = []
    age_colors = []
    for g in groups_in_data:
        vals = cohort_clin.loc[cohort_clin["disease_group"] == g, "age"].dropna()
        if len(vals) > 0:
            age_data.append(vals.values)
            age_labels.append(g)
            age_colors.append(GROUP_COLORS.get(g, "#999"))

    bp = ax_age.boxplot(age_data, vert=True, patch_artist=True, widths=0.6,
                         medianprops=dict(color="black", linewidth=1.5))
    for patch, c in zip(bp["boxes"], age_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax_age.set_xticklabels(age_labels, rotation=45, ha="right", fontsize=8)
    ax_age.set_ylabel("Age (years)")
    ax_age.set_title("B. Age Distribution by Group")
    ax_age.spines[["top", "right"]].set_visible(False)

    # ── Panel C: Sex ratio stacked bar ──
    ax_sex = fig.add_subplot(gs[0, 2])
    male_counts = []
    female_counts = []
    for g in groups_in_data:
        sub = cohort_clin[cohort_clin["disease_group"] == g]
        male_counts.append((sub["sex"] == "M").sum())
        female_counts.append((sub["sex"] == "F").sum())

    x_pos = range(len(groups_in_data))
    ax_sex.bar(x_pos, male_counts, color="#3498DB", label="Male", edgecolor="white")
    ax_sex.bar(x_pos, female_counts, bottom=male_counts, color="#E74C3C", label="Female", edgecolor="white")
    ax_sex.set_xticks(x_pos)
    ax_sex.set_xticklabels(groups_in_data, rotation=45, ha="right", fontsize=8)
    ax_sex.set_ylabel("Number of Subjects")
    ax_sex.set_title("C. Sex Distribution")
    ax_sex.legend(loc="upper right", fontsize=9)
    ax_sex.spines[["top", "right"]].set_visible(False)

    # ── Panel D: BMI distribution ──
    ax_bmi = fig.add_subplot(gs[1, 0])
    bmi_data = []
    bmi_labels = []
    bmi_colors = []
    for g in groups_in_data:
        vals = cohort_clin.loc[cohort_clin["disease_group"] == g, "bmi"].dropna()
        if len(vals) > 2:
            bmi_data.append(vals.values)
            bmi_labels.append(g)
            bmi_colors.append(GROUP_COLORS.get(g, "#999"))

    bp2 = ax_bmi.boxplot(bmi_data, vert=True, patch_artist=True, widths=0.6,
                          medianprops=dict(color="black", linewidth=1.5))
    for patch, c in zip(bp2["boxes"], bmi_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax_bmi.set_xticklabels(bmi_labels, rotation=45, ha="right", fontsize=8)
    ax_bmi.set_ylabel("BMI (kg/m²)")
    ax_bmi.set_title("D. BMI Distribution by Group")
    ax_bmi.spines[["top", "right"]].set_visible(False)

    # ── Panel E: SPAN demography comparison ──
    ax_span = fig.add_subplot(gs[1, 1])
    pan_groups = ["CPAN", "YPAN"]
    pan_data_age = []
    pan_labels_age = ["CPAN\n(CBNU)", "YPAN\n(Severance)", "SPAN\n(Samsung,\npost-op)"]
    pan_colors_age = [PAN_COLORS["CPAN"], PAN_COLORS["YPAN"], PAN_COLORS["SPAN"]]

    for g in pan_groups:
        vals = cohort_clin.loc[cohort_clin["disease_group"] == g, "age"].dropna()
        pan_data_age.append(vals.values)
    pan_data_age.append(span_clin["age"].dropna().values)

    bp3 = ax_span.boxplot(pan_data_age, vert=True, patch_artist=True, widths=0.5,
                           medianprops=dict(color="black", linewidth=1.5))
    for patch, c in zip(bp3["boxes"], pan_colors_age):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax_span.set_xticklabels(pan_labels_age, fontsize=9)
    ax_span.set_ylabel("Age (years)")
    ax_span.set_title("E. Pancreatic Subgroup — Age")
    ax_span.spines[["top", "right"]].set_visible(False)

    # Add N labels
    for i, data in enumerate(pan_data_age):
        ax_span.text(i + 1, ax_span.get_ylim()[1] - 2, f"n={len(data)}",
                     ha="center", fontsize=8, style="italic")

    # ── Panel F: SPAN sex comparison ──
    ax_span_sex = fig.add_subplot(gs[1, 2])
    pan_all = ["CPAN", "YPAN", "SPAN"]
    male_pan = []
    female_pan = []
    for g in pan_all:
        if g == "SPAN":
            sub = span_clin
        else:
            sub = cohort_clin[cohort_clin["disease_group"] == g]
        male_pan.append((sub["sex"] == "M").sum())
        female_pan.append((sub["sex"] == "F").sum())

    x_pan = range(3)
    ax_span_sex.bar(x_pan, male_pan, color="#3498DB", label="Male", edgecolor="white", width=0.5)
    ax_span_sex.bar(x_pan, female_pan, bottom=male_pan, color="#E74C3C", label="Female",
                     edgecolor="white", width=0.5)
    ax_span_sex.set_xticks(x_pan)
    ax_span_sex.set_xticklabels(["CPAN\n(CBNU)", "YPAN\n(Severance)", "SPAN\n(Samsung)"], fontsize=9)
    ax_span_sex.set_ylabel("Number of Subjects")
    ax_span_sex.set_title("F. Pancreatic Subgroup — Sex")
    ax_span_sex.legend(fontsize=9)
    ax_span_sex.spines[["top", "right"]].set_visible(False)

    # Add N labels
    for i, (m, f) in enumerate(zip(male_pan, female_pan)):
        ax_span_sex.text(i, m + f + 1, f"n={m+f}", ha="center", fontsize=8, style="italic")

    fig.suptitle("Cohort Demographics", fontsize=16, fontweight="bold", y=0.98)
    fig.savefig(OUT_DIR / "fig1_demography_overview.png")
    fig.savefig(OUT_DIR / "fig1_demography_overview.pdf")
    plt.close(fig)
    print("  Saved: fig1_demography_overview.png/pdf")


# ──────────────────────────────────────────────
# 3. Spectrum Figures
# ──────────────────────────────────────────────
def plot_individual_spectra(spec_df, wavenumbers, wn_cols):
    """Plot individual spectra faceted by group (random subset)."""
    groups = [g for g in GROUP_ORDER_ALL if g in spec_df["group"].values]
    n_groups = len(groups)
    n_cols = 4
    n_rows = int(np.ceil(n_groups / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, g in enumerate(groups):
        ax = axes[idx]
        sub = spec_df[spec_df["group"] == g]
        # Sample up to 50 spectra for visibility
        if len(sub) > 50:
            sub = sub.sample(50, random_state=42)
        for _, row in sub.iterrows():
            ax.plot(wavenumbers, row[wn_cols].values, color=SPEC_COLORS.get(g, "#999"),
                    alpha=0.15, linewidth=0.5)
        # Mean spectrum
        mean_spec = spec_df.loc[spec_df["group"] == g, wn_cols].mean()
        ax.plot(wavenumbers, mean_spec.values, color=SPEC_COLORS.get(g, "#999"),
                linewidth=1.5, label="Mean")
        ax.set_title(f"{GROUP_LABELS.get(g, g).replace(chr(10), ' ')} (n={spec_df[spec_df['group']==g]['sample_id'].nunique()})",
                     fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Raman Shift (cm⁻¹)")
        if idx % n_cols == 0:
            ax.set_ylabel("Intensity (a.u.)")

    # Hide unused axes
    for idx in range(n_groups, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Individual SERS Spectra by Group", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "fig2_individual_spectra.png")
    fig.savefig(OUT_DIR / "fig2_individual_spectra.pdf")
    plt.close(fig)
    print("  Saved: fig2_individual_spectra.png/pdf")


def plot_mean_spectra_all(spec_df, wavenumbers, wn_cols):
    """Plot mean spectra for all groups overlaid."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Cancer groups
    for g in GROUP_ORDER_CANCER:
        if g in spec_df["group"].values:
            mean_spec = spec_df.loc[spec_df["group"] == g, wn_cols].mean()
            n = spec_df[spec_df["group"] == g]["sample_id"].nunique()
            ax1.plot(wavenumbers, mean_spec.values, color=SPEC_COLORS[g],
                     linewidth=1.5, label=f"{g} (n={n})")
    ax1.set_ylabel("Intensity (a.u.)")
    ax1.set_title("A. Cancer Groups — Mean SERS Spectra")
    ax1.legend(fontsize=8, ncol=2, loc="upper right")
    ax1.spines[["top", "right"]].set_visible(False)

    # Non-cancer groups
    for g in GROUP_ORDER_NONCANCER:
        if g in spec_df["group"].values:
            mean_spec = spec_df.loc[spec_df["group"] == g, wn_cols].mean()
            n = spec_df[spec_df["group"] == g]["sample_id"].nunique()
            ax2.plot(wavenumbers, mean_spec.values, color=SPEC_COLORS[g],
                     linewidth=1.5, label=f"{g} (n={n})")
    ax2.set_xlabel("Raman Shift (cm⁻¹)")
    ax2.set_ylabel("Intensity (a.u.)")
    ax2.set_title("B. Non-Cancer Groups — Mean SERS Spectra")
    ax2.legend(fontsize=8, ncol=2, loc="upper right")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Mean Preprocessed SERS Spectra by Group", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "fig3_mean_spectra_all.png")
    fig.savefig(OUT_DIR / "fig3_mean_spectra_all.pdf")
    plt.close(fig)
    print("  Saved: fig3_mean_spectra_all.png/pdf")


def plot_pancreatic_comparison(spec_df, span_df, wavenumbers, wn_cols):
    """CPAN vs YPAN vs SPAN spectrum comparison."""
    # Align SPAN columns to main wavenumbers
    span_wn_cols = [c for c in span_df.columns if c.startswith("x_")]
    span_wavenumbers = np.array([float(c.replace("x_", "")) for c in span_wn_cols])

    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ── Panel A: Mean spectra overlay (CPAN vs YPAN vs SPAN) ──
    ax_mean = fig.add_subplot(gs[0, :])
    for g, color, label in [
        ("CPAN", PAN_COLORS["CPAN"], "CPAN (CBNU, n=70)"),
        ("YPAN", PAN_COLORS["YPAN"], "YPAN (Severance, n=30)"),
    ]:
        mean_spec = spec_df.loc[spec_df["group"] == g, wn_cols].mean()
        std_spec = spec_df.loc[spec_df["group"] == g, wn_cols].std()
        ax_mean.plot(wavenumbers, mean_spec.values, color=color, linewidth=2, label=label)
        ax_mean.fill_between(wavenumbers, (mean_spec - std_spec).values, (mean_spec + std_spec).values,
                             color=color, alpha=0.15)

    # SPAN
    span_mean = span_df[span_wn_cols].mean()
    span_std = span_df[span_wn_cols].std()
    ax_mean.plot(span_wavenumbers, span_mean.values, color=PAN_COLORS["SPAN"], linewidth=2,
                 label=f"SPAN (Samsung post-op, n={span_df['sample_id'].nunique()})")
    ax_mean.fill_between(span_wavenumbers, (span_mean - span_std).values, (span_mean + span_std).values,
                         color=PAN_COLORS["SPAN"], alpha=0.15)

    ax_mean.set_xlabel("Raman Shift (cm⁻¹)")
    ax_mean.set_ylabel("Intensity (a.u.)")
    ax_mean.set_title("A. Mean SERS Spectra — Pancreatic Cancer Subgroups (±1 SD)")
    ax_mean.legend(fontsize=10)
    ax_mean.spines[["top", "right"]].set_visible(False)

    # ── Panel B: CPAN individual spectra ──
    ax_cpan = fig.add_subplot(gs[1, 0])
    cpan_sub = spec_df[spec_df["group"] == "CPAN"].sample(min(50, len(spec_df[spec_df["group"] == "CPAN"])), random_state=42)
    for _, row in cpan_sub.iterrows():
        ax_cpan.plot(wavenumbers, row[wn_cols].values, color=PAN_COLORS["CPAN"], alpha=0.15, linewidth=0.5)
    cpan_mean = spec_df.loc[spec_df["group"] == "CPAN", wn_cols].mean()
    ax_cpan.plot(wavenumbers, cpan_mean.values, color=PAN_COLORS["CPAN"], linewidth=2)
    ax_cpan.set_title("B. CPAN — Individual Spectra (CBNU, n=70)")
    ax_cpan.set_xlabel("Raman Shift (cm⁻¹)")
    ax_cpan.set_ylabel("Intensity (a.u.)")
    ax_cpan.spines[["top", "right"]].set_visible(False)

    # ── Panel C: YPAN individual spectra ──
    ax_ypan = fig.add_subplot(gs[1, 1])
    ypan_sub = spec_df[spec_df["group"] == "YPAN"]
    for _, row in ypan_sub.iterrows():
        ax_ypan.plot(wavenumbers, row[wn_cols].values, color=PAN_COLORS["YPAN"], alpha=0.15, linewidth=0.5)
    ypan_mean = spec_df.loc[spec_df["group"] == "YPAN", wn_cols].mean()
    ax_ypan.plot(wavenumbers, ypan_mean.values, color=PAN_COLORS["YPAN"], linewidth=2)
    ax_ypan.set_title("C. YPAN — Individual Spectra (Severance, n=30)")
    ax_ypan.set_xlabel("Raman Shift (cm⁻¹)")
    ax_ypan.set_ylabel("Intensity (a.u.)")
    ax_ypan.spines[["top", "right"]].set_visible(False)

    # ── Panel D: SPAN individual spectra ──
    ax_span = fig.add_subplot(gs[2, 0])
    span_sub = span_df.sample(min(50, len(span_df)), random_state=42)
    for _, row in span_sub.iterrows():
        ax_span.plot(span_wavenumbers, row[span_wn_cols].values, color=PAN_COLORS["SPAN"],
                     alpha=0.15, linewidth=0.5)
    ax_span.plot(span_wavenumbers, span_mean.values, color=PAN_COLORS["SPAN"], linewidth=2)
    ax_span.set_title(f"D. SPAN — Individual Spectra (Samsung post-op, n={span_df['sample_id'].nunique()})")
    ax_span.set_xlabel("Raman Shift (cm⁻¹)")
    ax_span.set_ylabel("Intensity (a.u.)")
    ax_span.spines[["top", "right"]].set_visible(False)

    # ── Panel E: Difference spectra ──
    ax_diff = fig.add_subplot(gs[2, 1])
    # Use common wavenumber grid (they should be the same from preprocessing)
    # CPAN - SPAN difference
    if len(wn_cols) == len(span_wn_cols):
        diff_cpan_span = cpan_mean.values - span_mean.values
        diff_ypan_span = ypan_mean.values - span_mean.values
        ax_diff.plot(wavenumbers, diff_cpan_span, color=PAN_COLORS["CPAN"], linewidth=1.5,
                     label="CPAN − SPAN")
        ax_diff.plot(wavenumbers, diff_ypan_span, color=PAN_COLORS["YPAN"], linewidth=1.5,
                     label="YPAN − SPAN")
        ax_diff.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax_diff.set_xlabel("Raman Shift (cm⁻¹)")
        ax_diff.set_ylabel("Δ Intensity (a.u.)")
        ax_diff.set_title("E. Difference Spectra (vs SPAN)")
        ax_diff.legend(fontsize=10)
        ax_diff.spines[["top", "right"]].set_visible(False)
    else:
        ax_diff.text(0.5, 0.5, "Wavenumber grids differ\n— interpolation needed",
                     transform=ax_diff.transAxes, ha="center", va="center")

    fig.suptitle("Pancreatic Cancer Subgroup Comparison:\nPre-operative (CPAN, YPAN) vs Post-operative (SPAN)",
                 fontsize=15, fontweight="bold", y=1.0)
    fig.savefig(OUT_DIR / "fig4_pancreatic_comparison.png")
    fig.savefig(OUT_DIR / "fig4_pancreatic_comparison.pdf")
    plt.close(fig)
    print("  Saved: fig4_pancreatic_comparison.png/pdf")


# ──────────────────────────────────────────────
# 4. Demography Tables (CSV)
# ──────────────────────────────────────────────
def save_demography_tables(cohort_clin, span_clin):
    """Save demography summary tables as CSV."""
    # Main cohort
    cohort_table = make_demography_table(cohort_clin, GROUP_ORDER_ALL, "cohort")
    cohort_table.to_csv(OUT_DIR / "table1_cohort_demography.csv", index=False)
    print("  Saved: table1_cohort_demography.csv")
    print(cohort_table.to_string(index=False))

    # SPAN
    span_table = make_demography_table(span_clin, ["SPAN"], "SPAN")
    span_table.to_csv(OUT_DIR / "table2_span_demography.csv", index=False)
    print("\n  Saved: table2_span_demography.csv")
    print(span_table.to_string(index=False))

    # Pancreatic comparison table
    pan_clin = pd.concat([
        cohort_clin[cohort_clin["disease_group"].isin(["CPAN", "YPAN"])],
        span_clin,
    ])
    pan_table = make_demography_table(pan_clin, ["CPAN", "YPAN", "SPAN"], "pancreatic")
    pan_table.to_csv(OUT_DIR / "table3_pancreatic_demography.csv", index=False)
    print("\n  Saved: table3_pancreatic_demography.csv")
    print(pan_table.to_string(index=False))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Poster Figure Generation")
    print("=" * 60)

    print("\n[1/5] Loading clinical data...")
    cohort_clin, span_clin = load_cohort_clinical()
    print(f"  Cohort clinical: {len(cohort_clin)} rows")
    print(f"  SPAN clinical: {len(span_clin)} rows")

    print("\n[2/5] Loading processed spectra...")
    spec_df, wavenumbers, wn_cols = load_spectra()
    print(f"  Main spectra: {spec_df.shape}")

    print("\n[3/5] Processing SPAN spectra...")
    span_df, span_grid = process_span_spectra()
    print(f"  SPAN spectra: {span_df.shape}")

    print("\n[4/5] Generating demography tables...")
    save_demography_tables(cohort_clin, span_clin)

    print("\n[5/5] Generating figures...")
    plot_demography_overview(cohort_clin, span_clin)
    plot_individual_spectra(spec_df, wavenumbers, wn_cols)
    plot_mean_spectra_all(spec_df, wavenumbers, wn_cols)
    plot_pancreatic_comparison(spec_df, span_df, wavenumbers, wn_cols)

    print(f"\n{'=' * 60}")
    print(f"All figures saved to: {OUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
