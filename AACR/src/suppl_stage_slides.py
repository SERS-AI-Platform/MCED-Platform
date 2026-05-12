"""
Supplementary Slides: Stage-Stratified TOO (Tumor of Origin) Analysis

Slide 1: Stage distribution + TOO accuracy by stage (Early vs Late)
Slide 2: Non-cancer → Early → Late stage SERS spectra per cancer type
Slide 3: Early-stage TOO accuracy summary card

16:9, large fonts for tablet/laptop.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve
from collections import Counter
from nature_style import (
    apply_style, AACR_COLORS, AACR_LABELS,
    CANCER_COLORS, SERS_ROOT,
)

apply_style()

AACR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(AACR_DIR, "figures")
FUSION_PATH = os.path.join(AACR_DIR, "data", "early_fusion", "fold_predictions.npz")
SPECTRA_PATH = os.path.join(SERS_ROOT, "results", "processed_spectra.csv")

SLIDE_W, SLIDE_H = 16, 9

EARLY_COLOR = "#2980B9"
LATE_COLOR = "#E74C3C"
NORMAL_COLOR = "#7F8C8D"
BG_COLOR = "#FFFFFF"
CARD_BG = "#F8F9FA"
TEXT_DARK = "#1a1a1a"
TEXT_MED = "#555555"
TEXT_LIGHT = "#888888"
ACCENT = "#2C3E50"

CANCER_DISPLAY = {
    "PRO": ("PRC", "Prostate", CANCER_COLORS["PRO"]),
    "LUN": ("LC", "Lung", CANCER_COLORS["LUN"]),
    "CPAN": ("PAC", "Pancreatic", CANCER_COLORS["PAN"]),
    "CRC": ("CRC", "Colorectal", CANCER_COLORS["CRC"]),
}
CANCER_ORDER = ["PRO", "LUN", "CPAN", "CRC"]
# Cancer type label indices: PRO=0, LUN=1, CRC=2, CPAN=3, OVA=4
CANCER_TYPE_IDX = {"PRO": 0, "LUN": 1, "CRC": 2, "CPAN": 3}
NON_CANCER_GROUPS = {"NOR", "DIA", "HBP", "H.D."}


# ── Staging ──

def classify_ajcc(s):
    if pd.isna(s): return None
    s = str(s).strip().upper()
    if s in ("UNKNOWN", ""): return None
    if s.startswith("III") or s.startswith("IV"): return "Late"
    if s.startswith("I") or s.startswith("II"): return "Early"
    return None

def classify_tstage(s):
    if pd.isna(s): return None
    s = str(s).strip().upper()
    if s.startswith("T1") or s.startswith("T2"): return "Early"
    if s.startswith("T3") or s.startswith("T4"): return "Late"
    return None


def load_staging():
    staging = {}
    pro = pd.read_csv(os.path.join(AACR_DIR, "data", "PRO_with_staging.csv"))
    pro["sample_id"] = pro["patient_id"].str.replace("PRO ", "").astype(int)
    pro["stage_group"] = pro["t_stage"].apply(classify_tstage)
    staging["PRO"] = pro[["sample_id", "stage_group"]].dropna(subset=["stage_group"])

    lun = pd.read_csv(os.path.join(SERS_ROOT, "data", "clinical_data", "standardized",
                                    "LUN_clinical_standardized.csv"))
    lun["sample_id"] = np.arange(1, len(lun) + 1)
    lun["stage_group"] = lun["stage"].apply(classify_ajcc)
    staging["LUN"] = lun[["sample_id", "stage_group"]].dropna(subset=["stage_group"])

    pan = pd.read_csv(os.path.join(AACR_DIR, "data", "PAN_with_ajcc_stage.csv"))
    pan["sample_id"] = np.arange(1, len(pan) + 1)
    pan["stage_group"] = pan["ajcc_stage"].apply(classify_ajcc)
    staging["CPAN"] = pan[["sample_id", "stage_group"]].dropna(subset=["stage_group"])

    crc = pd.read_csv(os.path.join(AACR_DIR, "data", "CRC_with_ajcc_stage.csv"))
    crc["sample_id"] = np.arange(1, len(crc) + 1)
    crc["stage_group"] = crc["ajcc_stage"].apply(classify_ajcc)
    staging["CRC"] = crc[["sample_id", "stage_group"]].dropna(subset=["stage_group"])
    return staging


def load_too_results():
    """Compute patient-level TOO (cancer type) accuracy by stage."""
    fus = np.load(FUSION_PATH, allow_pickle=True)
    bl = fus["binary_labels"]
    ctl = fus["cancer_type_labels"]
    cl = fus["val_cancer_logits"]
    groups = fus["groups"]
    sids = fus["sample_ids"]

    cancer_mask = bl == 1
    pred_type = cl[cancer_mask].argmax(axis=1)
    true_type = ctl[cancer_mask]
    cancer_groups = groups[cancer_mask]
    cancer_sids = sids[cancer_mask]

    # Patient-level majority vote
    pat_votes = {}
    for i in range(len(cancer_groups)):
        key = (str(cancer_groups[i]), int(cancer_sids[i]))
        pat_votes.setdefault(key, {"true": int(true_type[i]), "preds": []})
        pat_votes[key]["preds"].append(int(pred_type[i]))

    pat_results = {}
    for key, v in pat_votes.items():
        majority = Counter(v["preds"]).most_common(1)[0][0]
        pat_results[key] = {"true": v["true"], "pred": majority,
                            "correct": majority == v["true"]}
    return pat_results


def compute_too_by_stage(pat_results, staging):
    """Returns dict: cancer -> {Early: {acc, n, correct}, Late: ...}."""
    results = {}
    for ct in CANCER_ORDER:
        st = staging[ct]
        results[ct] = {}
        for stage in ["Early", "Late"]:
            stage_sids = set(st.loc[st["stage_group"] == stage, "sample_id"].tolist())
            pts = {k: v for k, v in pat_results.items()
                   if k[0] == ct and k[1] in stage_sids}
            correct = sum(1 for v in pts.values() if v["correct"])
            total = len(pts)
            results[ct][stage] = {
                "acc": correct / total if total > 0 else 0,
                "correct": correct, "n": total,
            }
    return results


def load_spectra():
    df = pd.read_csv(SPECTRA_PATH)
    wn_cols = [c for c in df.columns if c.startswith("x_")]
    wavenumbers = np.array([float(c.replace("x_", "")) for c in wn_cols])
    return df, wn_cols, wavenumbers


# ══════════════════════════════════════════════════
#  SLIDE 1: Stage Distribution + TOO Accuracy
# ══════════════════════════════════════════════════

def make_slide1(staging, too_results):
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H), facecolor=BG_COLOR)

    fig.text(0.5, 0.94, "Cancer Type Identification (TOO) by Disease Stage",
             ha="center", va="top", fontsize=22, fontweight="bold", color=ACCENT)
    fig.text(0.5, 0.895,
             "uSERS-Net  |  Patient-level majority vote  |  5-fold CV  |  OVC excluded (staging <15%)",
             ha="center", va="top", fontsize=11, color=TEXT_LIGHT)

    gs = fig.add_gridspec(1, 2, left=0.06, right=0.96, top=0.83, bottom=0.10,
                          wspace=0.30)

    # ── Panel A: Stage distribution ──
    ax1 = fig.add_subplot(gs[0])
    early_n = [too_results[c]["Early"]["n"] for c in CANCER_ORDER]
    late_n = [too_results[c]["Late"]["n"] for c in CANCER_ORDER]
    labels = [CANCER_DISPLAY[c][0] for c in CANCER_ORDER]

    x = np.arange(len(CANCER_ORDER))
    w = 0.55
    bars_e = ax1.bar(x, early_n, w, color=EARLY_COLOR, label="Early (I–II)", zorder=3)
    bars_l = ax1.bar(x, late_n, w, bottom=early_n, color=LATE_COLOR, label="Late (III–IV)", zorder=3)

    for i, (e, l) in enumerate(zip(early_n, late_n)):
        if e > 0:
            ax1.text(i, e / 2, f"n={e}", ha="center", va="center",
                     fontsize=11, fontweight="bold", color="white")
        if l > 0:
            ax1.text(i, e + l / 2, f"n={l}", ha="center", va="center",
                     fontsize=11, fontweight="bold", color="white")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=13, fontweight="bold")
    for i, c in enumerate(CANCER_ORDER):
        ax1.get_xticklabels()[i].set_color(CANCER_DISPLAY[c][2])
    ax1.set_ylabel("Number of patients", fontsize=12)
    ax1.set_title("a   Cohort Stage Distribution", fontsize=14, fontweight="bold",
                   loc="left", pad=10, color=ACCENT)
    ax1.legend(fontsize=11, loc="upper right", frameon=True, edgecolor="#ddd")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(axis="y", labelsize=10)
    ax1.text(0, -0.12, "PRC: T1–T2 vs T3–T4 (T-stage proxy)",
             transform=ax1.transAxes, fontsize=9, color=TEXT_LIGHT, style="italic")

    # ── Panel B: TOO Accuracy grouped bar ──
    ax2 = fig.add_subplot(gs[1])

    bar_w = 0.30
    x = np.arange(len(CANCER_ORDER))

    early_acc = [too_results[c]["Early"]["acc"] * 100 for c in CANCER_ORDER]
    late_acc = [too_results[c]["Late"]["acc"] * 100 for c in CANCER_ORDER]

    b1 = ax2.bar(x - bar_w/2, early_acc, bar_w, color=EARLY_COLOR,
                 label="Early (I–II)", zorder=3)
    b2 = ax2.bar(x + bar_w/2, late_acc, bar_w, color=LATE_COLOR,
                 label="Late (III–IV)", zorder=3)

    # Value labels
    for i in range(len(CANCER_ORDER)):
        ax2.text(x[i] - bar_w/2, early_acc[i] + 1.2, f"{early_acc[i]:.1f}%",
                 ha="center", va="bottom", fontsize=11, fontweight="bold",
                 color=EARLY_COLOR)
        ax2.text(x[i] + bar_w/2, late_acc[i] + 1.2, f"{late_acc[i]:.1f}%",
                 ha="center", va="bottom", fontsize=11, fontweight="bold",
                 color=LATE_COLOR)

    ax2.set_xticks(x)
    ax2.set_xticklabels([CANCER_DISPLAY[c][0] for c in CANCER_ORDER],
                        fontsize=13, fontweight="bold")
    for i, c in enumerate(CANCER_ORDER):
        ax2.get_xticklabels()[i].set_color(CANCER_DISPLAY[c][2])

    ax2.set_ylabel("TOO Accuracy (%)", fontsize=12)
    ax2.set_ylim(70, 108)
    ax2.axhline(90, color="#ddd", linewidth=1, linestyle="--", zorder=0)
    ax2.text(3.5, 91, "90%", fontsize=9, color=TEXT_LIGHT)
    ax2.set_title("b   Cancer Type ID Accuracy by Stage", fontsize=14,
                   fontweight="bold", loc="left", pad=10, color=ACCENT)
    ax2.legend(fontsize=11, loc="lower right", frameon=True, edgecolor="#ddd")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="y", labelsize=10)

    outpath = os.path.join(FIGURE_DIR, "suppl_slide1_too_by_stage")
    fig.savefig(outpath + ".png", dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    fig.savefig(outpath + ".pdf", bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Slide 1 saved: {outpath}.png/.pdf")


# ══════════════════════════════════════════════════
#  SLIDE 2: Non-cancer → Early → Late Spectra
# ══════════════════════════════════════════════════

def make_slide2(staging, spectra_df, wn_cols, wavenumbers):
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H), facecolor=BG_COLOR)

    fig.text(0.5, 0.94, "SERS Spectral Signatures: Non-cancer → Early → Late Stage",
             ha="center", va="top", fontsize=22, fontweight="bold", color=ACCENT)
    fig.text(0.5, 0.895,
             "Mean spectra  |  Non-cancer (n=400) as baseline  |  Shaded = top 5% difference regions",
             ha="center", va="top", fontsize=11, color=TEXT_LIGHT)

    gs = fig.add_gridspec(2, 2, left=0.06, right=0.97, top=0.83, bottom=0.08,
                          wspace=0.20, hspace=0.35)

    # Non-cancer mean spectrum (shared baseline)
    nc_mask = spectra_df["group"].isin(NON_CANCER_GROUPS)
    nc_mean = spectra_df.loc[nc_mask, wn_cols].values.mean(axis=0)

    for idx, cancer in enumerate(CANCER_ORDER):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])

        abbr, full, color = CANCER_DISPLAY[cancer]
        st = staging[cancer]
        early_ids = set(st.loc[st["stage_group"] == "Early", "sample_id"].tolist())
        late_ids = set(st.loc[st["stage_group"] == "Late", "sample_id"].tolist())

        c_mask = spectra_df["group"] == cancer
        cancer_df = spectra_df[c_mask]

        early_spectra = cancer_df.loc[cancer_df["sample_id"].isin(early_ids), wn_cols].values
        late_spectra = cancer_df.loc[cancer_df["sample_id"].isin(late_ids), wn_cols].values

        n_early = cancer_df.loc[cancer_df["sample_id"].isin(early_ids), "sample_id"].nunique()
        n_late = cancer_df.loc[cancer_df["sample_id"].isin(late_ids), "sample_id"].nunique()

        mean_e = early_spectra.mean(axis=0) if len(early_spectra) > 0 else None
        mean_l = late_spectra.mean(axis=0) if len(late_spectra) > 0 else None

        # Plot Non-cancer baseline
        ax.plot(wavenumbers, nc_mean, color=NORMAL_COLOR, linewidth=1.2,
                label="Non-cancer (n=400)", alpha=0.7, zorder=2)

        # Plot Early
        if mean_e is not None:
            ax.plot(wavenumbers, mean_e, color=EARLY_COLOR, linewidth=1.8,
                    label=f"Early (n={n_early})", zorder=4)

        # Plot Late
        if mean_l is not None:
            ax.plot(wavenumbers, mean_l, color=LATE_COLOR, linewidth=1.5,
                    label=f"Late (n={n_late})", linestyle="--", zorder=3)

        # Highlight regions where cancer (early) differs from non-cancer
        if mean_e is not None:
            diff_from_nc = np.abs(mean_e - nc_mean)
            threshold = np.percentile(diff_from_nc, 95)
            highlight = diff_from_nc > threshold
            # Group consecutive highlighted regions
            in_region = False
            region_start = 0
            for i in range(len(wavenumbers)):
                if highlight[i] and not in_region:
                    region_start = i
                    in_region = True
                elif not highlight[i] and in_region:
                    ax.axvspan(wavenumbers[max(0, region_start - 3)],
                              wavenumbers[min(len(wavenumbers) - 1, i + 3)],
                              color=color, alpha=0.10, zorder=0)
                    in_region = False
            if in_region:
                ax.axvspan(wavenumbers[max(0, region_start - 3)],
                          wavenumbers[-1], color=color, alpha=0.10, zorder=0)

        stage_label = "T1–T2 / T3–T4" if cancer == "PRO" else "I–II / III–IV"
        ax.set_title(f"{abbr} — {full}", fontsize=14, fontweight="bold",
                     color=color, pad=8)
        ax.legend(fontsize=9, loc="upper right", frameon=True, edgecolor="#ddd",
                  title=stage_label, title_fontsize=8)
        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=10)
        ax.set_ylabel("Intensity (a.u.)", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    outpath = os.path.join(FIGURE_DIR, "suppl_slide2_spectra_normal_early_late")
    fig.savefig(outpath + ".png", dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    fig.savefig(outpath + ".pdf", bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Slide 2 saved: {outpath}.png/.pdf")


# ══════════════════════════════════════════════════
#  SLIDE 3: Early-Stage TOO Accuracy Cards
# ══════════════════════════════════════════════════

def make_slide3(too_results):
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H), facecolor=BG_COLOR)

    fig.text(0.5, 0.94, "Early-Stage Cancer Type Identification (TOO)",
             ha="center", va="top", fontsize=22, fontweight="bold", color=ACCENT)
    fig.text(0.5, 0.895,
             "uSERS-Net correctly identifies cancer type even at early stage (I–II)",
             ha="center", va="top", fontsize=12, color=TEXT_MED)

    card_width = 0.20
    card_height = 0.58
    gap = 0.035
    total_w = 4 * card_width + 3 * gap
    start_x = (1 - total_w) / 2
    card_y = 0.13

    for i, c in enumerate(CANCER_ORDER):
        abbr, full, color = CANCER_DISPLAY[c]
        r_early = too_results[c]["Early"]
        r_late = too_results[c]["Late"]
        x = start_x + i * (card_width + gap)

        # Card background
        rect = FancyBboxPatch((x, card_y), card_width, card_height,
                               boxstyle="round,pad=0.015",
                               facecolor=CARD_BG, edgecolor="#E0E0E0",
                               linewidth=1.5, transform=fig.transFigure, zorder=2)
        fig.patches.append(rect)

        # Color accent bar
        accent = plt.Rectangle((x, card_y + card_height - 0.04), card_width, 0.04,
                                facecolor=color, transform=fig.transFigure,
                                zorder=3, clip_on=False)
        fig.patches.append(accent)

        cx = x + card_width / 2

        # Cancer name
        fig.text(cx, card_y + card_height - 0.08, abbr,
                 ha="center", va="top", fontsize=18, fontweight="bold", color=color)
        fig.text(cx, card_y + card_height - 0.13, full,
                 ha="center", va="top", fontsize=10, color=TEXT_MED)

        # Early-stage TOO accuracy (big number)
        acc_str = f"{r_early['acc']:.1%}"
        fig.text(cx, card_y + card_height - 0.22, acc_str,
                 ha="center", va="top", fontsize=36, fontweight="bold",
                 color=EARLY_COLOR)
        fig.text(cx, card_y + card_height - 0.32, "Early-Stage\nTOO Accuracy",
                 ha="center", va="top", fontsize=10, color=TEXT_MED,
                 linespacing=1.4)

        # Divider
        line = plt.Line2D([x + 0.03, x + card_width - 0.03],
                          [card_y + card_height - 0.39, card_y + card_height - 0.39],
                          color="#E0E0E0", linewidth=1, transform=fig.transFigure)
        fig.lines.append(line)

        # Details
        fig.text(cx, card_y + card_height - 0.42,
                 f"{r_early['correct']}/{r_early['n']} patients",
                 ha="center", va="top", fontsize=10, color=EARLY_COLOR)

        fig.text(cx, card_y + card_height - 0.47,
                 f"Late: {r_late['acc']:.1%}  ({r_late['correct']}/{r_late['n']})",
                 ha="center", va="top", fontsize=10, color=LATE_COLOR)

        # Overall (all staged)
        total_correct = r_early["correct"] + r_late["correct"]
        total_n = r_early["n"] + r_late["n"]
        overall = total_correct / total_n if total_n > 0 else 0
        fig.text(cx, card_y + card_height - 0.52,
                 f"Overall: {overall:.1%}",
                 ha="center", va="top", fontsize=10, fontweight="bold",
                 color=TEXT_DARK)

        # Stage definition
        stage_def = "T1–T2 vs T3–T4" if c == "PRO" else "AJCC I–II vs III–IV"
        fig.text(cx, card_y + 0.03, stage_def,
                 ha="center", va="bottom", fontsize=8, color=TEXT_LIGHT,
                 style="italic")

    # Bottom message
    fig.text(0.5, 0.05,
             "TOO = Tumor of Origin  |  Even early-stage pancreatic cancer is correctly typed 87.2% of the time",
             ha="center", va="top", fontsize=12, fontweight="bold", color=ACCENT,
             style="italic")

    outpath = os.path.join(FIGURE_DIR, "suppl_slide3_too_early_stage")
    fig.savefig(outpath + ".png", dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    fig.savefig(outpath + ".pdf", bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Slide 3 saved: {outpath}.png/.pdf")


# ══════════════════════════════════════════════════
if __name__ == "__main__":
    print("Loading data...")
    staging = load_staging()
    pat_results = load_too_results()
    too_results = compute_too_by_stage(pat_results, staging)
    spectra_df, wn_cols, wavenumbers = load_spectra()

    print("\n=== TOO Accuracy by Stage ===")
    for c in CANCER_ORDER:
        abbr = CANCER_DISPLAY[c][0]
        e = too_results[c]["Early"]
        l = too_results[c]["Late"]
        print(f"  {abbr}: Early {e['acc']:.1%} ({e['correct']}/{e['n']})  "
              f"Late {l['acc']:.1%} ({l['correct']}/{l['n']})")

    print("\nGenerating slides...")
    make_slide1(staging, too_results)
    make_slide2(staging, spectra_df, wn_cols, wavenumbers)
    make_slide3(too_results)
    print("\nDone! All slides in:", FIGURE_DIR)
