#!/usr/bin/env python3
"""
2-Device Visualization: Confusion Matrix, Feature Importance (SHAP-like), Mean Spectra
=======================================================================================

Generates per-instrument figures for dashboard embedding.

Output: results/two_device_figures/
"""

from __future__ import annotations
import sys, warnings, logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

# Korean font support
plt.rcParams["font.family"] = ["Noto Sans KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from scipy.signal import savgol_filter

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.preprocessing import trim_spectrum, baseline_correction, normalize_spectrum, resample
from src.sers.io import find_spectra, read_spectrum, parse_filename

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "results" / "two_device_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANCER_TYPES = ("PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC")
NON_CANCER_GROUPS = ("NOR", "DIA", "HBP", "H.D.")
ALL_GROUPS_ORDER = ["NOR", "DIA", "HBP", "H.D.", "PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "BLC"]
SG_WINDOW, SG_POLY = 11, 3

THERMO_DIR = PROJECT_ROOT / "data" / "raw_data"
MEDICAL_DIR = PROJECT_ROOT / "data" / "raw_data_medical"
THERMO_MAP = {
    "1. Prostate cancer (100개)": "PRO", "2. Breast cancer (30개)": "BRE",
    "3. Ovarian cancer (70개)": "OVA", "4. Lung cancer (300개)": "LUN",
    "5. Normal (100개)": "NOR", "6. Diabetes (100개)": "DIA",
    "7. High blood pressure (100개)": "HBP",
    "8. High blood pressure + Diabetes (100개)": "H.D.",
    "9. Colorectal cancer (300개)": "CRC",
    "10-1. C-Pancreatic cancer (70개)": "CPAN",
    "10-3. Y-Pancreatic cancer (YPAN)": "YPAN",
    "11 BLC (299개)": "BLC", "12. Y-Normal (YNOR)": "YNOR",
}
MEDICAL_MAP = {
    "1. Prostate cancer (100개)": "PRO", "2. Breast cancer (30개)": "BRE",
    "3. Ovarian cancer (70개)": "OVA", "4. Lung cancer (300개)": "LUN",
    "5. Normal (100개)": "NOR", "6. Diabetes (100개)": "DIA",
    "7. High blood pressure (100개)": "HBP",
    "8. High blood pressure + Diabetes (100개)": "H.D.",
    "9. Colorectal cancer (300개)": "CRC",
    "10-1. C-Pancreatic cancer (70개)": "CPAN",
    "10-3. Y-Pancreatic cancer (30개)": "YPAN",
    "11. Bladdder Cancer (299개)": "BLC",
    "12. Y-Normal (29개)": "YNOR",
}


# =============================================================================
# Data loading
# =============================================================================
def preprocess_channel(x, y, grid, deriv_order):
    y_sg = savgol_filter(y, SG_WINDOW, SG_POLY, deriv=deriv_order)
    x_tr, y_tr = trim_spectrum(x.copy(), y_sg, region=(400, 2200))
    if deriv_order == 0:
        y_tr = baseline_correction(y_tr, window=101)
    y_tr = normalize_spectrum(y_tr, method="snv")
    return resample(x_tr, y_tr, grid)


def load_device_data(data_dir, folder_map, grid, pattern="*.CSV", max_rep=None):
    all_X, raw_ch0, meta_rows, failed = [], [], [], 0
    for folder_name, group in folder_map.items():
        folder = data_dir / folder_name
        if not folder.is_dir():
            continue
        files = find_spectra(folder, pattern=pattern, recursive=False)
        if not files:
            for p in ["*.txt", "*.csv", "*.CSV"]:
                files = find_spectra(folder, pattern=p, recursive=False)
                if files: break
        files = [f for f in files if "_ave" not in f.stem.lower() and "zone.identifier" not in f.name.lower()]
        for fp in files:
            try:
                sid = parse_filename(fp, fallback_group=group)
                if max_rep and sid.group in max_rep and sid.replicate > max_rep[sid.group]:
                    continue
                x, y = read_spectrum(fp)
                ch0 = preprocess_channel(x, y, grid, 0)
                ch1 = preprocess_channel(x, y, grid, 1)
                ch2 = preprocess_channel(x, y, grid, 2)
                all_X.append(np.concatenate([ch0, ch1, ch2]))
                raw_ch0.append(ch0)
                meta_rows.append({"group": sid.group, "sample_id": sid.sample_id, "replicate": sid.replicate})
            except Exception:
                failed += 1
    X = np.stack(all_X, axis=0).astype(np.float32)
    raw_spectra = np.stack(raw_ch0, axis=0).astype(np.float32)
    df = pd.DataFrame(meta_rows)
    # Prefix sample_ids for Y-subgroups to avoid collision before merging
    df["sample_id"] = df["sample_id"].astype(str)
    ypan_mask = df["group"] == "YPAN"
    df.loc[ypan_mask, "sample_id"] = "Y" + df.loc[ypan_mask, "sample_id"]
    ynor_mask = df["group"] == "YNOR"
    df.loc[ynor_mask, "sample_id"] = "Y" + df.loc[ynor_mask, "sample_id"]
    df["group"] = df["group"].replace({"YPAN": "PAN", "CPAN": "PAN", "YNOR": "NOR"})
    valid = set(CANCER_TYPES) | set(NON_CANCER_GROUPS)
    mask = df["group"].isin(valid).values
    logger.info(f"  {data_dir.name}: {mask.sum()} spectra loaded ({failed} failed)")
    return X[mask], raw_spectra[mask], df[mask].reset_index(drop=True)


# =============================================================================
# Train LR + collect OOF predictions
# =============================================================================
def train_oof(X, df_meta, n_splits=5):
    """Train LR (raw+d1+d2 flatten), return OOF predictions + trained model coefficients."""
    groups_arr = df_meta["group"].values
    sample_ids = (df_meta["group"] + "_" + df_meta["sample_id"].astype(str)).values
    binary_labels = np.array([1 if g in CANCER_TYPES else 0 for g in groups_arr])
    ct_map = {ct: i for i, ct in enumerate(CANCER_TYPES)}
    cancer_type_labels = np.array([ct_map.get(g, -1) for g in groups_arr])

    n = len(X)
    oof_s1 = np.full(n, np.nan)
    oof_s2 = np.full((n, len(CANCER_TYPES)), np.nan)
    oof_s2_pred = np.full(n, -1, dtype=int)
    all_s1_coefs = []
    all_s2_coefs = []

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X, binary_labels, sample_ids)):
        # S1
        s1 = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=500, solver="lbfgs", class_weight="balanced"))
        s1.fit(X[tr_idx], binary_labels[tr_idx])
        oof_s1[val_idx] = s1.predict_proba(X[val_idx])[:, 1]
        all_s1_coefs.append(s1[1].coef_[0])

        # S2
        cancer_mask = binary_labels[tr_idx] == 1
        if cancer_mask.sum() > 10:
            s2 = make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, max_iter=500, solver="lbfgs", class_weight="balanced", multi_class="multinomial"))
            s2.fit(X[tr_idx][cancer_mask], cancer_type_labels[tr_idx][cancer_mask])
            raw_prob = s2.predict_proba(X[val_idx])
            for i, cls in enumerate(s2.classes_):
                if cls < len(CANCER_TYPES):
                    oof_s2[val_idx, cls] = raw_prob[:, i]
            all_s2_coefs.append(s2[1].coef_)

        cancer_val = binary_labels[val_idx] == 1
        if cancer_val.sum() > 0:
            oof_s2_pred[val_idx[cancer_val]] = oof_s2[val_idx[cancer_val]].argmax(axis=1)

    return {
        "oof_s1": oof_s1, "oof_s2": oof_s2, "oof_s2_pred": oof_s2_pred,
        "binary_labels": binary_labels, "cancer_type_labels": cancer_type_labels,
        "groups_arr": groups_arr, "sample_ids": sample_ids,
        "s1_coefs": np.mean(all_s1_coefs, axis=0),
        "s2_coefs": np.mean(all_s2_coefs, axis=0) if all_s2_coefs else None,
    }


# =============================================================================
# Figure 1: Confusion Matrices (side by side)
# =============================================================================
def _aggregate_to_patient(res):
    """Aggregate spectrum-level predictions to patient-level via majority vote / mean prob."""
    from collections import Counter
    sample_ids = res["sample_ids"]
    unique_ids = []
    seen = {}
    for i, sid in enumerate(sample_ids):
        if sid not in seen:
            seen[sid] = len(unique_ids)
            unique_ids.append(sid)

    n_patients = len(unique_ids)
    pat_binary = np.zeros(n_patients)
    pat_ct = np.full(n_patients, -1, dtype=int)
    pat_s1 = np.zeros(n_patients)
    pat_s2_pred = np.full(n_patients, -1, dtype=int)
    pat_groups = np.empty(n_patients, dtype=object)

    for pid_idx, pid in enumerate(unique_ids):
        mask = sample_ids == pid
        pat_binary[pid_idx] = res["binary_labels"][mask][0]
        pat_ct[pid_idx] = res["cancer_type_labels"][mask][0]
        pat_groups[pid_idx] = res["groups_arr"][mask][0]
        pat_s1[pid_idx] = res["oof_s1"][mask].mean()

        preds = res["oof_s2_pred"][mask]
        valid_preds = preds[preds >= 0]
        if len(valid_preds) > 0:
            pat_s2_pred[pid_idx] = Counter(valid_preds).most_common(1)[0][0]

    return {
        "binary_labels": pat_binary, "cancer_type_labels": pat_ct,
        "oof_s1": pat_s1, "oof_s2_pred": pat_s2_pred,
        "groups_arr": pat_groups, "n_patients": n_patients,
    }


def plot_confusion_matrices(res_t, res_m):
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    for ax, res, title, cmap_color in [
        (axes[0], res_t, "Thermo Fisher", "Greens"),
        (axes[1], res_m, "Medical Instrument", "Blues"),
    ]:
        # Aggregate to patient level
        pat = _aggregate_to_patient(res)

        # Cancer type confusion (cancer patients only)
        cancer_mask = pat["binary_labels"] == 1
        valid = cancer_mask & (pat["oof_s2_pred"] >= 0)
        y_true = pat["cancer_type_labels"][valid]
        y_pred = pat["oof_s2_pred"][valid]

        present = sorted(set(y_true) | set(y_pred))
        labels = [CANCER_TYPES[i] for i in present if i < len(CANCER_TYPES)]
        cm = confusion_matrix(y_true, y_pred, labels=present)

        # Normalize by row
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)

        im = ax.imshow(cm_norm, cmap=cmap_color, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=16, fontweight="bold")
        ax.set_yticklabels(labels, fontsize=16, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=18)
        ax.set_ylabel("Actual", fontsize=18)

        # Annotate with patient counts
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = cm_norm[i, j]
                count = cm[i, j]  # this is now patient count
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.0%}\n({count}명)", ha="center", va="center",
                        fontsize=14, fontweight="bold" if i == j else "normal", color=color)

        # AUC/F1 annotation
        auc = roc_auc_score(pat["binary_labels"], pat["oof_s1"])
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        n_pat = pat["n_patients"]
        ax.set_title(f"{title} (n={n_pat}명)\nDet AUC={auc:.3f}, Type F1={f1:.3f}",
                     fontsize=20, fontweight="bold", pad=16)

    plt.suptitle("Cancer Type Confusion Matrix (LR raw+d1+d2, 5-fold CV)", fontsize=22, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved confusion_matrices.png")


# =============================================================================
# Figure 2: Feature Importance (LR coef as SHAP-like)
# =============================================================================
def plot_feature_importance(res_t, res_m, grid):
    fig, axes = plt.subplots(2, 1, figsize=(24, 12), sharex=True)
    n_wn = len(grid)

    for ax, res, title, color in [
        (axes[0], res_t, "Thermo Fisher", "#0f766e"),
        (axes[1], res_m, "Medical Instrument", "#1d4ed8"),
    ]:
        coef = res["s1_coefs"]
        # Split into 3 channels
        coef_raw = np.abs(coef[:n_wn])
        coef_d1 = np.abs(coef[n_wn:2*n_wn])
        coef_d2 = np.abs(coef[2*n_wn:3*n_wn])

        ax.fill_between(grid, 0, coef_raw, alpha=0.4, color=color, label="Raw")
        ax.fill_between(grid, 0, coef_d1, alpha=0.3, color="#E91E63", label="d1")
        ax.fill_between(grid, 0, coef_d2, alpha=0.2, color="#FF9800", label="d2")

        # Top peaks annotation
        combined = coef_raw + coef_d1 + coef_d2
        top_idx = np.argsort(combined)[-8:]
        for idx in top_idx:
            wn = grid[idx]
            val = combined[idx]
            ax.annotate(f"{wn:.0f}", xy=(wn, val), fontsize=12,
                       color="black", ha="center", va="bottom",
                       arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

        ax.set_ylabel("|LR Coefficient|", fontsize=16)
        ax.set_title(f"{title} — Stage 1 Feature Importance (Binary Detection)", fontsize=18, fontweight="bold")
        ax.legend(loc="upper right", fontsize=14)
        ax.tick_params(labelsize=13)
        ax.grid(True, alpha=0.2)

    axes[1].set_xlabel("Raman Shift (cm⁻¹)", fontsize=16)
    plt.suptitle("Feature Importance: LR Coefficient Magnitude (|w|) by Channel",
                 fontsize=20, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved feature_importance.png")

    # Stage 2: per-cancer-type detailed panels (each cancer gets its own subplot)
    if res_t["s2_coefs"] is not None and res_m["s2_coefs"] is not None:
        n_types = min(len(CANCER_TYPES), res_t["s2_coefs"].shape[0], res_m["s2_coefs"].shape[0])
        fig, axes = plt.subplots(n_types, 2, figsize=(26, 4.5 * n_types), sharex=True)

        cancer_colors = ["#E91E63", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4", "#795548"]

        for ci in range(n_types):
            ct_name = CANCER_TYPES[ci]
            for col, (res, dev_name, dev_color) in enumerate([
                (res_t, "Thermo", "#0f766e"),
                (res_m, "Medical", "#1d4ed8"),
            ]):
                ax = axes[ci, col]
                coefs = res["s2_coefs"]
                # raw, d1, d2 channels
                c_raw = np.abs(coefs[ci, :n_wn])
                c_d1 = np.abs(coefs[ci, n_wn:2*n_wn]) if coefs.shape[1] >= 2*n_wn else np.zeros(n_wn)
                c_d2 = np.abs(coefs[ci, 2*n_wn:3*n_wn]) if coefs.shape[1] >= 3*n_wn else np.zeros(n_wn)

                ax.fill_between(grid, 0, c_raw, alpha=0.5, color=cancer_colors[ci], label="raw")
                ax.fill_between(grid, 0, c_d1, alpha=0.3, color="#E91E63", label="d1")
                ax.fill_between(grid, 0, c_d2, alpha=0.2, color="#FF9800", label="d2")

                # Top 5 peaks
                combined = c_raw + c_d1 + c_d2
                top5 = np.argsort(combined)[-5:]
                for idx in top5:
                    ax.annotate(f"{grid[idx]:.0f}", xy=(grid[idx], combined[idx]),
                               fontsize=11, ha="center", va="bottom",
                               arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

                ax.grid(True, alpha=0.15)
                ax.tick_params(labelsize=12)
                if ci == 0:
                    ax.set_title(dev_name, fontsize=18, fontweight="bold")
                if col == 0:
                    ax.set_ylabel(f"{ct_name}\n|Coef|", fontsize=14, fontweight="bold")
                if ci == 0:
                    ax.legend(fontsize=12, loc="upper right", ncol=3)
                if ci == n_types - 1:
                    ax.set_xlabel("Raman Shift (cm⁻¹)", fontsize=16)

        plt.suptitle("암종별 Feature Importance — LR Coefficient (raw/d1/d2 채널)",
                     fontsize=22, fontweight="bold", y=1.01)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "feature_importance_s2.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved feature_importance_s2.png")


# =============================================================================
# Figure 3: Mean Spectra per Group (Thermo vs Medical overlay)
# =============================================================================
def _plot_spectra_page(groups_subset, raw_t, df_t, raw_m, df_m, grid, title, filename):
    """Helper: plot a page of mean spectra subplots for given groups."""
    n = len(groups_subset)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(26, 7 * rows), sharex=True)
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, g in enumerate(groups_subset):
        ax = axes[i]
        mask_t = df_t["group"] == g
        mean_t = raw_t[mask_t.values].mean(axis=0)
        ax.plot(grid, mean_t, color="#0f766e", linewidth=1.5, alpha=0.85, label="Thermo")

        mask_m = df_m["group"] == g
        mean_m = raw_m[mask_m.values].mean(axis=0)
        ax.plot(grid, mean_m, color="#1d4ed8", linewidth=1.5, alpha=0.85, label="Medical")

        std_t = raw_t[mask_t.values].std(axis=0)
        ax.fill_between(grid, mean_t - std_t, mean_t + std_t, color="#0f766e", alpha=0.1)
        std_m = raw_m[mask_m.values].std(axis=0)
        ax.fill_between(grid, mean_m - std_m, mean_m + std_m, color="#1d4ed8", alpha=0.1)

        corr = np.corrcoef(mean_t, mean_m)[0, 1]
        n_pat_t = df_t.loc[mask_t, "sample_id"].nunique()
        n_pat_m = df_m.loc[mask_m, "sample_id"].nunique()
        category = "cancer" if g in CANCER_TYPES else ("control" if g == "NOR" else "non-cancer")
        cat_color = "#E53935" if category == "cancer" else "#43A047" if category == "non-cancer" else "#5C6BC0"

        ax.set_title(f"{g} (T:{n_pat_t}, M:{n_pat_m}명)  r={corr:.3f}",
                     fontsize=20, fontweight="bold", color=cat_color)
        ax.grid(True, alpha=0.15)
        ax.tick_params(labelsize=14)
        if i == 0:
            ax.legend(fontsize=16, loc="upper right")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.text(0.5, -0.01, "Raman Shift (cm⁻¹)", ha="center", fontsize=20)
    fig.text(-0.01, 0.5, "SNV Intensity", va="center", rotation="vertical", fontsize=20)
    plt.suptitle(title, fontsize=24, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {filename}")


def plot_mean_spectra(raw_t, df_t, raw_m, df_m, grid):
    groups = [g for g in ALL_GROUPS_ORDER if g in df_t["group"].values and g in df_m["group"].values]

    # Split into non-cancer and cancer (cancer split into 2 pages: 4+3)
    non_cancer = [g for g in groups if g not in CANCER_TYPES]
    cancer = [g for g in groups if g in CANCER_TYPES]

    _plot_spectra_page(non_cancer, raw_t, df_t, raw_m, df_m, grid,
                       "Mean Spectra ± 1σ — 비암 그룹 (NOR, DIA, HBP, H.D.)",
                       "mean_spectra_noncancer.png")
    _plot_spectra_page(cancer[:4], raw_t, df_t, raw_m, df_m, grid,
                       "Mean Spectra ± 1σ — 암 그룹 1/2 (PRO, BRE, OVA, LUN)",
                       "mean_spectra_cancer1.png")
    _plot_spectra_page(cancer[4:], raw_t, df_t, raw_m, df_m, grid,
                       "Mean Spectra ± 1σ — 암 그룹 2/2 (CRC, PAN, BLC)",
                       "mean_spectra_cancer2.png")


# =============================================================================
# Figure 4: Binary detection — per-group probability distribution
# =============================================================================
def plot_probability_distribution(res_t, res_m):
    """환자 단위 mean probability로 boxplot + strip."""
    fig, axes = plt.subplots(1, 2, figsize=(26, 10))

    for ax, res, title, main_color in [
        (axes[0], res_t, "Thermo Fisher", "#0f766e"),
        (axes[1], res_m, "Medical Instrument", "#1d4ed8"),
    ]:
        # Aggregate to patient-level mean probability
        sample_ids = res["sample_ids"]
        unique_ids = sorted(set(sample_ids), key=lambda x: list(sample_ids).index(x))
        pat_probs, pat_groups = [], []
        for pid in unique_ids:
            mask = sample_ids == pid
            pat_probs.append(res["oof_s1"][mask].mean())
            pat_groups.append(res["groups_arr"][mask][0])
        pat_probs = np.array(pat_probs)
        pat_groups = np.array(pat_groups)

        groups = [g for g in ALL_GROUPS_ORDER if g in pat_groups]
        y_pos = np.arange(len(groups))

        box_data, colors, ylabels = [], [], []
        for g in groups:
            mask = pat_groups == g
            probs = pat_probs[mask]
            box_data.append(probs)
            colors.append("#E53935" if g in CANCER_TYPES else "#43A047")
            ylabels.append(f"{g} (n={mask.sum()})")

        # Horizontal boxplot
        bp = ax.boxplot(box_data, positions=y_pos, vert=False, widths=0.6,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(linewidth=1),
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        # Strip (jitter individual patients)
        for i, (data, color) in enumerate(zip(box_data, colors)):
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(data))
            ax.scatter(data, y_pos[i] + jitter, c=color, s=12, alpha=0.5, edgecolors="none", zorder=3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(ylabels, fontsize=16, fontweight="bold")
        ax.set_xlabel("암 예측 확률 (환자 단위 평균)", fontsize=16)
        ax.set_title(title, fontsize=20, fontweight="bold")
        ax.tick_params(axis='x', labelsize=14)
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1.2)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(axis="x", alpha=0.2)
        ax.invert_yaxis()

        # Annotate median
        for i, data in enumerate(box_data):
            med = np.median(data)
            ax.text(min(med + 0.03, 0.95), i, f"med={med:.3f}", va="center", fontsize=13, color="#333")

    plt.suptitle("질환군별 암 예측 확률 분포 (환자 단위, LR raw+d1+d2, 5-fold OOF)",
                 fontsize=22, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "probability_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved probability_distribution.png")


# =============================================================================
# Main
# =============================================================================
def main():
    logger.info("=" * 64)
    logger.info("  2-Device Visualization Generator")
    logger.info("=" * 64)

    grid = np.linspace(402.0, 2198.0, 933)

    # Load both instruments
    logger.info("\n  Loading Thermo...")
    X_t, raw_t, df_t = load_device_data(THERMO_DIR, THERMO_MAP, grid, "*.CSV")
    logger.info("  Loading Medical...")
    X_m, raw_m, df_m = load_device_data(MEDICAL_DIR, MEDICAL_MAP, grid, "*.txt", max_rep={"NOR": 6})

    # Train LR + OOF predictions
    logger.info("\n  Training LR (Thermo)...")
    res_t = train_oof(X_t, df_t)
    logger.info("  Training LR (Medical)...")
    res_m = train_oof(X_m, df_m)

    # Generate figures
    logger.info("\n  Generating figures...")
    plot_confusion_matrices(res_t, res_m)
    plot_feature_importance(res_t, res_m, grid)
    plot_mean_spectra(raw_t, df_t, raw_m, df_m, grid)
    plot_probability_distribution(res_t, res_m)

    logger.info(f"\n  All figures saved to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
