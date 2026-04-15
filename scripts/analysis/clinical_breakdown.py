"""
Clinical Analysis:
    1. Per-cancer sensitivity breakdown (which cancers are easy/hard?)
    2. LUN cancer stage analysis (can SERS detect early-stage?)

Uses the existing best model (LR on SNV spectra) and analyzes
per-sample predictions from 5-fold CV.

Usage:
    python models/run_clinical_analysis.py
"""

from __future__ import annotations

import sys
import logging
import json
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, confusion_matrix,
    classification_report, roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.train import (
    load_processed_spectra, get_feature_columns,
    apply_class_selection, resolve_aliases, aggregate_replicates,
    create_labels, ModelConfig,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import yaml
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

CANCER_TYPES = ["PRO", "LUN", "CRC", "CPAN", "OVA"]
NON_CANCER = ["NOR", "DIA", "HBP", "H.D."]
CANCER_LABELS = {0: "PRO", 1: "LUN", 2: "CRC", 3: "CPAN", 4: "OVA"}
CANCER_COLORS = {"PRO": "#E53935", "LUN": "#1976D2", "CRC": "#43A047",
                 "CPAN": "#FF9800", "OVA": "#9C27B0"}


def parse_lung_stage(s):
    """Parse LUN stage string to major stage (I/II/III/IV)."""
    if pd.isna(s):
        return None
    s = str(s).upper().strip()
    s = s.replace("STAGE ", "").replace("STGAE ", "").replace("STAGEI ", "")
    if s.startswith("IA") or s.startswith("1A") or s == "I":
        return "I"
    elif s.startswith("IB") or s.startswith("1B"):
        return "I"
    elif s.startswith("IIA") or s.startswith("IIB") or s.startswith("2"):
        return "II"
    elif s.startswith("IIIA") or s.startswith("IIIB") or s.startswith("LLLA") or s.startswith("3"):
        return "III"
    elif s.startswith("IV") or s.startswith("4"):
        return "IV"
    return None


def run_cv_collect_predictions(X, bl, ctl, sample_ids, mc):
    """Run 5-fold CV and collect per-sample predictions."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=mc.random_state)
    n = len(X)
    n_ct = mc.n_cancer_types

    val_bp = np.full(n, np.nan)
    val_cp = np.full((n, n_ct), 1.0 / n_ct)

    for fold_i, (tr, va) in enumerate(cv.split(X, bl, sample_ids)):
        # Stage 1
        m1 = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=1000, solver="saga", class_weight="balanced",
            random_state=mc.random_state + fold_i))
        m1.fit(X[tr], bl[tr])
        val_bp[va] = m1.predict_proba(X[va])[:, 1]

        # Stage 2
        cancer_mask = ctl[tr] >= 0
        if cancer_mask.sum() > 0:
            present = np.unique(ctl[tr][cancer_mask])
            if len(present) > 1:
                local = np.searchsorted(present, ctl[tr][cancer_mask])
                m2 = make_pipeline(StandardScaler(), LogisticRegression(
                    C=1.0, max_iter=1000, solver="saga", class_weight="balanced",
                    random_state=mc.random_state + fold_i + 100))
                m2.fit(X[tr][cancer_mask], local)
                probs = m2.predict_proba(X[va])
                for j, cls in enumerate(present):
                    val_cp[va, cls] = probs[:, j]

    return val_bp, val_cp


def main():
    out_dir = PROJECT_ROOT / "results" / "training" / "clinical_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(out_dir / "clinical_analysis.log", mode="w", encoding="utf-8")],
    )

    t0 = datetime.now()
    logger.info("=" * 64)
    logger.info("  Clinical Analysis: Per-cancer & Stage Analysis")
    logger.info("=" * 64)

    # Load data
    with open("config/config.yaml", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    df = load_processed_spectra()
    feat_cols = get_feature_columns(df)
    mc = ModelConfig.from_pipeline_config(raw_cfg, n_spectral_features=len(feat_cols))
    mc = apply_class_selection(mc, cancer_types=CANCER_TYPES, non_cancer_groups=NON_CANCER)
    df = resolve_aliases(df, mc)
    df_agg = aggregate_replicates(df, feat_cols, "none")

    X, bl, ctl, sample_ids, groups_arr = create_labels(df_agg, mc)

    # Keep group info for per-cancer analysis
    df_used = df_agg[df_agg["group"].isin(CANCER_TYPES + NON_CANCER)].reset_index(drop=True)
    groups = df_used["group"].values

    logger.info(f"  Spectra: {len(X)} (cancer={bl.sum()}, non-cancer={(bl==0).sum()})")

    # Run CV
    logger.info("\n  Running 5-fold CV...")
    val_bp, val_cp = run_cv_collect_predictions(X, bl, ctl, sample_ids, mc)

    # ================================================================
    # Analysis 1: Per-cancer sensitivity breakdown
    # ================================================================
    logger.info("\n" + "=" * 64)
    logger.info("  Analysis 1: Per-Cancer Sensitivity Breakdown")
    logger.info("=" * 64)

    # Stage 1: binary detection per cancer type
    logger.info("\n  Stage 1: Cancer Detection (cancer vs non-cancer)")
    logger.info(f"  {'Group':<8s} {'N':>6s} {'Sensitivity':>12s} {'Mean Prob':>10s} {'Median':>8s}")
    logger.info("  " + "-" * 50)

    s1_per_cancer = {}
    for ct_idx, ct_name in CANCER_LABELS.items():
        mask = groups == ct_name
        if mask.sum() == 0:
            continue
        probs = val_bp[mask]
        preds = (probs > 0.5).astype(int)
        sensitivity = preds.mean()  # All should be cancer (label=1)
        s1_per_cancer[ct_name] = {
            "n": int(mask.sum()),
            "sensitivity": float(sensitivity),
            "mean_prob": float(probs.mean()),
            "median_prob": float(np.median(probs)),
            "min_prob": float(probs.min()),
            "max_prob": float(probs.max()),
        }
        logger.info(f"  {ct_name:<8s} {mask.sum():>6d} {sensitivity:>12.4f} "
                     f"{probs.mean():>10.4f} {np.median(probs):>8.4f}")

    # Non-cancer specificity per group
    logger.info(f"\n  {'Group':<8s} {'N':>6s} {'Specificity':>12s} {'Mean Prob':>10s}")
    logger.info("  " + "-" * 40)

    s1_per_control = {}
    for grp in NON_CANCER:
        mask = groups == grp
        if mask.sum() == 0:
            continue
        probs = val_bp[mask]
        preds = (probs > 0.5).astype(int)
        specificity = 1 - preds.mean()  # All should be non-cancer (label=0)
        s1_per_control[grp] = {
            "n": int(mask.sum()),
            "specificity": float(specificity),
            "mean_prob": float(probs.mean()),
        }
        logger.info(f"  {grp:<8s} {mask.sum():>6d} {specificity:>12.4f} {probs.mean():>10.4f}")

    # Stage 2: Cancer type classification
    logger.info("\n  Stage 2: Cancer Type Classification (among cancer samples)")

    cancer_mask = ctl >= 0
    cancer_pred = val_cp[cancer_mask].argmax(axis=1)
    cancer_true = ctl[cancer_mask]

    report = classification_report(cancer_true, cancer_pred,
                                    target_names=list(CANCER_LABELS.values()),
                                    output_dict=True, zero_division=0)
    logger.info(f"\n  {'Type':<8s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s} {'Support':>8s}")
    logger.info("  " + "-" * 45)
    s2_per_cancer = {}
    for ct_name in CANCER_LABELS.values():
        r = report[ct_name]
        s2_per_cancer[ct_name] = r
        logger.info(f"  {ct_name:<8s} {r['precision']:>10.4f} {r['recall']:>8.4f} "
                     f"{r['f1-score']:>8.4f} {r['support']:>8.0f}")

    # Confusion matrix
    cm = confusion_matrix(cancer_true, cancer_pred)
    logger.info(f"\n  Confusion Matrix (rows=true, cols=predicted):")
    header = "        " + "  ".join(f"{n:>6s}" for n in CANCER_LABELS.values())
    logger.info(f"  {header}")
    for i, ct_name in CANCER_LABELS.items():
        row = "  ".join(f"{cm[i, j]:>6d}" for j in range(len(CANCER_LABELS)))
        logger.info(f"  {ct_name:<6s}  {row}")

    # ================================================================
    # Analysis 2: LUN Stage Analysis
    # ================================================================
    logger.info("\n" + "=" * 64)
    logger.info("  Analysis 2: LUN Cancer Stage Analysis")
    logger.info("=" * 64)

    # Load clinical staging
    clin = pd.read_csv(PROJECT_ROOT / "data" / "clinical_data" / "standardized" / "all_clinical_standardized.csv")
    clin["disease_group"] = clin["disease_group"].replace({"PAN": "CPAN"})
    clin["major_stage"] = clin["stage"].apply(parse_lung_stage)

    # Build stage lookup
    stage_lookup = {}
    for _, row in clin[clin["disease_group"] == "LUN"].iterrows():
        pid = row["patient_id"]
        if row["major_stage"]:
            stage_lookup[pid] = row["major_stage"]

    # Match stages to spectra
    lun_mask = groups == "LUN"
    lun_indices = np.where(lun_mask)[0]
    lun_sids = df_used.iloc[lun_indices]["sample_id"].values
    lun_stages = []
    for sid in lun_sids:
        key = f"LUN {sid}"
        lun_stages.append(stage_lookup.get(key, None))

    lun_stages = np.array(lun_stages)
    lun_bp = val_bp[lun_indices]
    lun_cp = val_cp[lun_indices]
    lun_ctl = ctl[lun_indices]

    logger.info(f"  LUN spectra: {len(lun_indices)}")
    logger.info(f"  With stage info: {np.sum([s is not None for s in lun_stages])}")

    # Per-stage analysis
    stage_results = {}
    for stage in ["I", "II", "III", "IV"]:
        mask = np.array([s == stage for s in lun_stages])
        if mask.sum() == 0:
            continue

        probs = lun_bp[mask]
        preds = (probs > 0.5).astype(int)
        sensitivity = preds.mean()

        # Stage 2: LUN correct classification rate
        cp = lun_cp[mask]
        ct = lun_ctl[mask]
        s2_correct = (cp.argmax(axis=1) == ct).mean() if (ct >= 0).any() else 0

        stage_results[stage] = {
            "n_spectra": int(mask.sum()),
            "n_subjects": int(len(set(lun_sids[mask]))),
            "sensitivity": float(sensitivity),
            "mean_prob": float(probs.mean()),
            "median_prob": float(np.median(probs)),
            "s2_correct_rate": float(s2_correct),
        }
        logger.info(f"  Stage {stage}: n={mask.sum():4d} ({len(set(lun_sids[mask])):3d} subjects) | "
                     f"Sensitivity={sensitivity:.4f} | Mean prob={probs.mean():.4f} | "
                     f"S2 correct={s2_correct:.4f}")

    # Early vs Late
    early_mask = np.array([s in ("I", "II") for s in lun_stages])
    late_mask = np.array([s in ("III", "IV") for s in lun_stages])

    if early_mask.sum() > 0 and late_mask.sum() > 0:
        early_sens = (lun_bp[early_mask] > 0.5).mean()
        late_sens = (lun_bp[late_mask] > 0.5).mean()
        logger.info(f"\n  Early (I+II): n={early_mask.sum()}, Sensitivity={early_sens:.4f}")
        logger.info(f"  Late (III+IV): n={late_mask.sum()}, Sensitivity={late_sens:.4f}")
        logger.info(f"  Delta: {late_sens - early_sens:+.4f}")

    # No-stage LUN (for completeness)
    no_stage = np.array([s is None for s in lun_stages])
    if no_stage.sum() > 0:
        ns_sens = (lun_bp[no_stage] > 0.5).mean()
        logger.info(f"  No stage info: n={no_stage.sum()}, Sensitivity={ns_sens:.4f}")

    # ================================================================
    # Operating point analysis
    # ================================================================
    logger.info("\n" + "=" * 64)
    logger.info("  Operating Point Analysis (Stage 1)")
    logger.info("=" * 64)

    fpr, tpr, thresholds = roc_curve(bl, val_bp)

    # Find key operating points
    for target_sens in [0.90, 0.95, 0.98, 0.99]:
        idx = np.argmin(np.abs(tpr - target_sens))
        spec = 1 - fpr[idx]
        logger.info(f"  Sensitivity={tpr[idx]:.3f} → Specificity={spec:.3f} (threshold={thresholds[idx]:.3f})")

    for target_spec in [0.90, 0.95, 0.98]:
        idx = np.argmin(np.abs((1 - fpr) - target_spec))
        logger.info(f"  Specificity={1-fpr[idx]:.3f} → Sensitivity={tpr[idx]:.3f} (threshold={thresholds[idx]:.3f})")

    # ================================================================
    # Plots
    # ================================================================
    logger.info("\n  Generating plots...")

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # (0,0) Per-cancer S1 sensitivity
    ax = fig.add_subplot(gs[0, 0])
    cancer_names = list(s1_per_cancer.keys())
    sens_vals = [s1_per_cancer[c]["sensitivity"] for c in cancer_names]
    colors = [CANCER_COLORS[c] for c in cancer_names]
    bars = ax.bar(cancer_names, sens_vals, color=colors, alpha=0.8)
    ax.set_ylabel("Sensitivity")
    ax.set_title("Stage 1: Cancer Detection Sensitivity\n(per cancer type)")
    ax.set_ylim(0.8, 1.02)
    ax.axhline(0.95, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="95%")
    ax.grid(True, alpha=0.3)
    for i, v in enumerate(sens_vals):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)

    # (0,1) Per-control specificity
    ax = fig.add_subplot(gs[0, 1])
    ctrl_names = list(s1_per_control.keys())
    spec_vals = [s1_per_control[c]["specificity"] for c in ctrl_names]
    ax.bar(ctrl_names, spec_vals, color=["#43A047", "#66BB6A", "#81C784", "#A5D6A7"], alpha=0.8)
    ax.set_ylabel("Specificity")
    ax.set_title("Stage 1: Non-cancer Specificity\n(per control group)")
    ax.set_ylim(0.7, 1.02)
    ax.axhline(0.95, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3)
    for i, v in enumerate(spec_vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    # (0,2) Stage 2 per-cancer F1
    ax = fig.add_subplot(gs[0, 2])
    s2_names = list(s2_per_cancer.keys())
    s2_f1 = [s2_per_cancer[c]["f1-score"] for c in s2_names]
    s2_recall = [s2_per_cancer[c]["recall"] for c in s2_names]
    s2_prec = [s2_per_cancer[c]["precision"] for c in s2_names]
    x = np.arange(len(s2_names))
    w = 0.25
    ax.bar(x - w, s2_prec, w, label="Precision", color="#1976D2", alpha=0.7)
    ax.bar(x, s2_recall, w, label="Recall", color="#E53935", alpha=0.7)
    ax.bar(x + w, s2_f1, w, label="F1", color="#FF9800", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(s2_names)
    ax.set_ylabel("Score")
    ax.set_title("Stage 2: Cancer Type Classification\n(precision / recall / F1)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    # (1,0) Confusion matrix heatmap
    ax = fig.add_subplot(gs[1, 0])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(CANCER_LABELS)))
    ax.set_yticks(range(len(CANCER_LABELS)))
    ax.set_xticklabels(list(CANCER_LABELS.values()), fontsize=9)
    ax.set_yticklabels(list(CANCER_LABELS.values()), fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Stage 2: Confusion Matrix\n(row-normalized)")
    for i in range(len(CANCER_LABELS)):
        for j in range(len(CANCER_LABELS)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.0%})",
                    ha="center", va="center", fontsize=8, color=color)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # (1,1) LUN stage sensitivity
    ax = fig.add_subplot(gs[1, 1])
    if stage_results:
        stages = list(stage_results.keys())
        stage_sens = [stage_results[s]["sensitivity"] for s in stages]
        stage_n = [stage_results[s]["n_spectra"] for s in stages]
        stage_colors = ["#43A047", "#8BC34A", "#FF9800", "#E53935"][:len(stages)]
        bars = ax.bar(stages, stage_sens, color=stage_colors, alpha=0.8)
        ax.set_ylabel("Sensitivity")
        ax.set_title("LUN: Detection Sensitivity by Stage")
        ax.set_ylim(0.7, 1.05)
        ax.axhline(0.95, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="95%")
        ax.grid(True, alpha=0.3)
        for i, (v, n) in enumerate(zip(stage_sens, stage_n)):
            ax.text(i, v + 0.008, f"{v:.3f}\n(n={n})", ha="center", fontsize=9, fontweight="bold")
        ax.legend(fontsize=8)

    # (1,2) LUN stage probability distribution
    ax = fig.add_subplot(gs[1, 2])
    for stage, color in zip(["I", "II", "III", "IV"],
                             ["#43A047", "#8BC34A", "#FF9800", "#E53935"]):
        mask = np.array([s == stage for s in lun_stages])
        if mask.sum() > 0:
            probs = lun_bp[mask]
            ax.hist(probs, bins=20, alpha=0.5, color=color, label=f"Stage {stage} (n={mask.sum()})",
                    edgecolor="white", density=True)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Threshold")
    ax.set_xlabel("Cancer Probability (Stage 1)")
    ax.set_ylabel("Density")
    ax.set_title("LUN: Prediction Distribution by Stage")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (2,0) ROC curve with operating points
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(fpr, tpr, color="#1976D2", linewidth=2,
            label=f"LR (AUC={roc_auc_score(bl, val_bp):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
    # Mark operating points
    for target_sens, marker in [(0.95, "o"), (0.98, "s")]:
        idx = np.argmin(np.abs(tpr - target_sens))
        ax.plot(fpr[idx], tpr[idx], marker, color="#E53935", markersize=8,
                label=f"Sens={tpr[idx]:.2f}, Spec={1-fpr[idx]:.2f}")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Stage 1: ROC Curve with Operating Points")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # (2,1) Cancer probability box plot by group
    ax = fig.add_subplot(gs[2, 1])
    all_groups_order = NON_CANCER + CANCER_TYPES
    prob_data = []
    prob_labels = []
    for grp in all_groups_order:
        mask = groups == grp
        if mask.sum() > 0:
            prob_data.append(val_bp[mask])
            prob_labels.append(grp)
    bp_plot = ax.boxplot(prob_data, labels=prob_labels, patch_artist=True, widths=0.6)
    grp_colors = {"NOR": "#43A047", "DIA": "#66BB6A", "HBP": "#81C784", "H.D.": "#A5D6A7",
                  "PRO": "#E53935", "LUN": "#1976D2", "CRC": "#43A047", "CPAN": "#FF9800", "OVA": "#9C27B0"}
    # Use cancer red tones for cancer groups in this plot
    box_colors = {"NOR": "#81C784", "DIA": "#81C784", "HBP": "#81C784", "H.D.": "#81C784",
                  "PRO": "#EF9A9A", "LUN": "#90CAF9", "CRC": "#A5D6A7", "CPAN": "#FFCC80", "OVA": "#CE93D8"}
    for i, grp in enumerate(prob_labels):
        bp_plot["boxes"][i].set_facecolor(box_colors.get(grp, "#999"))
        bp_plot["boxes"][i].set_alpha(0.7)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.axvspan(0.5, len(NON_CANCER) + 0.5, alpha=0.05, color="green")
    ax.axvspan(len(NON_CANCER) + 0.5, len(all_groups_order) + 0.5, alpha=0.05, color="red")
    ax.set_ylabel("Cancer Probability")
    ax.set_title("Stage 1: Prediction Distribution by Group")
    ax.grid(True, alpha=0.3)

    # (2,2) Summary text
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")

    # Build summary
    overall_auc = roc_auc_score(bl, val_bp)
    overall_sens = (val_bp[bl == 1] > 0.5).mean()
    overall_spec = (val_bp[bl == 0] <= 0.5).mean()

    best_cancer = max(s1_per_cancer.items(), key=lambda x: x[1]["sensitivity"])
    worst_cancer = min(s1_per_cancer.items(), key=lambda x: x[1]["sensitivity"])

    early_sens_val = stage_results.get("I", {}).get("sensitivity", 0)
    late_sens_val = stage_results.get("III", {}).get("sensitivity", 0) if "III" in stage_results else 0

    txt = (
        f"CLINICAL ANALYSIS SUMMARY\n"
        f"{'='*40}\n\n"
        f"Overall: AUC={overall_auc:.3f}\n"
        f"  Sens={overall_sens:.3f}, Spec={overall_spec:.3f}\n\n"
        f"Best detected:  {best_cancer[0]} ({best_cancer[1]['sensitivity']:.3f})\n"
        f"Worst detected: {worst_cancer[0]} ({worst_cancer[1]['sensitivity']:.3f})\n\n"
        f"LUN Early-Stage Detection:\n"
        f"  Stage I:  {stage_results.get('I', {}).get('sensitivity', 0):.3f} "
        f"(n={stage_results.get('I', {}).get('n_spectra', 0)})\n"
        f"  Stage II: {stage_results.get('II', {}).get('sensitivity', 0):.3f} "
        f"(n={stage_results.get('II', {}).get('n_spectra', 0)})\n"
    )
    if "III" in stage_results:
        txt += f"  Stage III: {stage_results['III']['sensitivity']:.3f} (n={stage_results['III']['n_spectra']})\n"

    ax.text(0.05, 0.95, txt, transform=ax.transAxes,
            fontsize=10, fontfamily="monospace", va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    plt.suptitle("Clinical Analysis: Per-Cancer Sensitivity & LUN Stage Detection",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.savefig(out_dir / "clinical_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out_dir / 'clinical_analysis.png'}")

    # Save JSON
    summary = {
        "experiment": "clinical_analysis",
        "timestamp": datetime.now().isoformat(),
        "overall": {"auc": overall_auc, "sensitivity": overall_sens, "specificity": overall_spec},
        "per_cancer_s1": s1_per_cancer,
        "per_control_s1": s1_per_control,
        "per_cancer_s2": {k: dict(v) for k, v in s2_per_cancer.items()},
        "confusion_matrix": cm.tolist(),
        "lung_stage": stage_results,
        "elapsed": str(datetime.now() - t0),
    }
    with open(out_dir / "clinical_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\n  Elapsed: {datetime.now() - t0}")
    logger.info(f"  Output: {out_dir}/")
    logger.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
