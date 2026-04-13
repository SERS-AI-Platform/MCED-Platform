"""
Phase 6c — Batch-Effect Leakage Investigation

Why this exists
---------------
06_preprocessing_validation reported raw L0 (NO preprocessing at all) hitting
AUC 0.985 on the hard 7-cancer screening task. This is suspicious — if raw
intensity alone discriminates cancer vs control, the model is learning a
non-biological signal (measurement day, instrument batch, protocol, etc.)
rather than spectral shape.

This script tests the hypothesis directly:

    Test 1 — Scalar features only:
        For each subject build a 4-feature vector:
            [mean intensity, std, max, total area]
        Run LR 5-fold CV on these alone.
        If AUC ≫ 0.5, intensity statistics carry the label → batch leakage.

    Test 2 — Per-group intensity distribution:
        Print mean intensity per group + boxplot. If groups are non-overlapping
        in absolute intensity, that's the smoking gun.

    Test 3 — Shape-only baseline:
        Apply SNV (which destroys absolute intensity but keeps shape).
        Compare AUC vs raw and vs scalar-only.

    Test 4 — File modification time grouping:
        If file mtime correlates with group, measurement day batch is the
        likely cause.

Output
------
results/preprocessing_validation/batch_leakage_report.md
results/preprocessing_validation/batch_leakage_stats.csv
results/preprocessing_validation/batch_leakage_intensity_box.png
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.config import load_config, RAW_DATA_DIR, RESULTS_DIR
from src.sers import read_spectrum, parse_filename
from src.sers.io import find_spectra
from src.sers.preprocessing import snv

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("batch_leak")

OUT_DIR = Path(RESULTS_DIR) / "preprocessing_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Same hard subset as 06
HARD_GROUPS = {
    "NOR": 60, "DIA": 60, "HBP": 60, "H.D.": 60,
    "PRO": 60, "BRE": 30, "OVA": 60, "LUN": 60, "CRC": 60,
    "CPAN": 60, "YPAN": 30, "BLC": 60,
}
CANCER_LABELS = {"PRO", "BRE", "OVA", "LUN", "CRC", "CPAN", "YPAN", "BLC"}


def load_subset(config) -> tuple[dict, list]:
    """Return (raw_dict, file_records). Records carry mtime + folder."""
    data_dir = Path(RAW_DATA_DIR)
    raw, records = {}, []
    for fp in find_spectra(data_dir, pattern="*.csv"):
        try:
            spec_id = parse_filename(
                fp,
                fallback_group=config.folder_to_group.get(fp.parent.name, "UNK"),
            )
        except Exception:
            continue
        if spec_id.group not in HARD_GROUPS:
            continue
        x, y = read_spectrum(fp)
        key = (spec_id.group, spec_id.sample_id, spec_id.replicate)
        raw[key] = (x, y)
        records.append({
            "group": spec_id.group, "sample_id": spec_id.sample_id,
            "replicate": spec_id.replicate, "folder": fp.parent.name,
            "mtime": fp.stat().st_mtime,
        })

    # Cap per group
    keep = set()
    for grp, n_max in HARD_GROUPS.items():
        sids = sorted({k[1] for k in raw if k[0] == grp})[:n_max]
        for k in raw:
            if k[0] == grp and k[1] in sids:
                keep.add(k)
    raw = {k: raw[k] for k in keep}
    records = [r for r in records
               if (r["group"], r["sample_id"], r["replicate"]) in keep]

    n_samples = len({(g, s) for g, s, _ in raw})
    logger.info(f"Subset: {len(raw)} spectra, {n_samples} subjects")
    return raw, records


def per_subject_aggregate(raw: dict) -> pd.DataFrame:
    """Build subject-level scalar features + mean spectrum on each subject."""
    subj = {}
    for (grp, sid, rep), (x, y) in raw.items():
        subj.setdefault((grp, sid), []).append(y)
    rows = []
    for (grp, sid), reps in subj.items():
        # Replicates may have slightly different lengths — interp to common len.
        min_len = min(len(r) for r in reps)
        stacked = np.vstack([r[:min_len] for r in reps])
        m = stacked.mean(axis=0)
        rows.append({
            "group": grp, "sample_id": sid,
            "label_cancer": int(grp in CANCER_LABELS),
            "mean_y": float(m.mean()),
            "std_y": float(m.std()),
            "max_y": float(m.max()),
            "min_y": float(m.min()),
            "area_y": float(m.sum()),
            "_mean_spec": m,
        })
    return pd.DataFrame(rows)


def lr_auc(X: np.ndarray, y: np.ndarray, label: str) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
    auc = float(np.mean(aucs))
    logger.info(f"  {label:35s} AUC = {auc:.4f}")
    return auc


def main():
    config = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    logger.info("[1] Loading hard subset ...")
    raw, records = load_subset(config)

    logger.info("[2] Subject aggregates ...")
    df = per_subject_aggregate(raw)

    # ── Test 1: scalar features only ──
    logger.info("\n[Test 1] LR on 4 scalar intensity features")
    X_scalar = df[["mean_y", "std_y", "max_y", "area_y"]].values
    y_label = df["label_cancer"].values
    auc_scalar = lr_auc(X_scalar, y_label, "scalar (mean/std/max/area)")
    auc_mean_only = lr_auc(df[["mean_y"]].values, y_label, "mean intensity ONLY")
    auc_max_only = lr_auc(df[["max_y"]].values, y_label, "max intensity ONLY")

    # ── Test 2: per-group intensity stats ──
    logger.info("\n[Test 2] Per-group intensity (mean of subject means)")
    grp_stats = df.groupby("group").agg(
        n=("sample_id", "count"),
        mean_y=("mean_y", "mean"),
        std_y=("mean_y", "std"),
        max_y=("max_y", "mean"),
        area_y=("area_y", "mean"),
    ).round(2)
    logger.info("\n" + grp_stats.to_string())

    # ── Test 3: shape-only (SNV per subject mean) ──
    logger.info("\n[Test 3] LR on full mean spectrum, raw vs SNV-normalized")
    min_len = min(len(s) for s in df["_mean_spec"])
    X_raw = np.vstack([s[:min_len] for s in df["_mean_spec"]])
    X_snv = np.vstack([snv(s[:min_len]) for s in df["_mean_spec"]])
    auc_raw_full = lr_auc(X_raw, y_label, "full spectrum (raw)")
    auc_snv_full = lr_auc(X_snv, y_label, "full spectrum (SNV)")

    # ── Test 4: mtime by group ──
    logger.info("\n[Test 4] File mtime distribution by group")
    rec_df = pd.DataFrame(records)
    rec_df["mtime_dt"] = pd.to_datetime(rec_df["mtime"], unit="s")
    mtime_summary = rec_df.groupby("group")["mtime_dt"].agg(["min", "max", "count"])
    logger.info("\n" + mtime_summary.to_string())

    # ── Save artifacts ──
    grp_stats.to_csv(OUT_DIR / "batch_leakage_stats.csv")
    rec_df[["group", "folder", "mtime_dt"]].to_csv(
        OUT_DIR / "batch_leakage_mtimes.csv", index=False
    )

    # Boxplot of mean intensity per group
    fig, ax = plt.subplots(figsize=(10, 5))
    groups_ordered = ["NOR", "DIA", "HBP", "H.D.", "PRO", "BRE", "OVA", "LUN",
                      "CRC", "CPAN", "YPAN", "BLC"]
    data = [df[df["group"] == g]["mean_y"].values for g in groups_ordered
            if g in df["group"].values]
    labels = [g for g in groups_ordered if g in df["group"].values]
    colors = ["#5C6BC0" if g == "NOR"
              else "#43A047" if g in ("DIA", "HBP", "H.D.")
              else "#E53935" for g in labels]
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel("Subject-mean raw intensity")
    ax.set_title(
        f"Raw mean intensity by group  |  scalar AUC = {auc_scalar:.3f}\n"
        f"(blue=control, green=non-cancer, red=cancer)",
        fontsize=10,
    )
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "batch_leakage_intensity_box.png", dpi=130)
    logger.info(f"  → batch_leakage_intensity_box.png")

    # ── Report ──
    lines = []
    lines.append("# Batch-effect Leakage 조사 (Phase 6c)\n\n")
    lines.append("## 가설\n")
    lines.append("Raw L0가 hard task에서 AUC 0.98 → 절대 intensity / batch가 그룹 라벨과 상관되어 누설되고 있을 가능성.\n\n")
    lines.append("## Test 1 — Scalar features only (4 numbers per subject)\n\n")
    lines.append("| Feature | AUC |\n|---|---|\n")
    lines.append(f"| mean+std+max+area | **{auc_scalar:.4f}** |\n")
    lines.append(f"| mean intensity ONLY | {auc_mean_only:.4f} |\n")
    lines.append(f"| max intensity ONLY | {auc_max_only:.4f} |\n\n")
    lines.append("→ 이 값들이 0.5 근처면 누설 없음. 0.8+ 이면 누설 강함.\n\n")
    lines.append("## Test 2 — Per-group raw intensity\n\n")
    lines.append("```\n" + grp_stats.to_string() + "\n```\n\n")
    lines.append("## Test 3 — Full spectrum: raw vs SNV-normalized\n\n")
    lines.append(f"- Raw full spectrum: AUC **{auc_raw_full:.4f}**\n")
    lines.append(f"- SNV-normalized:    AUC **{auc_snv_full:.4f}**\n\n")
    lines.append("→ SNV는 절대 intensity를 제거하므로, raw vs SNV 격차 = intensity가 기여한 분량.\n\n")
    lines.append("## Test 4 — File mtime by group\n\n")
    lines.append("```\n" + mtime_summary.to_string() + "\n```\n\n")
    lines.append("→ 그룹별 mtime 범위가 분리되어 있으면 측정일 batch가 confounder.\n\n")
    lines.append("## 결론\n")
    if auc_scalar >= 0.85:
        lines.append(f"⚠ **누설 확인.** 4개 scalar feature만으로 AUC {auc_scalar:.3f} → 모델이 spectral shape이 아닌 absolute intensity를 학습 가능. ")
        lines.append("Raw L0의 0.98 AUC는 신뢰할 수 없음. Production에서는 SNV 같은 intensity-removing 정규화 필수.\n")
    elif auc_scalar >= 0.7:
        lines.append(f"⚠ **부분 누설.** scalar AUC {auc_scalar:.3f} — 절대 intensity가 일부 신호 제공. 정규화 권장.\n")
    else:
        lines.append(f"✓ **누설 없음/약함.** scalar AUC {auc_scalar:.3f} — raw L0의 고AUC는 다른 원인.\n")

    (OUT_DIR / "batch_leakage_report.md").write_text("".join(lines), encoding="utf-8")
    logger.info(f"  → batch_leakage_report.md")


if __name__ == "__main__":
    main()
