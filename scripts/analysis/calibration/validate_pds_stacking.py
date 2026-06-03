#!/usr/bin/env python3
"""
Validate PDS calibration with Stacking V2 production artifacts.

Runs the existing 10-base-model Stacking V2 pipeline on every paired sample
from both Thermo and Medical raw data, under four conditions:

  1. Thermo raw → Thermo model              (upper bound / positive control)
  2. Medical raw → Thermo model, no PDS     (current failure mode)
  3. Medical raw → Thermo model + PDS       (★ main test)
  4. Thermo raw (reference)                  (sanity: same as #1)

For each condition we compute:
  - Stage-1 ROC AUC (cancer vs non-cancer)
  - Stage-1 F1 (macro)
  - Stage-2 F1 (cancer-type, macro)
  - Per-group mean cancer probability

Uses the FIRST replicate of each paired sample (sample-level 1-rep proxy —
faster than full 5-rep patient aggregation; noted as a caveat).

Output → results/cross_instrument/calibration/stacking_validation/
"""
from __future__ import annotations
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.calibration.cross_instrument_calibration import (
    MEDICAL_FOLDER_MAP, NOR_MAX_REPLICATE, THERMO_FOLDER_MAP,
    CANCER_TYPES, GROUP_ALIASES, NON_CANCER_GROUPS,
)
from scripts.deployment.sers_predict import StackingPredictor
from src.sers.io import find_spectra, parse_filename

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

THERMO_DIR = PROJECT_ROOT / "data" / "raw_data"
MEDICAL_DIR = PROJECT_ROOT / "data" / "raw_data_medical"
OUT_DIR = PROJECT_ROOT / "results" / "cross_instrument" / "calibration" / "stacking_validation"


def collect_first_replicate_paths(data_dir: Path, folder_map: dict,
                                   pattern: str,
                                   read_from_subdir: str | None = None) -> pd.DataFrame:
    """Return a DataFrame with one row per sample: first replicate file."""
    rows = []
    for folder_name, group in folder_map.items():
        folder = data_dir / folder_name
        source = folder / read_from_subdir if read_from_subdir else folder
        if not source.is_dir():
            continue
        files = find_spectra(source, pattern=pattern, recursive=False)
        if not files and pattern == "*.csv":
            files = find_spectra(source, pattern="*.CSV", recursive=False)
        files = [f for f in files
                 if "_ave" not in f.stem.lower()
                 and "zone.identifier" not in f.name.lower()
                 and "multidata" not in f.stem.lower()]
        for fp in files:
            try:
                sid = parse_filename(fp, fallback_group=group)
                if sid.group == "NOR" and sid.replicate > NOR_MAX_REPLICATE:
                    continue
                rows.append({"group": sid.group, "sample_id": sid.sample_id,
                              "replicate": sid.replicate, "path": str(fp)})
            except Exception:
                pass
    df = pd.DataFrame(rows)
    # Keep only the smallest replicate per (group, sample_id)
    df = df.sort_values(["group", "sample_id", "replicate"])
    df = df.drop_duplicates(["group", "sample_id"], keep="first").reset_index(drop=True)
    return df


def alias_group(g: str) -> str:
    return GROUP_ALIASES.get(g, g)


def run_condition(predictor: StackingPredictor, paths: list, instrument: str,
                   condition_label: str) -> list:
    """Call predict_single on every path and collect results."""
    from tqdm import tqdm
    rows = []
    for p in tqdm(paths, desc=condition_label):
        r = predictor.predict_single(p["path"], instrument=instrument)
        rows.append({
            "condition": condition_label,
            "group_raw": p["group"],
            "group": alias_group(p["group"]),
            "sample_id": p["sample_id"],
            "replicate": p["replicate"],
            "status": r.get("status"),
            "cancer_probability": r.get("cancer_probability", float("nan")),
            "cancer_detected": r.get("cancer_detected"),
            "cancer_type_prediction": r.get("cancer_type_prediction"),
        })
    return rows


def score(df: pd.DataFrame, label: str) -> dict:
    from sklearn.metrics import f1_score, roc_auc_score
    valid_groups = set(CANCER_TYPES) | set(NON_CANCER_GROUPS)
    df = df[df["group"].isin(valid_groups) & df["status"].eq("ok")].copy()
    if df.empty:
        return {"label": label, "n": 0}
    y = df["group"].isin(CANCER_TYPES).astype(int).values
    p = df["cancer_probability"].values
    try:
        auc = roc_auc_score(y, p)
    except ValueError:
        auc = float("nan")
    pred = (p > 0.5).astype(int)
    f1 = f1_score(y, pred, average="macro")

    # Stage 2 F1 among cancer samples (using predicted cancer type)
    cancer_mask = df["group"].isin(CANCER_TYPES)
    s2_df = df[cancer_mask]
    if len(s2_df) and s2_df["cancer_type_prediction"].notna().any():
        s2_true = s2_df["group"].values
        s2_pred = s2_df["cancer_type_prediction"].values
        s2_f1 = f1_score(s2_true, s2_pred, average="macro", zero_division=0)
    else:
        s2_f1 = float("nan")

    per_group = (df.groupby("group")
                 .agg(n=("cancer_probability", "size"),
                      mean_prob=("cancer_probability", "mean"),
                      std_prob=("cancer_probability", "std"))
                 .reset_index())

    return {
        "label": label,
        "n": int(len(df)),
        "s1_auc": float(auc),
        "s1_f1": float(f1),
        "s2_f1": float(s2_f1),
        "n_cancer": int(y.sum()),
        "n_non_cancer": int((1 - y).sum()),
        "per_group": per_group.to_dict(orient="records"),
    }


def main():
    t0 = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 70)
    logger.info("  Stacking V2 × PDS validation")
    logger.info("=" * 70)

    logger.info("\n[1] Collecting first-replicate paths")
    df_t_paths = collect_first_replicate_paths(THERMO_DIR, THERMO_FOLDER_MAP, "*.CSV")
    df_m_paths = collect_first_replicate_paths(
        MEDICAL_DIR, MEDICAL_FOLDER_MAP, "*.txt", read_from_subdir="Background",
    )
    logger.info(f"  Thermo  samples: {len(df_t_paths)}")
    logger.info(f"  Medical samples: {len(df_m_paths)}")

    # Paired intersection on (group, sample_id)
    paired_keys = set(zip(df_t_paths["group"], df_t_paths["sample_id"])) & \
                  set(zip(df_m_paths["group"], df_m_paths["sample_id"]))
    logger.info(f"  Paired samples:  {len(paired_keys)}")

    df_t_paths = df_t_paths[
        df_t_paths.set_index(["group", "sample_id"]).index.isin(paired_keys)
    ].reset_index(drop=True)
    df_m_paths = df_m_paths[
        df_m_paths.set_index(["group", "sample_id"]).index.isin(paired_keys)
    ].reset_index(drop=True)

    logger.info("\n[2] Loading StackingPredictor (PDS integrated)")
    pred = StackingPredictor()
    assert pred.pds is not None, "PDS not loaded — run fit_pds_artifact.py first"

    logger.info("\n[3] Running 3 conditions")
    rows = []
    rows.extend(run_condition(pred, df_t_paths.to_dict("records"), "thermo",
                               "Thermo→Thermo (ceiling)"))
    rows.extend(run_condition(pred, df_m_paths.to_dict("records"), "thermo",
                               "Medical→Thermo no PDS (failure)"))
    rows.extend(run_condition(pred, df_m_paths.to_dict("records"), "medical",
                               "Medical→Thermo + PDS (fix)"))

    df_all = pd.DataFrame(rows)
    df_all.to_csv(OUT_DIR / "predictions.csv", index=False)

    logger.info("\n[4] Scoring")
    results = {}
    for label in df_all["condition"].unique():
        results[label] = score(df_all[df_all["condition"] == label], label)
        r = results[label]
        logger.info(f"  {label}")
        logger.info(f"    n={r['n']}  AUC={r['s1_auc']:.4f}  "
                    f"S1F1={r['s1_f1']:.4f}  S2F1={r['s2_f1']:.4f}")

    # Summary table
    summary = pd.DataFrame([{
        "condition": k,
        "n": v["n"],
        "s1_auc": v["s1_auc"],
        "s1_f1": v["s1_f1"],
        "s2_f1": v["s2_f1"],
    } for k, v in results.items()])
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": (datetime.now() - t0).total_seconds(),
        "n_paired": len(paired_keys),
        "conditions": results,
        "note": "Single-replicate proxy (first rep per sample). Full 5-rep "
                "patient aggregation would likely score slightly higher.",
    }
    with open(OUT_DIR / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)

    logger.info(f"\nDone in {(datetime.now()-t0).total_seconds():.1f}s")
    logger.info(f"Outputs → {OUT_DIR}")
    logger.info("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
