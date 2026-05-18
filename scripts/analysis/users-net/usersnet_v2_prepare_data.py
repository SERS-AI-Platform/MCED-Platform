"""Re-prepare data for uSERS-Net v2 figure set.

Fix:
- Use composite_id (original_group + sample_id) before alias resolution to avoid
  CPAN/YPAN and NOR/YNOR sample_id collisions during mean aggregation → yields N=1628.
- mean spectrum aggregation (not medoid).

Output: replaces results/runs/2026-05-11_usersnet_alpha0.8_ovr_roc/resnet18_ckpts/fold_inputs.npz
"""
from __future__ import annotations
import importlib.util as _iu
import logging
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

_spec = _iu.spec_from_file_location("legacy_train_resnet",
                                    PROJECT_ROOT / "scripts" / "training" / "_legacy" / "train_resnet.py")
_legacy = _iu.module_from_spec(_spec)
sys.modules["legacy_train_resnet"] = _legacy
_spec.loader.exec_module(_legacy)

from sers.models._legacy.resnet_v1.model import ModelConfig

CANCER_TYPES = ["PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "BLC"]
NON_CANCER = ["NOR", "DIA", "HBP", "H.D."]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("v2_prep")

RUN_DIR = PROJECT_ROOT / "results" / "runs" / "2026-05-11_usersnet_alpha0.8_ovr_roc"
OUT_NPZ = RUN_DIR / "resnet18_ckpts" / "fold_inputs.npz"


def main():
    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    df = _legacy.load_processed_spectra()
    feat_cols = _legacy.get_feature_columns(df)
    wavenumbers = np.array([float(c.split("_")[1]) for c in feat_cols])

    mc = ModelConfig.from_pipeline_config(raw_cfg, n_spectral_features=len(feat_cols))
    mc = _legacy.apply_class_selection(mc, cancer_types=CANCER_TYPES, non_cancer_groups=NON_CANCER)

    # CRITICAL FIX: build composite_id from ORIGINAL group + sample_id before alias.
    df["composite_id"] = df["group"].astype(str) + "_" + df["sample_id"].astype(str)

    df_resolved = _legacy.resolve_aliases(df, mc)

    # Aggregate by (group_resolved, composite_id) — keeps CPAN_10 distinct from YPAN_10.
    agg = df_resolved.groupby(["group", "composite_id"], as_index=False)[feat_cols].mean()
    logger.info(f"Mean (composite_id): {len(df_resolved)} spectra → {len(agg)} patients")

    valid_groups = set(mc.cancer_types) | set(mc.non_cancer_groups)
    df_valid = agg[agg["group"].isin(valid_groups)].copy()
    logger.info(f"After group filter: {len(df_valid)} patients")
    cancer_n = df_valid["group"].isin(mc.cancer_types).sum()
    nc_n = df_valid["group"].isin(mc.non_cancer_groups).sum()
    logger.info(f"  Cancer={cancer_n}, Non-cancer={nc_n}")

    # Per-group counts
    counts = df_valid.groupby("group").size()
    logger.info(f"Per-group counts:\n{counts.to_string()}")

    X = df_valid[feat_cols].values.astype(np.float32)
    groups_arr = df_valid["group"].to_numpy()
    composite_ids = df_valid["composite_id"].to_numpy()
    # numeric sample_id (parse from composite_id) — used for fold grouping
    sample_ids_int = np.array([abs(hash(c)) % (10**9) for c in composite_ids], dtype=np.int64)
    bl = np.array([1 if g in mc.cancer_types else 0 for g in groups_arr], dtype=np.int64)
    ctl = np.array([mc.cancer_type_index(g) for g in groups_arr], dtype=np.int64)

    # Sex mapping using (group, original_sample_id) pairs — pull from AACR npz.
    aacr = np.load(PROJECT_ROOT / "AACR" / "data" / "early_fusion" / "fold_predictions_7cancer.npz",
                   allow_pickle=True)
    asid = aacr["sample_ids"].tolist(); asx = aacr["sex_for_constraint"].tolist()
    agr = aacr["groups"].tolist()
    pair_map: dict = {}
    for s, x, g in zip(asid, asx, agr):
        pair_map.setdefault((int(s), str(g)), []).append(int(x))
    pair_map = {k: Counter(v).most_common(1)[0][0] for k, v in pair_map.items()}

    sex_for_constraint = np.zeros(len(df_valid), dtype=np.int64)
    n_resolved = 0
    for i, cid in enumerate(composite_ids):
        # composite_id like "CPAN_10" or "BLC_241" — recover original group + sample_id
        orig_g, _, sid_str = cid.rpartition("_")
        try:
            sid_int = int(sid_str)
        except ValueError:
            sid_int = -1
        if (sid_int, orig_g) in pair_map:
            sex_for_constraint[i] = pair_map[(sid_int, orig_g)]
            n_resolved += 1
        elif orig_g == "PRO":
            sex_for_constraint[i] = 1
        elif orig_g in ("OVA", "BRE"):
            sex_for_constraint[i] = 0
    logger.info(f"Sex resolved via AACR pair-map: {n_resolved}/{len(composite_ids)}")

    np.savez_compressed(OUT_NPZ,
                        X=X, bl=bl, ctl=ctl,
                        sample_ids=sample_ids_int,
                        composite_ids=composite_ids,
                        groups_arr=groups_arr,
                        wavenumbers=wavenumbers,
                        cancer_types=np.array(CANCER_TYPES),
                        sex_for_constraint=sex_for_constraint)
    logger.info(f"Saved {OUT_NPZ}")

    # Also save sex array under extra_models for downstream scripts that read it directly
    extra = RUN_DIR / "extra_models"; extra.mkdir(parents=True, exist_ok=True)
    np.save(extra / "sex_for_constraint.npy", sex_for_constraint)


if __name__ == "__main__":
    main()
