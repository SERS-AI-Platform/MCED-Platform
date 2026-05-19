"""Rebuild correct sex_for_constraint mapping using (sample_id, group) pairs
and reapply sex constraint to all base-model val_cancer_logits."""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = PROJECT_ROOT / "results" / "runs" / "2026-05-11_usersnet_alpha0.8_ovr_roc"
EXTRA = RUN_DIR / "extra_models"

CANCER_TYPES = ["PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "BLC"]


def apply_sex_constraint(probs, sex):
    out = probs.copy()
    pro = CANCER_TYPES.index("PRO"); ova = CANCER_TYPES.index("OVA")
    out[sex == 0, pro] = 0.0
    out[sex == 1, ova] = 0.0
    s = out.sum(axis=1, keepdims=True)
    nz = s.squeeze() > 0
    out[nz] = out[nz] / s[nz]
    return out


def main():
    inputs = np.load(RUN_DIR / "resnet18_ckpts" / "fold_inputs.npz", allow_pickle=True)
    sid = inputs["sample_ids"]; groups = inputs["groups_arr"]

    aacr = np.load(PROJECT_ROOT / "AACR" / "data" / "early_fusion" / "fold_predictions_7cancer.npz",
                   allow_pickle=True)
    asid = aacr["sample_ids"].tolist()
    asx = aacr["sex_for_constraint"].tolist()
    agr = aacr["groups"].tolist()

    pair_map = {}
    for s, x, g in zip(asid, asx, agr):
        pair_map.setdefault((int(s), str(g)), []).append(int(x))
    pair_map = {k: Counter(v).most_common(1)[0][0] for k, v in pair_map.items()}

    sex_new = np.zeros(len(sid), dtype=int)
    missing = 0
    for i, (s, g) in enumerate(zip(sid, groups)):
        key = (int(s), str(g))
        if key in pair_map:
            sex_new[i] = pair_map[key]
        else:
            # group-based fallback
            if str(g) == "PRO":
                sex_new[i] = 1
            elif str(g) in ("OVA", "BRE"):
                sex_new[i] = 0
            else:
                # Use any sample_id match (group-agnostic) as last resort
                fallback = [v for k, v in pair_map.items() if k[0] == int(s)]
                sex_new[i] = Counter(fallback).most_common(1)[0][0] if fallback else 0
                missing += 1
    print(f"Sex remap: male={(sex_new==1).sum()}, female={(sex_new==0).sum()}, missing-with-fallback={missing}")
    np.save(EXTRA / "sex_for_constraint.npy", sex_new)

    for m in ["logistic_regression", "random_forest", "xgboost", "cnn1d"]:
        p = EXTRA / f"{m}_oof.npz"
        d = dict(np.load(p, allow_pickle=True))
        raw = d.get("val_cancer_logits_raw", d["val_cancer_logits"])
        d["val_cancer_logits"] = apply_sex_constraint(raw, sex_new)
        d["val_cancer_logits_raw"] = raw
        np.savez_compressed(p, **d)
        print(f"  rewrote {p.name} with corrected sex constraint")


if __name__ == "__main__":
    main()
