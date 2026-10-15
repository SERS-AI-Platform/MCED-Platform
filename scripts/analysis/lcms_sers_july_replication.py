"""Phase LCMS-3: replicate LCMS-2 (Aug mapping) LC-MS/MS x SERS correlations on July SERS sets.

Same 45 Boramae subjects, independent SERS acquisitions (pre-reagent-change, 5 points each):
- liquid: DB runs 98, 99 (2026-07-09/10)
- powder: DB run 110 (2026-07-16)
DB links July files by patient number (file "BPRO 20" -> BNOR_20, same patient as Aug mapping).

Pre-specified tests (fixed from LCMS-2 before looking at July):
1. The 9 feature x arm x wavenumber hits with LCMS-2 FWER p < 0.05. Value = mean over
   wavenumber +-3 cm-1 (absorbs small axis offsets; no axis calibration on either set).
   Pooled within-stratum Spearman, one-sided permutation p in the Aug direction.
2. Shape agreement: Pearson r between the Aug and July correlation spectra per feature x arm,
   null = same within-stratum permutation of July LC rows shared across features.
Strata = the Aug lot split (normal + benign vs cancer) so both sets are analysed identically.

Run: PYTHONPATH=. python scripts/analysis/lcms_sers_july_replication.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess

import numpy as np
import pandas as pd

from scripts.analysis.lcms_sers_candidate_comparison import GRID, ROOT, bh, connect, load_lcms, preprocess
from scripts.analysis.lcms_sers_wavenumber_corr import pooled_rho
from src.sers.config import load_config

AUG = ROOT / "results/lcms_sers_wavenumber_corr_20261015"
OUT = ROOT / "results/lcms_sers_july_replication_20261015"
SETS = {"july_liquid": (98, 99), "july_powder": (110,)}
HALF_WINDOW = 3.0
N_PERM = 2000
N_SHAPE_PERM = 200


def load_runs(run_ids: tuple[int, ...], labels: list[str]) -> pd.DataFrame:
    q = """select s.solum_label, rs.wavenumber, rs.intensities
           from measurement.measurements m
           join measurement.raw_spectra rs using(measurement_id)
           join master.samples s using(sample_id)
           where m.measurement_run_id = any(%s) and m.status = 'acquired'
             and rs.source_kind = 'replicate' and s.solum_label = any(%s)"""
    with connect() as c:
        return pd.read_sql(q, c, params=(list(run_ids), labels))


def permute_within(index: pd.Index, strata: pd.Series, rng: np.random.Generator) -> np.ndarray:
    perm = index.to_numpy().copy()
    for s in strata.unique():
        k = np.flatnonzero(strata.values == s)
        perm[k] = rng.permutation(perm[k])
    return perm


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pp = load_config().preprocessing
    meta, lc = load_lcms()
    strata = (meta.group == "cancer").map({True: "lot6", False: "lot1"})  # same split as Aug lots
    hits = pd.read_csv(AUG / "feature_max_corr.csv").query("p_fwer < 0.05")
    aug_spec = pd.read_csv(AUG / "corr_spectra.csv", index_col=0)
    rng = np.random.default_rng(0)

    rep_rows, shape_rows, summary = [], [], {}
    for set_name, runs in SETS.items():
        spec = load_runs(runs, list(meta.index))
        A, B = preprocess(spec, pp)
        arms = {arm: pd.DataFrame(X).groupby(spec.solum_label.values).median().set_axis(GRID, axis=1)
                for arm, X in (("raw", A), ("snv", B))}
        ids = arms["raw"].index.intersection(lc.index)
        lcs, st = lc.loc[ids], strata.loc[ids]
        pts = spec.groupby("solum_label").size()

        # 1) pre-specified hits
        for h in hits.itertuples():
            m = (GRID >= h.wavenumber_at_max - HALF_WINDOW) & (GRID <= h.wavenumber_at_max + HALF_WINDOW)
            v = arms[h.arm].loc[ids, m].mean(axis=1).to_frame("v")
            obs = pooled_rho(lcs[h.lcms_feature], v, st)[0]
            sign = np.sign(h.rho_at_max)
            null = np.array([pooled_rho(lcs.loc[permute_within(ids, st, rng), h.lcms_feature].set_axis(ids),
                                        v, st)[0] for _ in range(N_PERM)])
            rep_rows.append({"set": set_name, "arm": h.arm, "lcms_feature": h.lcms_feature,
                             "wavenumber": h.wavenumber_at_max, "aug_rho": h.rho_at_max, "july_rho": obs,
                             "same_direction": bool(np.sign(obs) == sign),
                             "p_one_sided": float((1 + (sign * null >= sign * obs).sum()) / (N_PERM + 1)),
                             "edge_wavenumber": bool(h.wavenumber_at_max <= 605 or h.wavenumber_at_max >= 1780)})

        # 2) spectrum shape agreement, shared permutations across features
        keys = [k for k in aug_spec.columns]
        def shape_r(lc_tab: pd.DataFrame) -> dict[str, float]:
            out = {}
            for k in keys:
                arm, f = k.split("|", 1)
                j = pooled_rho(lc_tab[f], arms[arm].loc[ids], st)
                if j is not None:
                    out[k] = float(np.corrcoef(aug_spec[k].values, j)[0, 1])
            return out
        obs_r = shape_r(lcs)
        null_r = [shape_r(lcs.loc[permute_within(ids, st, rng)].set_axis(ids)) for _ in range(N_SHAPE_PERM)]
        for k, r in obs_r.items():
            nr = np.array([d.get(k, np.nan) for d in null_r])
            shape_rows.append({"set": set_name, "key": k, "shape_r": r,
                               "p_one_sided": float((1 + np.nansum(nr >= r)) / (N_SHAPE_PERM + 1)),
                               "is_aug_hit": k in {f"{h.arm}|{h.lcms_feature}" for h in hits.itertuples()}})
        mean_obs = float(np.mean(list(obs_r.values())))
        mean_null = np.array([np.mean(list(d.values())) for d in null_r])
        summary[set_name] = {"n_subjects": int(len(ids)), "points_per_subject": [int(pts.min()), int(pts.max())],
                             "n_spectra": int(len(spec)), "mean_shape_r": mean_obs,
                             "mean_shape_r_null_median": float(np.median(mean_null)),
                             "mean_shape_r_null_95pct": float(np.percentile(mean_null, 95)),
                             "p_global_shape": float((1 + (mean_null >= mean_obs).sum()) / (N_SHAPE_PERM + 1))}

    rep = pd.DataFrame(rep_rows)
    rep["q_bh"] = rep.groupby("set").p_one_sided.transform(lambda p: bh(p))
    rep.to_csv(OUT / "prespecified_hits_replication.csv", index=False, encoding="utf-8-sig")
    shape = pd.DataFrame(shape_rows)
    shape["q_bh"] = shape.groupby("set").p_one_sided.transform(lambda p: bh(p))
    shape.sort_values(["set", "p_one_sided"]).to_csv(OUT / "shape_agreement.csv", index=False, encoding="utf-8-sig")
    for s in SETS:
        r = rep[rep.set == s]
        summary[s].update({"hits_same_direction": int(r.same_direction.sum()), "hits_total": int(len(r)),
                           "hits_p05": int((r.p_one_sided < 0.05).sum()), "hits_q05": int((r.q_bh < 0.05).sum())})
    summary.update({"phase": "LCMS-3", "half_window_cm1": HALF_WINDOW, "n_perm": N_PERM,
                    "n_shape_perm": N_SHAPE_PERM, "strata": "Aug lot split (normal+benign vs cancer)",
                    "config_sha256_12": hashlib.sha256((ROOT / "config/config.yaml").read_bytes()).hexdigest()[:12],
                    "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                          capture_output=True, text=True).stdout.strip()})
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(rep.round(3).to_string())


if __name__ == "__main__":
    main()
