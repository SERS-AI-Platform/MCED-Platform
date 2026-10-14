"""Phase LCMS-1: Boramae 45-subject untargeted LC-MS/MS candidates vs same-subject SERS bands.

Inputs
- data/06_supporting_or_previous_outputs/lcmsms_untargeted_BNOR-BPRO_20261014/
  (supplement 2 per-sample PQN log2 values for 28 standard-matched features, sample map)
- aecd_platform measurement runs 1-6 (2026-08 Boramae mapping, ~121 points per subject)
- BM-15 data-driven bands (15 + 727 split into 723a/744b)

Design notes
- Strip lot is aliased with group: normal + benign on lot 1 (08-10~13), cancer on lot 6 (08-13~14).
  Correlations are therefore computed inside each lot stratum and combined by Fisher z.
- "미검출" is non-detection (NaN), never 0. Correlations use detected subjects only.
- Arm A raw: trim -> rolling-min baseline, no smoothing/normalisation. Arm B production: + SG + SNV.
- Subject value = median over points (quantitative read-out; model training stays non-aggregated).

Run: python scripts/analysis/lcms_sers_candidate_comparison.py
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.sers.config import load_config
from src.sers.preprocessing import preprocess_single_spectrum

ROOT = Path(__file__).resolve().parents[2]
LC_DIR = ROOT / "data/06_supporting_or_previous_outputs/lcmsms_untargeted_BNOR-BPRO_20261014"
BANDS_CSV = Path("/home/user/SERS-AI-ci-tiered/results/su_strip_normalization_20261008/bands.csv")
OUT = ROOT / "results/lcms_sers_candidates_20261014"
RUN_IDS = (1, 2, 3, 4, 5, 6)
GRID = np.arange(600.0, 1800.0 + 0.5, 1.0)
MIN_DETECTED = 8  # per stratum, for a correlation to count
N_PERM = 200


def connect():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ.get("PGDATABASE", "aecd_platform"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ["PGPASSWORD"],
        options="-c default_transaction_read_only=on",
    )


def load_lcms() -> tuple[pd.DataFrame, pd.DataFrame]:
    m = pd.read_csv(LC_DIR / "sample_map_lcms_to_solum_label.csv")
    v = pd.read_excel(LC_DIR / "전립선암_소변대사체_보충자료2_표준품_시료별검출.xlsx", "시료별 값")
    feats = v.columns[5:]
    vals = v.set_index("시료 번호")[feats].replace("미검출", np.nan).apply(pd.to_numeric)
    vals.index = m.set_index("시료 번호").loc[vals.index, "solum_label"].values
    grp = {"정상": "normal", "전립선암": "cancer", "전립선 관련 질환": "benign"}
    meta = m.assign(group=m["군"].map(grp)).set_index("solum_label")[["group", "grade_group_v11"]]
    return meta, vals


def load_spectra(labels: list[str]) -> pd.DataFrame:
    q = """select s.solum_label, m.measurement_run_id, r.strip_lot_id, rs.wavenumber, rs.intensities
           from measurement.measurements m
           join measurement.raw_spectra rs using(measurement_id)
           join measurement.runs r using(measurement_run_id)
           join master.samples s using(sample_id)
           where m.measurement_run_id = any(%s) and m.status = 'acquired'
             and rs.source_kind = 'replicate' and s.solum_label = any(%s)"""
    with connect() as c:
        return pd.read_sql(q, c, params=(list(RUN_IDS), labels))


def preprocess(df: pd.DataFrame, pp) -> tuple[np.ndarray, np.ndarray]:
    common = dict(grid=GRID, trim_region=(GRID[0], GRID[-1]), do_baseline=True,
                  baseline_method=pp.baseline_method, baseline_window=pp.baseline_window)
    a, b = [], []
    for x, y in zip(df.wavenumber, df.intensities):
        x, y = np.asarray(x), np.asarray(y)
        a.append(preprocess_single_spectrum(x, y, do_smooth=False, normalization="none", **common))
        b.append(preprocess_single_spectrum(x, y, do_smooth=True, smooth_window=pp.smooth_window,
                                            smooth_poly=pp.smooth_poly, normalization="snv", **common))
    return np.array(a), np.array(b)


def band_area(X: np.ndarray, bands: pd.DataFrame) -> np.ndarray:
    """Area above the window minimum (same definition as BM-15 band_features)."""
    out = []
    for left, right in zip(bands.left, bands.right):
        m = (GRID >= left) & (GRID <= right)
        seg = X[:, m]
        out.append(np.trapezoid(seg - seg.min(axis=1, keepdims=True), GRID[m], axis=1))
    return np.column_stack(out)


def stratified_spearman(x: pd.Series, y: pd.Series, strata: pd.Series) -> dict:
    res, zs, ws = {}, [], []
    for s in sorted(strata.unique()):
        k = (strata == s) & x.notna() & y.notna()
        n = int(k.sum())
        res[f"n_{s}"] = n
        if n < MIN_DETECTED:
            res[f"rho_{s}"] = np.nan
            continue
        r = stats.spearmanr(x[k], y[k]).statistic
        res[f"rho_{s}"] = r
        zs.append(np.arctanh(np.clip(r, -0.999, 0.999)))
        ws.append(n - 3)
    if ws:
        z = np.dot(zs, ws) / sum(ws)
        res["rho_pooled"] = float(np.tanh(z))
        res["p"] = float(2 * stats.norm.sf(abs(z) * np.sqrt(sum(ws))))
    else:
        res["rho_pooled"] = res["p"] = np.nan
    return res


def cv_auc(X: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """All-band LR, 5-fold x5 repeated (mean AUC across repeats), matching the vendor panel protocol."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    aucs = []
    for rep in range(5):
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed + rep)
        p = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, p))
    return float(np.mean(aucs)), float(np.std(aucs))


def bh(p: pd.Series) -> pd.Series:
    q = pd.Series(np.nan, index=p.index)
    ok = p.notna()
    q[ok] = stats.false_discovery_control(p[ok])
    return q


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pp = load_config().preprocessing
    meta, lc = load_lcms()
    bands = pd.read_csv(BANDS_CSV, encoding="utf-8-sig")
    spec = load_spectra(list(meta.index))

    A, B = preprocess(spec, pp)
    rows = []
    for arm, X in (("raw", A), ("snv", B)):
        med = pd.DataFrame(X).groupby(spec.solum_label.values).median()
        feat = pd.DataFrame(band_area(med.values, bands), index=med.index, columns=bands.name.astype(str))
        feat.to_csv(OUT / f"sers_band_area_{arm}.csv", encoding="utf-8-sig")
        rows.append((arm, feat))
    lot = spec.groupby("solum_label").strip_lot_id.first().map(lambda v: f"lot{v}")
    meta = meta.join(lot.rename("lot")).join(spec.groupby("solum_label").size().rename("n_points"))
    meta.to_csv(OUT / "subjects.csv", encoding="utf-8-sig")

    # 1) candidate feature x band correlation inside lot strata
    corr = []
    for arm, feat in rows:
        for f in lc.columns:
            for b in feat.columns:
                r = stratified_spearman(lc[f], feat.loc[lc.index, b], meta.loc[lc.index, "lot"])
                corr.append({"arm": arm, "lcms_feature": f, "band": b, **r})
    corr = pd.DataFrame(corr)
    corr["q"] = bh(corr.p)
    corr.sort_values("p").to_csv(OUT / "corr_lcms_vs_band.csv", index=False, encoding="utf-8-sig")

    # permutation null: shuffle whole LC rows across subjects inside each lot (keeps feature and
    # band correlation structure), count p<0.05 hits and the strongest |rho_pooled|
    rng = np.random.default_rng(0)
    strata = meta.loc[lc.index, "lot"]
    null_hits, null_max = [], []
    for _ in range(N_PERM):
        perm = lc.index.to_numpy().copy()
        for s in strata.unique():
            k = np.flatnonzero(strata.values == s)
            perm[k] = rng.permutation(perm[k])
        lcp = lc.loc[perm].set_axis(lc.index)
        ps, rs = [], []
        for _, feat in rows:
            for f in lcp.columns:
                for b in feat.columns:
                    r = stratified_spearman(lcp[f], feat.loc[lc.index, b], strata)
                    ps.append(r["p"])
                    rs.append(abs(r["rho_pooled"]))
        null_hits.append(int(np.nansum(np.array(ps) < 0.05)))
        null_max.append(float(np.nanmax(rs)))
    obs_hits, obs_max = int((corr.p < 0.05).sum()), float(corr.rho_pooled.abs().max())
    perm_summary = {
        "n_perm": N_PERM, "obs_hits_p05": obs_hits,
        "null_hits_median": float(np.median(null_hits)), "null_hits_95pct": float(np.percentile(null_hits, 95)),
        "perm_p_hits": float((1 + sum(h >= obs_hits for h in null_hits)) / (N_PERM + 1)),
        "obs_max_abs_rho": obs_max, "null_max_abs_rho_95pct": float(np.percentile(null_max, 95)),
        "perm_p_max_rho": float((1 + sum(m >= obs_max for m in null_max)) / (N_PERM + 1)),
    }

    # 2) SERS group separation on the same contrasts the vendor used
    contrasts = {
        "S5_normal_vs_benign": (meta.group == "normal", meta.group == "benign", "same lot 1"),
        "S4dich2_GG12_vs_GG35": ((meta.group == "cancer") & (meta.grade_group_v11 <= 2),
                                 (meta.group == "cancer") & (meta.grade_group_v11 >= 3), "same lot 6"),
        "S2_normal_vs_cancer": (meta.group == "normal", meta.group == "cancer", "LOT CONFOUNDED"),
        "S3_benign_vs_cancer": (meta.group == "benign", meta.group == "cancer", "LOT CONFOUNDED"),
    }
    band_rows, panel_rows = [], []
    for arm, feat in rows:
        for name, (neg, pos, note) in contrasts.items():
            idx = meta.index[neg | pos]
            y = pos[idx].astype(int).values
            for b in feat.columns:
                v = feat.loc[idx, b].values
                band_rows.append({"arm": arm, "contrast": name, "lot_note": note, "band": b,
                                  "n_neg": int((y == 0).sum()), "n_pos": int(y.sum()),
                                  "auc": roc_auc_score(y, v),
                                  "p": stats.mannwhitneyu(v[y == 1], v[y == 0]).pvalue})
            m, s = cv_auc(feat.loc[idx].values, y)
            panel_rows.append({"arm": arm, "contrast": name, "lot_note": note, "n_neg": int((y == 0).sum()),
                               "n_pos": int(y.sum()), "cv_auc_mean": m, "cv_auc_sd": s})
    band_df = pd.DataFrame(band_rows)
    band_df["q"] = bh(band_df.p)
    band_df.to_csv(OUT / "sers_band_group_auc.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(panel_rows).to_csv(OUT / "sers_panel_cv_auc.csv", index=False, encoding="utf-8-sig")

    cfg_hash = hashlib.sha256((ROOT / "config/config.yaml").read_bytes()).hexdigest()[:12]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    summary = {
        "phase": "LCMS-1", "n_subjects": int(len(meta)), "n_spectra": int(len(spec)),
        "points_per_subject": [int(meta.n_points.min()), int(meta.n_points.max())],
        "lot_by_group": meta.groupby(["group", "lot"]).size().to_dict().__repr__(),
        "n_corr_tests": int(corr.p.notna().sum()), "n_corr_q05": int((corr.q < 0.05).sum()),
        "n_corr_p05": int((corr.p < 0.05).sum()),
        **perm_summary,
        "n_band_tests": int(len(band_df)), "n_band_q05": int((band_df.q < 0.05).sum()),
        "axis_calibration": "none (runs 1-6 calibration_id NULL)", "config_sha256_12": cfg_hash, "git": commit,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
