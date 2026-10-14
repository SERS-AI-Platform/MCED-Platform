"""Subject-count learning curves for cancer-vs-control screening.

Answers two planning questions with data instead of intuition:

1. Minimum number of cancer subjects per cancer type before the screening AUC
   plateaus (retrospective 7-cancer KAO cohort, 2026-06-10 update).
2. Expected AUC if the prostate cohort grows to a target size, by fitting an
   inverse power law ``AUC(n) = a - b * n**(-c)`` to the observed curve and
   extrapolating (Figueroa et al. 2012, BMC Med Inform Decis Mak).

Two independent curves are produced because they answer different questions:

* ``kao``      — retrospective, multi-site Thermo, each cancer vs. 430 controls
                 (in-distribution ceiling).
* ``boramae``  — prospective single-site Boramae mapping cohort, prostate (43)
                 vs. non-cancer (69). This is the domain the 450-subject
                 prostate target actually lives in.

Design (kept deliberately simple so that the curve reflects sample size, not
modelling choices):

* One row per subject = mean of QC-passed replicates (matches production
  aggregation used by STK-V2 inputs).
* StandardScaler + L2 LogisticRegression (C=0.1) — the project's strongest
  classical baseline (experiment.md Phase F/K).
* For each repeat: 30 % stratified hold-out test set fixed; then cancer
  subjects sub-sampled to n and controls sub-sampled 1:1 (min(n, available))
  from the remaining 70 %. Reported AUC is the hold-out AUC.

Run from repo root:

    python scripts/analysis/learning_curve_sample_size.py --repeats 30

Outputs go to ``results/learning_curve_20260909_v1/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
KAO_CSV = ROOT / "results/kao_20260610_updated_cohort/processed_spectra.csv"
BORAMAE_CSV = (
    ROOT / "notebooks/aecd_api_model_mean_spectrum_outputs/mean_representative_spectra.csv"
)
OUT_DIR = ROOT / "results/learning_curve_20260909_v1"

CANCER_GROUPS = ["PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC"]
CONTROL_GROUPS = ["NOR", "DIA", "HBP", "H.D."]
# KAO cohort merges CPAN+YPAN into PAN and YNOR into NOR.
SOURCE_TO_MODEL = {"CPAN": "PAN", "YPAN": "PAN", "YNOR": "NOR"}

GRID_KAO = [10, 15, 20, 30, 40, 50, 70, 100, 130, 160, 200, 210]
GRID_BORAMAE = [8, 12, 16, 20, 24, 28, 30]


def inverse_power(n, a, b, c):
    return a - b * np.power(n, -c)


def fit_curve(n, auc):
    """Fit AUC(n)=a-b*n^-c. Returns (a,b,c) or None if the fit fails."""
    try:
        popt, _ = curve_fit(
            inverse_power,
            n,
            auc,
            p0=[min(max(auc) + 0.02, 0.999), 1.0, 0.5],
            bounds=([0.5, 0.0, 0.01], [1.0, 50.0, 3.0]),
            maxfev=20000,
        )
        return popt
    except (RuntimeError, ValueError):
        return None


def load_kao() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(KAO_CSV)
    feat_cols = [c for c in df.columns if c.startswith("x_")]
    df["model_group"] = df["group"].map(lambda g: SOURCE_TO_MODEL.get(g, g))
    # subject key must keep source group: CPAN 25 and YPAN 25 are different people
    subj = df.groupby(["group", "sample_id"], sort=False)
    X = subj[feat_cols].mean().to_numpy(dtype=np.float32)
    g = subj["model_group"].first().to_numpy()
    ids = np.array([f"{a}_{b}" for a, b in subj.groups.keys()])
    return X, g, ids


def load_boramae() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(BORAMAE_CSV)
    feat_cols = [c for c in df.columns if c.startswith("wn_")]
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = (df["label"] == "prostate").astype(int).to_numpy()
    return X, y


def holdout_auc(Xtr, ytr, Xte, yte) -> float:
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.1, max_iter=5000)
    clf.fit(sc.transform(Xtr), ytr)
    return roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1])


def curve_one_task(
    X_pos, X_neg, grid, repeats, rng, test_frac=0.3, min_test_pos=5
) -> pd.DataFrame:
    rows = []
    n_pos, n_neg = len(X_pos), len(X_neg)
    for r in range(repeats):
        seed = int(rng.integers(0, 2**31 - 1))
        X = np.vstack([X_pos, X_neg])
        y = np.r_[np.ones(n_pos, int), np.zeros(n_neg, int)]
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_frac, stratify=y, random_state=seed
        )
        if yte.sum() < min_test_pos:
            continue
        pos_idx = np.flatnonzero(ytr == 1)
        neg_idx = np.flatnonzero(ytr == 0)
        rs = np.random.default_rng(seed)
        for n in grid:
            if n > len(pos_idx):
                continue
            n_ctrl = min(n, len(neg_idx))
            sel = np.r_[
                rs.choice(pos_idx, n, replace=False),
                rs.choice(neg_idx, n_ctrl, replace=False),
            ]
            auc = holdout_auc(Xtr[sel], ytr[sel], Xte, yte)
            rows.append({"repeat": r, "n_cancer_train": n, "n_control_train": n_ctrl, "auc": auc})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, task: str, n_available: int, target_n: int | None):
    s = (
        df.groupby("n_cancer_train")["auc"]
        .agg(mean="mean", sd="std", q05=lambda v: v.quantile(0.05), q95=lambda v: v.quantile(0.95), reps="count")
        .reset_index()
    )
    s.insert(0, "task", task)
    popt = fit_curve(s["n_cancer_train"].to_numpy(float), s["mean"].to_numpy(float))
    out = {"task": task, "n_cancer_available": int(n_available), "fit": None}
    if popt is not None:
        a, b, c = (float(v) for v in popt)
        out["fit"] = {"a_plateau": a, "b": b, "c": c}
        # minimum n: first grid point whose mean AUC is within 0.01 of the plateau
        within = s[s["mean"] >= a - 0.01]
        out["n_min_within_0.01_of_plateau"] = int(within["n_cancer_train"].min()) if len(within) else None
        ge95 = s[s["mean"] >= 0.95]
        out["n_min_auc_ge_0.95"] = int(ge95["n_cancer_train"].min()) if len(ge95) else None
        if target_n is not None:
            # bootstrap the extrapolation over per-repeat curves
            preds = []
            for _, grp in df.groupby("repeat"):
                g = grp.groupby("n_cancer_train")["auc"].mean()
                p = fit_curve(g.index.to_numpy(float), g.to_numpy(float))
                if p is not None:
                    preds.append(inverse_power(target_n, *p))
            preds = np.array(preds)
            out["target_n"] = target_n
            out["pred_auc_at_target"] = float(inverse_power(target_n, a, b, c))
            if len(preds):
                out["pred_auc_at_target_ci90"] = [
                    float(np.quantile(preds, 0.05)),
                    float(np.quantile(preds, 0.95)),
                ]
                out["pred_auc_at_target_median_of_repeat_fits"] = float(np.median(preds))
            # required n to reach a few AUC milestones (from the mean fit)
            req = {}
            for tgt in (0.80, 0.85, 0.90, 0.95):
                n_req = (b / (a - tgt)) ** (1.0 / c) if tgt < a else np.inf
                # beyond 10,000 subjects the fit says "not reachable by sample size alone"
                req[str(tgt)] = int(np.ceil(n_req)) if n_req <= 10_000 else None
            out["n_required_for_auc"] = req
    return s, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--target-n", type=int, default=450)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    curves, summaries = [], []

    # ---- retrospective KAO cohort: each cancer vs all controls ----
    X, g, _ = load_kao()
    X_neg = X[np.isin(g, CONTROL_GROUPS)]
    print(f"KAO subjects: {len(X)}  controls: {len(X_neg)}")
    for ct in CANCER_GROUPS:
        X_pos = X[g == ct]
        df = curve_one_task(X_pos, X_neg, GRID_KAO, args.repeats, rng)
        s, out = summarize(df, f"kao_{ct}_vs_control", len(X_pos), args.target_n if ct == "PRO" else None)
        df.insert(0, "task", f"kao_{ct}_vs_control")
        curves.append(df)
        summaries.append((s, out))
        print(f"  {ct:4s} n={len(X_pos):3d}  " + "  ".join(f"{int(r.n_cancer_train)}:{r['mean']:.3f}" for _, r in s.iterrows()))

    # ---- prospective Boramae cohort: prostate vs non-cancer ----
    Xb, yb = load_boramae()
    X_pos, X_neg = Xb[yb == 1], Xb[yb == 0]
    print(f"Boramae subjects: {len(Xb)}  prostate: {len(X_pos)}  non-cancer: {len(X_neg)}")
    df = curve_one_task(X_pos, X_neg, GRID_BORAMAE, args.repeats * 2, rng, test_frac=0.3, min_test_pos=5)
    s, out = summarize(df, "boramae_PRO_vs_noncancer", len(X_pos), args.target_n)
    df.insert(0, "task", "boramae_PRO_vs_noncancer")
    curves.append(df)
    summaries.append((s, out))
    print("  BORAMAE " + "  ".join(f"{int(r.n_cancer_train)}:{r['mean']:.3f}" for _, r in s.iterrows()))

    pd.concat(curves).to_csv(OUT_DIR / "learning_curve_raw.csv", index=False, encoding="utf-8-sig")
    pd.concat([s for s, _ in summaries]).to_csv(
        OUT_DIR / "learning_curve_summary.csv", index=False, encoding="utf-8-sig"
    )
    with open(OUT_DIR / "learning_curve_fits.json", "w", encoding="utf-8") as f:
        json.dump([o for _, o in summaries], f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "kao_csv": str(KAO_CSV.relative_to(ROOT)),
                "boramae_csv": str(BORAMAE_CSV.relative_to(ROOT)),
                "model": "StandardScaler + LogisticRegression(C=0.1, L2)",
                "aggregation": "subject mean of replicates",
                "controls_kao": CONTROL_GROUPS,
                "control_ratio": "1:1 (min(n, available))",
                "test_fraction": 0.3,
                "repeats": args.repeats,
                "seed": args.seed,
                "target_n": args.target_n,
                "curve_model": "AUC(n) = a - b * n^(-c)",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print("done ->", OUT_DIR)


if __name__ == "__main__":
    main()
