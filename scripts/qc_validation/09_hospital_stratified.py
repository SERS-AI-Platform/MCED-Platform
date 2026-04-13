"""
Phase 9 — Hospital-Stratified Training (Batch Correction)

Repeats Phase 8 experiments A-E after applying per-hospital batch correction.

Two correction methods:
    Method 1 — mean centering: subtract per-hospital mean (additive effect)
    Method 2 — z-score: subtract mean and divide by per-hospital std

CV leakage handling:
    For each train fold we recompute hospital statistics from the train fold
    only and apply them to both train and test. For leave-Severance-out (B),
    Severance test data is corrected using Severance's own statistics
    (deployment simulation: we know the test hospital).

Note on fundamental limitation
------------------------------
Most hospitals in our cohort hold only ONE class (양산부산대 = control,
충북대 = cancer, ...). Per-hospital centering removes the hospital signature,
but where hospital ≡ class, it also removes the class signal. Severance is
the only hospital with both classes in our training data, so after correction
the cancer signal must come from there (n=59) — strongly limiting what can
be learned.
"""

from __future__ import annotations

import json
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.config import load_config, RAW_DATA_DIR, RESULTS_DIR
from src.sers import read_spectrum, parse_filename
from src.sers.io import find_spectra, make_fixed_grid
from src.sers.preprocessing import preprocess_spectra

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("phase9")

OUT_DIR = Path(RESULTS_DIR) / "qc_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_CAPS = {
    "NOR": 100, "DIA": 100, "HBP": 100, "H.D.": 100,
    "PRO": 100, "BRE": 30, "OVA": 70, "LUN": 100,
    "CRC": 100, "CPAN": 70, "BLC": 100,
    "YPAN": 30, "YNOR": 29,
}
HOSPITAL = {
    "NOR": "양산부산대", "DIA": "양산부산대",
    "HBP": "양산부산대", "H.D.": "양산부산대",
    "PRO": "충북대", "CRC": "충북대",
    "CPAN": "충북대", "BLC": "충북대",
    "BRE": "부산백", "OVA": "부산백서울대",
    "LUN": "서울대성모",
    "YPAN": "세브란스", "YNOR": "세브란스",
}
CANCER_LABELS = {"PRO", "BRE", "OVA", "LUN", "CRC", "CPAN", "YPAN", "BLC"}
CONTROL_LABELS = {"NOR", "YNOR", "DIA", "HBP", "H.D."}


# ── Loading & matrix build ─────────────────────────────────────
def load_subset(config) -> dict:
    data_dir = Path(RAW_DATA_DIR)
    raw = {}
    for fp in find_spectra(data_dir, pattern="*.csv"):
        try:
            spec_id = parse_filename(
                fp,
                fallback_group=config.folder_to_group.get(fp.parent.name, "UNK"),
            )
        except Exception:
            continue
        if spec_id.group not in GROUP_CAPS:
            continue
        x, y = read_spectrum(fp)
        raw[(spec_id.group, spec_id.sample_id, spec_id.replicate)] = (x, y)
    keep = set()
    for grp, n_max in GROUP_CAPS.items():
        sids = sorted({k[1] for k in raw if k[0] == grp})[:n_max]
        for k in raw:
            if k[0] == grp and k[1] in sids:
                keep.add(k)
    raw = {k: raw[k] for k in keep}
    n_subj = len({(g, s) for g, s, _ in raw})
    logger.info(f"Loaded: {len(raw)} spectra, {n_subj} subjects")
    return raw


def subject_matrix(processed: dict):
    subj = {}
    for (g, s, _), y in processed.items():
        subj.setdefault((g, s), []).append(y)
    rows, groups, sids = [], [], []
    for (g, s), reps in subj.items():
        rows.append(np.mean(np.vstack(reps), axis=0))
        groups.append(g)
        sids.append(s)
    return np.vstack(rows), np.array(groups), sids


# ── Batch correction (leak-safe) ──────────────────────────────
def fit_hospital_stats(X: np.ndarray, hospitals: np.ndarray, method: str):
    """Compute per-hospital mean (and std if z-score) from given samples."""
    stats = {}
    for h in np.unique(hospitals):
        m = hospitals == h
        mu = X[m].mean(axis=0)
        if method == "zscore":
            sd = X[m].std(axis=0, ddof=0)
            sd = np.where(sd < 1e-9, 1.0, sd)
        else:
            sd = None
        stats[h] = (mu, sd)
    return stats


def apply_hospital_correction(X, hospitals, stats, method, fallback_stats=None):
    Xc = X.copy()
    for i, h in enumerate(hospitals):
        if h in stats:
            mu, sd = stats[h]
        elif fallback_stats is not None and h in fallback_stats:
            mu, sd = fallback_stats[h]
        else:
            continue
        Xc[i] = Xc[i] - mu
        if method == "zscore" and sd is not None:
            Xc[i] = Xc[i] / sd
    return Xc


# ── Classifier helper ─────────────────────────────────────────
def _fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def cv_with_correction(X, y, hospitals, method, n_splits=5, random_state=42):
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs = []
    for tr, te in skf.split(X, y):
        if method != "none":
            stats_tr = fit_hospital_stats(X[tr], hospitals[tr], method)
            # Test fold may include hospitals not seen in train fold (rare with random
            # CV but possible). Compute test-fold stats as fallback for unseen ones.
            stats_te_fallback = fit_hospital_stats(X[te], hospitals[te], method)
            Xtr = apply_hospital_correction(X[tr], hospitals[tr], stats_tr, method)
            Xte = apply_hospital_correction(X[te], hospitals[te], stats_tr, method,
                                            fallback_stats=stats_te_fallback)
        else:
            Xtr, Xte = X[tr], X[te]
        proba = _fit_predict(Xtr, ytr=y[tr], Xte=Xte)
        aucs.append(roc_auc_score(y[te], proba))
    return float(np.mean(aucs)), float(np.std(aucs))


def holdout_with_correction(Xtr, ytr, htr, Xte, yte, hte, method):
    if method != "none":
        stats_tr = fit_hospital_stats(Xtr, htr, method)
        stats_te = fit_hospital_stats(Xte, hte, method)  # deployment-time stats
        Xtr = apply_hospital_correction(Xtr, htr, stats_tr, method)
        Xte = apply_hospital_correction(Xte, hte, stats_te, method)
    if len(np.unique(yte)) < 2:
        return float("nan")
    proba = _fit_predict(Xtr, ytr, Xte)
    return float(roc_auc_score(yte, proba))


# ── Experiments (parameterized by correction method) ──────────
def run_all(X, groups, method: str) -> dict:
    hospitals = np.array([HOSPITAL[g] for g in groups])
    out = {}

    # A — naive random CV cancer vs control
    mask = np.array([g in CANCER_LABELS or g in CONTROL_LABELS for g in groups])
    Xa, ha = X[mask], hospitals[mask]
    ya = np.array([1 if g in CANCER_LABELS else 0 for g in groups[mask]])
    a_mean, a_std = cv_with_correction(Xa, ya, ha, method)
    out["A"] = {"auc": a_mean, "std": a_std, "n": int(len(ya))}

    # B — leave-Severance-out
    sev = np.array([g in ("YPAN", "YNOR") for g in groups])
    in_task = mask
    train = in_task & ~sev
    test = in_task & sev
    Xtr, Xte = X[train], X[test]
    htr, hte = hospitals[train], hospitals[test]
    ytr = np.array([1 if g in CANCER_LABELS else 0 for g in groups[train]])
    yte = np.array([1 if g in CANCER_LABELS else 0 for g in groups[test]])
    auc_b = holdout_with_correction(Xtr, ytr, htr, Xte, yte, hte, method)
    out["B"] = {"auc": auc_b, "n_train": int(len(ytr)), "n_test": int(len(yte))}

    # C — within-Severance CV (correction is no-op since one hospital)
    Xc = X[sev]
    yc = np.array([1 if g == "YPAN" else 0 for g in groups[sev]])
    hc = hospitals[sev]
    c_mean, c_std = cv_with_correction(Xc, yc, hc, method)
    out["C"] = {"auc": c_mean, "std": c_std, "n": int(len(yc))}

    # D — within-양산부산대 NOR vs chronic (also one hospital — correction no-op)
    yangsan = np.array([g in ("NOR", "DIA", "HBP", "H.D.") for g in groups])
    Xd = X[yangsan]
    yd = np.array([1 if g != "NOR" else 0 for g in groups[yangsan]])
    hd = hospitals[yangsan]
    d_mean, d_std = cv_with_correction(Xd, yd, hd, method)
    out["D"] = {"auc": d_mean, "std": d_std, "n": int(len(yd))}

    # E — NOR shuffle sanity (one hospital — correction no-op)
    nor_mask = np.array([g == "NOR" for g in groups])
    Xn = X[nor_mask]
    hn = hospitals[nor_mask]
    rng = np.random.RandomState(42)
    aucs = []
    for rep in range(20):
        y_shuf = (rng.permutation(len(Xn)) < (len(Xn) // 2)).astype(int)
        a, _ = cv_with_correction(Xn, y_shuf, hn, method, random_state=rep)
        aucs.append(a)
    out["E"] = {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                "auc_max": float(np.max(aucs)), "n": int(len(Xn))}
    return out


def main():
    config = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    grid = make_fixed_grid(config)

    logger.info("[1] Load + preprocess ...")
    raw = load_subset(config)
    processed_dict, _, _ = preprocess_spectra(raw, grid, config, qc_passed_keys=None)
    X, groups, _ = subject_matrix(processed_dict)
    logger.info(f"  Subject matrix: {X.shape}")

    all_results = {}
    for method in ["none", "mean", "zscore"]:
        logger.info(f"\n=== Method: {method} ===")
        res = run_all(X, groups, method)
        for k, v in res.items():
            if k == "E":
                logger.info(f"  {k} n={v['n']:>4d}  AUC={v['auc_mean']:.4f}±{v['auc_std']:.4f} max={v['auc_max']:.3f}")
            else:
                auc = v.get("auc")
                std = v.get("std", "")
                std_s = f"±{std:.4f}" if isinstance(std, float) else ""
                n = v.get("n", v.get("n_train", "?"))
                logger.info(f"  {k} n={n}  AUC={auc:.4f}{std_s}")
        all_results[method] = res

    (OUT_DIR / "hospital_stratified.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Side-by-side report
    def fmt(x, p=4):
        return f"{x:.{p}f}" if isinstance(x, (int, float)) and not pd.isna(x) else "—"

    def get_auc(d, k):
        if k == "E":
            return d[k]["auc_mean"]
        return d[k]["auc"]

    lines = []
    lines.append("# Phase 9 — Hospital-Stratified Training\n\n")
    lines.append("## 결과 (3 methods × 5 experiments)\n\n")
    lines.append("| 실험 | none (Phase 8) | mean centering | z-score |\n|---|---|---|---|\n")
    for k in ["A", "B", "C", "D", "E"]:
        line = f"| {k} | {fmt(get_auc(all_results['none'], k))} | "
        line += f"{fmt(get_auc(all_results['mean'], k))} | "
        line += f"{fmt(get_auc(all_results['zscore'], k))} |\n"
        lines.append(line)
    lines.append("\n## 해석 가이드\n\n")
    lines.append("- **A 가 크게 떨어졌다면** confound 제거 성공 — 정직한 random CV 메트릭.\n")
    lines.append("- **B 가 변하지 않았다면** Severance에서의 일반화는 hospital correction 영향 없음 (정직한 cross-hospital baseline).\n")
    lines.append("- **C, D, E는 변화 없음**이 정상 — 단일 병원 안의 실험이라 correction이 no-op.\n")
    lines.append("- **A → C 격차가 좁혀졌다면** confound가 거의 다 제거된 것.\n")
    lines.append("- **A 가 여전히 높다면** 단순 mean/scale 외에 nonlinear hospital effect 존재 → ComBat / DANN 필요.\n")
    (OUT_DIR / "hospital_stratified_report.md").write_text("".join(lines), encoding="utf-8")
    logger.info(f"\n  → {OUT_DIR / 'hospital_stratified_report.md'}")


if __name__ == "__main__":
    main()
