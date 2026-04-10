"""
Phase F — Within-Hospital Cancer-Type Classification

Tests whether SERS picks up real cancer-type biology independent of
hospital/protocol confound.

All classification is done WITHIN a single hospital (충북대 or 부산백) so
any discrimination cannot be hospital effect.

Subtests
--------
F1  충북대 4-class:   PRO vs CRC vs CPAN vs BLC (macro OvR AUC)
F2  충북대 binary:    PRO vs BLC   (intended as most-contrast pair)
F3  충북대 binary:    CRC vs CPAN  (adjacent-organ pair, harder)
F4  충북대 binary:    CRC vs BLC
F5  BRE vs OVA  (binary, note OVA spans 부산백+서울대 → mild hospital mix)

Sanity checks (S*) per subtest: randomly shuffle labels, repeat 20×, AUC should
be ~0.5. If >0.6 there is finer batch confound within the hospital.

Output
------
results/qc_validation/within_hospital_cancer.json
results/qc_validation/within_hospital_cancer_report.md
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
logger = logging.getLogger("phaseF")

OUT_DIR = Path(RESULTS_DIR) / "qc_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_CAPS = {
    "PRO": 100, "CRC": 100, "CPAN": 70, "BLC": 100,    # 충북대
    "BRE": 30, "OVA": 70,                               # 부산백 (OVA mixed)
}


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
    logger.info(f"Loaded: {len(raw)} spectra, "
                f"{len({(g, s) for g, s, _ in raw})} subjects")
    return raw


def subject_matrix(processed: dict):
    subj = {}
    for (g, s, _), y in processed.items():
        subj.setdefault((g, s), []).append(y)
    rows, groups = [], []
    for (g, s), reps in subj.items():
        rows.append(np.mean(np.vstack(reps), axis=0))
        groups.append(g)
    return np.vstack(rows), np.array(groups)


def lr_cv_binary(X, y, n_splits=5, random_state=42):
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    ns = min(n_splits, int(min(np.bincount(y.astype(int)))))
    ns = max(ns, 2)
    skf = StratifiedKFold(n_splits=ns, shuffle=True, random_state=random_state)
    aucs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def lr_cv_multiclass_macro(X, y, n_splits=5, random_state=42):
    """Macro one-vs-rest AUC for multiclass."""
    classes = np.unique(y)
    if len(classes) < 3:
        return float("nan"), float("nan")
    ns = min(n_splits, int(min([np.sum(y == c) for c in classes])))
    ns = max(ns, 2)
    skf = StratifiedKFold(n_splits=ns, shuffle=True, random_state=random_state)
    fold_aucs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="ovr").fit(Xtr, y[tr])
        proba = clf.predict_proba(Xte)
        per_class = []
        for i, c in enumerate(classes):
            y_bin = (y[te] == c).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                continue
            per_class.append(roc_auc_score(y_bin, proba[:, i]))
        if per_class:
            fold_aucs.append(np.mean(per_class))
    if not fold_aucs:
        return float("nan"), float("nan")
    return float(np.mean(fold_aucs)), float(np.std(fold_aucs))


def shuffle_baseline_binary(X, y, n_repeats=20, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(y)
    for rep in range(n_repeats):
        perm = rng.permutation(n) < (n // 2)
        a, _ = lr_cv_binary(X, perm.astype(int), random_state=rep)
        aucs.append(a)
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.max(aucs))


def shuffle_baseline_multi(X, y, n_repeats=20, seed=42):
    """Shuffle true labels to form a multiclass baseline."""
    rng = np.random.RandomState(seed)
    aucs = []
    for rep in range(n_repeats):
        y_sh = rng.permutation(y)
        a, _ = lr_cv_multiclass_macro(X, y_sh, random_state=rep)
        aucs.append(a)
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.max(aucs))


def main():
    config = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    grid = make_fixed_grid(config)

    logger.info("[1] Load + preprocess ...")
    raw = load_subset(config)
    processed, _, _ = preprocess_spectra(raw, grid, config, qc_passed_keys=None)
    X, groups = subject_matrix(processed)
    logger.info(f"  Subject matrix: {X.shape}")

    results = {}

    def run_binary(name, g1, g2, desc):
        mask = np.isin(groups, [g1, g2])
        Xb = X[mask]
        yb = (groups[mask] == g1).astype(int)
        if len(np.unique(yb)) < 2:
            logger.warning(f"  {name}: insufficient labels")
            return
        auc, std = lr_cv_binary(Xb, yb)
        sh_m, sh_s, sh_max = shuffle_baseline_binary(Xb, yb)
        logger.info(
            f"  {name} [{desc}]  n={len(yb)} ({g1}={int(yb.sum())}, {g2}={int(len(yb)-yb.sum())})"
        )
        logger.info(f"    real    AUC = {auc:.4f} ± {std:.4f}")
        logger.info(f"    shuffle AUC = {sh_m:.4f} ± {sh_s:.4f}  (max {sh_max:.3f})")
        results[name] = {
            "desc": desc, "n": int(len(yb)),
            "classes": {g1: int(yb.sum()), g2: int(len(yb) - yb.sum())},
            "auc": auc, "std": std,
            "shuffle_auc_mean": sh_m, "shuffle_auc_std": sh_s, "shuffle_auc_max": sh_max,
        }

    # F1 — 4-class within 충북대
    logger.info("\n[F1] 충북대 4-class: PRO vs CRC vs CPAN vs BLC")
    mask = np.isin(groups, ["PRO", "CRC", "CPAN", "BLC"])
    X4 = X[mask]
    y4 = groups[mask]
    auc_m, std_m = lr_cv_multiclass_macro(X4, y4)
    sh_m, sh_s, sh_max = shuffle_baseline_multi(X4, y4)
    logger.info(f"  n={len(y4)}")
    for c in np.unique(y4):
        logger.info(f"    {c}: {int((y4 == c).sum())}")
    logger.info(f"  real    macro-AUC = {auc_m:.4f} ± {std_m:.4f}")
    logger.info(f"  shuffle macro-AUC = {sh_m:.4f} ± {sh_s:.4f}  (max {sh_max:.3f})")
    results["F1_chungbuk_4class"] = {
        "desc": "충북대 PRO/CRC/CPAN/BLC 4-class macro OvR",
        "n": int(len(y4)),
        "classes": {c: int((y4 == c).sum()) for c in np.unique(y4)},
        "macro_auc": auc_m, "std": std_m,
        "shuffle_auc_mean": sh_m, "shuffle_auc_std": sh_s, "shuffle_auc_max": sh_max,
    }

    # F2 — F5 binaries
    run_binary("F2_chungbuk_PRO_vs_BLC", "PRO", "BLC", "충북대 극단 대비")
    run_binary("F3_chungbuk_CRC_vs_CPAN", "CRC", "CPAN", "충북대 인접 장기")
    run_binary("F4_chungbuk_CRC_vs_BLC", "CRC", "BLC", "충북대 비인접")
    run_binary("F5_BRE_vs_OVA", "BRE", "OVA", "부산백 (OVA는 부산백+서울대 혼합)")

    # Save
    (OUT_DIR / "within_hospital_cancer.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Report
    def fmt(v, p=4):
        return f"{v:.{p}f}" if isinstance(v, (int, float)) else "—"

    lines = []
    lines.append("# Phase F — Within-Hospital Cancer-Type Classification\n\n")
    lines.append("## 목적\n")
    lines.append("Hospital/protocol 교란이 없는 조건에서 SERS가 암종 특이적 biology를 잡는지 검증.\n\n")
    lines.append("## 결과\n\n")
    lines.append("| Subtest | n | real AUC | shuffle baseline | gap |\n|---|---|---|---|---|\n")

    for k, v in results.items():
        if "macro_auc" in v:
            real = v["macro_auc"]
            shuf = v["shuffle_auc_mean"]
        else:
            real = v["auc"]
            shuf = v["shuffle_auc_mean"]
        gap = real - shuf
        lines.append(
            f"| {k} ({v['desc']}) | {v['n']} | **{fmt(real)}** ± {fmt(v['std'],3)} | "
            f"{fmt(shuf)} (max {fmt(v['shuffle_auc_max'], 3)}) | **{fmt(gap)}** |\n"
        )

    lines.append("\n## 해석 가이드\n")
    lines.append("- **real − shuffle** 이 큼 (>0.15) → 실제 cancer-type biology signal 존재\n")
    lines.append("- **real ≈ shuffle (≈0.5)** → 같은 병원 안에서도 구분 불가 → SERS는 cancer 생화학 구분 못함 OR finer batch가 셔플 기반까지 오염\n")
    lines.append("- **shuffle max > 0.65** → finer batch confound 의심. 결과 신뢰도 낮춤\n\n")
    lines.append("## 의미\n")
    lines.append("- F가 높은 AUC + 낮은 shuffle 이면 Phase 8/9 결론 완화: 'SERS는 진짜 cancer biology를 본다, 다만 cross-hospital transfer를 못 한다'\n")
    lines.append("- F가 chance 수준이면 Phase 8/9 결론 강화: 'SERS 분류력 대부분이 hospital confound'\n")
    (OUT_DIR / "within_hospital_cancer_report.md").write_text("".join(lines), encoding="utf-8")
    logger.info(f"\n  → {OUT_DIR / 'within_hospital_cancer_report.md'}")


if __name__ == "__main__":
    main()
