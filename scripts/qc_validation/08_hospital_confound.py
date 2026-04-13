"""
Phase 8 — Hospital/Protocol Confound Magnitude

We measure how much of the cancer-screening AUC comes from real biology
vs hospital/protocol confound.

Five experiments (binary classification, subject-level mean spectrum, LR):

    A. Naive random 5-fold CV (cancer vs control, all 12 groups)
       — current "production" practice; confounded baseline
    B. Leave-Severance-out (train on rest, test on YPAN+YNOR)
       — Severance is the only hospital with both cancer and control,
         so it's the only unbiased test set for cross-hospital generalization
    C. Within-Severance only (YPAN vs YNOR, 5-fold CV)
       — pure biology in one hospital, no batch confound
    D. Within-양산부산대 NOR vs (DIA∪HBP∪H.D.) — single-hospital "healthy
       vs chronic disease" — does SERS pick up metabolic signal at all?
    E. NOR-shuffle sanity check — split NOR 100 into two random halves,
       repeat 20×; AUC should be ~0.5. If >0.6 there is finer batch
       confound *within* 양산부산대 (measurement-day, lot, etc.).

Outputs
-------
results/qc_validation/hospital_confound.json
results/qc_validation/hospital_confound_report.md
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
logger = logging.getLogger("phase8")

OUT_DIR = Path(RESULTS_DIR) / "qc_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Subject caps to keep runtime manageable; deterministic by sample_id order.
GROUP_CAPS = {
    "NOR": 100, "DIA": 100, "HBP": 100, "H.D.": 100,        # 양산부산대 controls
    "PRO": 100, "BRE": 30, "OVA": 70, "LUN": 100,            # cancers
    "CRC": 100, "CPAN": 70, "BLC": 100,                       # cancers
    "YPAN": 30, "YNOR": 29,                                   # 세브란스 (both)
}

# Hospital map (from config.yaml dataset.group_metadata)
HOSPITAL = {
    "NOR": "양산부산대", "DIA": "양산부산대",
    "HBP": "양산부산대", "H.D.": "양산부산대",
    "PRO": "충북대", "CRC": "충북대",
    "CPAN": "충북대", "BLC": "충북대",
    "BRE": "부산백", "OVA": "부산백+서울대",
    "LUN": "서울대+성모",
    "YPAN": "세브란스", "YNOR": "세브란스",
}
CANCER_LABELS = {"PRO", "BRE", "OVA", "LUN", "CRC", "CPAN", "YPAN", "BLC"}
CONTROL_LABELS = {"NOR", "YNOR", "DIA", "HBP", "H.D."}


# ── Loading ────────────────────────────────────────────────────
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


def subject_matrix(processed: dict) -> tuple[np.ndarray, list[str], list[str]]:
    """Return (X[n_subj × n_feat], group_per_subject, sid_per_subject)."""
    subj = {}
    for (g, s, _), y in processed.items():
        subj.setdefault((g, s), []).append(y)
    rows, groups, sids = [], [], []
    for (g, s), reps in subj.items():
        rows.append(np.mean(np.vstack(reps), axis=0))
        groups.append(g)
        sids.append(s)
    return np.vstack(rows), groups, sids


def lr_cv_auc(X: np.ndarray, y: np.ndarray, n_splits: int = 5,
              random_state: int = 42) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    if min(np.bincount(y.astype(int))) < n_splits:
        n_splits = max(2, int(min(np.bincount(y.astype(int)))))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def lr_holdout_auc(X_tr, y_tr, X_te, y_te) -> float:
    if len(np.unique(y_te)) < 2:
        return float("nan")
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X_tr), y_tr)
    return float(roc_auc_score(y_te, clf.predict_proba(sc.transform(X_te))[:, 1]))


# ── Experiments ────────────────────────────────────────────────
def main():
    config = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    grid = make_fixed_grid(config)

    logger.info("[1] Loading subset ...")
    raw = load_subset(config)

    logger.info("[2] Preprocessing (production pipeline) ...")
    processed_dict, _, proc_grid = preprocess_spectra(raw, grid, config, qc_passed_keys=None)
    logger.info(f"  preprocessed: {len(processed_dict)}")

    X, groups, sids = subject_matrix(processed_dict)
    groups = np.array(groups)
    logger.info(f"  Subject matrix: {X.shape}")

    results = {}

    # ─── Experiment A: naive random CV (cancer vs control) ─────
    logger.info("\n[A] Naive random CV — cancer vs control (all hospitals)")
    mask_a = np.array([g in CANCER_LABELS or g in CONTROL_LABELS for g in groups])
    Xa = X[mask_a]
    ya = np.array([1 if g in CANCER_LABELS else 0 for g in groups[mask_a]])
    auc_a, std_a = lr_cv_auc(Xa, ya)
    logger.info(f"  n={len(ya)} (cancer={ya.sum()}, control={len(ya)-ya.sum()})  AUC = {auc_a:.4f} ± {std_a:.4f}")
    results["A_naive_random"] = {"auc": auc_a, "std": std_a, "n": int(len(ya)),
                                  "n_cancer": int(ya.sum()), "n_control": int(len(ya) - ya.sum())}

    # ─── Experiment B: leave-Severance-out ──────────────────────
    logger.info("\n[B] Leave-Severance-out generalization")
    sev = np.array([g in ("YPAN", "YNOR") for g in groups])
    in_task = mask_a  # cancer or control
    train_mask = in_task & ~sev
    test_mask = in_task & sev
    Xtr, Xte = X[train_mask], X[test_mask]
    ytr = np.array([1 if g in CANCER_LABELS else 0 for g in groups[train_mask]])
    yte = np.array([1 if g in CANCER_LABELS else 0 for g in groups[test_mask]])
    auc_b = lr_holdout_auc(Xtr, ytr, Xte, yte)
    logger.info(f"  train n={len(ytr)} (cancer={ytr.sum()})  test n={len(yte)} "
                f"(YPAN={int((groups[test_mask]=='YPAN').sum())}, YNOR={int((groups[test_mask]=='YNOR').sum())})")
    logger.info(f"  AUC = {auc_b:.4f}")
    results["B_leave_severance_out"] = {
        "auc": auc_b, "n_train": int(len(ytr)), "n_test": int(len(yte))
    }

    # ─── Experiment C: within-Severance only ────────────────────
    logger.info("\n[C] Within-Severance CV (YPAN vs YNOR)")
    Xc = X[sev]
    yc = np.array([1 if g == "YPAN" else 0 for g in groups[sev]])
    auc_c, std_c = lr_cv_auc(Xc, yc)
    logger.info(f"  n={len(yc)} (YPAN={yc.sum()}, YNOR={len(yc)-yc.sum()})  AUC = {auc_c:.4f} ± {std_c:.4f}")
    results["C_within_severance"] = {"auc": auc_c, "std": std_c, "n": int(len(yc))}

    # ─── Experiment D: within-양산부산대 NOR vs (DIA+HBP+H.D.) ─
    logger.info("\n[D] Within-양산부산대 NOR vs chronic disease")
    yangsan = np.array([g in ("NOR", "DIA", "HBP", "H.D.") for g in groups])
    Xd = X[yangsan]
    yd = np.array([1 if g != "NOR" else 0 for g in groups[yangsan]])
    auc_d, std_d = lr_cv_auc(Xd, yd)
    logger.info(f"  n={len(yd)} (chronic={yd.sum()}, NOR={len(yd)-yd.sum()})  AUC = {auc_d:.4f} ± {std_d:.4f}")
    results["D_within_yangsan_nor_vs_chronic"] = {"auc": auc_d, "std": std_d, "n": int(len(yd))}

    # ─── Experiment E: NOR-shuffle sanity check ────────────────
    logger.info("\n[E] NOR shuffle sanity check (20 repeats)")
    nor_mask = np.array([g == "NOR" for g in groups])
    Xn = X[nor_mask]
    n_nor = len(Xn)
    rng = np.random.RandomState(42)
    aucs = []
    for rep in range(20):
        # Random binary labels
        y_shuf = rng.permutation(n_nor) < (n_nor // 2)
        y_shuf = y_shuf.astype(int)
        a, _ = lr_cv_auc(Xn, y_shuf, random_state=rep)
        aucs.append(a)
    auc_e_mean = float(np.mean(aucs))
    auc_e_std = float(np.std(aucs))
    auc_e_max = float(np.max(aucs))
    logger.info(f"  n={n_nor}  AUC mean={auc_e_mean:.4f} ± {auc_e_std:.4f}  max={auc_e_max:.4f}")
    results["E_nor_shuffle"] = {"auc_mean": auc_e_mean, "auc_std": auc_e_std,
                                 "auc_max": auc_e_max, "n_repeats": 20, "n": int(n_nor)}

    # ─── Save ──────────────────────────────────────────────────
    (OUT_DIR / "hospital_confound.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Markdown report
    lines = []
    lines.append("# Hospital Confound 측정 (Phase 8)\n\n")
    lines.append("## 결과 요약\n\n")
    lines.append("| 실험 | n | AUC | 해석 |\n|---|---|---|---|\n")
    lines.append(f"| **A** Naive random CV | {results['A_naive_random']['n']} | "
                 f"**{auc_a:.4f}** ± {std_a:.4f} | 현재 관행 (confound 포함) |\n")
    lines.append(f"| **B** Leave-Severance-out | train={results['B_leave_severance_out']['n_train']}, "
                 f"test={results['B_leave_severance_out']['n_test']} | "
                 f"**{auc_b:.4f}** | 처음 보는 병원에서의 일반화 |\n")
    lines.append(f"| **C** Within-Severance CV | {results['C_within_severance']['n']} | "
                 f"**{auc_c:.4f}** ± {std_c:.4f} | confound-free biology |\n")
    lines.append(f"| **D** Within-양산부산대 NOR vs chronic | {results['D_within_yangsan_nor_vs_chronic']['n']} | "
                 f"**{auc_d:.4f}** ± {std_d:.4f} | 같은 병원 metabolic discrimination |\n")
    lines.append(f"| **E** NOR shuffle sanity (20 rep) | {results['E_nor_shuffle']['n']} | "
                 f"**{auc_e_mean:.4f}** ± {auc_e_std:.4f} (max {auc_e_max:.3f}) | within-batch 라벨 무작위 baseline |\n")

    lines.append("\n## 해석\n\n")
    lines.append(f"- **Gap A − C = {auc_a - auc_c:+.4f}** : naive AUC와 confound-free 사이의 거리. 양수일수록 현재 관행이 부풀어 있음.\n")
    lines.append(f"- **Gap A − B = {auc_a - auc_b:+.4f}** : 새 병원에서의 성능 손실.\n")
    lines.append(f"- **D 결과 ({auc_d:.4f})** : 같은 병원 안에서 metabolic class 구분이 가능한지.\n")
    lines.append(f"- **E 결과 ({auc_e_mean:.4f})** : 만약 0.55 이상이면 양산부산대 안에 finer batch confound 존재 → D 결과 신뢰 불가.\n\n")

    lines.append("## 결정 표\n\n")
    lines.append("| 시나리오 | 의미 | 다음 행동 |\n|---|---|---|\n")
    lines.append("| B ≪ A 그리고 C ≪ A | confound가 거의 모든 신호 | 도메인 적응 / Boramae 코호트 우선 |\n")
    lines.append("| B ≈ C ≪ A | confound가 부분적, biology 진짜 존재 | 병원-층화 학습으로 진짜 모델 재학습 |\n")
    lines.append("| B ≈ A | confound 없음 | (이 시나리오는 거의 가능성 없음) |\n")
    lines.append("| E > 0.55 | 양산부산대 안에 finer batch | day/lot 메타데이터 확보 → finer stratification |\n")

    (OUT_DIR / "hospital_confound_report.md").write_text("".join(lines), encoding="utf-8")
    logger.info(f"\n  → {OUT_DIR / 'hospital_confound_report.md'}")


if __name__ == "__main__":
    main()
