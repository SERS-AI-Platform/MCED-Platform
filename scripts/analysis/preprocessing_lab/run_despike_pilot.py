#!/usr/bin/env python
"""Preprocessing Lab 파일럿: Whitaker-Hayes despike on/off 비교.

설계 근거: docs/ml/preprocessing_design_guide.md ②단계
절차: SERS-AI/.claude/skills/preprocessing-lab/SKILL.md 5단계

데이터는 aecd_platform에서 읽고(파일 아님), 결과는 같은 DB의 experiment
스키마에 기록한다. 사용한 measurement는 experiment.run_measurements에 FK로
연결되므로 이 실행이 정확히 어떤 임상 레코드를 썼는지 추적된다.

저장소 규칙 준수:
- seed=42 고정
- StratifiedGroupKFold(subject 단위) — 같은 환자가 train/test에 갈리지 않게
- 개별 spectrum으로 학습하고, 평가에서만 환자 단위로 집계 (medoid/mean
  aggregation으로 학습하지 않는다)

파일럿 한정 단순화 (의도적, notes에 기록됨):
- mapping ablation의 fit_logistic은 GridSearchCV(내부 4-fold)를 돌지만 이
  파일럿은 C=1.0 고정 LR을 쓴다. 두 조건에 동일하게 적용되므로 조건 간
  비교에는 영향이 없고, 절대 성능은 튜닝된 값보다 낮을 수 있다.

Usage:
    source scripts/db/pghost.sh
    python scripts/analysis/preprocessing_lab/run_despike_pilot.py --dry-run
    python scripts/analysis/preprocessing_lab/run_despike_pilot.py
"""

from __future__ import annotations

import argparse
import dataclasses
import subprocess
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sers.aecd_api.loader import load_spectra_from_aecd
from sers.aecd_api.models import SpectrumFilters
from sers.aecd_api.repository import DatabaseSettings, PostgresAecdRepository
from sers.config import load_config
from sers.preprocessing import preprocess_spectra
from sers.preprocessing_lab.db import ExperimentTracker
from sers.qc.qc import apply_per_spectrum_corr_qc, apply_stage1_qc, enforce_min_replicates

SEED = 42
# 분석 대상 코호트를 명시적으로 고정한다. aecd_platform은 다른 세션이 계속
# 적재/재라벨링하는 살아있는 DB이므로(2026-09-02 'Drop' 라벨 신설로 control
# 환자 1명이 이동), 코호트를 열거하지 않으면 실행 시점마다 데이터가 달라진다.
COHORTS = ("prostate", "prostate disease control", "control")
CANCER_COHORT = "prostate"          # 양성 클래스 = 전립선암
CONDITIONS = (("despike_off", False), ("despike_on", True))


def load_all(repo) -> tuple[dict, list[int], dict]:
    """반환: (spectra, measurement_ids, 로드 시점 코호트 카운트)

    measurement_ids는 반드시 **이 로드와 같은 호출**에서 얻는다. 나중에 다시
    조회하면 그 사이 라벨이 바뀌어 실행 데이터와 기록이 어긋난다.
    """
    spectra, ids, observed = {}, [], {}
    for cohort in COHORTS:
        res = load_spectra_from_aecd(repo, SpectrumFilters(cohort_group=cohort), page_size=500)
        spectra.update(res.spectra)
        ids.extend(res.measurement_ids)
        observed[cohort] = len(res.spectra)
        print(f"  {cohort:26s} {len(res.spectra):6d} spectra")
    return spectra, ids, observed


def run_condition(raw, cfg, do_despike: bool):
    """Stage1 QC -> preprocess -> Stage2 QC. 반환: (X, y, groups, qc통계)"""
    passed, _ = apply_stage1_qc(raw)
    prep_cfg = dataclasses.replace(cfg.preprocessing, do_despike=do_despike)
    cfg2 = dataclasses.replace(cfg, preprocessing=prep_cfg)
    grid = np.linspace(402.0, 2198.0, 935)
    processed, _, proc_grid = preprocess_spectra(raw, grid, cfg2, qc_passed_keys=passed)
    final, _ = apply_per_spectrum_corr_qc(processed)
    final, _ = enforce_min_replicates(final)

    keys = sorted(final)
    X = np.vstack([processed[k] for k in keys])
    y = np.array([1 if k[0] == CANCER_COHORT else 0 for k in keys])
    groups = np.array([f"{k[0]}|{k[1]}" for k in keys])
    return X, y, groups, {"stage1_passed": len(passed), "final": len(keys),
                          "dropped": len(raw) - len(keys)}


def evaluate(X, y, groups):
    """개별 spectrum 학습 + 환자 단위 OOF 집계 평가."""
    oof = np.zeros(len(y), dtype=float)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, te in cv.split(X, y, groups):
        model = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=5000, class_weight="balanced",
                                      solver="lbfgs", random_state=SEED)),
        ])
        model.fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]

    spec_auc = float(roc_auc_score(y, oof))
    # 환자 단위: OOF 확률을 환자별 평균
    pat_prob, pat_label = [], []
    for g in np.unique(groups):
        m = groups == g
        pat_prob.append(oof[m].mean())
        pat_label.append(int(y[m][0]))
    pat_auc = float(roc_auc_score(pat_label, pat_prob))
    return {"spectrum_auc": spec_auc, "patient_auc": pat_auc,
            "n_patients": len(pat_label), "n_cancer_patients": int(sum(pat_label))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="DB에 기록하지 않음")
    args = ap.parse_args()

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip() or None
    cfg = load_config()
    repo = PostgresAecdRepository(DatabaseSettings.from_environment())

    print("데이터 로드 (aecd_platform)")
    raw, measurement_ids, observed = load_all(repo)
    print(f"  총 {len(raw)} spectra / measurement {len(measurement_ids)}건\n")

    tracker = None if args.dry_run else ExperimentTracker()
    results = {}
    for name, do_despike in CONDITIONS:
        print(f"[{name}] 전처리 (do_despike={do_despike})")
        X, y, groups, qc = run_condition(raw, cfg, do_despike)
        print(f"  QC 후 {qc['final']} spectra / {len(np.unique(groups))} 환자")
        metrics = evaluate(X, y, groups)
        print(f"  spectrum AUC={metrics['spectrum_auc']:.4f}  "
              f"patient AUC={metrics['patient_auc']:.4f}\n")
        results[name] = (metrics, qc)

        if tracker is None:
            continue
        run_name = f"despike_pilot_{name}_{datetime.now(timezone.utc):%Y%m%d}"
        method_id = None
        if do_despike:
            method_id = tracker.select_method_id("despike_whitaker_hayes")
        run_id = tracker.create_run(
            run_name, method_id=method_id, git_commit=commit,
            config_snapshot={"do_despike": do_despike, "grid": "402-2198/935",
                             "model": "LogisticRegression(C=1.0, balanced)", "seed": SEED},
            data_query_filters={"cohort_group": list(COHORTS),
                                "observed_at_load": observed},
            n_subjects=metrics["n_patients"], n_spectra=qc["final"],
            qc_passed_n=qc["final"], qc_failed_n=qc["dropped"],
            started_at=datetime.now(timezone.utc), status="complete",
            notes="파일럿. LR은 GridSearchCV 없이 C=1.0 고정 (두 조건 동일 적용).",
        )
        tracker.link_measurements(run_id, measurement_ids)
        tracker.record_metrics(run_id, [
            {"metric_name": "screening_auc_spectrum", "metric_value": metrics["spectrum_auc"],
             "split": "oof", "model_name": "logistic_regression"},
            {"metric_name": "screening_auc_patient", "metric_value": metrics["patient_auc"],
             "split": "oof", "model_name": "logistic_regression"},
        ])
        tracker.finish_run(run_id, "complete", datetime.now(timezone.utc))
        print(f"  → experiment.runs #{run_id} 기록\n")

    print("=" * 56)
    off, on = results["despike_off"][0], results["despike_on"][0]
    print(f"patient AUC:  off={off['patient_auc']:.4f}  on={on['patient_auc']:.4f}  "
          f"Δ={on['patient_auc'] - off['patient_auc']:+.4f}")
    print("판단(채택 여부)은 사용자 몫 — 이 스크립트는 수치만 낸다.")


if __name__ == "__main__":
    main()
