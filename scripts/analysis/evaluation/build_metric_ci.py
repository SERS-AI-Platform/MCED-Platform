#!/usr/bin/env python
"""예측값이 남아 있는 run에 대해 표준 스키마 CI 표를 생성한다.

각 run의 결과 폴더 **안에** `metrics_ci_standard.csv`로 저장한다. 코호트
정보를 행마다 넣기 때문에 나중에 여러 run을 모아도 출처가 섞이지 않는다.

예측값이 없는 run은 건드리지 않는다 — CI를 추정으로 채우지 않는다.

Usage:
    python scripts/analysis/evaluation/build_metric_ci.py --run yonsei
    python scripts/analysis/evaluation/build_metric_ci.py --run all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sers.evaluation.bootstrap import bootstrap_metric_ci
from sers.evaluation.metrics import BINARY_METRICS, MULTICLASS_METRICS
from sers.evaluation.predictions import PredictionSet, load_fixed_split_test, load_mapping_oof
from sers.evaluation.schema import (
    CohortSpec,
    Evaluation,
    MetricRow,
    Task,
    write_metrics_csv,
)

RESULTS = Path("results")


def yonsei() -> tuple[list[MetricRow], Path]:
    directory = RESULTS / "yonsei_prospective_current_model"
    pred = pd.read_csv(directory / "yonsei_oof_predictions.csv", encoding="utf-8-sig")
    cohort = CohortSpec(
        cohort_id="yonsei_prospective_2026",
        site="연세",
        cancer_groups=("YPAN",),
        control_groups=("YNOR",),
        n_patients=len(pred),
        n_positive=int((pred["true_label"] == 1).sum()),
        n_negative=int((pred["true_label"] == 0).sum()),
        n_spectra=int(pred["n_replicates"].sum()),
        notes="전향 검체. 췌장암 vs 정상. 환자 1명당 1행(replicate는 사전 집계).",
    )
    return _binary_rows(
        y_true=pred["true_label"].to_numpy(),
        y_score=pred["meta_lr_prob"].to_numpy(),
        groups=pred["subject_id"].to_numpy(),
        run_id="yonsei_prospective_current_model",
        model_name="STK-V2 (base 10 + LR meta)",
        evaluation=Evaluation.EXTERNAL,
        cohort=cohort,
    ), directory


def boramae() -> tuple[list[MetricRow], Path]:
    directory = RESULTS / "boramae_20260602_validation"
    pred = pd.read_csv(directory / "per_sample_results.csv", encoding="utf-8-sig")
    subject = pred["group"].astype(str) + "_" + pred["sample"].astype(str)
    cohort = CohortSpec(
        cohort_id="boramae_prospective_20260602",
        site="보라매",
        cancer_groups=("BPRO",),
        control_groups=("BNOR",),
        n_patients=len(pred),
        n_positive=int((pred["y_true"] == 1).sum()),
        n_negative=int((pred["y_true"] == 0).sum()),
        n_spectra=int(pred["qc_pass"].sum()),
        notes="전향 검체. 전립선암 vs 정상. 정상군 7명뿐 — 특이도 계열 지표는 검정력이 매우 낮다.",
    )
    # ⚠️ 이 run의 y_pred는 prob에 임계값을 적용해 재현되지 않는다
    #    (양성 최소 확률 0.5426 < 음성 최대 확률 0.5616 — 구간이 겹친다).
    #    즉 기록된 prob 컬럼만으로는 결정이 설명되지 않으므로, 임계값을
    #    추측하지 않고 **기록된 결정(y_pred)을 그대로** 쓴다.
    #    임계값 무관 지표(auc/pr_auc)만 연속 확률에서 계산한다.
    return _binary_rows(
        y_true=pred["y_true"].to_numpy(),
        y_score=pred["prob"].to_numpy(),
        groups=subject.to_numpy(),
        run_id="boramae_20260602_validation",
        model_name="STK-V2 production (balanced mode)",
        evaluation=Evaluation.EXTERNAL,
        cohort=cohort,
        decision=pred["y_pred"].to_numpy(),
    ), directory


#: 임계값과 무관한 지표 — 항상 연속 점수에서 계산한다.
THRESHOLD_FREE = frozenset({"auc", "pr_auc"})


def _binary_rows(*, y_true, y_score, groups, run_id, model_name, evaluation,
                 cohort: CohortSpec, n_boot: int = 2000,
                 metrics: dict | None = None,
                 decision: np.ndarray | None = None) -> list[MetricRow]:
    """`decision`이 주어지면 임계값 의존 지표는 그 결정을 그대로 사용한다.

    운영 임계값이 0.5가 아니거나, 기록된 결정이 확률로 재현되지 않는 run에
    필요하다. 0/1 벡터에 임계값 0.5를 적용하면 그 결정이 그대로 재현된다.
    """
    rows = []
    for name, fn in (metrics or BINARY_METRICS).items():
        score = y_score if (decision is None or name in THRESHOLD_FREE) else decision
        ci = bootstrap_metric_ci(y_true, score, groups, fn, name=name, n_boot=n_boot)
        rows.append(MetricRow.from_ci(
            ci, run_id=run_id, model_name=model_name, task=Task.CANCER_SCREENING,
            evaluation=evaluation, aggregation="patient", cohort=cohort,
        ))
    return rows


def boramae_109() -> tuple[list[MetricRow], Path]:
    """보라매 109명 — publication 분석. 48명 검증과는 **다른 실험**이다.

    48명(`boramae_20260602_validation`)은 타 병원에서 학습한 프로덕션 모델의
    외부 검증(evaluation=external)이고, 이 109명은 보라매 코호트 안에서 새로
    학습한 LR의 5-fold OOF(evaluation=oof)다. 같은 "보라매 AUC"로 보이지만
    측정 대상이 달라 나란히 비교하면 안 된다 — `check_comparable`이 막는다.
    """
    directory = Path("publications/전향검체/보라매병원/tables")
    pred = pd.read_csv(directory / "screening_binary_oof_predictions.csv", encoding="utf-8-sig")
    y = (pred["true_label"] == "Cancer").astype(int).to_numpy()
    decision = (pred["pred_label"] == "Cancer").astype(int).to_numpy()
    cohort = CohortSpec(
        cohort_id="boramae_prospective_2026_full",
        site="보라매",
        cancer_groups=("Prostate",),
        control_groups=("Control", "Biopsy-negative"),
        n_patients=len(pred),
        n_positive=int(y.sum()),
        n_negative=int((1 - y).sum()),
        notes="publication 분석 코호트. 생검음성 47명은 non-cancer로 분류. "
              "보라매 내부 5-fold OOF — 48명 외부검증과 다른 실험.",
    )
    return _binary_rows(
        y_true=y,
        y_score=pred["prob_Cancer"].to_numpy(),
        groups=pred["case_id"].to_numpy(),
        run_id="boramae_publication_screening_oof",
        model_name="LR (Boramae-trained, 5-fold OOF)",
        evaluation=Evaluation.OOF,
        cohort=cohort,
        decision=decision,
    ), directory


#: 현 단계에서 쓰는 run. external(외부 검증) 두 건은 사용자 결정으로 제외
#: (2026-09-07: "현단계에서 쓸 데이터가 아니야"). 빌더 코드는 남겨두되
#: 기본 실행(`--run all`)에서 빠지며, 명시적으로 이름을 줘야만 돈다.
def _from_prediction_set(ps: PredictionSet, n_boot: int = 2000) -> list[MetricRow]:
    metrics = BINARY_METRICS if ps.is_binary else MULTICLASS_METRICS
    rows = []
    for name, fn in metrics.items():
        score = ps.y_score
        if ps.decision is not None and name not in THRESHOLD_FREE and ps.is_binary:
            score = ps.decision
        ci = bootstrap_metric_ci(ps.y_true, score, ps.groups, fn, name=name, n_boot=n_boot)
        rows.append(MetricRow.from_ci(
            ci, run_id=ps.run_id, model_name=ps.model_name, task=ps.task,
            evaluation=ps.evaluation, aggregation="patient", cohort=ps.cohort,
        ))
    return rows


def mapping_runs() -> list[tuple[list[MetricRow], Path]]:
    out = []
    for d in sorted(RESULTS.glob("mapping_*")):
        if not (d / "patient_oof_predictions.csv").exists():
            continue
        rows = [r for ps in load_mapping_oof(d) for r in _from_prediction_set(ps)]
        out.append((rows, d))
    return out


def fixed_split_runs() -> list[tuple[list[MetricRow], Path]]:
    out = []
    for d in sorted((RESULTS / "training").glob("*")):
        if not (d / "fixed_test_predictions.npz").exists():
            continue
        rows = [r for ps in load_fixed_split_test(d) for r in _from_prediction_set(ps)]
        out.append((rows, d))
    return out


BUILDERS = {"boramae_109": boramae_109}
MULTI_BUILDERS = {"mapping": mapping_runs, "fixed_split": fixed_split_runs}
EXCLUDED_EXTERNAL = {"yonsei": yonsei, "boramae": boramae}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="all",
                        choices=[*BUILDERS, *MULTI_BUILDERS, *EXCLUDED_EXTERNAL, "all"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--to-db", action="store_true",
                        help="aecd_platform experiment.run_metrics에도 적재")
    args = parser.parse_args()
    tracker = None
    if args.to_db and not args.dry_run:
        from sers.preprocessing_lab.db import ExperimentTracker
        tracker = ExperimentTracker()

    if args.run == "all":
        jobs = [(n, b) for n, b in BUILDERS.items()]
        jobs += [(n, b) for n, b in MULTI_BUILDERS.items()]
    elif args.run in MULTI_BUILDERS:
        jobs = [(args.run, MULTI_BUILDERS[args.run])]
    else:
        jobs = [(args.run, {**BUILDERS, **EXCLUDED_EXTERNAL}[args.run])]

    for name, builder in jobs:
        produced = builder()
        if isinstance(produced, tuple):
            produced = [produced]
        for rows, directory in produced:
            _emit(name, rows, directory, args, tracker)


def _emit(name, rows, directory, args, tracker) -> None:
    if True:
        by_model = {}
        for r in rows:
            by_model.setdefault((r.model_name, str(r.task)), []).append(r)
        cohort = rows[0].cohort
        print(f"\n[{name}] {directory.name} — {cohort.site}, "
              f"환자 {cohort.n_patients}명 (암 {cohort.n_positive} / 정상 {cohort.n_negative}), "
              f"{rows[0].evaluation}")
        for (model, task), group in by_model.items():
            head = ", ".join(f"{r.metric}={r.value:.3f}[{r.ci_low:.3f},{r.ci_high:.3f}]"
                             for r in group[:2])
            print(f"    {model:<34} {task:<22} {head}")
        if args.dry_run:
            return
        out = directory / "metrics_ci_standard.csv"
        write_metrics_csv(rows, out)
        print(f"    → {out} ({len(rows)}행)")
        if tracker is not None:
            for group in by_model.values():
                tracker.load_metric_rows(group)
            print("    → aecd_platform experiment.run_metrics 적재")


if __name__ == "__main__":
    main()
