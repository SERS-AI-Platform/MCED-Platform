#!/usr/bin/env python
"""Preprocessing Lab PL-3: 설계가이드 논문 기반 파이프라인을 조건 사다리로 비교.

설계 근거: docs/ml/preprocessing_design_guide.md §2 핵심 순서
    calibration → despike → truncation → smoothing → baseline → normalization
절차: SERS-AI/.claude/skills/preprocessing-lab/SKILL.md

PL-1(run_despike_pilot.py)의 모델·평가 규약을 그대로 쓰고, 조건만 가이드
순서대로 한 단계씩 더한다 (additive ladder). 두 코호트(aecd 전립선 3군,
보라매 매핑 113명)에 같은 QC·전처리·모델 코드를 적용한다.

조건 — 한 번에 한 요인만 바꾼다 (기준점 = cal_despike). 모두 trim 400–2200,
SG(config: 창 5, 차수 3), 개별 spectrum 학습, 환자 mean OOF:
    no_cal        : cal ✗  despike ✗  rolling_min  SNV   ← PL-1 despike_off 재현(단 SG 5)
    production    : cal ✓  despike ✗  rolling_min  SNV   ← 현재 프로덕션 (config.yaml)
    cal_despike   : cal ✓  despike ✓  rolling_min  SNV   ← 요인 교체의 기준점
    bl_airpls_1e{3..7} / bl_arpls_1e{3..7} / bl_als_1e{3..7}
                  : 기준점에서 baseline만 교체, λ 로그 그리드 (airPLS 감사 권고)
    norm_l2 / norm_minmax
                  : 기준점에서 정규화만 교체 (baseline은 rolling_min 유지)

⚠️ 게이트 면제: preprocessing-lab SKILL은 audit_status='pending'인 방법을 실행
단계로 넘기지 말라고 하지만, 사용자가 2026-09-08 탐색 실험으로 전부 실행하도록
결정했다. 감사 완료는 whitaker_hayes·airpls 2건뿐이며 audit_status는 건드리지
않는다. 결과는 채택 근거가 아니라 탐색 기록이다.

데이터: aecd_platform 전립선 3군만 쓴다. 매핑 폴더(data/mapping)는 같은 보라매
113검체×121점의 파일 사본이라 독립 재현이 아니고, xlsx 라벨은 DB에서 철회된
구 라벨(2026-09-02 'Drop' 1명)을 포함하므로 쓰지 않는다. `--cohort mapping`은
로더 일치 확인용으로만 남겨 두며 DB에는 적재하지 않는다.

calibration은 preprocess_spectra에 배선돼 있지 않으므로(preprocessing_methods
note 참조) 프로덕션 파이프라인(scripts/pipeline/run_qc_preprocess.py)과 같이
`calibrate_spectra_batch`를 Stage-1 QC 이전에 별도로 적용한다.

신뢰구간: seed 42 OOF 환자 점수에 환자 단위 BCa bootstrap(2000회). 5시드
mean/sd는 PL-1과 같은 metric 이름으로 보조 기록. 조건 간 차이는 같은 환자를
함께 재추출하는 짝지은 bootstrap으로 낸다. 채택 판단은 하지 않는다.

Usage:
    source scripts/db/pghost.sh
    python scripts/analysis/preprocessing_lab/run_guide_pipeline.py --cohort aecd \
        --conditions production,guide_airpls --seeds 42 --dry-run      # smoke
    python scripts/analysis/preprocessing_lab/run_guide_pipeline.py --cohort aecd
    python scripts/analysis/preprocessing_lab/run_guide_pipeline.py --cohort aecd --load-db
        # 저장된 예측 CSV로부터 CI 표를 만들고 DB에 적재 (재계산 없음)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "patent"))

from sers.config import load_config  # noqa: E402
from sers.evaluation.bootstrap import bootstrap_difference_ci, bootstrap_metric_ci  # noqa: E402
from sers.evaluation.metrics import BINARY_METRICS, roc_auc  # noqa: E402
from sers.evaluation.schema import (  # noqa: E402
    CohortSpec,
    Evaluation,
    MetricRow,
    Task,
    write_metrics_csv,
)
from sers.preprocessing import calibrate_spectra_batch, preprocess_spectra  # noqa: E402
from sers.qc.qc import (  # noqa: E402
    apply_per_spectrum_corr_qc,
    apply_stage1_qc,
    enforce_min_replicates,
)

SEEDS = (42, 7, 123, 2024, 31337)
PRIMARY_SEED = 42
N_BOOT = 2000
GRID = np.linspace(402.0, 2198.0, 935)
DATE_TAG = "20260907"
OUT_ROOT = REPO / "results" / "preprocessing_lab" / f"guide_pipeline_{DATE_TAG}"

# 코호트를 명시적으로 고정 (aecd_platform은 살아있는 DB — PL-1 주석 참조)
AECD_COHORTS = ("prostate", "prostate disease control", "control")
AECD_CANCER = "prostate"
MAPPING_CANCER = "Prostate cancer"

# 조건. 각 항목: config 덮어쓰기, calibration 여부, 연결할 method_key, 설명
_BASE = dict(do_despike=True, baseline_method="rolling_min", normalization="snv")
LAMBDA_GRID = (1e3, 1e4, 1e5, 1e6, 1e7)


def _lam_tag(lam: float) -> str:
    return f"1e{int(round(np.log10(lam)))}"


CONDITIONS: dict[str, dict] = {
    "no_cal": dict(
        overrides=dict(do_despike=False, baseline_method="rolling_min", normalization="snv"),
        calibrate=False, method_key=None,
        label="PL-1 재현: cal✗ despike✗ rolling_min SNV (SG 5)",
    ),
    "production": dict(
        overrides=dict(do_despike=False, baseline_method="rolling_min", normalization="snv"),
        calibrate=True, method_key="calibration_astm_reference",
        label="현재 프로덕션: cal✓ despike✗ rolling_min SNV",
    ),
    "cal_despike": dict(
        overrides=dict(_BASE),
        calibrate=True, method_key="despike_whitaker_hayes",
        label="기준점: cal✓ + Whitaker–Hayes despike(z=6) rolling_min SNV",
    ),
}
for _method, _param in (("airpls", "baseline_airpls_lam"), ("arpls", "baseline_arpls_lam"),
                        ("als", "baseline_als_lam")):
    for _lam in LAMBDA_GRID:
        CONDITIONS[f"bl_{_method}_{_lam_tag(_lam)}"] = dict(
            overrides={**_BASE, "baseline_method": _method, _param: _lam},
            calibrate=True, method_key=_method,
            label=f"기준점에서 baseline={_method} λ={_lam_tag(_lam)}",
        )
for _norm in ("l2", "minmax"):
    CONDITIONS[f"norm_{_norm}"] = dict(
        overrides={**_BASE, "normalization": _norm},
        calibrate=True, method_key=_norm,
        label=f"기준점에서 normalization={_norm} (rolling_min)",
    )

#: 짝지은 Δ를 낼 기준. production = 현재 배포 파이프라인, cal_despike = 요인 교체의 기준점
DELTA_REFERENCES = ("production", "cal_despike")


# ---------------------------------------------------------------------------
# 데이터 로드 — 두 코호트를 같은 dict[(group, subject, rep)] 형식으로
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True, slots=True)
class Cohort:
    name: str
    site: str
    spectra: dict
    cancer_group: str
    control_groups: tuple[str, ...]
    measurement_ids: list[int] | None      # aecd만 FK 연결
    data_query_filters: dict


def load_aecd() -> Cohort:
    from sers.aecd_api.loader import load_spectra_from_aecd
    from sers.aecd_api.models import SpectrumFilters
    from sers.aecd_api.repository import DatabaseSettings, PostgresAecdRepository

    repo = PostgresAecdRepository(DatabaseSettings.from_environment())
    spectra, ids, observed = {}, [], {}
    for cohort in AECD_COHORTS:
        res = load_spectra_from_aecd(repo, SpectrumFilters(cohort_group=cohort), page_size=500)
        spectra.update(res.spectra)
        ids.extend(res.measurement_ids)
        observed[cohort] = len(res.spectra)
        print(f"  {cohort:26s} {len(res.spectra):6d} spectra")
    return Cohort(
        name="aecd", site="aecd_platform(전립선 3군)", spectra=spectra,
        cancer_group=AECD_CANCER,
        control_groups=tuple(c for c in AECD_COHORTS if c != AECD_CANCER),
        measurement_ids=ids,
        data_query_filters={"cohort_group": list(AECD_COHORTS), "observed_at_load": observed},
    )


def load_mapping() -> Cohort:
    from mapping_repeat_average_core import GROUP_ORDER, load_subjects

    subjects, raw_axis = load_subjects(GRID)
    spectra = {}
    counts = {g: 0 for g in GROUP_ORDER}
    for s in subjects:
        counts[s.group] += 1
        x = np.asarray(s.native_axis, dtype=float)
        for rep, y in enumerate(np.asarray(s.native_replicates, dtype=float), start=1):
            spectra[(s.group, s.ordinal, rep)] = (x.copy(), y.copy())
    print(f"  mapping: {len(subjects)} subjects / {len(spectra)} spectra  {counts}")
    return Cohort(
        name="mapping", site="보라매(매핑 121점)", spectra=spectra,
        cancer_group=MAPPING_CANCER,
        control_groups=tuple(g for g in GROUP_ORDER if g != MAPPING_CANCER),
        measurement_ids=None,
        data_query_filters={"source": "data/mapping/2026*_mapping + clinical_df.xlsx",
                            "raw_axis": json.loads(json.dumps(raw_axis, default=float)),
                            "group_counts": counts},
    )


# ---------------------------------------------------------------------------
# 전처리 + QC (PL-1과 동일한 체인, calibration만 앞에 추가)
# ---------------------------------------------------------------------------

def run_condition(cohort: Cohort, cfg, spec: dict):
    raw = cohort.spectra
    shift_df = None
    if spec["calibrate"]:
        raw, shift_df = calibrate_spectra_batch(
            raw, target_wn=cfg.preprocessing.calibration_reference_wn,
            window=cfg.preprocessing.calibration_window,
        )
    passed, _ = apply_stage1_qc(raw)
    prep_cfg = dataclasses.replace(cfg.preprocessing, **spec["overrides"])
    cfg2 = dataclasses.replace(cfg, preprocessing=prep_cfg)
    processed, _, _ = preprocess_spectra(raw, GRID, cfg2, qc_passed_keys=passed)
    final, _ = apply_per_spectrum_corr_qc(processed)
    final, _ = enforce_min_replicates(final)

    keys = sorted(final)
    X = np.vstack([processed[k] for k in keys]).astype(np.float64)
    finite = np.isfinite(X).all(axis=1)
    if not finite.all():
        print(f"  ⚠️ non-finite spectra dropped: {(~finite).sum()}")
        keys = [k for k, f in zip(keys, finite) if f]
        X = X[finite]
    y = np.array([1 if k[0] == cohort.cancer_group else 0 for k in keys])
    groups = np.array([f"{k[0]}|{k[1]}" for k in keys])
    qc = {"input": len(raw), "stage1_passed": len(passed), "final": len(keys),
          "dropped": len(raw) - len(keys), "n_patients": int(len(np.unique(groups)))}
    return X, y, groups, qc, shift_df, prep_cfg


def evaluate(X, y, groups, seed: int) -> np.ndarray:
    """spectrum OOF 확률 (seed 고정)."""
    oof = np.zeros(len(y), dtype=float)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y, groups):
        model = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=5000, class_weight="balanced",
                                      solver="lbfgs", random_state=seed)),
        ])
        model.fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
    return oof


def patient_table(oof: np.ndarray, y: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"subject": groups, "true_label": y, "prob": oof})
    out = frame.groupby("subject").agg(true_label=("true_label", "first"),
                                       prob=("prob", "mean"), n_spectra=("prob", "size"))
    return out.reset_index()


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def run_name(cohort: str, cond: str) -> str:
    return f"guide_{cohort}_{cond}_{DATE_TAG}"


def compute(cohort: Cohort, conditions: list[str], seeds: tuple[int, ...], cfg, dry_run: bool):
    for cond in conditions:
        spec = CONDITIONS[cond]
        out_dir = OUT_ROOT / cohort.name / cond
        t0 = time.time()
        started_at = datetime.now(timezone.utc).isoformat()
        print(f"\n[{cohort.name}/{cond}] {spec['label']}")
        X, y, groups, qc, shift_df, prep_cfg = run_condition(cohort, cfg, spec)
        print(f"  QC: {qc}   전처리 {time.time() - t0:.0f}s")

        tables, seed_auc = [], {}
        spectrum_primary = None
        for seed in seeds:
            t1 = time.time()
            oof = evaluate(X, y, groups, seed)
            pt = patient_table(oof, y, groups)
            pt.insert(0, "seed", seed)
            tables.append(pt)
            seed_auc[seed] = float(roc_auc_score(pt["true_label"], pt["prob"]))
            if seed == PRIMARY_SEED:
                spectrum_primary = pd.DataFrame({"key": [f"{k}" for k in groups],
                                                 "true_label": y, "prob": oof})
            print(f"  seed {seed:>5}: patient AUC={seed_auc[seed]:.4f}  "
                  f"spectrum AUC={roc_auc_score(y, oof):.4f}  ({time.time() - t1:.0f}s)")
        if dry_run:
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(tables).to_csv(out_dir / "patient_oof_predictions.csv",
                                 index=False, encoding="utf-8-sig")
        if spectrum_primary is not None:
            spectrum_primary.to_csv(out_dir / f"spectrum_oof_seed{PRIMARY_SEED}.csv",
                                    index=False, encoding="utf-8-sig")
        if shift_df is not None:
            shift_df.to_csv(out_dir / "calibration_shifts.csv", index=False, encoding="utf-8-sig")
        meta = {
            "run_name": run_name(cohort.name, cond),
            "cohort": cohort.name, "site": cohort.site, "condition": cond,
            "label": spec["label"], "method_key": spec["method_key"],
            "calibrate": spec["calibrate"],
            "preprocessing": {k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in dataclasses.asdict(prep_cfg).items()
                              if isinstance(v, (int, float, str, bool, tuple, list)) or v is None},
            "grid": "402-2198/935",
            "model": "LogisticRegression(C=1.0, balanced, lbfgs) + StandardScaler",
            "cv": "StratifiedGroupKFold(5) by subject, spectrum-level train, patient mean OOF",
            "seeds": list(seeds), "primary_seed": PRIMARY_SEED,
            "qc": qc,
            "calibration_shift_cm1": (None if shift_df is None else {
                "mean": float(shift_df.shift_cm1.mean()), "sd": float(shift_df.shift_cm1.std()),
                "abs_max": float(shift_df.shift_cm1.abs().max()),
                "n_zero": int((shift_df.shift_cm1 == 0).sum())}),
            "seed_patient_auc": seed_auc,
            "patient_auc_mean5": float(np.mean(list(seed_auc.values()))),
            "patient_auc_sd5": float(np.std(list(seed_auc.values()), ddof=1)) if len(seeds) > 1 else None,
            "data_query_filters": cohort.data_query_filters,
            "measurement_ids_n": None if cohort.measurement_ids is None else len(cohort.measurement_ids),
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        (out_dir / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        if cohort.measurement_ids is not None:
            np.save(out_dir / "measurement_ids.npy", np.asarray(cohort.measurement_ids, dtype=np.int64))
        print(f"  → {out_dir}")


# ---------------------------------------------------------------------------
# CI 표 + DB 적재 (저장된 예측에서만 계산 — 재학습 없음)
# ---------------------------------------------------------------------------

def _cohort_spec(meta: dict, pt: pd.DataFrame, cohort_key: str) -> CohortSpec:
    if cohort_key == "aecd":
        cancer, control = (AECD_CANCER,), tuple(c for c in AECD_COHORTS if c != AECD_CANCER)
        cohort_id, site = f"aecd_prostate3_{DATE_TAG}", "aecd_platform"
        note = ("aecd_platform 전립선 3군(prostate / prostate disease control / control) = 보라매 "
                "113검체×121점 매핑 데이터의 DB 사본(2026-09-02 Drop 1명 제외). 양성=전립선암.")
    else:
        cancer, control = (MAPPING_CANCER,), ("Control", "Prostate disease control")
        cohort_id, site = "boramae_mapping_2026", "보라매"
        note = "보라매 매핑 코호트(121점/검체). 양성=전립선암. 기존 매핑 run과 달리 Stage1/corr QC 적용."
    return CohortSpec(
        cohort_id=cohort_id, site=site, cancer_groups=cancer, control_groups=control,
        n_patients=len(pt), n_positive=int((pt.true_label == 1).sum()),
        n_negative=int((pt.true_label == 0).sum()), n_spectra=int(pt.n_spectra.sum()),
        notes=note + f" 조건={meta['condition']}: {meta['label']}. LR C=1.0 고정(PL-1 규약).",
    )


def build_ci(cohort_key: str, conditions: list[str], load_db: bool) -> pd.DataFrame:
    tracker = None
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip() or None
    if load_db:
        from sers.preprocessing_lab.db import ExperimentTracker
        tracker = ExperimentTracker()

    summary, primary = [], {}
    for cond in conditions:
        out_dir = OUT_ROOT / cohort_key / cond
        if not (out_dir / "run_metadata.json").exists():
            print(f"  (skip) {out_dir} 결과 없음")
            continue
        meta = json.loads((out_dir / "run_metadata.json").read_text())
        allp = pd.read_csv(out_dir / "patient_oof_predictions.csv", encoding="utf-8-sig")
        pt = allp[allp.seed == PRIMARY_SEED].sort_values("subject").reset_index(drop=True)
        cohort = _cohort_spec(meta, pt, cohort_key)
        rname = meta["run_name"]
        rows = []
        for name, fn in BINARY_METRICS.items():
            ci = bootstrap_metric_ci(pt.true_label.to_numpy(), pt.prob.to_numpy(),
                                     pt.subject.to_numpy(), fn, name=name, n_boot=N_BOOT)
            rows.append(MetricRow.from_ci(
                ci, run_id=rname, model_name="LR C=1.0 (spectrum→patient mean)",
                task=Task.CANCER_SCREENING, evaluation=Evaluation.OOF,
                aggregation="patient", cohort=cohort))
        write_metrics_csv(rows, out_dir / "metrics_ci_standard.csv")
        primary[cond] = pt
        auc = rows[0]
        summary.append({
            "cohort": cohort_key, "condition": cond, "label": meta["label"], "run_name": rname,
            "n_patients": cohort.n_patients, "n_positive": cohort.n_positive,
            "n_spectra": cohort.n_spectra,
            "auc_seed42": auc.value, "auc_ci_low": auc.ci_low, "auc_ci_high": auc.ci_high,
            "auc_mean5": meta["patient_auc_mean5"], "auc_sd5": meta["patient_auc_sd5"],
            "calibration_abs_max_shift": (meta["calibration_shift_cm1"] or {}).get("abs_max"),
        })
        print(f"  {cohort_key}/{cond:<14} AUC={auc.value:.3f} [{auc.ci_low:.3f},{auc.ci_high:.3f}]"
              f"  5시드 {meta['patient_auc_mean5']:.3f}±{(meta['patient_auc_sd5'] or 0):.3f}")

        if tracker is not None:
            _load_run(tracker, meta, rows, out_dir, commit)

    # 조건 간 짝지은 차이 — 같은 환자를 함께 재추출. 기준 2개(production, cal_despike)
    for ref in DELTA_REFERENCES:
        if ref not in primary:
            continue
        base = primary[ref]
        for s in summary:
            cond = s["condition"]
            if cond == ref:
                s.update({f"d_{ref}": 0.0, f"d_{ref}_lo": 0.0, f"d_{ref}_hi": 0.0})
                continue
            merged = base.merge(primary[cond], on="subject", suffixes=("_b", "_c"))
            d = bootstrap_difference_ci(
                merged.true_label_b.to_numpy(), merged.prob_c.to_numpy(), merged.prob_b.to_numpy(),
                merged.subject.to_numpy(), roc_auc, name=f"auc_delta_vs_{ref}", n_boot=N_BOOT)
            s.update({f"d_{ref}": d.value, f"d_{ref}_lo": d.ci_low, f"d_{ref}_hi": d.ci_high,
                      f"d_{ref}_n": len(merged)})
    return pd.DataFrame(summary)


def _load_run(tracker, meta: dict, rows: list[MetricRow], out_dir: Path, commit: str | None) -> None:
    """runs 행을 method_id·config와 함께 만들고(없을 때만), CI 행과 시드 행을 얹는다."""
    rname = meta["run_name"]
    with tracker._connect() as conn, conn.cursor() as cur:  # noqa: SLF001 — 조회 전용
        cur.execute("SELECT run_id FROM experiment.runs WHERE run_name = %s", (rname,))
        hit = cur.fetchone()
    if hit is None:
        method_id = tracker.select_method_id(meta["method_key"]) if meta["method_key"] else None
        run_id = tracker.create_run(
            rname, method_id=method_id, git_commit=commit,
            config_snapshot={"condition": meta["condition"], "preprocessing": meta["preprocessing"],
                             "calibrate": meta["calibrate"], "grid": meta["grid"],
                             "model": meta["model"], "cv": meta["cv"], "seeds": meta["seeds"],
                             "primary_seed": meta["primary_seed"],
                             "calibration_shift_cm1": meta["calibration_shift_cm1"]},
            data_query_filters=meta["data_query_filters"],
            n_subjects=meta["qc"]["n_patients"], n_spectra=meta["qc"]["final"],
            qc_passed_n=meta["qc"]["final"], qc_failed_n=meta["qc"]["dropped"],
            started_at=meta.get("started_at", meta["finished_at"]), status="complete",
            notes=f"PL-3 설계가이드 요인별 비교 [{meta['cohort']}] {meta['label']}. "
                  f"LR C=1.0 고정(PL-1 규약). 탐색 실험 — audit gate 면제(사용자 결정 2026-09-08), "
                  f"채택 근거 아님. 산출물 {out_dir.relative_to(REPO)}",
        )
        # create_run은 코호트 배열 컬럼을 넣지 않고 load_metric_rows의 ON CONFLICT도
        # 갱신하지 않으므로 여기서 직접 채운다 (01_schema.sql: 비교 가능성 판단 근거).
        cohort = rows[0].cohort
        with tracker._connect() as conn, conn.cursor() as cur:  # noqa: SLF001
            cur.execute(
                "UPDATE experiment.runs SET cancer_types=%s, non_cancer_groups=%s, "
                "aggregation=%s, phase=%s, variable=%s, baseline=%s WHERE run_id=%s",
                (list(cohort.cancer_groups), list(cohort.control_groups), "patient", "PL-3",
                 meta["condition"], "cal_despike", run_id),
            )
            conn.commit()
        ids_path = out_dir / "measurement_ids.npy"
        if ids_path.exists():
            tracker.link_measurements(run_id, [int(i) for i in np.load(ids_path)])
        tracker.finish_run(run_id, "complete", meta["finished_at"])
    else:
        run_id = hit[0]
    tracker.load_metric_rows(rows, git_commit=commit)
    seed_rows = [{"metric_name": f"screening_auc_patient_seed{s}", "metric_value": v,
                  "split": "oof", "model_name": "logistic_regression"}
                 for s, v in meta["seed_patient_auc"].items()]
    seed_rows += [
        {"metric_name": "screening_auc_patient_mean5seeds", "metric_value": meta["patient_auc_mean5"],
         "split": "oof", "model_name": "logistic_regression"},
        {"metric_name": "screening_auc_patient_sd5seeds", "metric_value": meta["patient_auc_sd5"],
         "split": "oof", "model_name": "logistic_regression"},
    ]
    tracker.record_metrics(run_id, [r for r in seed_rows if r["metric_value"] is not None])
    print(f"    → experiment.runs #{run_id} ({rname})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", choices=["aecd", "mapping", "both"], default="both")
    ap.add_argument("--conditions", default=",".join(CONDITIONS),
                    help="쉼표 구분. 기본: 전부")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--dry-run", action="store_true", help="계산만 하고 아무것도 저장하지 않음")
    ap.add_argument("--load-db", action="store_true",
                    help="저장된 예측에서 CI 표를 만들고 aecd_platform experiment 스키마에 적재")
    ap.add_argument("--ci-only", action="store_true",
                    help="학습 없이 저장된 예측에서 CI 표·summary만 다시 만듦 (DB 미적재)")
    args = ap.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise SystemExit(f"unknown conditions: {sorted(unknown)}")
    seeds = tuple(int(s) for s in args.seeds.split(","))
    if PRIMARY_SEED not in seeds:
        raise SystemExit(f"--seeds must include primary seed {PRIMARY_SEED}")
    cohorts = ["aecd", "mapping"] if args.cohort == "both" else [args.cohort]

    if not (args.load_db or args.ci_only):
        cfg = load_config()
        for key in cohorts:
            print(f"\n=== 데이터 로드: {key} ===")
            cohort = load_aecd() if key == "aecd" else load_mapping()
            compute(cohort, conditions, seeds, cfg, args.dry_run)
        if args.dry_run:
            print("\n(dry-run) 저장·적재 없음")
            return

    print("\n=== CI 표 / summary ===")
    frames = [build_ci(key, conditions, load_db=args.load_db and key == "aecd") for key in cohorts]
    summary = pd.concat([f for f in frames if len(f)], ignore_index=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / "summary.csv"
    if path.exists():
        old = pd.read_csv(path, encoding="utf-8-sig")
        old = old[~old.set_index(["cohort", "condition"]).index.isin(
            summary.set_index(["cohort", "condition"]).index)]
        summary = pd.concat([old, summary], ignore_index=True)
    summary.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n→ {path}")
    print("판단(채택 여부)은 사용자 몫 — 이 스크립트는 수치만 낸다.")


if __name__ == "__main__":
    main()
