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
    trim_600_1800 : 기준점에서 ③ 절단만 가이드 예시(600–1800)로 교체
    sm_none / sm_sg11 / sm_sg21 / sm_median5 / sm_gauss1 / sm_wavelet
                  : 기준점에서 ④ smoothing만 교체 (기준점은 SG 창 5)
    order_baseline_first
                  : 기준점에서 ④↔⑤ 순서만 교체 (baseline → smooth)
    cal_ps        : 표준물질(PS) run 단위 축 보정만 (urea ✗ despike ✗)  ← 2026-09-09 추가
    cal_ps_urea   : PS 축 보정 후 urea 피크 보정 (despike ✗)            ← 2026-09-09 추가
    cal_ps_si     : PS 8피크 + Si 520.7 피크로 run 단위 1차(offset+slope) 축 보정, urea ✗
    cal_ps_si_sg11: 위와 같고 SG 창만 11                                 ← 2026-09-09 사용자 결정
                    ("urea 보정은 화학 신호를 볼 수 있으므로 PS/Si 표준물질로만 보정")

표준물질 보정(2026-09-09 추가): `measurement.calibrations`(PS, raman_shift, pass)의
run(측정일+장비) 단위 global_shift를 x축에서 빼는 것(x − shift)이며, AECD API
파이프라인(scripts/analysis/aecd_api_model_mean_spectrum_clinical_performance.py)의
정의와 같다. 기존 "production"의 calibration은 검체 내 urea 1001.4 피크 기준
스펙트럼별 보정(config.yaml)이고, DB method_key `calibration_astm_reference`와는
다른 알고리즘임에 주의 — PL-3 초기 기록의 명칭 불일치.

과제(task) 두 개를 같은 조건·같은 전처리 결과로 낸다:
    cancer_screening     — 전립선암 vs 비암(질환대조+control), 이진 LR
    prostate_three_class — control / 질환대조 / 전립선암, multinomial LR, macro OvR AUC

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

from joblib import Parallel, delayed  # noqa: E402

from sers.config import PreprocessingConfig, load_config  # noqa: E402
from sers.evaluation.bootstrap import bootstrap_difference_ci, bootstrap_metric_ci  # noqa: E402
from sers.evaluation.metrics import (  # noqa: E402
    BINARY_METRICS,
    MULTICLASS_METRICS,
    macro_roc_auc,
    roc_auc,
)
from sers.evaluation.schema import (  # noqa: E402
    CohortSpec,
    Evaluation,
    MetricRow,
    Task,
    write_metrics_csv,
)
from sers.preprocessing import (  # noqa: E402
    apply_standard_material_axis,
    calibrate_spectra_batch,
    fit_standard_material_axis,
    preprocess_spectra,
)
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

CONDITIONS["trim_600_1800"] = dict(
    overrides={**_BASE, "trim_region": (600.0, 1800.0)},
    calibrate=True, method_key="truncation_fingerprint",
    label="기준점에서 truncation=600–1800 (가이드 ③ 예시)",
)
for _key, _ov, _lab in (
    ("sm_none", {"do_smooth": False}, "smoothing 없음"),
    ("sm_sg11", {"smooth_window": 11}, "SG 창 11 (PL-1 당시 값)"),
    ("sm_sg21", {"smooth_window": 21}, "SG 창 21"),
    ("sm_median5", {"smoothing_method": "median", "median_window": 5}, "median 창 5"),
    ("sm_gauss1", {"smoothing_method": "gaussian", "gaussian_sigma": 1.0}, "gaussian σ=1"),
    ("sm_wavelet", {"smoothing_method": "wavelet_haar"}, "Haar wavelet threshold"),
):
    CONDITIONS[_key] = dict(
        overrides={**_BASE, **_ov},
        calibrate=True, method_key="savgol" if _key.startswith("sm_sg") else None,
        label=f"기준점에서 smoothing={_lab}",
    )
CONDITIONS["no_cal_sg11"] = dict(
    overrides=dict(do_despike=False, baseline_method="rolling_min", normalization="snv",
                   smooth_window=11),
    calibrate=False, method_key=None,
    label="PL-1 despike_off 정확 재현: cal✗ despike✗ SG 11 rolling_min SNV",
)
CONDITIONS["cal_ps"] = dict(
    overrides=dict(do_despike=False, baseline_method="rolling_min", normalization="snv"),
    calibrate=False, ps_calibrate=True, method_key="calibration_astm_reference",
    label="표준물질(PS) run 단위 축 보정만: ps✓ urea✗ despike✗ rolling_min SNV",
)
CONDITIONS["cal_ps_urea"] = dict(
    overrides=dict(do_despike=False, baseline_method="rolling_min", normalization="snv"),
    calibrate=True, ps_calibrate=True, method_key="calibration_astm_reference",
    label="PS 축 보정 후 urea 피크 보정: ps✓ urea✓ despike✗ rolling_min SNV",
)
CONDITIONS["cal_ps_si"] = dict(
    overrides=dict(do_despike=False, baseline_method="rolling_min", normalization="snv"),
    calibrate=False, ps_calibrate="linear", method_key="calibration_astm_reference",
    label="PS+Si 표준물질 run 단위 1차 축 보정(offset+slope): urea✗ despike✗ rolling_min SNV (SG 5)",
)
CONDITIONS["cal_ps_si_sg11"] = dict(
    overrides=dict(do_despike=False, baseline_method="rolling_min", normalization="snv",
                   smooth_window=11),
    calibrate=False, ps_calibrate="linear", method_key="calibration_astm_reference",
    label="PS+Si 표준물질 run 단위 1차 축 보정: urea✗ despike✗ rolling_min SNV (SG 11)",
)
CONDITIONS["order_baseline_first"] = dict(
    overrides={**_BASE, "baseline_before_smooth": True},
    calibrate=True, method_key=None,
    label="기준점에서 순서만 baseline → smooth (가이드 §2 '두 순서 다')",
)


# ---------------------------------------------------------------------------
# 2026-09-09 사용자 결정("축 보정은 PS+Si 표준물질로만") 이후의 사다리: 기존 요인 조건을
# urea 보정 없이 PS+Si 1차 축 보정 위에서 재실행한다. 기준점 = ps_despike
# (PS+Si ✓, WH despike ✓, rolling_min, SNV, SG 5 = 기존 cal_despike에서 urea→PS+Si만 교체).
# ---------------------------------------------------------------------------
_PS_SKIP = {"no_cal", "production", "no_cal_sg11", "cal_ps", "cal_ps_urea", "cal_ps_si", "cal_ps_si_sg11"}
for _name, _spec in list(CONDITIONS.items()):
    if _name in _PS_SKIP or not _spec["calibrate"]:
        continue
    _new = "ps_despike" if _name == "cal_despike" else f"ps_{_name}"
    CONDITIONS[_new] = dict(
        overrides=dict(_spec["overrides"]),
        calibrate=False, ps_calibrate="linear",
        method_key=_spec["method_key"] or "calibration_astm_reference",
        label=f"[PS+Si 축보정, urea✗] {_spec['label']}",
    )
# SG 창 5 vs 11 결정용 중간 창 (2026-09-09 사용자 요청) — PS+Si 기준점에서 smoothing 창만 교체
for _w in (7, 9):
    CONDITIONS[f"ps_sm_sg{_w}"] = dict(
        overrides={**_BASE, "smooth_window": _w},
        calibrate=False, ps_calibrate="linear", method_key="savgol",
        label=f"[PS+Si 축보정, urea✗] 기준점에서 smoothing=SG 창 {_w}",
    )
PS_LADDER = tuple(k for k in CONDITIONS if k.startswith("ps_"))


@dataclasses.dataclass(frozen=True)
class PreprocessingConfigExt(PreprocessingConfig):
    """`baseline_before_smooth`를 config.py를 건드리지 않고 추가한다.

    preprocess_spectra는 getattr로 읽으므로 기본 PreprocessingConfig와 호환된다.
    """

    baseline_before_smooth: bool = False


def _prep_config(cfg, overrides: dict) -> PreprocessingConfigExt:
    base = dataclasses.asdict(cfg.preprocessing)
    return PreprocessingConfigExt(**{**base, **overrides})


#: 짝지은 Δ를 낼 기준. production = 현재 배포 파이프라인, cal_despike = 요인 교체의 기준점
DELTA_REFERENCES = ("production", "cal_despike", "no_cal", "cal_ps_si", "ps_despike")
# 2026-09-09 사용자 결정 이후 기준 조건은 cal_ps_si(표준물질만). "production"은 urea 보정 시절 기록 유지용.

#: 3-class 라벨 (aecd cohort_group → 클래스 인덱스). 매핑 run의 GROUP_ORDER와 같은 순서.
THREE_CLASS = {"control": 0, "prostate disease control": 1, "prostate": 2,
               "Control": 0, "Prostate disease control": 1, "Prostate cancer": 2}
N_JOBS = 5


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
    ps_shift_by_key: dict | None = None    # aecd만: key → PS global_shift_cm1 (run 단위)
    ps_si_linear_by_key: dict | None = None  # aecd만: key → (a, b): x_corr = x − (a + b·x)


def load_aecd(groups: tuple[str, ...] = AECD_COHORTS, name: str = "aecd") -> Cohort:
    """aecd_platform 전립선 코호트. ``groups``로 비암군을 좁힐 수 있다 (aecd_ctrl = 암 vs 정상만)."""
    from sers.aecd_api.loader import load_spectra_from_aecd
    from sers.aecd_api.models import SpectrumFilters
    from sers.aecd_api.repository import DatabaseSettings, PostgresAecdRepository

    repo = PostgresAecdRepository(DatabaseSettings.from_environment())
    spectra, ids, observed, key_to_mid = {}, [], {}, {}
    for cohort in groups:
        res = load_spectra_from_aecd(repo, SpectrumFilters(cohort_group=cohort), page_size=500)
        # loader는 spectra dict와 measurement_ids를 같은 루프에서 같은 순서로 채운다
        key_to_mid.update(dict(zip(res.spectra.keys(), res.measurement_ids)))
        spectra.update(res.spectra)
        ids.extend(res.measurement_ids)
        observed[cohort] = len(res.spectra)
        print(f"  {cohort:26s} {len(res.spectra):6d} spectra")
    ps_shift_by_key, ps_si_linear_by_key = _load_ps_shifts(key_to_mid)
    return Cohort(
        name=name, site=f"aecd_platform(전립선 {len(groups)}군)", spectra=spectra,
        cancer_group=AECD_CANCER,
        control_groups=tuple(c for c in groups if c != AECD_CANCER),
        measurement_ids=ids,
        data_query_filters={"cohort_group": list(groups), "observed_at_load": observed},
        ps_shift_by_key=ps_shift_by_key,
        ps_si_linear_by_key=ps_si_linear_by_key,
    )


def _load_ps_shifts(key_to_mid: dict) -> dict:
    """key → 표준물질(PS) run 단위 global_shift_cm1.

    AECD API 파이프라인(`load_standard_material_calibrations`/`calibration_for_item`)과
    같은 정의: 측정일(date)+장비명으로 `measurement.calibrations`(PS, raman_shift, pass)를
    조인하고, 축 보정은 x_corrected = x_observed − global_shift_cm1. 대응 보정이 없는
    측정은 오류로 멈춘다 (조용히 0 처리하지 않음).
    """
    import os

    import psycopg2

    conn = psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"), port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ.get("PGDATABASE", "aecd_platform"),
        user=os.environ.get("PGUSER", "postgres"), password=os.environ.get("PGPASSWORD", ""),
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT c.calibration_date::date, i.instrument_name, c.standard_material, "
                "c.global_shift_cm1, c.reference_peaks_cm1, c.observed_peaks_cm1 "
                "FROM measurement.calibrations c "
                "JOIN measurement.instruments i ON i.instrument_id = c.instrument_id "
                "WHERE c.standard_material IN ('Polystyrene (PS)', 'Silicon (Si)') "
                "AND c.calibration_type = 'raman_shift' AND c.result = 'pass'")
            cal, peaks = {}, {}
            for d, inst, material, shift, ref, obs in cur.fetchall():
                k = (d.isoformat(), inst)
                ref_a = np.asarray(ref, dtype=float)
                err_a = np.asarray(obs, dtype=float) - ref_a
                if material.startswith("Polystyrene"):
                    if k in cal:
                        raise RuntimeError(f"PS calibration ambiguous for {k}")
                    if shift is None or not np.allclose(err_a, float(shift), atol=0.05):
                        raise RuntimeError(f"stored global_shift disagrees with peak errors for {k}")
                    cal[k] = float(shift)
                peaks.setdefault(k, []).append((ref_a, err_a, material))
            # PS+Si 1차 축 보정: 관측오차 err(x) = a + b·x 를 두 물질의 피크에 최소제곱 적합.
            # PS 관측피크는 DB에 global_shift로 파생 저장돼 있어 8점이 같은 오차를 가지며,
            # Si 520.7은 독립 관측이라 기울기는 사실상 Si↔PS 차이가 결정한다.
            linear = {}
            for k, items in peaks.items():
                mats = {m for _, _, m in items}
                if not {"Polystyrene (PS)", "Silicon (Si)"} <= mats:
                    raise RuntimeError(f"PS+Si calibration incomplete for {k}: {mats}")
                xs = np.concatenate([r for r, _, _ in items])
                es = np.concatenate([e for _, e, _ in items])
                linear[k] = fit_standard_material_axis(xs, xs + es)
            mids = [int(m) for m in key_to_mid.values()]
            cur.execute(
                "SELECT m.measurement_id, r.measurement_date, i.instrument_name "
                "FROM measurement.measurements m "
                "JOIN measurement.runs r ON r.measurement_run_id = m.measurement_run_id "
                "JOIN measurement.instruments i ON i.instrument_id = r.instrument_id "
                "WHERE m.measurement_id = ANY(%s)", (mids,))
            mid_to_run = {int(mid): (d.isoformat(), inst) for mid, d, inst in cur.fetchall()}
    finally:
        conn.close()
    out, out_lin = {}, {}
    for key, mid in key_to_mid.items():
        run = mid_to_run.get(int(mid))
        if run is None or run not in cal or run not in linear:
            raise RuntimeError(f"no PS/Si calibration for measurement {mid} (run={run})")
        out[key] = cal[run]
        out_lin[key] = linear[run]
    uniq = sorted(set(out.values()))
    print(f"  PS calibration: {len(cal)} run(s), shift(cm⁻¹) = {uniq}")
    for k in sorted(linear):
        a, b = linear[k]
        print(f"  PS+Si linear {k[0]}: err(x) = {a:+.4f} {b:+.2e}·x  "
              f"(corr @520={a + b * 520:+.3f}, @1000={a + b * 1000:+.3f}, @2000={a + b * 2000:+.3f})")
    return out, out_lin


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
    if spec.get("ps_calibrate") == "linear":
        if cohort.ps_si_linear_by_key is None:
            raise RuntimeError("PS+Si 보정은 aecd 코호트(measurement.calibrations)에서만 가능")
        # x_corrected = x_observed − (a + b·x_observed), run 단위 1차 보정
        raw = {}
        for k, (x, y) in cohort.spectra.items():
            a, b = cohort.ps_si_linear_by_key[k]
            raw[k] = (apply_standard_material_axis(x, a, b), y)
    elif spec.get("ps_calibrate"):
        if cohort.ps_shift_by_key is None:
            raise RuntimeError("PS 보정은 aecd 코호트(measurement.calibrations)에서만 가능")
        # x_corrected = x_observed − global_shift (AECD API 파이프라인과 동일 정의)
        raw = {k: (np.asarray(x, dtype=float) - cohort.ps_shift_by_key[k], y)
               for k, (x, y) in raw.items()}
    if spec["calibrate"]:
        raw, shift_df = calibrate_spectra_batch(
            raw, target_wn=cfg.preprocessing.calibration_reference_wn,
            window=cfg.preprocessing.calibration_window,
        )
    passed, _ = apply_stage1_qc(raw)
    prep_cfg = _prep_config(cfg, spec["overrides"])
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
    y3 = np.array([THREE_CLASS[k[0]] for k in keys])
    groups = np.array([f"{k[0]}|{k[1]}" for k in keys])
    qc = {"input": len(raw), "stage1_passed": len(passed), "final": len(keys),
          "dropped": len(raw) - len(keys), "n_patients": int(len(np.unique(groups)))}
    return X, y, y3, groups, qc, shift_df, prep_cfg


def evaluate(X, y, groups, seed: int) -> np.ndarray:
    """spectrum OOF 확률 (seed 고정). y가 3클래스면 (n, 3) multinomial 확률."""
    n_class = int(len(np.unique(y)))
    oof = np.zeros((len(y), n_class), dtype=float)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y, groups):
        model = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=5000, class_weight="balanced",
                                      solver="lbfgs", random_state=seed)),
        ])
        model.fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])
    return oof[:, 1] if n_class == 2 else oof


def patient_table(oof: np.ndarray, y: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """환자 mean 집계. 이진이면 prob, 3클래스면 prob_class_0..2."""
    if oof.ndim == 1:
        frame = pd.DataFrame({"subject": groups, "true_label": y, "prob": oof})
        cols = {"prob": ("prob", "mean")}
    else:
        frame = pd.DataFrame({"subject": groups, "true_label": y})
        cols = {}
        for k in range(oof.shape[1]):
            frame[f"prob_class_{k}"] = oof[:, k]
            cols[f"prob_class_{k}"] = (f"prob_class_{k}", "mean")
    out = frame.groupby("subject").agg(true_label=("true_label", "first"), **cols,
                                       n_spectra=("true_label", "size"))
    return out.reset_index()


def _seed_job(X, y, groups, seed):
    oof = evaluate(X, y, groups, seed)
    pt = patient_table(oof, y, groups)
    pt.insert(0, "seed", seed)
    if oof.ndim == 1:
        auc = float(roc_auc_score(pt["true_label"], pt["prob"]))
        s_auc = float(roc_auc_score(y, oof))
    else:
        cols = [c for c in pt.columns if c.startswith("prob_class_")]
        auc = macro_roc_auc(pt["true_label"].to_numpy(), pt[cols].to_numpy())
        s_auc = macro_roc_auc(y, oof)
    return seed, pt, auc, s_auc, oof


TASK_FILES = {"screening": "patient_oof_predictions.csv",
              "three_class": "patient_oof_three_class.csv"}


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def run_name(cohort: str, cond: str) -> str:
    return f"guide_{cohort}_{cond}_{DATE_TAG}"


def compute(cohort: Cohort, conditions: list[str], seeds: tuple[int, ...], cfg, dry_run: bool,
            tasks: tuple[str, ...] = ("screening", "three_class"), force: bool = False):
    for cond in conditions:
        spec = CONDITIONS[cond]
        out_dir = OUT_ROOT / cohort.name / cond
        todo = [t for t in tasks if force or dry_run or not (out_dir / TASK_FILES[t]).exists()]
        if not todo:
            print(f"\n[{cohort.name}/{cond}] 이미 있음 — 건너뜀 (--force로 재계산)")
            continue
        t0 = time.time()
        started_at = datetime.now(timezone.utc).isoformat()
        print(f"\n[{cohort.name}/{cond}] {spec['label']}  tasks={todo}")
        X, y, y3, groups, qc, shift_df, prep_cfg = run_condition(cohort, cfg, spec)
        print(f"  QC: {qc}   전처리 {time.time() - t0:.0f}s  features={X.shape[1]}")

        task_meta = {}
        for task in todo:
            target = y if task == "screening" else y3
            t1 = time.time()
            results = Parallel(n_jobs=min(N_JOBS, len(seeds)))(
                delayed(_seed_job)(X, target, groups, seed) for seed in seeds)
            tables, seed_auc, primary = [], {}, None
            for seed, pt, auc, s_auc, oof in results:
                tables.append(pt)
                seed_auc[seed] = auc
                if seed == PRIMARY_SEED:
                    primary = pd.DataFrame({"key": [f"{k}" for k in groups], "true_label": target})
                    if oof.ndim == 1:
                        primary["prob"] = oof
                    else:
                        for k in range(oof.shape[1]):
                            primary[f"prob_class_{k}"] = oof[:, k]
                print(f"  [{task}] seed {seed:>5}: patient AUC={auc:.4f}  spectrum AUC={s_auc:.4f}")
            print(f"  [{task}] {len(seeds)}시드 {time.time() - t1:.0f}s")
            task_meta[task] = {"tables": tables, "seed_auc": seed_auc, "primary": primary}
        if dry_run:
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / "run_metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        for task, tm in task_meta.items():
            pd.concat(tm["tables"]).to_csv(out_dir / TASK_FILES[task], index=False, encoding="utf-8-sig")
            suffix = "" if task == "screening" else "_three_class"
            tm["primary"].to_csv(out_dir / f"spectrum_oof_seed{PRIMARY_SEED}{suffix}.csv",
                                 index=False, encoding="utf-8-sig")
            vals = list(tm["seed_auc"].values())
            meta[f"{task}_seed_patient_auc"] = tm["seed_auc"]
            meta[f"{task}_patient_auc_mean5"] = float(np.mean(vals))
            meta[f"{task}_patient_auc_sd5"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        if "screening" in task_meta:      # 호환: 기존 키 유지
            meta["seed_patient_auc"] = meta["screening_seed_patient_auc"]
            meta["patient_auc_mean5"] = meta["screening_patient_auc_mean5"]
            meta["patient_auc_sd5"] = meta["screening_patient_auc_sd5"]
        if shift_df is not None:
            shift_df.to_csv(out_dir / "calibration_shifts.csv", index=False, encoding="utf-8-sig")
        meta.update({
            "run_name": run_name(cohort.name, cond),
            "cohort": cohort.name, "site": cohort.site, "condition": cond,
            "label": spec["label"], "method_key": spec["method_key"],
            "calibrate": spec["calibrate"],
            "ps_calibrate": spec.get("ps_calibrate", False),
            "ps_shift_cm1": (None if not spec.get("ps_calibrate") else
                             sorted({float(v) for v in cohort.ps_shift_by_key.values()})),
            "ps_si_linear": (None if spec.get("ps_calibrate") != "linear" else
                             sorted({(round(a, 5), round(b, 8))
                                     for a, b in cohort.ps_si_linear_by_key.values()})),
            "preprocessing": {k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in dataclasses.asdict(prep_cfg).items()
                              if isinstance(v, (int, float, str, bool, tuple, list)) or v is None},
            "grid": "402-2198/935 (trim 후 부분집합)", "n_features": int(X.shape[1]),
            "model": "LogisticRegression(C=1.0, balanced, lbfgs) + StandardScaler; "
                     "3-class는 multinomial",
            "cv": "StratifiedGroupKFold(5) by subject, spectrum-level train, patient mean OOF",
            "seeds": list(seeds), "primary_seed": PRIMARY_SEED,
            "qc": qc,
            "calibration_shift_cm1": (None if shift_df is None else {
                "mean": float(shift_df.shift_cm1.mean()), "sd": float(shift_df.shift_cm1.std()),
                "abs_max": float(shift_df.shift_cm1.abs().max()),
                "n_zero": int((shift_df.shift_cm1 == 0).sum())}),
            "data_query_filters": cohort.data_query_filters,
            "measurement_ids_n": None if cohort.measurement_ids is None else len(cohort.measurement_ids),
            "started_at": meta.get("started_at", started_at),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        if cohort.measurement_ids is not None:
            np.save(out_dir / "measurement_ids.npy", np.asarray(cohort.measurement_ids, dtype=np.int64))
        print(f"  → {out_dir}")


# ---------------------------------------------------------------------------
# CI 표 + DB 적재 (저장된 예측에서만 계산 — 재학습 없음)
# ---------------------------------------------------------------------------

def _cohort_spec(meta: dict, pt: pd.DataFrame, cohort_key: str, task: Task) -> CohortSpec:
    if cohort_key == "aecd_ctrl":
        cancer, control = (AECD_CANCER,), ("control",)
        cohort_id, site = f"aecd_prostate_vs_control_{DATE_TAG}", "aecd_platform"
        note = ("aecd_platform 전립선암 vs 정상(control)만 — 질환대조군 제외. 3군 코호트의 부분집합이며 "
                "같은 환자·같은 전처리에서 두 군만으로 학습·OOF. 양성=전립선암.")
    elif cohort_key == "aecd":
        cancer, control = (AECD_CANCER,), tuple(c for c in AECD_COHORTS if c != AECD_CANCER)
        cohort_id, site = f"aecd_prostate3_{DATE_TAG}", "aecd_platform"
        note = ("aecd_platform 전립선 3군(prostate / prostate disease control / control) = 보라매 "
                "113검체×121점 매핑 데이터의 DB 사본(2026-09-02 Drop 1명 제외). 양성=전립선암.")
    else:
        cancer, control = (MAPPING_CANCER,), ("Control", "Prostate disease control")
        cohort_id, site = "boramae_mapping_2026", "보라매"
        note = "보라매 매핑 코호트(121점/검체). 양성=전립선암. 기존 매핑 run과 달리 Stage1/corr QC 적용."
    if task is Task.CANCER_SCREENING:
        n_pos, n_neg = int((pt.true_label == 1).sum()), int((pt.true_label == 0).sum())
    else:   # 3-class: 매핑 run 규약과 같이 전 군을 cancer_groups에, n_negative=0
        cancer, control = cancer + control, ()
        n_pos, n_neg = len(pt), 0
        counts = pt.true_label.value_counts().sort_index().to_dict()
        note += f" 3-class counts(0=control,1=질환대조,2=암)={counts}."
    return CohortSpec(
        cohort_id=cohort_id, site=site, cancer_groups=cancer, control_groups=control,
        n_patients=len(pt), n_positive=n_pos, n_negative=n_neg, n_spectra=int(pt.n_spectra.sum()),
        notes=note + f" 조건={meta['condition']}: {meta['label']}. LR C=1.0 고정(PL-1 규약).",
    )


MODEL_NAME = "LR C=1.0 (spectrum→patient mean)"


def _task_rows(pt: pd.DataFrame, task: Task, rname: str, cohort: CohortSpec) -> list[MetricRow]:
    y = pt.true_label.to_numpy()
    if task is Task.CANCER_SCREENING:
        score, metrics = pt.prob.to_numpy(), BINARY_METRICS
    else:
        cols = sorted(c for c in pt.columns if c.startswith("prob_class_"))
        score, metrics = pt[cols].to_numpy(dtype=float), MULTICLASS_METRICS
    rows = []
    for name, fn in metrics.items():
        ci = bootstrap_metric_ci(y, score, pt.subject.to_numpy(), fn, name=name, n_boot=N_BOOT)
        rows.append(MetricRow.from_ci(ci, run_id=rname, model_name=MODEL_NAME, task=task,
                                      evaluation=Evaluation.OOF, aggregation="patient",
                                      cohort=cohort))
    return rows


def build_ci(cohort_key: str, conditions: list[str], load_db: bool) -> pd.DataFrame:
    tracker = None
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip() or None
    if load_db:
        from sers.preprocessing_lab.db import ExperimentTracker
        tracker = ExperimentTracker()

    summary, primary, primary3 = [], {}, {}
    for cond in conditions:
        out_dir = OUT_ROOT / cohort_key / cond
        if not (out_dir / "run_metadata.json").exists():
            print(f"  (skip) {out_dir} 결과 없음")
            continue
        meta = json.loads((out_dir / "run_metadata.json").read_text())
        rname = meta["run_name"]
        allp = pd.read_csv(out_dir / TASK_FILES["screening"], encoding="utf-8-sig")
        pt = allp[allp.seed == PRIMARY_SEED].sort_values("subject").reset_index(drop=True)
        cohort = _cohort_spec(meta, pt, cohort_key, Task.CANCER_SCREENING)
        rows = _task_rows(pt, Task.CANCER_SCREENING, rname, cohort)
        primary[cond] = pt
        auc = rows[0]
        rec = {
            "cohort": cohort_key, "condition": cond, "label": meta["label"], "run_name": rname,
            "n_patients": cohort.n_patients, "n_positive": cohort.n_positive,
            "n_spectra": cohort.n_spectra, "n_features": meta.get("n_features"),
            "auc_seed42": auc.value, "auc_ci_low": auc.ci_low, "auc_ci_high": auc.ci_high,
            "auc_mean5": meta["patient_auc_mean5"], "auc_sd5": meta["patient_auc_sd5"],
            "calibration_abs_max_shift": (meta["calibration_shift_cm1"] or {}).get("abs_max"),
        }
        line = (f"  {cohort_key}/{cond:<20} AUC={auc.value:.3f} [{auc.ci_low:.3f},{auc.ci_high:.3f}]"
                f" 5시드 {meta['patient_auc_mean5']:.3f}±{(meta['patient_auc_sd5'] or 0):.3f}")

        three = out_dir / TASK_FILES["three_class"]
        if three.exists():
            all3 = pd.read_csv(three, encoding="utf-8-sig")
            pt3 = all3[all3.seed == PRIMARY_SEED].sort_values("subject").reset_index(drop=True)
            cohort3 = _cohort_spec(meta, pt3, cohort_key, Task.PROSTATE_THREE_CLASS)
            rows3 = _task_rows(pt3, Task.PROSTATE_THREE_CLASS, rname, cohort3)
            rows += rows3
            primary3[cond] = pt3
            m = {r.metric: r for r in rows3}
            rec.update({
                "macro_auc_seed42": m["macro_auc"].value, "macro_auc_ci_low": m["macro_auc"].ci_low,
                "macro_auc_ci_high": m["macro_auc"].ci_high,
                "macro_auc_mean5": meta.get("three_class_patient_auc_mean5"),
                "macro_auc_sd5": meta.get("three_class_patient_auc_sd5"),
                "macro_f1_seed42": m["macro_f1"].value, "accuracy_seed42": m["accuracy"].value,
            })
            line += (f" | 3-class macroAUC={m['macro_auc'].value:.3f} "
                     f"[{m['macro_auc'].ci_low:.3f},{m['macro_auc'].ci_high:.3f}]")
        write_metrics_csv(rows, out_dir / "metrics_ci_standard.csv")
        summary.append(rec)
        print(line)

        if tracker is not None:
            _load_run(tracker, meta, rows, out_dir, commit)

    # 조건 간 짝지은 차이 — 같은 환자를 함께 재추출. 기준 2개(production, cal_despike)
    for ref in DELTA_REFERENCES:
        for tag, store, fn, scol in (("", primary, roc_auc, None),
                                     ("3_", primary3, macro_roc_auc, "prob_class_")):
            if ref not in store:
                continue
            base = store[ref]
            for rec in summary:
                cond = rec["condition"]
                key = f"d{tag}_{ref}"
                if cond == ref:
                    rec.update({key: 0.0, f"{key}_lo": 0.0, f"{key}_hi": 0.0})
                    continue
                if cond not in store:
                    continue
                merged = base.merge(store[cond], on="subject", suffixes=("_b", "_c"))
                if scol is None:
                    a, b = merged.prob_c.to_numpy(), merged.prob_b.to_numpy()
                else:
                    cb = sorted(c for c in merged.columns if c.startswith(scol) and c.endswith("_b"))
                    cc = sorted(c for c in merged.columns if c.startswith(scol) and c.endswith("_c"))
                    a, b = merged[cc].to_numpy(dtype=float), merged[cb].to_numpy(dtype=float)
                d = bootstrap_difference_ci(
                    merged.true_label_b.to_numpy(), a, b, merged.subject.to_numpy(), fn,
                    name=f"{key}", n_boot=N_BOOT)
                rec.update({key: d.value, f"{key}_lo": d.ci_low, f"{key}_hi": d.ci_high,
                            f"{key}_n": len(merged)})
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
    seed_rows = []
    for task, prefix in (("screening", "screening_auc_patient"),
                         ("three_class", "three_class_macro_auc_patient")):
        seeds = meta.get(f"{task}_seed_patient_auc") or (meta.get("seed_patient_auc")
                                                         if task == "screening" else None)
        if not seeds:
            continue
        seed_rows += [{"metric_name": f"{prefix}_seed{s}", "metric_value": v,
                       "split": "oof", "model_name": "logistic_regression"} for s, v in seeds.items()]
        vals = list(seeds.values())
        seed_rows += [
            {"metric_name": f"{prefix}_mean5seeds", "metric_value": float(np.mean(vals)),
             "split": "oof", "model_name": "logistic_regression"},
            {"metric_name": f"{prefix}_sd5seeds",
             "metric_value": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
             "split": "oof", "model_name": "logistic_regression"},
        ]
    tracker.record_metrics(run_id, [r for r in seed_rows if r["metric_value"] is not None])
    print(f"    → experiment.runs #{run_id} ({rname})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", choices=["aecd", "aecd_ctrl", "mapping", "both"], default="aecd",
                    help="aecd=전립선 3군, aecd_ctrl=암 vs 정상만(질환대조 제외), mapping=파일 로더 확인용")
    ap.add_argument("--conditions", default=",".join(CONDITIONS),
                    help="쉼표 구분. 기본: 전부")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--dry-run", action="store_true", help="계산만 하고 아무것도 저장하지 않음")
    ap.add_argument("--load-db", action="store_true",
                    help="저장된 예측에서 CI 표를 만들고 aecd_platform experiment 스키마에 적재")
    ap.add_argument("--ci-only", action="store_true",
                    help="학습 없이 저장된 예측에서 CI 표·summary만 다시 만듦 (DB 미적재)")
    ap.add_argument("--tasks", default="screening,three_class")
    ap.add_argument("--force", action="store_true", help="이미 있는 task 결과도 재계산")
    args = ap.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    if conditions == ["PS_LADDER"]:
        conditions = list(PS_LADDER)
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
            if key == "aecd":
                cohort = load_aecd()
            elif key == "aecd_ctrl":
                cohort = load_aecd(groups=(AECD_CANCER, "control"), name="aecd_ctrl")
            else:
                cohort = load_mapping()
            compute(cohort, conditions, seeds, cfg, args.dry_run,
                    tasks=tuple(t.strip() for t in args.tasks.split(",")), force=args.force)
        if args.dry_run:
            print("\n(dry-run) 저장·적재 없음")
            return

    print("\n=== CI 표 / summary ===")
    frames = [build_ci(key, conditions, load_db=args.load_db and key.startswith("aecd")) for key in cohorts]
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
