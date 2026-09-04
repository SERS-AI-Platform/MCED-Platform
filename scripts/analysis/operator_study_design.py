#!/usr/bin/env python3
"""측정자 효과를 실제로 추정하려면 며칠이 필요한가 -- 설계 및 검정력 계산.

현재 데이터로는 측정자 효과를 추정할 수 없다(operator_variability_analysis.py 참고).
측정자가 날짜와 완전히 aliasing 되어 있기 때문이다. 분리하려면 **같은 날 두 측정자가
같은 대상을 각각 측정하는 교차 설계**가 필요하고, 그러면 날짜 효과가 쌍 안에서 상쇄된다.

핵심: 교차 설계에서는 날짜 간 변동(가장 큰 성분)이 분석에서 빠진다. 필요한 표본을
결정하는 것은 **같은 날 같은 대상의 반복 측정 변동**이지 날짜 변동이 아니다.

두 가지 설계를 계산한다.

  A. QC 표준물질(폴리스티렌) 교차 측정 -- 매일 두 측정자가 PS를 각 n회 측정.
     검체 소모가 없어 일상 업무와 병행 가능. 반복성은 실측값을 쓴다
     (data/mapping/Thermo Reference, 2026-08-04~14, 9일 60개 스펙트럼).

  B. 임상 검체 분주(split-aliquot) 교차 측정 -- 같은 소변을 둘로 나눠 두 측정자가
     각각 준비·측정. 시료 전처리까지 포함한 측정자 효과를 잡는다. 다만 이 설계의
     기술적 반복성(같은 소변, 다른 스트립)은 **현재 데이터로 추정할 수 없다** --
     한 번도 같은 검체를 두 번 측정한 적이 없다. 그래서 가정값을 훑는다.

실행:
    python scripts/analysis/operator_study_design.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sers.config import load_config  # noqa: E402
from sers.io import read_spectrum  # noqa: E402
from sers.signal import baseline_correction, resample, smooth  # noqa: E402

ALPHA = 0.05
POWER = 0.80
REFERENCE_DIR = ROOT / "data" / "mapping" / "Thermo Reference"
SAMPLE_METRICS = ROOT / "results" / "operator_variability" / "sample_metrics.csv"


def required_units(effect: float, sd_difference: float,
                   alpha: float = ALPHA, power: float = POWER) -> int:
    """쌍 비교에서 effect를 탐지하는 데 필요한 쌍의 수 (정규 근사)."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil(z**2 * sd_difference**2 / effect**2))


def ps_repeatability(cfg) -> pd.DataFrame:
    """PS 표준물질의 일내 반복성과 일간 변동을 실측 파일에서 계산한다."""
    low, high = cfg.trim_region
    grid = np.arange(low, high + 1.93, 1.93)
    rows = []
    for path in sorted(REFERENCE_DIR.glob("*_PS_*.CSV")):
        match = re.match(r"\d{8}_(\d{8})Cali_PS_(\d+)\.CSV", path.name)
        if match is None:  # _ave 파일은 평균이라 반복성 추정에 쓰면 안 된다
            continue
        x, y = read_spectrum(path)
        resampled = resample(x, y, grid)
        smoothed = smooth(resampled, window=cfg.smooth_window, poly=cfg.smooth_poly)
        corrected = baseline_correction(smoothed, window=cfg.baseline_window)
        rows.append({
            "day": match.group(1),
            "auc": float(np.trapezoid(resampled, grid)),
            "peak": float(np.max(corrected)),
            "noise_sd": float(np.std(resampled - smoothed)),
        })
    frame = pd.DataFrame(rows)
    out = []
    for metric in ("auc", "peak", "noise_sd"):
        mean = frame[metric].mean()
        out.append({
            "metric": metric,
            "n_days": frame.day.nunique(),
            "n_spectra": len(frame),
            "within_day_cv_pct": 100 * frame.groupby("day")[metric].std(ddof=1).mean() / abs(mean),
            "between_day_cv_pct": 100 * frame.groupby("day")[metric].mean().std(ddof=1) / abs(mean),
        })
    return pd.DataFrame(out)


def design_a(repeatability: pd.DataFrame, reps_per_day=(3, 5, 10),
             effects_pct=(15, 10, 5, 3)) -> pd.DataFrame:
    """PS 교차 측정: 탐지하려는 효과 크기별 필요 일수.

    하루가 쌍 1개다. 쌍 차이의 SD = sqrt(2/n) * 일내반복 CV.
    측정자x날짜 상호작용은 0으로 가정한다 -- 추정할 데이터가 없기 때문이며,
    상호작용이 있으면 필요 일수는 이보다 늘어난다.
    """
    rows = []
    for _, r in repeatability.iterrows():
        for n in reps_per_day:
            sd_diff = np.sqrt(2 / n) * r.within_day_cv_pct
            for effect in effects_pct:
                rows.append({
                    "metric": r.metric,
                    "일내반복 CV(%)": round(r.within_day_cv_pct, 2),
                    "1인당 반복수/일": n,
                    "탐지목표(평균 대비 %)": effect,
                    "필요 일수": required_units(effect, sd_diff),
                })
    return pd.DataFrame(rows)


def design_b(sample_level: pd.DataFrame, tech_cv_multipliers=(0.2, 0.35, 0.5, 1.0),
             pairs_per_day=(10, 20, 30)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """임상 검체 분주 교차 측정: 필요한 쌍 수와 일수.

    탐지 목표는 '현재 관측된 날짜 간 SD'로 잡는다 -- 그보다 작은 측정자 효과는
    이미 존재하는 날짜 변동에 묻히므로 실무적으로 조치할 대상이 아니다.
    기술적 반복성 SD는 미지수라 검체 간 SD의 배수로 훑는다.
    """
    bnor = sample_level[sample_level.label_prefix == "BNOR"]
    stats_rows, plan_rows = [], []
    for metric in ("auc", "median_intensity", "noise_sd"):
        groups = [g[metric].to_numpy() for _, g in bnor.groupby("run_id")]
        counts = np.array([len(g) for g in groups])
        means = np.array([g.mean() for g in groups])
        grand = np.concatenate(groups).mean()
        k, total = len(groups), counts.sum()
        ms_between = float((counts * (means - grand) ** 2).sum()) / (k - 1)
        ms_within = float(sum(((g - g.mean()) ** 2).sum() for g in groups)) / (total - k)
        n0 = (total - (counts**2).sum() / total) / (k - 1)
        sd_day = np.sqrt(max((ms_between - ms_within) / n0, 0.0))
        sd_sample = np.sqrt(ms_within)
        stats_rows.append({"metric": metric, "검체간 SD": sd_sample, "날짜간 SD": sd_day,
                           "탐지목표 Δ(=날짜간 SD)": sd_day})
        for mult in tech_cv_multipliers:
            sd_tech = mult * sd_sample
            n_pairs = required_units(sd_day, np.sqrt(2) * sd_tech)
            row = {"metric": metric, "가정 기술반복 SD": f"{mult:g}×검체간SD",
                   "필요 쌍 수": n_pairs}
            for m in pairs_per_day:
                row[f"{m}쌍/일 → 일수"] = int(np.ceil(n_pairs / m))
            plan_rows.append(row)
    return pd.DataFrame(stats_rows), pd.DataFrame(plan_rows)


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    cfg = load_config(ROOT / "config" / "config.yaml").preprocessing

    print("=== PS 표준물질 반복성 (실측, data/mapping/Thermo Reference) ===")
    rep = ps_repeatability(cfg)
    print(rep.round(2).to_string(index=False))
    print("\n일간 CV가 일내 CV보다 2~3배 크다 -- 고정된 물질인데도 날짜가 바뀌면 변한다.")
    print("교차 설계에서는 이 일간 변동이 쌍 안에서 상쇄되므로 표본 계산에 들어가지 않는다.")

    print("\n=== 설계 A: PS 표준물질 교차 측정 -- 필요 일수 (α=0.05, power=0.80) ===")
    plan_a = design_a(rep)
    print(
        plan_a[plan_a.metric == "auc"]
        .pivot(index="1인당 반복수/일", columns="탐지목표(평균 대비 %)", values="필요 일수")
        .to_string()
    )
    print("\n(위 표는 AUC 기준. peak/noise_sd 전체는 CSV 참조)")

    print("\n=== 설계 B: 임상 검체 분주 교차 측정 ===")
    sample_level = pd.read_csv(SAMPLE_METRICS)
    stats_b, plan_b = design_b(sample_level)
    print(stats_b.round(4).to_string(index=False))
    print()
    print(plan_b.to_string(index=False))

    out_dir = ROOT / "results" / "operator_variability"
    plan_a.to_csv(out_dir / "design_a_qc_standard.csv", index=False, encoding="utf-8-sig")
    plan_b.to_csv(out_dir / "design_b_split_aliquot.csv", index=False, encoding="utf-8-sig")
    print(f"\n출력: {out_dir}/design_a_qc_standard.csv, design_b_split_aliquot.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
