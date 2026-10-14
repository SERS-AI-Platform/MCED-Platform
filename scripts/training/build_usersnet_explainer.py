#!/usr/bin/env python3
"""uSERS-Net 피크 기여도(설명) 아티팩트 생성 — artifacts/usersnet/<버전>/explainer/

기여도 계산(피크 가림, docs/ml/explainability_design_peak_attribution.md §5)에 쓰는 **참조 스펙트럼**과
**방법 판**을 모델 아티팩트에 넣는다. 앱은 이 폴더가 있을 때만 기여도를 계산하고, ``validated`` 가 true 일 때만
표시한다. 참조는 운영 모델 학습 데이터와 같은 전처리(3채널: raw·1차·2차 미분, SNV, 공통 격자)로 만든
비암 코호트(NOR·DIA·HBP·H.D.) 평균 스펙트럼이다. 검증용으로 암 코호트 평균과 암종별 평균도 함께 저장한다.

실행 (SERS-AI 루트, 원시 데이터 필요):
    python scripts/training/build_usersnet_explainer.py                 # v1.0.0
    python scripts/training/build_usersnet_explainer.py --artifact-name v1.0.0 --validated   # 검증 승인 후에만

산출물:
    explainer/reference_noncancer_mean.npy  (3, 935) float32 — 기여도 계산의 대조 스펙트럼
    explainer/reference_cancer_mean.npy     (3, 935)          — 검증(방향 대조)용
    explainer/reference_type_means.npz      암종별 평균       — 검증용
    explainer/peak_ranges.json              피크별 Voigt 면적 분포(비암·암 코호트 5/50/95 백분위) — 환자 피크가
                                            비암 범위 안/밖인지 설명하는 데 쓴다 (2026-09-29)
    explainer/explainer.json                method·version·background·validated·n_samples·source_commit
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models import build_production_stacking as bps  # noqa: E402  (운영 모델과 같은 로더·전처리)
from scripts.training.train_usersnet import KNOWN_PEAKS, extract_peak_features  # noqa: E402  (운영 피크 특징과 같은 적합)

METHOD_VERSION = "0.1"
METHOD = "peak-occlusion"
METHOD_TEXT = (
    "환자 대표 스펙트럼에서 알려진 피크 창(중심 ± 반폭×1.5)을 비암 학습 코호트 평균으로 바꿔 넣고 배포 모델 전체를 다시 실행해 l"
    "ogit P(암)의 변화(로그오즈 차이)를 기여도로 삼는다. 양수: 관측 피크가 판정을 암 쪽으로 밈. 확률 차이(delta_proba"
    "bility)와 비암 평균 대비 피크 면적 차이(delta_pct)를 함께 기록한다."
)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-name", default="v1.0.0")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--validated", action="store_true",
                        help="검증·규제 검토가 끝난 뒤에만 켠다. 켜지 않으면 앱은 기여도를 계산만 하고 표시하지 않는다.")
    args = parser.parse_args()

    artifact_dir = PROJECT_ROOT / "artifacts" / "usersnet" / args.artifact_name
    grid = np.load(artifact_dir / "common_grid.npy")
    prep = json.loads((artifact_dir / "preprocessing.json").read_text(encoding="utf-8"))
    data_dir = args.data_dir or PROJECT_ROOT / "data" / "02_sers_primary_pooled_acquisition" / "thermo_retro_12groups_undated"

    bps.DATA_TYPE = "raw_spectrum"
    X_raw, df_raw = bps.load_data(grid, data_dir)          # (n_spectra, 3, 935)
    X_agg, df_agg = bps.aggregate_mean(X_raw, df_raw)      # 반복측정 평균 = 운영 모델 학습 단위
    groups = df_agg["group"].values
    non_cancer = np.isin(groups, bps.NON_CANCER)
    cancer = np.isin(groups, bps.CANCER_TYPES)
    if non_cancer.sum() == 0 or cancer.sum() == 0:
        raise SystemExit(f"코호트가 비어 있음: non-cancer {non_cancer.sum()}, cancer {cancer.sum()}")

    out = artifact_dir / "explainer"
    out.mkdir(exist_ok=True)
    np.save(out / "reference_noncancer_mean.npy", X_agg[non_cancer].mean(axis=0).astype(np.float32))
    np.save(out / "reference_cancer_mean.npy", X_agg[cancer].mean(axis=0).astype(np.float32))
    np.savez(out / "reference_type_means.npz",
             **{ct: X_agg[groups == ct].mean(axis=0).astype(np.float32) for ct in bps.CANCER_TYPES if (groups == ct).any()})

    # 피크별 면적 분포: 운영 모델의 피크 특징(Voigt 면적, 앞 17열)과 같은 계산
    areas = extract_peak_features(X_agg[:, 0, :], grid)[:, : len(KNOWN_PEAKS)]
    def pct(values):
        return {q: round(float(v), 6) for q, v in zip(("p5", "p50", "p95"), np.percentile(values, [5, 50, 95]))}
    ranges = {name: {"wavenumber": center, "noncancer": pct(areas[non_cancer, i]), "cancer": pct(areas[cancer, i])}
              for i, (center, name, _hw) in enumerate(KNOWN_PEAKS)}
    (out / "peak_ranges.json").write_text(json.dumps(ranges, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    counts = {str(g): int((groups == g).sum()) for g in sorted(set(groups))}
    meta = {
        "method": METHOD,
        "version": METHOD_VERSION,
        "method_text": METHOD_TEXT,
        "background": f"비암 학습 코호트 평균 스펙트럼 (NOR·DIA·HBP·H.D., n={int(non_cancer.sum())}), 운영 모델 v1.0.0 학습 데이터",
        "reference_id": f"usersnet-{args.artifact_name}-noncancer-mean-{dt.date.today().isoformat()}",
        "reference_file": "reference_noncancer_mean.npy",
        "channels": prep.get("channels"),
        "grid_points": int(len(grid)),
        "window_factor": 1.5,
        "ranges_file": "peak_ranges.json",
        "n_samples": {"non_cancer": int(non_cancer.sum()), "cancer": int(cancer.sum()), "by_group": counts},
        "validated": bool(args.validated),
        "validation_note": "" if args.validated else "검증 전 — 앱은 값을 저장만 하고 화면·보고서에 표시하지 않는다.",
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source_commit": git_commit(),
        "design_doc": "docs/ml/explainability_design_peak_attribution.md",
    }
    (out / "explainer.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(PROJECT_ROOT)}: non-cancer n={non_cancer.sum()}, cancer n={cancer.sum()}, validated={args.validated}")


if __name__ == "__main__":
    main()
