#!/usr/bin/env python
"""표준물질(PS/Si) 기준 전처리 방법 비교 — ground truth 있는 평가.

임상 코호트로는 전처리 효과를 판별할 수 없다는 것이 PL-1에서 확인됐다
(환자 112명, 환자 1명 차이로 AUC 차이 부호가 뒤집힘). 표준물질은 참조 피크
위치가 알려져 있고 같은 시료를 반복 측정하므로, "노이즈를 얼마나 줄였나"와
"피크를 얼마나 망가뜨렸나"를 직접 잴 수 있다.

참조 피크는 aecd_platform `measurement.calibrations`에서 가져온다(단일 출처).
DB에 접속할 수 없으면 실행을 중단한다 — 값을 코드에 박아두면 DB와 어긋난다.

Usage:
    source scripts/db/pghost.sh
    python scripts/analysis/preprocessing_lab/run_reference_eval.py
    python scripts/analysis/preprocessing_lab/run_reference_eval.py --material Si
"""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np

from sers.aecd_api.repository import DatabaseSettings, PostgresAecdRepository
from sers.config import load_config
from sers.preprocessing import preprocess_single_spectrum
from sers.preprocessing_lab.reference_eval import evaluate_reference, load_reference_sessions

DB_MATERIAL = {"PS": "Polystyrene (PS)", "Si": "Silicon (Si)"}

# 비교할 전처리 조건. 이름 → PreprocessingConfig 오버라이드.
CONDITIONS: dict[str, dict[str, object]] = {
    "raw": {"do_smooth": False, "do_baseline": False, "normalization": "none",
            "do_despike": False},
    "production": {},                                   # config.yaml 기본값 그대로
    "production+despike": {"do_despike": True},
    "despike_only": {"do_smooth": False, "do_baseline": False, "normalization": "none",
                     "do_despike": True},
    "no_smooth": {"do_smooth": False},                  # Butler 2016의 "smoothing 신중히" 권고 검증
}


def reference_peaks(material: str) -> list[float]:
    repo = PostgresAecdRepository(DatabaseSettings.from_environment())
    page = repo.reference_peaks(DB_MATERIAL[material])
    if not page.items:
        raise SystemExit(f"{DB_MATERIAL[material]}의 참조 피크가 DB에 없습니다.")
    return [float(v) for v in page.items[0].reference_peaks_cm1]


def apply_condition(x: np.ndarray, spectra: np.ndarray, cfg, overrides: dict) -> np.ndarray:
    """replicate별로 전처리를 적용한다 (노이즈 추정이 replicate 분산에 의존)."""
    prep = dataclasses.replace(cfg.preprocessing, **overrides)
    grid = np.linspace(max(x.min(), 400.0), min(x.max(), 2200.0), 935)
    out = []
    for row in spectra:
        out.append(preprocess_single_spectrum(
            x, row, grid,
            do_despike=prep.do_despike, despike_method=prep.despike_method,
            despike_z_threshold=prep.despike_z_threshold, despike_window=prep.despike_window,
            do_trim=prep.do_trim, trim_region=prep.trim_region,
            do_smooth=prep.do_smooth, smoothing_method=prep.smoothing_method,
            smooth_window=prep.smooth_window, smooth_poly=prep.smooth_poly,
            do_baseline=prep.do_baseline, baseline_window=prep.baseline_window,
            baseline_method=prep.baseline_method,
            normalization=prep.normalization or ("snv" if prep.use_snv else "none"),
        ))
    return grid, np.vstack(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--material", default="PS", choices=["PS", "Si"])
    ap.add_argument("--halfwin", type=float, default=12.0)
    args = ap.parse_args()

    peaks = reference_peaks(args.material)
    sessions = load_reference_sessions(material=args.material)
    cfg = load_config()
    print(f"{args.material}: 참조 피크 {len(peaks)}개 {peaks}")
    print(f"세션 {len(sessions)}개 × replicate {next(iter(sessions.values()))[1].shape[0]}\n")

    rows = []
    for name, overrides in CONDITIONS.items():
        shifts, snrs, fwhms = [], [], []
        for key, (x, spectra) in sorted(sessions.items()):
            grid, processed = apply_condition(x, spectra, cfg, overrides)
            ev = evaluate_reference(grid, processed, peaks, material=args.material,
                                    session=key, halfwin=args.halfwin)
            shifts.append(ev.median_abs_shift_cm1)
            snrs.append(ev.median_snr)
            fwhms.append(ev.median_fwhm_cm1)
        rows.append((name, np.nanmedian(shifts), np.nanmedian(snrs), np.nanmedian(fwhms)))

    print(f"{'조건':<20} {'|위치이동| cm-1':>16} {'피크 SNR':>10} {'FWHM cm-1':>11}")
    print("-" * 62)
    for name, sh, snr, fw in rows:
        print(f"{name:<20} {sh:16.2f} {snr:10.1f} {fw:11.2f}")
    print("\n해석: 위치이동은 작을수록, SNR은 클수록, FWHM은 원본 대비 커지면"
          " 피크가 뭉개진 것. 채택 판단은 사용자 몫.")


if __name__ == "__main__":
    main()
