#!/usr/bin/env python
"""Smoothing 파라미터 스윕 — 표준물질(PS/Si) 기준 SNR 대 피크 폭 trade-off.

표준물질 평가에서 현 프로덕션 설정(SG window=11, poly=3)이 피크 FWHM을
9.76 → 12.44 cm-1로 약 27% 넓히는 것이 관측됐다. [17] Butler 2016도
"smoothing은 제한적·신중하게" 쓰라고 권고한다. 어느 설정이 SNR을 얻으면서
피크 폭을 지키는지를 데이터로 정한다.

smoothing만 격리해 적용한다(baseline/정규화/despike 없음). 그래야 FWHM 변화를
smoothing 탓으로 정확히 귀속할 수 있다.

Usage:
    python scripts/analysis/preprocessing_lab/run_smoothing_sweep.py
    python scripts/analysis/preprocessing_lab/run_smoothing_sweep.py --material Si
"""

from __future__ import annotations

import argparse

import numpy as np

from sers.preprocessing import resample, smooth_spectrum
from sers.preprocessing_lab.reference_eval import evaluate_reference, load_reference_sessions

REF_PEAKS = {
    "PS": [621.3, 794.8, 1001.2, 1032.0, 1157.4, 1327.1, 1448.6, 1602.8],
    "Si": [520.7],
}

# (라벨, kwargs) — smooth_spectrum에 그대로 전달
SETTINGS: list[tuple[str, dict]] = [
    ("none", {"method": "none"}),
    ("savgol w5 p3", {"method": "savgol", "window_length": 5, "polyorder": 3}),
    ("savgol w7 p3", {"method": "savgol", "window_length": 7, "polyorder": 3}),
    ("savgol w9 p3", {"method": "savgol", "window_length": 9, "polyorder": 3}),
    ("savgol w11 p3 *", {"method": "savgol", "window_length": 11, "polyorder": 3}),
    ("savgol w15 p3", {"method": "savgol", "window_length": 15, "polyorder": 3}),
    ("savgol w21 p3", {"method": "savgol", "window_length": 21, "polyorder": 3}),
    ("savgol w11 p2", {"method": "savgol", "window_length": 11, "polyorder": 2}),
    ("savgol w11 p4", {"method": "savgol", "window_length": 11, "polyorder": 4}),
    ("savgol w15 p4", {"method": "savgol", "window_length": 15, "polyorder": 4}),
    ("median w5", {"method": "median", "median_window": 5}),
    ("gaussian s1.0", {"method": "gaussian", "gaussian_sigma": 1.0}),
    ("gaussian s0.5", {"method": "gaussian", "gaussian_sigma": 0.5}),
    ("moving avg w5", {"method": "moving_average", "moving_window": 5}),
    ("wavelet haar", {"method": "wavelet_haar", "wavelet_threshold": 1.0}),
]


def measure(material: str) -> dict[str, tuple[float, float, float]]:
    """설정별 (SNR, FWHM, |위치이동|) 중앙값을 재서 반환한다."""
    peaks = REF_PEAKS[material]
    sessions = load_reference_sessions(material=material)
    grid = np.linspace(400.0, 2200.0, 935)
    out: dict[str, tuple[float, float, float]] = {}
    for label, kwargs in SETTINGS:
        snrs, fwhms, shifts = [], [], []
        for key, (x, spectra) in sorted(sessions.items()):
            processed = [resample(x, smooth_spectrum(row, **kwargs), grid) for row in spectra]
            ev = evaluate_reference(grid, np.vstack(processed), peaks,
                                    material=material, session=key)
            snrs.append(ev.median_snr)
            fwhms.append(ev.median_fwhm_cm1)
            shifts.append(ev.median_abs_shift_cm1)
        out[label] = (float(np.nanmedian(snrs)), float(np.nanmedian(fwhms)),
                      float(np.nanmedian(shifts)))
    return out


def combined_report() -> None:
    """PS와 Si를 함께 만족하는 설정을 고른다.

    두 표준물질은 피크 폭이 다르다(PS 약 9.8 cm-1, Si 약 8.0). 우리 대사체
    피크의 폭이 어느 쪽인지 확정되지 않았으므로, 한쪽만 보고 고르면 다른 쪽
    조건에서 실패한다. 지표는 물질별 절대 스케일이 크게 다르므로(SNR PS ~26 vs
    Si ~124) 반드시 `none` 대비 상대값으로 환산해 비교한다.
    """
    ps, si = measure("PS"), measure("Si")
    ps_base, si_base = ps["none"], si["none"]

    rows = []
    for label, _ in SETTINGS:
        g_ps = ps[label][0] / ps_base[0]
        g_si = si[label][0] / si_base[0]
        f_ps = (ps[label][1] / ps_base[1] - 1) * 100
        f_si = (si[label][1] / si_base[1] - 1) * 100
        rows.append((label, g_ps, g_si, f_ps, f_si, min(g_ps, g_si), max(f_ps, f_si),
                     (g_ps * g_si) ** 0.5))

    print(f"{'설정':<17} {'SNR배수':>16} {'FWHM증가':>16} | {'최악SNR':>8} {'최악FWHM':>9} {'기하평균':>9}")
    print(f"{'':17} {'PS':>7} {'Si':>8} {'PS':>7} {'Si':>8} |")
    print("-" * 82)
    for r in rows:
        star = " *" if r[0].endswith("*") else ""
        print(f"{r[0]:<17} {r[1]:7.2f} {r[2]:8.2f} {r[3]:+6.1f}% {r[4]:+7.1f}% |"
              f" {r[5]:8.2f} {r[6]:+8.1f}% {r[7]:9.2f}{star}")

    # 피크 손상 예산을 두고 pareto 최적 찾기
    print()
    print("피크 폭 증가를 두 물질 모두에서 예산 이내로 제한했을 때 최적 (SNR 기하평균 기준):")
    for budget in (5.0, 10.0, 15.0, 30.0):
        ok = [r for r in rows if r[6] <= budget and r[0] != "none"]
        best = max(ok, key=lambda r: r[7]) if ok else None
        if best:
            print(f"  FWHM 증가 ≤{budget:4.0f}% : {best[0]:<16} "
                  f"SNR 기하평균 {best[7]:.2f} (PS {best[1]:.2f} / Si {best[2]:.2f}), "
                  f"최악 FWHM {best[6]:+.1f}%")
        else:
            print(f"  FWHM 증가 ≤{budget:4.0f}% : 조건을 만족하는 smoothing 설정 없음")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--material", default="PS", choices=["PS", "Si"])
    ap.add_argument("--combined", action="store_true",
                    help="PS와 Si를 함께 평가해 두 물질 모두 만족하는 설정을 고른다")
    args = ap.parse_args()

    if args.combined:
        combined_report()
        return

    peaks = REF_PEAKS[args.material]
    sessions = load_reference_sessions(material=args.material)
    grid = np.linspace(400.0, 2200.0, 935)
    print(f"{args.material}: 세션 {len(sessions)}개 × replicate 5, 참조 피크 {len(peaks)}개")
    print("smoothing만 적용 (baseline/정규화 없음) 후 공통 격자로 리샘플\n")

    baseline_fwhm = None
    print(f"{'설정':<17} {'SNR':>7} {'FWHM':>8} {'FWHM증가':>9} {'|위치이동|':>10} {'면적비 변화':>11}")
    print("-" * 68)

    for label, kwargs in SETTINGS:
        snrs, fwhms, shifts, ratios = [], [], [], []
        for key, (x, spectra) in sorted(sessions.items()):
            processed = []
            for row in spectra:
                smoothed = smooth_spectrum(row, **kwargs)
                processed.append(resample(x, smoothed, grid))
            ev = evaluate_reference(grid, np.vstack(processed), peaks,
                                    material=args.material, session=key)
            snrs.append(ev.median_snr)
            fwhms.append(ev.median_fwhm_cm1)
            shifts.append(ev.median_abs_shift_cm1)
            if len(peaks) > 1:
                ratios.append(float(np.median([p.area_ratio for p in ev.peaks[1:]])))

        snr, fwhm = np.nanmedian(snrs), np.nanmedian(fwhms)
        if baseline_fwhm is None:
            baseline_fwhm = fwhm
            base_ratio = np.nanmedian(ratios) if ratios else float("nan")
        growth = (fwhm / baseline_fwhm - 1) * 100
        rchg = ((np.nanmedian(ratios) / base_ratio - 1) * 100) if ratios else float("nan")
        print(f"{label:<17} {snr:7.1f} {fwhm:8.2f} {growth:+8.1f}% "
              f"{np.nanmedian(shifts):10.2f} {rchg:+10.1f}%")

    print("\n  * = 현재 프로덕션 설정 (config.yaml)")
    print("  FWHM증가는 smoothing 없음(none) 대비. 피크가 뭉개진 정도.")
    print("  면적비 변화는 기준 피크 대비 상대 면적의 변화 — 0에 가까울수록 형상 보존.")


if __name__ == "__main__":
    main()
