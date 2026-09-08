"""
build_stkv2_dataset.py
======================
STK-V2 전처리 데이터셋 생성 · 저장 스크립트

stk_v2_preprocess.py 의 함수들을 실제 subject-level 스펙트럼에 호출해서
모델링에 바로 쓸 수 있는 배열(.npy)로 저장한다.

산출물 (OUT_DIR 에 저장)
------------------------
  X_ch0.npy    (n_subject, 935)  ch0 = smooth + rolling-min baseline + SNV
  X_d1.npy     (n_subject, 935)  ch1 = 1차 SG derivative + SNV
  X_d2.npy     (n_subject, 935)  ch2 = 2차 SG derivative + SNV
  X_peak.npy   (n_subject, 75)   17 고정 peak Voigt feature (area/height/fwhm/shift + ratio)
  y_screen.npy (n_subject,)      1 = cancer, 0 = non-cancer
  y_type.npy   (n_subject,)      7-class cancer type (non-cancer 는 -1)
  groups.npy   (n_subject,)      subject id (CV group key)

데이터 소스 연결
----------------
load_subject_spectra() 안의 TODO 를 실제 aecd_api 호출로 채운다.
지금은 DEMO(합성 신호)로 동작하며, --demo 를 빼고 실제 loader 를 연결하면 된다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.special import voigt_profile
from scipy.optimize import curve_fit

from stk_v2_preprocess import (
    STK_TARGET_GRID,
    stk_v2_channels,
    _savgol,
    _rolling_minimum_baseline,
    _snv,
    _interp_to_grid,
)

# STK-V2 production 17 고정 peak center (cm^-1)
STK_FIXED_PEAKS_CM1 = np.array([
    448.1, 538.7, 617.7, 683.3, 723.8, 795.1, 849.1,
    895.4, 933.9, 999.5, 1147.9, 1230.8, 1292.5,
    1352.3, 1448.7, 1597.1, 1651.1,
], dtype=np.float64)
PEAK_HALF_WINDOW_CM1 = 25.0   # 각 peak 주변 ±window (production peak_config.json 값으로 교체)

CANCER_TYPES = ["PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC"]


# ---------------------------------------------------------------------------
# 17 고정 peak Voigt feature (75개) 추출
# ---------------------------------------------------------------------------
def _voigt_single(x, amplitude, center, sigma, gamma):
    return amplitude * voigt_profile(x - center, sigma, gamma)


def _voigt_fwhm(sigma, gamma):
    fg = 2.0 * sigma * np.sqrt(2.0 * np.log(2.0))
    fl = 2.0 * gamma
    return 0.5346 * fl + np.sqrt(0.2166 * fl**2 + fg**2)


def extract_peak_features(grid: np.ndarray, ch0: np.ndarray) -> np.ndarray:
    """
    ch0 (baseline 제거 + SNV 된 spectrum)에서 17개 고정 peak Voigt fitting.
    peak당 area/height/fwhm/shift 4개 → 68개, + 7개 area ratio = 75개.
    fitting 실패 시 area/height/fwhm=0, shift=0.
    """
    grid = np.asarray(grid, dtype=np.float64)
    ch0 = np.asarray(ch0, dtype=np.float64)

    areas, heights, fwhms, shifts = [], [], [], []
    for center in STK_FIXED_PEAKS_CM1:
        lo = center - 1.5 * PEAK_HALF_WINDOW_CM1
        hi = center + 1.5 * PEAK_HALF_WINDOW_CM1
        mask = (grid >= lo) & (grid <= hi)
        xw, yw = grid[mask], ch0[mask]

        if len(xw) < 6:
            areas.append(0.0); heights.append(0.0); fwhms.append(0.0); shifts.append(0.0)
            continue
        try:
            amp0 = max(float(np.ptp(yw)), 1e-6)
            p0 = [amp0, center, 4.0, 4.0]
            bounds = ([0.0, lo, 0.1, 0.1], [np.inf, hi, 50.0, 50.0])
            popt, _ = curve_fit(_voigt_single, xw, yw, p0=p0, bounds=bounds, maxfev=5000)
            amp, cen, sig, gam = popt
            area = float(np.trapz(_voigt_single(xw, *popt), xw))
            areas.append(area)
            heights.append(float(amp * voigt_profile(0.0, sig, gam)))
            fwhms.append(float(_voigt_fwhm(sig, gam)))
            shifts.append(float(cen - center))
        except Exception:
            areas.append(0.0); heights.append(0.0); fwhms.append(0.0); shifts.append(0.0)

    areas = np.asarray(areas, dtype=np.float64)
    # 7개 area ratio: 대표 peak 대비 상대비 (999.5 = urea 기준 index 9)
    ref = areas[9] if areas[9] > 0 else (areas.max() if areas.max() > 0 else 1.0)
    ratio_idx = [0, 4, 8, 10, 12, 14, 15]  # 대표 7개 (peak_config 와 정합 필요)
    ratios = np.asarray([areas[i] / ref if ref > 0 else 0.0 for i in ratio_idx])

    return np.concatenate([areas, np.asarray(heights), np.asarray(fwhms),
                           np.asarray(shifts), ratios]).astype(np.float64)


# ---------------------------------------------------------------------------
# 데이터 소스 (실제 aecd_api 연결 지점)
# ---------------------------------------------------------------------------
def load_subject_spectra(demo: bool = True):
    """
    return: list of (subject_id, wavenumber, intensity, label_screen, label_type)
    label_type: cancer type string 또는 None(non-cancer)

    TODO(실서버 연결):
      - GET /v1/cohorts, /v1/spectra 로 subject별 replicate 스펙트럼 수집
      - QC pass replicate 의 subject-mean intensity 계산
      - (wavenumber, mean_intensity) 반환
    """
    if not demo:
        raise NotImplementedError(
            "load_subject_spectra: aecd_api 호출부를 여기에 구현하세요 "
            "(GET /v1/spectra → subject-mean)."
        )

    # ---- DEMO: 합성 subject 200명 ----
    rng = np.random.default_rng(42)
    x = np.linspace(400.0, 2200.0, 1000)
    out = []
    for sid in range(200):
        is_cancer = sid % 2 == 0
        bg = 500.0 * np.exp(-((x - 400.0) / 900.0))
        # cancer 는 특정 peak 강도 상승
        boost = 1.6 if is_cancer else 1.0
        peaks = (
            120.0 * boost * np.exp(-((x - 999.5) ** 2) / (2 * 6.0 ** 2))
            + 80.0 * np.exp(-((x - 1448.7) ** 2) / (2 * 8.0 ** 2))
            + 60.0 * boost * np.exp(-((x - 1352.3) ** 2) / (2 * 7.0 ** 2))
        )
        y = bg + peaks + rng.normal(0, 4.0, size=x.shape)
        ctype = CANCER_TYPES[sid % 7] if is_cancer else None
        out.append((f"S{sid:04d}", x.copy(), y, int(is_cancer), ctype))
    return out


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./stkv2_dataset", help="출력 디렉토리")
    ap.add_argument("--demo", action="store_true", help="합성 데이터로 실행")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = load_subject_spectra(demo=args.demo)
    grid = STK_TARGET_GRID

    X_ch0, X_d1, X_d2, X_peak = [], [], [], []
    y_screen, y_type, groups = [], [], []

    for sid, wn, inten, lab_s, lab_t in subjects:
        channels = stk_v2_channels(wn, inten, target_grid=grid)  # (3, 935)
        ch0, ch1, ch2 = channels
        X_ch0.append(ch0); X_d1.append(ch1); X_d2.append(ch2)
        X_peak.append(extract_peak_features(grid, ch0))
        y_screen.append(lab_s)
        y_type.append(CANCER_TYPES.index(lab_t) if lab_t in CANCER_TYPES else -1)
        groups.append(sid)

    X_ch0 = np.asarray(X_ch0); X_d1 = np.asarray(X_d1); X_d2 = np.asarray(X_d2)
    X_peak = np.asarray(X_peak)
    y_screen = np.asarray(y_screen); y_type = np.asarray(y_type)
    groups = np.asarray(groups)

    np.save(out_dir / "X_ch0.npy", X_ch0)
    np.save(out_dir / "X_d1.npy", X_d1)
    np.save(out_dir / "X_d2.npy", X_d2)
    np.save(out_dir / "X_peak.npy", X_peak)
    np.save(out_dir / "y_screen.npy", y_screen)
    np.save(out_dir / "y_type.npy", y_type)
    np.save(out_dir / "groups.npy", groups)

    print("[saved]", out_dir.resolve())
    print(f"  X_ch0 {X_ch0.shape}  X_d1 {X_d1.shape}  X_d2 {X_d2.shape}")
    print(f"  X_peak {X_peak.shape}  y_screen {y_screen.shape} "
          f"(cancer {int(y_screen.sum())}/{len(y_screen)})")


if __name__ == "__main__":
    main()
