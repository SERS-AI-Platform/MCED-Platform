"""
stk_v2_preprocess.py
====================
STK-V2 전처리 재현 모듈 (API 산출 방식과 동일하게 정합)

목적
----
현재 노트북에 들어간 신규 전처리(AsLS baseline + area-normalize + smoothing 비활성화)를
STK-V2 production이 실제로 쓰던 전처리로 되돌린다. STK-V2 스펙은 다음과 같다.

    장비 CSV (Raman shift x, intensity y)
     → 400–2200 cm⁻¹ crop
     → Savitzky–Golay(window=11, poly=3)
     → ch0: smoothing + Rolling-Minimum baseline(window=101) + SNV
     → ch1: 1차 SG derivative + SNV
     → ch2: 2차 SG derivative + SNV
     → 각 channel을 402–2198 cm⁻¹, 935-point grid로 interpolation
     → shape = (3, 935)

Baseline은 AsLS가 아니라 Rolling-Minimum(window=101). Phase R에서 Rolling-Min > ALS로
검증됨(Det AUC 0.977 vs 0.976, Id F1 0.892 vs 0.885, ALS 5배 느림).

이 모듈은 두 가지 진입점을 제공한다.
  1) stk_v2_channels(...)      : 모델 입력용 (3, 935) 3채널 재현 (STK-V2 완전 동일)
  2) stk_v2_peak_input(...)    : peak registry 파이프라인용 단일 채널(ch0-equivalent) 반환
                                 → 기존 _baseline_area_normalize(X_RAW_SUBJECT_MEAN) 자리에
                                   그대로 대체해서 넣으면 됨.

주의: rolling-minimum 세부 구현(엔벨로프 스무딩 포함 여부)은 src/sers/preprocessing.py의
production 함수와 반드시 대조 확인할 것. 아래 구현은 "rolling min → rolling mean 엔벨로프"라는
가장 표준적인 형태이며, STK-V2 preprocessing.py와 1:1로 맞추는 것이 최종 기준이다.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

# ---------------------------------------------------------------------------
# STK-V2 고정 상수 (artifact 기준)
# ---------------------------------------------------------------------------
STK_CROP_MIN_CM1 = 400.0            # 물리 범위 crop 하한
STK_CROP_MAX_CM1 = 2200.0           # 물리 범위 crop 상한
STK_GRID_MIN_CM1 = 402.0            # 최종 grid 하한
STK_GRID_MAX_CM1 = 2198.0           # 최종 grid 상한
STK_GRID_POINTS = 935               # 고정 wavenumber point 수 (약 1.923 cm⁻¹ 간격)

STK_SG_WINDOW = 11                  # Savitzky–Golay window
STK_SG_POLY = 3                     # Savitzky–Golay polynomial order
STK_BASELINE_WINDOW = 101           # Rolling-Minimum baseline window

# 최종 고정 grid (402–2198 cm⁻¹, 935 point)
STK_TARGET_GRID = np.linspace(
    STK_GRID_MIN_CM1,
    STK_GRID_MAX_CM1,
    STK_GRID_POINTS,
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# 기본 연산자
# ---------------------------------------------------------------------------
def _savgol(values: np.ndarray, deriv: int = 0) -> np.ndarray:
    """Savitzky–Golay smoothing 또는 derivative (window=11, poly=3)."""
    values_array = np.asarray(values, dtype=np.float64)
    length = len(values_array)
    if length < STK_SG_WINDOW:
        # 너무 짧으면 홀수·poly 조건을 만족하는 최대 window로 축소
        window = length if length % 2 == 1 else length - 1
        window = max(window, STK_SG_POLY + 1 + (STK_SG_POLY % 2 == 0))
    else:
        window = STK_SG_WINDOW
    return savgol_filter(
        values_array,
        window_length=window,
        polyorder=STK_SG_POLY,
        deriv=deriv,
    )


def _rolling_minimum_baseline(
    values: np.ndarray,
    window: int = STK_BASELINE_WINDOW,
) -> np.ndarray:
    """
    Rolling-Minimum baseline (window=101).

    각 point 기준 ±window/2 구간의 최소값으로 형광 배경 엔벨로프를 만들고,
    같은 window의 rolling mean으로 부드럽게 만들어 baseline으로 사용한다.
    (STK-V2 preprocessing.py의 production 구현과 반드시 대조 확인)
    """
    values_array = np.asarray(values, dtype=np.float64)
    length = len(values_array)
    half = window // 2

    # 경계 반사 패딩으로 edge 왜곡 최소화
    padded = np.pad(values_array, half, mode="reflect")

    # rolling minimum
    minimum_envelope = np.empty(length, dtype=np.float64)
    for index in range(length):
        minimum_envelope[index] = padded[index : index + window].min()

    # rolling mean으로 엔벨로프 스무딩
    smoothed_pad = np.pad(minimum_envelope, half, mode="reflect")
    kernel = np.ones(window, dtype=np.float64) / window
    baseline = np.convolve(smoothed_pad, kernel, mode="valid")

    # convolve 'valid' 길이 보정
    if len(baseline) != length:
        baseline = baseline[:length]
    return baseline


def _snv(values: np.ndarray) -> np.ndarray:
    """Standard Normal Variate: (x - mean) / std."""
    values_array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values_array))
    std = float(np.std(values_array))
    if std <= np.finfo(float).eps:
        return values_array - mean
    return (values_array - mean) / std


def _interp_to_grid(
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_grid: np.ndarray = STK_TARGET_GRID,
) -> np.ndarray:
    """source_x 좌표의 값을 STK-V2 고정 935 grid로 interpolation."""
    source_x = np.asarray(source_x, dtype=np.float64)
    source_y = np.asarray(source_y, dtype=np.float64)
    order = np.argsort(source_x)
    return np.interp(
        target_grid,
        source_x[order],
        source_y[order],
    )


# ---------------------------------------------------------------------------
# 진입점 1) 모델 입력용 3채널 (STK-V2 완전 동일)
# ---------------------------------------------------------------------------
def stk_v2_channels(
    wavenumber: np.ndarray,
    intensity: np.ndarray,
    target_grid: np.ndarray = STK_TARGET_GRID,
) -> np.ndarray:
    """
    한 spectrum을 STK-V2 (3, 935) 표현으로 변환한다.

    Parameters
    ----------
    wavenumber : 장비가 export한 Raman shift 축 (crop 전 원본)
    intensity  : 대응 intensity
    target_grid: 최종 935-point grid (기본 STK_TARGET_GRID)

    Returns
    -------
    np.ndarray, shape = (3, len(target_grid))
        ch0 = smoothing + rolling-min baseline + SNV
        ch1 = 1차 SG derivative + SNV
        ch2 = 2차 SG derivative + SNV
    """
    wavenumber = np.asarray(wavenumber, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)

    # 1) 400–2200 cm⁻¹ crop
    crop_mask = (wavenumber >= STK_CROP_MIN_CM1) & (wavenumber <= STK_CROP_MAX_CM1)
    x_crop = wavenumber[crop_mask]
    y_crop = intensity[crop_mask]

    # 2) Savitzky–Golay smoothing
    y_smooth = _savgol(y_crop, deriv=0)

    # 3) ch0: rolling-min baseline 제거 + SNV
    baseline = _rolling_minimum_baseline(y_smooth)
    ch0 = _snv(y_smooth - baseline)

    # 4) ch1: 1차 SG derivative + SNV
    ch1 = _snv(_savgol(y_crop, deriv=1))

    # 5) ch2: 2차 SG derivative + SNV
    ch2 = _snv(_savgol(y_crop, deriv=2))

    # 6) 935-point grid interpolation
    ch0_g = _interp_to_grid(x_crop, ch0, target_grid)
    ch1_g = _interp_to_grid(x_crop, ch1, target_grid)
    ch2_g = _interp_to_grid(x_crop, ch2, target_grid)

    return np.vstack([ch0_g, ch1_g, ch2_g]).astype(np.float64)


# ---------------------------------------------------------------------------
# 진입점 2) peak registry 파이프라인용 단일 채널 (ch0-equivalent)
# ---------------------------------------------------------------------------
def stk_v2_peak_input(
    spectra: np.ndarray,
    source_grid: np.ndarray,
    target_grid: np.ndarray = STK_TARGET_GRID,
) -> np.ndarray:
    """
    기존 노트북의 `X = _baseline_area_normalize(X_RAW_SUBJECT_MEAN)` 자리를
    그대로 대체하기 위한 함수.

    입력이 이미 (subject, point) 형태의 raw subject-mean 스펙트럼이고
    x축 좌표(source_grid)를 알고 있다고 가정한다. 각 row에 STK-V2 ch0
    (SG smooth → rolling-min baseline → SNV)을 적용한 뒤 935 grid로 맞춰 반환한다.

    Parameters
    ----------
    spectra     : 1D (point,) 또는 2D (subject, point)
    source_grid : spectra의 x축 (길이 = spectra의 마지막 축)
    target_grid : 최종 935-point grid

    Returns
    -------
    np.ndarray, 입력이 1D면 (935,), 2D면 (subject, 935)
    """
    values = np.asarray(spectra, dtype=np.float64)
    source_grid = np.asarray(source_grid, dtype=np.float64)
    matrix = values[None, :] if values.ndim == 1 else values

    processed = np.empty((matrix.shape[0], len(target_grid)), dtype=np.float64)
    for index, row in enumerate(matrix):
        y_smooth = _savgol(row, deriv=0)
        baseline = _rolling_minimum_baseline(y_smooth)
        ch0 = _snv(y_smooth - baseline)
        processed[index] = _interp_to_grid(source_grid, ch0, target_grid)

    return processed[0] if values.ndim == 1 else processed


# ---------------------------------------------------------------------------
# 간단한 자가 점검
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = np.linspace(400.0, 2200.0, 1000)
    # 형광 배경 + 몇 개 peak + noise 로 합성
    background = 500.0 * np.exp(-((x - 400.0) / 900.0))
    peaks = (
        120.0 * np.exp(-((x - 1001.0) ** 2) / (2 * 6.0 ** 2))
        + 80.0 * np.exp(-((x - 1448.0) ** 2) / (2 * 8.0 ** 2))
    )
    y = background + peaks + rng.normal(0, 3.0, size=x.shape)

    channels = stk_v2_channels(x, y)
    print("3-channel shape:", channels.shape)  # (3, 935)

    peak_in = stk_v2_peak_input(np.vstack([y, y * 1.02]), x)
    print("peak-input shape:", peak_in.shape)   # (2, 935)
    print("ch0 mean≈0, std≈1 (SNV 확인):",
          round(float(channels[0].mean()), 4),
          round(float(channels[0].std()), 4))
