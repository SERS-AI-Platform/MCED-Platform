#!/usr/bin/env python
"""Whitaker-Hayes threshold 스윕 — 표준물질(PS) 기준.

PL-1에서 despike가 AUC를 떨어뜨렸고, 표준물질 평가에서 SNR도 낮췄다. 원인은
이 장비의 샘플링 격자(1.93 cm-1)에서 PS 피크가 약 5포인트뿐이라 1차 차분이
커져 **진짜 피크가 spike로 오인**되기 때문으로 보인다. threshold를 올리면
해결되는지, 올리면 정작 진짜 spike를 놓치지는 않는지를 함께 잰다.

두 축을 동시에 측정한다:
  harm    : 원본(깨끗한) PS에 적용했을 때 피크 손상 (오인 flag 수, SNR, FWHM)
  benefit : 인공 spike를 주입한 뒤 복구 정확도 (주입 전 스펙트럼이 ground truth)

Usage:
    python scripts/analysis/preprocessing_lab/run_despike_threshold_sweep.py
"""

from __future__ import annotations

import argparse

import numpy as np

from sers.preprocessing import whitaker_hayes_despike
from sers.preprocessing_lab.reference_eval import (
    evaluate_reference,
    load_reference_sessions,
)

REF_PEAKS_PS = [621.3, 794.8, 1001.2, 1032.0, 1157.4, 1327.1, 1448.6, 1602.8]
THRESHOLDS = (4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0)
PEAK_GUARD_CM1 = 6.0


def flag_mask(y: np.ndarray, threshold: float) -> np.ndarray:
    """whitaker_hayes_despike 내부와 동일한 flag 규칙 (force_endpoints 제외)."""
    grad = np.diff(y)
    mad = float(np.median(np.abs(grad - np.median(grad))))
    if mad <= 0:
        return np.zeros_like(y, dtype=bool)
    z = np.zeros_like(y, dtype=float)
    z[1:] = 0.6745 * (grad - np.median(grad)) / mad
    return np.abs(z) > threshold


def local_noise_sigma(y: np.ndarray) -> float:
    """1차 차분 기반 노이즈 추정. 인접 차분의 분산은 노이즈 분산의 2배."""
    return float(np.std(np.diff(y)) / np.sqrt(2.0))


def inject_spikes(y: np.ndarray, x: np.ndarray, rng: np.random.Generator,
                  n_spikes: int = 3, amplitude_ratio: float = 3.0,
                  amplitude_sigma: float | None = None) -> tuple[np.ndarray, list[int], float]:
    """참조 피크에서 떨어진 위치에 단일 포인트 spike를 주입한다.

    cosmic ray는 CCD 단일 픽셀 사건이므로 1포인트 spike로 모사한다. 진짜 피크
    위에 놓으면 복구 평가가 오염되므로 피크 근방은 피한다.

    `amplitude_sigma`가 주어지면 진폭을 국소 노이즈 σ의 배수로 지정한다
    (탐지 이론상 자연스러운 단위). 아니면 스펙트럼 전체 범위의 배수를 쓴다.
    """
    out = y.copy()
    span = float(y.max() - y.min())
    if amplitude_sigma is not None:
        amp = amplitude_sigma * local_noise_sigma(y)
    else:
        amp = amplitude_ratio * span
    far = np.ones_like(x, dtype=bool)
    for r in REF_PEAKS_PS:
        far &= np.abs(x - r) > 15.0
    far[:10] = far[-10:] = False
    candidates = np.flatnonzero(far)
    idx = rng.choice(candidates, size=n_spikes, replace=False)
    out[idx] += amp
    return out, sorted(int(i) for i in idx), amp


def amplitude_sweep(sessions, rng, window: int, thresholds, amplitudes) -> None:
    """진폭(노이즈 σ 배수)을 바꿔가며 threshold별 탐지·복구 성능을 잰다.

    threshold 상한을 정하려면 "얼마나 작은 spike까지 잡아야 하는가"를 알아야
    한다. 앞선 스윕은 진폭이 전체 범위의 3배(=σ의 수백 배)로 지나치게 커서
    threshold 차이가 드러나지 않았다.
    """
    print(f"{'진폭(σ배수)':>12} | " + " | ".join(f"thr={t:<5.0f}" for t in thresholds))
    print("-" * (14 + 11 * len(thresholds)))
    for amp_sigma in amplitudes:
        cells = []
        for thr in thresholds:
            detected, residual = [], []
            for _key, (x, spectra) in sorted(sessions.items()):
                for row in spectra:
                    spiked, idx, amp = inject_spikes(row, x, rng, amplitude_sigma=amp_sigma)
                    fixed = whitaker_hayes_despike(spiked, z_threshold=thr, window=window)
                    mask = flag_mask(spiked, thr)
                    detected.append(np.mean([mask[i] for i in idx]) * 100)
                    # 남은 spike 비율: 0 = 완전 제거, 1 = 그대로 남음
                    residual.append(np.mean([abs(fixed[i] - row[i]) / amp for i in idx]))
            cells.append(f"{np.mean(detected):5.0f}% {np.mean(residual):.2f}")
        print(f"{amp_sigma:12.1f} | " + " | ".join(cells))
    print("\n  각 칸 = 탐지율% / 잔여비율(0=완전제거, 1=그대로)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--mode", choices=["threshold", "amplitude"], default="threshold")
    args = ap.parse_args()

    sessions = load_reference_sessions(material="PS")
    rng = np.random.default_rng(args.seed)
    print(f"PS 세션 {len(sessions)}개 × replicate 5, 참조 피크 {len(REF_PEAKS_PS)}개")
    print("격자 간격 약 1.93 cm-1 → PS 피크(FWHM ~10)당 약 5포인트\n")

    if args.mode == "amplitude":
        amplitude_sweep(sessions, rng, args.window,
                        thresholds=(6.0, 10.0, 20.0, 30.0, 50.0),
                        amplitudes=(3.0, 5.0, 8.0, 12.0, 20.0, 40.0, 100.0))
        return

    print(f"{'thr':>5} | {'flag율%':>7} {'피크오인':>8} {'SNR':>7} {'FWHM':>7} "
          f"| {'spike제거%':>10} {'복구RMSE':>10}")
    print("-" * 72)

    for thr in THRESHOLDS:
        flag_rates, peak_hits, snrs, fwhms = [], [], [], []
        removed, rmses = [], []

        for _key, (x, spectra) in sorted(sessions.items()):
            # --- harm: 깨끗한 원본에 적용 ---
            for row in spectra:
                mask = flag_mask(row, thr)
                flag_rates.append(mask.mean() * 100)
                hits = sum(1 for r in REF_PEAKS_PS
                           if mask[np.abs(x - r) <= PEAK_GUARD_CM1].any())
                peak_hits.append(hits)

            cleaned = np.vstack([whitaker_hayes_despike(r, z_threshold=thr,
                                                        window=args.window) for r in spectra])
            ev = evaluate_reference(x, cleaned, REF_PEAKS_PS, material="PS", session=_key)
            snrs.append(ev.median_snr)
            fwhms.append(ev.median_fwhm_cm1)

            # --- benefit: spike 주입 후 복구 ---
            for row in spectra:
                spiked, idx, _amp = inject_spikes(row, x, rng)
                fixed = whitaker_hayes_despike(spiked, z_threshold=thr, window=args.window)
                # 주입 지점이 원본 수준으로 되돌아왔는가 (원본 노이즈 폭 기준)
                tol = 3.0 * np.std(np.diff(row)) / np.sqrt(2)
                removed.append(np.mean([abs(fixed[i] - row[i]) < tol for i in idx]) * 100)
                # 끝점 강제 교체 영향을 빼고 비교
                rmses.append(float(np.sqrt(np.mean((fixed[1:-1] - row[1:-1]) ** 2))))

        print(f"{thr:5.0f} | {np.mean(flag_rates):7.2f} {np.mean(peak_hits):8.2f} "
              f"{np.median(snrs):7.1f} {np.median(fwhms):7.2f} | "
              f"{np.mean(removed):10.1f} {np.mean(rmses):10.1f}")

    print("\n해석")
    print("  피크오인: 참조 피크 8개 중 ±6cm-1 안에 flag가 생긴 피크 수 (0이어야 안전)")
    print("  spike제거%: 주입한 인공 spike가 원본 수준으로 복구된 비율 (높을수록 좋음)")
    print("  복구RMSE : 주입 전 원본과의 차이 (낮을수록 좋음, 끝점 2개 제외)")


if __name__ == "__main__":
    main()
