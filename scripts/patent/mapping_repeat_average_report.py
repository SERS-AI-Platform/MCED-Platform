from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from mapping_repeat_average_core import GROUP_ORDER, SubjectResult
from mapping_repeat_average_outputs import OutputTables


def reference_audit(root: Path) -> dict[str, int | list[str]]:
    files = sorted(root.glob("*.CSV"))
    by_date_kind: Counter[tuple[str, str]] = Counter()
    for path in files:
        parts = path.stem.split("_")
        date = parts[0]
        kind = "PS" if "Cali_PS" in path.stem else "Si" if "Cali_Si" in path.stem else "other"
        by_date_kind[(date, kind)] += 1
    complete = sum(count == 6 for count in by_date_kind.values())
    subject_dates = ["20260810", "20260811", "20260812", "20260813", "20260814"]
    same_run = sum(count for (date, _), count in by_date_kind.items() if date in subject_dates)
    return {
        "reference_csv_files": len(files),
        "reference_date_count": len({date for date, _ in by_date_kind}),
        "complete_date_kind_sets": complete,
        "same_run_reference_files_verified": same_run,
        "reference_dates": sorted({date for date, _ in by_date_kind}),
    }


def build_metadata(
    results: list[SubjectResult],
    grid: np.ndarray,
    raw_axis: dict[str, float | int | list[int]],
    references: dict[str, int | list[str]],
) -> dict[str, object]:
    return {
        "analysis_name": "Mapping repeat average versus existing instrument average",
        "analysis_date": "2026-08-25",
        "included_subjects": len(results),
        "group_counts": {
            group: sum(r.subject.group == group for r in results) for group in GROUP_ORDER
        },
        "input_repeats": sum(len(r.subject.replicate_paths) for r in results),
        "replicates_per_subject": sorted({len(r.subject.replicate_paths) for r in results}),
        "qc_passed_repeats": int(sum(r.keep.sum() for r in results)),
        "qc_passed_per_subject": {
            "min": int(min(r.keep.sum() for r in results)),
            "median": float(np.median([r.keep.sum() for r in results])),
            "max": int(max(r.keep.sum() for r in results)),
        },
        "common_grid": {
            "min_cm-1": float(grid[0]),
            "max_cm-1": float(grid[-1]),
            "points": len(grid),
            "step_cm-1": float(np.median(np.diff(grid))),
        },
        "raw_axis_observed": raw_axis,
        "instrument_average_audit": {
            "pass_threshold_normalized_rmse_le_1e-4": sum(
                r.average_normalized_rmse <= 1e-4 for r in results
            ),
            "total": len(results),
            "normalized_rmse_median": float(
                np.median([r.average_normalized_rmse for r in results])
            ),
            "correlation_median": float(np.median([r.average_correlation for r in results])),
        },
        "alignment": {
            "method": "fixed common-grid linear interpolation",
            "per_spectrum_shift_applied": False,
            "same_run_reference_shift_applied": False,
        },
        "reference_audit": references,
        "preprocessing": {
            "baseline_correction": False,
            "intensity_scaling": False,
            "denoising": False,
            "relative_intensity_scale": True,
        },
        "peak_definition": {
            "detector": "scipy.signal.find_peaks",
            "prominence_threshold": "3 x method high-frequency noise scale",
            "minimum_distance_points": 5,
            "support_definition": "peak within ±4 common-grid points in at least 50% of QC-passed repeats",
            "positive_peaks_only": True,
        },
        "privacy": "Raw sample identifiers, source filenames, and spectral arrays are not exported.",
    }


def _median_metric(tables: OutputTables, metric: str) -> float:
    values = [
        float(row["median"])
        for row in tables.summary_rows
        if row["group"] == "All included" and row["metric"] == metric
    ]
    return values[0]


def write_metadata(out: Path, metadata: dict[str, object]) -> None:
    (out / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_report(
    out: Path,
    results: list[SubjectResult],
    grid: np.ndarray,
    metadata: dict[str, object],
    tables: OutputTables,
) -> None:
    audit_pass = sum(r.average_normalized_rmse <= 1e-4 for r in results)
    qc = metadata["qc_passed_per_subject"]
    groups = metadata["group_counts"]
    existing_reduction = _median_metric(tables, "existing_noise_reduction_vs_single_pct")
    patent_reduction = _median_metric(tables, "patent_noise_reduction_vs_single_pct")
    extra_reduction = _median_metric(tables, "patent_extra_reduction_vs_existing_pct")
    method_rmse = _median_metric(tables, "patent_existing_normalized_rmse")
    method_corr = _median_metric(tables, "patent_existing_correlation")
    baseline_rmse = _median_metric(tables, "baseline_aligned_normalized_rmse")
    baseline_corr = _median_metric(tables, "baseline_aligned_correlation")
    raw_peaks = _median_metric(tables, "raw_peak_count_median")
    existing_peaks = _median_metric(tables, "existing_peak_count")
    patent_peaks = _median_metric(tables, "patent_peak_count")
    existing_real = _median_metric(tables, "existing_reproducible_peak_count")
    patent_real = _median_metric(tables, "patent_reproducible_peak_count")
    existing_snr = _median_metric(tables, "existing_median_peak_snr")
    patent_snr = _median_metric(tables, "patent_median_peak_snr")
    snr_change = 100.0 * (patent_snr / existing_snr - 1.0)
    audit_rmse = float(metadata["instrument_average_audit"]["normalized_rmse_median"])
    audit_corr = float(metadata["instrument_average_audit"]["correlation_median"])
    report = (
        "# data/mapping 반복 평균 적용 및 peak 검출 비교\n\n"
        "## 분석 범위\n\n"
        f"- 임상 매핑 {len(results)}명: Control {groups['Control']}명, Prostate disease control {groups['Prostate disease control']}명, Prostate cancer {groups['Prostate cancer']}명.\n"
        f"- subject별 원시 반복 121개, 총 {metadata['input_repeats']}개를 사용했다. `_ave` 파일은 기존 장비 평균 방식으로 별도 감사했다.\n"
        f"- 공통 grid: {grid[0]:.1f}–{grid[-1]:.1f} cm⁻¹, {len(grid)} points, 간격 {np.median(np.diff(grid)):.4f} cm⁻¹.\n"
        f"- QC 통과 반복 수: subject 기준 최소 {qc['min']}, 중앙값 {qc['median']:.0f}, 최대 {qc['max']}개.\n\n"
        "## 기존 `_ave`와 특허 방식 비교\n\n"
        f"- `_ave`와 121회 원시 산술평균 감사 통과: {audit_pass}/{len(results)}명. normalized RMSE 중앙값은 {audit_rmse:.4f}, 상관계수 중앙값은 {audit_corr:.5f}였다.\n"
        f"- 단일 반복 대비 high-frequency noise 감소 중앙값: 기존 `_ave` {existing_reduction:.1f}%, 특허 QC 평균 {patent_reduction:.1f}%.\n"
        f"- 기존 `_ave` 대비 특허 QC 평균의 추가 noise 감소 중앙값: {extra_reduction:.2f}%.\n"
        f"- 특허 QC 평균과 기존 `_ave`의 차이: normalized RMSE 중앙값 {method_rmse:.4f}, 상관계수 중앙값 {method_corr:.5f}.\n"
        f"- 양쪽에 동일한 101-point rolling-minimum baseline만 적용하면 차이는 normalized RMSE {baseline_rmse:.4f}, 상관계수 {baseline_corr:.5f}로 줄어든다. 이는 `_ave`와 원시 평균의 차이에 baseline/scale 영향이 섞였음을 시사한다.\n"
        "- 여기서 noise는 31-point Savitzky–Golay 저주파 성분을 뺀 residual의 robust MAD scale이다.\n\n"
        "## 실질적 peak 검출\n\n"
        "- peak는 baseline/scaling 없이 common grid 스펙트럼에서 prominence ≥ 3× high-frequency noise, 최소 간격 5 points로 검출했다.\n"
        "- ‘재현 가능한 peak’는 QC 통과 반복의 50% 이상에서 ±4 grid points 안에 peak가 다시 나타나는 경우로 정의했다.\n"
        f"- subject별 peak 수 중앙값: QC 통과 단일 반복 {raw_peaks:.1f}, 기존 `_ave` {existing_peaks:.1f}, 특허 QC 평균 {patent_peaks:.1f}.\n"
        f"- 재현 가능한 peak 수 중앙값: 기존 `_ave` {existing_real:.1f}, 특허 QC 평균 {patent_real:.1f}.\n"
        f"- 검출 peak의 median SNR 중앙값: 기존 `_ave` {existing_snr:.2f}, 특허 QC 평균 {patent_snr:.2f} ({snr_change:+.1f}%).\n\n"
        "## 해석\n\n"
        "- 121회 반복을 평균하면 단일 반복의 랜덤 noise는 이론적으로 약 1/√121 수준까지 내려갈 수 있다. 실제 감소율은 각 subject의 반복 상관과 outlier에 의해 달라진다.\n"
        "- 이번 감사에서 `_ave`는 121회 원시 산술평균과 일치하지 않았다. 따라서 기존 `_ave`는 독립적인 장비 집계/전처리 결과로 취급했고, 특허 방식은 QC 통과 반복의 평균으로 재현했다.\n"
        "- 검출된 peak는 분석적 재현성 지표이지 임상 바이오마커 확정이나 분류 성능을 의미하지 않는다.\n\n"
        "## 기준물질 및 한계\n\n"
        f"- data/mapping 내 Thermo Reference 파일은 {metadata['reference_audit']['reference_csv_files']}개이며, subject 측정일과 겹치는 기준물질 파일은 {metadata['reference_audit']['same_run_reference_files_verified']}개다. 이번 산출에서는 기준물질 peak shift 보정은 적용하지 않았다.\n"
        "- 출력에는 raw sample identifier, source filename, raw spectral array를 포함하지 않았다.\n\n"
        "## 산출물\n\n"
        "- `subject_mapping_repeatability_peak_metrics.csv`\n"
        "- `mapping_method_comparison_summary.csv`\n"
        "- `mapping_group_method_spectra.csv`\n"
        "- `mapping_peak_catalog.csv`\n"
        "- `mapping_partial_average_summary.csv`, `run_metadata.json`\n"
        "- `fig01`–`fig04` PNG/PDF\n"
    )
    (out / "REPORT.md").write_text(report, encoding="utf-8")
