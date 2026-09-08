from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from boramae_repeat_average_core import GROUP_ORDER
from boramae_repeat_average_utils import stats


def build_metadata(
    subjects: list[dict[str, object]],
    excluded_rows: int,
    grid: np.ndarray,
    raw_axis: dict[str, object],
    audits: list[dict[str, object]],
) -> dict[str, object]:
    counts = [int(s["keep"].sum()) for s in subjects]
    return {
        "analysis_name": "Boramae prospective repeat-measurement average representative spectrum",
        "analysis_date": "2026-08-25",
        "included_subjects": len(subjects),
        "excluded_clinical_rows_with_null_group": excluded_rows,
        "group_counts": {g: sum(s["group"] == g for s in subjects) for g in GROUP_ORDER},
        "input_replicates": len(subjects) * 5,
        "instrument_average_files_audited": len(audits),
        "instrument_average_files_passing_normalized_rmse_le_1e-4": sum(
            bool(r["target_usable"]) for r in audits
        ),
        "common_grid": {
            "min_cm-1": float(grid[0]),
            "max_cm-1": float(grid[-1]),
            "points": len(grid),
            "step_cm-1": float(np.median(np.diff(grid))),
        },
        "raw_axis_observed": raw_axis,
        "alignment": {
            "method": "fixed common-grid linear interpolation",
            "per_spectrum_shift_applied": False,
            "same_run_external_reference_files_verified": 0,
        },
        "qc": {
            "method": "replicate-wise robust MAD distance",
            "mad_threshold": 3.5,
            "minimum_valid_replicates": 3,
            "counts": {
                "min": min(counts),
                "median": float(np.median(counts)),
                "max": max(counts),
            },
        },
        "preprocessing": {
            "baseline_correction": False,
            "intensity_scaling": False,
            "denoising": False,
            "relative_intensity_scale": True,
        },
        "aggregate": {
            "average_audit_normalized_rmse_median": stats([r["normalized_rmse"] for r in audits])[
                "median"
            ],
            "average_audit_correlation_median": stats([r["correlation"] for r in audits])["median"],
            "noise_floor_median": stats([s["noise_floor"] for s in subjects])["median"],
            "mean_replicate_correlation_median": stats(
                [s["mean_replicate_correlation"] for s in subjects]
            )["median"],
        },
        "privacy": "Raw sample identifiers, patient codes, source filenames, and spectral arrays are not exported.",
    }


def write_metadata(out: Path, metadata: dict[str, object]) -> None:
    (out / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_report(
    out: Path,
    subjects: list[dict[str, object]],
    excluded_rows: int,
    grid: np.ndarray,
    metadata: dict[str, object],
    audits: list[dict[str, object]],
) -> None:
    audit_rmse = stats([r["normalized_rmse"] for r in audits])
    audit_corr = stats([r["correlation"] for r in audits])
    qc = metadata["qc"]["counts"]
    groups = metadata["group_counts"]
    report = (
        "# 보라매 전향검체 적용 결과: 반복측정 평균 대표 스펙트럼 생성\n\n"
        "## 결론\n\n"
        f"- 임상표 120행 중 라벨이 있는 {len(subjects)}명을 분석했다. Control {groups['Control']}명, Biopsy-negative {groups['Biopsy-negative']}명, Prostate cancer {groups['Prostate cancer']}명이다. 라벨이 비어 있는 {excluded_rows}행은 제외했고 임상 라벨을 재매핑하지 않았다.\n"
        f"- 포함 검체의 `_1.._5` 반복 {len(subjects) * 5}개를 사용했고, 별도 `_ave` 파일 {len(audits)}개는 원시 5회 산술평균과 독립적으로 대조했다.\n"
        f"- 공통 grid는 {grid[0]:.1f}–{grid[-1]:.1f} cm⁻¹, {len(grid)} points, 간격 {np.median(np.diff(grid)):.4f} cm⁻¹이다. 원시 파수축은 1종으로 관찰되어 per-spectrum shift는 적용하지 않았다.\n"
        "- 같은 측정 런의 외부 기준물질 파일이 확인되지 않아 외부 기준물질 보정은 주장하지 않는다. 이번 적용은 “공통 grid 보간 → 반복 QC → 평균 대표 스펙트럼 → 잔차 통계” 범위다.\n"
        f"- QC 통과 반복 수는 subject 기준 최소 {qc['min']}, 중앙값 {qc['median']:.0f}, 최대 {qc['max']}회다. 평균 대표 스펙트럼은 QC 통과 반복의 산술평균이다.\n\n"
        "## 평균 파일 감사\n\n"
        f"- normalized RMSE ≤ 1e-4 기준 통과: {sum(bool(r['target_usable']) for r in audits)}/{len(audits)}개.\n"
        f"- normalized RMSE 중앙값 {audit_rmse['median']:.3e}, 상관계수 중앙값 {audit_corr['median']:.6f}.\n\n"
        "## 특허 초안 항목 대응\n\n"
        "- 파수축 정렬/보간: 공통 grid 선형 보간. 외부 기준 피크 이동 보정은 미적용.\n"
        "- 평균 대표 스펙트럼: 각 subject의 QC 통과 반복 평균.\n"
        "- 잔차/noise floor: 반복−subject 평균 잔차, 파수별 SD, 전체 grid median.\n"
        "- 공분산 구조: residual lag autocovariance/autocorrelation 및 반복 간 off-diagonal residual covariance/correlation.\n"
        "- 평균화 효과: 가능한 반복 subset 조합과 n별 1/n 분산 기준.\n"
        "- baseline correction, absolute intensity scaling, DWT denoising은 핵심 평균 생성 단계에 미적용.\n\n"
        "## 한계\n\n"
        "- 반복측정 재현성과 평균 생성의 기술 검증이며 임상 분류 성능 검증이 아니다.\n"
        "- 같은 2026-07-09 런의 기준물질이 없으므로 run-specific reference peak/FWHM 보정 결과로 해석하면 안 된다.\n"
        "- 출력에는 raw sample identifier, patient code, source filename, raw spectral array를 포함하지 않았다.\n\n"
        "## 산출물\n\n"
        "- `cohort_summary.csv`, `average_file_audit_deidentified.csv`\n"
        "- `subject_repeatability_metrics.csv`, `repeatability_metric_summary.csv`\n"
        "- `subject_average_representative_spectra.csv`\n"
        "- `group_mean_representative_spectra.csv`, `within_subject_noise_by_wavenumber.csv`\n"
        "- `residual_autocovariance_summary.csv`, `inter_replicate_residual_covariance_summary.csv`\n"
        "- `partial_average_summary.csv`, `run_metadata.json`\n"
        "- `fig01`–`fig04` PNG/PDF\n"
    )
    (out / "REPORT.md").write_text(report, encoding="utf-8")
