from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from mapping_preprocessing_compare_core import PreprocessingComparison
from mapping_preprocessing_compare_outputs import ComparisonTables


def _metric(tables: ComparisonTables, group: str, metric: str) -> float:
    values = [
        float(row["median"])
        for row in tables.summary_rows
        if row["group"] == group and row["metric"] == metric
    ]
    return values[0]


def metadata(
    results: list[PreprocessingComparison],
    grid: np.ndarray,
    raw_axis: dict[str, float | int | list[int]],
    groups: tuple[str, ...],
) -> dict[str, object]:
    return {
        "analysis_name": "Per-subject previous versus changed preprocessing comparison",
        "analysis_date": "2026-08-25",
        "included_subjects": len(results),
        "group_counts": {group: sum(r.subject.group == group for r in results) for group in groups},
        "raw_repeats": sum(len(r.subject.replicate_paths) for r in results),
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
        "comparison_design": "Both methods use the same raw-QC-passed repeat indices; only preprocessing differs.",
        "previous_production_preprocessing": {
            "calibration": "urea reference 1001.4 cm-1, ±10 cm-1 search",
            "trim": "400–2200 cm-1",
            "smoothing": "Savitzky–Golay, window 11, polynomial 3",
            "baseline": "rolling minimum, window 101",
            "normalization": "SNV",
            "resampling": "fixed common grid",
        },
        "changed_preprocessing": {
            "calibration": False,
            "smoothing": False,
            "baseline_correction": False,
            "intensity_scaling": False,
            "steps": "fixed common-grid interpolation → raw repeat QC → arithmetic average",
        },
        "privacy": "No raw filenames, sample identifiers, or spectral arrays are exported.",
    }


def write_report(
    out: Path,
    results: list[PreprocessingComparison],
    tables: ComparisonTables,
    metadata_value: dict[str, object],
    groups: tuple[str, ...],
) -> None:
    counts = metadata_value["group_counts"]
    report = (
        "# 검체별 기존 전처리와 변경 전처리 비교\n\n"
        "## 분석 설계\n\n"
        f"- 포함 검체 {len(results)}명: Control {counts['Control']}명, Prostate disease control {counts['Prostate disease control']}명, Prostate cancer {counts['Prostate cancer']}명.\n"
        f"- 검체별 원시 반복 121개, 총 {metadata_value['raw_repeats']}개. 동일한 raw-QC 통과 반복 집합을 두 방법에 공통으로 적용했고, QC 통과 반복은 총 {metadata_value['qc_passed_repeats']}개였다.\n"
        "- 변경 방식: common grid 보간 → raw 반복 QC → QC 통과 반복 산술평균. baseline, smoothing, SNV, absolute intensity scaling은 평균 생성 단계에 적용하지 않았다.\n"
        "- 기존 전처리: 1001.4 cm⁻¹ calibration → 400–2200 cm⁻¹ trim → Savitzky–Golay(11, 3) → rolling-min baseline(101) → SNV → common grid.\n\n"
        "## 전체 검체 중앙값\n\n"
        f"- 평균 스펙트럼 high-frequency noise: 변경 raw 평균 { _metric(tables, 'All included', 'changed_average_hf_noise'):.4f}, 기존 전처리 { _metric(tables, 'All included', 'previous_average_hf_noise'):.4f}. 두 방법은 scale이 달라 절대값을 직접 noise 우열로 해석하지 않는다.\n"
        f"- 재현 가능한 peak 수: 변경 { _metric(tables, 'All included', 'changed_reproducible_peak_count'):.1f}, 기존 전처리 { _metric(tables, 'All included', 'previous_reproducible_peak_count'):.1f}.\n"
        f"- native 출력의 shape correlation: 변경 raw 평균 vs 기존 전처리 평균 {_metric(tables, 'All included', 'changed_previous_shape_correlation'):.4f}.\n"
        f"- 변경 raw 평균에도 기존 전처리 변환을 동일하게 적용한 뒤의 shape correlation: {_metric(tables, 'All included', 'changed_after_previous_transform_shape_correlation'):.4f}.\n"
        f"- calibration shift 중앙값: {_metric(tables, 'All included', 'calibration_shift_median_cm-1'):+.4f} cm⁻¹; 절대 shift 최대값 중앙값: {_metric(tables, 'All included', 'calibration_shift_abs_max_cm-1'):.4f} cm⁻¹.\n\n"
        "## 해석\n\n"
        "- 이번 비교의 목적은 QC 정책 차이가 아니라 전처리 차이를 분리하는 것이다. 따라서 동일한 raw-QC 반복을 기존 전처리와 변경 방식에 각각 통과시켰다.\n"
        "- native 출력 상관은 낮지만, 변경 평균에 기존의 calibration·trim·SG·rolling-min·SNV 변환을 사후 적용하면 기존 전처리 평균과 거의 일치한다. 즉, 현재 차이의 주된 원인은 평균 자체보다 baseline·normalization을 적용한 위치와 유무다.\n"
        "- 기존 방식은 baseline 제거와 SNV 때문에 peak shape를 비교하기에는 유리하지만, absolute intensity와 baseline 정보는 보존하지 않는다. 변경 방식은 원래 상대 intensity 스케일을 보존하므로 후속 peak/평균 분석에서 baseline 처리를 별도 단계로 결정해야 한다.\n"
        "- 평균 스펙트럼의 high-frequency noise 절대값은 SNV 적용 여부에 따라 scale이 달라지므로, noise 저감률만으로 두 전처리 중 하나를 선택하지 않고 peak 재현성·shape correlation·임상 목적을 함께 봐야 한다.\n\n"
        "## 산출물\n\n"
        "- `subject_preprocessing_comparison.csv`: 검체별 QC 수, calibration, noise, peak, shape 지표\n"
        "- `preprocessing_comparison_summary.csv`: 그룹별 및 전체 요약\n"
        "- `preprocessing_group_spectra.csv`: 그룹별 평균 스펙트럼\n"
        "- `fig01`–`fig03` PNG/PDF\n"
    )
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    (out / "run_metadata.json").write_text(
        json.dumps(metadata_value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
