from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from boramae_data import OUT
from prostate_comparison_model import OofResult
from prostate_shift_alignment import PeakSummary

REPORT = OUT / "PROSTATE_COMPARISON.md"


@dataclass(frozen=True, slots=True)
class CohortCounts:
    control: int
    psa_bx_negative: int
    prostate_cancer: int

    @property
    def total(self) -> int:
        return self.control + self.psa_bx_negative + self.prostate_cancer


@dataclass(frozen=True, slots=True)
class AnalysisResults:
    screening_binary: OofResult
    aligned_screening_binary: OofResult
    strict_binary: OofResult
    aligned_strict_binary: OofResult
    three_group: OofResult
    aligned_three_group: OofResult


@dataclass(frozen=True, slots=True)
class AlignmentResults:
    common_grid: PeakSummary
    urea_aligned: PeakSummary
    source_auc: float
    aligned_source_auc: float
    mean_spectrum_correlation: float
    clean_anchor_count: int
    prospective_anchor_count: int


def metric_ci(result: OofResult, name: str) -> str:
    low, high = result.intervals[name]
    return f"{result.metrics[name]:.3f} ({low:.3f}-{high:.3f})"


def count_cohort(labels: np.ndarray) -> CohortCounts:
    return CohortCounts(
        control=int(np.sum(labels == "Control")),
        psa_bx_negative=int(np.sum(labels == "Biopsy-negative")),
        prostate_cancer=int(np.sum(labels == "Prostate cancer")),
    )


def write_report(
    results: AnalysisResults,
    counts: CohortCounts,
    alignment: AlignmentResults,
    report: Path = REPORT,
) -> None:
    with report.open("w", encoding="utf-8") as handle:
        handle.write("# 보라매 전향검체 Screening 및 3군 성능\n\n")
        handle.write("## 분석 대상\n\n")
        handle.write(
            f"- 총 {counts.total}명: Control {counts.control}명, "
            f"Biopsy-negative (PSA 상승·전립선 생검 음성) {counts.psa_bx_negative}명, "
            f"전립선암 {counts.prostate_cancer}명\n"
        )
        handle.write("- `_ave` 제외, `_1.._5` replicate를 subject mean으로 집계\n")
        handle.write(
            "- 성능은 subject-level nested 5-fold OOF, 내부 4-fold에서 Logistic Regression `C` 선택\n"
        )
        handle.write("- 95% CI는 OOF 예측의 2,000회 bootstrap\n\n")
        handle.write("## 2군 성능\n\n")
        handle.write("| 정의 | N | ROC-AUC | Balanced accuracy | Sensitivity | Specificity |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        rows = (
            ("Screening: Control+Biopsy-negative vs Cancer", results.screening_binary),
            ("Screening + urea alignment", results.aligned_screening_binary),
            ("Strict: Control vs Cancer", results.strict_binary),
            ("Strict + urea alignment", results.aligned_strict_binary),
        )
        for label, result in rows:
            handle.write(
                f"| {label} | {len(result.y_true)} | {metric_ci(result, 'roc_auc')} | {metric_ci(result, 'balanced_accuracy')} | {metric_ci(result, 'sensitivity')} | {metric_ci(result, 'specificity')} |\n"
            )
        handle.write("\n- Screening은 Biopsy-negative 47명을 non-cancer에 포함한다.\n")
        handle.write("- Strict는 Biopsy-negative를 제외한 Control 21명 vs Cancer 41명 비교다.\n")
        handle.write(
            "- Figure: `figures/fig03_screening_binary_auc_confusion_matrix.png` / `.pdf`\n\n"
        )
        handle.write("## 3-group 성능\n\n")
        handle.write("| 정의 | N | Macro OVR ROC-AUC | Balanced accuracy | Macro F1 |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for label, result in (
            ("3-group: Control/Biopsy-negative/Cancer", results.three_group),
            ("3-group + urea alignment", results.aligned_three_group),
        ):
            handle.write(
                f"| {label} | {len(result.y_true)} | {metric_ci(result, 'macro_ovr_roc_auc')} | {metric_ci(result, 'balanced_accuracy')} | {metric_ci(result, 'macro_f1')} |\n"
            )
        handle.write(
            "\n- Figure: `figures/fig06_three_group_auc_confusion_matrix.png` / `.pdf`\n\n"
        )
        handle.write("## Train overfitting 처리\n\n")
        handle.write(
            "- 기존 single split은 train accuracy 1.000, test accuracy 0.697로 과적합 신호가 명확했다.\n"
        )
        handle.write(
            "- 새 주 결과는 outer fold에 한 번도 학습되지 않은 nested OOF 예측만 사용한다.\n\n"
        )
        handle.write("## Clean PRO와 전향 암군 차이 및 peak alignment\n\n")
        handle.write("- Clean retrospective PRO 91명(CBNUH) vs 보라매 전향 전립선암 41명\n")
        handle.write(
            f"- 평균 전처리 스펙트럼 Pearson r: {alignment.mean_spectrum_correlation:.3f}\n"
        )
        handle.write(
            f"- Cohort-source ROC-AUC: {alignment.source_auc:.3f}; urea alignment 후 {alignment.aligned_source_auc:.3f}\n"
        )
        handle.write(
            "- 높은 source AUC는 병원·채취시점·batch 차이이며 생물학적 차이로 해석하지 않는다.\n\n"
        )
        handle.write(
            "| 단계 | Clean peaks | Prospective peaks | Matched | ±5 cm⁻¹ | Median abs shift | Max abs shift |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for item in (alignment.common_grid, alignment.urea_aligned):
            handle.write(
                f"| {item.stage} | {item.clean_peaks} | {item.prospective_peaks} | {item.matched_peaks} | {item.matched_within_5} | {item.median_absolute_shift:.2f} | {item.max_absolute_shift:.2f} |\n"
            )
        handle.write(
            f"\n- Urea anchor: Clean {alignment.clean_anchor_count}/455, prospective cancer {alignment.prospective_anchor_count}/205 replicates\n"
        )
        handle.write(
            "- Figure: `figures/fig07_legacy_vs_prospective_peak_alignment.png` / `.pdf`\n\n"
        )
        handle.write("## 사용한 파일\n\n")
        handle.write("- `data/clinical_data/보라매 병원 임상정보.xlsx`\n")
        handle.write("- `results/clean_cohort_20260605/clean_cohort_manifest.csv`\n")
        handle.write("- `data/raw_data/1. Prostate cancer (100개)/PRO *_1..5.CSV`\n")
        handle.write("- `data/raw_data/20260709_BPRO,BNOR_1mW_0.05s_Ave100/*_1..5.CSV`\n")
        handle.write("- `artifacts/usersnet/v1.0.0/common_grid.npy`\n\n")
        handle.write("## 해석 주의\n\n")
        handle.write("- 결과는 exploratory nested OOF 성능이며 외부검증 성능이 아니다.\n")
        handle.write("- Cancer Screening AUC를 cross-hospital 일반화 근거로 사용하면 안 된다.\n")


def write_summary(
    results: AnalysisResults,
    counts: CohortCounts,
    alignment: AlignmentResults,
    summary: Path = OUT / "SUMMARY.md",
) -> None:
    with summary.open("w", encoding="utf-8") as handle:
        handle.write("# 보라매병원 전향검체 SERS 분석 산출물\n\n")
        handle.write(
            f"- 분석 포함 {counts.total}명: Control {counts.control}, Biopsy-negative {counts.psa_bx_negative}, Cancer {counts.prostate_cancer}\n"
        )
        handle.write(
            f"- Screening ROC-AUC {results.screening_binary.metrics['roc_auc']:.3f}; aligned {results.aligned_screening_binary.metrics['roc_auc']:.3f}\n"
        )
        handle.write(
            f"- Strict ROC-AUC {results.strict_binary.metrics['roc_auc']:.3f}; aligned {results.aligned_strict_binary.metrics['roc_auc']:.3f}\n"
        )
        handle.write(
            f"- 3-group macro ROC-AUC {results.three_group.metrics['macro_ovr_roc_auc']:.3f}; aligned {results.aligned_three_group.metrics['macro_ovr_roc_auc']:.3f}\n"
        )
        handle.write(
            f"- Peaks: clean {alignment.common_grid.clean_peaks}, prospective {alignment.common_grid.prospective_peaks}, matched {alignment.common_grid.matched_peaks}\n"
        )
        handle.write(f"- Mean-spectrum Pearson r {alignment.mean_spectrum_correlation:.3f}\n")
        handle.write(
            f"- Cohort-source ROC-AUC {alignment.source_auc:.3f}; aligned {alignment.aligned_source_auc:.3f}\n"
        )
        handle.write("- Train 1.000 single-split 결과는 과적합 확인용이며 주 결과는 nested OOF\n")
        handle.write("- 상세 결과와 95% CI: `PROSTATE_COMPARISON.md`\n")
