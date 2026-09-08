from __future__ import annotations

from pathlib import Path

import markdown
import numpy as np

from powder_comparison.delong import correlated_auc_test
from powder_comparison.evaluation import ComparisonBlock
from powder_comparison.group_separation_report import (
    group_separation_section,
    group_separation_summary,
)
from powder_comparison.preprocessing_report import preprocessing_section
from powder_comparison.report_output import html_document
from powder_comparison.runner import AnalysisResult
from powder_comparison.signal_noise_report import signal_noise_section, signal_noise_summary
from powder_comparison.statistics import MetricComparison, holm_adjust
from powder_comparison.threshold_analysis import threshold_strategies


def _metric(block: ComparisonBlock, name: str) -> MetricComparison:
    return next(row for row in block.metrics if row.name == name)


def _metric_text(block: ComparisonBlock, name: str) -> str:
    row = _metric(block, name)
    return (
        f"{row.legacy_value:.3f} → {row.powder_value:.3f} "
        f"(Δ {row.delta:+.3f}, 95% CI {row.ci_low:+.3f}–{row.ci_high:+.3f})"
    )


def _order_table(result: AnalysisResult) -> str:
    rows = [row for row in result.order_effects if row.scope == "BPRO_label_adjusted"]
    adjusted = holm_adjust(np.array([row.result.p_value for row in rows]))
    body = "\n".join(
        f"| {row.endpoint} | {row.result.n} | {row.result.rho:.3f} | "
        f"{row.result.p_value:.4f} | {holm_p:.4f} |"
        for row, holm_p in zip(rows, adjusted, strict=True)
    )
    return (
        "| 평가변수 | N | 층화 Spearman ρ | permutation p | Holm p |\n"
        "|---|---:|---:|---:|---:|\n"
        f"{body}"
    )


def _mcnemar_table(result: AnalysisResult) -> str:
    adjusted = holm_adjust(np.array([row.result.p_value for row in result.mcnemar]))
    body = "\n".join(
        f"| {row.scope} | {row.result.legacy_only} | {row.result.powder_only} | "
        f"{row.result.p_value:.4g} | {holm_p:.4g} |"
        for row, holm_p in zip(result.mcnemar, adjusted, strict=True)
    )
    return (
        "| 범위 | 기존만 정분류 | powder만 정분류 | exact p | Holm p |\n"
        "|---|---:|---:|---:|---:|\n"
        f"{body}"
    )


def _delong_table(result: AnalysisResult) -> str:
    rows = (
        (
            "Cancer Screening LR",
            correlated_auc_test(
                result.screening.y_true,
                result.screening.legacy_probabilities[:, 1],
                result.screening.transfer_probabilities[:, 1],
            ),
        ),
    )
    body = "\n".join(
        f"| {name} | {row.legacy_auc:.3f} | {row.powder_auc:.3f} | "
        f"{row.delta:+.3f} | {row.p_value:.4g} |"
        for name, row in rows
    )
    return (
        "| 분석 | 기존 AUROC | powder AUROC | 차이 | correlated DeLong p |\n"
        "|---|---:|---:|---:|---:|\n"
        f"{body}"
    )


def _comparison_table(result: AnalysisResult) -> str:
    screening_transfer, _, three_transfer, _ = result.comparisons
    return "\n".join(
        (
            "| 분석 | 지표 | 기존 액상 | 신규 powder | 차이 (95% CI) |",
            "|---|---|---:|---:|---:|",
            _row(screening_transfer, "Cancer Screening LR", "roc_auc"),
            _row(screening_transfer, "Cancer Screening LR", "sensitivity"),
            _row(screening_transfer, "Cancer Screening LR", "specificity"),
            _row(screening_transfer, "Cancer Screening LR", "balanced_accuracy"),
            _row(three_transfer, "3군 LR", "macro_ovr_roc_auc"),
            _row(three_transfer, "3군 LR - Control", "control_ovr_roc_auc"),
            _row(three_transfer, "3군 LR - Biopsy-negative", "biopsy_negative_ovr_roc_auc"),
            _row(three_transfer, "3군 LR - Cancer", "cancer_ovr_roc_auc"),
            _row(three_transfer, "3군 LR - Control", "control_sensitivity"),
            _row(three_transfer, "3군 LR - Control", "control_specificity"),
            _row(three_transfer, "3군 LR - Biopsy-negative", "biopsy_negative_sensitivity"),
            _row(three_transfer, "3군 LR - Biopsy-negative", "biopsy_negative_specificity"),
            _row(three_transfer, "3군 LR - Cancer", "cancer_sensitivity"),
            _row(three_transfer, "3군 LR - Cancer", "cancer_specificity"),
        )
    )


def _threshold_table(result: AnalysisResult) -> str:
    labels = ("기존 액상", "Powder transfer", "Powder native")
    thresholds = result.screening.binary_thresholds
    assert thresholds is not None
    means = (
        float(np.mean(thresholds.legacy_thresholds)),
        float(np.mean(thresholds.legacy_thresholds)),
        float(np.mean(thresholds.native_thresholds)),
    )
    rows = []
    for label, mean, block in zip(labels, means, result.threshold_comparisons, strict=True):
        sensitivity = _metric(block, "sensitivity")
        specificity = _metric(block, "specificity")
        balanced = _metric(block, "balanced_accuracy")
        rows.append(
            f"| {label} | {mean:.3f} | {sensitivity.legacy_value:.3f} → "
            f"{sensitivity.powder_value:.3f} | {specificity.legacy_value:.3f} → "
            f"{specificity.powder_value:.3f} | {balanced.legacy_value:.3f} → "
            f"{balanced.powder_value:.3f} |"
        )
    return "\n".join(
        (
            "| Modality | 평균 Youden threshold | 민감도 | 특이도 | Balanced accuracy |",
            "|---|---:|---:|---:|---:|",
            *rows,
        )
    )


def _psa_error_table(result: AnalysisResult) -> str:
    strategies = threshold_strategies(result)
    selected_strategies = (strategies[0], strategies[1], strategies[2], strategies[3])
    groups = result.spectra.clinical_groups
    rows = []
    for strategy in selected_strategies:
        label = strategy.name.replace("_", " ")
        for band in ("<4", "4-<10", ">=10"):
            selected = (groups == "Biopsy-negative") & (result.clinical.psa_bands == band)
            count = int(np.sum(selected))
            errors = int(np.sum(strategy.predictions[selected] == 1))
            rows.append(f"| {label} | {band} | {count} | {errors} | {errors / count:.1%} |")
    return "\n".join(
        (
            "| Strategy | PSA 구간 | Biopsy-negative n | Cancer 오분류 | 오분류율 |",
            "|---|---:|---:|---:|---:|",
            *rows,
        )
    )


def _row(block: ComparisonBlock, label: str, metric_name: str) -> str:
    metric = _metric(block, metric_name)
    return (
        f"| {label} | {metric_name} | {metric.legacy_value:.3f} | {metric.powder_value:.3f} | "
        f"{metric.delta:+.3f} ({metric.ci_low:+.3f}–{metric.ci_high:+.3f}) |"
    )


def build_markdown(result: AnalysisResult) -> str:
    screening_transfer, screening_native, three_transfer, three_native = result.comparisons
    reclassification = screening_transfer.reclassification
    primary_order = [row for row in result.order_effects if row.scope == "BPRO_label_adjusted"]
    order_adjusted = holm_adjust(np.array([row.result.p_value for row in primary_order]))
    order_by_endpoint = {row.endpoint: row for row in primary_order}
    order_holm_by_endpoint = {
        row.endpoint: adjusted for row, adjusted in zip(primary_order, order_adjusted, strict=True)
    }
    significant_order = [
        row for row, adjusted in zip(primary_order, order_adjusted, strict=True) if adjusted < 0.05
    ]
    order_statement = (
        "BPRO 내부에서 임상라벨을 층화한 순서효과 검정에서 유의한 평가변수가 관찰되었다."
        if significant_order
        else "BPRO 내부 임상라벨 층화 검정에서는 유의한 단조 순서효과가 확인되지 않았다."
    )
    strategies = threshold_strategies(result)
    legacy_false_positive = (result.screening.y_true == 0) & (strategies[0].predictions == 1)
    legacy_biopsy_false_positive = int(
        np.sum(legacy_false_positive & (result.spectra.clinical_groups == "Biopsy-negative"))
    )
    transfer_threshold = result.threshold_comparisons[1]
    optimized = result.screening.binary_thresholds
    assert optimized is not None
    return f"""# 기존 액상 SERS와 신규 powder SERS의 LR paired 비교 분석

## 기술 요약

- 분석 대상은 동일 보라매 환자 {len(result.pairs)}명(Control 21, Biopsy-negative 47, Cancer 41)이다.
- 기존 분석과 동일한 LR nested 5-fold OOF 설정에서 Cancer Screening AUROC는 {_metric_text(screening_transfer, "roc_auc")}였다.
- 동일 액상 학습 LR을 powder에 적용했을 때 재분류는 개선 {reclassification.improved}명, 악화 {reclassification.worsened}명, 순개선율 {reclassification.net_improvement_rate:+.1%}였다.
- 3군 macro OVR AUROC는 {_metric_text(three_transfer, "macro_ovr_roc_auc")}였다.
- {group_separation_summary(result)}
- {signal_noise_summary(result)}
- 액상 training fold에서 최적화한 Youden threshold를 powder에 적용해도 balanced accuracy는 {_metric_text(transfer_threshold, "balanced_accuracy")}로 제한적으로만 변했다.
- 측정순서 평가 결과: {order_statement}
- 사전 정의된 의사결정 규칙에 따른 결론은 **{result.decision}**이다.

## 동일 모델에서 관찰된 성능 변화

{_comparison_table(result)}

![액상 학습 LR의 powder transfer 평가](figures/fig07a_powder_transfer_summary.png)

모든 모델은 StandardScaler와 class-weighted Logistic Regression으로 통일했다. 각 outer fold에서 액상 training 환자로만 `C`를 선택하고 동일 fold의 액상·powder를 함께 예측했다. 기존 0.714 결과를 재현하도록 outer seed 42와 inner seed 101–105를 고정했다.

![기존 액상과 신규 powder의 paired 확률 및 ROC](figures/fig08_powder_paired_probability_roc.png)

## Threshold 최적화는 powder 전이 성능을 복구하지 못했다

Threshold는 각 outer fold의 평가 환자를 제외한 training 환자의 inner-CV 예측값에서 Youden J(민감도+특이도-1)를 최대화하도록 선택했다. AUROC와 Brier score는 threshold와 무관하므로 변하지 않는다.

{_threshold_table(result)}

액상 training 기준 fold threshold는 {float(np.min(optimized.legacy_thresholds)):.3f}–{float(np.max(optimized.legacy_thresholds)):.3f}로 폭이 넓어 안정적이지 않았다. Powder transfer에서 민감도는 {_metric(transfer_threshold, "sensitivity").legacy_value:.3f}→{_metric(transfer_threshold, "sensitivity").powder_value:.3f}로 소폭 증가했지만, 특이도는 {_metric(transfer_threshold, "specificity").legacy_value:.3f}→{_metric(transfer_threshold, "specificity").powder_value:.3f}로 감소했다. 따라서 현재 성능 저하는 0.5 threshold 하나의 문제로 설명되지 않는다.

![Threshold별 confusion matrix](figures/fig11_lr_threshold_confusion_matrices.png)

![Threshold 성능·오분류·PSA 진단](figures/fig12_lr_threshold_error_diagnostics.png)

![환자별 예측확률과 threshold trace](figures/fig13_lr_threshold_patient_trace.png)

### Biopsy-negative와 PSA 구간별 오분류

기존 액상 LR의 false positive {int(np.sum(legacy_false_positive))}명 중 {legacy_biopsy_false_positive}명이 Biopsy-negative였다. 즉 기존 분석에서도 Cancer와 가장 헷갈리는 non-cancer 군은 Biopsy-negative였다.

{_psa_error_table(result)}

PSA는 Biopsy-negative 47명에서만 모두 확인되고 Cancer 41명은 전원 결측이다. 따라서 현재 PSA 구간 분석은 Biopsy-negative 내 Cancer false positive 구성만 확인할 수 있으며, Cancer false negative와 PSA의 관계는 평가할 수 없다. PSA `<4` 구간은 2명에 불과하므로 구간별 차이를 일반화하지 않는다. 환자별 확률·fold threshold·고정/최적화 판정·판정 변경 방향은 `tables/lr_screening_paired_predictions.csv`에서 추적할 수 있다.

## Powder 재학습 LR을 기존과 동일한 figure로 평가했다

Powder-native LR의 Cancer Screening AUROC는 {_metric(screening_native, "roc_auc").powder_value:.3f}, balanced accuracy는 {_metric(screening_native, "balanced_accuracy").powder_value:.3f}, 민감도는 {_metric(screening_native, "sensitivity").powder_value:.3f}, 특이도는 {_metric(screening_native, "specificity").powder_value:.3f}이다. 3군 macro OVR AUROC는 {_metric(three_native, "macro_ovr_roc_auc").powder_value:.3f}, balanced accuracy는 {_metric(three_native, "balanced_accuracy").powder_value:.3f}, macro F1은 {_metric(three_native, "macro_f1").powder_value:.3f}이다. 두 figure는 기존 액상 fig03·fig06과 동일하게 confusion matrix, ROC, nested OOF metric panel로 구성했다.

![액상·powder transfer·powder-native LR 전체 비교](figures/fig07b_powder_overall_analysis_summary.png)

![Powder-native Cancer Screening ROC·confusion matrix](figures/fig14_powder_native_screening_auc_confusion_matrix.png)

![Powder-native 3분류 ROC·confusion matrix](figures/fig15_powder_native_three_group_auc_confusion_matrix.png)

환자 추적 CSV와 fig13의 식별자는 임의로 생성한 `BRM-PWD` 순번 대신 원 임상표의 `patient_code`를 사용했다. `BPRO n` 또는 `BNOR n`은 원 스펙트럼으로 역추적할 수 있게 별도 `sample_id` 열로 남겼다.

## 측정순서 효과는 별도 검정했다

Powder 측정순서는 기록된 규칙에 따라 BNOR 번호순 후 BPRO 번호순으로 재구성했다. 전체 BNOR-BPRO 비교는 prefix·임상군·순서가 분리되지 않으므로 인과적 순서효과 검정으로 사용하지 않았다. 주 검정은 BPRO 내부에서 clinical group별 rank를 제거한 층화 Spearman ρ와 9,999회 label-stratified permutation으로 수행했다. LR 전이 예측값과 절대확률잔차 `|실제값-예측확률|`를 포함한 6개 주 평가변수에 Holm 보정을 적용했다.

{_order_table(result)}

LR 전이 예측확률의 순서 연관은 ρ={order_by_endpoint["lr_transfer_probability"].result.rho:.3f}, Holm p={order_holm_by_endpoint["lr_transfer_probability"]:.4f}였고, 절대확률잔차는 ρ={order_by_endpoint["lr_transfer_absolute_residual"].result.rho:.3f}, Holm p={order_holm_by_endpoint["lr_transfer_absolute_residual"]:.4f}였다. 예측확률 이동과 분류오차·스펙트럼/QC drift를 함께 확인해 순서효과를 판단한다.

![측정순서 진단](figures/fig09a_powder_measurement_order_diagnostics.png)

![BPRO 내부 순서효과](figures/fig09b_powder_measurement_order_effect.png)

초기·중기·후기 구간의 오분류율, spectral PC1, raw intensity와 replicate correlation을 함께 비교했다. 따라서 예측확률만 이동한 경우와 스펙트럼/QC drift가 동반된 경우를 구분할 수 있다.

## 판정 일치도와 재분류

- Cancer Screening LR exact agreement: {screening_transfer.agreement.exact_agreement:.3f}
- Cancer Screening LR Cohen's κ: {screening_transfer.agreement.kappa:.3f} (95% CI {screening_transfer.agreement.ci_low:.3f}–{screening_transfer.agreement.ci_high:.3f})
- 3군 LR Cohen's κ: {three_transfer.agreement.kappa:.3f}
- Powder-native 탐색모델의 binary AUROC 변화: {_metric_text(screening_native, "roc_auc")}
- Powder-native 탐색모델의 3군 macro OVR AUROC 변화: {_metric_text(three_native, "macro_ovr_roc_auc")}

환자별 개선·악화 목록과 확률 변화는 `tables/lr_screening_paired_predictions.csv`에, 재분류 환자의 Top-5 Raman shift 변화는 `tables/reclassified_patient_peaks.csv`에 기록했다.

### 민감도·특이도 변화의 paired 검정

{_mcnemar_table(result)}

### AUROC 변화의 correlated DeLong 보조검정

{_delong_table(result)}

## Important peak 비교

- Top-5 Jaccard overlap: {result.peaks.agreement.jaccard:.3f}
- 전체 peak 중요도 Spearman ρ: {result.peaks.agreement.spearman_rho:.3f}
- 공통 Top peak: {", ".join(result.peaks.agreement.shared_top_peaks) or "없음"}

![LR 중요 Raman shift 비교](figures/fig10_powder_important_peak_comparison.png)

Important peak는 Cancer Screening LR의 표준화 계수 절댓값에서 18 cm^-1 이상 떨어진 국소 극값을 추출하고 모델별 합계가 1이 되도록 상대 중요도를 정규화했다. 전체 환자로 적합한 설명용 LR의 계수이므로 OOF 성능 추정치와 구분해 해석하며, 재분류 환자는 동일 액상 학습 LR의 액상·powder local contribution을 비교했다.

{preprocessing_section(result)}

## 임상 subgroup 스펙트럼 육안 비교

![Cancer·Biopsy-negative·Control의 최종 SNV 스펙트럼](figures/fig17_clinical_group_spectra_comparison.png)

위 Figure는 동일한 최종 전처리 결과에서 Control, Biopsy-negative, Cancer의 중앙 스펙트럼과 IQR을 비교한다. A·B는 기존 액상과 신규 powder 각각에서 subgroup 간 형태를 보여주고, C는 동일 환자의 `powder − 액상` 차이를 subgroup별로 표시한다. D–F는 각 subgroup 안에서 액상과 powder를 직접 겹쳐 표시한다. 모든 스펙트럼 panel의 y축 범위를 동일하게 고정했으므로 subgroup 차이와 측정조건 차이의 상대적 크기를 육안으로 비교할 수 있다. 다만 곡선 중첩만으로 분류 가능성이나 개별 peak의 통계적 유의성을 확정하지 않는다.

{group_separation_section(result)}

{signal_noise_section(result)}

## 데이터와 분석 방법

- 기존 액상: `data/raw_data/20260709_BPRO,BNOR_1mW_0.05s_Ave100`
- 신규 powder: `data/20260716_Urine test (Powder_BNOR, BPRO)`
- Powder 5-lot 재현성: `data/20260715_Powder_Reproducibility test`
- Reference label: `data/clinical_data/보라매 병원 임상정보.xlsx`
- `_1.._5` replicate를 전처리 후 환자 평균으로 집계했고 `_ave` 및 Zone.Identifier는 제외했다.
- AUROC와 성능 차이의 95% CI는 임상군 층화 subject bootstrap 10,000회로 계산하고, binary AUROC는 correlated DeLong 검정을 보조적으로 제시했다.
- 민감도·특이도 변화는 동일 환자의 discordant pair를 대상으로 exact McNemar 검정을 사용했다.
- 3군 평가는 Control/Biopsy-negative/Cancer의 macro one-vs-rest AUROC를 사용했다.

## 제한점과 불확실성

1. 측정순서가 무작위화되지 않았고 BNOR가 BPRO보다 먼저 측정되어, prefix 간 차이는 순서·임상군·검체 처리 효과로 분해할 수 없다.
2. Paired 임상 비교는 1개 powder production sequence이며, 별도 5-lot 자료는 동일 BPRO 5검체에 한정된다. 정식 키트 성능시험이나 임상적 비열등성 검증이 아니다.
3. LR는 동일 보라매 코호트 내부 nested OOF 결과이므로 외부검증 또는 cross-hospital 일반화 근거로 해석하지 않는다.
4. Powder-native baseline은 신규 모델 개발 필요성을 탐색하기 위한 단순 모델이며 운영 배포 후보가 아니다.
5. Youden threshold는 fold별 변동이 크고 현 코호트의 탐색적 추정치이므로 운영 threshold로 고정할 수 없다.
6. Cancer의 PSA가 모두 결측이므로 PSA 구간별 분석은 Biopsy-negative 내 false positive 기술에 한정된다.
7. PPV와 NPV는 본 연구 코호트의 인위적 유병률에 의존하므로 주 결과에서 제외했다.
8. Powder 임상 측정과 같은 run의 blank·4-ATP·instrument reference가 없어 기판 background와 sample–powder interaction을 최종 분리할 수 없다.

## 권고되는 다음 단계

1. `POWDER_SIGNAL_NOISE_VALIDATION_PROTOCOL.md`에 따라 liquid/powder를 같은 날 block randomization하여 blank·4-ATP·artificial urine·3개 pooled clinical QC를 측정한다.
2. 본 분석의 성능 저하 유형에 따라 고정 모델 유지, threshold/전처리 전이 보정 또는 powder 전용 모델 개발 중 하나를 선택한다.
3. 재분류 악화 환자는 원시 스펙트럼, QC, local peak contribution을 함께 검토하고 재측정 우선순위를 정한다.

## 추가 확인 질문

- 동일 결과가 다른 powder lot와 다른 측정일에서도 재현되는가?
- 순서효과가 유의한 경우 blank/reference 신호 또는 실험 시간대 변화와 연결되는가?
- Powder-native 모델의 개선이 새 생물학적 정보인지 assay-specific shift인지 독립 lot에서 구분 가능한가?
"""


def write_report(result: AnalysisResult) -> tuple[Path, Path]:
    result.sources.output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = result.sources.output_dir / "POWDER_COMPARISON.md"
    html_path = result.sources.output_dir / "powder_comparison.html"
    text = build_markdown(result)
    markdown_path.write_text(text, encoding="utf-8")
    body = markdown.markdown(text, extensions=("tables", "fenced_code"))
    html_path.write_text(html_document(body), encoding="utf-8")
    return markdown_path, html_path
