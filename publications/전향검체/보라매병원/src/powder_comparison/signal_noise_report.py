from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .runner import AnalysisResult


def _ablation_table(result: AnalysisResult) -> str:
    rows = []
    for block in (result.signal_noise.screening_ablation, result.signal_noise.three_group_ablation):
        for row in block:
            rows.append(
                f"| {row.task} | {row.feature_set} | {np.median(row.feature_counts):.0f} | "
                f"{row.metrics.roc_auc:.3f} | {row.metrics.balanced_accuracy:.3f} |"
            )
    return "\n".join(
        (
            "| Task | Feature set | Median features | OOF AUROC | Balanced accuracy |",
            "|---|---|---:|---:|---:|",
            *rows,
        )
    )


def signal_noise_summary(result: AnalysisResult) -> str:
    summaries = result.signal_noise.effect_summaries
    variance = result.signal_noise.lot_variance
    technical = float(np.median(variance.lot_fraction + variance.residual_fraction))
    return (
        f"Cancer–Control 관련 liquid effect의 powder 방향 일치율은 "
        f"{summaries[0].sign_agreement:.1%}, 중앙 |effect| 보존율은 "
        f"{summaries[0].median_absolute_retention:.3f}였고, 5-lot 자료의 Raman shift별 "
        f"중앙 technical variance fraction은 {technical:.3f}이었다."
    )


def signal_noise_section(result: AnalysisResult) -> str:
    summaries = result.signal_noise.effect_summaries
    variance = result.signal_noise.lot_variance
    technical = float(np.median(variance.lot_fraction + variance.residual_fraction))
    stage_rows = result.signal_noise.repeatability.stage_rows
    liquid = [row for row in stage_rows if row.modality == "legacy_liquid" and row.stage == "snv"]
    powder = [row for row in stage_rows if row.modality == "powder" and row.stage == "snv"]
    full = next(
        row for row in result.signal_noise.screening_ablation if row.feature_set == "full_spectrum"
    )
    common = next(
        row for row in result.signal_noise.screening_ablation if row.feature_set == "common_stable"
    )
    powder_only = next(
        row for row in result.signal_noise.screening_ablation if row.feature_set == "powder_only"
    )
    excluded = next(
        row
        for row in result.signal_noise.screening_ablation
        if row.feature_set == "powder_only_excluded"
    )
    return f"""## 육안상 powder peak 증가는 반복 가능한 peak 증가로 확인되지 않았다

SNV 단계에서 환자별 apparent peak 중앙값은 liquid {np.median([row.median_peak_count for row in liquid]):.1f}개, powder {np.median([row.median_peak_count for row in powder]):.1f}개였다. 그러나 5회 중 4회 이상 같은 위치(±12 cm⁻¹)에 존재한 consensus fraction 중앙값은 liquid {np.median([row.consensus_fraction for row in liquid]):.3f}, powder {np.median([row.consensus_fraction for row in powder]):.3f}였다. 따라서 육안상 peak 수와 반복 가능한 peak 수를 분리해 해석해야 한다.

![Liquid와 powder의 반복 peak 진단](figures/fig19_powder_peak_reproducibility.png)

Fig19는 raw→smoothed→baseline-corrected→SNV 전 단계에서 동일 peak detector를 적용하고, 개별 replicate peak 수와 4-of-5 consensus를 함께 보여준다. Residual noise는 `1.4826 × MAD(spectrum − Savitzky–Golay smooth)`로 측정했다. Biofluid SERS에서 전통적 단일 SNR만으로 quality를 판단하지 않고 replicate correlation과 consensus를 주 판정축으로 사용했다.

## 기존 임상 subgroup signal은 powder에서 일부 약화되거나 방향이 바뀌었다

Cancer–Control의 liquid relevant feature(|Hedges' g|≥0.10) 중 powder에서도 effect 방향이 일치한 비율은 {summaries[0].sign_agreement:.1%}, 중앙 absolute effect retention은 {summaries[0].median_absolute_retention:.3f}였다. Biopsy-negative–Control은 {summaries[1].sign_agreement:.1%}/{summaries[1].median_absolute_retention:.3f}, Cancer–Biopsy-negative는 {summaries[2].sign_agreement:.1%}/{summaries[2].median_absolute_retention:.3f}였다.

![기존 subgroup effect와 LR peak의 powder 보존](figures/fig20_biological_signal_retention.png)

Fig20의 effect size는 LR 계수와 독립적으로 계산했다. 따라서 모델이 peak를 못 찾은 문제와 실제 subgroup contrast가 약해진 문제를 분리할 수 있다. 기존 LR peak별 중요도·effect 방향·반복 검출률은 `tables/powder_legacy_lr_peak_signal_retention.csv`에서 확인할 수 있다.

## 5-lot 자료는 powder의 patient·lot·spot 분산을 분리했다

동일 BPRO 5검체를 powder 5 lots에서 각각 5회 측정한 balanced crossed 자료에서 Raman shift별 중앙 technical fraction(lot+spot/residual)은 {technical:.3f}이었다. Patient×lot interaction은 별도 반복이 없어 residual에 포함되므로, 이 값은 임상적 일반화가 아니라 기판·spot 변동의 진단값이다.

![Powder 5-lot variance components](figures/fig21_powder_lot_variance_components.png)

Fig21은 각 Raman shift와 기존 LR 중요 peak에서 patient, lot, spot/residual 분산이 차지하는 비율을 보여준다. 정확한 shift별 값은 `tables/powder_lot_variance_components.csv`에 저장했다.

## Powder-only peak 제거만으로 분류 성능이 회복되는지 검증했다

{_ablation_table(result)}

![Powder-native LR peak ablation](figures/fig22_peak_ablation_performance.png)

Peak mask는 전체 자료에서 한 번 고정하지 않고 각 outer training fold의 replicate consensus만으로 다시 만들었다. 따라서 `common stable`, `powder-only`, `powder-only excluded` 비교에서 평가 환자의 label이나 peak presence가 feature selection에 유입되지 않는다. 모든 feature set의 confusion matrix는 `tables/powder_peak_ablation_confusion_matrices.csv`에 기록했다.

Binary AUROC는 full spectrum {full.metrics.roc_auc:.3f}, common-stable {common.metrics.roc_auc:.3f}, powder-only {powder_only.metrics.roc_auc:.3f}, powder-only excluded {excluded.metrics.roc_auc:.3f}였다. Powder-only 영역만으로도 일부 구분력이 있었고 이를 제거해도 full spectrum보다 개선되지 않았으므로, 신규 peak를 전부 무작위 noise로 보는 해석은 맞지 않는다. 반대로 common-stable 영역으로 제한했을 때 성능이 높아진 결과는 고차원 전체 spectrum에서 임상 signal이 희석되거나 regularization이 불안정해졌을 가능성을 제시한다. 이 ablation 간 차이는 탐색적 결과이며 별도 외부 검증 전에는 우월성을 확정하지 않는다.

## 현재 자료는 random-noise peak 증가보다 biological contrast 변화와 feature dilution을 더 지지한다

![Noise amplification과 biological signal loss의 통합 판정](figures/fig23_noise_vs_signal_conclusion.png)

Fig23은 powder peak의 반복성, subgroup effect, 환자별 replicate correlation과 native LR 오분류를 연결한다. 현재 powder에서 apparent peak 수가 늘지 않았고 consensus fraction은 더 높았으므로 `추가 peak가 반복되지 않는 technical noise`가 주원인이라는 증거는 약하다. 반면 subgroup effect 방향 보존이 낮고 common-stable feature 제한에서 성능이 상승해, 현재 자료의 주된 신호는 `기존 biological contrast의 약화·반전`과 `전체 spectrum에서의 feature dilution`이다. 다만 5-lot technical fraction {technical:.3f}에서 보듯 기술 변동도 무시할 수 없고, powder 임상 측정과 같은 run의 blank·4-ATP/reference가 없어 기판 자체 background나 non-specific enhancement는 확정할 수 없다. 최종 원인 확인은 [동일-day liquid/powder 검증 프로토콜](POWDER_SIGNAL_NOISE_VALIDATION_PROTOCOL.md)에 따라 진행한다.
"""
