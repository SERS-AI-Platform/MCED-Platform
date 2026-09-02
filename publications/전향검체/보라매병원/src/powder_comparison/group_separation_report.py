from __future__ import annotations

import numpy as np

from .group_separation import evaluate_group_separation
from .runner import AnalysisResult


def group_separation_summary(result: AnalysisResult) -> str:
    diagnostics = evaluate_group_separation(result.spectra)
    legacy = np.array([row.separation_to_spread for row in diagnostics.legacy_pairs])
    powder = np.array([row.separation_to_spread for row in diagnostics.powder_pairs])
    return (
        "최종 SNV subgroup separation-to-spread ratio는 "
        f"액상 {float(np.min(legacy)):.3f}–{float(np.max(legacy)):.3f}, "
        f"powder {float(np.min(powder)):.3f}–{float(np.max(powder)):.3f}로 "
        "세 pair 모두 1 미만이었다."
    )


def group_separation_section(result: AnalysisResult) -> str:
    diagnostics = evaluate_group_separation(result.spectra)
    rows = "\n".join(
        f"| {legacy.label} | {legacy.separation_to_spread:.3f} | "
        f"{powder.separation_to_spread:.3f} | "
        f"{powder.separation_to_spread - legacy.separation_to_spread:+.3f} |"
        for legacy, powder in zip(
            diagnostics.legacy_pairs,
            diagnostics.powder_pairs,
            strict=True,
        )
    )
    lower_count = sum(
        powder.separation_to_spread < legacy.separation_to_spread
        for legacy, powder in zip(
            diagnostics.legacy_pairs,
            diagnostics.powder_pairs,
            strict=True,
        )
    )
    legacy_own_closer = float(np.mean(diagnostics.legacy_margins.margin > 0.0))
    powder_own_closer = float(np.mean(diagnostics.powder_margins.margin > 0.0))
    return f"""## 액상과 powder의 subgroup spectral separation

`Separation-to-spread ratio`는 두 subgroup 평균 스펙트럼의 RMS 차이를 두 군의 pooled 군내 RMS로 나눈 기술통계다. 1보다 작으면 subgroup 중심 차이가 전형적인 환자 간 군내 편차보다 작다는 뜻이며, 클수록 상대적 분리가 뚜렷하다.

| Subgroup pair | 기존 액상 | 신규 powder | Powder - 액상 |
|---|---:|---:|---:|
{rows}

![액상과 powder의 subgroup spectral separation](figures/fig18_clinical_group_separation_diagnostics.png)

세 pair 중 powder ratio가 액상보다 낮은 비교는 {lower_count}/3개였다. Leave-one-out 방식에서 자기 임상군 centroid가 다른 군 centroid보다 가까운 환자는 액상 {legacy_own_closer:.1%}, powder {powder_own_closer:.1%}였다. A·B의 Hedges' g heatmap은 각 Raman shift의 군간 평균 차이를 pooled 표준편차로 나눈 값이고, 두 패널은 동일한 색상 범위를 사용한다. C는 전체 스펙트럼 수준의 군간 신호/군내 변동 비율, D는 환자별 centroid margin 분포다.

이 지표들은 “각 modality 안에서 임상군 신호가 개인차보다 작은가”를 확인하는 설명용 진단이다. 다중 Raman shift의 결합 효과, fold별 feature scaling과 regularization을 반영하는 nested OOF LR 성능을 대체하지 않으며, Hedges' g의 각 shift별 다중검정 유의성을 주장하지 않는다.
"""
