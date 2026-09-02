from __future__ import annotations

from powder_comparison.runner import AnalysisResult


def _stage_table(result: AnalysisResult) -> str:
    body = "\n".join(
        f"| {row.stage} | {row.median_correlation:.3f} "
        f"({row.correlation_q1:.3f}–{row.correlation_q3:.3f}) | "
        f"{row.median_nrmse:.3f} | {row.median_log2_rms_ratio:+.3f} |"
        for row in result.preprocessing.rows
    )
    return (
        "| 단계 | Paired Pearson r, median (IQR) | Median NRMSE | "
        "Median log2 powder/액상 RMS |\n"
        "|---|---:|---:|---:|\n"
        f"{body}"
    )


def preprocessing_section(result: AnalysisResult) -> str:
    diagnostics = result.preprocessing
    raw = diagnostics.rows[0]
    baseline = diagnostics.rows[2]
    raw_rms_ratio = 2**raw.median_log2_rms_ratio
    return f"""## Preprocessing domain-shift 역추적

{_stage_table(result)}

![전처리 단계별 domain-shift와 LR 예측 이동](figures/fig16_preprocessing_domain_shift_diagnostics.png)

각 replicate를 `Raw → Savitzky–Golay(11, 3) → rolling-minimum baseline(101) → SNV → model grid` 순서로 처리한 뒤 5회 평균했다. 최종 SNV paired RMSE와 동일 액상 학습 LR의 암 예측확률 절대변화 간 Spearman ρ는 {diagnostics.distance_probability_rho:.3f}(p={diagnostics.distance_probability_p_value:.4g})이다. Legacy LR Top-5 peak ±9 cm^-1 구간의 최종 SNV 변화는 그 밖의 구간 대비 {diagnostics.peak_shift_enrichment:.2f}배다.

이 Figure는 차이가 처음 나타나거나 증폭되는 전처리 단계, 최종 차이와 LR 예측 이동의 연관, legacy LR 중요 peak와의 공간적 중첩을 함께 진단한다. 이는 원인 후보를 좁히는 paired 진단이며, 전처리 자체의 인과효과나 peak의 생물학적 타당성을 단독으로 확정하지 않는다.

현재 데이터에서 powder의 raw RMS는 액상의 {raw_rms_ratio:.2f}배였다. SNV가 RMS scale은 거의 1:1로 맞췄지만 paired 스펙트럼 상관은 {raw.median_correlation:.3f}에서 baseline correction 후 {baseline.median_correlation:.3f}로 낮아진 뒤 회복되지 않았다. 따라서 단순 intensity scale 차이 외에 형태 domain shift가 남아 있다. 그러나 전처리 거리와 예측확률 이동이 연관되지 않고 Top-5 peak 구간에 차이가 농축되지도 않아, 전처리 전체 거리 또는 일부 peak만으로 LR 성능 저하를 설명할 수는 없다. Fig. 10의 Top-5 공통 peak가 없고 중요도 순위 상관도 음수인 결과를 함께 보면, powder domain에서 LR 계수·peak 선택이 불안정하게 바뀌는 문제와 전처리 후에도 남는 형태 shift를 둘 다 추가 분리해야 한다."""
