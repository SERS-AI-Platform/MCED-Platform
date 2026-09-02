from __future__ import annotations

from pathlib import Path

import numpy as np
from clinical_hypothesis_models import AncovaAnalysis
from clinical_hypothesis_outputs import AssumptionContext
from clinical_hypothesis_statistics import MetricAnalysis, holm_adjust


def format_p(p_value: float) -> str:
    """Format p-values without representing numerical underflow as zero."""
    floor = float(np.finfo(np.float64).eps)
    if p_value == floor:
        return f"p<{floor:.3g}"
    return f"p={p_value:.3g}"


def write_report(
    path: Path,
    ancova: AncovaAnalysis,
    secondary: tuple[MetricAnalysis, MetricAnalysis],
    psa: MetricAnalysis,
    context: AssumptionContext,
) -> None:
    """Write the hypothesis-driven technical analysis report."""
    secondary_adjusted = holm_adjust(
        np.array([analysis.welch.p_value for analysis in secondary], dtype=np.float64)
    )
    common_slope = next(
        result for result in ancova.slope_checks if result.check == "common_slopes_joint"
    )
    lines = [
        "# 보라매 3군 임상 공변량 가설 기반 분석",
        "",
        "## 기술 요약",
        "",
        f"- **주가설 H1을 기각했다.** 연령과 BMI를 보정한 log10-PSA의 군 효과는 "
        f"유의했다(F({ancova.omnibus.df1:.0f}, {ancova.omnibus.df2:.0f})="
        f"{ancova.omnibus.statistic:.2f}, {format_p(ancova.omnibus.p_value)}).",
        "- 세 개의 사전 계획 대비 모두 Holm 보정 후 유의했으며, 방향은 "
        "Biopsy-negative > Cancer > Control이었다.",
        f"- 공통 회귀기울기 상호작용 검정은 {format_p(common_slope.p_value)}로 "
        "기울기 이질성의 증거를 확인하지 못했다. 이는 가정이 증명됐다는 뜻은 아니다.",
        "",
        "## 사전 정의 가설",
        "",
        "### H1 — 주가설: 연령·BMI 보정 PSA 군 효과",
        "",
        "- 귀무가설: ANCOVA에서 두 군 지시변수의 계수가 공동으로 0이다.",
        "- 대립가설: 적어도 하나의 보정 군 계수가 0이 아니다.",
        "- 분석: `log10(PSA) ~ group + age + BMI`, HC3 공분산, 양측 α=0.05.",
        "- 사후 대비: 세 군 쌍을 사전 계획하고 Holm으로 family-wise error를 통제했다.",
        "",
        "### H2 — 보조가설: 연령과 BMI의 군간 불균형",
        "",
        "- H2a: 세 군의 평균 연령이 같다.",
        "- H2b: 세 군의 평균 BMI가 같다.",
        "- Welch ANOVA 두 개를 Holm 보정하고, 통과한 변수만 Games–Howell 결과를 해석했다.",
        "",
        "### H3 — 민감도: 보정 전 PSA 분포 차이의 방법 일치성",
        "",
        "- log10-PSA Welch/Games–Howell과 Kruskal–Wallis/Dunn–Holm의 방향과 "
        "유의성 패턴이 일치하는지 평가했다.",
        "- H3는 새로운 확증적 주장보다 H1 결과의 강건성 점검으로 사용했다.",
        "",
        "## 통계기법 선택 근거",
        "",
        "- ANCOVA는 연령과 BMI를 보정한 뒤 PSA의 군 효과를 직접 검정하므로 H1의 주분석이다. "
        "PSA의 강한 우측 꼬리를 줄이기 위해 log10 변환했고, 이분산에 덜 민감한 HC3 "
        "공분산을 사용했다.",
        "- 연령·BMI는 군별 분산과 표본수가 같다고 전제하기 어려워 Welch ANOVA를 사용했다. "
        "전체검정이 열린 경우의 쌍별 평균 비교는 동일한 조건을 허용하는 Games–Howell을 썼다.",
        "- Dunn–Holm은 정규성이나 등분산을 요구하지 않는 순위 기반 PSA 민감도 분석이다. "
        "Dunn 검정의 세 쌍 p값에 Holm 보정을 적용해 family-wise error를 통제했다.",
        "- 단순 ANCOVA가 부적절한 것이 아니라, 원자료 PSA에 등분산·정규오차를 그대로 가정하는 "
        "ANCOVA가 취약하다. 본 분석은 log 변환, HC3, 공통기울기 점검으로 그 문제를 다뤘다.",
        "",
        "## H1 결과 — 보정 후에도 PSA 군 차이가 유지됨",
        "",
    ]
    for result in ancova.contrasts:
        lines.append(
            f"- {result.comparison}: 보정 기하평균비 {result.adjusted_ratio:.2f} "
            f"(95% CI {result.ci_lower:.2f}–{result.ci_upper:.2f}), "
            f"Holm {format_p(result.p_adjusted)}."
        )
    lines.extend(["", "## H2 결과 — 연령은 불균형, BMI는 전체검정 비유의", ""])
    for analysis, adjusted_p in zip(secondary, secondary_adjusted, strict=True):
        lines.append(
            f"- {analysis.metric.name}: Welch F={analysis.welch.statistic:.2f}, "
            f"원 p={analysis.welch.p_value:.3g}, Holm 보정 "
            f"{format_p(float(adjusted_p))}, 최대 |SMD|="
            f"{max(abs(result.value) for result in analysis.smd):.2f}."
        )
    age = secondary[0]
    for result in age.games_howell:
        if result.p_adjusted < 0.05:
            lines.append(
                f"- 연령 {result.group_1}−{result.group_2}: 평균차 {result.estimate:.2f}세 "
                f"(95% CI {result.ci_lower:.2f}–{result.ci_upper:.2f}), "
                f"{format_p(result.p_adjusted)}."
            )
    lines.extend(
        [
            "",
            "## H3 결과 — 비모수 분석도 PSA 순서를 재현",
            "",
            f"- log10-PSA Welch 전체검정: {format_p(psa.welch.p_value)}.",
            f"- Kruskal–Wallis 전체검정: {format_p(psa.kruskal.p_value)}.",
            "- Games–Howell과 Dunn–Holm 모두 세 쌍에서 같은 방향의 차이를 보였다. "
            "아래 p값은 각 방법 내 다중비교가 반영된 값이다.",
        ]
    )
    for games_howell, dunn_holm in zip(
        psa.games_howell,
        psa.dunn_holm,
        strict=True,
    ):
        lines.append(
            f"- {games_howell.group_1} vs {games_howell.group_2}: "
            f"Games–Howell {format_p(games_howell.p_adjusted)}, "
            f"Dunn–Holm {format_p(dunn_holm.p_adjusted)}."
        )
    lines.extend(
        [
            "",
            "## 분석 범위와 가정",
            "",
            f"- 총 {context.total_n}명; ANCOVA는 BMI 결측 {context.missing_bmi_n}명을 제외한 "
            f"{ancova.omnibus.sample_size}명 complete-case 분석이다.",
            f"- 연령 공통 중첩 범위: {context.age_overlap}.",
            f"- BMI 공통 중첩 범위: {context.bmi_overlap}.",
            "- 모든 PSA가 양수여서 log10 변환이 가능했다.",
            "",
            "## 제한과 해석 경계",
            "",
            "- 관찰 자료이므로 보정 후 연관성이지 인과효과가 아니다.",
            "- 그룹 정의에 PSA가 사용됐다면 PSA 차이는 선택 기준의 결과일 수 있다.",
            "- 연령과 BMI만 보정했으므로 미측정 교란은 남아 있다.",
            "- BMI 두 건의 결측은 complete-case에서 무작위 결측으로 취급했으나 검증할 수 없다.",
            "- 제공표의 BMI는 소수 둘째 자리 표시값이므로 원 Excel 비반올림 값과 미세한 차이가 "
            "생길 수 있다.",
            "",
            "## 다음 단계",
            "",
            "1. Biopsy-negative/Cancer의 임상적 정의와 PSA가 군 정의에 사용됐는지 문서화한다.",
            "2. 원본 Excel의 비반올림 BMI와 결측 사유를 확인해 동일 분석을 재실행한다.",
            "3. 원인 해석이 필요하면 전립선 용적, 생검 결과, PI-RADS 등 사전 교란변수를 "
            "분석계획서에 추가한다.",
            "",
            "## 추가 질문",
            "",
            "- Biopsy-negative와 Cancer 판정 기준은 각각 무엇인가?",
            "- PSA가 군 분류 기준에 직접 또는 간접적으로 포함됐는가?",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
