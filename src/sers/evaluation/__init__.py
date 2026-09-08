"""모델 성능 평가 — 신뢰구간, 통계 검정, 표준 스키마.

`scripts/`와 `publications/` 안에 흩어져 있던 일회성 통계 코드를 설치 가능한
패키지로 모은 것. 보고 레이어가 여기서만 지표·구간을 가져가도록 한다.

핵심 원칙: 신뢰구간은 **환자 단위**로 계산한다. 스펙트럼 단위로 계산하면
구간이 실제보다 좁아져 없는 차이를 있다고 말하게 된다. `bootstrap` 모듈 설명
참고.
"""

from sers.evaluation.bootstrap import (
    BootstrapError,
    MetricCI,
    bootstrap_difference_ci,
    bootstrap_metric_ci,
)

__all__ = [
    "BootstrapError",
    "MetricCI",
    "bootstrap_difference_ci",
    "bootstrap_metric_ci",
]
