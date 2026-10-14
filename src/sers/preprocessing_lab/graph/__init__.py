"""Preprocessing Lab LangGraph 파일럿 — 논문→구현→검수→실행→비교→채택→기록 흐름 강제.

설계: docs/ml/preprocessing_lab_langgraph_pilot_design.md
실행 환경: conda env `sers-langgraph` (sers-analysis env와 분리)

M0 상태: 사람 승인 node(interrupt)·분기 규칙·체크포인트 재개는 실제로 동작하고,
코드·LLM node는 stub이다 (nodes/ 각 함수 docstring에 M1/M2 구현 내용 명시).
"""

import os

# 설계 §7: 실행 추적을 LangSmith 등 외부로 보내지 않는다 (setdefault가 아니라 강제).
os.environ["LANGSMITH_TRACING"] = "false"

from .builder import build_graph  # noqa: E402

__all__ = ["build_graph"]
