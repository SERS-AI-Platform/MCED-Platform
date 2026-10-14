"""분기 규칙 — 설계 §6.

모든 통과 조건은 값이 정확히 ``True``(또는 0)일 때만 인정한다. 필드가 없거나 stub이
값을 채우지 않으면 통과가 아니라 ABORT로 간다 (기억에 의존하지 않고 구조로 막는다).
"""

from __future__ import annotations

from langgraph.graph import END

from .state import LabState

ABORT = "abort"

#: 설계 §6 "retry < 2" = 최초 구현 1회 + 재시도 2회
MAX_IMPLEMENT_ATTEMPTS = 3


def _flag(state: LabState, key: str, field: str) -> bool:
    return (state.get(key) or {}).get(field) is True


def _attempts(state: LabState) -> int:
    return int(state.get("implement_attempts", 0))


def route_preflight(state: LabState) -> str:
    return "acquire_paper" if _flag(state, "preflight", "ok") else ABORT


def route_acquire_paper(state: LabState) -> str:
    status = (state.get("paper") or {}).get("status")
    return "extract_method" if status == "acquired" else "hitl_paper_needed"


def route_paper_needed(state: LabState) -> str:
    status = (state.get("paper") or {}).get("status")
    return "extract_method" if status == "acquired" else ABORT


def route_verify_impl(state: LabState) -> str:
    if _flag(state, "verify_report", "passed"):
        return "audit"
    return "hitl_implement" if _attempts(state) < MAX_IMPLEMENT_ATTEMPTS else ABORT


def route_audit_review(state: LabState) -> str:
    """E1: approved 외에는 어떤 값으로도 set_audit_status(→ 실행)로 가지 않는다. 면제 경로 없음."""
    decision = (state.get("human_audit") or {}).get("decision")
    if decision == "approved":
        return "set_audit_status"
    if decision == "changes_requested" and _attempts(state) < MAX_IMPLEMENT_ATTEMPTS:
        return "hitl_implement"
    return ABORT


def route_pre_run_check(state: LabState) -> str:
    return "run_benchmark" if _flag(state, "pre_run", "ok") else ABORT


def route_run_benchmark(state: LabState) -> str:
    returncode = (state.get("benchmark") or {}).get("returncode")
    return "comparability_gate" if returncode == 0 else ABORT


def route_comparability_gate(state: LabState) -> str:
    return "compare" if _flag(state, "comparability", "cohort_match") else ABORT


def route_verify_record(state: LabState) -> str:
    return "hitl_record_confirm" if _flag(state, "record_check", "ok") else ABORT


def route_record_confirm(state: LabState) -> str:
    action = (state.get("record_confirmation") or {}).get("action")
    return END if action == "confirm" else ABORT


def abort_reason(state: LabState) -> str:
    """ABORT에 도달한 이유를 그래프 순서대로 찾아 한 줄로 남긴다."""
    if "preflight" in state and not _flag(state, "preflight", "ok"):
        return "preflight 실패 (브랜치 또는 DB 접속)"
    if (state.get("paper") or {}).get("status") == "given_up":
        return "논문 확보 포기 (HITL-0)"
    if "verify_report" in state and not _flag(state, "verify_report", "passed"):
        return f"구현 검증 실패 — 시도 {_attempts(state)}회로 한도 도달"
    decision = (state.get("human_audit") or {}).get("decision")
    if decision == "rejected":
        return "구현 검수 기각 (HITL-1)"
    if decision == "changes_requested":
        return f"검수 수정요청 반복 — 시도 {_attempts(state)}회로 한도 도달"
    if "pre_run" in state and not _flag(state, "pre_run", "ok"):
        return "실행 직전 검사 실패 (작업 트리 dirty 또는 HEAD != impl_commit)"
    if "benchmark" in state and (state.get("benchmark") or {}).get("returncode") != 0:
        return f"벤치마크 실행 실패 (returncode={state['benchmark'].get('returncode')})"
    if "comparability" in state and not _flag(state, "comparability", "cohort_match"):
        return "코호트 스냅샷 불일치 — 실행 중 데이터 구성 변경"
    if "record_check" in state and not _flag(state, "record_check", "ok"):
        return "기록 초안 수치가 DB/CSV와 불일치"
    if (state.get("record_confirmation") or {}).get("action") == "reject":
        return "기록 초안 반려 (HITL-3)"
    return "원인 미상 — state 확인 필요"
