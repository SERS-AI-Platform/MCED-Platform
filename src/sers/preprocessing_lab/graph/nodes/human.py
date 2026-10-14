"""사람 승인 node — ``interrupt()``로 멈추고 resume payload를 검증한다 (설계 §2).

잘못된 payload는 통과로 해석하지 않고, 오류 메시지를 붙여 **같은 자리에서 다시 묻는다**.
예외로 멈추면 안 되는 이유: LangGraph는 resume 값을 체크포인트에 기록하므로, node가
예외를 던지면 이후 올바른 값을 넣어도 기록된 잘못된 값이 다시 읽혀 run이 막힌다
(LangGraph 1.2.11에서 확인). 한 node 안의 여러 ``interrupt()``는 resume 값을 순서대로
받으므로 반복해서 묻는 방식은 재개와 충돌하지 않는다.

interrupt 이후 node는 재개 시 처음부터 다시 실행되므로, interrupt 앞에서는 state를
읽기만 하고 부수효과를 만들지 않는다.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import TypeVar

from langgraph.types import interrupt

from ..routing import MAX_IMPLEMENT_ATTEMPTS
from ..state import (
    ADOPTION_DECISIONS,
    AUDIT_DECISIONS,
    PAPER_ACTIONS,
    RECORD_ACTIONS,
    LabState,
)

T = TypeVar("T")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require(reply: object, key: str, allowed: frozenset[str] | None = None) -> str:
    if not isinstance(reply, dict) or not isinstance(reply.get(key), str) or not reply[key].strip():
        raise ValueError(f"resume payload에 문자열 '{key}'가 필요하다: {reply!r}")
    value = reply[key]
    if allowed is not None and value not in allowed:
        raise ValueError(f"'{key}'={value!r} — 허용값: {sorted(allowed)}")
    return value


def _ask(payload: dict, parse: Callable[[object], T]) -> T:
    """유효한 답이 올 때까지 같은 질문을 반복한다. 직전 오류는 payload['error']로 보여준다."""
    error = None
    while True:
        reply = interrupt({**payload, "error": error} if error else payload)
        try:
            return parse(reply)
        except ValueError as exc:
            error = str(exc)


def paper_needed(state: LabState) -> dict:
    """HITL-0 — 논문 전문 확보 실패. 사람이 PDF 경로를 주거나 포기한다."""
    paper = dict(state.get("paper") or {})

    def parse(reply: object) -> dict:
        if _require(reply, "action", PAPER_ACTIONS) == "give_up":
            return {**paper, "status": "given_up", "note": reply.get("note", "")}
        return {**paper, "status": "acquired", "fulltext_path": _require(reply, "fulltext_path")}

    return {"paper": _ask({
        "kind": "paper_needed",
        "method_key": state.get("method_key"),
        "paper": state.get("paper"),
        "expected": {"action": sorted(PAPER_ACTIONS), "fulltext_path": "action=provided일 때"},
    }, parse)}


def implement(state: LabState) -> dict:
    """HITL-impl — 파일럿에서는 Claude Code 세션에서 구현·커밋한 뒤 결과를 입력한다.

    플러그인 가이드 1~5단계 + run_guide_pipeline.CONDITIONS에 조건 추가가 끝난 커밋이어야 한다.
    """
    attempt = int(state.get("implement_attempts", 0)) + 1

    def parse(reply: object) -> dict:
        return {
            "implement_attempts": attempt,
            "impl_commit": _require(reply, "impl_commit"),
            "condition_name": _require(reply, "condition_name"),
        }

    return _ask({
        "kind": "implement",
        "method_key": state.get("method_key"),
        "attempt": attempt,
        "max_attempts": MAX_IMPLEMENT_ATTEMPTS,
        "extracted": state.get("extracted"),
        "previous_verify_report": state.get("verify_report"),
        "previous_human_audit": state.get("human_audit"),
        "expected": {"impl_commit": "git hash", "condition_name": "CONDITIONS 키"},
    }, parse)


def _decision(key: str, allowed: frozenset[str]) -> Callable[[object], dict]:
    def parse(reply: object) -> dict:
        value = _require(reply, key, allowed)
        return {key: value, "note": reply.get("note", ""), "at": _now()}
    return parse


def audit_review(state: LabState) -> dict:
    """HITL-1 — 감사 보고서를 보고 사람이 결정한다. approved만 실행 단계로 이어진다."""
    return {"human_audit": _ask({
        "kind": "audit_review",
        "method_key": state.get("method_key"),
        "impl_commit": state.get("impl_commit"),
        "audit": state.get("audit"),
        "attempt": state.get("implement_attempts"),
        "expected": {"decision": sorted(AUDIT_DECISIONS), "note": "선택"},
    }, _decision("decision", AUDIT_DECISIONS))}


def adoption(state: LabState) -> dict:
    """HITL-2 — 비교표를 보고 사람이 채택/보류/기각을 정한다. 어떤 결정이든 기록 단계로 간다."""
    return {"adoption": _ask({
        "kind": "adoption",
        "method_key": state.get("method_key"),
        "comparison": state.get("comparison"),
        "comparability": state.get("comparability"),
        "expected": {"decision": sorted(ADOPTION_DECISIONS), "note": "선택"},
    }, _decision("decision", ADOPTION_DECISIONS))}


def record_confirm(state: LabState) -> dict:
    """HITL-3 — 수치 대조를 통과한 기록 초안을 사람이 확인한다. 반영·커밋은 confirm 이후."""
    return {"record_confirmation": _ask({
        "kind": "record_confirm",
        "method_key": state.get("method_key"),
        "record_draft": state.get("record_draft"),
        "record_check": state.get("record_check"),
        "expected": {"action": sorted(RECORD_ACTIONS), "note": "선택"},
    }, _decision("action", RECORD_ACTIONS))}
