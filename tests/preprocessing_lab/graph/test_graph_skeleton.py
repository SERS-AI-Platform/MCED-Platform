"""Preprocessing Lab 그래프 M0 — 흐름·게이트·재개 검증 (설계 §10 중 구조로 확인 가능한 항목).

실행: conda env `sers-langgraph`에서 ``pytest tests/preprocessing_lab/graph``
"""

from __future__ import annotations

import sqlite3

import pytest

langgraph = pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from sers.preprocessing_lab.graph import build_graph  # noqa: E402
from sers.preprocessing_lab.graph.routing import MAX_IMPLEMENT_ATTEMPTS  # noqa: E402

DESIGN_NODES = {
    "preflight", "acquire_paper", "hitl_paper_needed", "extract_method", "hitl_implement",
    "verify_impl", "audit", "hitl_audit_review", "set_audit_status", "pre_run_check",
    "run_benchmark", "comparability_gate", "compare", "hitl_adoption", "record",
    "verify_record", "hitl_record_confirm", "abort",
}
IMPL = {"impl_commit": "abc1234", "condition_name": "ps_example_method"}


def _config(thread_id: str = "t") -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _start(graph, config: dict) -> None:
    graph.invoke({"run_key": "m:1", "method_key": "m", "implement_attempts": 0}, config)


def _pending(graph, config: dict) -> dict | None:
    pending = [i.value for t in graph.get_state(config).tasks for i in t.interrupts]
    assert len(pending) <= 1
    return pending[0] if pending else None


def _pending_kind(graph, config: dict) -> str | None:
    pending = _pending(graph, config)
    return pending["kind"] if pending else None


def _drive(graph, config: dict, answers: dict[str, dict]) -> list[str]:
    """대기 중인 interrupt 종류에 맞는 답을 계속 넣어 그래프가 끝날 때까지 진행한다."""
    seen = []
    while (kind := _pending_kind(graph, config)) is not None:
        seen.append(kind)
        graph.invoke(Command(resume=answers[kind]), config)
    return seen


def _predecessors(graph, node: str) -> set[str]:
    return {e.source for e in graph.get_graph().edges if e.target == node}


def test_graph_nodes_match_design() -> None:
    graph = build_graph()
    assert set(graph.get_graph().nodes) == DESIGN_NODES | {"__start__", "__end__"}
    mermaid = graph.get_graph().draw_mermaid()
    assert all(name in mermaid for name in DESIGN_NODES)


def test_happy_path_stops_at_each_human_gate_in_order() -> None:
    graph = build_graph(checkpointer=InMemorySaver())
    config = _config()
    _start(graph, config)
    seen = _drive(graph, config, {
        "implement": IMPL,
        "audit_review": {"decision": "approved"},
        "adoption": {"decision": "hold", "note": "표본 부족"},
        "record_confirm": {"action": "confirm"},
    })
    state = graph.get_state(config)
    assert seen == ["implement", "audit_review", "adoption", "record_confirm"]
    assert state.next == ()
    assert "abort_reason" not in state.values
    assert state.values["adoption"]["decision"] == "hold"


def test_benchmark_is_reachable_only_through_audit_approval() -> None:
    """E1 구조 검사 — 면제·우회 edge가 없다."""
    graph = build_graph()
    assert _predecessors(graph, "run_benchmark") == {"pre_run_check"}
    assert _predecessors(graph, "pre_run_check") == {"set_audit_status"}
    assert _predecessors(graph, "set_audit_status") == {"hitl_audit_review"}


@pytest.mark.parametrize(("decision", "expected_attempts"), [
    ("rejected", 1),
    ("changes_requested", MAX_IMPLEMENT_ATTEMPTS),
])
def test_non_approval_never_runs_benchmark(decision: str, expected_attempts: int) -> None:
    calls = []
    graph = build_graph(checkpointer=InMemorySaver(), overrides={
        "run_benchmark": lambda s: calls.append(s) or {"benchmark": {"returncode": 0}},
    })
    config = _config()
    _start(graph, config)
    _drive(graph, config, {"implement": IMPL, "audit_review": {"decision": decision}})
    state = graph.get_state(config)
    assert calls == []
    assert state.next == ()
    assert state.values["implement_attempts"] == expected_attempts
    assert "검수" in state.values["abort_reason"]


def test_invalid_audit_decision_is_asked_again_and_gate_stays_closed() -> None:
    """오타는 통과도 아니고 run을 막지도 않는다 — 같은 자리에서 오류와 함께 다시 묻는다."""
    calls = []
    graph = build_graph(checkpointer=InMemorySaver(), overrides={
        "set_audit_status": lambda s: calls.append(s) or {"audit_status_update": {}},
    })
    config = _config()
    _start(graph, config)
    graph.invoke(Command(resume=IMPL), config)
    graph.invoke(Command(resume={"decision": "approve"}), config)  # 오타
    pending = _pending(graph, config)
    assert calls == []
    assert pending["kind"] == "audit_review"
    assert "approve" in pending["error"]
    graph.invoke(Command(resume={"decision": "approved"}), config)
    assert len(calls) == 1
    assert _pending_kind(graph, config) == "adoption"
    assert graph.get_state(config).values["human_audit"]["decision"] == "approved"


def test_invalid_payload_survives_sqlite_resume(tmp_path) -> None:
    """잘못된 답 → 프로세스 종료 → 새 인스턴스에서 올바른 답으로 이어가기."""
    path = tmp_path / "checkpoints.sqlite"
    config = _config("m:typo")
    first = sqlite3.connect(path, check_same_thread=False)
    graph = build_graph(checkpointer=SqliteSaver(first))
    _start(graph, config)
    graph.invoke(Command(resume={"impl_commit": "abc1234"}), config)  # condition_name 누락
    first.close()

    second = sqlite3.connect(path, check_same_thread=False)
    resumed = build_graph(checkpointer=SqliteSaver(second))
    pending = _pending(resumed, config)
    assert pending["kind"] == "implement" and "condition_name" in pending["error"]
    resumed.invoke(Command(resume=IMPL), config)
    assert _pending_kind(resumed, config) == "audit_review"
    assert resumed.get_state(config).values["implement_attempts"] == 1
    second.close()


def test_verify_failure_loops_back_then_aborts_at_limit() -> None:
    graph = build_graph(checkpointer=InMemorySaver(), overrides={
        "verify_impl": lambda s: {"verify_report": {"passed": False}},
    })
    config = _config()
    _start(graph, config)
    seen = _drive(graph, config, {"implement": IMPL})
    state = graph.get_state(config)
    assert seen == ["implement"] * MAX_IMPLEMENT_ATTEMPTS
    assert "구현 검증 실패" in state.values["abort_reason"]


@pytest.mark.parametrize(("node", "update", "reason"), [
    ("preflight", {"preflight": {"ok": False}}, "preflight"),
    ("pre_run_check", {"pre_run": {"ok": False}}, "HEAD"),
    ("comparability_gate", {"comparability": {"cohort_match": False}}, "코호트"),
    ("verify_record", {"record_check": {"ok": False}}, "불일치"),
])
def test_failed_code_check_aborts_with_reason(node: str, update: dict, reason: str) -> None:
    graph = build_graph(checkpointer=InMemorySaver(), overrides={node: lambda s: update})
    config = _config()
    _start(graph, config)
    _drive(graph, config, {
        "implement": IMPL,
        "audit_review": {"decision": "approved"},
        "adoption": {"decision": "adopt"},
    })
    state = graph.get_state(config)
    assert state.next == ()
    assert reason in state.values["abort_reason"]


def test_stub_that_omits_pass_flag_is_not_treated_as_pass() -> None:
    graph = build_graph(checkpointer=InMemorySaver(), overrides={"preflight": lambda s: {}})
    config = _config()
    _start(graph, config)
    assert graph.get_state(config).next == ()
    assert _pending_kind(graph, config) is None


@pytest.mark.parametrize(("answer", "next_kind"), [
    ({"action": "provided", "fulltext_path": "docs/ml/papers/example.pdf"}, "implement"),
    ({"action": "give_up", "note": "유료 장벽"}, None),
])
def test_paper_needed_gate(answer: dict, next_kind: str | None) -> None:
    graph = build_graph(checkpointer=InMemorySaver(), overrides={
        "acquire_paper": lambda s: {"paper": {"status": "needs_human"}},
    })
    config = _config()
    _start(graph, config)
    assert _pending_kind(graph, config) == "paper_needed"
    graph.invoke(Command(resume=answer), config)
    assert _pending_kind(graph, config) == next_kind


def test_record_rejection_aborts() -> None:
    graph = build_graph(checkpointer=InMemorySaver())
    config = _config()
    _start(graph, config)
    _drive(graph, config, {
        "implement": IMPL,
        "audit_review": {"decision": "approved"},
        "adoption": {"decision": "reject"},
        "record_confirm": {"action": "reject"},
    })
    assert "반려" in graph.get_state(config).values["abort_reason"]


def test_run_resumes_from_sqlite_in_a_new_graph_instance(tmp_path) -> None:
    path = tmp_path / "checkpoints.sqlite"
    config = _config("m:resume")
    first = sqlite3.connect(path, check_same_thread=False)
    graph = build_graph(checkpointer=SqliteSaver(first))
    _start(graph, config)
    graph.invoke(Command(resume=IMPL), config)
    first.close()

    second = sqlite3.connect(path, check_same_thread=False)
    resumed = build_graph(checkpointer=SqliteSaver(second))
    assert _pending_kind(resumed, config) == "audit_review"
    assert resumed.get_state(config).values["impl_commit"] == "abc1234"
    second.close()


def test_unknown_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown node"):
        build_graph(overrides={"run_benchmrk": lambda s: {}})
