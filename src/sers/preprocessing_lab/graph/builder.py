"""StateGraph 조립 — 설계 §3의 흐름도를 코드로 옮긴 것.

edge 목록이 곧 규칙이다. 특히 run_benchmark의 선행 node는 pre_run_check 하나이고,
pre_run_check의 선행은 set_audit_status, 그 선행은 hitl_audit_review(approved 분기)뿐이다.
이 구조는 tests/preprocessing_lab/graph/test_graph_skeleton.py가 검사한다.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from langgraph.graph import END, START, StateGraph

from . import routing as r
from .nodes import human, stubs
from .state import LabState

NodeFn = Callable[[LabState], dict]

DEFAULT_NODES: dict[str, NodeFn] = {
    "preflight": stubs.preflight,
    "acquire_paper": stubs.acquire_paper,
    "hitl_paper_needed": human.paper_needed,
    "extract_method": stubs.extract_method,
    "hitl_implement": human.implement,
    "verify_impl": stubs.verify_impl,
    "audit": stubs.audit,
    "hitl_audit_review": human.audit_review,
    "set_audit_status": stubs.set_audit_status,
    "pre_run_check": stubs.pre_run_check,
    "run_benchmark": stubs.run_benchmark,
    "comparability_gate": stubs.comparability_gate,
    "compare": stubs.compare,
    "hitl_adoption": human.adoption,
    "record": stubs.record,
    "verify_record": stubs.verify_record,
    "hitl_record_confirm": human.record_confirm,
    r.ABORT: lambda state: {"abort_reason": r.abort_reason(state)},
}


def build_graph(*, checkpointer: Any = None, overrides: Mapping[str, NodeFn] | None = None):
    """그래프를 컴파일한다.

    Parameters
    ----------
    checkpointer
        재개가 필요하면 ``SqliteSaver``를 넘긴다. None이면 재개 불가(구조 확인용).
    overrides
        node 이름 → 함수. 테스트나 M1/M2 단계별 교체에 쓴다. 없는 이름이면 ValueError.
    """
    unknown = set(overrides or {}) - set(DEFAULT_NODES)
    if unknown:
        raise ValueError(f"unknown node(s): {sorted(unknown)}")
    nodes = {**DEFAULT_NODES, **(overrides or {})}

    graph = StateGraph(LabState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "preflight")
    graph.add_conditional_edges("preflight", r.route_preflight, ["acquire_paper", r.ABORT])
    graph.add_conditional_edges(
        "acquire_paper", r.route_acquire_paper, ["extract_method", "hitl_paper_needed"])
    graph.add_conditional_edges(
        "hitl_paper_needed", r.route_paper_needed, ["extract_method", r.ABORT])
    graph.add_edge("extract_method", "hitl_implement")
    graph.add_edge("hitl_implement", "verify_impl")
    graph.add_conditional_edges(
        "verify_impl", r.route_verify_impl, ["audit", "hitl_implement", r.ABORT])
    graph.add_edge("audit", "hitl_audit_review")
    graph.add_conditional_edges(
        "hitl_audit_review", r.route_audit_review,
        ["set_audit_status", "hitl_implement", r.ABORT])
    graph.add_edge("set_audit_status", "pre_run_check")
    graph.add_conditional_edges("pre_run_check", r.route_pre_run_check, ["run_benchmark", r.ABORT])
    graph.add_conditional_edges(
        "run_benchmark", r.route_run_benchmark, ["comparability_gate", r.ABORT])
    graph.add_conditional_edges(
        "comparability_gate", r.route_comparability_gate, ["compare", r.ABORT])
    graph.add_edge("compare", "hitl_adoption")
    graph.add_edge("hitl_adoption", "record")
    graph.add_edge("record", "verify_record")
    graph.add_conditional_edges(
        "verify_record", r.route_verify_record, ["hitl_record_confirm", r.ABORT])
    graph.add_conditional_edges("hitl_record_confirm", r.route_record_confirm, [END, r.ABORT])
    graph.add_edge(r.ABORT, END)
    return graph.compile(checkpointer=checkpointer)
