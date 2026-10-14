"""Preprocessing Lab 그래프 CLI (M0).

conda env `sers-langgraph`에서 프로젝트 루트 기준으로 실행한다::

    python -m sers.preprocessing_lab.graph.cli show              # mermaid 텍스트 (네트워크 없음)
    python -m sers.preprocessing_lab.graph.cli start --method-key despike_example
    python -m sers.preprocessing_lab.graph.cli status --run-key despike_example:20260914-110000
    python -m sers.preprocessing_lab.graph.cli resume --run-key despike_example:20260914-110000 \
        --payload '{"decision": "approved"}'

이미지가 필요하면 ``show > graph.mmd`` 후 ``mmdc -i graph.mmd -o graph.svg`` (로컬 렌더).
M0: 코드·LLM node는 stub이라 실제 DB·git·스크립트를 건드리지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from . import build_graph

REPO = Path(__file__).resolve().parents[4]
#: results/는 git-ignored — 최종 SSOT는 DB·experiment.md이고 checkpoint는 재개·추적용 (설계 §7)
DEFAULT_CHECKPOINT = REPO / "results" / "preprocessing_lab" / "langgraph" / "checkpoints.sqlite"


def _dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _report(graph, config: dict) -> None:
    state = graph.get_state(config)
    pending = [i.value for task in state.tasks for i in task.interrupts]
    if pending:
        print("⏸  사람 입력 대기:")
        print(_dump(pending))
    elif not state.next:
        reason = state.values.get("abort_reason")
        print(f"■ 종료 — {'ABORT: ' + reason if reason else '정상 완료'}")
    else:
        print(f"다음 node: {state.next}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("show", help="그래프 구조를 mermaid 텍스트로 출력")
    start = sub.add_parser("start", help="새 run 시작 (첫 사람 입력 지점까지 진행)")
    start.add_argument("--method-key", required=True)
    for name in ("status", "resume"):
        p = sub.add_parser(name)
        p.add_argument("--run-key", required=True)
        if name == "resume":
            p.add_argument("--payload", required=True, help="JSON — 대기 중인 interrupt의 expected 참고")
    args = parser.parse_args()

    if args.command == "show":
        print(build_graph().get_graph().draw_mermaid())
        return

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.checkpoint, check_same_thread=False) as conn:
        graph = build_graph(checkpointer=SqliteSaver(conn))
        if args.command == "start":
            run_key = f"{args.method_key}:{datetime.now():%Y%m%d-%H%M%S}"
            config = {"configurable": {"thread_id": run_key}}
            print(f"run_key = {run_key}")
            graph.invoke({"run_key": run_key, "method_key": args.method_key,
                          "implement_attempts": 0}, config)
        else:
            config = {"configurable": {"thread_id": args.run_key}}
            if not graph.get_state(config).values:
                raise SystemExit(f"checkpoint에 없는 run_key: {args.run_key}")
            if args.command == "resume":
                graph.invoke(Command(resume=json.loads(args.payload)), config)
        _report(graph, config)


if __name__ == "__main__":
    main()
