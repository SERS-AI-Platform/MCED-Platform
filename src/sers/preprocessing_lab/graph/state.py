"""그래프 state 정의 — 설계 §4.

원칙: 스펙트럼·환자 단위 데이터는 state에 넣지 않는다. 경로, run_name, 집계 지표만 둔다.
"""

from __future__ import annotations

from typing import Literal, TypedDict

AuditDecision = Literal["approved", "changes_requested", "rejected"]
AdoptionDecision = Literal["adopt", "hold", "reject"]

AUDIT_DECISIONS: frozenset[str] = frozenset({"approved", "changes_requested", "rejected"})
ADOPTION_DECISIONS: frozenset[str] = frozenset({"adopt", "hold", "reject"})
PAPER_ACTIONS: frozenset[str] = frozenset({"provided", "give_up"})
RECORD_ACTIONS: frozenset[str] = frozenset({"confirm", "reject"})


class CohortSnapshot(TypedDict):
    cohort_groups: list[str]
    counts_by_group: dict[str, int]
    measurement_ids_sha256: str  # E5: 개수가 같아도 구성이 바뀌면 잡힘
    label_map_sha256: str  # measurement_id → cohort_group 매핑 해시 (라벨 맞교환 대비)
    sites: list[str]
    instruments: list[str]
    taken_at: str


class AuditVerdict(TypedDict):
    matches_core_idea: bool
    mismatches: list[dict]  # {"item", "paper", "code", "severity"}
    unverifiable: list[str]
    source_used: Literal["paper_fulltext", "author_code", "design_guide_summary"]


class LabState(TypedDict, total=False):
    run_key: str  # f"{method_key}:{YYYYMMDD-HHMMSS}" = thread_id
    method_key: str  # preprocessing.py dispatcher 문자열
    stage: str
    paper: dict  # status: acquired | needs_human | given_up
    extracted: dict
    impl_commit: str
    condition_name: str  # run_guide_pipeline.CONDITIONS 키
    implement_attempts: int  # HITL-impl 진입 횟수 (최초 1 + 재시도)
    verify_report: dict  # passed: bool
    audit: AuditVerdict
    human_audit: dict  # decision, note, at
    audit_status_update: dict
    preflight: dict  # ok: bool, git_branch, cohort_snapshot
    pre_run: dict  # ok: bool, head, dirty
    benchmark: dict  # returncode: int, run_name, out_dir, summary_csv, log_path
    comparability: dict  # cohort_match: bool, separated_conditions
    comparison: dict
    adoption: dict  # decision, note, at
    record_draft: dict
    record_check: dict  # ok: bool
    record_confirmation: dict  # action, note, at
    abort_reason: str
