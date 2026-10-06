"""M0 stub — 코드·LLM node 자리표시자.

모든 stub은 결과에 ``"stub": True``를 남기고 통과 값만 채운다. M1(코드 node)과
M2(LLM node)에서 설계 §9의 모듈(preflight/paper/implement/audit/benchmark/report)로
하나씩 교체한다. docstring은 교체 시 구현해야 할 내용이다.
"""

from __future__ import annotations

from ..state import LabState

_STUB = {"stub": True}


def preflight(state: LabState) -> dict:
    """[M1 코드] 브랜치(detached HEAD 거부)·DB 접속 확인, 코호트 스냅샷(§4) 저장."""
    return {"preflight": {**_STUB, "ok": True}}


def acquire_paper(state: LabState) -> dict:
    """[M2 LLM+도구] 전문·저자 공개 코드 확보. 유료 장벽이면 status=needs_human (우회 금지)."""
    return {"paper": {**_STUB, "status": "acquired"}}


def extract_method(state: LabState) -> dict:
    """[M2 LLM] 알고리즘·파라미터·기본값을 구조화 추출."""
    return {"extracted": dict(_STUB)}


def verify_impl(state: LabState) -> dict:
    """[M1 코드] pytest·ruff, dispatcher 등록, CONDITIONS에 condition_name 존재, 기본 config 회귀."""
    return {"verify_report": {**_STUB, "passed": True}}


def audit(state: LabState) -> dict:
    """[M2 LLM] paper-code-audit 판정 → AuditVerdict."""
    return {"audit": dict(_STUB)}


def set_audit_status(state: LabState) -> dict:
    """[M1 코드] experiment.preprocessing_methods.audit_status만 UPDATE (db.py 신규 메서드)."""
    return {"audit_status_update": dict(_STUB)}


def pre_run_check(state: LabState) -> dict:
    """[M1 코드] 작업 트리 clean, HEAD == impl_commit (E4)."""
    return {"pre_run": {**_STUB, "ok": True}}


def run_benchmark(state: LabState) -> dict:
    """[M1 subprocess] run_guide_pipeline.py --dry-run → 본 실행 → --load-db (sers-analysis env)."""
    return {"benchmark": {**_STUB, "returncode": 0}}


def comparability_gate(state: LabState) -> dict:
    """[M1 코드] 코호트 해시 재대조(E5), 기준 cal_ps_si 대비 환자 수 분리(E6), 병원·장비 구성."""
    return {"comparability": {**_STUB, "cohort_match": True}}


def compare(state: LabState) -> dict:
    """[M2 LLM] 코드가 DB에서 만든 비교표에 관찰 서술만 붙인다. 수치 생성·채택 표현 금지."""
    return {"comparison": dict(_STUB)}


def record(state: LabState) -> dict:
    """[M1 코드] 템플릿으로 기록 초안 4곳(§5.4) 생성. 파일 쓰기는 HITL-3 이후."""
    return {"record_draft": dict(_STUB)}


def verify_record(state: LabState) -> dict:
    """[M1 코드] 초안의 모든 수치를 DB·summary.csv와 대조."""
    return {"record_check": {**_STUB, "ok": True}}
