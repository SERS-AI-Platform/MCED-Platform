# plan-executability-reviewer — Preprocessing Lab LangGraph 파일럿 설계안 v1

- 날짜: 2026-09-14
- 대상: `docs/ml/preprocessing_lab_langgraph_pilot_design.md` (v1)
- 판정: **ITERATE** — 참조는 모두 실재·일치, 갭 3건

## 참조 확인 (모두 PASS)
- `run_guide_pipeline.py` 43-46 게이트 면제 문구, :142 CONDITIONS, 플래그 4개(:844-848)
- `experiment.md` 350-352 / 353-354 / 360-362 / 372-373 / 403-407 인용 내용 일치
- `db.py`: audit_status 갱신 코드 없음, `upsert_method`(:66-86)도 해당 컬럼 미사용
- `schema.py:155` `check_comparable`, `pghost.sh`, whitaker-hayes 감사 보고서(불일치 2건), 플러그인 가이드, comparator agent, `docs/ml/papers/`

## 이슈
1. **E4 방어 시점 오류** — git dirty 검사가 `implement` 이전 1회뿐. 실행 직전에 clean + `HEAD == impl_commit` 재검사 필요.
2. **"스크립트 구조 불변" 전제와 충돌** — `_load_run`에 `notes`(감사 면제 문구, :794-796), `phase="PL-3"`, `baseline="cal_despike"`(:805-806) 고정, `DATE_TAG="20260907"`(:125) 고정 + 기존 결과 있으면 skip(:549-552). 승인된 방법도 "면제된 PL-3 탐색"으로 적재됨.
3. **§10 입력 미정** — (a) E6 기준 조건·임계값 없음, `check_comparable`은 cohort_id만 비교(schema.py:169) (b) `gate_exemption` edge가 §6에만 있고 §3에 없음 (c) W-H 초기 구현 재현 커밋 미지정.

## 반영 (v2)
- 1 → `pre_run_check` node 추가 (§3, §5, §6, §10 E4)
- 2 → §5.1에 고정값 표 + 인자 4개 추가 제안, §13 결정 2로 사용자 확인
- 3a → §5.2 기준 `cal_ps_si`(:286), 임계값은 §13 결정 4로 미정 처리
- 3b → 면제 경로 자체 삭제 (critic BLOCKER와 함께 처리)
- 3c → `b9483f0` 지정 (threshold 7.0 여부는 착수 시 확인)
