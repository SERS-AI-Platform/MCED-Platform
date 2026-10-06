# sers-plan-critic — Preprocessing Lab LangGraph 파일럿 설계안 v1

- 날짜: 2026-09-14
- 대상: `docs/ml/preprocessing_lab_langgraph_pilot_design.md` (v1)
- 판정: **GAPS FOUND**

## BLOCKER
1. **E1 게이트 모순** — §6이 "우회 없음"이라면서 `gate_exemption` 사유 입력 시 통과. SKILL.md:61은 pending 실행을 예외 없이 금지하며, 이 면제 경로가 E1의 원인(experiment.md:372-373). 입력 주체·테스트도 미정.
2. **SKILL 중복·충돌** — SKILL.md:15-19 "전용 로직 신설 금지", :63 "평가는 항상 @model-bench"와 충돌. 어느 쪽이 기준인지 사용자 결정 필요.
3. **기록 3곳 중 1곳만** — 대시보드 `SERS_AI_Experiment_History.html` PHASES, SERS-AI memory 갱신, 사후 `sers-experiment-auditor` 누락.

## WATCH
- record가 "코드+LLM"인데 수치 생성 금지 제약·테스트 없음 (append-only 문서 오염 위험)
- preflight 1회 검사로 E4 미방어, 브랜치 검사 없음 (detached HEAD / 미병합 worktree)
- 코호트 스냅샷이 그룹별 개수만 비교 → 라벨 맞교환 미탐지. E6는 경고만 하고 채택 판단까지 흘러감
- 병원·장비 confound 검사 없음, 단일 코호트 한계가 compare 입력에 없음
- LLM API 전송이 M2 착수 조건으로 막혀 있지 않음, `LANGSMITH_TRACING` 미명시
- 임상 스키마 읽기 전용이 선언뿐 (권한 장치 없음)
- §12 Supabase에 "(보안정책 확인 전)" 단서 → 무조건 금지로 표기해야 함, `sync_data.py` 미실행 명시
- 기타: verify_impl의 CONDITIONS 등록 미확인, `thread_id = method_key` 재실행 충돌, method_key 명칭 불일치(run_guide_pipeline.py:36-37), set_audit_status/upsert_method 경로 중복 가능성, E3 검증 누락

## 반영 (v2)
- B1 → 면제 경로 삭제, 탐색 실행은 그래프 밖에서만 (§6, §12)
- B2 → §13 결정 1로 사용자 결정 요청 (선택지 a/b)
- B3 → §5.4 기록 대상 4곳 + `verify_record` 코드 node(수치 대조)
- record → 코드 템플릿 전용으로 변경, LLM node 4개 유지
- 브랜치 검사 → preflight, E4 → `pre_run_check`
- 스냅샷 → measurement_id 집합 해시 + 라벨 매핑 해시, E6 → 본표에서 분리
- confound → 스냅샷에 sites/instruments, 단일 병원 한계 고정 문구 (§5.3)
- 보안 → M2는 보안정책 확인 후 착수, `LANGSMITH_TRACING=false`, 전용 DB role은 §13 결정 6
- Supabase → 무조건 금지 표기, sync 스크립트 미실행 명시
- 기타 → verify_impl에 CONDITIONS 확인, `thread_id = run_key`, set_audit_status는 audit_status 컬럼만 UPDATE (upsert_method는 해당 컬럼 미사용 — executability 리뷰에서 확인), §10에 E3 추가
- 미반영: method_key 명칭 불일치(`calibration_astm_reference`)는 기존 스크립트 이슈라 파일럿 범위 밖으로 둠
