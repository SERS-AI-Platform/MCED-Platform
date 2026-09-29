# SERS-AI 리뷰 에이전트 로그 인덱스

`sers-experiment-auditor` / `sers-report-reviewer` / `sers-data-integrity-checker` /
`sers-plan-critic`이 실행될 때마다 남기는 리뷰 기록의 인덱스입니다.

**규칙: append-only.** 기존 행을 삭제·수정하지 않습니다 — 판정이 바뀌었으면
새 행을 추가하고 이전 행은 그대로 둡니다 (경위 추적 목적).

| 날짜 | 에이전트 | 대상 | 판정 | 파일 |
|---|---|---|---|---|
| 2026-08-31 | sers-data-integrity-checker | Dashboard v3 QC Threshold 라우트 (백테스트) | BLOCK | [파일](2026-08-31-sers-data-integrity-checker-qc-threshold-route.md) |
| 2026-09-02 | paper-code-audit | whitaker_hayes_despike 구현 vs 저자 참조구현 | REQUEST_CHANGES → 수정완료 | [파일](2026-09-02-paper-code-audit-whitaker-hayes.md) |
| 2026-09-02 | paper-code-audit | airpls_baseline 구현 vs Zhang 2010 | CLEAR(핵심)/WATCH(가장자리·λ기본값) | [파일](2026-09-02-paper-code-audit-airpls.md) |
| 2026-09-02 | deep-research 출처검증 | preprocessing_design_guide.md 주장 vs 인용 논문 원문 | FINDING 4건 (Zhao 인용이 주장과 반대) | [파일](2026-09-02-design-guide-source-verification.md) |
| 2026-09-08 | sers-experiment-auditor | PL-3 기록 vs 결과 파일 전수 대조 | 수치 CLEAR / 서술 1건 정정 / 대시보드 지표 정의 통일 | [파일](2026-09-08-sers-experiment-auditor-pl3.md) |
| 2026-09-11 | sers-report-reviewer | 환원제 변경 전/후 대표님 덱 v4 (동일 측정조건 5점 기준, 23장) | REQUEST_CHANGES → 11건 수정 후 전달 | [파일](2026-09-11-sers-report-reviewer-reagent-deck-v4.md) |
| 2026-09-14 | plan-executability-reviewer | Preprocessing Lab LangGraph 파일럿 설계안 v1 | ITERATE (갭 3건) → v2 반영 | [파일](2026-09-14-plan-executability-reviewer-langgraph-pilot.md) |
| 2026-09-14 | sers-plan-critic | Preprocessing Lab LangGraph 파일럿 설계안 v1 | GAPS FOUND (BLOCKER 3) → v2 반영, SKILL 관계는 사용자 결정 대기 | [파일](2026-09-14-sers-plan-critic-langgraph-pilot.md) |
| 2026-09-14 | sers-report-reviewer | 대표님 덱 v6 신규 3장 + 측정점 곡선 (무엇을 시험했나 / 판정 기준 / 36점 근거) | REQUEST_CHANGES → 7건 수정 후 전달 | [파일](2026-09-14-sers-report-reviewer-reagent-deck-v6-new-slides.md) |
| 2026-09-14 | sers-report-reviewer | 환원제 변경 전·후 동일 검체 427개 비교 덱 (36점 기준, 11장) | REQUEST_CHANGES(13건) → 2차 APPROVE, 선택 권고 5건 반영 | [파일](2026-09-14-sers-report-reviewer-reagent-deck-427-36pt.md) |
| 2026-09-14 | sers-plan-critic | Spike-in · LC-MS/MS 검증 실험 설계서 v1 (E0–E6) | GAPS FOUND (BLOCKER 6) → v2 전체 반영 | [파일](2026-09-14-sers-plan-critic-spike-in-design.md) |
| 2026-09-14 | sers-report-reviewer | 환원제 변경 전·후 427검체 덱 18장판 (STK-V2 1단계 → 2단계 05장·부록 C 추가) | REQUEST_CHANGES (11건) → 반영 진행 | [파일](2026-09-14-sers-report-reviewer-reagent-deck-427-36pt-stk.md) |
| 2026-09-14 | sers-report-reviewer | 427검체 덱 18장판 재검토 (1차 11건 반영 후) | 11건 해결 / REQUEST_CHANGES 3건(05장 각주 문구·부록 A·B 근거) → 반영 진행 | [파일](2026-09-14-sers-report-reviewer-reagent-deck-427-36pt-stk-r2.md) |
| 2026-09-14 | sers-report-reviewer | 427검체 덱 18장판 05장 기존 모델(1,630명) 기준 막대 추가 | REQUEST_CHANGES (6건, 이미 전달된 판) → 사용자 확인 대기 | [파일](2026-09-14-sers-report-reviewer-reagent-deck-427-36pt-stk-r3.md) |
| 2026-09-14 | sers-report-reviewer | 427검체 덱 18장판 재검토 (r3 6건 반영 후) | r3 해결 확인·겹침 93명이 맞음(r3의 91은 리뷰 매칭 오류) / REQUEST_CHANGES 1건(05장 각주 잘림) → 반영 진행 | [파일](2026-09-14-sers-report-reviewer-reagent-deck-427-36pt-stk-r4.md) |
| 2026-09-14 | sers-report-reviewer | 427검체 덱 17장판 (구분 AUC 장 삭제, 0.335 단일 제시) | 수치·장 번호·레이아웃 PASS / 표현 1건(0.335 비교 기준) 사용자 확인 대기 | [파일](2026-09-14-sers-report-reviewer-reagent-deck-427-36pt-stk-r5.md) |
| 2026-09-14 | sers-plan-critic | 실험 전체 흐름 LangGraph 설계안 v2 | GAPS FOUND (BLOCKER 3: 예약 장부 스키마 불일치·도메인 게이트 근거 부재·기존 agent 충돌) → v3 반영 | [파일](2026-09-14-sers-plan-critic-experiment-flow-langgraph.md) |
| 2026-09-14 | plan-executability-reviewer | 실험 전체 흐름 LangGraph 설계안 v2 | ITERATE (M0 재사용 주장 오류·node 입력 순서·F1–F8 시나리오) → v3 반영 | [파일](2026-09-14-plan-executability-reviewer-experiment-flow-langgraph.md) |
| 2026-09-14 | sers-report-reviewer | 427검체 덱 17장판 (혼선 줄이기 반영) | 수치 PASS / REQUEST_CHANGES 필수 1(02장 췌장암 합산 안내)·권고 6 → 사용자 확인 대기 | [파일](2026-09-14-sers-report-reviewer-reagent-deck-427-36pt-stk-r6.md) |
| 2026-09-14 | sers-report-reviewer | 427검체 덱 13장판 (교차 적용 장 삭제, 암종 구분만) | REQUEST_CHANGES 필수 2(부록 C 그림 재빌드, 사라진 통계 근거 복원)·권고 4 → 사용자 확인 대기 | [파일](2026-09-14-sers-report-reviewer-reagent-deck-427-36pt-stk-r7.md) |
| 2026-09-16 | sers-report-reviewer | 대표님 보고 덱 (574검체 Paired Analysis) | REQUEST_CHANGES → 반영 완료 | [파일](2026-09-16-sers-report-reviewer-paired-analysis-574.md) |
| 2026-09-17 | sers-report-reviewer | Paired Analysis 덱 768검체 장 추가분 (07장·08장·부록 A/B/C-4) | REQUEST_CHANGES → 11건 반영, 표지·요약 반영 여부 사용자 확인 대기 | [파일](2026-09-17-sers-report-reviewer-paired-analysis-768-chapter.md) |
| 2026-09-17 | sers-report-reviewer | Paired Analysis 덱 07·08장 수정분 재검토 (카드 삭제·결정 필요 삭제) | 거의 통과, 필수 1·권장 2 → 전부 반영 | [파일](2026-09-17-sers-report-reviewer-paired-analysis-768-r2.md) |
| 2026-09-18 | sers-report-reviewer | Paired Analysis 덱 (QC 2b 기준·926/667 코호트·추이 장 추가, 16장) | REQUEST_CHANGES → 전 항목 반영 | [파일](2026-09-18-sers-report-reviewer-paired-analysis-qc2b.md) |
| 2026-09-21 | sers-report-reviewer | Paired Analysis 덱 (KIMS 제외 871·코호트 패널·부록 D/E, 18장) | REQUEST_CHANGES(수치 일치, 표기 7건) → 전 항목 반영 | [파일](2026-09-21-sers-report-reviewer-paired-analysis-871.md) |
| 2026-09-29 | sers-report-reviewer | 분석 보고 덱 1038검체 (09·10장 재작성, 10-1·11-2 신설, 24장) | REQUEST_CHANGES(26건, 가짜 숫자 0) → 18건 반영·레이아웃 2건 미반영 후 전달 | [파일](2026-09-29-sers-report-reviewer-analysis-deck-1038.md) |
