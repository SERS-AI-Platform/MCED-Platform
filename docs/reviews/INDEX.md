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
