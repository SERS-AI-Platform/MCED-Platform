# 2026-09-08 sers-experiment-auditor — PL-3 기록 감사

대상: PL-3 설계가이드 요인별 비교 (`results/preprocessing_lab/guide_pipeline_20260907`). 원천 파일과
experiment.md / 대시보드 / registry / memory의 수치를 전수 대조.

## 판정
- **수치: CLEAR** — 20조건 × (환자, spectra, AUC[CI], 5시드 mean±sd, Δ[CI]) 전 셀 일치. append-only 준수(삭제 0줄).
- **서술: 1건 정정** — 관찰 5 "세 baseline 모두 1e7에서 떨어진다"는 arPLS에 거짓(1e6→1e7 상승). `0ead9f7`에서 정정.
- **정의 통일: 1건 수정** — 대시보드 PL-3 `auc`가 seed42 점추정(0.817)이라 PL-1(5시드 평균)과 정의가 달랐음 → 0.796(5시드 평균)으로 통일, desc에 seed42 값 병기.

## 미해결 (사용자 결정 필요)
1. `logs/experiment_registry.json`이 `.gitignore`(`logs/`)에 걸려 한 번도 커밋된 적 없음. SSOT라면 `!logs/experiment_registry.json` 예외 처리 검토.
2. PL-2 smoothing sweep(SG 11→5 근거) 결과가 experiment.md·registry에 없음. PL-3 관찰 3이 이를 인용하므로 우선순위 높음.
3. 감사자 프롬프트가 참조하는 `scripts/analysis/check_unlogged_experiments.py`가 존재하지 않음.
