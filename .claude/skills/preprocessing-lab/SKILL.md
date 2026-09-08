---
name: preprocessing-lab
description: 논문에서 가져온 SERS 전처리 방법을 구현→검수→평가→비교하는 SERS-AI 전용 워크플로우. 사용자가 논문(링크/PDF/DOI)을 주며 "이 전처리 방법 추가해줘"/"비교해줘"라고 할 때 사용.
---

# Preprocessing Lab — 논문 기반 전처리 벤치마크 워크플로우

> **저장소 (2026-09-02 확정)**: Supabase 금지로 `aecd_platform`의 `experiment`
> 스키마를 쓴다 (DDL: `scripts/db/experiment_tracking/01_schema.sql`,
> 쓰기: `src/sers/preprocessing_lab/db.py`). 임상 원본 스키마
> (`master`/`clinical`/`measurement`)는 여전히 읽기 전용.

전체 설계 근거: `/home/user/.claude/plans/fluttering-jumping-lagoon.md`.

이 스킬은 **오케스트레이션 순서만 정의**한다 — 실제 작업은 기존 SERS-AI
프로젝트 에이전트(`preprocess-lab`, `model-bench`, `experiment-runner`)와
Feynman 스킬(`alpha-research`, `paper-code-audit`), 그리고 이번에 추가된
`aecd-data-ops`/`sers-preprocessing-comparator`에 위임한다. 이 워크플로우
전용 로직을 새로 만들지 않는다 — 그러면 기존 4-에이전트 시스템과 중복된다.

사용자 역할 분담(고정): **레퍼런스 소싱 + 구현이 논문 핵심 아이디어와
맞는지 검수 + metrics 비교 검토 + 최종 채택 판단**은 항상 사람이 한다.

## 단계

### 1. 논문 확보
사용자가 링크/PDF/DOI를 준다. 없으면 `alpha-research` 스킬로 검색.

### 2. 방법 추출
`alpha ask <id> "이 논문이 설명하는 전처리 알고리즘과 파라미터는?"` 또는
직접 논문을 읽고 핵심 알고리즘을 추출한다.

### 3. 구현
`docs/ml/preprocessing_lab_plugin_guide.md`의 절차를 그대로 따른다:
`src/sers/preprocessing.py`의 dispatcher에 등록, `PreprocessingConfig`에
opt-in 필드 추가, `tests/test_preprocessing.py`에 테스트 추가, docstring에
인용 명시. `src/sers/preprocessing_lab/db.py`의 `ExperimentTracker`로
`experiment.papers`(없으면)와 `experiment.preprocessing_methods`
(`audit_status='pending'`)에 등록.

### 4. 핵심 아이디어 검수 (사람이 하는 단계)
`paper-code-audit` 스킬을 논문 + 3단계 diff에 대해 실행 → claims-vs-code
비교 아티팩트 생성 → **사용자가 읽고 판단**. 승인되면
`experiment.preprocessing_methods.audit_status`를 `'approved'`로 갱신 (아직
`'pending'`이면 뒤 단계로 진행하지 않는다).

### 5. 실행
`@aecd-data-ops`로 `status` 확인 → `@preprocess-lab`을 DB 모드로 호출해
`aecd_platform` 데이터로 전처리 실행 → `@model-bench`에 평가 위임. 결과를
`experiment.preprocessing_runs`/`run_metrics`에 기록하고 사용한 measurement를
`experiment.run_measurements`에 연결,
`@experiment-runner`의 기존 절차로 `docs/ml/experiment.md` +
`logs/experiment_registry.json`도 갱신.

### 6. 비교 (사람이 최종 판단하는 단계)
`@sers-preprocessing-comparator`를 호출해 승인된 방법들을 레퍼런스 인용과
함께 나란히 비교. **채택 여부는 이 에이전트가 아니라 사용자가 결정한다.**

## 금지 사항

- 4단계(사람 검수)를 건너뛰고 바로 실행하지 않는다 — `audit_status='pending'`인
  방법은 5단계로 진행하지 않는다.
- 다운스트림 평가 모델을 새로 만들지 않는다 — 항상 `@model-bench`.
- `preprocessing.py`의 dispatcher를 우회해 signal.py나 독자 스크립트로 구현하지
  않는다.
- `experiment.preprocessing_runs`를 만들면서 `experiment.run_measurements`
  연결을 빠뜨리거나 근거 없는 수치로 기록하지 않는다 — 근거 없는 숫자는 `feedback_data_fabrication.md`
  규칙 위반.
