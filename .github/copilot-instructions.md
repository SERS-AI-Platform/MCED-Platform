# SERS-AI Copilot Instructions

You are working on SERS-AI — surface-enhanced Raman spectroscopy (SERS) based urine screening pipeline for multi-cancer detection at SOLUM Healthcare.

## Project context
- Path: `/home/user/SERS-AI`
- Two-stage classifier: (1) Cancer Screening (cancer vs non-cancer), (2) Cancer Type ID (subtype)
- Code: `src/sers/` (data/QC pipeline), `models/` (training/evaluation)
- Scripts: `scripts/{pipeline,analysis,deployment,db,qc_validation}/`
- Main data: `results/processed_spectra.csv`, `results/metadata_raw.csv`
- Dashboard: `/home/user/workspace/solum-dashboard/`
- Publications: `publications/aacr/`
- MLflow: SQLite backend (`mlflow.db`)
- Obsidian vault (SSOT for dynamic facts): `/mnt/c/Users/user/OneDrive - solum/바탕 화면/obsidian/01_Projects/SERS-AI/`

## Permanent rules

### 1. Data integrity
- **PAN composition**: PAN = CPAN (충북대 retro, n=70) + YPAN (연세 prosp, n=30) = 100명. **SPAN (삼성 post-op, n=72)은 절대 PAN에 포함 금지** — 스크리닝 부적합
- **감별 피크 출처**: LR coef, SHAP, AACR figure 등 **우리 모델/데이터에서 나온 결과만**. 문헌 피크는 보조 해석용. 데이터 없는 암종(예: GAS)에는 "데이터 미보유"로 표기
- **CSV 인코딩**: 항상 `encoding='utf-8-sig'` (Windows/Excel 한글 BOM)

### 2. Hospital confound disclaimer (2026-04-09 발견)
모든 Cancer Screening (cancer vs control) AUC는 **hospital-confounded** 가능성 있음 — non-cancer는 양산부산대 SMCXD03, cancer는 다른 병원에서 옴. 외부 보고/문서/대시보드 작성 시 cross-hospital 일반화 가정에는 반드시 disclaimer 명시. Cancer Type ID는 대체로 honest biology. 자세한 evidence: vault `Findings/2026-04-09__hospital-confound.md`

### 3. QC 처리 (2026-04-01 발견)
`results/processed_spectra.csv` 재생성 시 `--skip-qc` 사용. QC(RSD<5%, Corr>0.95)는 inference 단계에만 적용. 재생성 후 1,700명/8,500 스펙트럼 전체 데이터. 자세히: vault `Findings/2026-04-01__qc-filtering-issue.md`

### 4. Run 비교 규칙
서로 다른 run을 비교할 때 다음 4가지가 모두 같지 않으면 비교 불가:
- aggregation mode (mean vs medoid — mean이 일반적으로 우월)
- cancer set (5/7/8 class 등)
- non-cancer set (NOR vs NOR+DIA+HBP+HD)
- sample count

8-class (with BLC) 결과는 5-class benchmark와 직접 비교 불가.

### 5. BLC dataset
- Folder: `data/raw_data/11 BLC (299개)/`
- 299 samples (1~300, **#241 missing**)
- Protocol: SMCXD06, Retrospective, 충북대학교병원

### 6. Sync 명령 (변경 후 필수)
- 대시보드 메타 변경: `cd /home/user/workspace/solum-dashboard && python sync_dashboard.py --url ... --name ... --description ... --icon ... --icon-color ... --category ... --tags ...`
- 데이터 변경 (preprocess/train 후): `cd /home/user/workspace/solum-dashboard && python sync_data.py`

### 7. Terminology
- Stage 1 → "Cancer Screening"
- Stage 2 → "Cancer Type ID"
- (Stage 1/2 표현은 사용 금지)

## Work completion convention

작업이 끝나면 마지막에 항상 **"VSCode 확인 파일"** 섹션 포함:
- 새로 생성된 파일 경로 + 1줄 설명
- 수정된 파일 + 무엇이 바뀌었는지
- 결과물(그래프, CSV 등) 포함

## Communication
- 한국어 응답 (코드/식별자는 영문 그대로)
- 모호한 지시는 추측하지 말고 되묻기 (특히 데이터 삭제/수정, 컬럼 정의 매핑, 실험 결과 해석, 외부 공유 자료)
- 라이브러리/프레임워크/API 질문 시 Context7 MCP로 최신 공식 문서 참조 (학습 데이터에 의존 금지)

## Mode prompts (사용자가 명시적 호출)
- `/ceo-report` — 대표님 보고 모드
- `/chairman-report` — 회장님 보고 모드
- `/timeline-update` — Master Timeline 업데이트
- `/research-outputs` — 리서치 산출물 작성 (vault 저장 + 경쟁사 양식)
