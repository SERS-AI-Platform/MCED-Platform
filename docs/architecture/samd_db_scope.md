# SaMD Software + DB 개발 범위 (dGMP 대응)

SOLUM Healthcare · SERS 기반 다중암 선별 SaMD

> **정정 이력**: "5) AI 분석 엔진" 섹션은 2026-07-02 세션에서 실제 코드(`scripts/deployment/sers_predict.py`, `clinical_decision.py`, `src/sers/scoring.py`)와 대조해 정합화했다. 나머지 섹션(1~4, 6~9)은 원문 그대로이며 코드 대조 검토는 이번 범위에 포함되지 않았다.

## 1. 전체 개발 범위 재정의

### 제품 관점 개발 범위

```
[라만 장비/벤더 SW]
    ↓ CSV/TXT 생성
[SaMD 소프트웨어]
    1) 로그인/권한
    2) 환자 등록
    3) 스펙트럼 업로드
    4) QC 검사
    5) AI 분석
    6) 결과 해석
    7) 보고서 생성
    8) Audit Trail 기록
    ↓
[DB + Raw File Storage]
```

여기서 라만 장비 자체는 측정 장비이고, SaMD 소프트웨어는 입력된 SERS 스펙트럼을 QC/AI 분석하여 결과를 생성하는 제품 영역으로 잡는 게 맞다. 내부 소프트웨어 명세에서도 CancerX/AECD는 소변 SERS 스펙트럼을 입력받아 딥러닝 알고리즘으로 암/비암 및 암종 확률 벡터, QC 적합 여부를 출력하는 독립형 SaMD로 정리되어 있다.

## 2. DB를 왜 붙이는지

UI/기능은 화면 기능이고, DB는 그 기능들이 규제적으로 증명 가능하도록 연결하는 장부이다. 즉 DB의 역할은 단순 저장이 아니라:

- 누가 로그인했는가
- 어떤 환자를 등록했는가
- 어떤 스펙트럼 파일 5개를 업로드했는가
- 각 파일의 QC 결과는 무엇인가
- QC Pass가 몇 개인가
- 어떤 모델 버전으로 분석했는가
- 환자 단위 SSI, 내부 기준 초과 반복 수, 최종 선별 결과, 암종별 패턴 비교 결과와 해석 수준이 어떻게 나왔는가
- 어떤 보고서가 생성되었는가
- 이 모든 과정이 언제/누구에 의해 수행되었는가

이것을 기록하는 것이다. 내부 회의 요약에서도 NAS 도입 후 파일 해시값, IP, 접근자, 수정 이력, 저장 기간을 문서화해 소프트웨어 GMP 대응 자료로 활용한다는 후속 검토 사항이 있었고, Raw Data 접근 권한과 알고리즘 분석 수치 접근 권한을 분리하는 권한 체계도 Action Item으로 정리되어 있다.

## 3. 시스템 기능 기준 개발 범위 + DB 연결 구조

### 1) 사용자 인터페이스 UI

| 기능 | 화면 기능 | 연결 DB | Audit Trail |
|---|---|---|---|
| 로그인/인증 | ID/PW 로그인 | users, user_sessions | LOGIN_SUCCESS / LOGIN_FAIL |
| 환자 정보 입력 | ID, 나이, 성별, BMI 입력 | patients | PATIENT_CREATE / PATIENT_UPDATE |
| 스펙트럼 업로드 | CSV/TXT 5개 업로드 | spectrum_files | FILE_UPLOAD |
| QC 결과 표시 | Pass/Fail 및 사유 표시 | qc_results | QC_RUN |
| 분석 결과 표시 | 환자 단위 SSI, 내부 기준 초과 반복 수, 최종 선별 결과, 암종별 패턴 비교 결과 표시 | analysis_results | ANALYSIS_RUN |
| 보고서 생성/조회 | PDF/CSV 생성 및 조회 | reports | REPORT_CREATE / REPORT_VIEW |

AECD 소프트웨어 사용을 위해 사용자 인증이 필요하며, 관리자와 임상의 역할에 따라 접근 권한을 적용한다. 환자 ID, 나이, 성별, BMI 입력 후 SERS 스펙트럼 CSV 또는 TXT 파일 5개를 업로드한다.

### 2) 사용자 인증 및 권한관리

기능 범위
- ID/PW 로그인
- 계정 생성/비활성화
- 비밀번호 초기화
- 역할 기반 접근제어 RBAC
- 자동 로그아웃
- 로그인 실패 기록

권한 구조 제안

| Role | 가능 기능 | 제한 기능 |
|---|---|---|
| 관리자 | 계정 관리, 권한 설정, audit 조회 | 분석 결과 임의 수정 불가 |
| 임상의 | 환자 등록, 스펙트럼 업로드, QC 실행, AI 분석, 본인 소유 이력 및 보고서 확인 | 사용자 권한 변경 불가 |
| 진검과 의사 | 환자 정보 입력/조회, 결과지 조회 | Raw 파일 삭제/수정 불가 |

이 구조는 관리자와 임상의의 이력 접근 범위를 분리한다.

DB 테이블

```
users
- user_id
- login_id
- password_hash
- role_id
- name
- department
- is_active
- created_at
- last_login_at

roles
- role_id
- role_name
- description

permissions
- permission_id
- permission_code
- description

role_permissions
- role_id
- permission_id
```

Audit 대상: LOGIN_SUCCESS / LOGIN_FAIL / LOGOUT / USER_CREATE / USER_DISABLE / PASSWORD_RESET / ROLE_CHANGE

여기서 ID 공유 금지는 단순 보안정책이 아니라, 분석 실행자와 보고서 생성자를 추적하기 위한 Audit Trail 전제 조건으로 봐야 한다. 통합 로드맵 문서에서도 Data Integrity는 데이터가 신뢰 가능하고 추적 가능한지 보는 항목이며 Audit Trail 로그, Data flow diagram, 데이터 저장/보존 정책이 필수 산출물로 정리되어 있다.

### 3) 환자 데이터 입력 및 검증

기능 범위
- 환자 ID 입력
- 나이 입력
- 성별 입력
- BMI 입력
- 필수값 검증
- 형식 오류 검증
- 동일 환자/동일 검사 중복 방지
- 환자-검체-검사 연결

DB 테이블

```
patients
- patient_id
- external_patient_code
- age
- sex
- bmi
- created_by
- created_at
- updated_by
- updated_at

specimens
- specimen_id
- patient_id
- specimen_type        -- urine 등
- collection_date
- measurement_date
- device_id
- substrate_lot
- operator_id
- status
```

개발 포인트: 환자 정보 입력 UI는 단순 입력창이 아니라 patient mismatch 방지 장치가 돼야 한다. 예를 들어 스펙트럼 업로드 화면과 결과 화면, PDF 보고서 상단에 환자 ID를 계속 표시해야 한다. 내부 제품 설명서에서도 환자 식별 정보는 모든 단계와 PDF 보고서 상단에 항상 표시되어 환자 혼동을 방지한다고 되어 있다.

### 4) SERS 데이터 처리 파이프라인

기능 범위

```
입력:
- CSV/TXT 파일 5개 업로드

처리:
- 파일 파싱
- 포맷 검증
- 파수 범위 검증
- 신호 강도 QC
- Noise QC
- Saturation QC
- Correlation QC

출력:
- Spectrum별 QC Pass/Fail
- Fail 사유
- 환자 단위 Valid/Invalid
```

내부 IFU에서는 업로드된 5개 스펙트럼을 자동으로 신호강도, 스파이크 잡음, 포화 여부, replicate 간 상관도 기준으로 검사하고, QC 결과를 스펙트럼별 Pass/Fail 및 실패 사유별 건수로 표시한다고 되어 있다. 검사 유효 조건은 QC Pass 3회 이상이면서 강도 부족·스파이크 잡음·포화가 한 건도 없는 경우이다. 이 세 치명적 QC 실패가 있거나 QC Pass가 3회 미만이면 검사 무효이며 AI 추론 및 결과 해석을 진행하지 않는다.

#### 스펙트럼 데이터 저장 책임 분리

SaMD의 스펙트럼 데이터는 DB, raw store, 분석용 배열 저장소의 책임을 분리한다.

| 저장 계층 | 책임 | 포함 정보 | 규제·운영 의미 |
|---|---|---|---|
| DB | 데이터 관계와 계보 관리 | 환자·검체·measurement 연결, 장비, 측정일, replicate, Lot, 공복 여부, 수술 후 여부, 원본 URI, SHA-256, 보존 위치, QC·분석 버전 | 누가 어떤 원본을 어떤 조건과 버전으로 처리했는지 추적하는 Traceability Layer |
| Content-addressed raw store | 실제 원본 스펙트럼 보존 | 장비가 생성한 CSV/TXT의 원본 bytes | 원본 불변성, 중복 제거, hash 재검증 및 감사 증거 |
| Parquet/NPZ | 전처리·학습용 숫자 배열 보관 | 정렬된 Raman shift–intensity 배열, 전처리 결과, feature matrix | 반복 학습·분석 성능을 위한 파생 artifact이며 원본을 대체하지 않음 |

SQLite DB에는 Raman shift–intensity 측정점마다 한 행을 생성하는 것을 기본 설계로 삼지 않는다. 대량의 측정점을 DB 행으로 펼치면 DB 크기, 백업, migration 및 학습 입력 생성 비용이 불필요하게 증가한다. 대신 한 spectrum 파일을 하나의 `measurement`로 등록하고, `source_assets`와 artifact 관계를 통해 실제 파일에 연결한다.

원본 파일은 OneDrive·WSL·사용자 PC의 절대경로를 실행 시점의 읽기 위치로 사용하지 않는다. 원래 발견 경로는 `source_assets.uri`에 provenance로 남기고, SaMD가 실제로 읽는 보존본은 SHA-256 기반 content-addressed raw store의 `raw_uri`로 관리한다. 같은 URI의 bytes가 변경되거나 hash가 일치하지 않으면 기존 원본을 덮어쓰지 않고 적재를 실패시켜야 한다.

Parquet/NPZ는 전처리 및 학습 성능을 위한 재생성 가능한 파생물이다. 각 파생 artifact는 입력 measurement, 원본 SHA-256, preprocessing version, feature schema version 및 생성 시점을 DB manifest에 연결해야 한다. Parquet/NPZ만 존재하거나 원본 raw file과 hash 연결이 끊긴 상태는 완전한 SaMD 계보로 인정하지 않는다.

DB 테이블

```
spectrum_files
- spectrum_file_id
- specimen_id
- replicate_no          -- 1~5
- original_filename
- stored_file_path
- file_hash_sha256
- file_size
- upload_status
- uploaded_by
- uploaded_at

qc_runs
- qc_run_id
- specimen_id
- qc_algorithm_version
- total_files
- pass_count
- fail_count
- overall_qc_status     -- VALID / INVALID
- executed_by
- executed_at

qc_results
- qc_result_id
- qc_run_id
- spectrum_file_id
- intensity_status
- noise_status
- saturation_status
- correlation_status
- overall_status        -- PASS / FAIL
- fail_reason_code
- fail_reason_detail
```

조건 로직

```
if QC_PASS_COUNT >= 3 and CRITICAL_QC_FAILURE_COUNT == 0:
    specimen_status = "VALID_FOR_ANALYSIS"
else:
    specimen_status = "INVALID_REMEASUREMENT_REQUIRED"
```

이 로직의 목적은 잘못된/불충분한 스펙트럼이 AI 모델로 들어가는 것을 차단하는 것이다. 내부 제품 설명서에서도 QC 단계에서 부적합 스펙트럼이 감지되면 분석 단계로 진행할 수 없다는 사용자 안전 설계가 정리되어 있다.

### 5) AI 분석 엔진

기능 범위

```
입력:
- QC 통과 데이터

처리:
- 전처리
- STK-V2 v1.0.0 inference (StackingPredictor: 10개 base model + ElasticNet meta-learner)
- 단일 고정 threshold 적용 (balanced / Youden's J — 3-preset 선택 UI/코드 없음)
- mean SSI three-band rule (QC 통과 반복측정의 모델 신호 평균을 SSI로 변환하여 `<1.0`, `1.0~4.0` 포함, `>4.0`의 세 조치로 판정)
- risk stratification (SSI 3단계: LOW / MODERATE / HIGH)
- 암종 비교 결과 해석 수준 산출 (1순위 비교값·2순위와의 차이 기반)
- explainability — 설계 문서만 존재, 코드 미구현 (개발 예정)

출력:
- SSI (0~10, threshold가 항상 4.0에 매핑되는 구간별 선형 변환)
- 추가 확인 권고/추가 평가 고려/기준 미만 (평균 SSI 3단계 기준)
- 평균 SSI 판정 점수와 암종별 패턴 비교 결과
- 암종 분류 점수(1순위 암종 비교값 × 10)와 해석 수준(높음/중간/낮음)
- Top peaks / contributed peaks — 개발 예정 (미구현)
```

현재 임상 웹앱은 QC 통과 반복측정의 모델 신호값을 평균하여 환자 단위 SSI로 변환하고, 평균 SSI가 4.0을 초과하면 `추가 확인 권고`, 1.0 이상 4.0 이하이면 `추가 평가 고려`, 1.0 미만이면 `기준 미만`으로 표시한다. 반복측정별 암 신호 스펙트럼 수는 참고값으로 보존한다. 추가 확인 권고인 경우에만 유효 반복측정의 암종별 상대 분류값을 평균하고 성별 제약 후 재정규화하여 가장 비슷한 암종 패턴과 상대 패턴 비교 결과를 표시한다.

> **[2026-07-02 코드 검증 정정 사항]**
>
> - **암종 비교 결과 정의**: 화면에는 1순위 암종 비교값을 10점 척도로 환산한 **암종 분류 점수**와 `scripts/deployment/clinical_decision.py`의 `type_confidence()`가 산출하는 **해석 수준**을 함께 표시한다. 해석 수준은 top_prob≥0.50 또는 gap≥0.20이면 **높음**, top_prob≥0.30 또는 gap≥0.10이면 **중간**, 그 외에는 **낮음**이다(gap = 1순위 비교값 − 2순위 비교값). 두 값 모두 SSI, 개인별 확진 가능성, 병기 또는 중증도를 뜻하지 않는 보조 정보이다.
> - **Threshold 정책**: screening/balanced/confirmatory 3-preset 선택 로직이 과거 코드에 남아 있었으나 실제 임상 웹앱에는 노출되지 않았음이 2026-07-01 세션에서 확인되어, 코드 자체를 단일 threshold로 정리했다(`sers_predict.py`, `models/build_production_stacking.py`). production 모델(v1.0.0)의 balanced threshold는 원시 확률 **0.60**이며, 이 값이 SSI 4.0에 매핑된다. 모델 버전이 바뀌면 balanced 값도 재계산해 `threshold_policy_id`를 갱신해야 한다.
> - **Risk Stratification**: STK-V2 nested 5-fold OOF CV(n=1628)로 검증한 3단계를 사용한다 — **LOW** [SSI 0-1), **MODERATE** [SSI 1-4] 포함, **HIGH** (SSI 4-10]. 관측 암 비율: LOW 1.8%(n=388), MODERATE 46.0%(n=50, 95% CI 32.7-59.7%), HIGH 98.2%(n=1190). 관측 비율은 개인의 확진 가능성을 뜻하지 않으며, 평균 SSI 구간은 신규 검사의 최종 선별 조치 문구를 결정한다.
> - **Explainability(Top peaks / contributed peaks)**: 출력 항목으로 나열돼 있으나 **현재 미구현**이다. 존재하는 것은 AACR 논문용 cohort-level SHAP(암종 평균 스펙트럼 vs 전체 평균 스펙트럼)뿐이며, 환자 1건 단위 기여 피크 계산은 설계 문서(`docs/ml/explainability_design_peak_attribution.md`)만 작성되어 있다. IFU에 "제공 기능"으로 표기하려면 구현이 선행되어야 한다.

DB 테이블

```
model_versions
- model_version_id
- model_name
- model_version
- model_file_hash
- threshold_policy_id      -- 단일 balanced threshold (모델 버전별 자체 값, 3-preset 아님)
- qc_policy_id
- risk_band_policy_id      -- [추가] LOW/MODERATE/HIGH 경계값 버전 관리
- release_status
- released_at
- approved_by

analysis_runs
- analysis_run_id
- specimen_id
- qc_run_id
- model_version_id
- analysis_status
- executed_by
- executed_at
- input_data_hash
- processing_config_hash

analysis_results
- analysis_result_id
- analysis_run_id
- ssi_score
- recommendation_result  -- 추가 확인 권고 / 기준 미만 (고정 n_detected >= 3)
- majority_vote_result   -- 호환성/참고 필드; 값은 "n_detected/N_valid", 신규 검사 판정에는 사용하지 않음
- risk_level             -- [추가] LOW / MODERATE / HIGH
- type_score             -- 1순위 암종 비교값을 0~10 척도로 환산한 보조 점수
- type_interpretation    -- 암종 비교 결과 해석 수준(high/medium/low)
- top_cancer_type
- interpretation_status

cancer_probabilities
- probability_id
- analysis_result_id
- cancer_type_code
- probability
- rank_order
```

개발 포인트: AI 결과값만 저장하면 안 되고, 결과를 만든 조건도 같이 저장해야 한다. 반드시 같이 묶어야 하는 것:

```
분석 결과
+ 사용한 raw file ID
+ QC run ID
+ model version
+ threshold policy (단일 balanced 값, 모델 버전별 고정)
+ risk band policy (LOW/MODERATE/HIGH 경계값 버전)
+ execution timestamp
+ executing user
```

이게 없으면 나중에 같은 결과를 재현할 수 없다.

### 6) 결과 해석 및 출력 로직

표시 정보
- 추가 확인 권고/추가 평가 고려/기준 미만
- SSI
- QC 상태
- 암종별 패턴 비교 결과
- 패턴 구분 수준
- Top Peaks
- 보고서 생성일
- 모델 버전
- 검사 유효/무효 여부

내부 직무발명신고서에서도 SERS 스펙트럼 분석 결과를 품질관리 투명성, 정량적 위험도 지수, 다중 암종 확률 분포를 포함한 임상 보고서로 자동 생성하는 시스템으로 설명하고 있으며, 모델 버전별로 사전에 고정 저장된 단일 판정 기준값과 QC 기준을 적용하여 검사 결과의 일관성과 추적성을 확보한다고 되어 있다.

DB 테이블

```
reports
- report_id
- analysis_run_id
- patient_id
- specimen_id
- report_type          -- PDF / CSV
- report_status
- report_file_path
- report_hash_sha256
- generated_by
- generated_at
- viewed_count
```

해석 로직 예시

```
IF QC_STATUS = INVALID:
    결과 표시 금지
    "재측정 권고" 표시
    보고서 생성 제한 또는 INVALID 보고서만 생성

IF QC_STATUS = VALID AND n_detected >= 3:
    "추가 확인 권고" 표시
    가장 비슷한 암종 패턴과 상대 패턴 비교 결과 표시
    임상 전문의 상담 및 필요한 추가 검사 확인 안내

IF QC_STATUS = VALID AND n_detected < 3:
    "기준 미만" 표시
    암종 패턴 비교 결과 숨김
    "암 배제를 의미하지 않는 선별 참고 결과" 표시
```

## 4. Audit Trail DB는 별도 핵심 테이블로 잡아야 함

Audit Trail은 모든 테이블에 컬럼 조금 넣는 수준으로 끝내면 안 되고, 별도 audit_log 테이블로 중앙 기록해야 한다.

Audit Log 테이블 제안

```
audit_logs
- audit_id
- event_time
- event_type
- user_id
- user_role
- patient_id
- specimen_id
- spectrum_file_id
- analysis_run_id
- report_id
- source_ip
- device_name
- action_summary
- before_value_hash
- after_value_hash
- reason_required
- reason_text
- system_version
```

반드시 기록할 이벤트

| 구간 | Event Type | 의미 |
|---|---|---|
| 로그인 | LOGIN_SUCCESS / LOGIN_FAIL | 누가 접속했는지 |
| 환자 | PATIENT_CREATE / UPDATE | 환자 정보 생성/수정 |
| 파일 | FILE_UPLOAD / FILE_HASH_CREATED | 어떤 파일이 들어왔는지 |
| QC | QC_RUN / QC_FAIL / QC_PASS | QC 실행과 결과 |
| 분석 | ANALYSIS_RUN | AI 분석 실행 |
| 결과 | RESULT_CREATED | 결과 생성 |
| 보고서 | REPORT_CREATED / VIEWED / EXPORTED | 보고서 생성/조회/출력 |
| 권한 | ROLE_CHANGE / USER_DISABLED | 권한 변경 |
| 예외 | DELETE_ATTEMPT / OVERRIDE_ATTEMPT | 삭제/우회 시도 |

내부 로드맵에서도 Data Integrity 항목은 "데이터가 누가/언제 어떻게 형성, 수정, 조회됐는지 추적 가능한가"로 표현되어 있고, Audit Trail 로그와 데이터 저장/보존 정책이 필요 산출물로 정리되어 있다.

## 5. Raw File Storage와 DB의 관계

여기서 제일 중요한 구조는 다음과 같다.

```
Raw CSV/TXT 파일 자체
→ NAS / Local App Storage / Immutable Storage에 보관

DB
→ 파일 경로, 파일 해시, 환자/검체 연결, QC/분석/보고서 이력 관리
```

즉, DB에 raw 스펙트럼 전체를 넣는 것이 목적이 아니다. 권장 구조는:

```
/NAS_RAW/2026/PRO/Patient001/
    rep1.csv
    rep2.csv
    rep3.csv
    rep4.csv
    rep5.csv

DB:
    patient_id  = Patient001
    specimen_id = S20260701_001
    file_path   = /NAS_RAW/2026/PRO/Patient001/rep1.csv
    file_hash   = ...
    qc_status   = PASS
```

내부 NAS 거버넌스 문서에서도 파일명 규칙, Raw 데이터 수정 금지, QC 실패 데이터 별도 관리, Sample ID/Patient ID/Cancer Type/Device Lot/Measurement Date/File Path/QC Status를 필수 메타데이터로 정리하고 있다.

## 6. 최종 DB 구조 요약

**Core DB**: users, roles, permissions, role_permissions, user_sessions

**Clinical / Specimen DB**: patients, specimens, devices, substrate_lots

**File / Raw Data DB**: spectrum_files, file_hash_records, storage_locations

**QC DB**: qc_policies, qc_runs, qc_results, qc_fail_reasons

**AI / Model DB**: model_versions, threshold_policies, analysis_runs, analysis_results, cancer_probabilities, explainability_results

**Report DB**: reports, report_templates, report_exports

**Audit / Compliance DB**: audit_logs, data_retention_policies, access_control_logs, change_requests

## 7. 개발 요구사항으로 다시 정리한 버전

아래는 바로 SRS/개발범위에 넣을 수 있는 형태다.

**A. 사용자/권한 모듈**
- REQ-AUTH-001: 시스템은 사용자 ID/PW 기반 로그인 기능을 제공해야 한다.
- REQ-AUTH-002: 시스템은 관리자와 임상의 역할을 구분해야 한다.
- REQ-AUTH-003: 시스템은 역할별 접근 가능한 기능을 제한해야 한다.
- REQ-AUTH-004: 시스템은 로그인 성공/실패, 로그아웃, 권한 변경을 audit log에 기록해야 한다.

**B. 환자/검체 모듈**
- REQ-PAT-001: 시스템은 환자 ID, 나이, 성별, BMI를 입력받아 저장해야 한다.
- REQ-PAT-002: 시스템은 필수값 누락 및 형식 오류를 검증해야 한다.
- REQ-PAT-003: 시스템은 환자와 검체를 1:N 또는 N:N 정책에 따라 연결할 수 있어야 한다.
- REQ-PAT-004: 환자 정보 생성/수정은 audit log에 기록되어야 한다.

**C. 스펙트럼 업로드 모듈**
- REQ-SPEC-001: 시스템은 1개 검사당 CSV/TXT 스펙트럼 파일 5개를 업로드할 수 있어야 한다.
- REQ-SPEC-002: 시스템은 업로드 파일별 hash를 생성하고 저장해야 한다.
- REQ-SPEC-003: 시스템은 파일 경로, 파일명, 업로드 사용자, 업로드 시간을 DB에 저장해야 한다.
- REQ-SPEC-004: 시스템은 raw 파일 수정/삭제를 제한해야 한다.

**D. QC 모듈**
- REQ-QC-001: 시스템은 신호강도, noise, saturation, correlation 기준으로 QC를 수행해야 한다.
- REQ-QC-002: 시스템은 spectrum별 Pass/Fail과 Fail 사유를 표시해야 한다.
- REQ-QC-003: 시스템은 QC Pass 개수가 3개 이상이고 강도 부족·스파이크 잡음·포화가 없을 때만 Valid로 판정해야 한다.
- REQ-QC-004: Invalid인 경우 AI 분석 단계로 진행할 수 없어야 한다.
- REQ-QC-005: QC 실행 결과는 audit log에 기록되어야 한다.

**E. AI 분석 모듈**
- REQ-AI-001: 시스템은 QC 통과 데이터를 AI inference 입력으로 사용해야 한다.
- REQ-AI-002: 시스템은 사용된 모델 버전과 threshold 정책(단일 balanced 값)을 기록해야 한다.
- REQ-AI-003: QC 유효 검사에 한해 시스템은 평균 SSI, 추가 확인 권고/추가 평가 고려/기준 미만, 암종별 패턴 비교 결과, 패턴 구분 수준을 산출해야 한다.
- REQ-AI-004: 시스템은 분석 실행자, 실행 시간, 입력 데이터 ID, 모델 버전을 저장해야 한다.
- REQ-AI-005: 분석 실행은 audit log에 기록되어야 한다.
- REQ-AI-006: [추가] QC 유효 검사에 한해 시스템은 SSI 기반 3단계 Risk Stratification(LOW/MODERATE/HIGH)을 산출하고 결과에 함께 저장해야 한다.
- REQ-AI-007: [추가, 개발 예정] Explainability(기여 피크)는 구현 완료 전까지 IFU/사양서에 "제공 기능"으로 표기하지 않는다.

**F. 결과/보고서 모듈**
- REQ-REP-001: QC 유효 검사는 분석 결과 기반 PDF/CSV 보고서를 생성하고, QC 무효 검사는 QC 상태와 재검 안내만 포함한 보고서를 생성해야 한다.
- REQ-REP-002: 모든 보고서에는 환자 ID, QC 상태, 모델 버전, 생성 시간이 포함되어야 하며, SSI와 분석 결과는 QC 유효 보고서에만 포함되어야 한다.
- REQ-REP-003: 보고서 파일 hash를 생성하고 저장해야 한다.
- REQ-REP-004: 보고서 생성/조회/출력은 audit log에 기록되어야 한다.

## 8. 한 장 슬라이드용 요약

```
SaMD Software + DB 개발 범위

1. 사용자/권한
   - Login, RBAC, role별 화면/기능 제한
   - DB: users, roles, permissions
   - Audit: login, 권한 변경

2. 환자/검체 등록
   - Patient ID, age, sex, BMI, specimen 연결
   - DB: patients, specimens
   - Audit: 환자 생성/수정

3. SERS 파일 업로드
   - CSV/TXT 5개 업로드, hash 생성, raw 저장
   - DB: spectrum_files, storage_locations
   - Audit: 파일 업로드/해시 생성

4. QC
   - intensity, noise, saturation, correlation
   - QC Pass ≥ 3이고 강도 부족·스파이크 잡음·포화가 없을 때만 Valid, 그 외 Invalid
   - DB: qc_runs, qc_results
   - Audit: QC 실행/결과

5. AI 분석
   - STK-V2 v1.0.0 inference, 환자 단위 평균 SSI, 암 신호 스펙트럼 참고 수, 평균 SSI 3단계 최종 결과, 암종별 패턴 비교 결과
   - DB: model_versions, analysis_runs, analysis_results
   - Audit: 분석 실행/모델 버전 기록

6. 보고서
   - PDF/CSV 생성, 조회, export
   - DB: reports
   - Audit: 보고서 생성/조회/출력
```

## 9. 현재 구조에서 가장 중요한 수정점

지금 정리된 기능 목록은 좋고, 여기에 아래 4개가 반드시 추가되어야 한다.

반드시 추가해야 할 것

```
1. spectrum_files 테이블
   → raw 파일 5개를 환자/검체와 연결

2. qc_runs / qc_results 테이블
   → QC 기준, 결과, 실패 사유 저장

3. model_versions / analysis_runs 테이블
   → 어떤 모델과 threshold로 결과가 나왔는지 저장

4. audit_logs 테이블
   → 모든 행위의 자동 기록
```

이 4개가 없으면 화면은 있어도 dGMP 대응 소프트웨어라고 말하기 어렵다. 반대로 이 4개만 제대로 잡으면, 지금 정리한 UI 기능은 바로 IEC 62304 SRS/SDD/V&V 요구사항으로 전환 가능하다. 통합 로드맵 문서에서는 IEC 62304가 SDP, SRS, SDD, V&V 보고서, Traceability Matrix, 변경관리 기록을 요구하는 SW 통제 영역으로 정리되어 있다.

## 최종 결론

DB는 단순 저장소가 아니라, SaMD 결과의 신뢰성을 증명하는 Traceability Layer다.

Raw 파일은 NAS/저장소에 보존하고, DB는 환자–검체–스펙트럼–QC–AI 모델–결과–보고서를 연결하며, Audit Trail은 이 모든 행위가 누가/언제/무엇을 했는지 자동 기록한다.

따라서 개발 범위는 다음과 같이 정의하는 것이 가장 적합하다.

> "AECD SaMD는 환자 정보 입력, SERS 스펙트럼 업로드, QC, AI 분석, 결과 보고서 생성 기능을 제공하며, 모든 입력·처리·출력 과정은 DB와 Audit Trail을 통해 추적 가능하도록 설계한다."

**미해결 항목 (섹션 1~4, 6~9는 이번 세션에서 코드 대조 검토 안 함)**: DB 스키마(약 30개 테이블)는 목표 아키텍처이며, 현재 `scripts/deployment/clinical_db.py`의 실제 스키마(sessions/spectra/reports/users 중심의 단순 구조)와는 상당한 격차가 있다. RBAC 3-role 구조도 `clinical_auth.py`의 실제 구현과 대조 검토가 필요하다. 이 부분은 별도 세션에서 다룰 것.
