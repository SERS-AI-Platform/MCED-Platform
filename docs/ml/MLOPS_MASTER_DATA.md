# Clinical–Spectrum Master Data MLOps Runbook

## 목적과 데이터 grain

이 모델은 임상정보와 Raman spectrum을 파일명 순서가 아니라 명시적인 계보로 연결한다.

| 엔터티 | 한 행의 의미 |
|---|---|
| `subjects` | 한 기관의 환자; 병원 가명 환자번호가 없어도 내부 UUID로 존재 |
| `samples` | `solum_label`로 spectrum과 연결되는 한 시점의 물리 검체 |
| `analytical_materials` | 원검체 또는 liquid/powder aliquot |
| `measurements` | 한 technical replicate |
| `clinical_events` | 원 임상자료의 한 행 또는 한 사건 |
| `clinical_observations` | 임상 사건의 한 필드 값 |
| `qc_evaluations` | 특정 규칙·평가 버전의 QC 결과 |
| `sample_labels` | 근거가 연결된 버전별 정답 라벨 |
| `prediction_runs` | 특정 manifest와 모델 버전의 한 실행 |
| `dataset_manifests` | 정확한 measurement·label·QC 조합의 동결 스냅샷 |

Liquid와 Powder는 동일 원검체에서 파생된 별도 `analytical_materials`로 관리한다. CSV 한 파일은 하나의 `measurement`이며 `_ave` 파일은 원 측정이 아니라 파생 artifact다.

## 환자 가명키와 개인정보 경계

임상기관이 제공한 환자번호는 이미 가명키다. 따라서 HMAC-SHA256으로 다시
변환하거나 별도 비밀키를 운용하지 않는다. 값이 있으면
`subjects.patient_id`에 그대로 저장하고 `(site_id, patient_id)`로 동일 기관 내
환자를 연결한다. 병원 환자번호가 없으면 `patient_id`는 `NULL`이며 내부
`subjects.id`만 생성한다.

`samples.solum_label`은 환자번호가 아니라 임상정보와 spectrum 파일을 연결하는
별도 Spectrum 매칭 ID다. 병원 환자번호가 비어 있어도 `solum_label`을
`patient_id`에 복사하지 않는다. 같은 문자열이 우연히 두 컬럼에 있더라도 두
식별자의 의미와 사용 목적은 분리한다.

이 가명키는 여전히 민감정보로 취급한다.

- CLI stdout/stderr, 애플리케이션 로그, MLflow parameter/tag/metric에 기록하지 않는다.
- dataset CSV에는 가명키, source sample code, 원 임상 필드 값, 원 파일 URI를 내보내지 않는다.
- `source_assets.uri`는 원본 위치 추적용 DB 내부 필드다. 접근 권한이 있는 운영자만 조회한다.
- export의 `raw_uri`는 SHA-256 content-addressed raw store 위치만 가리킨다.
- 직접 식별정보가 포함된 원자료는 이 DB에 적재하지 않는다.

## 원본과 불변성

임상·spectrum 원본은 SHA-256으로 식별하고 content-addressed raw store에 보관한다. `source_assets` 한 행이 원 URI, hash, 보존 위치 `raw_uri`를 함께 가진다. 같은 URI의 bytes가 바뀌면 새 버전으로 조용히 덮어쓰지 않고 적재를 실패시킨다.

저장 책임은 다음과 같이 고정한다.

- DB는 환자·검체·measurement 관계, 측정 조건, 원본 URI, SHA-256, `raw_uri`, QC·전처리·모델·manifest 계보를 관리한다.
- Content-addressed raw store는 장비가 생성한 CSV/TXT 원본 bytes를 보존하며 SaMD 원본의 권위 저장소다.
- Parquet/NPZ는 Raman shift–intensity 정렬 배열과 전처리·학습용 feature를 저장하는 재생성 가능한 파생 artifact다.

원래 OneDrive/WSL 절대경로는 provenance인 `source_assets.uri`로만 보존한다. 실행 시 사용하는 `raw_uri`는 프로젝트가 관리하는 content-addressed raw store를 가리켜야 한다. Parquet/NPZ를 생성할 때는 입력 measurement와 원본 hash, preprocessing version, feature schema version을 manifest에 연결하며, 해당 파일을 원본의 대체물로 취급하지 않는다. SQLite에는 spectrum 측정점마다 한 행을 생성하지 않는 것을 기본 원칙으로 한다.

QC, 라벨 근거, prediction, manifest는 append-only다. 규칙·전처리·모델·threshold가 바뀌면 기존 행을 수정하지 않고 새 버전을 추가한다. 기존 임상 운영용 `predictions` 테이블은 호환 projection으로 유지하고, 재현 가능한 MLOps 계보는 `prediction_runs`와 `prediction_input_measurements`를 사용한다.

## 결정론적 matching

자동 matching은 spectrum의 `canonical_source_code`와 같은 site의
`samples.solum_label`이 정확히 일치할 때만 수행한다. `subjects.patient_id`,
파일 순서, Excel 행 순서, fuzzy string, 암종 alias만으로 자동 승인하지 않는다.
`CPAN → PAN`, `YPAN → PAN`, `BLA → BLC`는 검토 metadata이며 identity rewrite가
아니다.

| 상태 | 의미 |
|---|---|
| `matched` | 결정론적 키로 한 대상에 연결됨 |
| `unmatched_clinical` | 임상 사건에 대응하는 spectrum material이 없음 |
| `unmatched_spectrum` | spectrum material에 대응하는 임상 사건이 없음 |
| `ambiguous` | 둘 이상의 대상 또는 기존 연결 충돌 |
| `duplicate` | 외부 inventory 검토에서 중복으로 판정 |

수동 판정은 `match_resolutions`에 추가하며 후보 행을 덮어쓰지 않는다. `sers data reconcile`은 이 상태의 집계만 출력하고 개별 키를 출력하지 않는다.

## immutable dataset export

`dataset_manifests`는 preprocessing, feature schema, decision policy, QC policy와 정확한 measurement·artifact hash·label evidence·QC evaluation을 묶는다. manifest hash는 정렬된 canonical content에서 계산되며 동결 후 수정할 수 없다.

`sers data export-dataset`은 동결된 manifest만 UTF-8-SIG CSV로 내보낸다. 같은 파일이 있으면 기본적으로 실패하며 `--overwrite`를 명시한 경우에만 원자적으로 교체한다. 임시 파일은 최종 경로와 같은 filesystem에 작성하고 `os.replace`로 publish한다.

현재 legacy `sers train`은 외부 학습 script에 인자를 전달하는 compatibility wrapper이며 manifest 입력 계약이 없다. 따라서 `--dataset-manifest`를 임의로 추가하지 않는다. 학습 script가 이 export schema를 공식 입력으로 채택한 뒤 별도 변경으로 연결해야 한다.

## SQLite에서 Fabric으로

논리 모델과 ID/hash 계약은 이전 후에도 유지한다.

| 현재 SQLite/raw store | Microsoft Fabric |
|---|---|
| `subjects`, `samples`, `clinical_events` | Lakehouse Delta tables |
| `measurements`, `qc_evaluations`, `prediction_runs` | Lakehouse Delta tables |
| `dataset_manifests`와 member/evidence tables | Lakehouse Delta tables |
| content-addressed raw files | OneLake Files |
| `source_assets.raw_uri` | OneLake URI |
| 로컬 MLflow SQLite backend | Fabric/관리형 MLflow tracking endpoint |

Delta 적재 시 primary key와 unique/check 제약을 데이터 품질 검증으로 재현하고, append-only 테이블은 overwrite가 아닌 append 작업만 허용한다. `raw_uri`를 바꿀 때도 `source_assets.sha256`은 유지해 동일 bytes임을 검증한다.

## 운영 순서

1. `sers data inventory`로 raw source 수와 quarantine 수를 확인한다.
2. `ingest-clinical-registry --validate-only`로 내장 임상 source 계약을 검증한다.
3. 검증된 임상 registry를 `ingest-clinical-registry`로, spectrum root를 `ingest-spectra`로 적재한다. registry 밖의 단일 원본만 `ingest-clinical`로 적재한다.
4. `match` 실행 후 `reconcile`에서 unmatched/ambiguous/duplicate를 확인한다.
5. 승인된 clinical observation 규칙으로 `build-labels`를 실행한다.
6. QC 평가와 label evidence가 포함된 immutable manifest를 생성한다.
7. `export-dataset`으로 학습 입력을 원자적으로 publish한다.
8. 선택적으로 `--mlflow-db`를 사용해 manifest/version과 집계 수만 로컬 MLflow에 기록한다.
9. 재실행 후 row count와 manifest hash가 동일한지 확인하고 다음 source version을 적재한다.

## 적재 transaction 소유권

공개 적재 API `ingest_clinical_source`, `ingest_clinical_registry`,
`ingest_spectra`와 두 schema initializer는 전달받은 SQLite connection의
transaction을 직접 소유한다. 호출 시 `connection.in_transaction`이 참이면
schema, DB row, raw blob을 변경하기 전에 `ActiveCallerTransactionError`로
실패한다. 따라서 이 API들을 호출자 `BEGIN` 또는 savepoint 안에 중첩하지 않는다.

각 공개 적재 호출은 자체 savepoint에서 전체 batch를 처리한다. 성공하면
savepoint를 release하고, 파싱·DB·raw publish 중 하나라도 실패하면 모든 DB
변경을 rollback하고 새로 만든 미참조 raw blob을 제거한 뒤 예외를 그대로
호출자에게 전달한다. 호출자가 별도 작업과 원자성을 묶어야 한다면 먼저 별도
connection에서 그 작업을 commit하거나 rollback한 뒤 깨끗한 connection으로
적재 API를 호출한다.

## 임상 source 계약과 site cutover

`src/sers/master_data/clinical_inventory.py`가 source-file → protocol mapping의 유일한 권위 원본이다. `config/config.yaml`의 protocol은 cohort discovery metadata이고, `scripts/db/clinical_unified/01_schema.sql`의 map은 기존 SQL migration을 위한 mirror이므로 registry와 다를 때 registry를 기준으로 수정한다.

내장 registry는 파일명·protocol·sheet·물리 header 행·병원 환자번호 필드를
하나의 source 계약으로 고정한다. `SoluM Label`은 이 환자번호 필드 계약에 넣지
않고 별도 `solum_label`로 적재한다. 한 workbook의 모든 configured sheet를 먼저
파싱한 뒤 하나의 savepoint에서 적재하므로 일부 sheet만 성공 상태로 남지 않는다.
재실행 시 source URI, row locator, field name 기반 ID가 같아 row count가 증가하지
않는다.

| 원본 source | canonical site | sheet/header/key 계약 |
|---|---|---|
| 전립선 `SMCXD01_전립선암 임상정보.xlsx` | `CBNUH` | `Sheet1`, header 1, data 3, `NO` |
| 유방 `SMCXD01_유방암.xlsx` | `IJBPH` | `C50 임상정보`, header 1, `제공자bCODE` |
| 난소 1 / 난소 2 | `IJBPH` / `SNUH` | `C56 임상정보` / `난소`, header 1 |
| 폐 1 / 폐 2 / 폐 3 | `SNUH` / `SSMH` / `SNUH` | `폐`; `Sheet1` + lowercase `no`; 폐 3의 네 clinical sheet |
| 정상·당뇨·고혈압·복합질환 | `YPNUH` | 질환별 sheet, header 1, 첫 source 번호 |
| 대장 `SMCXD06_대장암.xlsx` | `CBNUH` | 두 cohort sheet, header 1, non-overlapping `제공자:제공자bCODE` |
| 충북 췌장 `SMCMD06_췌장암.xlsx` / 방광 | `CBNUH` | `CPAN`은 `SMCMD06`, 췌장 provider code / 방광 `분양명단`, physical header 2 |
| SPAN CSV / YPAN CSV / `SMCXD04_CRF_data.xlsx` | `SAMSUNG` / `YONSEI` | SPAN CSV 19개는 `SMCXD02`; YPAN CSV 20개와 YPAN/YNOR workbook은 `SMCXD04`; CSV `SUBJID`, CRF key `스크리닝번호` |
| 보라매 current/history | `BORAMAE` | `BPRO`/`BNOR` protocol alias `SMCXD07`; `Sheet1`, header 1, `patient_code` |

`SMC`와 `CBNU`는 canonical site가 아니다. 기존 DB에서 이 코드로 적재된 행은 전체를 한 기관으로 일괄 변경하면 안 된다. `source_assets.uri`의 정확한 원본 파일명을 위 표와 `scripts/db/clinical_unified/01_schema.sql`의 source-hospital map에 대조해 `CBNUH`, `IJBPH`, `SNUH`, `SSMH`, `YPNUH`로 migration한다. SPAN, YPAN, 보라매는 각각 `SAMSUNG`, `YONSEI`, `BORAMAE`를 유지한다.

protocol migration도 exact source URI를 기준으로 수행한다. `SMCMD06_췌장암.xlsx`에 잘못 기록된 `SMCXD06`만 `SMCMD06`으로 교정하고, 기존 `SPAN_CRF` 19개 source protocol은 `SMCXD02`로, `YPAN_CRF` 20개 source protocol은 `SMCXD04`로 교정한다. YPAN/YNOR는 `SMCXD04`, Boramae BPRO/BNOR current와 history는 `SMCXD07`로 기록한다. cohort alias나 파일·Excel 행 순서만으로 protocol 또는 identity를 일괄 변경하지 않는다.

질환 표기의 migration metadata는 identity rewrite가 아니다.

| source 표기 | canonical review alias | 처리 |
|---|---|---|
| `CPAN` | `PAN` | 모델 group 검토 metadata만 기록 |
| `YPAN` | `PAN` | 모델 group 검토 metadata만 기록 |
| `BLA` | `BLC` | source 표기는 보존하고 canonical review alias 기록 |
| `SPAN` | 없음 | post-operative cohort이므로 `PAN`에 병합 금지 |

legacy `sers data standardize`는 호환을 위한 비관리형 predecessor다. 이 경로가 생성하는 `CPAN → PAN`, `BLA → BLC` rewrite와 파일/Excel 행 순서 기반 산출물은 `source_assets`, matching, label evidence, manifest의 입력으로 금지한다. governed ingestion은 `sers data ingest-clinical-registry`만 사용한다.

`src/sers/reingest_staging.py`와 `scripts/db/clinical_unified/`도 과거 결과 재현 전용이며 canonical `sers data`에서 호출하지 않는다. 전자는 기본 실행을 거부하고 historical reproduction에만 `--allow-legacy-row-order-linkage`를 명시해야 한다. 이 override로 생성한 row-order CRC linkage와 legacy SQL의 `BLA → BLC` rewrite도 governed lineage 입력으로 금지한다.

cutover 전후에는 환자 값을 출력하지 않는 다음 명령으로 계약과 row count를 검증한다.

```bash
sers data ingest-clinical-registry \
  --clinical-root data/clinical_data \
  --validate-only

sers data ingest-clinical-registry \
  --clinical-root data/clinical_data \
  --db data/sers_master.db \
  --raw-store data/raw_store
```

첫 명령은 `sources`, `sheets`, `rows`, `issues` 집계만 출력하며 `issues=0`이어야 한다. 두 번째 명령을 연속 두 번 실행한 뒤 `source_assets`, `clinical_events`, `clinical_observations` 집계가 동일해야 한다. 기존 generic site migration 후에는 다음 집계가 0이어야 한다.

```sql
SELECT COUNT(*)
FROM source_assets AS sa
JOIN sites AS s ON s.id = sa.site_id
WHERE sa.asset_kind = 'clinical'
  AND s.code IN ('SMC', 'CBNU');
```

코어 CLI는 MLflow 없이 동작한다. 로컬 tracking이 필요할 때만 `pip install -e ".[mlops]"`로 optional dependency를 설치한다. tracking URI는 `sqlite:////absolute/path/mlflow.db` 형태로 설정되며 네트워크 registry를 기본 사용하지 않는다.

## 임상 운영 prediction cutover와 backfill

schema v7는 기존 `predictions`를 삭제하거나 history를 합성하지 않는다. 대신
`legacy_prediction_lineage_status`에 각 legacy 결과 JSON hash의
`pending_context`/`linked` 상태를 기록하고,
`operational_prediction_run_links`로 legacy projection과 immutable run을 연결한다.
`pending_context`는 lineage가 없다는 사실을 나타내며 임의 manifest, measurement,
모델 버전을 채우는 placeholder가 아니다.

v3/v4/v5/v6에서 v7로 migration할 때는 보존된 모든 legacy `predictions` 결과를
canonical JSON hash로 식별해 누락된 `pending_context` 행만 idempotent하게
추가한다. migration은 운영 앱 DB에 `prediction_runs`,
`prediction_input_measurements`, `operational_session_links`,
`operational_prediction_run_links`를 준비하지만 검증된 context가 없는
legacy 결과에 연결 행을 합성하지 않는다.
검증된 context가 없는 행은 migration 후에도 pending으로 남는다.

배포 전 migration은 운영 DB 백업 후 다음처럼 실행한다.

```bash
SERS_CLINICAL_DB_PATH=/absolute/path/clinical.db \
  uv run python -c "from scripts.deployment.clinical_db import init_db; init_db()"
sqlite3 /absolute/path/clinical.db "PRAGMA user_version;"
```

두 번째 명령은 `7`을 출력해야 한다. cutover 후 prediction 호출자는 검증된
`DatasetManifestId`, 정확한 `measurement_ids`, model/preprocessing/feature/policy
version을 모두 전달해야 한다.

```python
from scripts.deployment import clinical_db
from sers.master_data.lineage_types import DatasetManifestId
from sers.master_data.operational_lineage import OperationalPredictionContext

context = OperationalPredictionContext(
    manifest_id=DatasetManifestId(verified_manifest_id),
    model_name=verified_model_name,
    model_version=verified_model_version,
    preprocessing_version=verified_preprocessing_version,
    feature_schema_version=verified_feature_schema_version,
    decision_policy_version=verified_decision_policy_version,
    measurement_ids=tuple(verified_measurement_ids),
)
clinical_db.save_prediction_with_lineage(session_id, result, context)
```

이 API는 legacy `predictions` upsert, `prediction_runs` append, 정확한
`prediction_input_measurements`, sample별 `operational_session_links`, run link를
한 SQLite transaction에서 기록한다. 어느 한 insert라도 실패하면 전체가
rollback된다. 같은 inference를 다시 실행하면 legacy projection은 현재값을
유지하지만 prediction run은 새 행으로 append된다.

기존 `save_prediction(session_id, result)`는 호환을 위해 유지하며 결과를
`pending_context`로 남긴다. 운영자가 원본 manifest와 measurement 목록 및 모든
version을 확인한 뒤에만 다음 idempotent API를 실행한다.

```python
pending = clinical_db.get_pending_prediction_lineage()
run_id = clinical_db.backfill_prediction_lineage(pending[0].session_id, context)
```

같은 legacy 결과/context를 다시 backfill하면 기존 run ID를 반환하며 새 run을
만들지 않는다. 이미 linked된 결과에 다른 context를 전달하면 실패한다. 원자료로
확인할 수 없는 pending 행은 pending 상태로 보존한다.

cutover 검증 쿼리는 다음과 같다.

```sql
SELECT status, COUNT(*)
FROM legacy_prediction_lineage_status
GROUP BY status;

SELECT opl.session_id, pr.id, COUNT(pim.measurement_id) AS input_count
FROM operational_prediction_run_links AS opl
JOIN prediction_runs AS pr ON pr.id = opl.prediction_run_id
JOIN prediction_input_measurements AS pim ON pim.prediction_run_id = pr.id
GROUP BY opl.session_id, pr.id
HAVING input_count = 0;

SELECT osl.session_id, COUNT(DISTINCT osl.sample_id) AS linked_samples
FROM operational_session_links AS osl
GROUP BY osl.session_id;

SELECT opl.prediction_run_id
FROM operational_prediction_run_links AS opl
LEFT JOIN prediction_runs AS pr ON pr.id = opl.prediction_run_id
WHERE pr.id IS NULL;
```

정상 cutover에서는 두 번째와 네 번째 쿼리가 0행이다. 첫 번째 쿼리의
`pending_context` 수는 승인된 backfill마다 감소하며, 세 번째 쿼리는 linked
session마다 1개 이상의 sample을 보여야 한다.

## 장애 복구

- 임상 파싱 또는 DB 오류: transaction과 raw blob publish가 함께 롤백됐는지 확인한 뒤 원본을 수정하지 말고 수정본을 새 URI로 등록한다.
- spectrum quarantine: `spectrum_inventory_records.reason_code`를 확인하고 원본 보존 상태에서 별도 수정본을 준비한다.
- matching ambiguity: source key를 로그로 복사하지 말고 접근 통제된 DB 검토 화면에서 판단해 resolution을 append한다.
- export 실패: 최종 CSV가 없고 `.tmp`가 정리됐는지 확인한 뒤 동일 manifest ID로 재실행한다.
- MLflow 미설치: `mlops` extra를 설치하거나 `--mlflow-db` 없이 export한다. MLflow 요청은 export 전에 preflight된다.
- operational dual-write 실패: legacy `predictions`, `prediction_runs`,
  `prediction_input_measurements`, operational link count가 모두 증가하지 않았는지
  확인하고, 누락된 명시적 context를 원자료에서 확인한 뒤 전체 호출을 재실행한다.
- legacy pending backfill: `pending_context`를 임의 버전이나 추정 measurement로
  해소하지 않는다. 확인 불가능한 행은 pending으로 유지하고 접근 통제된
  reconciliation 기록에 조사 상태를 남긴다.
