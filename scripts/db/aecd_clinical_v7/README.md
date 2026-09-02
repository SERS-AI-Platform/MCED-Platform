# aecd_platform 임상정보 적재 (전체환자_임상정보_정규화_v7.xlsx)

정규화 워크북 v7 → `master.sites/subjects/samples` + `clinical.diagnoses/medical_histories/treatments/observations`.

## 실행

```bash
python scripts/db/aecd_clinical_v7/build_staging_csv.py     # 워크북 → 적재용 CSV
./scripts/db/aecd_clinical_v7/run_load.sh --dry-run         # 전 단계 실행, 04만 롤백
./scripts/db/aecd_clinical_v7/run_load.sh                   # 실제 반영
```

`run_load.sh`는 실행 전에 `pg_dump -n master -n clinical -n ingest`로 백업을 뜬다
(`data/processed/aecd_platform_ingest/backup_*.sql`). `measurement` 스키마는 이 파이프라인이
건드리지 않으므로 백업 대상에서 제외한다.

## 파일

| 파일 | 역할 |
|---|---|
| `site_map.py` | solum_label → 병원(site_code) 매핑. 워크북 `Label definition` 시트의 라벨 범위를 코드로 옮긴 것 |
| `build_staging_csv.py` | 워크북 → `ingest.clinical_master_staging` 적재용 CSV + 제외행/사이트매핑 리포트 |
| `01_sites.sql` | `master.sites`를 병원 단위 8행으로 정리 |
| `02_load_staging.sql` | staging COPY (템플릿 — `__STAGING_CSV__`를 `run_load.sh`가 치환) |
| `03_preflight.sql` | 쓰기 전 제약조건 위반 사전 점검 |
| `04_load_master_clinical.sql` | 실제 변환 적재 (단일 트랜잭션) |
| `05_verify.sql` | 적재 후 상태 및 무결성 확인 |
| `06_fixups.sql` | 2026-09-02 적재 이후 사후 보정 2건. 위 스크립트들에 이미 반영되어 있어 처음부터 다시 돌리면 no-op |

## 결정 사항 (2026-09-02, 데이터 오너 확인)

**적재 범위 — 2,850행.** 워크북 2,898행 중 `solum_label` 없는 40행과 `group='Drop'` 8행을 제외.
제외 목록은 `data/processed/aecd_platform_ingest/*_excluded.csv`.

**site_code = 병원 약어.** 프로토콜 코드(SMCXD01~07, SMCMD06)는 site로 쓸 수 없다 — 한 병원이
여러 프로토콜을 돌린다(CBNUH만 4종). `master.subjects`의 유일성이 `(site_id, patient_code)`이므로
site가 프로토콜 단위가 되면 같은 병원의 같은 환자가 프로토콜마다 다른 subject로 쪼개진다.
프로토콜은 이 스키마에 저장하지 않는다(오너 결정). 파일 단위 프로토콜 정보는
`src/sers/master_data/clinical_source_contracts.py`에 그대로 남아 있다.

기존 `site_code='smcxd07'`(보라매)은 `'BORAMAE'`로 rename. `site_id`는 그대로라 기존 FK 영향 없음.
2026-08에 적재된 staging 113행의 `site_code`도 함께 갱신해 join이 계속 성립하게 했다.

**기존 데이터 — 유지, 신규만 추가.** 2026-08에 들어간 보라매 113 subject는 v5 기준 값을 그대로 둔다.
`04`의 모든 statement가 `v7_new_subjects` 임시 테이블을 경유하는 이유가 이것이다 —
`clinical.diagnoses/treatments/observations`에는 유니크 제약이 없어서 범위를 좁히지 않으면
재실행이 조용히 행을 2배로 만든다.

**BNOR_110.** v7에서 `Drop`이지만 이미 DB에 있고 스펙트럼 측정이 붙어 있어 subject/sample은 유지하고
`clinical.diagnoses.cohort_group`만 `'Drop'`으로 표시한다.

**환자 1명 = subject 1행.** staging 1행은 *검체*다. CBNUH 환자 2명이 두 암종 코호트에 중복 등장한다
(BLC_247/PRO_60, CRC_186/PAN_78 — patient_code·생년월일 동일). 이들은 subject 1행 + sample 2행 +
diagnosis 2행이 된다. 과거력은 `(subject_id, sequence_number)` 유일 제약 때문에 첫 행 것만 남는다
(중복 5셀).

**solum_label 표기 통일.** 워크북이 같은 코호트를 두 가지로 적어놨다 — `PRO_ 60`(1~100)과
`PRO_101`(101~400). DIA_/HBP_/OVA_/PRO_1~100 총 370건의 언더바 뒤 공백을 제거해 통일했다.
그대로 두면 나중에 스펙트럼을 `solum_label`로 join할 때 370건이 조용히 누락된다.
`H. D._1`의 공백은 그룹명 자체의 일부라 건드리지 않는다(정규식이 언더바 *뒤* 공백만 잡는다).

**age는 observations로.** `master.subjects`에는 birth_date만 있는데 워크북의 birth_date는
2,850행 중 1,400행뿐이고 age는 2,845행에 있다. age를 staging에만 두면 약 1,400명은
어디에서도 나이를 복원할 수 없다. `panel='demographics', code='age'`로 적재(2,733건).

**정규화하지 않은 것.**
- `metastasis_status`(boolean)는 NULL. 워크북의 `metastasis`는 무/유, N/Y, 자유서술이 섞여 있어
  boolean 판정은 임상 판단이다. 원문은 `metastasis_raw`에 보존.
- `clinical.observations`는 전부 `normalization_status='raw_only'` — 원문만 `raw_value`에 넣고
  `numeric_value`/`unit`은 후속 정규화 단계로 미룬다. 2026-08 적재와 같은 방식.
- `treatment_info`의 `'Atypical small acinar proliferation'` 1건은 치료가 아니라 병리 소견이라
  2026-08 적재와 동일하게 제외.

## 재실행

`01`은 멱등, `02`는 자기 배치(`source_mapping='clinical_v7_20260902'`)만 지우고 다시 넣으므로 멱등,
`03`은 읽기 전용. `04`는 **멱등이 아니다** — 이미 적재된 subject는 건너뛰지만, 한 번 성공한 뒤
다시 돌리면 `v7_new_subjects`가 비어 아무것도 하지 않는다. 처음부터 다시 하려면 이번 적재분을 지워야 하는데, `master.subjects.subject_id > 113`이 정확히
이번에 만든 2,736 subject다(identity가 순차 발급). `measurement.measurements`가
`master.samples`를 FK로 참조하므로 `pg_dump` 파일을 그대로 replay하는 방식의 복원은 안 된다 —
`clinical.*` → `master.samples` → `master.subjects` 순서로 `subject_id > 113` 범위를 지우고
staging 배치를 지우는 것이 실질적인 되돌리기 경로다.

## 남은 정리 대상

- `cohort_group`이 NULL인 diagnosis 2건 (BNOR_152, BNOR_153 — v7에서 `group`이 비어 있음).
  코호트 필터에 걸리지 않으므로 그룹을 확정하면 채워야 한다.
