# clinical_unified — 임상 데이터 통합 파이프라인

raw schema의 각 엑셀별 테이블(`raw.raw_pro`, `raw.raw_ova1`, ..., `raw.raw_cpan`)을
std.clinical_unified 통합 테이블로 정규화하는 SQL 파이프라인.

## 실행 순서

```bash
# DBeaver에서 아래 순서대로 실행
01_schema.sql           # std schema + lookup + DDL (최초 1회)
02_ingest_pro.sql       # PRO  (100)
02_ingest_ova.sql       # OVA  (30 + 40 = 70)
02_ingest_lun.sql       # LUN  (30 + 170 + 100 = 300)
02_ingest_crc.sql       # CRC  (300)
02_ingest_cpan.sql      # CPAN (~)
03_derive.sql           # 파생 컬럼 (hospital, smoking, prior_cancer, stage_group)
99_verify.sql           # 검증 쿼리 (블록 단위 선택 실행)
```

## 현재 커버 범위

| 질환그룹 | 프로토콜 | 병원  | n    | 소스 파일 |
|---------|---------|-------|------|----------|
| PRO     | SMCXD01 | CBNUH | 100  | 전립선암 |
| OVA     | SMCXD01 | IJBPH | 30   | 난소암 1 |
| OVA     | SMCXD01 | SNUH  | 40   | 난소암 2 |
| LUN     | SMCXD01 | SNUH  | 30   | 폐암 1 |
| LUN     | SMCXD06 | SSMH  | 170  | 폐암 2 |
| LUN     | SMCXD06 | SNUH  | 100  | 폐암 3 |
| CRC     | SMCXD06 | CBNUH | 300  | 대장암 (sheet1=early 270, sheet2=advanced 30) |
| CPAN    | SMCMD06 | CBNUH | ~    | 췌장암 |

## 미수행 (TODO)

- [ ] YPAN (세브란스 췌장암) — 도착 후 ingest, 이후 PAN 통합 VIEW
- [ ] BRE (유방암) — 소스 확인 필요
- [ ] BLC (방광암, 299) — raw table 생성 후 ingest
- [ ] NOR, DIA, HBP, HD — 비암종 4종 ingest
- [ ] cancer_stage_group에 `stage_source` 메타컬럼 (sheet_based vs tnm_based) 추가

## 주의사항

- `raw.raw_col`, `raw.raw_cpan`의 **한글 컬럼명**은 여분 공백 포함 → `DO $$ format('%I', ...)` 패턴으로 동적 rename
- CPAN의 `surgery_date`에 `"2007-09-07(타병원)"` 같은 주석 있음 → `substring(..., '^\d{4}-\d{2}-\d{2}')` 로 날짜만 추출
- CRC의 `cancer_stage_group`은 TNM이 아니라 **sheet 구조(source_no ≤ 270)** 기준 (임상 근거)
- PAN = CPAN + YPAN 통합, SPAN(Samsung Pancreatic)은 **절대 포함 금지**
