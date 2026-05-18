-- =====================================================================
-- 02_ingest_cpan.sql  —  췌장암 (CPAN, SMCMD06, CBNUH)
-- 입력: raw.raw_cpan (췌장암.xlsx)
-- 주의:
--   - 일부 surgery_date에 "2007-09-07(타병원)" 같은 주석 포함 → substring으로 날짜만 추출
--   - 원본 컬럼에 여분 공백 존재 → 동적 rename으로 통일
--   - YPAN(세브란스) = 별도 raw 테이블로 추후 INSERT, PAN 통합은 뷰/라벨링에서 처리
-- =====================================================================

-- --- 원본 컬럼명 rename (공백/특수문자 제거) ------------------------
DO $$
DECLARE col_name TEXT;
BEGIN
    SELECT column_name INTO col_name
    FROM information_schema.columns
    WHERE table_schema='raw' AND table_name='raw_cpan'
      AND column_name LIKE '%질병%';
    IF col_name IS NOT NULL THEN
        EXECUTE format('ALTER TABLE raw.raw_cpan RENAME COLUMN %I TO medical_history', col_name);
    END IF;
END $$;

DO $$
DECLARE col_name TEXT;
BEGIN
    SELECT column_name INTO col_name
    FROM information_schema.columns
    WHERE table_schema='raw' AND table_name='raw_cpan'
      AND column_name LIKE '%Histologic%';
    IF col_name IS NOT NULL THEN
        EXECUTE format('ALTER TABLE raw.raw_cpan RENAME COLUMN %I TO histologic_type', col_name);
    END IF;
END $$;

-- --- INSERT ---------------------------------------------------------
INSERT INTO std.clinical_unified (
    disease_group, protocol, source_file, source_no, provider_code, lot_no,
    sex, age, birth_year, birth_month, birth_day,
    height_cm, weight_kg,
    collection_date, diagnosis_date, surgery_date, chemo_date,
    medical_history, fasting, treatment_info,
    t_stage, n_stage, m_stage,
    surgery_name, histologic_type
)
SELECT
    disease_group, protocol, source_file,
    no::TEXT, provider_code::TEXT, lot_no,
    CASE WHEN UPPER(TRIM(sex)) IN ('M','남','남자') THEN 'M'
         WHEN UPPER(TRIM(sex)) IN ('F','여','여자') THEN 'F'
         ELSE sex END,
    age::INTEGER,
    birth_year::INTEGER,
    birth_month::INTEGER,
    birth_day::INTEGER,
    height::NUMERIC,
    weight::NUMERIC,
    collection_date::DATE,
    diagnosis_date::DATE,
    CASE WHEN surgery_date ~ '^\d{8}$'            THEN to_date(surgery_date, 'YYYYMMDD')
         WHEN surgery_date ~ '^\d{4}-\d{2}-\d{2}' THEN substring(surgery_date from '^\d{4}-\d{2}-\d{2}')::DATE
         ELSE NULL END,
    CASE WHEN chemo_date ~ '^\d{8}$'              THEN to_date(chemo_date, 'YYYYMMDD')
         WHEN chemo_date ~ '^\d{4}-\d{2}-\d{2}'   THEN substring(chemo_date from '^\d{4}-\d{2}-\d{2}')::DATE
         ELSE NULL END,
    medical_history, fasting, treatment_info,
    t_stage, n_stage, m_stage,
    surgery_name, histologic_type
FROM raw.raw_cpan
WHERE no IS NOT NULL;

-- --- solum_label ----------------------------------------------------
UPDATE std.clinical_unified
SET solum_label = 'CPAN ' || source_no
WHERE disease_group = 'CPAN' AND source_no IS NOT NULL;
