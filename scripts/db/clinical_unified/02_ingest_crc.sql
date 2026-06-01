-- =====================================================================
-- 02_ingest_crc.sql  —  대장암 (CRC, SMCXD06, CBNUH, n=300)
-- 입력: raw.raw_col (대장암.xlsx)
-- 주의: 원본 엑셀은 2시트 구성
--       sheet 1 (no 1~270): 조기암  (early)
--       sheet 2 (no 271~300): 진행암 (advanced)
--       → cancer_stage_group은 sheet 구조 기반으로 03_derive.sql에서 설정
-- =====================================================================

-- --- 원본 한글 컬럼명 rename (최초 1회) -----------------------------
-- WHY: 질병 관련 한글 컬럼을 표준 이름으로 통일
DO $$
DECLARE col_name TEXT;
BEGIN
    SELECT column_name INTO col_name
    FROM information_schema.columns
    WHERE table_schema = 'raw' AND table_name = 'raw_col'
      AND column_name LIKE '%질병%';
    IF col_name IS NOT NULL THEN
        EXECUTE format('ALTER TABLE raw.raw_col RENAME COLUMN %I TO medical_history', col_name);
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
    surgery_name, histologic_type, diagnosis_name
)
SELECT
    'CRC', protocol, source_file,
    no::TEXT, provider_code::TEXT, lot_no::TEXT,
    CASE WHEN UPPER(TRIM(sex::TEXT)) IN ('M','남','남자') THEN 'M'
         WHEN UPPER(TRIM(sex::TEXT)) IN ('F','여','여자') THEN 'F'
         ELSE sex::TEXT END,
    CASE WHEN age::TEXT         ~ '^\d+$' THEN age::INTEGER         ELSE NULL END,
    CASE WHEN birth_year::TEXT  ~ '^\d+$' THEN birth_year::INTEGER  ELSE NULL END,
    CASE WHEN birth_month::TEXT ~ '^\d+$' THEN birth_month::INTEGER ELSE NULL END,
    CASE WHEN birth_day::TEXT   ~ '^\d+$' THEN birth_day::INTEGER   ELSE NULL END,
    CASE WHEN height::TEXT ~ '^\d+\.?\d*$' THEN height::NUMERIC ELSE NULL END,
    CASE WHEN weight::TEXT ~ '^\d+\.?\d*$' THEN weight::NUMERIC ELSE NULL END,
    CASE WHEN LENGTH(TRIM(collection_date::TEXT)) = 8 AND collection_date::TEXT ~ '^\d{8}$'
         THEN to_date(collection_date::TEXT, 'YYYYMMDD')
         WHEN collection_date::TEXT ~ '^\d{4}-' THEN collection_date::TEXT::DATE
         ELSE NULL END,
    CASE WHEN LENGTH(TRIM(diagnosis_date::TEXT)) = 8 AND diagnosis_date::TEXT ~ '^\d{8}$'
         THEN to_date(diagnosis_date::TEXT, 'YYYYMMDD')
         WHEN diagnosis_date::TEXT ~ '^\d{4}-' THEN diagnosis_date::TEXT::DATE
         ELSE NULL END,
    CASE WHEN LENGTH(TRIM(surgery_date::TEXT)) = 8 AND surgery_date::TEXT ~ '^\d{8}$'
         THEN to_date(surgery_date::TEXT, 'YYYYMMDD')
         WHEN surgery_date::TEXT ~ '^\d{4}-\d{2}-\d{2}'
         THEN substring(surgery_date::TEXT from '^\d{4}-\d{2}-\d{2}')::DATE
         ELSE NULL END,
    CASE WHEN LENGTH(TRIM(chemo_date::TEXT)) = 8 AND chemo_date::TEXT ~ '^\d{8}$'
         THEN to_date(chemo_date::TEXT, 'YYYYMMDD')
         WHEN chemo_date::TEXT ~ '^\d{4}-\d{2}-\d{2}'
         THEN substring(chemo_date::TEXT from '^\d{4}-\d{2}-\d{2}')::DATE
         ELSE NULL END,
    medical_history::TEXT, fasting::TEXT, treatment_info::TEXT,
    t_stage::TEXT, n_stage::TEXT, m_stage::TEXT,
    surgery_name::TEXT, histologic_type::TEXT, diagnosis_name::TEXT
FROM raw.raw_col
WHERE no IS NOT NULL
  AND NOT EXISTS (
      SELECT 1
      FROM std.clinical_unified cu
      WHERE cu.disease_group = 'CRC'
        AND cu.source_file = raw.raw_col.source_file
        AND cu.source_no = raw.raw_col.no::TEXT
  );

-- --- solum_label ----------------------------------------------------
UPDATE std.clinical_unified
SET solum_label = 'CRC ' || source_no
WHERE disease_group = 'CRC' AND source_no IS NOT NULL;
