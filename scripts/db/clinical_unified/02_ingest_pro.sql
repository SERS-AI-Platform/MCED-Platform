-- =====================================================================
-- 02_ingest_pro.sql  —  전립선암 (PRO, SMCXD01, CBNUH, n=100)
-- 입력: raw.raw_pro
-- =====================================================================

INSERT INTO std.clinical_unified (
    disease_group, protocol, source_file, source_no, resource_code, solum_label,
    sex, age, height_cm, weight_kg,
    collection_date, diagnosis_date,
    smoking, drinking, diagnosis_name, fasting,
    medical_history, pathology_location, histologic_type, grade,
    tnm_stage, metastasis, differentiation, lvi
)
SELECT
    disease_group, protocol, source_file,
    no::TEXT, resource_code::TEXT, solum_label::TEXT,
    CASE WHEN UPPER(TRIM(sex::TEXT)) IN ('M','남','남자','MALE')   THEN 'M'
         WHEN UPPER(TRIM(sex::TEXT)) IN ('F','여','여자','FEMALE') THEN 'F'
         ELSE sex::TEXT END,
    CASE WHEN age::TEXT    ~ '^\d+$'        THEN age::INTEGER    ELSE NULL END,
    CASE WHEN height::TEXT ~ '^\d+\.?\d*$'  THEN height::NUMERIC ELSE NULL END,
    CASE WHEN weight::TEXT ~ '^\d+\.?\d*$'  THEN weight::NUMERIC ELSE NULL END,
    CASE WHEN collection_date::TEXT ~ '^\d{4}-' THEN collection_date::DATE ELSE NULL END,
    CASE WHEN diagnosis_date::TEXT  ~ '^\d{4}-' THEN diagnosis_date::DATE  ELSE NULL END,
    smoking::TEXT, drinking::TEXT, diagnosis_name::TEXT, fasting::TEXT,
    medical_history::TEXT, pathology_location::TEXT, histologic_type::TEXT, grade::TEXT,
    tnm_stage::TEXT, metastasis::TEXT, differentiation::TEXT, lvi::TEXT
FROM raw.raw_pro
WHERE no IS NOT NULL;
