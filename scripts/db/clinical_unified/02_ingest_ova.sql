-- =====================================================================
-- 02_ingest_ova.sql  —  난소암 (OVA, SMCXD01, IJBPH+SNUH, n=30+40=70)
-- 입력: raw.raw_ova1 (난소암 1.xlsx, IJBPH)
--       raw.raw_ova2 (난소암 2.xlsx, SNUH)
-- =====================================================================

-- --- OVA 1 (IJBPH) --------------------------------------------------
INSERT INTO std.clinical_unified (
    disease_group, protocol, source_file, solum_label, provider_code,
    resource_code,
    sex, age, weight_kg, height_cm,
    collection_date, diagnosis_name, diagnosis_date,
    smoking, drinking, medical_history,
    tnm_stage, treatment_info, blood_test_results
)
SELECT
    disease_group, protocol, source_file,
    solum_label::TEXT, provider_code::TEXT, resource_code::TEXT,
    CASE WHEN UPPER(TRIM(sex::TEXT)) IN ('M','남','남자') THEN 'M'
         WHEN UPPER(TRIM(sex::TEXT)) IN ('F','여','여자') THEN 'F'
         ELSE sex::TEXT END,
    CASE WHEN age::TEXT    ~ '^\d+$'       THEN age::INTEGER    ELSE NULL END,
    CASE WHEN weight::TEXT ~ '^\d+\.?\d*$' THEN weight::NUMERIC ELSE NULL END,
    CASE WHEN height::TEXT ~ '^\d+\.?\d*$' THEN height::NUMERIC ELSE NULL END,
    CASE WHEN LENGTH(TRIM(collection_date::TEXT)) = 8 AND collection_date::TEXT ~ '^\d{8}$'
         THEN to_date(collection_date::TEXT, 'YYYYMMDD')
         WHEN collection_date::TEXT ~ '^\d{4}-' THEN collection_date::TEXT::DATE
         ELSE NULL END,
    diagnosis_name::TEXT,
    CASE WHEN LENGTH(TRIM(diagnosis_date::TEXT)) = 8 AND diagnosis_date::TEXT ~ '^\d{8}$'
         THEN to_date(diagnosis_date::TEXT, 'YYYYMMDD')
         WHEN diagnosis_date::TEXT ~ '^\d{4}-' THEN diagnosis_date::TEXT::DATE
         ELSE NULL END,
    smoking::TEXT, drinking::TEXT, medical_history::TEXT,
    pathology_result::TEXT, treatment_info::TEXT, blood_test_results::TEXT
FROM raw.raw_ova1
WHERE provider_code IS NOT NULL;

-- --- OVA 2 (SNUH) ---------------------------------------------------
INSERT INTO std.clinical_unified (
    disease_group, protocol, source_file, solum_label, provider_code,
    resource_code, lot_no,
    sex, age, weight_kg, height_cm,
    collection_date, diagnosis_name, diagnosis_date,
    smoking, drinking, medical_history, blood_test_results
)
SELECT
    disease_group, protocol, source_file,
    COALESCE(solum_label, solum_label_alt)::TEXT, provider_code::TEXT, resource_code::TEXT,
    lot_no::TEXT,
    CASE WHEN UPPER(TRIM(sex::TEXT)) IN ('M','남','남자') THEN 'M'
         WHEN UPPER(TRIM(sex::TEXT)) IN ('F','여','여자') THEN 'F'
         ELSE sex::TEXT END,
    CASE WHEN age::TEXT    ~ '^\d+$'       THEN age::INTEGER    ELSE NULL END,
    CASE WHEN weight::TEXT ~ '^\d+\.?\d*$' THEN weight::NUMERIC ELSE NULL END,
    CASE WHEN height::TEXT ~ '^\d+\.?\d*$' THEN height::NUMERIC ELSE NULL END,
    CASE WHEN LENGTH(TRIM(collection_date::TEXT)) = 8 AND collection_date::TEXT ~ '^\d{8}$'
         THEN to_date(collection_date::TEXT, 'YYYYMMDD')
         WHEN collection_date::TEXT ~ '^\d{4}-' THEN collection_date::TEXT::DATE
         ELSE NULL END,
    diagnosis_name::TEXT,
    CASE WHEN LENGTH(TRIM(diagnosis_date::TEXT)) = 8 AND diagnosis_date::TEXT ~ '^\d{8}$'
         THEN to_date(diagnosis_date::TEXT, 'YYYYMMDD')
         WHEN diagnosis_date::TEXT ~ '^\d{4}-' THEN diagnosis_date::TEXT::DATE
         ELSE NULL END,
    smoking::TEXT, drinking::TEXT, medical_history::TEXT, blood_test_results::TEXT
FROM raw.raw_ova2
WHERE provider_code IS NOT NULL;
