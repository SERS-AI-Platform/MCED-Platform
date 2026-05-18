-- =====================================================================
-- 02_ingest_lun.sql  —  폐암 (LUN, n=30+170+100=300)
-- 입력: raw.raw_lun_smcxd01 (폐암 1.xlsx, SNUH, SMCXD01)
--       raw.raw_lun_smcxd06 (폐암 2.xlsx, SSMH, SMCXD06)
--       raw.raw_lun3        (폐암 3.xlsx, SNUH, SMCXD06)
-- =====================================================================

-- --- LUN 1 (SMCXD01, SNUH, 폐암 1.xlsx) -----------------------------
INSERT INTO std.clinical_unified (
    disease_group, protocol, source_file, solum_label, provider_code,
    resource_code, lot_no,
    sex, age, weight_kg, height_cm,
    collection_date, diagnosis_name, diagnosis_date,
    smoking, drinking, medical_history, blood_test_results
)
SELECT
    disease_group, protocol, source_file,
    COALESCE(solum_label, solum_label_alt)::TEXT, provider_code::TEXT,
    resource_code::TEXT, lot_no::TEXT,
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
FROM raw.raw_lun_smcxd01
WHERE provider_code IS NOT NULL;

-- --- LUN 2 (SMCXD06, SSMH, 폐암 2.xlsx) -----------------------------
INSERT INTO std.clinical_unified (
    disease_group, protocol, source_file, source_no, resource_code,
    sex, age, height_cm, weight_kg,
    surgery_date, diagnosis_date,
    tnm_stage, diagnosis_name, cancer_stage, metastasis
)
SELECT
    disease_group, protocol, source_file, no::TEXT, resource_code::TEXT,
    CASE WHEN UPPER(TRIM(sex::TEXT)) IN ('M','남','남자') THEN 'M'
         WHEN UPPER(TRIM(sex::TEXT)) IN ('F','여','여자') THEN 'F'
         ELSE sex::TEXT END,
    CASE WHEN age::TEXT    ~ '^\d+$'       THEN age::INTEGER    ELSE NULL END,
    CASE WHEN height::TEXT ~ '^\d+\.?\d*$' THEN height::NUMERIC ELSE NULL END,
    CASE WHEN weight::TEXT ~ '^\d+\.?\d*$' THEN weight::NUMERIC ELSE NULL END,
    CASE WHEN surgery_date::TEXT ~ '^\d{4}-' THEN surgery_date::TEXT::DATE
         WHEN LENGTH(TRIM(surgery_date::TEXT)) = 8 AND surgery_date::TEXT ~ '^\d{8}$'
            THEN to_date(surgery_date::TEXT, 'YYYYMMDD')
         ELSE NULL END,
    CASE WHEN diagnosis_date::TEXT ~ '^\d{4}-' THEN diagnosis_date::TEXT::DATE
         WHEN LENGTH(TRIM(diagnosis_date::TEXT)) = 8 AND diagnosis_date::TEXT ~ '^\d{8}$'
            THEN to_date(diagnosis_date::TEXT, 'YYYYMMDD')
         ELSE NULL END,
    tnm_stage::TEXT, diagnosis_name::TEXT, cancer_stage::TEXT, metastasis::TEXT
FROM raw.raw_lun_smcxd06
WHERE no IS NOT NULL;

-- --- LUN 3 (SMCXD06, SNUH, 폐암 3.xlsx) — 혈액검사 포함 -----------
INSERT INTO std.clinical_unified (
    disease_group, protocol, source_file, solum_label, provider_code,
    resource_code, lot_no,
    sex, age, height_cm, weight_kg, bmi,
    collection_date, diagnosis_date, diagnosis_name,
    smoking, drinking, medical_history, treatment_info,
    bp_systolic, bp_diastolic,
    pathology_location, histologic_type, cancer_stage, tnm_stage,
    metastasis, differentiation, lvi,
    blood_wbc, blood_rbc, blood_hb, blood_hct, blood_platelet,
    blood_ast, blood_alt, blood_alp, blood_bilirubin_total,
    blood_glucose, blood_creatinine, blood_bun, blood_uric_acid,
    blood_ggt, blood_cholesterol, blood_hba1c
)
SELECT
    disease_group, protocol, source_file,
    solum_label::TEXT, provider_code::TEXT, resource_code::TEXT, lot_no::TEXT,
    CASE WHEN UPPER(TRIM(sex::TEXT)) IN ('M','남','남자') THEN 'M'
         WHEN UPPER(TRIM(sex::TEXT)) IN ('F','여','여자') THEN 'F'
         ELSE sex::TEXT END,
    CASE WHEN age::TEXT    ~ '^\d+$'       THEN age::INTEGER    ELSE NULL END,
    CASE WHEN height::TEXT ~ '^\d+\.?\d*$' THEN height::NUMERIC ELSE NULL END,
    CASE WHEN weight::TEXT ~ '^\d+\.?\d*$' THEN weight::NUMERIC ELSE NULL END,
    CASE WHEN bmi::TEXT    ~ '^\d+\.?\d*$' THEN bmi::NUMERIC    ELSE NULL END,
    CASE WHEN LENGTH(TRIM(collection_date::TEXT)) = 8 AND collection_date::TEXT ~ '^\d{8}$'
         THEN to_date(collection_date::TEXT, 'YYYYMMDD')
         WHEN collection_date::TEXT ~ '^\d{4}-' THEN collection_date::TEXT::DATE
         ELSE NULL END,
    CASE WHEN diagnosis_date::TEXT ~ '^\d{4}-' THEN diagnosis_date::TEXT::DATE
         WHEN LENGTH(TRIM(diagnosis_date::TEXT)) = 8 AND diagnosis_date::TEXT ~ '^\d{8}$'
            THEN to_date(diagnosis_date::TEXT, 'YYYYMMDD')
         ELSE NULL END,
    diagnosis_code::TEXT,
    smoking::TEXT, drinking::TEXT, medical_history::TEXT, treatment_info::TEXT,
    CASE WHEN blood_pressure::TEXT ~ '/' THEN NULLIF(SPLIT_PART(blood_pressure::TEXT,'/',1),'')::INTEGER ELSE NULL END,
    CASE WHEN blood_pressure::TEXT ~ '/' THEN NULLIF(SPLIT_PART(blood_pressure::TEXT,'/',2),'')::INTEGER ELSE NULL END,
    pathology_location::TEXT, histologic_type::TEXT, cancer_stage::TEXT, tnm_stage::TEXT,
    metastasis::TEXT, differentiation::TEXT, lvi::TEXT,
    CASE WHEN blood_wbc::TEXT             ~ '^\d+\.?\d*$' THEN blood_wbc::NUMERIC             ELSE NULL END,
    CASE WHEN blood_rbc::TEXT             ~ '^\d+\.?\d*$' THEN blood_rbc::NUMERIC             ELSE NULL END,
    CASE WHEN blood_hb::TEXT              ~ '^\d+\.?\d*$' THEN blood_hb::NUMERIC              ELSE NULL END,
    CASE WHEN blood_hct::TEXT             ~ '^\d+\.?\d*$' THEN blood_hct::NUMERIC             ELSE NULL END,
    CASE WHEN blood_platelet::TEXT        ~ '^\d+\.?\d*$' THEN blood_platelet::NUMERIC        ELSE NULL END,
    CASE WHEN blood_ast::TEXT             ~ '^\d+\.?\d*$' THEN blood_ast::NUMERIC             ELSE NULL END,
    CASE WHEN blood_alt::TEXT             ~ '^\d+\.?\d*$' THEN blood_alt::NUMERIC             ELSE NULL END,
    CASE WHEN blood_alp::TEXT             ~ '^\d+\.?\d*$' THEN blood_alp::NUMERIC             ELSE NULL END,
    CASE WHEN blood_bilirubin_total::TEXT ~ '^\d+\.?\d*$' THEN blood_bilirubin_total::NUMERIC ELSE NULL END,
    CASE WHEN blood_glucose::TEXT         ~ '^\d+\.?\d*$' THEN blood_glucose::NUMERIC         ELSE NULL END,
    CASE WHEN blood_creatinine::TEXT      ~ '^\d+\.?\d*$' THEN blood_creatinine::NUMERIC      ELSE NULL END,
    CASE WHEN blood_bun::TEXT             ~ '^\d+\.?\d*$' THEN blood_bun::NUMERIC             ELSE NULL END,
    CASE WHEN blood_uric_acid::TEXT       ~ '^\d+\.?\d*$' THEN blood_uric_acid::NUMERIC       ELSE NULL END,
    CASE WHEN blood_ggt::TEXT             ~ '^\d+\.?\d*$' THEN blood_ggt::NUMERIC             ELSE NULL END,
    CASE WHEN blood_cholesterol::TEXT     ~ '^\d+\.?\d*$' THEN blood_cholesterol::NUMERIC     ELSE NULL END,
    CASE WHEN blood_hba1c::TEXT           ~ '^\d+\.?\d*$' THEN blood_hba1c::NUMERIC           ELSE NULL END
FROM raw.raw_lun3
WHERE provider_code IS NOT NULL;
