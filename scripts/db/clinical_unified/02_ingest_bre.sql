-- =====================================================================
-- 02_ingest_bre.sql  —  유방암 (BRE, SMCXD01, n=30)
-- 입력: raw.raw_bre
-- 주의: 기존 clinical_unified를 drop하지 않고 누락 row만 추가한다.
-- =====================================================================

INSERT INTO std.clinical_unified (
    disease_group, protocol, source_file, source_no, provider_code, resource_code,
    solum_label,
    sex, age, height_cm, weight_kg, bmi,
    collection_date, diagnosis_date, diagnosis_name,
    smoking, drinking, smoking_status, drinking_status,
    medical_history, prior_cancer_flag,
    tnm_stage, t_stage, n_stage, cancer_stage_group,
    treatment_info, blood_test_results, blood_cea
)
SELECT
    disease_group, protocol, source_file,
    provider_code::TEXT, provider_code::TEXT, resource_code::TEXT,
    solum_label::TEXT,
    CASE WHEN UPPER(TRIM(sex::TEXT)) IN ('F','여','여자','FEMALE') THEN 'F'
         WHEN UPPER(TRIM(sex::TEXT)) IN ('M','남','남자','MALE') THEN 'M'
         ELSE sex::TEXT END,
    CASE WHEN age::TEXT    ~ '^\d+$'       THEN age::INTEGER    ELSE NULL END,
    CASE
        WHEN height::TEXT ~ '^\d+\.?\d*$'
         AND height::NUMERIC BETWEEN 100 AND 250
        THEN height::NUMERIC ELSE NULL
    END,
    CASE
        WHEN weight::TEXT ~ '^\d+\.?\d*$'
         AND weight::NUMERIC BETWEEN 20 AND 250
        THEN weight::NUMERIC ELSE NULL
    END,
    CASE
        WHEN height::TEXT ~ '^\d+\.?\d*$'
         AND weight::TEXT ~ '^\d+\.?\d*$'
         AND height::NUMERIC BETWEEN 100 AND 250
         AND weight::NUMERIC BETWEEN 20 AND 250
        THEN ROUND(weight::NUMERIC / POWER(height::NUMERIC / 100.0, 2), 1)
        ELSE NULL
    END,
    CASE
        WHEN collection_date::TEXT ~ '^\d{8}$' THEN to_date(collection_date::TEXT, 'YYYYMMDD')
        WHEN collection_date::TEXT ~ '^\d{4}-\d{2}-\d{2}' THEN substring(collection_date::TEXT from '^\d{4}-\d{2}-\d{2}')::DATE
        ELSE NULL
    END,
    CASE
        WHEN diagnosis_date::TEXT ~ '^\d{8}$' THEN to_date(diagnosis_date::TEXT, 'YYYYMMDD')
        WHEN diagnosis_date::TEXT ~ '^\d{4}-\d{2}-\d{2}' THEN substring(diagnosis_date::TEXT from '^\d{4}-\d{2}-\d{2}')::DATE
        ELSE NULL
    END,
    diagnosis_name::TEXT,
    smoking::TEXT, drinking::TEXT,
    CASE
        WHEN smoking::TEXT ~ '안함|비흡연|never' THEN 'never'
        WHEN smoking::TEXT ~ '과거|former|ex' THEN 'former'
        WHEN smoking::TEXT ~ '함|현재|current' THEN 'current'
        ELSE 'unknown'
    END,
    CASE
        WHEN drinking::TEXT ~ '안함|비음주|never' THEN 'never'
        WHEN drinking::TEXT ~ '과거|former|ex' THEN 'former'
        WHEN drinking::TEXT ~ '함|현재|current' THEN 'current'
        ELSE 'unknown'
    END,
    medical_history::TEXT,
    CASE
        WHEN regexp_replace(
            COALESCE(medical_history::TEXT, ''),
            '(?i)breast|invasive ductal carcinoma|ductal carcinoma|Rt\.?breast|Lt\.?breast|BCS',
            '',
            'gi'
        ) ~* '암|cancer|carcinoma|sarcoma|lymphoma|leukemia|HCC'
        THEN TRUE ELSE FALSE
    END,
    NULLIF(pathology_result::TEXT, ''),
    NULLIF(regexp_replace(substring(UPPER(pathology_result::TEXT) from 'P?(TIS|T[0-9][A-Z]*)'), '^P', ''), ''),
    NULLIF(substring(UPPER(pathology_result::TEXT) from '(N[0-9X][A-Z]*)'), ''),
    CASE
        WHEN pathology_result IS NULL OR TRIM(pathology_result::TEXT) = '' OR TRIM(pathology_result::TEXT) = '안함'
            THEN 'unknown'
        WHEN UPPER(pathology_result::TEXT) ~ 'M1|N[23]|T[34]' THEN 'advanced'
        WHEN UPPER(pathology_result::TEXT) ~ 'TIS|T[012]|N[01]' THEN 'early'
        ELSE 'unknown'
    END,
    treatment_info::TEXT, blood_test_results::TEXT,
    CASE
        WHEN substring(blood_test_results::TEXT from '([0-9]+\.?[0-9]*)') ~ '^\d+\.?\d*$'
        THEN substring(blood_test_results::TEXT from '([0-9]+\.?[0-9]*)')::NUMERIC
        ELSE NULL
    END
FROM raw.raw_bre r
WHERE provider_code IS NOT NULL
  AND NOT EXISTS (
      SELECT 1
      FROM std.clinical_unified cu
      WHERE cu.disease_group = r.disease_group
        AND cu.source_file = r.source_file
        AND cu.provider_code = r.provider_code::TEXT
  );
