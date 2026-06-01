-- =====================================================================
-- 02_ingest_bla.sql  —  방광암 임상정보 (BLC, SMCXD06, n=300)
-- 입력: raw.raw_bla
-- 주의:
--   - 원본 임상 disease_group은 BLA지만 std disease_group은 config 기준 BLC로 통일한다.
--   - SERS 스펙트럼 그룹명도 BLC이므로 solum_label은 BLC 1..300으로 만든다.
--   - 기존 clinical_unified를 drop하지 않고 누락 row만 추가한다.
-- =====================================================================

INSERT INTO std.clinical_unified (
    disease_group, protocol, hospital_code, hospital_name, source_file, source_no,
    provider_code, solum_label,
    sex, age, birth_year, birth_month, birth_day,
    height_cm, weight_kg, bmi,
    collection_date, diagnosis_date, surgery_date, chemo_date,
    smoking, drinking, smoking_status, drinking_status,
    medical_history, hx_surgery, hx_other, prior_cancer_flag,
    t_stage, n_stage, m_stage, cancer_stage, cancer_stage_group,
    histologic_type, treatment_info,
    bp_systolic, bp_diastolic,
    blood_wbc, blood_rbc, blood_hb, blood_hct, blood_platelet,
    blood_ast, blood_alt, blood_alp, blood_ggt, blood_bilirubin_total,
    blood_bun, blood_creatinine, blood_uric_acid, blood_glucose, blood_hba1c,
    blood_cholesterol, blood_triglyceride, blood_ca125,
    urine_sg, urine_ph, urine_protein, urine_glucose, urine_occult_blood, urine_leukocyte
)
SELECT
    'BLC', r.protocol,
    'CBNUH', 'Chungbuk National University Hospital',
    r.source_file, r.no::TEXT,
    r.provider_code::TEXT,
    'BLC ' ||
        CASE
            WHEN r.no::TEXT LIKE '1기~2기-%'
                THEN substring(r.no::TEXT from '(\d+)$')::INTEGER
            WHEN r.no::TEXT LIKE '3기~4기-%'
                THEN 290 + substring(r.no::TEXT from '(\d+)$')::INTEGER
            ELSE NULL
        END,
    CASE WHEN UPPER(TRIM(r.sex::TEXT)) IN ('M','남','남자','MALE') THEN 'M'
         WHEN UPPER(TRIM(r.sex::TEXT)) IN ('F','여','여자','FEMALE') THEN 'F'
         ELSE r.sex::TEXT END,
    CASE WHEN r.age::TEXT         ~ '^\d+$' THEN r.age::INTEGER         ELSE NULL END,
    CASE WHEN r.birth_year::TEXT  ~ '^\d+$' THEN r.birth_year::INTEGER  ELSE NULL END,
    CASE WHEN r.birth_month::TEXT ~ '^\d+$' THEN r.birth_month::INTEGER ELSE NULL END,
    CASE WHEN r.birth_day::TEXT   ~ '^\d+$' THEN r.birth_day::INTEGER   ELSE NULL END,
    CASE WHEN r.height::TEXT ~ '^\d+\.?\d*$' THEN r.height::NUMERIC ELSE NULL END,
    CASE WHEN r.weight::TEXT ~ '^\d+\.?\d*$' THEN r.weight::NUMERIC ELSE NULL END,
    CASE
        WHEN r.height::TEXT ~ '^\d+\.?\d*$'
         AND r.weight::TEXT ~ '^\d+\.?\d*$'
         AND r.height::NUMERIC > 0
        THEN ROUND(r.weight::NUMERIC / POWER(r.height::NUMERIC / 100.0, 2), 1)
        ELSE NULL
    END,
    CASE
        WHEN r.collection_date::TEXT ~ '^\d{8}$' THEN to_date(r.collection_date::TEXT, 'YYYYMMDD')
        WHEN r.collection_date::TEXT ~ '^\d{4}-\d{2}-\d{2}' THEN substring(r.collection_date::TEXT from '^\d{4}-\d{2}-\d{2}')::DATE
        ELSE NULL
    END,
    CASE
        WHEN r.diagnosis_date::TEXT ~ '^\d{8}$' THEN to_date(r.diagnosis_date::TEXT, 'YYYYMMDD')
        WHEN r.diagnosis_date::TEXT ~ '^\d{4}-\d{2}-\d{2}' THEN substring(r.diagnosis_date::TEXT from '^\d{4}-\d{2}-\d{2}')::DATE
        ELSE NULL
    END,
    CASE
        WHEN substring(r.treatment_info::TEXT from '(\d{8})\s*-?\s*수술') ~ '^\d{8}$'
        THEN to_date(substring(r.treatment_info::TEXT from '(\d{8})\s*-?\s*수술'), 'YYYYMMDD')
        ELSE NULL
    END,
    CASE
        WHEN substring(r.treatment_info::TEXT from '(\d{8})\s*-?\s*항암') ~ '^\d{8}$'
        THEN to_date(substring(r.treatment_info::TEXT from '(\d{8})\s*-?\s*항암'), 'YYYYMMDD')
        ELSE NULL
    END,
    r.smoking::TEXT, r.drinking::TEXT,
    CASE
        WHEN TRIM(r.smoking::TEXT) = '1' THEN 'never'
        WHEN TRIM(r.smoking::TEXT) = '2' THEN 'former'
        WHEN TRIM(r.smoking::TEXT) = '3' THEN 'current'
        ELSE 'unknown'
    END,
    CASE
        WHEN TRIM(r.drinking::TEXT) = '1' THEN 'never'
        WHEN TRIM(r.drinking::TEXT) = '2' THEN 'former'
        WHEN TRIM(r.drinking::TEXT) = '3' THEN 'current'
        ELSE 'unknown'
    END,
    r.medical_history::TEXT,
    CASE
        WHEN r.surgery_history IS NULL OR TRIM(r.surgery_history::TEXT) = '' THEN NULL
        WHEN r.surgery_history::TEXT ~* '없음|아니오|no|none' THEN '아니오'
        ELSE '예'
    END,
    r.surgery_history::TEXT,
    CASE
        WHEN regexp_replace(
            COALESCE(r.medical_history::TEXT, ''),
            '(?i)bladder\s*(cancer|ca|carcinoma|tumou?r)?|방광암|urothelial carcinoma',
            '',
            'gi'
        ) ~* '암|cancer|carcinoma|sarcoma|lymphoma|leukemia|HCC|CA'
        THEN TRUE ELSE FALSE
    END,
    r.t_stage::TEXT, r.n_stage::TEXT, r.m_stage::TEXT,
    CASE
        WHEN r.no::TEXT LIKE '1기~2기-%' THEN 'I-II'
        WHEN r.no::TEXT LIKE '3기~4기-%' THEN 'III-IV'
        ELSE NULL
    END,
    CASE
        WHEN r.no::TEXT LIKE '1기~2기-%' THEN 'early'
        WHEN r.no::TEXT LIKE '3기~4기-%' THEN 'advanced'
        WHEN UPPER(CONCAT_WS('', r.t_stage, r.n_stage, r.m_stage)) ~ 'M1|T[3-9]|N[1-9]' THEN 'advanced'
        WHEN UPPER(CONCAT_WS('', r.t_stage, r.n_stage, r.m_stage)) ~ 'T[12]' THEN 'early'
        ELSE 'unknown'
    END,
    r.histologic_type::TEXT, r.treatment_info::TEXT,
    CASE WHEN r.bp_systolic::TEXT  ~ '^\d+$' THEN r.bp_systolic::INTEGER  ELSE NULL END,
    CASE WHEN r.bp_diastolic::TEXT ~ '^\d+$' THEN r.bp_diastolic::INTEGER ELSE NULL END,
    CASE WHEN r.blood_wbc::TEXT             ~ '^\d+\.?\d*$' THEN r.blood_wbc::NUMERIC             ELSE NULL END,
    CASE WHEN r.blood_rbc::TEXT             ~ '^\d+\.?\d*$' THEN r.blood_rbc::NUMERIC             ELSE NULL END,
    CASE WHEN r.blood_hb::TEXT              ~ '^\d+\.?\d*$' THEN r.blood_hb::NUMERIC              ELSE NULL END,
    CASE WHEN r.blood_hct::TEXT             ~ '^\d+\.?\d*$' THEN r.blood_hct::NUMERIC             ELSE NULL END,
    CASE WHEN r.blood_platelet::TEXT        ~ '^\d+\.?\d*$' THEN r.blood_platelet::NUMERIC        ELSE NULL END,
    CASE WHEN r.blood_ast::TEXT             ~ '^\d+\.?\d*$' THEN r.blood_ast::NUMERIC             ELSE NULL END,
    CASE WHEN r.blood_alt::TEXT             ~ '^\d+\.?\d*$' THEN r.blood_alt::NUMERIC             ELSE NULL END,
    CASE WHEN r.blood_alp::TEXT             ~ '^\d+\.?\d*$' THEN r.blood_alp::NUMERIC             ELSE NULL END,
    CASE WHEN r.blood_ggt::TEXT             ~ '^\d+\.?\d*$' THEN r.blood_ggt::NUMERIC             ELSE NULL END,
    CASE WHEN r.blood_bilirubin_total::TEXT ~ '^\d+\.?\d*$' THEN r.blood_bilirubin_total::NUMERIC ELSE NULL END,
    CASE WHEN r.blood_bun::TEXT             ~ '^\d+\.?\d*$' THEN r.blood_bun::NUMERIC             ELSE NULL END,
    CASE WHEN r.blood_creatinine::TEXT      ~ '^\d+\.?\d*$' THEN r.blood_creatinine::NUMERIC      ELSE NULL END,
    CASE WHEN r.blood_uric_acid::TEXT       ~ '^\d+\.?\d*$' THEN r.blood_uric_acid::NUMERIC       ELSE NULL END,
    CASE WHEN r.blood_glucose::TEXT         ~ '^\d+\.?\d*$' THEN r.blood_glucose::NUMERIC         ELSE NULL END,
    CASE WHEN r.blood_hba1c::TEXT           ~ '^\d+\.?\d*$' THEN r.blood_hba1c::NUMERIC           ELSE NULL END,
    CASE WHEN r.blood_cholesterol::TEXT     ~ '^\d+\.?\d*$' THEN r.blood_cholesterol::NUMERIC     ELSE NULL END,
    CASE WHEN r.blood_triglyceride::TEXT    ~ '^\d+\.?\d*$' THEN r.blood_triglyceride::NUMERIC    ELSE NULL END,
    NULL,
    CASE WHEN r.urine_sg::TEXT              ~ '^\d+\.?\d*$' THEN r.urine_sg::NUMERIC              ELSE NULL END,
    CASE WHEN r.urine_ph::TEXT              ~ '^\d+\.?\d*$' THEN r.urine_ph::NUMERIC              ELSE NULL END,
    r.urine_protein::TEXT, r.urine_glucose::TEXT, r.urine_occult_blood::TEXT, r.urine_leukocyte::TEXT
FROM raw.raw_bla r
WHERE r.no IS NOT NULL
  AND NOT EXISTS (
      SELECT 1
      FROM std.clinical_unified cu
      WHERE cu.disease_group = 'BLC'
        AND cu.source_file = r.source_file
        AND cu.source_no = r.no::TEXT
  );
