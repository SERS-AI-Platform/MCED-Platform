-- =====================================================================
-- 02_ingest_controls.sql  —  대조군/비암군 (NOR/DIA/HBP/H.D.)
-- 입력: raw.raw_nor, raw.raw_dia, raw.raw_hbp, raw.raw_hd
-- 주의:
--   - raw.raw_hd의 disease_group은 HD지만 std disease_group은 config 기준 H.D.로 통일한다.
--   - raw 테이블별 컬럼 타입이 조금씩 달라서 to_jsonb로 text 추출 후 안전 cast한다.
--   - 기존 clinical_unified를 drop하지 않고 누락 row만 추가한다.
-- =====================================================================

CREATE OR REPLACE FUNCTION pg_temp.safe_numeric(v TEXT)
RETURNS NUMERIC
LANGUAGE SQL
IMMUTABLE
AS $$
    SELECT CASE
        WHEN NULLIF(regexp_replace(COALESCE(v, ''), '[^0-9.+-]+', '', 'g'), '') ~ '^[+-]?[0-9]+(\.[0-9]+)?$'
        THEN NULLIF(regexp_replace(COALESCE(v, ''), '[^0-9.+-]+', '', 'g'), '')::NUMERIC
        ELSE NULL
    END;
$$;

CREATE OR REPLACE FUNCTION pg_temp.safe_integer(v TEXT)
RETURNS INTEGER
LANGUAGE SQL
IMMUTABLE
AS $$
    SELECT CASE
        WHEN NULLIF(regexp_replace(COALESCE(v, ''), '[^0-9-]+', '', 'g'), '') ~ '^-?[0-9]+$'
        THEN NULLIF(regexp_replace(COALESCE(v, ''), '[^0-9-]+', '', 'g'), '')::INTEGER
        ELSE NULL
    END;
$$;

WITH src AS (
    SELECT 'NOR'::TEXT AS canonical_group, to_jsonb(r) AS j FROM raw.raw_nor r
    UNION ALL
    SELECT 'DIA'::TEXT AS canonical_group, to_jsonb(r) AS j FROM raw.raw_dia r
    UNION ALL
    SELECT 'HBP'::TEXT AS canonical_group, to_jsonb(r) AS j FROM raw.raw_hbp r
    UNION ALL
    SELECT 'H.D.'::TEXT AS canonical_group, to_jsonb(r) AS j FROM raw.raw_hd r
),
norm AS (
    SELECT
        canonical_group,
        j,
        j->>'no' AS no_text,
        j->>'exam_date' AS exam_date_text,
        j->>'birth_year' AS birth_year_text,
        j->>'birth_month' AS birth_month_text,
        j->>'smoking' AS smoking_text,
        j->>'drinking' AS drinking_text
    FROM src
)
INSERT INTO std.clinical_unified (
    disease_group, protocol, hospital_code, hospital_name, source_file, source_no,
    provider_code, resource_code, solum_label,
    sex, age, birth_year, birth_month,
    height_cm, weight_kg, bmi,
    collection_date,
    smoking, smoking_years, smoking_per_day,
    drinking, drinking_days, smoking_status, drinking_status,
    diagnosis_name, medical_history,
    hx_diabetes, hx_hypertension, hx_hyperlipidemia, hx_angina, hx_mi,
    hx_stroke, hx_hepatitis, hx_tb, hx_thyroid, hx_surgery, hx_other,
    prior_cancer_flag,
    bp_systolic, bp_diastolic,
    blood_wbc, blood_rbc, blood_hb, blood_hct, blood_platelet,
    blood_ast, blood_alt, blood_alp, blood_ggt, blood_bilirubin_total,
    blood_protein_total, blood_albumin, blood_bun, blood_creatinine,
    blood_egfr, blood_uric_acid, blood_glucose, blood_hba1c,
    blood_cholesterol, blood_triglyceride, blood_hdl, blood_ldl,
    blood_hscrp, blood_afp, blood_cea, blood_ca125, blood_ca199,
    blood_ca153, blood_psa, blood_free_psa, blood_tsh, blood_ft4,
    blood_insulin,
    urine_sg, urine_ph, urine_protein, urine_glucose, urine_ketone,
    urine_bilirubin, urine_urobilinogen, urine_nitrite,
    urine_occult_blood, urine_leukocyte, urine_color, urine_clarity,
    urine_rbc, urine_wbc
)
SELECT
    n.canonical_group,
    n.j->>'protocol',
    'YPNUH',
    'Yangsan Pusan National University Hospital',
    n.j->>'source_file',
    n.no_text,
    n.j->>'bank_id',
    n.j->>'bank_barcode',
    n.canonical_group || ' ' || n.no_text,
    CASE
        WHEN UPPER(TRIM(n.j->>'sex')) IN ('M','남','남자','MALE') THEN 'M'
        WHEN UPPER(TRIM(n.j->>'sex')) IN ('F','여','여자','FEMALE') THEN 'F'
        ELSE n.j->>'sex'
    END,
    CASE
        WHEN n.exam_date_text ~ '^\d{6,8}$'
         AND n.birth_year_text ~ '^\d{4}$'
        THEN substring(n.exam_date_text from 1 for 4)::INTEGER
             - n.birth_year_text::INTEGER
             - CASE
                   WHEN n.birth_month_text ~ '^\d{1,2}$'
                    AND substring(n.exam_date_text from 5 for 2)::INTEGER < n.birth_month_text::INTEGER
                   THEN 1 ELSE 0
               END
        ELSE NULL
    END,
    pg_temp.safe_integer(n.birth_year_text),
    pg_temp.safe_integer(n.birth_month_text),
    CASE
        WHEN pg_temp.safe_numeric(n.j->>'height') BETWEEN 100 AND 250
        THEN pg_temp.safe_numeric(n.j->>'height') ELSE NULL
    END,
    CASE
        WHEN pg_temp.safe_numeric(n.j->>'weight') BETWEEN 20 AND 250
        THEN pg_temp.safe_numeric(n.j->>'weight') ELSE NULL
    END,
    CASE
        WHEN pg_temp.safe_numeric(n.j->>'bmi') BETWEEN 5 AND 100
        THEN pg_temp.safe_numeric(n.j->>'bmi') ELSE NULL
    END,
    CASE
        WHEN n.exam_date_text ~ '^\d{6}$' THEN to_date(n.exam_date_text || '01', 'YYYYMMDD')
        WHEN n.exam_date_text ~ '^\d{8}$' THEN to_date(n.exam_date_text, 'YYYYMMDD')
        ELSE NULL
    END,
    n.smoking_text,
    pg_temp.safe_integer(n.j->>'smoking_years'),
    pg_temp.safe_integer(n.j->>'smoking_per_day'),
    n.drinking_text,
    n.j->>'drinking_days',
    CASE
        WHEN n.smoking_text ~* '전혀|피운 적이 없다|비흡연|never' THEN 'never'
        WHEN n.smoking_text ~* '지금은 피우지 않는다|과거|former|quit|ex' THEN 'former'
        WHEN n.smoking_text ~* '현재도|지금도|current' THEN 'current'
        ELSE 'unknown'
    END,
    CASE
        WHEN n.drinking_text ~* '아니오|못 마시|처음부터|비음주|never' THEN 'never'
        WHEN n.drinking_text ~* '과거|끊|former|quit|ex' THEN 'former'
        WHEN n.drinking_text ~* '마신다|지금도|current' THEN 'current'
        ELSE 'unknown'
    END,
    n.j->>'category',
    CONCAT_WS('; ', NULLIF(n.j->>'category', ''), NULLIF(n.j->>'hx_other', '')),
    n.j->>'hx_diabetes',
    n.j->>'hx_hypertension',
    n.j->>'hx_hyperlipidemia',
    n.j->>'hx_angina',
    n.j->>'hx_mi',
    n.j->>'hx_stroke',
    n.j->>'hx_hepatitis',
    n.j->>'hx_tb',
    n.j->>'hx_thyroid',
    CASE
        WHEN n.j->>'hx_surgery' IS NULL OR TRIM(n.j->>'hx_surgery') = '' THEN NULL
        WHEN n.j->>'hx_surgery' ~* '아니오|no|none|없음' THEN '아니오'
        ELSE '예'
    END,
    CONCAT_WS('; ', NULLIF(n.j->>'hx_other', ''), NULLIF(n.j->>'hx_surgery', '')),
    EXISTS (
        SELECT 1
        FROM jsonb_each_text(n.j) AS kv(key, value)
        WHERE kv.key LIKE 'hx_cancer_%'
          AND NULLIF(TRIM(kv.value), '') IS NOT NULL
          AND kv.value !~* '^(아니오|no|n|none|없음)$'
    ),
    pg_temp.safe_integer(n.j->>'bp_systolic'),
    pg_temp.safe_integer(n.j->>'bp_diastolic'),
    pg_temp.safe_numeric(n.j->>'blood_wbc'),
    pg_temp.safe_numeric(n.j->>'blood_rbc'),
    pg_temp.safe_numeric(n.j->>'blood_hb'),
    pg_temp.safe_numeric(n.j->>'blood_hct'),
    pg_temp.safe_numeric(n.j->>'blood_platelet'),
    pg_temp.safe_numeric(n.j->>'blood_ast'),
    pg_temp.safe_numeric(n.j->>'blood_alt'),
    pg_temp.safe_numeric(n.j->>'blood_alp'),
    pg_temp.safe_numeric(n.j->>'blood_ggt'),
    pg_temp.safe_numeric(n.j->>'blood_bilirubin_total'),
    pg_temp.safe_numeric(n.j->>'blood_protein_total'),
    pg_temp.safe_numeric(n.j->>'blood_albumin'),
    pg_temp.safe_numeric(n.j->>'blood_bun'),
    pg_temp.safe_numeric(n.j->>'blood_creatinine'),
    pg_temp.safe_numeric(n.j->>'blood_egfr'),
    pg_temp.safe_numeric(n.j->>'blood_uric_acid'),
    pg_temp.safe_numeric(n.j->>'blood_glucose'),
    pg_temp.safe_numeric(n.j->>'blood_hba1c'),
    pg_temp.safe_numeric(n.j->>'blood_cholesterol'),
    pg_temp.safe_numeric(n.j->>'blood_triglyceride'),
    pg_temp.safe_numeric(n.j->>'blood_hdl'),
    pg_temp.safe_numeric(n.j->>'blood_ldl'),
    pg_temp.safe_numeric(n.j->>'blood_hscrp'),
    pg_temp.safe_numeric(n.j->>'blood_afp'),
    pg_temp.safe_numeric(n.j->>'blood_cea'),
    pg_temp.safe_numeric(n.j->>'blood_ca125'),
    pg_temp.safe_numeric(n.j->>'blood_ca199'),
    pg_temp.safe_numeric(n.j->>'blood_ca153'),
    pg_temp.safe_numeric(n.j->>'blood_psa'),
    pg_temp.safe_numeric(n.j->>'blood_free_psa'),
    pg_temp.safe_numeric(n.j->>'blood_tsh'),
    pg_temp.safe_numeric(n.j->>'blood_ft4'),
    pg_temp.safe_numeric(n.j->>'blood_insulin'),
    pg_temp.safe_numeric(n.j->>'urine_sg'),
    pg_temp.safe_numeric(n.j->>'urine_ph'),
    n.j->>'urine_protein',
    n.j->>'urine_glucose',
    n.j->>'urine_ketone',
    n.j->>'urine_bilirubin',
    n.j->>'urine_urobilinogen',
    n.j->>'urine_nitrite',
    n.j->>'urine_occult_blood',
    n.j->>'urine_leukocyte',
    n.j->>'urine_color',
    n.j->>'urine_clarity',
    pg_temp.safe_numeric(n.j->>'urine_rbc'),
    pg_temp.safe_numeric(n.j->>'urine_wbc')
FROM norm n
WHERE n.no_text IS NOT NULL
  AND NOT EXISTS (
      SELECT 1
      FROM std.clinical_unified cu
      WHERE cu.disease_group = n.canonical_group
        AND cu.source_file = n.j->>'source_file'
        AND cu.source_no = n.no_text
  );
