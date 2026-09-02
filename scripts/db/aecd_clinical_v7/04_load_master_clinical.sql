-- Purpose:
--   Transform the staged v7 rows into master.subjects / master.samples and the
--   four clinical.* tables, in a single transaction.
--
-- Scope decision (confirmed with the data owner, 2026-09-02):
--   "기존 유지, 신규만 추가" -- subjects already in the database keep the values
--   they were loaded with in 2026-08. Only subjects that do not yet exist are
--   inserted, and clinical rows are written only for those new subjects. This is
--   why every statement below joins v7_new_subjects rather than the staging table
--   directly: clinical.diagnoses / treatments / observations have no unique
--   constraint, so an unscoped re-run would silently double their contents.
--
-- Grain:
--   A staging row is a *sample*. A subject is (site_id, patient_code). Two CBNUH
--   patients appear twice in v7 under two cancer cohorts (BLC_247/PRO_60 and
--   CRC_186/PAN_78) -- same person, two urine samples, two diagnoses. They
--   collapse to one subject with two samples, which is what the schema models.
--
-- Expected result: see 05_verify.sql.

\set tag 'clinical_v7_20260902'

BEGIN;

DO $$
BEGIN
    IF current_database() <> 'aecd_platform' THEN
        RAISE EXCEPTION 'Wrong database: %.', current_database();
    END IF;
END
$$;

-- ---------------------------------------------------------------------------
-- master.subjects
-- Subject-level attributes are picked from the first staging row that actually
-- has a value (array_agg ordered by "is null" first), so a patient whose second
-- cohort row omits birth_date still gets it from the first.
-- ---------------------------------------------------------------------------
CREATE TEMP TABLE v7_new_subjects ON COMMIT DROP AS
WITH scoped AS (
    SELECT
        site.site_id,
        btrim(stg.patient_code) AS patient_code,
        stg.ingest_row_id,
        NULLIF(btrim(stg.sex), '') AS sex,
        NULLIF(btrim(stg.birth_date), '')::date AS birth_date,
        NULLIF(btrim(stg.smoking_history), '') AS smoking_history,
        NULLIF(btrim(stg.drinking_history), '') AS drinking_history
    FROM ingest.clinical_master_staging AS stg
    JOIN master.sites AS site ON site.site_code = btrim(stg.site_code)
    WHERE stg.source_mapping = :'tag'
),
merged AS (
    SELECT
        site_id,
        patient_code,
        (array_agg(sex ORDER BY (sex IS NULL), ingest_row_id))[1] AS sex,
        (array_agg(birth_date ORDER BY (birth_date IS NULL), ingest_row_id))[1]
            AS birth_date,
        (array_agg(smoking_history
                   ORDER BY (smoking_history IS NULL), ingest_row_id))[1]
            AS smoking_history,
        (array_agg(drinking_history
                   ORDER BY (drinking_history IS NULL), ingest_row_id))[1]
            AS drinking_history
    FROM scoped
    GROUP BY site_id, patient_code
),
inserted AS (
    INSERT INTO master.subjects (
        site_id, patient_code, sex, birth_date, smoking_history, drinking_history
    )
    SELECT
        merged.site_id, merged.patient_code, merged.sex, merged.birth_date,
        merged.smoking_history, merged.drinking_history
    FROM merged
    WHERE NOT EXISTS (
        SELECT 1 FROM master.subjects AS existing
        WHERE existing.site_id = merged.site_id
          AND existing.patient_code = merged.patient_code
    )
    RETURNING subject_id, site_id, patient_code
)
SELECT * FROM inserted;

CREATE INDEX ON v7_new_subjects (site_id, patient_code);

\echo '--- inserted subjects ---'
SELECT count(*) AS inserted_subjects FROM v7_new_subjects;

-- ---------------------------------------------------------------------------
-- Staging rows that belong to a newly created subject. Everything below is
-- driven off this table, so nothing can touch a pre-existing subject.
-- ---------------------------------------------------------------------------
CREATE TEMP TABLE v7_rows ON COMMIT DROP AS
SELECT
    stg.*,
    ns.subject_id
FROM ingest.clinical_master_staging AS stg
JOIN master.sites AS site ON site.site_code = btrim(stg.site_code)
JOIN v7_new_subjects AS ns
  ON ns.site_id = site.site_id
 AND ns.patient_code = btrim(stg.patient_code)
WHERE stg.source_mapping = :'tag';

CREATE INDEX ON v7_rows (ingest_row_id);

-- ---------------------------------------------------------------------------
-- master.samples
-- ---------------------------------------------------------------------------
CREATE TEMP TABLE v7_new_samples ON COMMIT DROP AS
WITH inserted AS (
    INSERT INTO master.samples (
        subject_id, solum_label, sample_type, sample_amount_raw,
        collection_date, receipt_date, sample_timing,
        post_treatment_collection_date
    )
    SELECT
        r.subject_id,
        btrim(r.solum_label),
        NULLIF(btrim(r.sample_type), ''),
        NULLIF(btrim(r.sample_amount), ''),
        NULLIF(btrim(r.collection_date), '')::date,
        NULLIF(btrim(r.receipt_date), '')::date,
        NULLIF(btrim(r.primary_sample_timing_interpretation), ''),
        NULLIF(btrim(r.post_treatment_collection_date), '')::date
    FROM v7_rows AS r
    RETURNING sample_id, subject_id, solum_label
)
SELECT * FROM inserted;

CREATE INDEX ON v7_new_samples (solum_label);

\echo '--- inserted samples ---'
SELECT count(*) AS inserted_samples FROM v7_new_samples;

-- ---------------------------------------------------------------------------
-- clinical.diagnoses -- one row per sample row, so a patient in two cohorts
-- keeps both diagnoses.
-- metastasis_status stays NULL: the workbook's metastasis column mixes 무/유,
-- N/Y and free-text ("from rectal cancer(T3N1M0)"), and turning that into a
-- boolean is a clinical judgement, not a load-time transform. The raw text is
-- preserved in metastasis_raw.
-- ---------------------------------------------------------------------------
WITH inserted AS (
    INSERT INTO clinical.diagnoses (
        subject_id, cohort_group, cancer_type, diagnosis_code, diagnosis_name,
        diagnosis_date, pathology_result, histologic_type, gleason_score,
        grade_group, overall_stage, tnm_stage_raw, t_stage, n_stage, m_stage,
        metastasis_status, metastasis_raw
    )
    SELECT
        r.subject_id,
        NULLIF(btrim(r."group"), ''),
        NULLIF(btrim(r.cancer_type), ''),
        NULLIF(btrim(r.diagnosis_code), ''),
        NULLIF(btrim(r.diagnosis_name), ''),
        NULLIF(btrim(r.diagnosis_date), '')::date,
        NULLIF(btrim(r.pathology_result), ''),
        NULLIF(btrim(r.histologic_type), ''),
        NULLIF(btrim(r.gleason), ''),
        NULLIF(btrim(r."grade group"), '')::integer,
        NULLIF(btrim(r.stage), ''),
        NULLIF(btrim(r.tnm_stage), ''),
        NULLIF(btrim(r.t_stage), ''),
        NULLIF(btrim(r.n_stage), ''),
        NULLIF(btrim(r.m_stage), ''),
        NULL::boolean,
        NULLIF(btrim(r.metastasis), '')
    FROM v7_rows AS r
    RETURNING diagnosis_id
)
SELECT count(*) AS inserted_diagnoses FROM inserted;

-- ---------------------------------------------------------------------------
-- clinical.medical_histories
-- sequence_number is the past_history_N column index. ON CONFLICT DO NOTHING
-- covers the two patients who contribute two rows to the same subject.
-- ---------------------------------------------------------------------------
WITH inserted AS (
    INSERT INTO clinical.medical_histories (
        subject_id, sequence_number, condition_raw
    )
    SELECT
        r.subject_id,
        history.sequence_number,
        btrim(history.condition_raw)
    FROM v7_rows AS r
    CROSS JOIN LATERAL (
        VALUES
            (1, r.past_history_1), (2, r.past_history_2), (3, r.past_history_3),
            (4, r.past_history_4), (5, r.past_history_5), (6, r.past_history_6),
            (7, r.past_history_7), (8, r.past_history_8), (9, r.past_history_9)
    ) AS history(sequence_number, condition_raw)
    WHERE NULLIF(btrim(history.condition_raw), '') IS NOT NULL
    ORDER BY r.ingest_row_id, history.sequence_number
    ON CONFLICT ON CONSTRAINT histories_subject_sequence_unique DO NOTHING
    RETURNING medical_history_id
)
SELECT count(*) AS inserted_medical_histories FROM inserted;

-- ---------------------------------------------------------------------------
-- clinical.treatments -- three kinds from three column groups.
-- 'Atypical small acinar proliferation' is a pathology finding mis-filed in the
-- treatment_info column; the 2026-08 load already excluded it, kept consistent.
-- ---------------------------------------------------------------------------
WITH inserted AS (
    INSERT INTO clinical.treatments (
        subject_id, treatment_type, treatment_name, start_date, end_date,
        details_raw
    )
    SELECT
        r.subject_id,
        'surgery',
        NULLIF(btrim(r.surgery_name), ''),
        NULLIF(btrim(r.surgery_date), '')::date,
        NULL::date,
        NULL::text
    FROM v7_rows AS r
    WHERE NULLIF(btrim(r.surgery_date), '') IS NOT NULL
       OR NULLIF(btrim(r.surgery_name), '') IS NOT NULL

    UNION ALL

    SELECT
        r.subject_id,
        'chemotherapy',
        NULL::text,
        NULLIF(btrim(r.chemo_start_date), '')::date,
        NULLIF(btrim(r.chemo_end_date), '')::date,
        NULL::text
    FROM v7_rows AS r
    WHERE NULLIF(btrim(r.chemo_start_date), '') IS NOT NULL
       OR NULLIF(btrim(r.chemo_end_date), '') IS NOT NULL

    UNION ALL

    SELECT
        r.subject_id,
        'other',
        'medication_record',
        NULL::date,
        NULL::date,
        btrim(r.treatment_info)
    FROM v7_rows AS r
    WHERE NULLIF(btrim(r.treatment_info), '') IS NOT NULL
      AND btrim(r.treatment_info) <> 'Atypical small acinar proliferation'
    RETURNING treatment_id
)
SELECT count(*) AS inserted_treatments FROM inserted;

-- ---------------------------------------------------------------------------
-- clinical.observations -- every non-empty lab / marker / body-measure cell,
-- one row each, linked to both the subject and the sample it was measured with.
-- raw_value keeps the workbook text verbatim; numeric_value / unit are left for
-- a later normalization pass, which is what normalization_status='raw_only' means.
-- ---------------------------------------------------------------------------
WITH inserted AS (
    INSERT INTO clinical.observations (
        subject_id, sample_id, observation_date, panel, code, raw_value,
        numeric_value, text_value, unit, normalization_status
    )
    SELECT
        r.subject_id,
        smp.sample_id,
        NULL::date,
        obs.panel,
        obs.code,
        btrim(obs.raw_value),
        NULL::numeric,
        NULL::text,
        NULL::text,
        'raw_only'
    FROM v7_rows AS r
    JOIN v7_new_samples AS smp ON smp.solum_label = btrim(r.solum_label)
    CROSS JOIN LATERAL (
        VALUES
            ('anthropometry',    'weight_kg',              r.weight_kg),
            ('anthropometry',    'height_cm',              r.height_cm),
            ('anthropometry',    'bmi',                    r.bmi),
            ('cbc',              'wbc',                    r.wbc),
            ('cbc',              'rbc',                    r.rbc),
            ('cbc',              'hb',                     r.hb),
            ('cbc',              'hct',                    r.hct),
            ('cbc',              'platelet',               r.platelet),
            ('cbc',              'neutrophil',             r.neutrophil),
            ('cbc',              'lymphocyte',             r.lymphocyte),
            ('cbc',              'monocyte',               r.monocyte),
            ('cbc',              'eosinophil',             r.eosinophil),
            ('cbc',              'basophil',               r.basophil),
            ('chemistry',        'ast',                    r.ast),
            ('chemistry',        'alt',                    r.alt),
            ('chemistry',        'alp',                    r.alp),
            ('chemistry',        'ggt',                    r.ggt),
            ('chemistry',        'total_bilirubin',        r.total_bilirubin),
            ('chemistry',        'bun',                    r.bun),
            ('chemistry',        'creatinine',             r.creatinine),
            ('chemistry',        'uric_acid',              r.uric_acid),
            ('chemistry',        'glucose',                r.glucose),
            ('chemistry',        'calcium',                r.calcium),
            ('chemistry',        'sodium',                 r.sodium),
            ('chemistry',        'potassium',              r.potassium),
            ('chemistry',        'chloride',               r.chloride),
            ('chemistry',        'cholesterol',            r.cholesterol),
            ('chemistry',        'triglyceride',           r.triglyceride),
            ('chemistry',        'hba1c',                  r.hba1c),
            ('receptor',         'ER',                     r."ER"),
            ('receptor',         'PR',                     r."PR"),
            ('receptor',         'HER2',                   r."HER2"),
            ('tumor_marker',     'CA 15-3',                r."CA 15-3"),
            ('tumor_marker',     'CEA',                    r."CEA"),
            ('tumor_marker',     'ca19_9',                 r.ca19_9),
            ('tumor_marker',     'ca125',                  r.ca125),
            ('tumor_marker',     'afp',                    r.afp),
            ('tumor_marker',     'psa',                    r.psa),
            ('urinalysis',       'ua_sg',                  r.ua_sg),
            ('urinalysis',       'ua_ph',                  r.ua_ph),
            ('urinalysis',       'ua_protein',             r.ua_protein),
            ('urinalysis',       'ua_glucose',             r.ua_glucose),
            ('urinalysis',       'ua_blood(Heme)',         r."ua_blood(Heme)"),
            ('urinalysis',       'ua_ketone',              r.ua_ketone),
            ('urinalysis',       'ua_urobilinogen',        r.ua_urobilinogen),
            ('urinalysis',       'ua_nitrite',             r.ua_nitrite),
            ('urinalysis',       'ua_leukocyte esterase',  r."ua_leukocyte esterase"),
            ('urine_microscopy', 'Microscopy_WBC',         r."Microscopy_WBC"),
            ('urine_microscopy', 'Microscopy_RBC',         r."Microscopy_RBC"),
            ('urine_microscopy', 'Microscopy_Bacteria',    r."Microscopy_Bacteria")
    ) AS obs(panel, code, raw_value)
    WHERE NULLIF(btrim(obs.raw_value), '') IS NOT NULL
    RETURNING observation_id
)
SELECT count(*) AS inserted_observations FROM inserted;

-- ---------------------------------------------------------------------------
-- BNOR_110 -- marked Drop in v7 but already loaded in 2026-08 and carrying
-- spectra, so the subject and sample stay. Record the exclusion on the
-- diagnosis instead, per the data owner's instruction (2026-09-02).
-- ---------------------------------------------------------------------------
WITH updated AS (
    UPDATE clinical.diagnoses AS d
    SET cohort_group = 'Drop'
    FROM master.samples AS smp
    WHERE smp.solum_label = 'BNOR_110'
      AND d.subject_id = smp.subject_id
    RETURNING d.diagnosis_id
)
SELECT count(*) AS bnor_110_marked_drop FROM updated;

COMMIT;
