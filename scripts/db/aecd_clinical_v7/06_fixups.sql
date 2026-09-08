-- Purpose:
--   Two corrections to the 2026-09-02 load, applied after the fact. Both are
--   already folded into build_staging_csv.py and 04_load_master_clinical.sql, so
--   a load run from scratch produces this state directly and this file becomes a
--   no-op.
--
-- 1. solum_label spelling.
--    The workbook writes DIA_/HBP_/OVA_/PRO_1-100 as "PRO_ 60" and PRO_101-400
--    as "PRO_101" -- same cohort, two spellings. One sample must have one label,
--    or later spectrum ingests joining on solum_label miss 370 rows silently.
--    The space inside the "H. D._1" prefix is part of the group name; the
--    pattern '_\s+' only touches whitespace *after* the underscore, so those
--    100 labels are untouched.
--
-- 2. age.
--    master.subjects stores birth_date, which the workbook has for only 1400 of
--    2850 rows, while age is present on 2845. Without this, ~1400 patients have
--    no recoverable age anywhere outside the staging table.
--
-- Scope of 2: subject_id > 113 -- exactly the 2736 subjects this load created.
-- The 113 loaded in 2026-08 keep their data untouched, as agreed.

BEGIN;

DO $$
BEGIN
    IF current_database() <> 'aecd_platform' THEN
        RAISE EXCEPTION 'Wrong database: %.', current_database();
    END IF;
END
$$;

WITH updated AS (
    UPDATE master.samples
    SET solum_label = regexp_replace(solum_label, '_\s+', '_')
    WHERE solum_label ~ '_\s+'
    RETURNING sample_id
)
SELECT count(*) AS relabelled_samples FROM updated;

WITH updated AS (
    UPDATE ingest.clinical_master_staging
    SET solum_label = regexp_replace(solum_label, '_\s+', '_')
    WHERE solum_label ~ '_\s+'
    RETURNING ingest_row_id
)
SELECT count(*) AS relabelled_staging_rows FROM updated;

WITH inserted AS (
    INSERT INTO clinical.observations (
        subject_id, sample_id, observation_date, panel, code, raw_value,
        numeric_value, text_value, unit, normalization_status
    )
    SELECT
        sub.subject_id,
        smp.sample_id,
        NULL::date,
        'demographics',
        'age',
        btrim(stg.age),
        NULL::numeric,
        NULL::text,
        NULL::text,
        'raw_only'
    FROM ingest.clinical_master_staging AS stg
    JOIN master.samples AS smp ON smp.solum_label = btrim(stg.solum_label)
    JOIN master.subjects AS sub ON sub.subject_id = smp.subject_id
    WHERE stg.source_mapping = 'clinical_v7_20260902'
      AND sub.subject_id > 113
      AND NULLIF(btrim(stg.age), '') IS NOT NULL
      AND NOT EXISTS (
          SELECT 1 FROM clinical.observations AS existing
          WHERE existing.sample_id = smp.sample_id
            AND existing.panel = 'demographics'
            AND existing.code = 'age'
      )
    RETURNING observation_id
)
SELECT count(*) AS inserted_age_observations FROM inserted;

COMMIT;

\echo '--- labels still holding whitespace after the underscore (must be 0) ---'
SELECT count(*) AS remaining
FROM master.samples
WHERE solum_label ~ '_\s+';

\echo '--- observations per panel ---'
SELECT panel, count(*) AS observations
FROM clinical.observations
GROUP BY panel
ORDER BY count(*) DESC;
