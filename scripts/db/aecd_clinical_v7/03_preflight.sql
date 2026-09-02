-- Purpose:
--   Prove, before any write, that the staged v7 rows satisfy every CHECK and
--   UNIQUE constraint they will meet in master.* / clinical.*.
--
-- How to read the output:
--   Every "violations" column must be 0. Anything else means stop and fix the
--   workbook or the builder script -- a failure inside 04 aborts the whole
--   transaction after doing tens of thousands of rows of work.

\set tag 'clinical_v7_20260902'

\echo '--- 1. site_code resolves to a real site ---'
SELECT count(*) AS unknown_site_violations
FROM ingest.clinical_master_staging AS stg
LEFT JOIN master.sites AS site ON site.site_code = btrim(stg.site_code)
WHERE stg.source_mapping = :'tag' AND site.site_id IS NULL;

\echo '--- 2. NOT NULL / not-blank keys ---'
SELECT
    count(*) FILTER (WHERE NULLIF(btrim(stg.solum_label), '') IS NULL)
        AS blank_solum_label_violations,
    count(*) FILTER (WHERE NULLIF(btrim(stg.patient_code), '') IS NULL)
        AS blank_patient_code_violations
FROM ingest.clinical_master_staging AS stg
WHERE stg.source_mapping = :'tag';

\echo '--- 3. every date column casts cleanly ---'
SELECT column_name, count(*) AS uncastable_violations
FROM (
    SELECT c.column_name, c.value
    FROM ingest.clinical_master_staging AS stg
    CROSS JOIN LATERAL (
        VALUES
            ('birth_date', stg.birth_date),
            ('collection_date', stg.collection_date),
            ('receipt_date', stg.receipt_date),
            ('diagnosis_date', stg.diagnosis_date),
            ('surgery_date', stg.surgery_date),
            ('chemo_start_date', stg.chemo_start_date),
            ('chemo_end_date', stg.chemo_end_date),
            ('post_treatment_collection_date', stg.post_treatment_collection_date)
    ) AS c(column_name, value)
    WHERE stg.source_mapping = :'tag'
      AND NULLIF(btrim(c.value), '') IS NOT NULL
) AS dates
WHERE value !~ '^\d{4}-\d{2}-\d{2}$'
GROUP BY column_name
ORDER BY column_name;

\echo '--- 4. grade group casts to a positive integer ---'
SELECT count(*) AS grade_group_violations
FROM ingest.clinical_master_staging AS stg
WHERE stg.source_mapping = :'tag'
  AND NULLIF(btrim(stg."grade group"), '') IS NOT NULL
  AND (stg."grade group" !~ '^\d+$' OR stg."grade group"::integer <= 0);

\echo '--- 5. chemotherapy end date is not before the start date ---'
SELECT count(*) AS chemo_date_order_violations
FROM ingest.clinical_master_staging AS stg
WHERE stg.source_mapping = :'tag'
  AND NULLIF(btrim(stg.chemo_start_date), '')::date
      > NULLIF(btrim(stg.chemo_end_date), '')::date;

\echo '--- 6. solum_label is unique inside the batch ---'
SELECT count(*) AS duplicate_label_violations
FROM (
    SELECT btrim(solum_label)
    FROM ingest.clinical_master_staging
    WHERE source_mapping = :'tag'
    GROUP BY 1
    HAVING count(*) > 1
) AS dupes;

\echo '--- 7. labels already in master.samples belong to already-known subjects ---'
\echo '      (these rows are skipped by design; the count must equal 112)'
SELECT count(*) AS preexisting_label_rows
FROM ingest.clinical_master_staging AS stg
JOIN master.samples AS smp ON smp.solum_label = btrim(stg.solum_label)
WHERE stg.source_mapping = :'tag';

SELECT count(*) AS preexisting_label_owned_by_other_subject_violations
FROM ingest.clinical_master_staging AS stg
JOIN master.sites AS site ON site.site_code = btrim(stg.site_code)
JOIN master.samples AS smp ON smp.solum_label = btrim(stg.solum_label)
JOIN master.subjects AS sub ON sub.subject_id = smp.subject_id
WHERE stg.source_mapping = :'tag'
  AND (sub.site_id <> site.site_id OR sub.patient_code <> btrim(stg.patient_code));

\echo '--- 8. what the load will create ---'
WITH scoped AS (
    SELECT
        site.site_id,
        btrim(stg.patient_code) AS patient_code,
        btrim(stg.solum_label) AS solum_label
    FROM ingest.clinical_master_staging AS stg
    JOIN master.sites AS site ON site.site_code = btrim(stg.site_code)
    WHERE stg.source_mapping = :'tag'
),
new_subjects AS (
    SELECT DISTINCT site_id, patient_code
    FROM scoped
    WHERE NOT EXISTS (
        SELECT 1 FROM master.subjects AS e
        WHERE e.site_id = scoped.site_id
          AND e.patient_code = scoped.patient_code
    )
)
SELECT
    (SELECT count(*) FROM scoped) AS staged_rows,
    (SELECT count(DISTINCT (site_id, patient_code)) FROM scoped) AS staged_subjects,
    (SELECT count(*) FROM new_subjects) AS subjects_to_insert,
    (SELECT count(*) FROM scoped JOIN new_subjects USING (site_id, patient_code))
        AS samples_to_insert;
