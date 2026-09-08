-- Purpose:
--   Post-load state of the platform, and the checks that would catch a partial
--   or duplicated load.

\echo '--- sites ---'
SELECT site_id, site_code, site_name FROM master.sites ORDER BY site_id;

\echo '--- row counts ---'
SELECT
    (SELECT count(*) FROM master.sites)               AS sites,
    (SELECT count(*) FROM master.subjects)            AS subjects,
    (SELECT count(*) FROM master.samples)             AS samples,
    (SELECT count(*) FROM clinical.diagnoses)         AS diagnoses,
    (SELECT count(*) FROM clinical.medical_histories) AS medical_histories,
    (SELECT count(*) FROM clinical.treatments)        AS treatments,
    (SELECT count(*) FROM clinical.observations)      AS observations,
    (SELECT count(*) FROM measurement.measurements)   AS measurements;

\echo '--- subjects and samples per site ---'
SELECT
    site.site_code,
    count(DISTINCT sub.subject_id) AS subjects,
    count(smp.sample_id) AS samples
FROM master.sites AS site
LEFT JOIN master.subjects AS sub ON sub.site_id = site.site_id
LEFT JOIN master.samples AS smp ON smp.subject_id = sub.subject_id
GROUP BY site.site_code
ORDER BY site.site_code;

\echo '--- cohort groups ---'
SELECT cohort_group, count(*) AS diagnoses
FROM clinical.diagnoses
GROUP BY cohort_group
ORDER BY count(*) DESC;

\echo '--- observations per panel ---'
SELECT panel, count(*) AS observations
FROM clinical.observations
GROUP BY panel
ORDER BY count(*) DESC;

\echo '--- treatments per type ---'
SELECT treatment_type, count(*) AS treatments
FROM clinical.treatments
GROUP BY treatment_type
ORDER BY count(*) DESC;

\echo '--- integrity: every staged label reached master.samples ---'
SELECT count(*) AS missing_samples
FROM ingest.clinical_master_staging AS stg
WHERE stg.source_mapping = 'clinical_v7_20260902'
  AND NOT EXISTS (
      SELECT 1 FROM master.samples AS smp
      WHERE smp.solum_label = btrim(stg.solum_label)
  );

\echo '--- integrity: no subject has more than one diagnosis per cohort_group ---'
\echo '      (a duplicated load would show up here)'
SELECT count(*) AS duplicated_diagnosis_groups
FROM (
    SELECT subject_id, cohort_group
    FROM clinical.diagnoses
    GROUP BY subject_id, cohort_group
    HAVING count(*) > 1
) AS dupes;

\echo '--- integrity: spectra still attached to their samples ---'
SELECT count(DISTINCT m.sample_id) AS samples_with_measurements,
       count(*) AS measurement_rows
FROM measurement.measurements AS m;

\echo '--- BNOR_110 ---'
SELECT smp.solum_label, d.cohort_group, count(m.measurement_id) AS measurements
FROM master.samples AS smp
JOIN clinical.diagnoses AS d ON d.subject_id = smp.subject_id
LEFT JOIN measurement.measurements AS m ON m.sample_id = smp.sample_id
WHERE smp.solum_label = 'BNOR_110'
GROUP BY smp.solum_label, d.cohort_group;
