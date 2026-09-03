-- 적재 결과 점검용 쿼리 모음. 전부 읽기 전용이라 아무 때나 돌려도 안전하다.
-- 05_verify.sql이 적재 직후 자동 확인용이라면, 이 파일은 사람이 눈으로 훑는 용도다.

\echo '=== 1. 전체 규모 ==='
SELECT
    (SELECT count(*) FROM master.sites)               AS sites,
    (SELECT count(*) FROM master.subjects)            AS subjects,
    (SELECT count(*) FROM master.samples)             AS samples,
    (SELECT count(*) FROM clinical.diagnoses)         AS diagnoses,
    (SELECT count(*) FROM clinical.medical_histories) AS histories,
    (SELECT count(*) FROM clinical.treatments)        AS treatments,
    (SELECT count(*) FROM clinical.observations)      AS observations,
    (SELECT count(*) FROM measurement.measurements)   AS measurements;

\echo '=== 2. 병원별 / 코호트별 진단 수 (워크북 group 분포와 대조) ==='
-- 주의: clinical.diagnoses는 subject에 붙어 있고 sample에는 붙어 있지 않다.
-- 그래서 samples를 join해서 세면 두 코호트에 걸친 환자 2명(질의 10) 때문에
-- 숫자가 부풀려진다. 워크북의 group 분포와 맞춰볼 때는 이 형태로 센다.
-- 검체 단위로 세고 싶으면 clinical.uti_indicators.cancer_group을 쓴다 -- 그쪽은
-- 라벨 접두사로 중복암 환자의 진단을 검체별로 갈라놓았다.
SELECT
    site.site_code,
    d.cohort_group,
    count(*) AS diagnoses
FROM clinical.diagnoses AS d
JOIN master.subjects AS sub ON sub.subject_id = d.subject_id
JOIN master.sites AS site   ON site.site_id = sub.site_id
GROUP BY site.site_code, d.cohort_group
ORDER BY site.site_code, count(*) DESC;

\echo '=== 3. 라벨 접두사별 개수 (워크북 라벨 범위와 대조) ==='
SELECT
    substring(solum_label FROM '^[A-Za-z. ]+?(?=_)') AS prefix,
    count(*)     AS samples,
    min(split_part(solum_label, '_', 2)::int) AS min_no,
    max(split_part(solum_label, '_', 2)::int) AS max_no
FROM master.samples
GROUP BY 1
ORDER BY 1;

\echo '=== 4. 환자 1명 전체 드릴다운 (라벨만 바꿔서 재사용) ==='
\set label 'BLC_1'
SELECT site.site_code, sub.patient_code, sub.sex, sub.birth_date,
       smp.solum_label, smp.collection_date, smp.sample_timing,
       d.cohort_group, d.diagnosis_name, d.overall_stage, d.t_stage, d.n_stage, d.m_stage
FROM master.samples AS smp
JOIN master.subjects AS sub ON sub.subject_id = smp.subject_id
JOIN master.sites AS site   ON site.site_id = sub.site_id
LEFT JOIN clinical.diagnoses AS d ON d.subject_id = sub.subject_id
WHERE smp.solum_label = :'label';

SELECT panel, code, raw_value
FROM clinical.observations AS o
JOIN master.samples AS smp ON smp.sample_id = o.sample_id
WHERE smp.solum_label = :'label'
ORDER BY panel, code;

SELECT sequence_number, condition_raw
FROM clinical.medical_histories AS h
JOIN master.samples AS smp ON smp.subject_id = h.subject_id
WHERE smp.solum_label = :'label'
ORDER BY sequence_number;

SELECT treatment_type, treatment_name, start_date, end_date, details_raw
FROM clinical.treatments AS t
JOIN master.samples AS smp ON smp.subject_id = t.subject_id
WHERE smp.solum_label = :'label'
ORDER BY treatment_type;

\echo '=== 5. 이상 징후 (전부 0이어야 정상) ==='
SELECT
    (SELECT count(*) FROM ingest.clinical_master_staging AS stg
      WHERE stg.source_mapping = 'clinical_v7_20260902'
        AND NOT EXISTS (SELECT 1 FROM master.samples AS s
                        WHERE s.solum_label = btrim(stg.solum_label)))
        AS staging_rows_without_sample,
    (SELECT count(*) FROM (SELECT subject_id, cohort_group FROM clinical.diagnoses
                           GROUP BY 1,2 HAVING count(*) > 1) AS x)
        AS duplicated_diagnoses,
    (SELECT count(*) FROM (SELECT sample_id, panel, code FROM clinical.observations
                           WHERE sample_id IS NOT NULL
                           GROUP BY 1,2,3 HAVING count(*) > 1) AS x)
        AS duplicated_observations,
    (SELECT count(*) FROM master.samples WHERE solum_label ~ '_\s+')
        AS labels_with_stray_space,
    (SELECT count(*) FROM master.samples AS s
      WHERE NOT EXISTS (SELECT 1 FROM master.subjects AS u
                        WHERE u.subject_id = s.subject_id))
        AS orphan_samples,
    (SELECT count(*) FROM measurement.measurements AS m
      WHERE m.sample_id IS NOT NULL
        AND NOT EXISTS (SELECT 1 FROM master.samples AS s
                        WHERE s.sample_id = m.sample_id))
        AS orphan_measurements;

\echo '=== 6. 워크북 원본과의 대사: staging 대비 적재 누락 ==='
SELECT
    count(*) FILTER (WHERE smp.sample_id IS NOT NULL) AS staged_and_loaded,
    count(*) FILTER (WHERE smp.sample_id IS NULL)     AS staged_but_missing
FROM ingest.clinical_master_staging AS stg
LEFT JOIN master.samples AS smp ON smp.solum_label = btrim(stg.solum_label)
WHERE stg.source_mapping = 'clinical_v7_20260902';

\echo '=== 7. 스펙트럼이 붙어 있는 검체 (아직 보라매 113건뿐이어야 정상) ==='
SELECT smp.solum_label, count(m.measurement_id) AS points
FROM measurement.measurements AS m
JOIN master.samples AS smp ON smp.sample_id = m.sample_id
GROUP BY smp.solum_label
ORDER BY smp.solum_label
LIMIT 10;

SELECT count(DISTINCT sample_id) AS samples_with_spectra
FROM measurement.measurements
WHERE sample_id IS NOT NULL;

\echo '=== 8. 검사값 panel별 / 결측 현황 ==='
SELECT panel, count(*) AS observations, count(DISTINCT sample_id) AS samples
FROM clinical.observations
GROUP BY panel
ORDER BY count(*) DESC;

\echo '=== 9. 확인이 필요한 잔여 건 ==='
SELECT smp.solum_label, d.cohort_group, 'cohort_group 미확정' AS note
FROM clinical.diagnoses AS d
JOIN master.samples AS smp ON smp.subject_id = d.subject_id
WHERE d.cohort_group IS NULL OR d.cohort_group = 'Drop'
ORDER BY smp.solum_label;

\echo '=== 10. 같은 환자가 두 코호트에 들어간 케이스 ==='
SELECT site.site_code, sub.patient_code, sub.birth_date,
       array_agg(smp.solum_label ORDER BY smp.solum_label) AS labels
FROM master.subjects AS sub
JOIN master.sites AS site  ON site.site_id = sub.site_id
JOIN master.samples AS smp ON smp.subject_id = sub.subject_id
GROUP BY site.site_code, sub.patient_code, sub.birth_date
HAVING count(*) > 1;
