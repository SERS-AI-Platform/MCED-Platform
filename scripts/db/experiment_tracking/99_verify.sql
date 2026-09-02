-- =====================================================================
-- 99_verify.sql  —  experiment 도메인 검증 쿼리
-- 이 파일은 "실행"이 아니라 한 블록씩 복사해서 확인용으로 사용
-- =====================================================================

-- ---------------------------------------------------------------------
-- A. 적용 직후 확인 (데이터가 없어도 실행 가능)
-- ---------------------------------------------------------------------

-- (A1) 5개 테이블이 모두 생성됐는지
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'experiment'
ORDER BY table_name;
-- 기대: papers, preprocessing_methods, runs, run_measurements, run_metrics

-- (A2) 임상 원본 스키마로 나가는 FK가 실제로 걸렸는지.
--      이게 aecd_platform 안에 둔 이유이므로 반드시 확인할 것.
SELECT
    tc.table_name        AS from_table,
    kcu.column_name      AS from_column,
    ccu.table_schema     AS to_schema,
    ccu.table_name       AS to_table,
    ccu.column_name      AS to_column
FROM information_schema.table_constraints AS tc
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
JOIN information_schema.constraint_column_usage AS ccu
    ON tc.constraint_name = ccu.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY'
  AND tc.table_schema = 'experiment'
ORDER BY from_table, from_column;
-- 기대: run_measurements.measurement_id → measurement.measurements.measurement_id 가 포함될 것

-- (A3) 지표 유니크 제약이 NULLS NOT DISTINCT인지 (재실행 시 중복 방지)
SELECT conname, pg_get_constraintdef(oid) AS definition
FROM pg_constraint
WHERE conrelid = 'experiment.run_metrics'::regclass
  AND contype = 'u';
-- 기대: UNIQUE NULLS NOT DISTINCT (run_id, metric_name, split, model_name)

-- (A4) 기존 도메인이 건드려지지 않았는지 (테이블 수 대조)
SELECT table_schema, COUNT(*) AS n_tables
FROM information_schema.tables
WHERE table_schema IN ('master', 'clinical', 'measurement', 'experiment')
GROUP BY table_schema
ORDER BY table_schema;
-- 기대: master/clinical/measurement의 테이블 수가 적용 전과 동일

-- ---------------------------------------------------------------------
-- B. 실험이 쌓인 뒤 확인
-- ---------------------------------------------------------------------

-- (B0) 과거 registry 임포트 결과 — phase별 건수.
--      logs/experiment_registry.json의 55건과 대조할 것.
SELECT
    COUNT(*)                                   AS total_runs,
    COUNT(*) FILTER (WHERE method_id IS NULL)  AS general_experiments,
    COUNT(*) FILTER (WHERE method_id IS NOT NULL) AS preprocessing_lab_runs,
    MIN(run_date)                              AS earliest,
    MAX(run_date)                              AS latest
FROM experiment.runs;

-- (B1) 방법별 최신 실행 + 논문 출처 (사람 검수 통과분만)
SELECT
    m.method_key,
    m.stage,
    m.audit_status,
    p.citation,
    r.run_name,
    r.status,
    r.finished_at
FROM experiment.preprocessing_methods AS m
LEFT JOIN experiment.papers AS p ON p.paper_id = m.paper_id
LEFT JOIN LATERAL (
    SELECT run_name, status, finished_at
    FROM experiment.runs
    WHERE method_id = m.method_id
    ORDER BY run_id DESC
    LIMIT 1
) AS r ON TRUE
ORDER BY m.audit_status, m.method_key;

-- (B2) 실행별 지표를 나란히 (같은 조건끼리만 비교할 것 — data_query_filters 확인)
SELECT
    r.run_name,
    m.method_key,
    r.data_query_filters,
    x.model_name,
    x.split,
    x.metric_name,
    x.metric_value
FROM experiment.run_metrics AS x
JOIN experiment.runs AS r ON r.run_id = x.run_id
JOIN experiment.preprocessing_methods AS m ON m.method_id = r.method_id
WHERE r.status = 'complete'
ORDER BY x.metric_name, x.metric_value DESC;

-- (B3) 각 실행이 실제로 사용한 measurement 수 vs 기록된 카운트.
--      불일치하면 데이터 소스 문제 의심 (근거 없는 숫자 감지).
SELECT
    r.run_name,
    r.n_spectra                          AS recorded_n_spectra,
    COUNT(rm.measurement_id)             AS linked_measurements
FROM experiment.runs AS r
LEFT JOIN experiment.run_measurements AS rm ON rm.run_id = r.run_id
GROUP BY r.run_id, r.run_name, r.n_spectra
ORDER BY r.run_id DESC;

-- (B4) 특정 실행이 어떤 코호트를 썼는지 임상 원본까지 join해서 확인
--      (:run_name 을 실제 값으로 바꿔서 실행)
SELECT
    d.cohort_group,
    s.site_code,
    COUNT(DISTINCT sub.subject_id) AS subjects,
    COUNT(*)                       AS measurements
FROM experiment.runs AS r
JOIN experiment.run_measurements AS rm ON rm.run_id = r.run_id
JOIN measurement.measurements AS meas ON meas.measurement_id = rm.measurement_id
JOIN master.samples AS smp ON smp.sample_id = meas.sample_id
JOIN master.subjects AS sub ON sub.subject_id = smp.subject_id
JOIN master.sites AS s ON s.site_id = sub.site_id
LEFT JOIN clinical.diagnoses AS d ON d.subject_id = sub.subject_id
WHERE r.run_name = :'run_name'
GROUP BY d.cohort_group, s.site_code
ORDER BY d.cohort_group, s.site_code;
