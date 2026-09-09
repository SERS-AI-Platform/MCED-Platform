-- =====================================================================
-- 보라매 전립선 매핑(2026-08-10~14) — lot · 측정일 · 측정자 · 군 교차 확인
-- DB: aecd_platform (DBeaver에서 그대로 실행)
-- 근거 문서: docs/ml/experiment.md Phase PL-3 보충 2·3,
--           docs/reports/2026-09-09_보라매매핑_측정데이터_검토보고.docx
-- =====================================================================

-- [1] run별 측정일·측정자·시약 lot·strip lot × 군별 검체 수
--     기대: 비암(control, prostate disease control)은 전부 strip_lot_id=1,
--           암(prostate)은 전부 strip_lot_id=6, reagent_lot_id는 전부 1
SELECT r.measurement_run_id                AS run_id,
       r.measurement_date::date            AS measurement_day,
       r.operator_name,
       r.reagent_lot_id,
       r.strip_lot_id,
       r.temperature_c,
       r.humidity_percent,
       d.cohort_group,
       COUNT(DISTINCT s.sample_id)         AS n_samples,
       COUNT(DISTINCT m.measurement_id)    AS n_points
FROM measurement.runs r
JOIN measurement.measurements m USING (measurement_run_id)
JOIN master.samples s           USING (sample_id)
JOIN clinical.diagnoses d       USING (subject_id)
WHERE d.cohort_group IN ('prostate', 'prostate disease control', 'control')
GROUP BY 1, 2, 3, 4, 5, 6, 7, 8
ORDER BY measurement_day, run_id, cohort_group;

-- [2] 같은 내용을 날짜 × 군 피벗으로 (보고서 표 1)
SELECT r.measurement_date::date AS measurement_day,
       r.strip_lot_id,
       r.operator_name,
       COUNT(DISTINCT s.sample_id) FILTER (WHERE d.cohort_group = 'control')                  AS control,
       COUNT(DISTINCT s.sample_id) FILTER (WHERE d.cohort_group = 'prostate disease control') AS disease_control,
       COUNT(DISTINCT s.sample_id) FILTER (WHERE d.cohort_group = 'prostate')                 AS cancer
FROM measurement.runs r
JOIN measurement.measurements m USING (measurement_run_id)
JOIN master.samples s           USING (sample_id)
JOIN clinical.diagnoses d       USING (subject_id)
WHERE d.cohort_group IN ('prostate', 'prostate disease control', 'control')
GROUP BY 1, 2, 3
ORDER BY 1, 2;

-- [3] lot 마스터 — lot 번호·제조일·수령일 (strip lot 1 = SK20260804B01, 6 = SK20260804C01)
SELECT strip_lot_id, strip_name, lot_number, manufactured_date, received_date, expiration_date
FROM measurement.strip_lots ORDER BY strip_lot_id;

SELECT reagent_lot_id, reagent_id, lot_number, received_date, opened_date, expiration_date
FROM measurement.reagent_lots ORDER BY reagent_lot_id;

-- [4] PS/Si 표준물질 일별 축 보정 결과 — 파수축 drift 배제 근거 (shift 0.02~0.20 cm-1)
SELECT calibration_date::date AS calibration_day,
       standard_material,
       result,
       ROUND(global_shift_cm1::numeric, 3)          AS global_shift_cm1,
       ROUND(alignment_score::numeric, 3)           AS alignment_score,
       ROUND(replicate_shift_std_cm1::numeric, 4)   AS replicate_shift_std_cm1,
       observed_peaks_cm1
FROM measurement.calibrations
WHERE calibration_date::date BETWEEN '2026-08-10' AND '2026-08-14'
ORDER BY calibration_day, standard_material;

-- [5] 검체 → 측정일·lot 매핑 (AI OOF 확률 CSV와 조인용)
--     CSV: results/preprocessing_lab/guide_pipeline_20260907/aecd/production/patient_oof_predictions.csv
--     CSV의 subject = '<cohort_group>|<sample_id>' 이므로 sample_id로 조인
SELECT s.sample_id,
       s.solum_label,
       d.cohort_group,
       r.measurement_date::date AS measurement_day,
       r.strip_lot_id,
       r.operator_name,
       COUNT(m.measurement_id)  AS n_points
FROM master.samples s
JOIN clinical.diagnoses d       USING (subject_id)
JOIN measurement.measurements m USING (sample_id)
JOIN measurement.runs r         USING (measurement_run_id)
WHERE d.cohort_group IN ('prostate', 'prostate disease control', 'control')
GROUP BY 1, 2, 3, 4, 5, 6
ORDER BY measurement_day, cohort_group, s.sample_id;

-- [6] AI 실험 쪽 — PL-3 run과 성능 지표 (evaluation=oof, CI 포함)
SELECT r.run_id, r.run_name, r.n_subjects, r.n_positive, r.n_negative,
       x.task, x.metric_name,
       ROUND(x.metric_value::numeric, 3) AS value,
       ROUND(x.ci_low::numeric, 3)       AS ci_low,
       ROUND(x.ci_high::numeric, 3)      AS ci_high
FROM experiment.runs r
JOIN experiment.run_metrics x USING (run_id)
WHERE r.run_name IN ('guide_aecd_production_20260907',
                     'guide_aecd_ctrl_production_20260907',
                     'guide_aecd_no_cal_sg11_20260907')
  AND x.ci_low IS NOT NULL
  AND x.metric_name IN ('auc', 'pr_auc', 'sensitivity', 'specificity', 'precision_ppv', 'npv', 'macro_auc')
ORDER BY r.run_id, x.task, x.metric_name;

-- [7] 위 run이 실제로 어떤 measurement를 썼는지 (FK 추적)
SELECT r.run_name, COUNT(rm.measurement_id) AS n_measurements,
       COUNT(DISTINCT m.sample_id)          AS n_samples
FROM experiment.runs r
JOIN experiment.run_measurements rm USING (run_id)
JOIN measurement.measurements m     USING (measurement_id)
WHERE r.run_name LIKE 'guide_aecd%_20260907'
GROUP BY r.run_name ORDER BY r.run_name;
