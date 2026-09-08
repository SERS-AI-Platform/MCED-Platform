-- Purpose:
--   Load the v7 workbook rows produced by build_staging_csv.py into the
--   all-TEXT staging table. Nothing is transformed here.
--
-- Expected result:
--   2850 rows tagged source_mapping = 'clinical_v7_20260902'.
--
-- Note:
--   __STAGING_CSV__ is substituted by run_load.sh -- psql does not interpolate
--   variables inside \copy, so this file is a template, not a standalone script.
--
-- Re-run safety:
--   Deletes any earlier rows carrying the same source_mapping tag first, so the
--   script is idempotent. Rows from other ingests (the 2026-08 *_mapping ones)
--   are never touched.

BEGIN;

DELETE FROM ingest.clinical_master_staging
WHERE source_mapping = 'clinical_v7_20260902';

\copy ingest.clinical_master_staging ("source_mapping","site_code","group","solum_label","source_cancer_file","patient_code","cancer_type","sex","age","birth_date","weight_kg","height_cm","bmi","smoking_history","drinking_history","collection_date","post_treatment_collection_date","has_post_treatment_sample","sample_type","sample_amount","receipt_date","diagnosis_code","diagnosis_name","diagnosis_date","past_history_1","past_history_2","past_history_3","past_history_4","past_history_5","past_history_6","past_history_7","past_history_8","past_history_9","pathology_result","histologic_type","gleason","grade group","stage","tnm_stage","t_stage","n_stage","m_stage","metastasis","surgery_date","surgery_name","chemo_start_date","chemo_end_date","treatment_info","post_surgery_or_chemo_urine_result_available","primary_sample_timing_interpretation","wbc","rbc","hb","hct","platelet","neutrophil","lymphocyte","monocyte","eosinophil","basophil","ast","alt","alp","ggt","total_bilirubin","bun","creatinine","uric_acid","glucose","calcium","sodium","potassium","chloride","cholesterol","triglyceride","hba1c","ER","PR","HER2","CA 15-3","CEA","ca19_9","ca125","afp","psa","ua_sg","ua_ph","ua_protein","ua_glucose","ua_blood(Heme)","ua_ketone","ua_urobilinogen","ua_nitrite","ua_leukocyte esterase","Microscopy_WBC","Microscopy_RBC","Microscopy_Bacteria","source_row_number","standardization_notes") FROM '__STAGING_CSV__' WITH (FORMAT csv, HEADER true)

COMMIT;

SELECT source_mapping, count(*) AS rows
FROM ingest.clinical_master_staging
GROUP BY source_mapping
ORDER BY source_mapping;
