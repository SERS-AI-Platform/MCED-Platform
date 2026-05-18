-- =====================================================================
-- 99_verify.sql  —  통합 임상 테이블 검증 쿼리
-- 이 파일은 "실행"이 아니라 한 블록씩 복사해서 확인용으로 사용
-- =====================================================================

-- (1) 질환그룹 × 파일별 샘플 수 + 핵심 컬럼 커버리지
SELECT disease_group, source_file, COUNT(*) AS n,
       COUNT(sex)             AS has_sex,
       COUNT(age)             AS has_age,
       COUNT(height_cm)       AS has_height,
       COUNT(weight_kg)       AS has_weight,
       COUNT(medical_history) AS has_mh,
       COUNT(tnm_stage)       AS has_tnm,
       COUNT(t_stage)         AS has_t,
       COUNT(*) FILTER (WHERE cancer_stage_group = 'unknown') AS stage_unknown,
       COUNT(*) FILTER (WHERE prior_cancer_flag)              AS prior_cancer
FROM std.clinical_unified
GROUP BY disease_group, source_file
ORDER BY disease_group, source_file;

-- (2) solum_label prefix ↔ disease_group 일치 여부
SELECT disease_group, LEFT(solum_label, 4) AS label_prefix, COUNT(*)
FROM std.clinical_unified
WHERE solum_label IS NOT NULL
GROUP BY disease_group, label_prefix
ORDER BY disease_group;

-- (3) hospital_code / hospital_name 채움 상태
SELECT disease_group, hospital_code, hospital_name, COUNT(*)
FROM std.clinical_unified
GROUP BY disease_group, hospital_code, hospital_name
ORDER BY disease_group;

-- (4) staging 분포
SELECT disease_group, cancer_stage_group, COUNT(*),
       COUNT(*) FILTER (WHERE prior_cancer_flag) AS prior_cancer
FROM std.clinical_unified
GROUP BY disease_group, cancer_stage_group
ORDER BY disease_group, cancer_stage_group;

-- (5) smoking / drinking status
SELECT disease_group, smoking_status, drinking_status, COUNT(*)
FROM std.clinical_unified
GROUP BY disease_group, smoking_status, drinking_status
ORDER BY disease_group;

-- (6) 전체 요약 (한 줄짜리)
SELECT COUNT(*)                        AS total,
       COUNT(DISTINCT disease_group)   AS n_groups,
       COUNT(DISTINCT source_file)     AS n_files,
       COUNT(DISTINCT hospital_code)   AS n_hospitals,
       COUNT(*) FILTER (WHERE prior_cancer_flag) AS prior_cancer_total
FROM std.clinical_unified;
