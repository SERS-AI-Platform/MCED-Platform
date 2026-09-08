-- 요로감염 지표 집계. 전부 읽기 전용이고 clinical.uti_indicators 뷰만 읽는다.
--
-- 분모에 주의할 것: 지표마다 검사된 환자 수가 다르다. ua_nitrite는 1,587명,
-- 요침사 백혈구는 238명에만 있다. "양성 n명"보다 "검사된 m명 중 n명"이 맞는 읽기다.

\echo ''
\echo '========== 1. 지표별: 검사된 환자 중 몇 명이 양성인가 =========='
WITH v AS (SELECT * FROM clinical.uti_indicators),
     total AS (SELECT count(DISTINCT subject_id) AS n FROM v)
SELECT indicator,
       tested   AS "검사된 환자",
       positive AS "양성 환자",
       round(100.0 * positive / nullif(tested, 0), 1) AS "양성률(%)",
       (SELECT n FROM total) - tested AS "미검사 환자"
FROM (
    SELECT 1 AS ord, 'ua_nitrite 양성' AS indicator,
           count(DISTINCT subject_id) FILTER (WHERE nitrite IS NOT NULL)  AS tested,
           count(DISTINCT subject_id) FILTER (WHERE nitrite = 'Positive') AS positive
    FROM v
    UNION ALL
    SELECT 2, 'ua_leukocyte esterase 양성',
           count(DISTINCT subject_id) FILTER (WHERE leukocyte_esterase IS NOT NULL),
           count(DISTINCT subject_id) FILTER (WHERE leukocyte_esterase = 'Positive')
    FROM v
    UNION ALL
    SELECT 3, '요침사 백혈구 >= 5/HPF (농뇨)',
           count(DISTINCT subject_id) FILTER (WHERE microscopy_wbc_low IS NOT NULL),
           count(DISTINCT subject_id) FILTER (WHERE microscopy_wbc_low >= 5)
    FROM v
    UNION ALL
    SELECT 4, '요침사 세균 보고됨',
           count(DISTINCT subject_id),
           count(DISTINCT subject_id) FILTER (WHERE microscopy_bacteria IS NOT NULL)
    FROM v
    UNION ALL
    SELECT 5, 'ua_blood(Heme) 양성 (참고: 혈뇨)',
           count(DISTINCT subject_id) FILTER (WHERE ua_blood_heme IS NOT NULL),
           count(DISTINCT subject_id) FILTER (WHERE ua_blood_heme = 'Positive')
    FROM v
    UNION ALL
    SELECT 6, '조합: nitrite + LE 동시 양성',
           count(DISTINCT subject_id) FILTER (WHERE nitrite IS NOT NULL
                                                AND leukocyte_esterase IS NOT NULL),
           count(DISTINCT subject_id) FILTER (WHERE nitrite = 'Positive'
                                                AND leukocyte_esterase = 'Positive')
    FROM v
    UNION ALL
    SELECT 7, '조합: nitrite 또는 LE 양성',
           count(DISTINCT subject_id) FILTER (WHERE nitrite IS NOT NULL
                                                 OR leukocyte_esterase IS NOT NULL),
           count(DISTINCT subject_id) FILTER (WHERE nitrite = 'Positive'
                                                 OR leukocyte_esterase = 'Positive')
    FROM v
    UNION ALL
    SELECT 8, '조합: (nitrite 또는 LE) + 농뇨',
           count(DISTINCT subject_id) FILTER (WHERE microscopy_wbc_low IS NOT NULL
                 AND (nitrite IS NOT NULL OR leukocyte_esterase IS NOT NULL)),
           count(DISTINCT subject_id) FILTER (WHERE (nitrite = 'Positive'
                                                  OR leukocyte_esterase = 'Positive')
                 AND microscopy_wbc_low >= 5)
    FROM v
) AS x
ORDER BY ord;

\echo ''
\echo '========== 2. 패턴별 검체 수 =========='
SELECT indicator_pattern, count(*) AS samples,
       count(DISTINCT subject_id) AS subjects
FROM clinical.uti_indicators
GROUP BY indicator_pattern
ORDER BY count(*) DESC;

\echo ''
\echo '========== 3. 병원 x 암종별 (미검사 분모 주의) =========='
SELECT site_code, cancer_group,
       count(*) AS "검체",
       count(*) FILTER (WHERE indicator_pattern = 'not_tested') AS "미검사",
       count(*) FILTER (WHERE indicator_pattern = 'nitrite_positive') AS "nitrite 양성",
       round(100.0 * count(*) FILTER (WHERE indicator_pattern = 'nitrite_positive')
             / nullif(count(*) FILTER (WHERE indicator_pattern <> 'not_tested'), 0), 1)
           AS "검사분모 양성률(%)",
       count(*) FILTER (WHERE leukocyte_esterase = 'Positive') AS "(참고) LE 양성"
FROM clinical.uti_indicators
GROUP BY site_code, cancer_group
ORDER BY site_code, cancer_group;
