-- Purpose:
--   요로감염 관련 요검사 지표를 검체 1행으로 펼쳐 보는 조회용 뷰.
--
-- 이 뷰는 "요로감염이다"라고 판정하지 않는다. 원문 값과 그 조합 패턴만 노출하고,
-- 어떤 검체를 제외할지는 쓰는 쪽에서 정한다. 이유:
--   - 세균뇨(Microscopy_Bacteria)는 2,851건 중 62건, 요침사 백혈구는 238건에만 있어
--     농뇨+세균뇨 확진 조합을 갖춘 검체가 거의 없다.
--   - leukocyte esterase 양성률이 코호트마다 크게 다르다 (방광암 45.8% vs
--     충북대 대조군 0.0%). 방광암은 종양 관련 염증·혈뇨로도 LE가 양성이 되므로
--     감염과 구분되지 않는다. 단일 기준으로 걸러내면 암군에서만 검체가 빠져
--     병원·코호트 confound가 커진다.
--
-- 사용 예:
--   SELECT * FROM clinical.uti_indicators WHERE indicator_pattern = 'nitrite+LE';
--   SELECT cohort_group, indicator_pattern, count(*) FROM clinical.uti_indicators
--   GROUP BY 1,2 ORDER BY 1,2;

CREATE OR REPLACE VIEW clinical.uti_indicators AS
WITH pivoted AS (
    SELECT
        o.sample_id,
        max(o.raw_value) FILTER (WHERE o.code = 'ua_nitrite')            AS nitrite,
        max(o.raw_value) FILTER (WHERE o.code = 'ua_leukocyte esterase') AS leukocyte_esterase,
        max(o.raw_value) FILTER (WHERE o.code = 'Microscopy_WBC')        AS microscopy_wbc_raw,
        max(o.raw_value) FILTER (WHERE o.code = 'Microscopy_Bacteria')   AS microscopy_bacteria,
        max(o.raw_value) FILTER (WHERE o.code = 'ua_blood(Heme)')        AS ua_blood_heme,
        max(o.raw_value) FILTER (WHERE o.code = 'ua_protein')            AS ua_protein,
        max(o.raw_value) FILTER (WHERE o.code = 'ua_ph')                 AS ua_ph
    FROM clinical.observations AS o
    WHERE o.code IN (
        'ua_nitrite', 'ua_leukocyte esterase', 'Microscopy_WBC',
        'Microscopy_Bacteria', 'ua_blood(Heme)', 'ua_protein', 'ua_ph'
    )
      AND o.sample_id IS NOT NULL
    GROUP BY o.sample_id
)
SELECT
    smp.sample_id,
    smp.solum_label,
    sub.subject_id,
    sub.patient_code,
    site.site_code,
    -- 두 코호트에 걸친 환자는 'bladder+prostate' 형태로 합쳐 한 행을 유지한다.
    (SELECT string_agg(DISTINCT d.cohort_group, '+')
       FROM clinical.diagnoses AS d
      WHERE d.subject_id = sub.subject_id) AS cohort_group,
    p.nitrite,
    p.leukocyte_esterase,
    p.microscopy_wbc_raw,
    -- "<1" / "< 1 (3.6/㎕)" 는 1개 미만이라 0으로, 그 외에는 보고된 구간의 하한을
    -- 취한다 ("10 - 19" -> 10, ">=100" -> 100).
    CASE
        WHEN p.microscopy_wbc_raw IS NULL THEN NULL
        WHEN p.microscopy_wbc_raw ~ '^\s*<' THEN 0
        ELSE (regexp_match(p.microscopy_wbc_raw, '(\d+)'))[1]::int
    END AS microscopy_wbc_low,
    p.microscopy_bacteria,
    p.ua_blood_heme,
    p.ua_protein,
    p.ua_ph,
    CASE
        WHEN p.nitrite IS NULL AND p.leukocyte_esterase IS NULL THEN 'not_tested'
        WHEN p.nitrite = 'Positive' AND p.leukocyte_esterase = 'Positive' THEN 'nitrite+LE'
        WHEN p.nitrite = 'Positive' THEN 'nitrite only'
        WHEN p.leukocyte_esterase = 'Positive' THEN 'LE only'
        ELSE 'negative'
    END AS indicator_pattern
FROM master.samples AS smp
JOIN master.subjects AS sub ON sub.subject_id = smp.subject_id
JOIN master.sites AS site   ON site.site_id = sub.site_id
LEFT JOIN pivoted AS p      ON p.sample_id = smp.sample_id;

COMMENT ON VIEW clinical.uti_indicators IS
    '검체별 요로감염 관련 요검사 지표(원문 + 조합 패턴). 판정용이 아니라 조회용 - '
    '세균뇨는 62건, 요침사 백혈구는 238건에만 있고 LE 양성률이 코호트마다 크게 달라 '
    '단일 기준 필터는 코호트 confound를 키운다.';
