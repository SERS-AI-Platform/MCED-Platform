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
--   SELECT cancer_group, indicator_pattern, count(*) FROM clinical.uti_indicators
--   GROUP BY 1,2 ORDER BY 1,2;
--
-- cancer_group은 검체 자기 자신의 진단이다. clinical.diagnoses가 subject에만 붙어
-- 있어서 중복암 환자 2명(19022041: BLC_247/PRO_60, 19276017: CRC_186/PAN_78)은
-- 진단이 2개 달리는데, 라벨 접두사와 cancer_type을 맞춰 검체별로 하나를 고른다.

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
    lbl.label_prefix,
    dx.cancer_group,
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
LEFT JOIN pivoted AS p      ON p.sample_id = smp.sample_id
CROSS JOIN LATERAL (
    SELECT
        substring(smp.solum_label FROM '^[A-Za-z. ]+?(?=_)') AS label_prefix
) AS lbl
CROSS JOIN LATERAL (
    SELECT CASE lbl.label_prefix
        WHEN 'BLC'  THEN 'bladder'
        WHEN 'BRE'  THEN 'breast'
        WHEN 'CRC'  THEN 'colorectal'
        WHEN 'LUN'  THEN 'lung'
        WHEN 'OVA'  THEN 'OVA'
        WHEN 'PAN'  THEN 'pancreatic'
        WHEN 'SPAN' THEN 'pancreatic'
        WHEN 'YPAN' THEN 'pancreatic'
        WHEN 'PRO'  THEN 'prostate'
        WHEN 'BPRO' THEN 'prostate'
    END AS expected_cancer_type
) AS exp
LEFT JOIN LATERAL (
    SELECT d.cohort_group AS cancer_group
    FROM clinical.diagnoses AS d
    WHERE d.subject_id = sub.subject_id
    -- 중복암 환자는 진단이 2개 달려 있다 (clinical.diagnoses가 subject에 붙어 있고
    -- sample에는 붙어 있지 않기 때문). 검체 라벨 접두사와 cancer_type이 맞는 진단을
    -- 먼저 고르므로, BLC_247은 bladder, PRO_60은 prostate가 된다.
    -- 진단이 하나뿐인 나머지 2,846명은 그 하나가 그대로 선택된다.
    ORDER BY (d.cancer_type IS DISTINCT FROM exp.expected_cancer_type), d.diagnosis_id
    LIMIT 1
) AS dx ON true;

COMMENT ON VIEW clinical.uti_indicators IS
    '검체별 요로감염 관련 요검사 지표(원문 + 조합 패턴). 판정용이 아니라 조회용 - '
    '세균뇨는 62건, 요침사 백혈구는 238건에만 있고 LE 양성률이 코호트마다 크게 달라 '
    '단일 기준 필터는 코호트 confound를 키운다.';
