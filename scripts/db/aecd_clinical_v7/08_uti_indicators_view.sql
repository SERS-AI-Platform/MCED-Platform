-- Purpose:
--   요로감염 관련 요검사 지표를 검체 1행으로 펼쳐 보는 조회용 뷰.
--
-- 왜 indicator_pattern이 nitrite 하나만 보는가
--   nitrite는 세균이 질산염을 아질산염으로 환원한 결과라 특이도가 높다(~95%).
--   나머지는 감염의 증거가 아니다:
--     - leukocyte esterase / 요침사 백혈구(농뇨)는 백혈구가 있다는 뜻이지 세균이
--       있다는 뜻이 아니다. 종양 염증·기구조작·오염으로도 양성이 된다.
--     - 혈뇨는 요로감염과 무관하게 종양·결석으로 나온다.
--     - 요침사 세균은 배뇨 검체 오염이 흔해 단독으로는 진단 근거가 못 된다.
--   실제로 LE 단독 양성 260건 중 요침사로 확인 가능한 건 6건, 세균 보고는 0건인
--   반면 119건이 혈뇨를 동반한다. 방광암만 보면 LE 단독 126건 중 114건(90%)이
--   혈뇨 동반이라, 감염이 아니라 종양 출혈·염증으로 보는 편이 자연스럽다.
--   LE를 기준에 넣으면 방광암 코호트가 통째로 걸려 병원·코호트 confound가 커진다.
--
-- 확진 기준인 요배양(>=10^5 CFU/mL)은 이 데이터셋에 아예 없다. 그래서 이 뷰는
-- "요로감염이다"라고 판정하지 않는다 -- nitrite로 의심 검체를 좁혀 보여줄 뿐이고,
-- LE / 농뇨 / 세균 / 혈뇨는 판정에 쓰지 않고 컬럼으로만 남겨 직접 조건을 걸 수 있게 했다.
--
-- 사용 예:
--   SELECT * FROM clinical.uti_indicators WHERE indicator_pattern = 'nitrite_positive';
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
    -- 패턴은 nitrite 하나로만 가른다. 아래 '왜 nitrite만인가' 참고.
    CASE
        WHEN p.nitrite IS NULL     THEN 'not_tested'
        WHEN p.nitrite = 'Positive' THEN 'nitrite_positive'
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
    '검체별 요로감염 관련 요검사 지표. indicator_pattern은 특이도가 높은 ua_nitrite '
    '하나로만 가른다. leukocyte esterase / 농뇨 / 세균 / 혈뇨는 감염의 증거가 아니라 '
    '컬럼으로만 제공한다 - LE 단독 양성은 방광암에서 90%가 혈뇨를 동반해 종양 염증과 '
    '구분되지 않으며, 기준에 넣으면 코호트 confound가 커진다. 확진 기준인 요배양은 '
    '이 데이터셋에 없다.';
