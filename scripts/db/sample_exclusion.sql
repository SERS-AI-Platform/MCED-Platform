-- 검체 제외 판정 (aecd_platform)
--
-- 실행:
--   source scripts/db/pghost.sh
--   psql -d aecd_platform -f scripts/db/sample_exclusion.sql
--
-- 무엇인가
--   측정 대상에서 빼야 할 검체를 임상정보로 판정한다. 2026-09-07 에 Lot_Powder
--   워크북의 제외 대상을 빨간색으로 표시하면서 확정한 기준을, DB에서 언제든
--   다시 돌려볼 수 있게 옮겨 놓은 것이다.
--
-- 판정 기준
--   1) 요로감염      ua_nitrite = Positive
--   2) 타암 이력     medical_histories 에 현재 암종과 다른 암/종양
--   3) 치료 후 채취  collection_date 가 수술일 또는 항암 시작일보다 뒤.
--                    단 sample_timing 에 pre_treatment_specimen 이 있으면 제외 안 함
--   4) 다중암        진단이 2개 이상인 환자 -- 검체 전부
--   5) Drop          cohort_group = 'Drop'
--
-- 기준마다 이렇게 정한 이유
--   1) 요로감염을 nitrite 하나로 가르는 근거는 08_uti_indicators_view.sql 에 있다.
--      요약하면 nitrite 는 특이도가 높고(~95%), leukocyte esterase 단독 양성은
--      감염의 증거가 아니다 -- 방광암 코호트에서 LE 단독 양성 126건 중 114건이
--      혈뇨를 동반해 종양 염증과 구분되지 않는다. LE 를 넣으면 방광암이 통째로
--      걸려 코호트 confound 가 커진다.
--   2) 자기 암은 '타암'이 아니므로 뺀다. 단어 경계로 봐야 한다 --
--      'Gallbladder cancer'(담낭암) 안에 'bladder' 가 들어 있어서, 부분문자열로
--      비교하면 방광암 환자의 담낭암 이력이 자기 암으로 잘못 걸러진다.
--   3) sample_timing 문자열이 아니라 날짜로 가른다. 문자열에 post_* 가 붙은
--      검체 231건 중 61건은 날짜상 치료 후가 아니고(치료일이 없거나, 채취일과
--      같은 날이거나, 치료일이 오히려 더 늦다), 반대로 문자열이 pre_* 인데
--      날짜상 치료 후인 검체가 3건 있다. pre_treatment_specimen 태그가 붙은
--      건을 살리는 것은 2026-09-07 사용자 판단이다.
--
-- DB 로 재현되지 않는 기준이 하나 있다
--   원본 임상 워크북(전체환자_임상정보_정규화_v8.xlsx)에는 사람이 손으로 칠한
--   노란색 하이라이트가 있는데, 색은 DB에 적재되지 않는다. 다만 노란 행 144개 중
--   142개(98.6%)가 위 1~2 로 설명되므로 실질적인 누락은 거의 없다. 워크북 기준
--   226건 중 노란 표시만이 유일한 근거인 검체는 1건(CRC_294)뿐이다.
--
-- 이 파일이 DB(v7)에서 내는 숫자는 워크북(v8) 작업 결과와 다르다
--   DB 는 전체 2,851 검체, 워크북 표시는 Lot 측정 대상 2,465개만 대상이었다.
--   (Lot 시트의 KPAN 열은 재료연 검체라 임상표에 대응 라벨이 없어 판정 대상이
--   아니다 -- scripts/db/lot_powder_exclusion/lot_map.py 참고.)


\set ON_ERROR_STOP on
\pset null '·'

-- 세션이 끝나면 사라지는 임시 뷰다. DB에 아무것도 남기지 않는다.
-- 영구 뷰로 두고 싶으면 파일 맨 아래 블록을 참고할 것.
CREATE TEMP VIEW sample_exclusion AS
WITH tx AS (
    SELECT subject_id,
           min(start_date) FILTER (WHERE treatment_type = 'surgery')      AS first_surgery,
           min(start_date) FILTER (WHERE treatment_type = 'chemotherapy') AS first_chemo
    FROM clinical.treatments
    WHERE start_date IS NOT NULL
    GROUP BY subject_id
),
multi AS (      -- 진단이 2개 이상인 환자 (중복암)
    SELECT subject_id FROM clinical.diagnoses GROUP BY subject_id HAVING count(*) > 1
),
-- 검체 자신의 암종. uti_indicators 가 이미 라벨 접두사로 검체당 진단 1행을
-- 골라 두었으므로(중복암 환자 대응) 그대로 쓴다.
base AS (
    SELECT u.sample_id, u.solum_label, u.subject_id, u.patient_code, u.site_code,
           u.label_prefix, u.cancer_group, u.nitrite, u.leukocyte_esterase,
           u.indicator_pattern,
           smp.collection_date, smp.sample_timing,
           tx.first_surgery, tx.first_chemo,
           (m.subject_id IS NOT NULL) AS is_multi_cancer
    FROM clinical.uti_indicators AS u
    JOIN master.samples AS smp ON smp.sample_id = u.sample_id
    LEFT JOIN tx    ON tx.subject_id = u.subject_id
    LEFT JOIN multi AS m ON m.subject_id = u.subject_id
),
-- 현재 암종과 다른 암/종양 이력. 자기 장기는 단어 경계로 뺀다.
other_cancer AS (
    SELECT b.sample_id, string_agg(DISTINCT h.condition_raw, ', ') AS conditions
    FROM base AS b
    JOIN clinical.medical_histories AS h ON h.subject_id = b.subject_id
    WHERE h.condition_raw ~* 'cancer|carcinoma|tumor|tumour|neoplasm|lymphoma'
                             '|leukemia|leukaemia|sarcoma|myeloma|malignan'
      AND NOT CASE b.label_prefix
              WHEN 'BLC'  THEN h.condition_raw ~* '(^|[^a-z])bladder'
              WHEN 'BRE'  THEN h.condition_raw ~* '(^|[^a-z])breast'
              WHEN 'CRC'  THEN h.condition_raw ~* '(^|[^a-z])(colon|rectal|rectum|colorectal)'
              WHEN 'LUN'  THEN h.condition_raw ~* '(^|[^a-z])lung'
              WHEN 'OVA'  THEN h.condition_raw ~* '(^|[^a-z])ovar'
              WHEN 'PAN'  THEN h.condition_raw ~* '(^|[^a-z])pancrea'
              WHEN 'SPAN' THEN h.condition_raw ~* '(^|[^a-z])pancrea'
              WHEN 'YPAN' THEN h.condition_raw ~* '(^|[^a-z])pancrea'
              WHEN 'PRO'  THEN h.condition_raw ~* '(^|[^a-z])prostate'
              WHEN 'BPRO' THEN h.condition_raw ~* '(^|[^a-z])prostate'
              ELSE false      -- 대조군은 자기 암이 없으므로 모든 암 이력이 '타암'
          END
    GROUP BY b.sample_id
)
SELECT
    b.sample_id, b.solum_label, b.patient_code, b.site_code, b.label_prefix,
    b.cancer_group,
    CASE b.label_prefix
        WHEN 'BLC' THEN '방광암'   WHEN 'BRE'  THEN '유방암'
        WHEN 'CRC' THEN '대장암'   WHEN 'LUN'  THEN '폐암'
        WHEN 'OVA' THEN '난소암'
        WHEN 'PAN' THEN '췌장암'   WHEN 'SPAN' THEN '췌장암'  WHEN 'YPAN' THEN '췌장암'
        WHEN 'PRO' THEN '전립선암' WHEN 'BPRO' THEN '전립선암'
        ELSE '대조군'
    END AS 암종,
    b.collection_date, b.first_surgery, b.first_chemo, b.sample_timing,
    b.nitrite, b.leukocyte_esterase,

    (b.indicator_pattern = 'nitrite_positive')          AS 요로감염,
    (oc.sample_id IS NOT NULL)                          AS 타암이력,
    (b.sample_timing IS DISTINCT FROM NULL
     AND b.sample_timing LIKE '%pre\_treatment\_specimen%') AS pre_treatment_태그,
    (COALESCE(b.collection_date > b.first_surgery
           OR b.collection_date > b.first_chemo, false)
     AND NOT COALESCE(b.sample_timing LIKE '%pre\_treatment\_specimen%', false))
                                                        AS 치료후채취,
    b.is_multi_cancer                                   AS 다중암,
    (b.cancer_group = 'Drop')                           AS drop된건,
    oc.conditions                                       AS 타암이력_내용
FROM base AS b
LEFT JOIN other_cancer AS oc ON oc.sample_id = b.sample_id;


\echo ''
\echo '=== 1. 암종별 제외 현황 (근거 중복 포함) ==='
SELECT 암종,
       count(*)                                                   AS 전체,
       count(*) FILTER (WHERE 요로감염 OR 타암이력 OR 치료후채취
                           OR 다중암 OR drop된건)                  AS 제외,
       round(100.0 * count(*) FILTER (WHERE 요로감염 OR 타암이력 OR 치료후채취
                                         OR 다중암 OR drop된건) / count(*), 1) AS 비율,
       count(*) FILTER (WHERE 요로감염)   AS 요로감염,
       count(*) FILTER (WHERE 타암이력)   AS 타암이력,
       count(*) FILTER (WHERE 치료후채취) AS 치료후채취,
       count(*) FILTER (WHERE 다중암)     AS 다중암,
       count(*) FILTER (WHERE drop된건)   AS "Drop"
FROM sample_exclusion
GROUP BY 암종
ORDER BY 제외 DESC;


\echo ''
\echo '=== 2. 라벨 접두사별 (병원 구분) ==='
SELECT label_prefix, site_code,
       count(*)                                                   AS 전체,
       count(*) FILTER (WHERE 요로감염 OR 타암이력 OR 치료후채취
                           OR 다중암 OR drop된건)                  AS 제외,
       count(*) FILTER (WHERE 요로감염)   AS 요로감염,
       count(*) FILTER (WHERE 타암이력)   AS 타암이력,
       count(*) FILTER (WHERE 치료후채취) AS 치료후채취
FROM sample_exclusion
GROUP BY label_prefix, site_code
ORDER BY 제외 DESC, label_prefix;


\echo ''
\echo '=== 3. 근거를 중복 없이 (검체 1개당 대표 근거 하나) ==='
\echo '     우선순위: 타암이력 > 요로감염 > 치료후채취 > 다중암 > Drop'
SELECT 암종,
       count(*) AS 제외,
       count(*) FILTER (WHERE 타암이력)                                   AS 타암이력,
       count(*) FILTER (WHERE NOT 타암이력 AND 요로감염)                   AS 요로감염,
       count(*) FILTER (WHERE NOT 타암이력 AND NOT 요로감염
                          AND 치료후채취)                                  AS 치료후채취,
       count(*) FILTER (WHERE NOT 타암이력 AND NOT 요로감염
                          AND NOT 치료후채취 AND 다중암)                   AS 다중암,
       count(*) FILTER (WHERE NOT 타암이력 AND NOT 요로감염
                          AND NOT 치료후채취 AND NOT 다중암 AND drop된건)  AS "Drop"
FROM sample_exclusion
WHERE 요로감염 OR 타암이력 OR 치료후채취 OR 다중암 OR drop된건
GROUP BY 암종
ORDER BY 제외 DESC;


\echo ''
\echo '=== 4. 제외 검체 전체 목록 ==='
SELECT solum_label, 암종, site_code, collection_date, sample_timing,
       요로감염, 타암이력, 치료후채취, 다중암, drop된건,
       concat_ws('; ',
           CASE WHEN 요로감염 THEN '요로감염(nitrite Positive)' END,
           CASE WHEN 타암이력 THEN '타암 이력(' || 타암이력_내용 || ')' END,
           CASE WHEN 치료후채취 THEN '치료 후 채취' END,
           CASE WHEN 다중암 THEN '다중암 환자' END,
           CASE WHEN drop된건 THEN 'cohort_group=Drop' END) AS 제외근거
FROM sample_exclusion
WHERE 요로감염 OR 타암이력 OR 치료후채취 OR 다중암 OR drop된건
ORDER BY 암종, solum_label;


\echo ''
\echo '=== 5. 경계 케이스: sample_timing 문자열과 날짜가 어긋나는 검체 ==='
\echo '     기준은 날짜다. 이 목록은 원본 데이터 점검용이다.'
SELECT solum_label, 암종, collection_date, first_surgery, first_chemo, sample_timing,
       CASE
           WHEN 치료후채취 AND sample_timing NOT LIKE '%post\_%'
               THEN 'A. 날짜는 치료후인데 timing 에 post_ 없음'
           WHEN first_surgery IS NULL AND first_chemo IS NULL
               THEN 'B. timing 은 post_ 인데 치료일이 DB에 없음'
           WHEN collection_date IN (first_surgery, first_chemo)
               THEN 'C. timing 은 post_ 인데 채취일 = 치료일'
           WHEN pre_treatment_태그
               THEN 'D. 날짜는 치료후지만 pre_treatment 태그로 살림'
           ELSE 'E. timing 은 post_ 인데 치료일이 채취일보다 늦음'
       END AS 유형
FROM sample_exclusion
WHERE (치료후채취 AND sample_timing NOT LIKE '%post\_%')
   OR (sample_timing LIKE '%post\_%' AND NOT 치료후채취)
   OR (pre_treatment_태그
       AND COALESCE(collection_date > first_surgery
                 OR collection_date > first_chemo, false))
ORDER BY 유형, solum_label;


-- ---------------------------------------------------------------------------
-- 영구 뷰로 두고 싶을 때
--
-- 위 CREATE TEMP VIEW 는 psql 세션이 끝나면 사라진다. 대시보드나 다른 쿼리에서
-- 계속 참조하려면 아래처럼 영구 뷰로 만들면 된다. DB 를 바꾸는 작업이므로
-- 일부러 실행하지 않고 남겨 둔다.
--
--   CREATE OR REPLACE VIEW clinical.sample_exclusion AS
--   <위 CREATE TEMP VIEW 의 WITH ... SELECT 부분을 그대로 붙여넣기>;
--
--   COMMENT ON VIEW clinical.sample_exclusion IS
--       '검체 제외 판정. 요로감염(nitrite)/타암 이력/치료 후 채취/다중암/Drop. '
--       '원본 워크북의 노란 하이라이트는 색이 DB에 적재되지 않아 반영되지 않는다.';
-- ---------------------------------------------------------------------------
