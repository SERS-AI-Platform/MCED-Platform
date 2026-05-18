-- =============================================================================
-- 01_engineering_exploration.sql
-- 목적: 어제 적재한 6개 테이블을 엔지니어링 관점에서 전수 조사
-- 대상 테이블:
--   1) prostate_전립선암           (100행 × 23열)
--   2) breast_유방암_box            (30행  ×  7열)
--   3) breast_유방암_clinical       (30행  × 14열)
--   4) ovarian_난소암1_box          (30행  ×  7열)
--   5) ovarian_난소암1_clinical     (30행  × 14열)
--   6) ovarian_난소암2_clinical     (40행  × 23열, 구조 다름)
--
-- 사용법: DBeaver에서 섹션별로 블록 선택 → Ctrl+Enter
-- 학습 포인트는 각 쿼리 위 주석에 [LEARN] 으로 표시
-- =============================================================================


-- =============================================================================
-- SECTION 0. 메타 — 내가 가진 것이 뭔지 먼저 본다
-- =============================================================================

-- [LEARN] information_schema 는 ANSI 표준 메타카탈로그.
--         "DB에 있는 객체 정보"는 항상 여기서 조회. \dt 같은 psql 명령보다
--         쿼리로 배우는 게 이식성 있음.
-- 0-1. 현재 public 스키마의 테이블 목록 + 컬럼 수
SELECT table_name,
       COUNT(*) AS col_count
FROM   information_schema.columns
WHERE  table_schema = 'public'
GROUP  BY table_name
ORDER  BY table_name;

-- 0-2. 각 테이블의 row 수 (한 번에)
-- [LEARN] UNION ALL = "결과를 그냥 이어붙임" (중복 제거 X → 빠름).
--         UNION은 중복 제거를 위해 정렬/해시가 들어가 느려짐.
SELECT 'prostate_전립선암'        AS table_name, COUNT(*) AS n FROM prostate_전립선암
UNION ALL SELECT 'breast_유방암_box',         COUNT(*) FROM breast_유방암_box
UNION ALL SELECT 'breast_유방암_clinical',    COUNT(*) FROM breast_유방암_clinical
UNION ALL SELECT 'ovarian_난소암1_box',       COUNT(*) FROM ovarian_난소암1_box
UNION ALL SELECT 'ovarian_난소암1_clinical',  COUNT(*) FROM ovarian_난소암1_clinical
UNION ALL SELECT 'ovarian_난소암2_clinical',  COUNT(*) FROM ovarian_난소암2_clinical;

-- 0-3. 한글 컬럼명은 반드시 쌍따옴표 "..."로 감싸야 한다 (식별자 인용)
-- [LEARN] PostgreSQL 규칙:
--         - 작은따옴표 '...' → 문자열 리터럴 (값)
--         - 큰따옴표  "..." → 식별자 (테이블/컬럼 이름)
--         - 식별자에 한글/공백/특수문자/대문자가 있으면 쌍따옴표 필수
-- 예시 — 에러나는 버전 vs 올바른 버전:
-- 에러:  SELECT 성별 FROM breast_유방암_clinical LIMIT 5;
-- OK:    SELECT "성별" FROM breast_유방암_clinical LIMIT 5;


-- =============================================================================
-- SECTION 1. 각 테이블 샘플 눈으로 보기 (Peek)
-- =============================================================================

-- [LEARN] 탐색의 첫 단계는 항상 LIMIT로 "몇 줄만 보기". SELECT * FROM ... 을
--         그냥 돌리면 30행이어도 의미 없고, 운영 DB라면 수백만 row가 튀어나옴.
SELECT * FROM prostate_전립선암         LIMIT 5;
SELECT * FROM breast_유방암_box         LIMIT 5;
SELECT * FROM breast_유방암_clinical    LIMIT 5;
SELECT * FROM ovarian_난소암1_box       LIMIT 5;
SELECT * FROM ovarian_난소암1_clinical  LIMIT 5;
SELECT * FROM ovarian_난소암2_clinical  LIMIT 5;


-- =============================================================================
-- SECTION 2. PK 후보 찾기 — "이 컬럼만 보면 행을 유일하게 식별할 수 있나?"
-- =============================================================================

-- [LEARN] PK(Primary Key) 후보의 3가지 조건:
--   (1) NULL 없음
--   (2) 중복 없음 (COUNT(*) == COUNT(DISTINCT col))
--   (3) 값이 안정적 (수정되지 않음 — 도메인 지식 필요)
-- 아래 쿼리는 (1)+(2)를 확인.

-- 2-1. prostate_전립선암 — patient_id 가 PK 후보인가?
SELECT
    COUNT(*)                                   AS total_rows,
    COUNT("patient_id")                        AS non_null_patient_id,  -- NULL 제외한 count
    COUNT(DISTINCT "patient_id")               AS distinct_patient_id,
    COUNT(*) - COUNT("patient_id")             AS null_count,
    COUNT(*) = COUNT(DISTINCT "patient_id")
        AND COUNT(*) = COUNT("patient_id")     AS is_pk_candidate
FROM prostate_전립선암;

-- 2-2. breast_유방암_clinical — 제공자bCODE 가 PK 후보인가?
SELECT
    COUNT(*)                             AS total_rows,
    COUNT("제공자bCODE")                 AS non_null,
    COUNT(DISTINCT "제공자bCODE")        AS distinct_,
    COUNT(*) = COUNT(DISTINCT "제공자bCODE")
        AND COUNT(*) = COUNT("제공자bCODE") AS is_pk_candidate
FROM breast_유방암_clinical;

-- 2-3. 같은 체크를 나머지 테이블에도 — box는 자원bCODE 또는 SoluM Label이 후보
SELECT
    '자원bCODE'                       AS col,
    COUNT(*) - COUNT(DISTINCT "자원bCODE") AS dup_count
FROM breast_유방암_box
UNION ALL SELECT 'SoluM Label', COUNT(*) - COUNT(DISTINCT "SoluM Label") FROM breast_유방암_box;

-- [과제] 같은 패턴을 ovarian_난소암1_box, ovarian_난소암1_clinical, ovarian_난소암2_clinical
--        에도 직접 적용해 보세요. 어느 컬럼이 PK 후보인가?


-- =============================================================================
-- SECTION 3. NULL 분포 — "어느 컬럼이 얼마나 비어있나"
-- =============================================================================

-- [LEARN] COUNT(col) 은 NULL을 제외하고 센다.
--         반대로 "NULL인 행 수"는 COUNT(*) - COUNT(col) 또는
--         SUM(CASE WHEN col IS NULL THEN 1 ELSE 0 END) 으로 구함.
--
-- 3-1. breast_유방암_clinical — 컬럼별 NULL 비율
SELECT
    COUNT(*) AS n,
    COUNT(*) - COUNT("성별")              AS null_성별,
    COUNT(*) - COUNT("나이")              AS null_나이,
    COUNT(*) - COUNT("체중")              AS null_체중,
    COUNT(*) - COUNT("신장")              AS null_신장,
    COUNT(*) - COUNT("흡연력")            AS null_흡연력,
    COUNT(*) - COUNT("음주력")            AS null_음주력,
    COUNT(*) - COUNT("과거력")            AS null_과거력,
    COUNT(*) - COUNT("병리결과")          AS null_병리결과,
    COUNT(*) - COUNT("혈액검사결과 : CEA, CA19-9") AS null_CEA
FROM breast_유방암_clinical;

-- 3-2. prostate_전립선암 — NULL 비율을 % 로 (좀 더 고급)
-- [LEARN] FILTER 절 = WHERE가 특정 집계함수에만 적용. 읽기 좋고 빠름.
--         (동일: SUM(CASE WHEN col IS NULL THEN 1 ELSE 0 END))
-- [LEARN] ROUND(x, 1) = 소수 1자리 반올림. 100.0 곱하고 나누는 건 정수 나눗셈 회피용.
SELECT
    ROUND(100.0 * COUNT(*) FILTER (WHERE "tnm_stage"    IS NULL) / COUNT(*), 1) AS pct_null_tnm,
    ROUND(100.0 * COUNT(*) FILTER (WHERE "grade"        IS NULL) / COUNT(*), 1) AS pct_null_grade,
    ROUND(100.0 * COUNT(*) FILTER (WHERE "metastasis"   IS NULL) / COUNT(*), 1) AS pct_null_metastasis,
    ROUND(100.0 * COUNT(*) FILTER (WHERE "과거력"       IS NULL) / COUNT(*), 1) AS pct_null_과거력,
    ROUND(100.0 * COUNT(*) FILTER (WHERE "histologic_type" IS NULL) / COUNT(*), 1) AS pct_null_histo
FROM prostate_전립선암;


-- =============================================================================
-- SECTION 4. 값 분포 — 범주형/연속형 별로 다르게 본다
-- =============================================================================

-- 4-1. 범주형: GROUP BY + COUNT 가 제1원칙
-- [LEARN] ORDER BY COUNT(*) DESC → 흔한 값부터 보기 (Pareto)
SELECT "성별", COUNT(*) AS n
FROM prostate_전립선암
GROUP BY "성별"
ORDER BY n DESC;

SELECT "흡연력", COUNT(*) AS n
FROM prostate_전립선암
GROUP BY "흡연력"
ORDER BY n DESC;

SELECT "grade", COUNT(*) AS n
FROM prostate_전립선암
GROUP BY "grade"
ORDER BY n DESC;

SELECT "tnm_stage", COUNT(*) AS n
FROM prostate_전립선암
GROUP BY "tnm_stage"
ORDER BY n DESC;

-- [과제] breast_유방암_clinical "진단코드 및 진단명"의 고유값 몇 개인가?
--        — ICD-10 코드와 설명이 한 셀에 같이 들어있으면 나중에 split 필요.


-- 4-2. 연속형: 요약통계
-- [LEARN] PostgreSQL 기본 집계 함수 + percentile_cont (연속 분위수).
--         percentile_cont(0.5) WITHIN GROUP (ORDER BY col) = 중앙값
SELECT
    MIN("나이")                                                       AS age_min,
    MAX("나이")                                                       AS age_max,
    ROUND(AVG("나이")::numeric, 1)                                    AS age_mean,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY "나이")               AS age_median,
    ROUND(stddev_samp("나이")::numeric, 2)                            AS age_sd
FROM prostate_전립선암;

-- [LEARN] ::numeric 은 캐스트(타입 변환). double precision → numeric 으로 바꿔야
--         ROUND(x, 1) 이 소수점 1자리까지 지원. PostgreSQL 특수 규칙.


-- =============================================================================
-- SECTION 5. 이상치/더러운 값 탐지
-- =============================================================================

-- 5-1. 나이가 범위를 벗어난 경우 (0~120 벗어나면 의심)
SELECT *
FROM prostate_전립선암
WHERE "나이" IS NULL OR "나이" < 0 OR "나이" > 120;

-- 5-2. 체중/신장이 text로 저장된 이유? → 숫자가 아닌 값이 섞여있을 것
-- [LEARN] 정규식으로 숫자 아닌 값 찾기. ~ 는 정규식 일치, !~ 는 불일치.
--         '^[0-9.]+$' = "0~9와 . 으로만 구성된 문자열".
SELECT "체중", COUNT(*) AS n
FROM prostate_전립선암
WHERE "체중" !~ '^[0-9.]+$' OR "체중" IS NULL
GROUP BY "체중"
ORDER BY n DESC;

SELECT "신장", COUNT(*) AS n
FROM prostate_전립선암
WHERE "신장" !~ '^[0-9.]+$' OR "신장" IS NULL
GROUP BY "신장"
ORDER BY n DESC;

-- 5-3. 문자열 좌우 공백/이상 문자 (트리밍 이슈)
-- [LEARN] LENGTH vs LENGTH(TRIM(...)) 차이 → 앞뒤 공백 존재 여부
SELECT "성별", LENGTH("성별") AS len
FROM prostate_전립선암
GROUP BY "성별", LENGTH("성별");

-- 5-4. 날짜 컬럼이 bigint로 들어간 것 (Excel serial date?)
-- [LEARN] Excel은 1900-01-01을 1로 시작하는 일자 시리얼을 씀.
--         TO_DATE('1899-12-30','YYYY-MM-DD') + serial * INTERVAL '1 day' 가 표준 트릭.
SELECT
    "자원 수집일"                                                AS raw_bigint,
    DATE '1899-12-30' + ("자원 수집일" || ' days')::interval     AS as_date
FROM breast_유방암_clinical
LIMIT 10;


-- =============================================================================
-- SECTION 6. 테이블 간 관계 — 같은 환자가 여러 테이블에 있나?
-- =============================================================================

-- 6-1. box ↔ clinical 1:1 매칭 여부 (breast)
-- [LEARN] FULL OUTER JOIN 은 양쪽 모두에 있는 키 + 한쪽에만 있는 키까지 전부 보여줌
--         → "매칭 안 되는" 행 찾는 표준 패턴
SELECT
    b."제공자:제공자bCODE"::bigint AS box_code,
    c."제공자bCODE"                AS clin_code,
    CASE
        WHEN b."제공자:제공자bCODE" IS NULL THEN 'clinical_only'
        WHEN c."제공자bCODE" IS NULL        THEN 'box_only'
        ELSE 'matched'
    END AS match_status
FROM breast_유방암_box b
FULL OUTER JOIN breast_유방암_clinical c
    ON b."제공자:제공자bCODE"::bigint = c."제공자bCODE"
ORDER BY match_status;

-- 6-2. 매칭 카운트 요약
SELECT
    CASE
        WHEN b."제공자:제공자bCODE" IS NULL THEN 'clinical_only'
        WHEN c."제공자bCODE" IS NULL        THEN 'box_only'
        ELSE 'matched'
    END AS status,
    COUNT(*) AS n
FROM breast_유방암_box b
FULL OUTER JOIN breast_유방암_clinical c
    ON b."제공자:제공자bCODE"::bigint = c."제공자bCODE"
GROUP BY status;

-- 6-3. 난소암1 도 동일 패턴
SELECT
    CASE
        WHEN b."제공자:제공자bCODE" IS NULL THEN 'clinical_only'
        WHEN c."제공자bCODE" IS NULL        THEN 'box_only'
        ELSE 'matched'
    END AS status,
    COUNT(*) AS n
FROM ovarian_난소암1_box b
FULL OUTER JOIN ovarian_난소암1_clinical c
    ON b."제공자:제공자bCODE"::bigint = c."제공자bCODE"
GROUP BY status;


-- =============================================================================
-- SECTION 7. 암종 간 컬럼 공통성 — unified 테이블 설계 근거
-- =============================================================================

-- 7-1. 각 테이블의 컬럼을 한 줄씩 전부 나열 (비교 편의용)
SELECT table_name, column_name, data_type
FROM   information_schema.columns
WHERE  table_schema = 'public'
  AND  table_name IN (
       'prostate_전립선암',
       'breast_유방암_clinical',
       'ovarian_난소암1_clinical',
       'ovarian_난소암2_clinical')
ORDER  BY column_name, table_name;

-- 7-2. "어느 컬럼이 몇 개 테이블에 존재하나" (교집합/차집합 파악)
-- [LEARN] CTE(Common Table Expression) = WITH 절. 복잡한 쿼리를 이름 붙여
--         단계별로 구성. 가독성↑, 성능은 동일 또는 비슷.
WITH cols AS (
    SELECT table_name, column_name
    FROM   information_schema.columns
    WHERE  table_schema = 'public'
      AND  table_name LIKE '%clinical%' OR table_name = 'prostate_전립선암'
)
SELECT
    column_name,
    COUNT(*)                       AS table_count,
    string_agg(table_name, ', ')   AS present_in  -- [LEARN] string_agg = 문자열 concat 집계
FROM cols
GROUP BY column_name
ORDER BY table_count DESC, column_name;


-- =============================================================================
-- SECTION 8. 다음 단계를 위한 체크리스트 (본인 답해보기)
-- =============================================================================
-- [과제 Q1] 6개 테이블 각각의 PK로 쓸 만한 컬럼은?
-- [과제 Q2] clinical 테이블들이 공통으로 갖는 컬럼 Top-N은? → unified 스키마 초안
-- [과제 Q3] "체중/신장" 이 text 인 이유는 뭘까? 어떻게 정리해야 할까?
-- [과제 Q4] "자원 수집일 (bigint)" 을 date로 바꾸는 ALTER TABLE 은?
-- [과제 Q5] box와 clinical을 항상 JOIN해야 한다면 VIEW로 만들어둘까?
-- =============================================================================
