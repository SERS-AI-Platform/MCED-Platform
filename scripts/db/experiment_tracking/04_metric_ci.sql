-- =====================================================================
-- 04_metric_ci.sql  —  run_metrics에 신뢰구간·표본수·평가종류 컬럼 추가
-- WHY: 성능 지표가 결과 폴더마다 CSV로 흩어져 있어 찾기 어렵고, 같은 "AUC"가
--      OOF/val/test 구분 없이 섞인다. 한 테이블에서 SQL 한 줄로 모든 run의
--      지표를 CI·표본수·코호트와 함께 조회할 수 있게 한다.
-- 실행 순서: 01_schema.sql 이후. ADD COLUMN IF NOT EXISTS라 재실행 안전.
-- 롤백: 아래 컬럼들을 DROP COLUMN (기존 행의 다른 컬럼은 영향 없음)
-- =====================================================================

BEGIN;

DO $$
BEGIN
    IF current_database() <> 'aecd_platform' THEN
        RAISE EXCEPTION 'Wrong database: %. Connect to aecd_platform.', current_database();
    END IF;
END
$$;

-- 신뢰구간과 그 근거
ALTER TABLE experiment.run_metrics
    ADD COLUMN IF NOT EXISTS ci_low    DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS ci_high   DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS ci_method TEXT,            -- 'bca' | 'percentile' | 'wilson' | 'undefined'
    ADD COLUMN IF NOT EXISTS n_boot    INTEGER,
    ADD COLUMN IF NOT EXISTS n_units   INTEGER,         -- 실질 표본수 = 재추출 단위(환자) 수
    ADD COLUMN IF NOT EXISTS n_rows    INTEGER;         -- 그 아래 행(스펙트럼) 수

-- 무엇을·어떻게 잰 값인지. 이게 없어서 OOF와 test가 섞였다.
ALTER TABLE experiment.run_metrics
    ADD COLUMN IF NOT EXISTS task        TEXT,          -- cancer_screening | cancer_type_id | prostate_three_class
    ADD COLUMN IF NOT EXISTS evaluation  TEXT,          -- oof | val | test_heldout | external
    ADD COLUMN IF NOT EXISTS aggregation TEXT;          -- patient | spectrum

-- 01_schema.sql의 split 어휘(train|val|test|oof)에 external(외부 코호트)을 추가.
-- split은 evaluation과 같은 값을 갖는다 — UNIQUE 제약의 일부라 남겨둔다.
ALTER TABLE experiment.run_metrics
    DROP CONSTRAINT IF EXISTS metrics_split_allowed;
ALTER TABLE experiment.run_metrics
    ADD CONSTRAINT metrics_split_allowed
        CHECK (split IS NULL OR split IN ('train', 'val', 'test', 'test_heldout', 'oof', 'external'));

ALTER TABLE experiment.run_metrics
    DROP CONSTRAINT IF EXISTS metrics_task_allowed,
    DROP CONSTRAINT IF EXISTS metrics_evaluation_allowed,
    DROP CONSTRAINT IF EXISTS metrics_ci_ordered;
ALTER TABLE experiment.run_metrics
    ADD CONSTRAINT metrics_task_allowed CHECK (task IS NULL OR task IN
        ('cancer_screening', 'cancer_type_id', 'prostate_three_class')),
    ADD CONSTRAINT metrics_evaluation_allowed CHECK (evaluation IS NULL OR evaluation IN
        ('oof', 'val', 'test_heldout', 'external')),
    ADD CONSTRAINT metrics_ci_ordered CHECK (
        ci_low IS NULL OR ci_high IS NULL OR ci_low <= ci_high);

-- 코호트 구성은 runs에 있다 (cancer_types[], non_cancer_groups[], n_subjects).
-- 여기에 암/정상 수를 추가해 특이도 검정력 판단이 가능하게 한다.
ALTER TABLE experiment.runs
    ADD COLUMN IF NOT EXISTS cohort_id   TEXT,
    ADD COLUMN IF NOT EXISTS site        TEXT,
    ADD COLUMN IF NOT EXISTS n_positive  INTEGER,
    ADD COLUMN IF NOT EXISTS n_negative  INTEGER;

CREATE INDEX IF NOT EXISTS run_metrics_task_eval_idx
    ON experiment.run_metrics(task, evaluation, metric_name);
CREATE INDEX IF NOT EXISTS runs_cohort_idx
    ON experiment.runs(cohort_id);

COMMIT;
