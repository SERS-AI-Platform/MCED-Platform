-- =====================================================================
-- 05_qc_and_measurement_context.sql — 지점 QC 결과와 측정 맥락을 experiment 스키마에 저장
-- WHY: 지점별 QC 통과/탈락, 스트립 판(unit), 검체별 측정 시각이 DB에 없어 분석할 때마다
--      원본 파일·경로에서 다시 계산했다(Phase BM-13, 2026-09-29). measurement 스키마는
--      읽기 전용 규칙이라, 사용자 결정에 따라 파생 테이블로 experiment 스키마에 둔다.
--        point_qc            측정 지점마다 QC 결과 (QC 버전별로 여러 벌 보관 가능)
--        sample_qc           검체마다 지점 탈락 수·비율과 파이프라인 검체 탈락(2b) 여부
--        measurement_context 측정(run × 검체)마다 스트립 판·측정 시각·그날 측정 순서
-- 실행 순서: 01_schema.sql 이후. CREATE ... IF NOT EXISTS라 재실행 안전.
-- 채우기: python scripts/db/experiment_tracking/load_qc_context.py (PG* 환경변수 필요)
-- 롤백: DROP TABLE experiment.point_qc, experiment.sample_qc, experiment.measurement_context;
-- =====================================================================

BEGIN;

DO $$
BEGIN
    IF current_database() <> 'aecd_platform' THEN
        RAISE EXCEPTION 'Wrong database: %. Connect to aecd_platform.', current_database();
    END IF;
END
$$;

-- 지점 QC. stage2a는 stage1을 통과한 지점에만 정의되므로 stage1 탈락이면 NULL.
CREATE TABLE IF NOT EXISTS experiment.point_qc (
    measurement_id  BIGINT      NOT NULL REFERENCES measurement.measurements(measurement_id),
    qc_version      TEXT        NOT NULL,   -- 예: 'stage1+2a_ver1_allpoints_20260929'
    stage1_pass     BOOLEAN     NOT NULL,   -- 세기 게이트·우주선·포화
    stage1_reason   TEXT,                   -- 탈락 사유 (세미콜론 구분)
    stage2a_pass    BOOLEAN,                -- 전처리 후 같은 검체 다른 지점 평균과의 상관
    run_id          BIGINT      REFERENCES experiment.runs(run_id),   -- 계산한 실험 run
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (measurement_id, qc_version),
    CONSTRAINT point_qc_stage2a_only_after_stage1
        CHECK (stage1_pass OR stage2a_pass IS NULL)
);

CREATE TABLE IF NOT EXISTS experiment.sample_qc (
    sample_id       BIGINT      NOT NULL REFERENCES master.samples(sample_id),
    qc_version      TEXT        NOT NULL,
    n_points        INTEGER     NOT NULL,   -- 측정 지점 수 (36 또는 121)
    n_fail          INTEGER     NOT NULL,   -- stage1 또는 stage2a 탈락 지점 수
    dropped_2b      BOOLEAN,                -- 분석 파이프라인에서 검체째 빠졌는가
    dropped_2b_rule TEXT,                   -- 예: '36점 서브샘플 후 25점 미만'
    run_id          BIGINT      REFERENCES experiment.runs(run_id),
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (sample_id, qc_version),
    CONSTRAINT sample_qc_counts CHECK (n_fail >= 0 AND n_fail <= n_points)
);

-- 측정 맥락. 원본에는 없고 경로·파일 시각에서 복원한 값이라 source에 방법을 남긴다.
CREATE TABLE IF NOT EXISTS experiment.measurement_context (
    measurement_run_id BIGINT   NOT NULL REFERENCES measurement.runs(measurement_run_id),
    sample_id       BIGINT      NOT NULL REFERENCES master.samples(sample_id),
    strip_unit_id   BIGINT      REFERENCES measurement.strip_units(strip_unit_id),
    strip_unit_label TEXT,                  -- 경로의 원문, 예: 'SK20260805B01-2'
    acquired_at     TIMESTAMP,              -- 검체 첫 지점 파일 수정 시각(장비 저장 시각, 현지 시간)
    order_in_day    INTEGER,                -- 같은 측정일 안에서 acquired_at 순서 (1부터)
    source          TEXT        NOT NULL,   -- 복원 방법
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (measurement_run_id, sample_id)
);

CREATE INDEX IF NOT EXISTS point_qc_run_idx ON experiment.point_qc(run_id);
CREATE INDEX IF NOT EXISTS sample_qc_run_idx ON experiment.sample_qc(run_id);
CREATE INDEX IF NOT EXISTS measurement_context_unit_idx ON experiment.measurement_context(strip_unit_id);

COMMIT;
