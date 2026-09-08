-- =====================================================================
-- LEGACY / NON-GOVERNED historical reproduction only
-- 01_schema.sql
-- WHY: std schema + 병원 lookup + 소스↔병원 매핑 VIEW + clinical_unified DDL
-- 실행 순서: 최초 한 번만. 이후 INSERT 스크립트 실행.
-- Do not use this schema or mapping as governed lineage input.
-- Canonical source contracts live in src/sers/master_data/clinical_inventory.py.
-- =====================================================================

CREATE SCHEMA IF NOT EXISTS std;

-- ---------------------------------------------------------------------
-- 병원 lookup
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS std.hospital_lookup CASCADE;
CREATE TABLE std.hospital_lookup (
    hospital_code    VARCHAR(10)  PRIMARY KEY,
    hospital_name    VARCHAR(100) NOT NULL,
    hospital_name_kr VARCHAR(100) NOT NULL
);

INSERT INTO std.hospital_lookup (hospital_code, hospital_name, hospital_name_kr) VALUES
('CBNUH', 'Chungbuk National University Hospital',       '충북대학교병원'),
('SNUH',  'Seoul National University Hospital',          '서울대학교병원'),
('IJBPH', 'Inje University Busan Paik Hospital',         '인제대학교 부산백병원'),
('SSMH',  'Seoul St. Mary''s Hospital',                  '서울성모병원'),
('YPNUH', 'Yangsan Pusan National University Hospital',  '양산부산대병원');

-- ---------------------------------------------------------------------
-- 원본 엑셀 파일 ↔ 질환그룹 ↔ 프로토콜 ↔ 병원 매핑 VIEW
-- ---------------------------------------------------------------------
DROP VIEW IF EXISTS std.v_source_hospital_map;
CREATE VIEW std.v_source_hospital_map AS
SELECT * FROM (VALUES
    ('SMCXD01_전립선암 임상정보.xlsx', 'PRO',  'SMCXD01', 'CBNUH'),
    ('SMCXD01_난소암 1.xlsx',          'OVA',  'SMCXD01', 'IJBPH'),
    ('SMCXD01_난소암 2.xlsx',          'OVA',  'SMCXD01', 'SNUH'),
    ('SMCXD01_폐암 1.xlsx',            'LUN',  'SMCXD01', 'SNUH'),
    ('SMCXD06_폐암 2.xlsx',            'LUN',  'SMCXD06', 'SSMH'),
    ('SMCXD06_폐암 3.xlsx',            'LUN',  'SMCXD06', 'SNUH'),
    ('SMCXD06_대장암.xlsx',            'CRC',  'SMCXD06', 'CBNUH'),
    ('SMCMD06_췌장암.xlsx',            'CPAN', 'SMCMD06', 'CBNUH'),
    ('SMCXD01_유방암.xlsx',            'BRE',  'SMCXD01', 'IJBPH'),
    ('SMCXD06_방광암.xlsm',            'BLC',  'SMCXD06', 'CBNUH'),
    ('SMCXD03_정상인 1.xlsx',          'NOR',  'SMCXD03', 'YPNUH'),
    ('SMCXD03_정상인 2.xlsx',          'NOR',  'SMCXD03', 'YPNUH'),
    ('SMCXD03_당뇨 1.xlsx',            'DIA',  'SMCXD03', 'YPNUH'),
    ('SMCXD03_당뇨 2.xlsx',            'DIA',  'SMCXD03', 'YPNUH'),
    ('SMCXD03_고혈압.xlsx',            'HBP',  'SMCXD03', 'YPNUH'),
    ('SMCXD05_당뇨+고혈압.xlsx',       'H.D.', 'SMCXD05', 'YPNUH')
) AS t(source_file, disease_group, protocol, hospital_code);

-- ---------------------------------------------------------------------
-- 통합 임상 테이블
-- WHY: config group_metadata 기준 질환그룹 공통 스키마
--      blood_ = 혈액검사, urine_ = 소변검사, hx_ = 과거력
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS std.clinical_unified;
CREATE TABLE std.clinical_unified (
    id              SERIAL PRIMARY KEY,

    -- 식별
    disease_group   VARCHAR(10)  NOT NULL,
    protocol        VARCHAR(20),
    hospital_code   VARCHAR(10),
    hospital_name   VARCHAR(100),
    source_file     VARCHAR(200),
    source_no       VARCHAR(50),
    resource_code   VARCHAR(50),
    provider_code   VARCHAR(50),
    lot_no          VARCHAR(50),
    solum_label     VARCHAR(50),

    -- 인구통계
    sex             VARCHAR(10),
    age             INTEGER,
    birth_year      INTEGER,
    birth_month     INTEGER,
    birth_day       INTEGER,
    height_cm       NUMERIC(5,1),
    weight_kg       NUMERIC(5,1),
    bmi             NUMERIC(4,1),

    -- 날짜
    collection_date DATE,
    diagnosis_date  DATE,
    surgery_date    DATE,
    chemo_date      DATE,

    -- 생활습관
    smoking           TEXT,
    smoking_years     INTEGER,
    smoking_per_day   INTEGER,
    drinking          TEXT,
    drinking_days     TEXT,
    smoking_status    VARCHAR(10),
    drinking_status   VARCHAR(10),

    -- 진단
    diagnosis_name  TEXT,
    fasting         VARCHAR(20),

    -- 과거력
    medical_history TEXT,
    hx_diabetes     VARCHAR(20),
    hx_hypertension VARCHAR(20),
    hx_hyperlipidemia VARCHAR(20),
    hx_angina       VARCHAR(20),
    hx_mi           VARCHAR(20),
    hx_stroke       VARCHAR(20),
    hx_hepatitis    VARCHAR(20),
    hx_tb           VARCHAR(20),
    hx_thyroid      VARCHAR(20),
    hx_surgery      VARCHAR(20),
    hx_other        TEXT,
    prior_cancer_flag BOOLEAN DEFAULT FALSE,

    -- 병리 (암환자)
    pathology_location TEXT,
    histologic_type    TEXT,
    grade              TEXT,
    tnm_stage          TEXT,
    t_stage            VARCHAR(20),
    n_stage            VARCHAR(20),
    m_stage            VARCHAR(20),
    cancer_stage       VARCHAR(20),
    cancer_stage_group VARCHAR(20),
    metastasis         TEXT,
    differentiation    TEXT,
    lvi                TEXT,
    treatment_info     TEXT,
    surgery_name       TEXT,

    -- 혈액검사
    blood_test_results    TEXT,
    blood_wbc             NUMERIC,
    blood_rbc             NUMERIC,
    blood_hb              NUMERIC,
    blood_hct             NUMERIC,
    blood_platelet        NUMERIC,
    blood_ast             NUMERIC,
    blood_alt             NUMERIC,
    blood_alp             NUMERIC,
    blood_ggt             NUMERIC,
    blood_bilirubin_total NUMERIC,
    blood_protein_total   NUMERIC,
    blood_albumin         NUMERIC,
    blood_bun             NUMERIC,
    blood_creatinine      NUMERIC,
    blood_egfr            NUMERIC,
    blood_uric_acid       NUMERIC,
    blood_glucose         NUMERIC,
    blood_hba1c           NUMERIC,
    blood_cholesterol     NUMERIC,
    blood_triglyceride    NUMERIC,
    blood_hdl             NUMERIC,
    blood_ldl             NUMERIC,
    blood_hscrp           NUMERIC,
    blood_afp             NUMERIC,
    blood_cea             NUMERIC,
    blood_ca125           NUMERIC,
    blood_ca199           NUMERIC,
    blood_ca153           NUMERIC,
    blood_psa             NUMERIC,
    blood_free_psa        NUMERIC,
    blood_tsh             NUMERIC,
    blood_ft4             NUMERIC,
    blood_insulin         NUMERIC,

    -- 소변검사
    urine_sg           NUMERIC,
    urine_ph           NUMERIC,
    urine_protein      TEXT,
    urine_glucose      TEXT,
    urine_ketone       TEXT,
    urine_bilirubin    TEXT,
    urine_urobilinogen TEXT,
    urine_nitrite      TEXT,
    urine_occult_blood TEXT,
    urine_leukocyte    TEXT,
    urine_color        TEXT,
    urine_clarity      TEXT,
    urine_rbc          NUMERIC,
    urine_wbc          NUMERIC,

    -- 혈압
    bp_systolic        INTEGER,
    bp_diastolic       INTEGER,

    -- 메타
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_cu_disease_group ON std.clinical_unified(disease_group);
CREATE INDEX idx_cu_protocol      ON std.clinical_unified(protocol);
CREATE INDEX idx_cu_hospital      ON std.clinical_unified(hospital_code);
CREATE INDEX idx_cu_solum_label   ON std.clinical_unified(solum_label);
CREATE INDEX idx_cu_resource_code ON std.clinical_unified(resource_code);

COMMENT ON TABLE std.clinical_unified IS
  'config group_metadata 기준 통합 임상 테이블 — blood_=혈액, urine_=소변';
