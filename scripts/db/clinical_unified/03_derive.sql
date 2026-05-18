-- =====================================================================
-- 03_derive.sql  —  파생 컬럼 계산
--   - hospital_code, hospital_name (source_file 기반 JOIN)
--   - smoking_status, drinking_status (never/former/current/unknown)
--   - prior_cancer_flag (본인 암 제외 후 타암 키워드)
--   - cancer_stage_group (early/advanced/unknown)
-- 각 암종별로 원본 형식이 달라서 disease_group별로 분기
-- =====================================================================

-- =====================================================================
-- (1) hospital_code / hospital_name 일괄 채우기
-- =====================================================================
UPDATE std.clinical_unified cu
SET hospital_code = m.hospital_code,
    hospital_name = h.hospital_name
FROM std.v_source_hospital_map m
JOIN std.hospital_lookup h ON h.hospital_code = m.hospital_code
WHERE cu.source_file = m.source_file;

-- =====================================================================
-- (2) PRO (전립선암, SMCXD01)
-- =====================================================================
UPDATE std.clinical_unified
SET smoking_status = CASE
    WHEN smoking::TEXT = '1' OR smoking ~* '^안함$|^비흡연$' THEN 'never'
    WHEN smoking::TEXT = '2' OR smoking ~* '^과거'            THEN 'former'
    WHEN smoking::TEXT = '3' OR smoking ~* '^함|^현재'        THEN 'current'
    ELSE 'unknown'
END,
drinking_status = CASE
    WHEN drinking::TEXT = '1' OR drinking ~* '^안함$|^비음주$' THEN 'never'
    WHEN drinking::TEXT = '2' OR drinking ~* '^과거'            THEN 'former'
    WHEN drinking::TEXT = '3' OR drinking ~* '^함|^현재'        THEN 'current'
    ELSE 'unknown'
END
WHERE disease_group = 'PRO';

-- prior_cancer_flag (본인 전립선암 제외 후 타암 검색)
UPDATE std.clinical_unified SET prior_cancer_flag = FALSE WHERE disease_group = 'PRO';
UPDATE std.clinical_unified
SET prior_cancer_flag = TRUE
WHERE disease_group = 'PRO'
  AND medical_history IS NOT NULL
  AND (
    regexp_replace(medical_history,
      '(?i)prostate\s*(cancer|ca)|전립선암|PCa\b', '', 'gi')
    ~* '암|cancer|carcinoma|sarcoma|tumor|[A-Za-z]CA(?=[,\s]|$)|HCC'
  );

UPDATE std.clinical_unified
SET cancer_stage_group = CASE
    WHEN tnm_stage IS NULL OR TRIM(tnm_stage) = '' THEN 'unknown'
    WHEN tnm_stage ~* 'T[3-9]|T1[0-9]' THEN 'advanced'
    WHEN metastasis ~* 'bone'          THEN 'advanced'
    ELSE 'early'
END
WHERE disease_group = 'PRO';

-- =====================================================================
-- (3) OVA (난소암, SMCXD01)
-- =====================================================================
UPDATE std.clinical_unified
SET smoking_status = CASE
    WHEN smoking::TEXT = '1' OR smoking ~* '^안함$|^비흡연$' THEN 'never'
    WHEN smoking::TEXT = '2' OR smoking ~* '^과거'            THEN 'former'
    WHEN smoking::TEXT = '3' OR smoking ~* '^함|^현재'        THEN 'current'
    WHEN smoking ~* '갑|pack|PY|Y$'                           THEN 'current'
    ELSE 'unknown'
END,
drinking_status = CASE
    WHEN drinking::TEXT = '1' OR drinking ~* '^안함$|^비음주$' THEN 'never'
    WHEN drinking::TEXT = '2' OR drinking ~* '^과거'            THEN 'former'
    WHEN drinking::TEXT = '3' OR drinking ~* '^함|^현재'        THEN 'current'
    ELSE 'unknown'
END
WHERE disease_group = 'OVA';

UPDATE std.clinical_unified
SET prior_cancer_flag = TRUE
WHERE disease_group = 'OVA'
  AND medical_history IS NOT NULL
  AND (
    regexp_replace(medical_history,
      '(?i)ovarian?\s*(cancer|ca|tumor)|ovary\s*(cancer|ca)|peritoneal carcinomatosis',
      '', 'gi')
    ~* '암|cancer|carcinoma|sarcoma|tumor|[A-Za-z]CA(?=[,\s]|$)|HCC'
  );

UPDATE std.clinical_unified
SET cancer_stage_group = CASE
    WHEN tnm_stage IS NULL OR TRIM(tnm_stage) = ''       THEN 'unknown'
    WHEN tnm_stage ~* '^T1(?![0-9])' AND tnm_stage !~* 'M1' THEN 'early'
    WHEN tnm_stage ~* '^T[2-9]|M1'                        THEN 'advanced'
    ELSE 'unknown'
END
WHERE disease_group = 'OVA';

-- =====================================================================
-- (4) LUN (폐암) — 파일별 원본 형식 상이
-- =====================================================================
-- LUN1 (SMCXD01, 폐암 1): staging 미비 → unknown
UPDATE std.clinical_unified
SET smoking_status = CASE
    WHEN smoking ~* 'ex.?smoker|ex,|quit|중단|stop' THEN 'former'
    WHEN smoking ~* '갑/일|PY|ppd'                   THEN 'current'
    ELSE 'unknown'
END,
drinking_status = CASE
    WHEN drinking ~* '병/월|소주|맥주|alcohol|social' THEN 'current'
    ELSE 'unknown'
END,
cancer_stage_group = 'unknown'
WHERE disease_group = 'LUN' AND source_file = 'SMCXD01_폐암 1.xlsx';

UPDATE std.clinical_unified
SET prior_cancer_flag = TRUE
WHERE disease_group = 'LUN'
  AND source_file = 'SMCXD01_폐암 1.xlsx'
  AND medical_history IS NOT NULL
  AND (
    regexp_replace(medical_history, '(?i)lung\s*(cancer|ca|tumor)|폐암', '', 'gi')
    ~* '암|cancer|carcinoma|sarcoma|tumor|[A-Za-z]CA(?=[,\s]|$)|HCC'
  );

-- LUN2 (SMCXD06, 폐암 2): staging from tnm_stage + metastasis
UPDATE std.clinical_unified
SET smoking_status  = 'unknown',
    drinking_status = 'unknown'
WHERE disease_group = 'LUN' AND source_file = 'SMCXD06_폐암 2.xlsx';

UPDATE std.clinical_unified
SET prior_cancer_flag = TRUE
WHERE disease_group = 'LUN'
  AND source_file = 'SMCXD06_폐암 2.xlsx'
  AND (
    metastasis ~* 'from|stomach ca|breast ca|colon ca|rectal cancer|sigmoid'
    OR tnm_stage ~* 'Esophgeal ca'
  );

UPDATE std.clinical_unified
SET cancer_stage_group = CASE
    WHEN tnm_stage IS NULL OR TRIM(tnm_stage) = ''          THEN 'unknown'
    WHEN tnm_stage ~* 'T[3-9]|T[1-9]\d|M1'                   THEN 'advanced'
    ELSE 'early'
END
WHERE disease_group = 'LUN' AND source_file = 'SMCXD06_폐암 2.xlsx';

-- LUN3 (SMCXD06, 폐암 3): smoking/drinking '.'=never, 'O'=current
UPDATE std.clinical_unified
SET smoking_status = CASE
    WHEN TRIM(smoking::TEXT) = '.' THEN 'never'
    WHEN TRIM(smoking::TEXT) = 'O' THEN 'current'
    ELSE 'unknown'
END,
drinking_status = CASE
    WHEN TRIM(drinking::TEXT) = '.' THEN 'never'
    WHEN TRIM(drinking::TEXT) = 'O' THEN 'current'
    ELSE 'unknown'
END
WHERE disease_group = 'LUN' AND source_file = 'SMCXD06_폐암 3.xlsx';

UPDATE std.clinical_unified
SET prior_cancer_flag = TRUE
WHERE disease_group = 'LUN'
  AND source_file = 'SMCXD06_폐암 3.xlsx'
  AND medical_history IS NOT NULL
  AND (
    regexp_replace(medical_history, '(?i)lung\s*(cancer|ca|tumor)|폐암', '', 'gi')
    ~* '암|cancer|carcinoma|sarcoma|tumor|[A-Za-z]CA(?=[,\s]|$)|HCC'
  );

UPDATE std.clinical_unified
SET cancer_stage_group = CASE
    WHEN tnm_stage IS NULL OR TRIM(tnm_stage) = '' THEN 'unknown'
    WHEN tnm_stage ~* 'T[3-9]|M1'                   THEN 'advanced'
    ELSE 'early'
END
WHERE disease_group = 'LUN' AND source_file = 'SMCXD06_폐암 3.xlsx';

-- =====================================================================
-- (5) CRC (대장암, SMCXD06)
--   WHY: 원본 엑셀 2시트 구조 기반 — sheet1(1~270)=early, sheet2(271~300)=advanced
-- =====================================================================
UPDATE std.clinical_unified
SET smoking_status  = 'unknown',
    drinking_status = 'unknown'
WHERE disease_group = 'CRC';

UPDATE std.clinical_unified
SET prior_cancer_flag = TRUE
WHERE disease_group = 'CRC'
  AND medical_history IS NOT NULL
  AND (
    regexp_replace(medical_history,
      '(?i)colon\s*(cancer|ca)|rectal\s*(cancer|ca)|대장암|직장암|colorectal',
      '', 'gi')
    ~* '암|cancer|carcinoma|sarcoma|tumor|[A-Za-z]CA(?=[,\s]|$)|HCC'
  );

UPDATE std.clinical_unified
SET cancer_stage_group = CASE
    WHEN source_no ~ '^\d+$' AND source_no::INTEGER <= 270 THEN 'early'
    WHEN source_no ~ '^\d+$' AND source_no::INTEGER >  270 THEN 'advanced'
    ELSE 'unknown'
END
WHERE disease_group = 'CRC';

-- =====================================================================
-- (6) CPAN (췌장암, SMCMD06)
-- =====================================================================
UPDATE std.clinical_unified
SET smoking_status  = 'unknown',
    drinking_status = 'unknown'
WHERE disease_group = 'CPAN';

UPDATE std.clinical_unified
SET prior_cancer_flag = TRUE
WHERE disease_group = 'CPAN'
  AND medical_history IS NOT NULL
  AND (
    regexp_replace(medical_history,
      '(?i)pancrea(tic|s)\s*(cancer|ca|tumor)?|췌장암|PDAC',
      '', 'gi')
    ~* '암|cancer|carcinoma|sarcoma|tumor|[A-Za-z]CA(?=[,\s]|$)|HCC'
  );

UPDATE std.clinical_unified
SET cancer_stage_group = CASE
    WHEN t_stage IS NULL AND m_stage IS NULL THEN 'unknown'
    WHEN m_stage ~* 'M1'                      THEN 'advanced'
    WHEN t_stage ~* '^T[3-9]'                 THEN 'advanced'
    WHEN t_stage ~* '^T[12]'                  THEN 'early'
    ELSE 'unknown'
END
WHERE disease_group = 'CPAN';
