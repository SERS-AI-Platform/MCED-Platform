-- =====================================================================
-- 03_design_guide.sql  —  전처리 설계 가이드(docx) → experiment 스키마
-- WHY: docs/ml/preprocessing_design_guide.md 의 레퍼런스 24편과 단계별 method를
--      DB에 등록해, 앞으로의 실험이 논문 출처와 FK로 연결되게 한다.
-- 실행 순서: 01_schema.sql 이후. 멱등(ON CONFLICT)이라 재실행 안전.
--
-- 주의: title/DOI는 채우지 않는다. Vancouver 인용문에서 기계적으로 분리하면
--       오분리 위험이 있어, 인용문 전체를 citation에 원문 그대로 보관한다.
-- =====================================================================

BEGIN;

DO $$
BEGIN
    IF current_database() <> 'aecd_platform' THEN
        RAISE EXCEPTION 'Wrong database: %. Connect to aecd_platform.', current_database();
    END IF;
END
$$;

-- ---------------------------------------------------------------------
-- (1) stage 어휘 확장
-- 가이드는 8단계(calibration/despike/truncation/denoise/baseline/
-- normalization/순서최적화/model transfer)인데 최초 스키마는 4개 값만 허용했다.
-- ---------------------------------------------------------------------
ALTER TABLE experiment.preprocessing_methods
    DROP CONSTRAINT IF EXISTS methods_stage_allowed;
ALTER TABLE experiment.preprocessing_methods
    ADD CONSTRAINT methods_stage_allowed CHECK (stage IN (
        'calibration', 'despike', 'truncation', 'smoothing',
        'baseline', 'normalization', 'transfer', 'combo'
    ));

-- ---------------------------------------------------------------------
-- (2) 레퍼런스 24편
-- ---------------------------------------------------------------------
INSERT INTO experiment.papers (citation, source_tool, notes) VALUES
    ('ASTM E1840-96(2014). Standard guide for Raman shift standards for spectrometer calibration. West Conshohocken (PA): ASTM International; 2014.', 'design_guide_docx', '설계가이드 참조 [1]'),
    ('Bocklitz TW, Dörfer T, Heinke R, Schmitt M, Popp J. Spectrometer calibration protocol for Raman spectra recorded with different excitation wavelengths. Spectrochim Acta A Mol Biomol Spectrosc. 2015;149:544–9.', 'design_guide_docx', '설계가이드 참조 [2]'),
    ('Guo S, Popp J, Bocklitz T. Chemometric analysis in Raman spectroscopy from experimental design to machine learning–based modeling. Nat Protoc. 2021;16(12):5426–59.', 'design_guide_docx', '설계가이드 참조 [3]'),
    ('Whitaker DA, Hayes K. A simple algorithm for despiking Raman spectra. Chemom Intell Lab Syst. 2018;179:82–4.', 'design_guide_docx', '설계가이드 참조 [4]'),
    ('Ryabchykov O, Bocklitz T, Ramoji A, Neugebauer U, Förster M, Kroegel C, et al. Automatization of spike correction in Raman spectra of biological samples. Chemom Intell Lab Syst. 2016;155:1–6.', 'design_guide_docx', '설계가이드 참조 [5]'),
    ('Afseth NK, Segtnan VH, Wold JP. Raman spectra of biological samples: a study of preprocessing methods. Appl Spectrosc. 2006;60(12):1358–67.', 'design_guide_docx', '설계가이드 참조 [6]'),
    ('Savitzky A, Golay MJE. Smoothing and differentiation of data by simplified least squares procedures. Anal Chem. 1964;36(8):1627–39.', 'design_guide_docx', '설계가이드 참조 [7]'),
    ('Barton SJ, Ward TE, Hennelly BM. Algorithm for optimal denoising of Raman spectra. Anal Methods. 2018;10(30):3759–69.', 'design_guide_docx', '설계가이드 참조 [8]'),
    ('Han M, Dang Y, Han J. Denoising and baseline correction methods for Raman spectroscopy based on convolutional autoencoder: a unified solution. Sensors (Basel). 2024;24(10):3161.', 'design_guide_docx', '설계가이드 참조 [9]'),
    ('Bai Y, Liu Q. Denoising Raman spectra by Wiener estimation with a numerical calibration dataset. Biomed Opt Express. 2020;11(1):200–14.', 'design_guide_docx', '설계가이드 참조 [10]'),
    ('Lieber CA, Mahadevan-Jansen A. Automated method for subtraction of fluorescence from biological Raman spectra. Appl Spectrosc. 2003;57(11):1363–7.', 'design_guide_docx', '설계가이드 참조 [11]'),
    ('Zhao J, Lui H, McLean DI, Zeng H. Automated autofluorescence background subtraction algorithm for biomedical Raman spectroscopy. Appl Spectrosc. 2007;61(11):1225–32.', 'design_guide_docx', '설계가이드 참조 [12]'),
    ('Eilers PHC, Boelens HFM. Baseline correction with asymmetric least squares smoothing. Leiden: Leiden University Medical Centre Report; 2005.', 'design_guide_docx', '설계가이드 참조 [13]'),
    ('Zhang ZM, Chen S, Liang YZ. Baseline correction using adaptive iteratively reweighted penalized least squares. Analyst. 2010;135(5):1138–46.', 'design_guide_docx', '설계가이드 참조 [14]'),
    ('Baek SJ, Park A, Ahn YJ, Choo J. Baseline correction using asymmetrically reweighted penalized least squares smoothing. Analyst. 2015;140(1):250–7.', 'design_guide_docx', '설계가이드 참조 [15]'),
    ('Bocklitz T, Walter A, Hartmann K, Rösch P, Popp J. How to pre-process Raman spectra for reliable and stable models? Anal Chim Acta. 2011;704(1–2):47–56.', 'design_guide_docx', '설계가이드 참조 [16]'),
    ('Butler HJ, Ashton L, Bird B, Cinque G, Curtis K, Dorney J, et al. Using Raman spectroscopy to characterize biological materials. Nat Protoc. 2016;11(4):664–87.', 'design_guide_docx', '설계가이드 참조 [17]'),
    ('Bell SEJ, Charron G, Cortés E, Kneipp J, Lamy de la Chapelle M, Langer J, et al. Towards reliable and quantitative surface-enhanced Raman scattering (SERS): from key parameters to good analytical practice. Angew Chem Int Ed. 2020;59(14):5454–62.', 'design_guide_docx', '설계가이드 참조 [18]'),
    ('Guo S, Kohler A, Zimmermann B, Heinke R, Stöckel S, Rösch P, et al. Extended multiplicative signal correction based model transfer for Raman spectroscopy in biological applications. Anal Chem. 2018;90(16):9787–95.', 'design_guide_docx', '설계가이드 참조 [19]'),
    ('Guo S, Heinke R, Stöckel S, Rösch P, Bocklitz T, Popp J. Towards an improvement of model transferability for Raman spectroscopy in biological applications. Vib Spectrosc. 2017;91:111–8.', 'design_guide_docx', '설계가이드 참조 [20]'),
    ('Zhao Y. On the measurements of the surface-enhanced Raman scattering spectrum: effective enhancement factor, optical configuration, spectral distortion, and baseline variation. Nanomaterials (Basel). 2023;13(23):2998.', 'design_guide_docx', '설계가이드 참조 [21]'),
    ('Picot F, Dallaire F, Daoust F, Chaikho L, Sheehy G, Bégin T, et al. Data consistency and classification model transferability across biomedical Raman spectroscopy systems. Transl Biophotonics. 2021;3(4):e202000019.', 'design_guide_docx', '설계가이드 참조 [22]'),
    ('Liu Q, Azziz A, Cucuiet V, Majdinasab M, Arib C, Yang X, et al. Investigating the reproducibility and repeatability of commercial SERS substrates using a new methodological approach. Anal Methods. 2026;18(9):1917–27.', 'design_guide_docx', '설계가이드 참조 [23]'),
    ('Xue B, Bi X, Dong Z, Xu Y, Liang M, Fang X, et al. Deep spectral component filtering as a foundation model for spectral analysis demonstrated in metabolic profiling. Nat Mach Intell. 2025;7(5):743–57.', 'design_guide_docx', '설계가이드 참조 [24]')
ON CONFLICT DO NOTHING;

-- ---------------------------------------------------------------------
-- (3) 가이드가 명시한 method 등록
-- audit_status는 전부 'pending' — 구현체가 이미 있는 것도 "논문 핵심 아이디어와
-- 일치하는지" 사람 검수를 아직 받지 않았으므로 approved로 올리지 않는다.
-- ---------------------------------------------------------------------
INSERT INTO experiment.preprocessing_methods
    (method_key, stage, display_name, paper_id, implementation_path, audit_status, config_overrides)
SELECT 'calibration_astm_reference', 'calibration', '파수축 표준물질 보정 (ASTM E1840)', p.paper_id, 'src/sers/preprocessing.py:calibrate_spectrum', 'pending',
       jsonb_build_object('note', '구현체는 있으나 preprocess_spectra 파이프라인에 연결돼 있지 않음 — 배선 필요')
FROM experiment.papers p WHERE p.notes = '설계가이드 참조 [1]'
ON CONFLICT (method_key) DO UPDATE SET
    stage = EXCLUDED.stage, display_name = EXCLUDED.display_name,
    paper_id = EXCLUDED.paper_id, implementation_path = EXCLUDED.implementation_path,
    config_overrides = EXCLUDED.config_overrides;
INSERT INTO experiment.preprocessing_methods
    (method_key, stage, display_name, paper_id, implementation_path, audit_status, config_overrides)
SELECT 'despike_whitaker_hayes', 'despike', 'Whitaker–Hayes modified Z-score despiking', p.paper_id, NULL, 'pending',
       jsonb_build_object('note', '미구현. 가이드 ②단계. 1차 차분 detrended 스펙트럼의 modified Z-score')
FROM experiment.papers p WHERE p.notes = '설계가이드 참조 [4]'
ON CONFLICT (method_key) DO UPDATE SET
    stage = EXCLUDED.stage, display_name = EXCLUDED.display_name,
    paper_id = EXCLUDED.paper_id, implementation_path = EXCLUDED.implementation_path,
    config_overrides = EXCLUDED.config_overrides;
INSERT INTO experiment.preprocessing_methods
    (method_key, stage, display_name, paper_id, implementation_path, audit_status, config_overrides)
SELECT 'despike_ryabchykov', 'despike', 'Ryabchykov 자동 spike 보정', p.paper_id, NULL, 'pending',
       jsonb_build_object('note', '미구현. ②단계 대안')
FROM experiment.papers p WHERE p.notes = '설계가이드 참조 [5]'
ON CONFLICT (method_key) DO UPDATE SET
    stage = EXCLUDED.stage, display_name = EXCLUDED.display_name,
    paper_id = EXCLUDED.paper_id, implementation_path = EXCLUDED.implementation_path,
    config_overrides = EXCLUDED.config_overrides;
INSERT INTO experiment.preprocessing_methods
    (method_key, stage, display_name, paper_id, implementation_path, audit_status, config_overrides)
SELECT 'truncation_fingerprint', 'truncation', 'fingerprint 영역 절단 (예: 600–1800 cm⁻¹)', p.paper_id, 'src/sers/preprocessing.py:trim_spectrum', 'pending',
       jsonb_build_object('note', '구현됨. 절단 경계는 config trim_region')
FROM experiment.papers p WHERE p.notes = '설계가이드 참조 [6]'
ON CONFLICT (method_key) DO UPDATE SET
    stage = EXCLUDED.stage, display_name = EXCLUDED.display_name,
    paper_id = EXCLUDED.paper_id, implementation_path = EXCLUDED.implementation_path,
    config_overrides = EXCLUDED.config_overrides;
INSERT INTO experiment.preprocessing_methods
    (method_key, stage, display_name, paper_id, implementation_path, audit_status, config_overrides)
SELECT 'savgol', 'smoothing', 'Savitzky–Golay 다항 평활', p.paper_id, 'src/sers/preprocessing.py:smooth_spectrum', 'pending',
       jsonb_build_object('note', '구현됨 (창 길이 m, 차수 p)')
FROM experiment.papers p WHERE p.notes = '설계가이드 참조 [7]'
ON CONFLICT (method_key) DO UPDATE SET
    stage = EXCLUDED.stage, display_name = EXCLUDED.display_name,
    paper_id = EXCLUDED.paper_id, implementation_path = EXCLUDED.implementation_path,
    config_overrides = EXCLUDED.config_overrides;
INSERT INTO experiment.preprocessing_methods
    (method_key, stage, display_name, paper_id, implementation_path, audit_status, config_overrides)
SELECT 'denoise_barton_optimal', 'smoothing', 'Barton 최적 denoising', p.paper_id, NULL, 'pending',
       jsonb_build_object('note', '미구현. ④단계 대안')
FROM experiment.papers p WHERE p.notes = '설계가이드 참조 [8]'
ON CONFLICT (method_key) DO UPDATE SET
    stage = EXCLUDED.stage, display_name = EXCLUDED.display_name,
    paper_id = EXCLUDED.paper_id, implementation_path = EXCLUDED.implementation_path,
    config_overrides = EXCLUDED.config_overrides;
INSERT INTO experiment.preprocessing_methods
    (method_key, stage, display_name, paper_id, implementation_path, audit_status, config_overrides)
SELECT 'denoise_wiener_estimation', 'smoothing', 'Wiener estimation 기반 denoising', p.paper_id, NULL, 'pending',
       jsonb_build_object('note', '미구현. ④단계 대안')
FROM experiment.papers p WHERE p.notes = '설계가이드 참조 [10]'
ON CONFLICT (method_key) DO UPDATE SET
    stage = EXCLUDED.stage, display_name = EXCLUDED.display_name,
    paper_id = EXCLUDED.paper_id, implementation_path = EXCLUDED.implementation_path,
    config_overrides = EXCLUDED.config_overrides;
INSERT INTO experiment.preprocessing_methods
    (method_key, stage, display_name, paper_id, implementation_path, audit_status, config_overrides)
SELECT 'denoise_cdae', 'smoothing', 'CDAE 통합 denoise+baseline', p.paper_id, NULL, 'pending',
       jsonb_build_object('note', '미구현. 딥러닝. 가이드가 FWHM 창 peak fidelity 검증 없이는 임상 적용 불가라고 명시')
FROM experiment.papers p WHERE p.notes = '설계가이드 참조 [9]'
ON CONFLICT (method_key) DO UPDATE SET
    stage = EXCLUDED.stage, display_name = EXCLUDED.display_name,
    paper_id = EXCLUDED.paper_id, implementation_path = EXCLUDED.implementation_path,
    config_overrides = EXCLUDED.config_overrides;
INSERT INTO experiment.preprocessing_methods
    (method_key, stage, display_name, paper_id, implementation_path, audit_status, config_overrides)
SELECT 'baseline_modpoly_lieber', 'baseline', 'ModPoly 자동 형광 차감 (Lieber)', p.paper_id, NULL, 'pending',
       jsonb_build_object('note', '미구현. ⑤단계')
FROM experiment.papers p WHERE p.notes = '설계가이드 참조 [11]'
ON CONFLICT (method_key) DO UPDATE SET
    stage = EXCLUDED.stage, display_name = EXCLUDED.display_name,
    paper_id = EXCLUDED.paper_id, implementation_path = EXCLUDED.implementation_path,
    config_overrides = EXCLUDED.config_overrides;
INSERT INTO experiment.preprocessing_methods
    (method_key, stage, display_name, paper_id, implementation_path, audit_status, config_overrides)
SELECT 'baseline_zhao_autofluorescence', 'baseline', 'Zhao 자동 autofluorescence 차감', p.paper_id, NULL, 'pending',
       jsonb_build_object('note', '미구현. ⑤단계 대안')
FROM experiment.papers p WHERE p.notes = '설계가이드 참조 [12]'
ON CONFLICT (method_key) DO UPDATE SET
    stage = EXCLUDED.stage, display_name = EXCLUDED.display_name,
    paper_id = EXCLUDED.paper_id, implementation_path = EXCLUDED.implementation_path,
    config_overrides = EXCLUDED.config_overrides;
INSERT INTO experiment.preprocessing_methods
    (method_key, stage, display_name, paper_id, implementation_path, audit_status, config_overrides)
SELECT 'als', 'baseline', 'Asymmetric Least Squares (Eilers)', p.paper_id, 'src/sers/preprocessing.py:estimate_baseline', 'pending',
       jsonb_build_object('note', '구현됨')
FROM experiment.papers p WHERE p.notes = '설계가이드 참조 [13]'
ON CONFLICT (method_key) DO UPDATE SET
    stage = EXCLUDED.stage, display_name = EXCLUDED.display_name,
    paper_id = EXCLUDED.paper_id, implementation_path = EXCLUDED.implementation_path,
    config_overrides = EXCLUDED.config_overrides;
INSERT INTO experiment.preprocessing_methods
    (method_key, stage, display_name, paper_id, implementation_path, audit_status, config_overrides)
SELECT 'airpls', 'baseline', 'adaptive iteratively reweighted PLS', p.paper_id, 'src/sers/preprocessing.py:estimate_baseline', 'pending',
       jsonb_build_object('note', '구현됨')
FROM experiment.papers p WHERE p.notes = '설계가이드 참조 [14]'
ON CONFLICT (method_key) DO UPDATE SET
    stage = EXCLUDED.stage, display_name = EXCLUDED.display_name,
    paper_id = EXCLUDED.paper_id, implementation_path = EXCLUDED.implementation_path,
    config_overrides = EXCLUDED.config_overrides;
INSERT INTO experiment.preprocessing_methods
    (method_key, stage, display_name, paper_id, implementation_path, audit_status, config_overrides)
SELECT 'arpls', 'baseline', 'asymmetrically reweighted PLS', p.paper_id, 'src/sers/preprocessing.py:estimate_baseline', 'pending',
       jsonb_build_object('note', '구현됨')
FROM experiment.papers p WHERE p.notes = '설계가이드 참조 [15]'
ON CONFLICT (method_key) DO UPDATE SET
    stage = EXCLUDED.stage, display_name = EXCLUDED.display_name,
    paper_id = EXCLUDED.paper_id, implementation_path = EXCLUDED.implementation_path,
    config_overrides = EXCLUDED.config_overrides;
INSERT INTO experiment.preprocessing_methods
    (method_key, stage, display_name, paper_id, implementation_path, audit_status, config_overrides)
SELECT 'l2', 'normalization', 'vector norm x/‖x‖₂', p.paper_id, 'src/sers/preprocessing.py:normalize_spectrum', 'pending',
       jsonb_build_object('note', '구현됨')
FROM experiment.papers p WHERE p.notes = '설계가이드 참조 [16]'
ON CONFLICT (method_key) DO UPDATE SET
    stage = EXCLUDED.stage, display_name = EXCLUDED.display_name,
    paper_id = EXCLUDED.paper_id, implementation_path = EXCLUDED.implementation_path,
    config_overrides = EXCLUDED.config_overrides;
INSERT INTO experiment.preprocessing_methods
    (method_key, stage, display_name, paper_id, implementation_path, audit_status, config_overrides)
SELECT 'snv', 'normalization', 'Standard Normal Variate', p.paper_id, 'src/sers/preprocessing.py:normalize_spectrum', 'pending',
       jsonb_build_object('note', '구현됨. 현재 프로덕션 기본값')
FROM experiment.papers p WHERE p.notes = '설계가이드 참조 [17]'
ON CONFLICT (method_key) DO UPDATE SET
    stage = EXCLUDED.stage, display_name = EXCLUDED.display_name,
    paper_id = EXCLUDED.paper_id, implementation_path = EXCLUDED.implementation_path,
    config_overrides = EXCLUDED.config_overrides;
INSERT INTO experiment.preprocessing_methods
    (method_key, stage, display_name, paper_id, implementation_path, audit_status, config_overrides)
SELECT 'minmax', 'normalization', 'min–max band normalization', p.paper_id, 'src/sers/preprocessing.py:normalize_spectrum', 'pending',
       jsonb_build_object('note', '구현됨')
FROM experiment.papers p WHERE p.notes = '설계가이드 참조 [18]'
ON CONFLICT (method_key) DO UPDATE SET
    stage = EXCLUDED.stage, display_name = EXCLUDED.display_name,
    paper_id = EXCLUDED.paper_id, implementation_path = EXCLUDED.implementation_path,
    config_overrides = EXCLUDED.config_overrides;
INSERT INTO experiment.preprocessing_methods
    (method_key, stage, display_name, paper_id, implementation_path, audit_status, config_overrides)
SELECT 'emsc', 'transfer', 'EMSC 기반 model transfer', p.paper_id, 'src/sers/preprocessing.py:normalize_spectrum', 'pending',
       jsonb_build_object('note', '구현됨. 장비간 전이는 calibration_transfer.py(PDS)도 참조')
FROM experiment.papers p WHERE p.notes = '설계가이드 참조 [20]'
ON CONFLICT (method_key) DO UPDATE SET
    stage = EXCLUDED.stage, display_name = EXCLUDED.display_name,
    paper_id = EXCLUDED.paper_id, implementation_path = EXCLUDED.implementation_path,
    config_overrides = EXCLUDED.config_overrides;

COMMIT;