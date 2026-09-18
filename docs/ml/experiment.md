# SERS-AI Experiment Tracker

> **Single source of truth** for experiment planning and status.
> 기계용 이력은 `logs/experiment_registry.json` 참조.
> **통합 문서 원칙 (2026-08-31)**: 개별 run마다 별도 `docs/ml/*.md` 리포트를 만들지 않고 여기에 직접 기록한다.
> 이전에 별도 문서였던 `ALL_SOURCE_SCREENING_20260827.md`, `mapping_preprocessing_ablation_20260826.md`는
> 아래 Phase AD/AE/AC에 전체 내용을 병합하고 삭제했다.
> **수정 후 반드시 즉시 커밋할 것** — 2026-08-31에 커밋 전 상태로 두었다가 동시 작업 세션의 `git reset`으로
> 5개월치 append 작업이 통째로 유실된 사고가 있었음.

## 상태 표기
- `[ ]` TODO — 실행 대상
- `[~]` 진행중
- `[x]` 완료
- `[!]` 실패/재실행 필요

---

## Completed Experiments

### Phase A~B: 초기 탐색 (2026-02~03)
- [x] Phase A: Initial medoid baseline (PAN merged) ✅ (2026-02-26)
  - Det AUC 0.793, Id F1 0.356 | 6-cancer, medoid, 1,272 samples
- [x] Phase A-v2: Improved medoid baseline (CPAN/SPAN separated) ✅ (2026-02-27)
  - Det AUC 0.948, Id F1 0.548 | 7-cancer (CPAN+SPAN), medoid, 1,342 samples
- [x] Phase B: All-spectra training ✅ (2026-02-27)
  - Det AUC 0.958, Id F1 0.613 | 7-cancer, all spectra, 6,710 samples
  - Key insight: medoid → all spectra = +6.5pp F1

### Phase F~K: 모델 벤치마크 (2026-03)
- [x] Phase F: Full benchmark (5 cancers, classical baselines) ✅ (2026-03-13)
  - Det AUC 0.981, Id F1 0.87 | LR established as strongest baseline
  - Key insight: Deep learning does NOT beat classical models
- [x] Phase K: Ensemble (80% LR + 20% ResNet18) ✅ (2026-03-17)
  - Det AUC 0.984, Id F1 0.875 | Marginal +0.3pp F1 over pure LR

### Phase L~M: Ablation & Confounding (2026-03)
- [x] Phase L: Normalization ablation ✅ (2026-03)
  - Key insight: Linear separability is intrinsic to data, not artifact of normalization
- [x] Phase M: Confounding analysis ✅ (2026-03)
  - Key insight: SERS detects metabolites, not demographics

### Phase N~Q: Fusion & Constraint (2026-03)
- [x] Phase N: Early fusion (SERS + age/sex/BMI) ✅ (2026-03-18)
  - Det AUC 0.988, Id F1 0.877 | 5-cancer, fusion, 5,050 samples
- [x] Phase P: Held-out test validation (SERS-only) ✅ (2026-03-18)
  - Det AUC 0.977±0.005, Id F1 0.852±0.026 | CV estimates confirmed
- [x] Phase P-fusion: Held-out test validation (Fusion) ✅ (2026-03-18)
  - Det AUC 0.986±0.004, Id F1 0.877±0.018
- [x] Phase Q: SERS + sex constraint ✅ (2026-03-19) ⭐ **BEST 5-cancer**
  - Det AUC 0.977±0.005, Id F1 0.892±0.026
  - Key insight: +4.0pp F1 from sex constraint alone, zero model retraining
- [x] Phase Q-fusion: Fusion + sex constraint ✅ (2026-03-19)
  - Det AUC 0.986±0.004, Id F1 0.884±0.018

### Phase T~V: BLC 추가 & Fixed Grid (2026-03-23)
- [x] Phase T: BLC added (8-class medoid) ✅ (2026-03-23)
  - Det AUC 0.969, Id F1 0.738 | BLC 299 samples (충북대학교병원)
- [x] Phase V-6cancer: Fixed grid 6-cancer ✅ (2026-03-23)
  - Det AUC 0.972, Id F1 0.869 | PAN = merged CPAN+SPAN
- [x] Phase V-7cancer: Fixed grid 7-cancer ✅ (2026-03-23)
  - Det AUC 0.98, Id F1 0.844 | CPAN+SPAN separate, BRE dropped
- [x] Phase V-8cancer: Fixed grid 8-cancer ✅ (2026-03-23)
  - Det AUC 0.98, Id F1 0.813 | All 8 types incl BRE (n=30)

### Phase U~W: BRE 재추가 & 7-cancer 최종 (2026-03-25)
- [x] Phase U: 7c add BRE ✅ (2026-03-25)
  - Det AUC 0.970, Id F1 0.813 | BRE re-added as 7th cancer
- [x] Phase W: 7-cancer held-out test ✅ (2026-03-25) ⭐ **BEST 7-cancer**
  - SERS only: Det AUC 0.966±0.005, Type ID F1 0.818±0.026
  - Fusion: Det AUC 0.979±0.005, Type ID F1 0.923±0.022
  - 8,140 spectra, ~1,628 subjects

### Phase X: Calibration 실험 (2026-03-27)
- [x] Phase X: Wavenumber calibration ✅ (2026-03-27)
  - 결과: 성능 하락 → 미적용 결정

### Phase AB: CRC↔PAN Confusion xAI Analysis (2026-04-02)
- [x] Phase AB: CRC↔PAN 혼동 원인 xAI 분석 ✅ (2026-04-02)
  - **Analysis 1 — LR Coefficient Overlay**: CRC-PAN cosine similarity -0.317, 1900~1970 cm⁻¹ 구간 강한 동일방향 overlap
  - **Analysis 2 — SHAP Spectrum Overlay**: CRC-PAN SHAP cosine -0.010 (거의 직교), PAN 시그널 비특이적
  - **Analysis 3 — Misclassified Spectra**: LR CRC→PAN 24건, PAN→CRC 21건; DL CRC→PAN 42건, PAN→CRC 17건
  - **Supplementary — t-SNE Embedding**: CRC/PAN latent space에서 크게 겹침, 오분류 샘플이 경계에 분포
  - 결론: 두 암종의 소변 SERS 스펙트럼이 본질적으로 유사 + PAN 샘플 수(100) 부족으로 CRC(300)쪽 bias
  - → results/training/phase_ab_xai/

> ⚠️ **아래 Phase AC~AT는 2026-08-31 소급 기록 (append-only)**: 실행일은 각 항목 표기, 기록일은 2026-08-31.
> 조사 agent가 각 디렉토리의 REPORT.md/README.md/JSON/CSV 원본에서 직접 읽은 수치만 사용, 추정·계산 없음.

### Phase AC: Mapping cohort 전처리/depth ablation (2026-08-25~26)

> Cohort: mapping study, 113명 (Prostate cancer 43 / Prostate disease control 49 / Control 21), 13,673 finite repeats
> (data/mapping/, 121 finite spectra/patient, no non-finite repeats removed) — **메인 7-cancer 소변 패널과 별개 서브스터디**.
> Grid 402–2198 cm⁻¹(933pt). Outer validation 5-fold StratifiedGroupKFold(patient-grouped), inner 4-fold(LR). ResNet: 3 residual
> block, multi-scale kernel 3/5/7, channel 32→64→128, 30 epoch max, seed `20260826`. Repeat study: candidate counts
> 1,3,5,9,16,25,36,49,64,81,100,121 × 100 Monte-Carlo. Cancer Screening AUC는 병원/측정 조건 confounding 가능성 있어
> 외부 일반화 성능으로 해석하지 않음(원본 명시).
> **MLOps 메타데이터** (mapping_aligned_baseline_area_dwt run 기준): source commit `46b0f0e17f1ca192c77b34daffa37ed3eac34519`,
> MLflow experiment `aecd-mapping-preprocessing-ablation`(id 2) run `c3d3eb76cf184acca55d8273de2cf418`.
> Data fingerprint(SHA-256): `data/mapping/clinical_df.xlsx`=`c49eb12812faa807f4b81ce747058d0e1d65fde1b7205925354e9838ccb1dbdb`,
> `artifacts/usersnet/v1.0.0/common_grid.npy`=`9b4fc501edff7e1e268fc6f167f6fab9b271a19f96e59fcc1bf4ad08c7b7b8e8`.
> 재현: `python scripts/analysis/run_mapping_aligned_baseline_area_dwt.py --condition <cond> --epochs 30 --mc-iterations 100 --n-blocks 3 --device cuda`
> (전 조건 완료 후 `compare_mapping_aligned_baseline_area_dwt.py` + `plot_mapping_aligned_baseline_area_dwt.py`로 집계 재생성)

- [x] Phase AC-1: 3-block CNN 전처리 ablation (standard vs trim-only vs no-trim raw) ✅ (2026-08-25~26)
  - Screening AUC: standard 0.4419 < no-trim raw 0.6083 < **trim-only 0.6528 (최고)**; LR-reference는 모든 조건에서 CNN 상회 (0.64~0.70)
  - 결론: trim이 유용하나 SG+baseline+SNV 표준 번들이 3-block CNN 성능을 오히려 악화시킴 (원인 세부 연산 미분리)
  - → results/mapping_trim_only_20260826_v1/, results/mapping_no_trim_raw_20260826_v1/, results/mapping_multiscale_resnet_20260826_standard_repro/
- [x] Phase AC-2: baseline correction + area norm + DWT 5조건 ablation ✅ (2026-08-26)
  - 5조건: `raw_aligned`(baseline/norm/DWT 없음) / `baseline`(rolling-min window101) / `baseline_area`(+area norm `y/sum(abs(y))`) /
    `baseline_dwt`(+db4 symmetric level7 soft-threshold DWT) / `baseline_area_dwt`
  - ResNet Screening AUC/BA/TypeID macroAUC/macroF1/LR-ref AUC:
    raw_aligned 0.653/0.570/0.556/0.251/0.696 · baseline 0.506/0.507/0.499/0.234/0.704 ·
    baseline_area 0.523/0.500/0.522/0.104/0.696 · baseline_dwt 0.489/0.496/0.502/0.262/0.659 ·
    baseline_area_dwt 0.512/0.500/0.543/0.104/0.654
  - train/val AUC gap: baseline ~0.202, baseline_dwt ~0.206 (overfitting/표현력 손실 패턴과 일치)
  - repeat-count 최소 후보(Screening/TypeID, 각 조건 자기 121-repeat 기준): raw_aligned 36/49, baseline 1/9,
    baseline_area 1/1, baseline_dwt 1/9, baseline_area_dwt 1/1 — chance-level 조건의 낮은 임계값은 "효율적 측정 프로토콜의 증거로 부적절"(원본 명시)
  - 결론: baseline correction이 CNN이 쓰던 raw intensity/scale feature를 제거했을 가능성, DWT는 현재 조건에서 CNN 성능 개선 못 함
  - → results/mapping_aligned_baseline_area_dwt_20260826_v1/
- [x] Phase AC-3: depth ablation (3-block vs 12-block) ✅ (2026-08-26)
  - trim-only 12-block 전환 시 0.6528→0.5352로 **하락**, train-val AUC gap 0.0018→0.1290으로 악화 (depth 증가가 generalization 악화)
  - → results/mapping_trim_only_20260826_deep12/, results/mapping_no_trim_raw_20260826_deep12/, results/mapping_multiscale_resnet_20260826_deep12_standard/, results/mapping_depth_comparison_20260826_v1/
- [x] Phase AC-4: Clinical+Spectrum LR fusion (참고군, ablation과 직접비교 금지) ✅ (2026-08-25)
  - Screening AUC 0.7389 / BA 0.6621 — spectrum-only ResNet과 다른 입력 체계이므로 별도 참고 기준
  - → results/mapping_clinical_spectrum_logistic_20260825_v1/
- [x] Phase AC-5: 반복 평균/signal audit ✅ (2026-08-25)
  - 121-repeat 산술평균 vs 기존 `_ave`: normalized RMSE median 0.0459, correlation median 0.98759, noise reduction 49.8%(기존) vs 50.3%(patent QC average)
  - → results/mapping_repeat_average_patent_20260825_v1/
  - smoke 실행 3건은 1 epoch/MC=2 pipeline 동작확인용으로 성능 표에서 제외됨 (숫자 없음, 실행 성공 여부만 확인):
    results/mapping_trim_only_20260826_smoke/, results/mapping_no_trim_raw_20260826_smoke/,
    results/mapping_clinical_spectrum_logistic_20260825_smoke/
- [x] Phase AC-6: Covariance eigenspectrum 신호 감사 ✅ (2026-08-26)
  - within-repeat effective rank 2.05(95%분산 4성분) vs between-subject corrected effective rank 1.40(2성분); subject-mean noise/between-subject SD비 6.05%
  - → results/mapping_covariance_eigenspectrum_20260826_v1/
- [x] Phase AC-7: 전처리 종합비교 + 역사적 참고값 ✅ (2026-08-26)
  - AC-1과 동일 3조건 비교를 LR/ResNet/blend 3-way로 재확인(trim-only 최고), 참고용 과거 STK-v2 locked external reference ROC-AUC 0.5993(935-point grid, 직접비교 불가 라벨)
  - → results/mapping_preprocessing_summary_20260826_v1/

### Phase AD: MLflow 레거시 모델 계보 재구성 (인프라, 2026-08-27)

> 새 실험이 아니라 기존 legacy 체크포인트를 MLflow에 등록/추적 가능하게 만든 인프라 작업.
> `pre_change_unverified`로 명시 — reagent identity를 보증하지 않으며, 등록된 모델은
> non-serving artifact (배포용 아님). Backfill: 95 bundle / model artifact 파일 667개(.pt 329 + .joblib 338) /
> registered family 13 / Model Registry version 95 / metadata 확인 93 / checkpoint-only 2.
> 재현: `uv run scripts/db/aecd_model_artifact_backfill.py`

- [x] Phase AD-1: Legacy 모델 아티팩트 backfill ✅ (2026-08-27)
  - 95개 legacy 체크포인트 번들을 13개 registered model family로 MLflow Model Registry에 등록, 93개는 메타데이터 확보
  - Reconstruction Git SHA: `efb086e071361682f1d76deb06da13e384aeaada`
  - → results/mlflow_historical_model_backfill_20260827_v1/ (REPORT.md, backfill_inventory.csv)
- [x] Phase AD-2: 모델 계보 타임라인 생성 ✅ (2026-08-27)
  - 95개 번들 → 13개 family, source timestamp 93건 확인/2건 누락, hyperparameter transition 68행. Source 시간범위 2026-02-26~2026-04-07
  - → results/mlflow_model_lineage_timeline_20260827_v1/ (MODEL_LINEAGE_TIMELINE.md, model_lineage_timeline.csv, parameter_change_matrix.csv)

### Phase AE: 데이터 소스 스크리닝 (2026-08-27)

> 목적: 환원제 변경 전·후 데이터를 동일 source inventory 규칙으로 확인, 학습·평가 재사용 후보군과 calibration/control 자료 분리.
> 환자 식별자/원본 파일명/URI/스펙트럼 배열은 기록하지 않음. `mapping` 반복측정=`post_change_verified`(DB reagent lot+측정일 확인),
> 과거 Thermo/Medical/remeasurement=`pre_change_unverified`(변경 전 환원제 lot 미확인). `_ave` 파생평균/reference/calibration/
> background/equipment test/metabolite reference는 inventory엔 기록하되 임상 screening 후보군에서 제외.

- [x] Phase AE-1: AECD 전체 소스 스캔·중복검증 ✅ (2026-08-27)
  - 전체 확인 77,722 파일 → 유한값 valid 69,735 → reject 12 → 파생평균 제외 7,975 → screening 후보 45,199 (총 spectrum point 127,712,605)
  - source family별 valid: mapping clinical(post_change_verified) 13,673 / historical Thermo(pre_change_unverified) 11,785 /
    Medical Raman(pre_change_unverified) 11,599 / remeasurement(pre_change_unverified) 8,142
  - mapping archive 13,937건 vs filesystem 13,937건 완전 일치(size mismatch 0) → duplicate alias로 기록, 신규 dataset으로 세지 않음
  - manifest SHA-256: `c47345f5d37a70c94ba73979edde2ab1ed518d0807e598497818c75521d45292`
  - MLflow: experiment `aecd-all-source-screening`(id 3) run `f19485cb393d4ca4928305adb2f57df7`;
    experiment history registry `aecd-experiment-registry`(id 4)에 후보/완료 107/107건 별도 등록(post_change_verified 13건, pre_change_unverified 94건,
    증거수준: 주분석 11/참고 2/historical summary 8/historical log metadata-only 86 — 86건은 direct_comparison=not_allowed 태그 고정)
  - **코드 정리 screening (조치 아직 안 됨, 승인 대기)**: `scripts/legacy/analysis/backfill_experiment_registry.py`는 참조하는
    `models/manifest.json`이 없고 artifact 경로도 확인 안 됨 → **삭제 후보로 식별만 됨, 삭제 승인 안 받음**. 나머지 legacy 스크립트
    (`notebooks/build_stkv2_dataset.py`, `explore_data.py`, `models/legacy/scripts/train.py`, `train_resnet.py`, `test.py`, `model.py`)는
    실사용 확인되어 유지 결정.
  - 재현: `uv run python scripts/db/aecd_source_inventory.py [--log-mlflow]`, `uv run python scripts/db/aecd_experiment_registry.py`
  - → results/data_source_screening_20260827_v1/

### Phase AF: 109 vs 113 코호트 피크 비교 (2026-08-26)
- [x] Phase AF-1: 후보 피크 6종 p-value 비교 ✅ (2026-08-26)
  - 785cm⁻¹(Ring breathing) p_109=0.0018 vs p_113=0.211; 1368cm⁻¹ p_109=0.0033 vs p_113=0.051 — 코호트 크기에 따라 유의성 반전되는 피크 존재
  - → results/109_vs_113_individual_peaks/

### Phase AG: Yonsei 전향적 코호트 외부검증 ⚠️ (2026-07-29 실행, 근거일 불명확)
> **핵심 발견**: STK-V2를 신규 기관(단일기관) 소규모 코호트에 적용 시 거의 chance 수준으로 붕괴.
- [x] Phase AG-1: STK-V2 meta ensemble Yonsei prospective 외부검증 ✅
  - YPAN 30 + YNOR 30 (총 60명/300 spectra), meta ensemble AUROC 0.4933 (95%CI 0.3437–0.6411) — base model 개별 AUROC(0.557~0.619)보다도 낮음, meta ensemble이 개선 효과 없음
  - → results/yonsei_prospective_current_model/

### Phase AH: 보라매병원 외부검증 ⚠️ (실데이터 2026-06-02, 처리 2026-07-29)
> **핵심 발견**: Phase AG와 같은 패턴 — 새 기관 코호트에서 AUC≈0.5, batch/domain 이탈로 진단(버그 아님, 대조실험 확인).
- [x] Phase AH-1: STK-V2 (usersnet v1.0.0) 보라매 BNOR/BPRO 외부검증 ✅
  - n=48(BPRO 41/BNOR 7), ROC-AUC 0.4652, Balanced accuracy 0.4251, Specificity 0.14(underpowered)
  - 모델공간 코사인 유사도: PRO-vs-PRO(같은 코호트) 0.999 → PRO-vs-BPRO(신규 코호트) 0.778 — domain shift로 성능 붕괴 설명
  - → results/boramae_20260602_validation/

### Phase AI: Clean cohort 정의 및 학습 (2026-06-05)
- [x] Phase AI-1: Clean retrospective cohort 정의 (pre/effective, 타암이력 제외) ✅ (2026-06-05)
  - 1,570명 → 1,395명(1,023 암/372 대조) clean cohort 확정, 6,975 spectra, fixed split train837/val279/test279
  - → results/clean_cohort_20260605/
- [x] Phase AI-2: Clean cohort 학습 (train_usersnet.py, fixed split) ✅ (2026-06-05)
  - Test cancer-vs-noncancer AUC 0.9851, cancer-type F1 0.9175, best meta learner=lr; cancer-type별 f1 0.769(BRE)~0.975(BLC)
  - → results/clean_cohort_package_20260605/

### Phase AJ: KAO 업데이트 코호트 (2026-06-10)
- [x] Phase AJ-1: KAO 코호트 카운트/임상요약 갱신 ✅ (2026-06-10)
  - 1,630 subjects(암 1,200/대조 430), BLC_241·YNOR_21 fallback spectrum 추가. 모델 성능 지표는 이 폴더에 없음 — Phase AG(yonsei_prospective)가 이 산출물을 입력으로 사용
  - → results/kao_20260610_updated_cohort/

### Phase AK: Cross-instrument calibration transfer (실행 2026-04-07~09, 기록 2026-08-31)
- [x] Phase AK-1: Thermo↔Medical PDS calibration transfer ✅
  - Self-reference(동일 기기) s1_auc 1.0 vs raw cross-instrument s1_auc 0.41~0.45(붕괴) → PDS 적용 시 단일시드 0.74, 멀티시드 평균 0.82~0.91로 회복(시드 변동성 큼, 두 수치 함께 병기 필요)
  - → results/cross_instrument/

### Phase AL: DACR 전처리 파이프라인 (2026-05-18)
- [x] Phase AL-1: DACR 전처리 + QC 2단계 필터링 ✅ (2026-05-18)
  - 9,485 raw → QC drop(low_corr 550건, min_reps 95명 subject) → 최종 7,202 replicate/1,493 subjects
  - → results/preprocessing_dacr/
- [x] Phase AL-2: DACR 전처리 (QC 필터링 없는 all 버전) ✅ (2026-05-18)
  - 동일 파이프라인, QC drop 없이 7,990 replicate/1,598 subjects 보존
  - → results/preprocessing_dacr_all/

### Phase AM: Medical vs Raw 장비 강도 비교 (2026-06-09)
- [x] Phase AM-1: Medical-Primary 강도비 비교 ✅ (2026-06-09)
  - 1,610 sample-date 쌍, medical/primary abs-y p95 비율 median 1.117(mean 1.322, 범위 0.067~10.317)
  - → results/medical_vs_raw_data_medical/

### Phase AN: Model revision — calibration/peak registry/설명모델 (2026-07-29, 원 학습 run 2026-06-10)
> 대한암학회 제출용 아님, 실험적 검토 자료로 명시됨. `docs/ml/sers_model_v3_peak_calibration_standard.md`(2026-06-10 KAO rerun 이후
> 요청된 "SERS Model V3" 스펙 — calibrated evidence model + 데이터기반 peak-window 표준 정의)의 구현 결과.
- [x] Phase AN-1: STK-V2 calibration + peak registry + 설명모델 ✅
  - Calibrated STK-V2 test AUC 0.9834/ECE 0.0285; peak registry 153개 발견 중 accepted 127/disease-discriminating 71
  - Peak-only 설명모델(127 peaks) binary AUC 0.9367이나 cancer-type macro F1 0.5985로 성능모델 단독대체 부적합 → hybrid 구조 제안(결론)
  - → results/model_revision/

### Phase AO: 특허 SEED 예시 패키지 ⚠️ 합성 데이터, 실제 환자 아님 (2026-06-04)
- [x] Phase AO-1: QC/전처리 흐름 시연용 합성 예시 8 case ✅
  - README에 "합성 예시 데이터, 실제 환자 데이터 아님, 환자 식별자 미포함" 명시. 성능 수치 없음(구조 시연용)
  - → results/patent_seed_examples/

### Phase AP: Raw data ↔ 임상 데이터 매칭 검증 (2026-06-08)
- [x] Phase AP-1: 1차 raw ↔ 2차 clinical reference replicate 매칭/보정 비교 ✅ (2026-06-08)
  - 8,050 matched replicate keys/1,611 subjects; axis_intensity_corrected 보정 전후 상관도 median 변화는 대체로 미미(그룹별 차이 있음)
  - → results/raw_data_vs_clinical_reference/

### Phase AQ: Raw set 평균 감사 + smoke 토이 테스트 (2026-07-30)
- [x] Phase AQ-1: `_ave` 평균파일 대 실측평균 감사 ✅ (2026-07-30)
  - 2,069개 average 파일 중 2,064개 정책 기준(rmse<0.0001) 통과, 5개 rejected
  - → results/raw_set/average_audit/
- [x] Phase AQ-2: PRO-vs-NOR 2-fold/1-epoch smoke 테스트 ⚠️ 토이 실행, 본 실험 아님 ✅ (2026-07-30)
  - raw_mean_logistic 베이스라인 AUC 0.995 vs raw_set 모델 AUC 0.68~0.79 (베이스라인이 훨씬 높음 — smoke 규모라 해석 주의)
  - → results/raw_set/smoke_pro_nor/

### Phase AR: 전립선 3-코호트 측정 반복성/변동성 연구 (2026-08-28)
- [x] Phase AR-1: 3-코호트(후향91/액상41/분말43) 스펙트럼 95% CI 재산출 ✅
  - Bootstrap(10,000회) 재계산. 분말-후향 raw intensity 차이 +1685.15 (95%CI 1586~1789, p=2e-31); "진단성능 비교 아닌 코호트 분포 비교"로 명시 제한
  - → results/three_cohort_spectra_ci_20260828_v1/
- [x] Phase AR-2: 검체 내 반복측정 변동성(CV) 비교 ✅
  - 전체보정면적 CV: 후향 9.12% / 액상 7.10% / 분말 19.53% — 분말군이 3개 지표 모두 변동성 최고
  - → results/within_subject_variability_20260828_v1/
- [x] Phase AR-3: 반복측정 개수별 pooled OOF AUC 민감도 분석 ✅ (재학습 아닌 사후 분석)
  - n=1→100 반복 사용 시 AUC: direct 0.61→0.67, raw 0.57→0.71, legacy 0.60→0.74; "전처리 인과효과로 해석 금지"로 명시 제한
  - → results/three_method_repeat_auc_20260828_v1/

### Phase AS: AECD API 모델 계열 — prostate 3-class/screening 전처리 비교 (2026-08-25~27, notebooks/)
> 모든 하위 실험 공통: prostate 3-class(control/prostate disease control/prostate) 또는 cancer-vs-non-cancer screening, subject-level nested GroupKFold(5×5), 113 subjects. **메인 7-cancer 소변 패널과 별개 서브스터디** (Phase AC~AR와 동일 mapping/AECD 코호트 계열).
- [x] Phase AS-1: Baseline 노트북 (screening/3-class/legacy STK-V2 비교 + repeat-count sweep) ✅ (2026-08-25, 2026-09-03 코호트 갱신)
  - Screening AUC 0.586, 3-class macro AUC 0.575; legacy STK-V2 screening AUC 0.7405(BAcc 0.6621)로 baseline 대비 높음
  - **2026-09-03 재실행 (Drop 제외, 113→112명)**: screening AUC 0.629 [CI 0.527–0.728], 3-class macro AUC 0.570, legacy STK-V2 screening AUC 0.7506(BAcc 0.7096) / 3-class macro OvR AUC 0.6343. legacy가 baseline보다 높다는 결론 불변
  - 노트북의 하드코딩된 코호트 크기를 `EXPECTED_SUBJECTS`/`EXPECTED_CLASS_COUNTS` 상수로 분리 — 다음 코호트 변경 시 한 곳만 고치면 됨
  - → notebooks/aecd_api_model_baseline_outputs/
- [x] Phase AS-2: 3-class 5조건 비교 (direct/DWT비교/특허DWT/all-QC spectra/legacy) ✅ (2026-08-25, 2026-09-03 코호트 갱신)
  - macro OvR AUC: DWT비교(raw subject mean) 0.6595(최고) > legacy STK-V2 0.6053 > all-QC 0.5547 > direct 0.5459 > 특허DWT 0.4523(최저)
  - **2026-09-03 재실행 (Drop 제외, 113→112명, QC spectra 12,826)**: legacy STK-V2 0.6343(최고) > DWT비교 0.5943 > direct 0.5593 > all-QC 0.5423 > 특허DWT 0.4861(최저)
  - ⚠️ **순위가 바뀌었다.** 최고 조건이 DWT비교 → legacy STK-V2로, direct와 all-QC의 순서도 뒤집혔다. DWT비교는 0.6595 → 0.5943 (−0.065). 정상군 subject 1명 차이로 macro AUC가 이 폭으로 움직인다 — n=20 코호트에서 조건 간 순위를 확정적으로 읽으면 안 된다 (AS-10의 peak 불안정성과 같은 현상)
  - legacy STK-V2 3-class는 baseline 노트북(AS-1)에서 재생성됨. `notebooks/aecd_api_model_3class_outputs/legacy_stkv2_3class_*.csv`(2026-08-25)는 갱신되지 않은 별도 산출물
  - → notebooks/aecd_api_model_3class_outputs/
- [x] Phase AS-3: All QC-passed spectra(12,926개) 직접 입력 screening 모델 ✅ (2026-08-25)
  - subject OOF AUC 0.5957, spectrum OOF AUC 0.5763
  - ⚠️ **2026-09-03 기준 재실행 불가 / 산출물 stale.** `notebooks/aecd_api_model_all_qc_spectra_clinical_performance.py` L337이 `pipeline.group_and_align_subjects(items, common_grid)`로 2인자 호출하는데, 해당 함수는 최초 커밋(880864a)부터 `calibrations`를 포함한 3인자다. 즉 이 스크립트는 오늘 이전부터 돌지 않았고 위 수치는 그 이전 버전의 결과다. 고치려면 "이 분석에도 PS 축 보정을 적용할 것인가"를 먼저 정해야 한다 (미결정)
  - → notebooks/aecd_api_model_all_qc_spectra_outputs/
- [x] Phase AS-4: 특허 "반복측정 평균스펙트럼생성" 검증 — direct mean-spectrum vs legacy STK-V2 ✅ (2026-08-28)
  - Direct raw mean-spectrum OOF AUC 0.6698 vs STK-V2 0.7316 (legacy가 더 높음); 3-class도 STK-V2가 소폭 우위(0.6210 vs 0.5616)
  - → notebooks/aecd_api_model_mean_spectrum_outputs/
- [x] Phase AS-5: 특허 "DWT 노이즈제거" 검증 — DWT 전처리 vs raw subject-mean ✅ (2026-08-25, 2026-09-03 코호트 갱신)
  - Raw subject-mean OOF AUC 0.7163 vs 특허 DWT 0.5233 — "이 데이터/설계에서는 DWT가 raw보다 높은 AUC를 만들지 않음"(원본 결론 그대로 인용)
  - **2026-09-03 재실행 (Drop 제외, 113→112명)**: raw 0.7145 vs 특허 DWT 0.5201. 결론 불변
  - → notebooks/aecd_api_model_patent_dwt_outputs/
- [x] Phase AS-6: Subject mean spectrum peak 검출/반복성 (SNR 2/3/5) — descriptive, 분류 성능 지표 아님 ✅ (2026-08-25)
  - SNR=3(primary) 기준 subject당 평균 31.04개 peak, 80% 반복 검출 401개
  - → notebooks/aecd_api_subject_peak_outputs/
- [x] Phase AS-7: OOF 오분류 임상변수 탐색 (exploratory, 인과추론 아님) ✅ (2026-08-27)
  - Grade group 3 FN rate 72.7%(8/11); Cancer FN PSA median 17.45 vs TP 10.52
  - → notebooks/aecd_clinical_feature_extraction_outputs/
- [x] Phase AS-8: OOF 오분류 진단 리포트 ✅ (2026-08-26)
  - FP 21건 중 prostate disease control 15/control 6; ⚠️ TN/FP 수치가 Phase AS-4(dir4) 재계산본과 다름(스냅샷 시점 차이 추정) — 각 출처 그대로 인용
  - → notebooks/aecd_clinical_misclassification_outputs/
- [x] Phase AS-9: AS-1~AS-4 통합 비교표 (RUN_EXPERIMENTS=False, 재취합만) ✅ (2026-08-27)
  - AS-4(legacy STK-V2 raw subject-mean)가 binary/3-class 모두 최고 macro AUC — 5개 조건 통틀어 최고 성능 방식으로 확인
  - ⚠️ **2026-09-03 재취합 보류.** 입력 4개(`aecd_direct_mean`/`aecd_dwt_and_raw`/`aecd_three_class`/`aecd_all_qc`) 중 앞의 3개는 112명으로 갱신됐으나 `aecd_all_qc`는 AS-3 파손으로 113명 산출물 그대로다. `model_pipeline_comparison.csv`에 코호트 크기 컬럼이 없어 지금 취합하면 112명·113명 결과가 구분 표시 없이 섞인다. AS-3 해결 후 재취합할 것
  - → notebooks/aecd_model_pipeline_results_outputs/
- [x] Phase AS-10: 검출된 공통 peak별 암/비암/정상 fold change ✅ (2026-09-02)
  - 코호트: 보라매(BORAMAE) 112명 = 정상 20 / 비암 49 / 암 43. clinical v7에서 오너가 Drop으로 지정한 BNOR_110(121 spectra)을 로더에서 제외 — `EXCLUDED_COHORT_GROUPS`, 근거 scripts/db/aecd_clinical_v7/README.md
  - 방법: subject별 평균 스펙트럼에 61-point rolling-median baseline 제거(clip 없음) → subject별 400–2200 cm⁻¹ 총 면적 정규화(총 면적 CV 22.5%) → 대표 파수 ±1 grid point 평균 → 그룹 median 비. Mann-Whitney U + BH-FDR(주 지표: 비교당 11검정 / 보수적: 전체 33검정), median ratio percentile bootstrap 10,000회
  - **결과: 피크 11개 × 비교 3종 33건 중 BH-FDR q<0.05 통과 0건** (두 검정군 정의 모두 동일)
  - ⚠️ **효과추정치도 불안정**: 정상군 1명 제거로 1001.9 cm⁻¹ 비암 vs 정상 CI가 [0.969–1.804](1 포함) → [1.023–1.835](1 제외)로, R05 비암 vs 정상도 1 포함 → 1 제외로 바뀌었다. 아래 "CI가 1을 제외"는 더 강한 증거가 아니라 n=20에서 subject 1명에 뒤집히는 값이다
  - 명목상 CI가 1을 제외한 항목 4건 (모두 FDR 탈락, 위 불안정성 전제하에 읽을 것):
    - 1001.9 cm⁻¹ 비암 vs 정상 FC 1.298 [1.023–1.835], p=0.013 → q=0.139
    - 1364.6/1366.6 cm⁻¹(분해능상 동일 피처 R05) 암 vs 비암 FC 1.66 [1.01–2.31], p=0.048/0.051 → q=0.271
    - 같은 R05 비암 vs 정상 FC 0.58–0.61 [0.41–0.98], p=0.084/0.123 → q=0.362 (비암군이 정상·암 양쪽보다 낮음)
  - Kruskal-Wallis 3군 최소 p: 1001.9 cm⁻¹ p=0.037. 이 피크는 암·비암이 모두 정상 대비 ~1.29배 높고(비암 vs 정상 p=0.013, 암 vs 정상 p=0.034) 암/비암 간 차이 없음(FC 0.996, p=0.633) — 단일기관·배치 대조 없음이라 암 신호와 검체 수집/취급 차이를 구분할 수 없음
  - ⚠️ **peak 검출 불안정성**: 정상군에서 subject 1명만 빠져도(21→20) resolution-aware peak registry가 9개 → 11개로 바뀌었다. 신규 2개(893.9, 1449.5)는 정상군에서만 검출되고, 1457.2는 control|prostate → prostate로 바뀜. 즉 검출 비대칭을 읽으라고 넣은 `detected_in_labels` 컬럼 자체가 4개 피크에서 뒤집혔다 — 검출 비대칭을 생물학적 차이로 해석하면 안 되는 근거. 그룹 median 기반 peak 검출이 n=20 규모에서 개별 subject에 민감하다
  - 참고 지표: 같은 입력 screening OOF AUC 0.6721 (Drop 포함 113명일 때 0.6698), balanced accuracy 0.5964, QC pass 12826/13552 (94.64%)
  - 정규화 기준(총 면적)과 1001.9 cm⁻¹를 PS 잔류 아님으로 취급한 것은 2026-09-02 사용자 확인 사항
  - 해석 한계: fold change는 subject 수준 기술통계이며 판별력 아님. 정상군 n=20이 통계적 제약
  - 구현: scripts/analysis/aecd_peak_group_fold_change.py → notebooks/aecd_api_model_mean_spectrum_outputs/peak_fold_change/

---

## TODO Experiments

> 아래 실험은 사용자 승인 후에만 실행합니다.

<!-- 새 실험을 여기에 추가하세요:
- [ ] Phase Y: 실험 설명
  - Hypothesis: ...
  - Variable: ...
  - Cancer set: ...
  - Baseline: Phase X
-->

---

## Suggested Next Experiments

> experiment-runner 또는 사용자가 제안한 실험 후보. 승인 시 TODO로 이동.

<!-- 제안 실험:
1. [ ] BRE 샘플 추가 수집 후 재학습 — 근거: BRE n=30으로 통계적 파워 부족
2. [ ] 보라매병원 전향적 데이터 검증 — 근거: retrospective bias 해소
-->

## Phase PL — Preprocessing Lab (논문 기반 전처리 벤치마크)

- [x] PL-1: Whitaker-Hayes despike on/off ablation ✅ (2026-09-02)
  - **결론: 개선 근거 없음.** 환자 단위 AUC (5시드, LR, StratifiedGroupKFold-by-subject)
    - despike off: 0.7941 ± 0.0112
    - despike on : 0.7582 ± 0.0323  → Δ = **-0.0359 ± 0.0428**, 3/5 시드 악화
  - despike on 쪽 분산이 3배 크다 (안정성도 나빠짐)
  - ⚠️ 표본 한계: 환자 112명. 실행 중 다른 세션이 control 1명을 'Drop'으로 재라벨했는데,
    그 **환자 1명 차이로 Δ 부호가 +0.0086 → -0.0359로 뒤집혔다.** 이 코호트 크기로는
    0.03 수준 효과를 판별할 수 없다.
  - 구현 감사 완료: 저자 참조구현(Mendeley sxjgbgg95y) 대조, 불일치 2건 정정
    (threshold 7.0→6.0, force_endpoints 추가) → `docs/reviews/2026-09-02-paper-code-audit-whitaker-hayes.md`
  - 후속 가설: Stage-1 QC가 이미 cosmic ray 스펙트럼을 폐기하므로 despike가 복구할
    대상이 거의 없다 (QC 통과 수 11331 vs 11334, 3건 차이). QC 게이트를 끄고 비교하는
    설계가 효과 검출에 더 적합할 수 있다.
  - → aecd_platform `experiment.runs` #58(off), #59(on) / 데이터: 13,552 measurements FK 연결

- [ ] PL-2: SG smoothing 창 sweep (표준물질 PS/Si 결합 평가, 2026-09-03) — **기록 누락**
  - `config/config.yaml` smooth_window 11→5 변경 근거로 인용됨(`scripts/analysis/preprocessing_lab/run_smoothing_sweep.py --combined`)
    이나 이 문서·registry에 결과가 없다. 해당 세션에서 append 필요. 번호 PL-2는 이 실험에 예약.

- [x] PL-3: 설계가이드 요인별 비교 — calibration / despike / baseline(λ 그리드) / normalization ✅ (2026-09-07~08)
  - **목적**: `docs/ml/preprocessing_design_guide.md` §2 순서(cal→despike→trim→SG→baseline→norm)의 각 단계를
    실제 run으로 실행해 CI 표(`experiment.run_metrics`)에 넣는다. 한 번에 한 요인만 바꾼다.
  - **데이터**: aecd_platform 전립선 3군 112명 / 13,552 spectra (= PL-1과 동일 로드: prostate 5203 / PDC 5929 / control 2420).
    `data/mapping` 파일은 같은 보라매 113검체×121점의 사본이며 xlsx 라벨이 DB에서 철회된 'Drop' 1명을 포함하므로 쓰지 않음.
  - **모델·평가**: PL-1 규약 그대로 — 개별 spectrum LR(C=1.0, balanced, StandardScaler), StratifiedGroupKFold(5) by subject,
    환자 mean OOF, 5시드(42/7/123/2024/31337). CI는 seed 42 환자 점수의 환자 단위 BCa bootstrap 2000회.
    Δ는 같은 환자를 함께 재추출하는 짝지은 bootstrap (production 기준, cal_despike 기준 둘 다).
  - ⚠️ **audit gate 면제**: SKILL은 `audit_status='pending'` 방법의 실행을 금하지만 사용자 결정(2026-09-08)으로 탐색 실험으로 전부 실행.
    감사 완료는 whitaker_hayes·airpls 2건뿐. **채택 근거가 아닌 탐색 기록.** audit_status는 변경하지 않음.
  - **조건** (모두 trim 400–2200, SG 창 5 차수 3 = 현재 config): 기준점 = cal_despike (cal✓ WH despike z=6, rolling_min 101, SNV)

    | 조건 | 환자 | spectra(QC 후) | AUC seed42 [95% CI] | 5시드 mean±sd | Δ vs production [CI] |
    |---|---|---|---|---|---|
    | no_cal (PL-1 재현, 단 SG 5) | 112 | 10,823 | 0.735 [0.637, 0.821] | 0.730±0.015 | **−0.082 [−0.158, −0.023]** |
    | production (cal✓ despike✗) | 112 | 10,413 | 0.817 [0.728, 0.885] | 0.796±0.028 | 0 |
    | cal_despike (기준점) | 112 | 10,414 | 0.792 [0.696, 0.868] | 0.798±0.021 | −0.025 [−0.079, 0.026] |
    | bl_airpls λ=1e3 | **91** | 4,771 | 0.696 [0.580, 0.790] | 0.674±0.068 | −0.098 [−0.200, 0.003] |
    | bl_airpls λ=1e4 | 111 | 7,565 | 0.677 [0.568, 0.770] | 0.720±0.062 | **−0.139 [−0.230, −0.060]** |
    | bl_airpls λ=1e5 (config 기본) | 112 | 9,967 | 0.796 [0.706, 0.871] | 0.769±0.048 | −0.022 [−0.083, 0.038] |
    | bl_airpls λ=1e6 | 112 | 10,454 | 0.804 [0.711, 0.876] | 0.815±0.011 | −0.014 [−0.086, 0.056] |
    | bl_airpls λ=1e7 | 112 | 11,930 | 0.775 [0.676, 0.854] | 0.792±0.030 | −0.042 [−0.119, 0.020] |
    | bl_arpls λ=1e3 | **84** | 3,595 | 0.628 [0.497, 0.748] | 0.634±0.067 | **−0.154 [−0.256, −0.048]** |
    | bl_arpls λ=1e4 | 110 | 6,786 | 0.712 [0.612, 0.798] | 0.682±0.022 | **−0.106 [−0.201, −0.021]** |
    | bl_arpls λ=1e5 (config 기본) | 111 | 6,825 | 0.739 [0.635, 0.822] | 0.721±0.030 | **−0.076 [−0.157, −0.004]** |
    | bl_arpls λ=1e6 | 112 | 8,680 | 0.758 [0.660, 0.841] | 0.780±0.026 | −0.059 [−0.137, 0.006] |
    | bl_arpls λ=1e7 | 112 | 9,972 | 0.785 [0.693, 0.863] | 0.806±0.025 | −0.032 [−0.100, 0.028] |
    | bl_als λ=1e3 | 102 | 5,569 | 0.708 [0.594, 0.806] | 0.667±0.024 | **−0.095 [−0.188, −0.019]** |
    | bl_als λ=1e4 | 112 | 7,930 | 0.675 [0.569, 0.771] | 0.687±0.027 | **−0.142 [−0.227, −0.067]** |
    | bl_als λ=1e5 | 112 | 9,645 | 0.792 [0.699, 0.868] | 0.787±0.028 | −0.026 [−0.095, 0.038] |
    | bl_als λ=1e6 (config 기본) | 112 | 10,712 | 0.839 [0.755, 0.900] | 0.820±0.019 | +0.022 [−0.038, 0.093] |
    | bl_als λ=1e7 | 112 | 11,748 | 0.759 [0.662, 0.836] | 0.789±0.035 | −0.059 [−0.140, 0.010] |
    | norm_l2 (rolling_min) | 112 | 10,409 | 0.818 [0.730, 0.887] | 0.795±0.019 | +0.001 [−0.054, 0.052] |
    | norm_minmax (rolling_min) | 112 | 10,413 | 0.845 [0.760, 0.905] | 0.802±0.030 | +0.028 [−0.024, 0.084] |

    (굵게 = Δ의 95% CI가 0을 제외. Δ vs cal_despike는 `summary.csv`의 `d_cal_despike*` 컬럼.)
  - **관찰 (해석·채택은 사용자 몫)**:
    1. production보다 좋은 조건은 없다 — 양의 Δ(als 1e6, minmax, l2)는 모두 CI가 0을 포함.
    2. calibration 제거(no_cal)만 CI가 0을 벗어나며 나쁘다. shift 분포: sd 2.2 cm⁻¹, |max| 6.9, fail-safe(shift=0) 629/13,552(4.6%) → calibration은 명목이 아니라 실제로 작동.
    3. **no_cal은 PL-1 despike_off와 데이터·코드가 같고 SG 창만 다르다(11→5, config.yaml 2026-09-03 변경, 미커밋).**
       PL-1 off 0.794±0.011 / QC 통과 11,331 → PL-3 no_cal 0.730±0.015 / QC 통과 10,823. SG 창 축소가 임상 AUC를 −0.06 낮춘
       것으로 보이나 이 비교는 사후적(같은 실행 안의 조건이 아님)이라 **PL-2 smoothing sweep을 임상 AUC로 재확인해야 한다.**
    4. λ가 작을수록 Stage-2 corr QC 탈락이 급증(arpls 1e3: 환자 84명, spectra 3,595만 남음). 낮은 λ의 나쁜 AUC는 baseline 효과와
       QC 탈락(환자·spectra 집합 변화)이 섞여 있다. 짝지은 Δ는 공통 환자만 쓰지만(`d_*_n`), 학습 데이터 자체가 달라진 점은 보정되지 않는다.
    5. airPLS·ALS는 λ=1e6이 정점이고 1e7에서 떨어지며, arPLS는 1e7(그리드 상한)까지 계속 오른다. 각 방법의 최적 λ(airpls 1e6, arpls 1e7, als 1e6)에서는 rolling_min과 구분되지 않는다.
  - **한계**: 환자 112명이라 0.03 수준 차이 판별 불가(PL-1과 동일). LR C 고정. smoothing/baseline 선후 순서는 한 가지만. 가이드 ③의 600–1800 절단은 미실행(400–2200 유지).
  - 스크립트(보존): `scripts/analysis/preprocessing_lab/run_guide_pipeline.py` (`--cohort aecd`, `--load-db`)
  - 산출물: `results/preprocessing_lab/guide_pipeline_20260907/summary.csv`, `aecd/{condition}/{patient_oof_predictions.csv, spectrum_oof_seed42.csv,
    metrics_ci_standard.csv, run_metadata.json, calibration_shifts.csv, measurement_ids.npy}`
  - DB: `experiment.runs` #126~#164 (run_name `guide_aecd_{condition}_20260907`, method_id·config_snapshot·measurement FK 연결), `run_metrics` CI 행(metric `auc` 등 8종) + 시드 행(PL-1 이름)
  - 코드: 평가 스키마 `2be7be3`, 스크립트 `8459643`

- [x] PL-3 추가 (2026-09-08 오후): ③ truncation · ④ smoothing · ④↔⑤ 순서 조건 8개 + PL-1 정확 재현 1개 + **3-class task 전 조건** ✅
  - 사용자 지적 두 건 반영: (a) 가이드 6단계 중 ③④를 고정해 둔 것 → 요인별로 추가, (b) 암 vs 비암만 있었음 → control / 질환대조 / 전립선암
    3-class(multinomial LR, 환자 mean, macro OvR AUC)를 같은 전처리 결과에서 함께 산출. 총 29조건 × 2 task.
  - 코드: `preprocess_single_spectrum(baseline_before_smooth=...)` 옵션 추가(기본 off, 프로덕션 결과 불변, 회귀 테스트 추가) — `d870ee5`, PL-1 재현 조건 `no_cal_sg11` 추가.
  - **전체 표** (환자 112명 기준. 이진 = 전립선암 vs 비암 AUC, 3-class = macro OvR AUC. seed 42 값 [95% CI], 5시드 mean±sd, Δ vs production [짝지은 CI], 굵게 = CI가 0 제외)

    | 조건 | 환자 | spectra | 이진 AUC [CI] | 이진 5시드 | 이진 Δ | 3-class macroAUC [CI] | 3-class 5시드 | 3-class Δ |
    |---|---|---|---|---|---|---|---|---|
    | no_cal | 112 | 10,823 | 0.735 [0.637, 0.821] | 0.730±0.015 | **-0.082 [-0.158, -0.023]** | 0.649 [0.573, 0.720] | 0.664±0.015 | -0.066 [-0.134, +0.001] |
    | no_cal_sg11 | 112 | 11,331 | 0.782 [0.679, 0.859] | 0.794±0.011 | -0.036 [-0.116, +0.035] | 0.696 [0.617, 0.767] | 0.674±0.020 | -0.018 [-0.067, +0.034] |
    | production | 112 | 10,413 | 0.817 [0.728, 0.885] | 0.796±0.028 | 0 | 0.715 [0.633, 0.780] | 0.682±0.025 | 0 |
    | cal_despike | 112 | 10,414 | 0.792 [0.696, 0.868] | 0.798±0.021 | -0.025 [-0.079, +0.026] | 0.700 [0.614, 0.772] | 0.688±0.021 | -0.015 [-0.056, +0.031] |
    | trim_600_1800 | 112 | 10,015 | 0.830 [0.744, 0.896] | 0.810±0.041 | +0.013 [-0.043, +0.072] | 0.694 [0.610, 0.766] | 0.688±0.023 | -0.021 [-0.077, +0.038] |
    | sm_none | 112 | 10,009 | 0.835 [0.749, 0.897] | 0.791±0.042 | +0.017 [-0.044, +0.082] | 0.676 [0.593, 0.746] | 0.685±0.052 | -0.039 [-0.093, +0.022] |
    | sm_sg11 | 112 | 11,061 | 0.818 [0.722, 0.885] | 0.793±0.019 | +0.000 [-0.057, +0.055] | 0.718 [0.637, 0.788] | 0.699±0.032 | +0.003 [-0.045, +0.049] |
    | sm_sg21 | 112 | 11,463 | 0.742 [0.633, 0.827] | 0.773±0.029 | **-0.075 [-0.147, -0.018]** | 0.704 [0.620, 0.774] | 0.691±0.026 | -0.011 [-0.063, +0.042] |
    | sm_median5 | 112 | 10,940 | 0.754 [0.654, 0.834] | 0.776±0.032 | -0.063 [-0.132, +0.005] | 0.642 [0.557, 0.724] | 0.685±0.048 | **-0.073 [-0.124, -0.016]** |
    | sm_gauss1 | 112 | 10,905 | 0.809 [0.711, 0.882] | 0.780±0.028 | -0.009 [-0.069, +0.049] | 0.728 [0.649, 0.794] | 0.700±0.028 | +0.013 [-0.029, +0.058] |
    | sm_wavelet | 112 | 11,343 | 0.763 [0.668, 0.842] | 0.774±0.013 | -0.054 [-0.127, +0.010] | 0.596 [0.524, 0.666] | 0.622±0.024 | **-0.118 [-0.181, -0.064]** |
    | order_baseline_first | 112 | 10,328 | 0.811 [0.725, 0.877] | 0.796±0.056 | -0.006 [-0.052, +0.045] | 0.694 [0.615, 0.766] | 0.680±0.029 | -0.021 [-0.058, +0.016] |
    | bl_airpls_1e3 | 91 | 4,771 | 0.696 [0.580, 0.790] | 0.674±0.068 | -0.098 [-0.200, +0.003] | 0.602 [0.516, 0.690] | 0.571±0.046 | **-0.092 [-0.166, -0.017]** |
    | bl_airpls_1e4 | 111 | 7,565 | 0.677 [0.568, 0.770] | 0.720±0.062 | **-0.139 [-0.230, -0.060]** | 0.625 [0.542, 0.707] | 0.616±0.022 | **-0.091 [-0.148, -0.039]** |
    | bl_airpls_1e5 | 112 | 9,967 | 0.796 [0.706, 0.871] | 0.769±0.048 | -0.022 [-0.083, +0.038] | 0.676 [0.602, 0.748] | 0.669±0.010 | -0.039 [-0.092, +0.010] |
    | bl_airpls_1e6 | 112 | 10,454 | 0.804 [0.711, 0.876] | 0.815±0.011 | -0.014 [-0.086, +0.056] | 0.693 [0.616, 0.766] | 0.692±0.026 | -0.022 [-0.068, +0.030] |
    | bl_airpls_1e7 | 112 | 11,930 | 0.775 [0.676, 0.854] | 0.792±0.030 | -0.042 [-0.119, +0.020] | 0.706 [0.619, 0.776] | 0.705±0.019 | -0.009 [-0.057, +0.041] |
    | bl_arpls_1e3 | 84 | 3,595 | 0.628 [0.497, 0.748] | 0.634±0.067 | **-0.154 [-0.256, -0.048]** | 0.534 [0.439, 0.622] | 0.586±0.045 | **-0.144 [-0.232, -0.058]** |
    | bl_arpls_1e4 | 110 | 6,786 | 0.712 [0.612, 0.798] | 0.682±0.022 | **-0.106 [-0.201, -0.021]** | 0.619 [0.534, 0.704] | 0.597±0.030 | **-0.096 [-0.161, -0.038]** |
    | bl_arpls_1e5 | 111 | 6,825 | 0.739 [0.635, 0.822] | 0.721±0.030 | **-0.076 [-0.157, -0.004]** | 0.652 [0.564, 0.737] | 0.636±0.022 | **-0.060 [-0.120, -0.003]** |
    | bl_arpls_1e6 | 112 | 8,680 | 0.758 [0.660, 0.841] | 0.780±0.026 | -0.059 [-0.137, +0.006] | 0.687 [0.606, 0.763] | 0.685±0.010 | -0.028 [-0.083, +0.029] |
    | bl_arpls_1e7 | 112 | 9,972 | 0.785 [0.693, 0.863] | 0.806±0.025 | -0.032 [-0.100, +0.028] | 0.695 [0.621, 0.771] | 0.706±0.010 | -0.019 [-0.072, +0.040] |
    | bl_als_1e3 | 102 | 5,569 | 0.708 [0.594, 0.806] | 0.667±0.024 | **-0.095 [-0.188, -0.019]** | 0.622 [0.534, 0.713] | 0.628±0.041 | **-0.077 [-0.148, -0.015]** |
    | bl_als_1e4 | 112 | 7,930 | 0.675 [0.569, 0.771] | 0.687±0.027 | **-0.142 [-0.227, -0.067]** | 0.580 [0.489, 0.663] | 0.614±0.025 | **-0.134 [-0.220, -0.069]** |
    | bl_als_1e5 | 112 | 9,645 | 0.792 [0.699, 0.868] | 0.787±0.028 | -0.026 [-0.095, +0.038] | 0.679 [0.604, 0.746] | 0.681±0.016 | -0.036 [-0.091, +0.015] |
    | bl_als_1e6 | 112 | 10,712 | 0.839 [0.755, 0.900] | 0.820±0.019 | +0.022 [-0.038, +0.093] | 0.668 [0.583, 0.744] | 0.669±0.033 | -0.047 [-0.129, +0.020] |
    | bl_als_1e7 | 112 | 11,748 | 0.759 [0.662, 0.836] | 0.789±0.035 | -0.059 [-0.140, +0.010] | 0.702 [0.623, 0.774] | 0.667±0.025 | -0.013 [-0.063, +0.037] |
    | norm_l2 | 112 | 10,409 | 0.818 [0.730, 0.887] | 0.795±0.019 | +0.001 [-0.054, +0.052] | 0.595 [0.516, 0.666] | 0.656±0.041 | **-0.120 [-0.182, -0.070]** |
    | norm_minmax | 112 | 10,413 | 0.845 [0.760, 0.905] | 0.802±0.030 | +0.028 [-0.024, +0.084] | 0.678 [0.598, 0.750] | 0.682±0.018 | -0.037 [-0.078, +0.006] |

  - **관찰 (해석·채택은 사용자 몫)**:
    1. **PL-1 정확 재현 성공**: `no_cal_sg11`(cal✗ despike✗ SG 11) = 0.794±0.011, QC 통과 11,331 — PL-1 despike_off와 동일. 코드·데이터 drift 없음.
    2. **⚠️ 앞선 관찰 2·3 정정 — calibration 효과는 SG 창과 얽혀 있다.** calibration 없이 SG 11→5는 −0.064(0.794→0.730)지만,
       calibration+despike가 있으면 SG 11↔5 차이가 없다(0.793 vs 0.798, Δ +0.000 [−0.057, +0.055]). 반대로 "calibration 제거"의 −0.082는
       SG 5에서만 나타나고, SG 11에서는 no_cal_sg11 0.794 ≈ production 0.796이다. 따라서 (a) "calibration이 유의하게 좋다"와
       (b) "SG 5가 유의하게 나쁘다"는 **분리되지 않는 하나의 관찰**이며, 2요인 상호작용으로 봐야 한다. 단독 주장을 하지 말 것.
    3. 이진 task에서 production을 이기는 조건은 여전히 없다(양의 Δ 모두 CI가 0 포함: minmax +0.028, ALS 1e6 +0.022, sm_none +0.017, trim 600–1800 +0.013).
    4. 3-class task는 production이 0.715 [0.633, 0.780]로 사실상 최상위이며, 이진에서 무해했던 L2 정규화(−0.120)·wavelet(−0.118)·median(−0.073)이
       3-class에서는 유의하게 나쁘다 — **task마다 전처리의 영향이 다르므로 이진 결과만으로 전처리를 고르면 안 된다.**
    5. ③ 절단 600–1800(feature 625개): 이진 +0.013, 3-class −0.021, 모두 CI 0 포함 — 400–2200과 구분 안 됨.
    6. ④↔⑤ 순서 교체(baseline→smooth): 이진 −0.006, 3-class −0.021, 모두 CI 0 포함 — 순서는 이 데이터에서 문제가 아님. 단 이진 5시드 sd가 0.056으로 가장 큼.
    7. SG 창 21은 유의하게 나쁨(−0.075). 낮은 λ baseline(1e3~1e4)은 두 task 모두 유의하게 나쁘며 corr QC 탈락(환자 84~111명)과 얽혀 있음(앞선 관찰 4와 동일).
  - 한계: 앞과 동일(112명, C 고정, 탐색 실험). 3-class는 클래스 21/48/43으로 불균형 — macro AUC이므로 소수 클래스(control 21명)의 CI 폭이 큼.
  - DB: `experiment.runs` #126~#164 갱신 + 신규 9건(run_name `guide_aecd_{조건}_20260907`), `run_metrics`에 task=`prostate_three_class` 행 87건 추가.
  - 산출물: `aecd/{condition}/patient_oof_three_class.csv`, `spectrum_oof_seed42_three_class.csv`, `summary.csv`에 `macro_auc_*`, `d3__*` 컬럼 추가.

- [x] PL-3 보충 (2026-09-09): 암 vs **정상만**(질환대조 제외) 이진 OOF + PR-AUC 정리
  - 코호트 `aecd_ctrl` = 전립선암 43 / 정상 20 (63명). 같은 환자·같은 전처리, 두 군만으로 학습·OOF. 임계값 0.5.

    | 조건 | AUC [CI] | PR-AUC [CI] | Sens | Spec | PPV | NPV |
    |---|---|---|---|---|---|---|
    | production (cal✓ SG5) | 0.755 [0.611, 0.867] | 0.870 [0.718, 0.935] | 0.86 [0.72, 0.95] | 0.45 [0.22, 0.68] | 0.77 | 0.60 |
    | no_cal_sg11 (PL-1 재현) | 0.705 [0.550, 0.831] | 0.821 [0.624, 0.913] | 0.79 [0.65, 0.90] | 0.50 [0.26, 0.70] | 0.77 | 0.53 |

  - 관찰: 질환대조를 빼면 AUC가 0.82→0.76으로 내려간다(정상 20명뿐이라 CI 폭 ±0.13). 양성 비율이 68%라 PR-AUC는 0.87로 높게 보이지만
    유병률 기준선이 0.68이므로 그 위의 이득만 의미 있다. 특이도 0.45 [0.22, 0.68]은 정상 20명으로는 판별 불가.
  - PR-AUC (양성=전립선암, 3군 코호트): production 0.678 [0.493, 0.795] / no_cal_sg11 0.683 [0.484, 0.802] / 매핑 LR-reference 0.577 [0.445, 0.712] /
    매핑 ResNet 0.348 [0.246, 0.448] / 보라매 109 publication 0.588 [0.408, 0.720]. 유병률(0.38~0.38)이 PR 기준선.
  - DB: `experiment.runs` #204 (`guide_aecd_ctrl_production_20260907`), #206 (`guide_aecd_ctrl_no_cal_sg11_20260907`). 산출물 `results/preprocessing_lab/guide_pipeline_20260907/aecd_ctrl/`.
  - 스크립트: `run_guide_pipeline.py --cohort aecd_ctrl` (다른 세션의 PS 보정 조건 추가와 같은 파일에 미커밋 상태 — 해당 세션 커밋 후 반영).
  - 그림: `src/sers/visualization/performance.py` — 혼동행렬은 행(실제 클래스)마다 클래스 색 램프로 칠한다 (`cbb3f30` 이후 수정).

- [x] PL-3 추가 (2026-09-09): 표준물질(PS) 축 보정 조건 2개 — `cal_ps`, `cal_ps_urea` ✅
  - **배경**: PL-3 "production" 조건의 calibration은 검체 내 urea 1001.4 cm⁻¹ 피크 기준 스펙트럼별 보정(config.yaml)인데,
    DB `experiment.preprocessing_methods`에는 `calibration_astm_reference`(표준물질 보정, ASTM E1840)로 연결돼 있었다 —
    **명칭 불일치**. 결정된 표준물질 보정은 AECD API 파이프라인에만 있었고 AUC 비교가 없었다. 사용자 요청으로 추가 실행.
  - **정의**: `measurement.calibrations`(PS, raman_shift, pass)의 run(측정일+장비) 단위 global_shift를 x축에서 뺌
    (x − shift; `aecd_api_model_mean_spectrum_clinical_performance.py`와 동일). 5 run 값: −0.027 / +0.024 / +0.111 / +0.125 / +0.203 cm⁻¹.
    urea 보정은 스펙트럼별 shift sd 2.2 / |max| 6.9 cm⁻¹, fail-safe(0) 4.6% — 두 보정의 크기가 한 자릿수 이상 다르다.
  - **결과** (112명, 5시드, 짝지은 bootstrap Δ; 나머지 규약 PL-3과 동일):

    | 조건 | spectra(QC 후) | 이진 AUC seed42 [CI] | 5시드 | Δ vs production | Δ vs no_cal | 3-class macro AUC seed42 | 5시드 | Δ3 vs production |
    |---|---|---|---|---|---|---|---|---|
    | no_cal | 10,823 | 0.735 [0.637, 0.821] | 0.730±0.015 | −0.082 [−0.158, −0.023] | 0 | 0.649 | 0.664 | −0.066 [−0.134, 0.001] |
    | production (urea) | 10,413 | 0.817 [0.728, 0.885] | 0.796±0.028 | 0 | +0.082 [0.023, 0.158] | 0.715 | 0.682 | 0 |
    | **cal_ps** (PS만) | 10,821 | 0.730 [0.628, 0.813] | 0.745±0.010 | **−0.087 [−0.158, −0.030]** | −0.005 [−0.030, 0.016] | 0.647 | 0.665 | **−0.067 [−0.135, −0.003]** |
    | **cal_ps_urea** (PS→urea) | 10,413 | 0.811 [0.719, 0.883] | 0.799±0.022 | −0.006 [−0.055, 0.046] | +0.076 [0.017, 0.151] | 0.661 | 0.682 | −0.054 [−0.091, −0.016] |

  - **관찰 (해석·채택은 사용자 몫)**:
    1. PS 보정 단독은 보정 없음(no_cal)과 구별되지 않는다 (Δ −0.005, CI가 0 포함). run 간 이동 0.03~0.20 cm⁻¹는 그리드 간격
       1.92 cm⁻¹의 1/10 이하라 이 코호트(단일 장비, 5일)에서는 효과가 나올 수 없는 크기다.
    2. PS 뒤에 urea 보정을 얹으면 production과 같다 (이진 Δ −0.006, 5시드 0.799 vs 0.796). urea 보정 후 잔여 shift 분포도 동일(sd 2.2).
    3. 3-class에서 cal_ps_urea seed42 Δ −0.054 [−0.091, −0.016]는 CI가 0을 제외하지만 5시드 평균은 0.682 = production 0.682로 같다 —
       seed 42 하나의 fold 배치 효과. 단일 seed CI만으로 읽지 말 것.
    4. **해석상 주의**: 이 코호트에서 성능에 영향을 주는 것은 장비 축 드리프트(PS가 잡는 것)가 아니라 스펙트럼별 urea 피크 위치 변동
       (sd 2.2 cm⁻¹)이다. 장비 드리프트가 0.2 이하인데 검체 피크가 ±7 움직인다면 urea "보정"은 장비 보정이 아니라 생화학적 피크 위치
       차이(예: 1001 urea vs 1004 phenylalanine 겹침)를 정렬해 버리는 것일 수 있다 — 이것이 신호 제거인지 정렬인지는 미확인.
       PL-3 앞 관찰(urea 효과가 SG 5에서만 나타나고 SG 11에서는 사라짐)과 함께 봐야 한다.
  - **기록 정정 필요(미조치)**: DB `experiment.runs`의 기존 `production`/`cal_*` run들은 method `calibration_astm_reference`(id 1)에
    연결돼 있으나 실제 알고리즘은 urea 피크 보정이다. 새 run #208(cal_ps)·#210(cal_ps_urea)만 이 method가 맞다. 기존 run의 method 연결을
    바꿀지(또는 urea 보정 method를 새로 등록할지)는 사용자 결정 후 처리.
  - 코드: `run_guide_pipeline.py`에 `ps_calibrate` 옵션·`_load_ps_shifts()`(DB 조인, 대응 없으면 오류) 추가, `DELTA_REFERENCES`에 `no_cal` 추가.
  - 산출물: `results/preprocessing_lab/guide_pipeline_20260907/aecd/{cal_ps,cal_ps_urea}/`, `summary.csv` 갱신.
    DB: `experiment.runs` #208, #210 (run_measurements 각 13,552 연결, 시드·CI 행 적재).

- [x] PL-3 보충 2 (2026-09-09): "암 vs 정상이 왜 0.76인가" — 기존 OOF 점수·DB 메타데이터만으로 확인 (재학습 없음)
  - (a) 3군으로 학습한 production 모델의 암-vs-비암 OOF 점수를 부분집합에서 평가(5시드 평균): 암 vs 전체 비암 0.796,
    **암 vs 정상 0.791**, 암 vs 질환대조 0.797. → 정상이 더 어려운 군이 아니다. 두 군만으로 학습한 0.741은 **학습 음성 예시가 69→20명으로
    준 효과**다. 3-class 모델의 pairwise: 암 vs 정상 0.768, 암 vs 질환대조 0.791, 질환대조 vs 정상 0.576(거의 구분 안 됨).
  - (b) **측정일 교차표** (`measurement.runs` × `clinical.diagnoses`):

    | 측정일 | 정상 | 질환대조 | 암 | 측정자 |
    |---|---|---|---|---|
    | 8/10 | 2 | 3 | 0 | 엄찬호 |
    | 8/11 | 5 | 17 | 0 | 엄찬호 |
    | 8/12 | 9 | 14 | 0 | 엄찬호 |
    | 8/13 | 4 | 15 | 8 | 고은혜 |
    | 8/14 | 0 | 0 | 35 | 고은혜 |

    암 43명 중 35명이 8/14 단독 측정(그날 비암 0). 정상 20명 중 16명은 암이 하나도 없던 8/10~12에 측정.
  - (c) 비암 환자의 OOF 암 확률이 **측정일 순서대로 오른다**: 8/10 0.09 → 8/11 0.22 → 8/12 0.40 → 8/13 0.42 (암은 8/13 0.55, 8/14 0.56).
    같은 날(8/13, 암 8 / 비암 19) 안에서만 평가하면 AUC 0.62~0.76(5시드), 전체 0.80보다 낮다. 8/13 정상 4 vs 암 8은 0.50~0.81로 판별 불가.
  - **결론(사실 수준)**: 현재 3군 코호트의 AUC 0.8은 암 여부와 측정일/측정자가 완전히 얽힌 값이며, 날짜 효과가 성능을 부풀리는 방향이다.
    암 vs 정상 0.76이 낮은 직접 원인은 학습 음성 수 감소이고, 그 위에 날짜 혼입이 얹혀 있다. 해석·조치는 사용자 판단.
  - 확인 방법(제안): ① 정상·암을 같은 날 같은 측정자가 교차 측정(OV-2 설계 A/B) — 유일한 근본 해결. ② 기존 검체 일부를 날짜 교차 재측정.
    ③ 분석 측: leave-one-day-out CV, 날짜 공변량 포함 모델, 날짜별 중심화 후 재평가. ④ 정상군 n 확대(현재 20명, 특이도 CI ±0.23).
  - 산출: 이 절의 수치는 `results/preprocessing_lab/guide_pipeline_20260907/aecd/production/patient_oof_*.csv` + DB 조인으로 재현 (스크립트 없음, 세션 내 계산).

- [x] PL-3 추가 (2026-09-09, 사용자 결정): **축 보정은 PS+Si 표준물질로만** — `cal_ps_si`, `cal_ps_si_sg11` ✅
  - **결정 근거**: urea 1001.4 피크 보정은 검체 화학을 정렬할 수 있다(위 cal_ps 항목 관찰 4). 후향 DACR 데이터의 urea shift는
    sd 2.41 / |max| 7.06 cm⁻¹, fail-safe 12.5%이고 **그룹별 평균이 다르다** (CRC +2.37, CPAN +2.04, LUN +1.65 vs PRO 0.03, DIA 0.01,
    HBP −0.65; `results/preprocessing_dacr/calibration_shifts.csv`). 그룹 = 병원이므로 urea 보정은 병원 배치 효과와 생화학 차이를
    구분 없이 지운다. 장비 축 오차는 표준물질로 잰 값이 run당 PS −0.03~+0.20, Si +0.28~+0.51 cm⁻¹로 한 자릿수 작다.
  - **구현**: `sers.preprocessing.fit_standard_material_axis(ref, obs)` → err(x)=a+b·x 최소제곱, `apply_standard_material_axis(x,a,b)` →
    x−(a+b·x). run(측정일+장비)별로 `measurement.calibrations`의 PS 8피크 + Si 520.7 피크(총 9점)에 적합. 적합 결과(5 run):
    b = −1.5e-4 (5일 모두 동일), a = +0.17~+0.40 → 보정량 520 cm⁻¹에서 +0.09~+0.32, 2000 cm⁻¹에서 −0.14~+0.10.
    PS 관측피크는 DB에 global_shift 파생값으로 저장돼 있어 기울기는 사실상 Si↔PS 차이가 정한다 (Si만 독립 관측).
    `config.yaml do_calibration: false`(urea OFF). 테스트 2건 추가(`tests/test_preprocessing.py`).
  - **결과** (112명, 5시드; 이전 항목과 같은 규약):

    | 조건 | spectra | 이진 AUC seed42 [CI] | 5시드 | Δ vs production(urea) | Δ vs no_cal | 3-class 5시드 | Δ3 vs production |
    |---|---|---|---|---|---|---|---|
    | cal_ps_si (SG 5) | 10,828 | 0.766 [0.671, 0.842] | 0.745±0.030 | −0.051 [−0.118, 0.004] | +0.031 [−0.011, 0.074] | 0.665±0.043 | −0.060 [−0.116, −0.013] |
    | cal_ps_si_sg11 | 11,332 | 0.765 [0.660, 0.851] | 0.765±0.023 | −0.052 [−0.134, 0.017] | +0.030 [−0.036, 0.100] | **0.690±0.023** | +0.005 [−0.045, 0.060] |
    | (참고) no_cal_sg11 | 11,331 | 0.782 | 0.794±0.011 | −0.036 [−0.116, 0.035] | | 0.674±0.020 | |
    | (참고) production urea SG5 | 10,413 | 0.817 | 0.796±0.028 | 0 | | 0.682±0.025 | 0 |

  - **관찰 (채택 판단은 사용자)**:
    1. urea를 빼면 이진 AUC 5시드 평균이 0.03~0.05 낮아지지만 짝지은 CI는 모두 0을 포함한다. 3-class는 SG 11에서 0.690으로 전 조건 중 최고.
    2. urea 없는 조건들끼리(no_cal 0.730, cal_ps 0.745, cal_ps_si 0.745, no_cal_sg11 0.794, cal_ps_si_sg11 0.765)의 차이는
       그리드 간격(1.92 cm⁻¹)보다 작은 축 이동에서 나온 것이라 **측정 잡음 수준(±0.03)** 으로 읽어야 한다. PS+Si 보정의 효과를
       이 코호트에서 AUC로 증명할 수는 없다 — 단일 장비·5일이라 보정할 드리프트가 거의 없기 때문이며, 이는 예상된 결과다.
    3. urea 제거 후에는 SG 창 선택이 다시 열린다(SG 11이 5보다 이진·3-class 모두 높음). PL-2 기록 누락과 함께 재확인 필요.
  - **후속 영향 (미조치, 사용자 확인 필요)**:
    (a) 후향 7암종 데이터(1,630명)는 표준물질 기록이 없어 PS/Si 보정이 불가능하다 — `do_calibration: false`로 재전처리하면 보정 없음.
    (b) STK-V2(production 모델)는 urea 보정된 데이터로 학습됐다. 추론 코드(`scripts/deployment`)는 urea 보정을 적용하지 않으므로
        이미 학습/추론 불일치가 있었고, 이번 결정으로 새 데이터 파이프라인과 기존 모델의 전처리는 공식적으로 갈라진다.
    (c) `experiment.preprocessing_methods` `calibration_astm_reference`에 연결된 기존 urea run(production 등)의 method 정정.
  - DB: `experiment.runs` cal_ps_si / cal_ps_si_sg11 적재(run_measurements 각 13,552). `summary.csv` Δ 기준에 `cal_ps_si` 추가.

- [x] PL-3 재실행 (2026-09-09): **PS+Si 축 보정 기준 사다리 28조건 + SG 창 5/7/9/11/21 확인** ✅
  - **왜**: 위 결정으로 urea 보정이 빠지면 기존 29조건(모두 urea ✓)의 Δ는 새 기준과 비교 불가. 요인 조건을 urea 없이 PS+Si 위에서
    재실행. 기준점 `ps_despike` = PS+Si 1차 축보정 ✓, WH despike ✓, trim 400–2200, SG 5, rolling_min 101, SNV (기존 cal_despike에서
    urea→PS+Si만 교체). 조건명은 `ps_` 접두어. SG 7·9는 사용자 요청("SG 5로 할지 11로 할지 확인")으로 추가.
  - **기준점**: 이진 0.746 [seed42] / 5시드 0.766±0.017, 3-class 0.656±0.027 (spectra 10,829).
  - **이진 AUC, 5시드 평균 상위** (Δ = seed42 짝지은 bootstrap vs ps_despike; 굵게 = CI가 0 제외):

    | 조건 | spectra | 5시드 | Δ [CI] | 3-class 5시드 | Δ3 [CI] |
    |---|---|---|---|---|---|
    | ps_trim_600_1800 | 10,516 | 0.817±0.021 | **+0.067 [0.003, 0.148]** | 0.687 | +0.042 [−0.020, 0.112] |
    | ps_sm_wavelet | 11,574 | 0.804±0.039 | +0.051 [−0.015, 0.118] | 0.673 | +0.031 |
    | ps_sm_none | 10,475 | 0.790±0.033 | **+0.086 [0.036, 0.160]** | 0.683 | +0.020 |
    | ps_sm_sg9 | 11,222 | 0.782±0.052 | **+0.068 [0.015, 0.134]** | 0.693 | +0.038 |
    | ps_sm_gauss1 | 11,208 | 0.780±0.019 | +0.044 | 0.670 | +0.038 |
    | ps_bl_airpls_1e6 | 10,841 | 0.778±0.035 | +0.029 | 0.688 | **+0.068 [0.015, 0.126]** |
    | ps_bl_als_1e5 | 10,246 | 0.776±0.037 | **+0.071 [0.019, 0.135]** | 0.660 | +0.022 |
    | ps_sm_sg11 | 11,335 | 0.774±0.031 | +0.014 [−0.071, 0.074] | 0.675 | **+0.091 [0.042, 0.146]** |
    | ps_sm_median5 | 11,257 | 0.774±0.025 | +0.031 | **0.704±0.015** | **+0.089 [0.038, 0.141]** |
    | ps_sm_sg7 | 11,078 | 0.769±0.041 | −0.023 [−0.089, 0.040] | 0.698 | **+0.060 [0.007, 0.117]** |
    | ps_despike (기준) | 10,829 | 0.766±0.017 | 0 | 0.656 | 0 |
    | ps_norm_minmax | 10,823 | 0.758±0.036 | +0.014 | 0.658 | **+0.050 [0.006, 0.095]** |
    | ps_norm_l2 | 10,818 | 0.727±0.028 | +0.011 | 0.637 | +0.010 |
    | ps_bl_{als,arpls,airpls}_1e3~1e4 | 4,427~8,952 | 0.629~0.700 | **−0.07~−0.16** (als_1e4, arpls_1e4, als_1e3, arpls_1e3 CI 0 제외) | 0.58~0.65 | — |
    (전체 32행: `summary.csv`, `d_ps_despike*`/`d3__ps_despike*` 컬럼)

  - **SG 창 곡선** (PS+Si 기준, despike ✓; cal_ps_si 계열은 despike ✗):

    | 창 | 이진 5시드 | Δ vs SG5 [CI] | 3-class 5시드 | Δ3 vs SG5 [CI] |
    |---|---|---|---|---|
    | 없음 | 0.790±0.033 | **+0.086 [0.036, 0.160]** | 0.683±0.023 | +0.020 [−0.037, 0.079] |
    | 5 (현 config) | 0.766±0.017 | 0 | 0.656±0.027 | 0 |
    | 7 | 0.769±0.041 | −0.023 [−0.089, 0.040] | 0.698±0.012 | **+0.060 [0.007, 0.117]** |
    | 9 | 0.782±0.052 | **+0.068 [0.015, 0.134]** | 0.693±0.029 | +0.038 [−0.011, 0.089] |
    | 11 | 0.774±0.031 | +0.014 [−0.071, 0.074] | 0.675±0.028 | **+0.091 [0.042, 0.146]** |
    | 21 | 0.771±0.043 | +0.007 [−0.063, 0.079] | 0.679±0.030 | +0.041 [−0.015, 0.100] |
    | (despike ✗) 5 / 11 | 0.745±0.030 / 0.765±0.023 | | 0.665±0.043 / 0.690±0.023 | |

  - **관찰 (해석·채택은 사용자 몫)**:
    1. 잡음 바닥: 5시드 sd 0.02~0.05, 조건 30개 다중비교라 seed42 Δ의 CI가 0을 벗어난 항목 중 1~2개는 우연으로 기대. **5시드 평균 차이가
       ±0.03 이내인 것은 구분 불가**로 읽는다(PL-3 원 결론과 동일). 이 기준으로 기준점을 확실히 넘는 조건은 없고, 확실히 나쁜 조건은
       λ≤1e4 baseline(QC 탈락 급증, spectra 4~9천으로 감소)뿐이다. 즉 "이기는 조건 없음"은 urea 제거 후에도 유지.
    2. 이진에서 seed42 Δ가 유의한 4건(trim 600–1800, smoothing 없음, SG 9, ALS 1e5)은 5시드 평균으로는 +0.01~+0.05. trim 600–1800은
       5시드 0.817로 이번 사다리 최고이며 urea 사다리에서도 +0.013으로 같은 방향 — 유일하게 두 사다리에서 방향이 일치하는 양의 조건.
    3. **SG 창**: 이진은 smoothing 없음(0.790) > 9(0.782) > 11(0.774) ≈ 21 ≈ 7 > **5(0.766)**; 3-class는 7(0.698) ≈ 9(0.693) > 11 ≈ 21 ≈ 없음 > **5(0.656)**.
       현 config의 SG 5는 두 task 모두 최하위이나 곡선은 단조가 아니고 sd 안에서 겹친다. PL-2(표준물질 PS/Si FWHM·SNR)가 5를 고른
       근거와 임상 AUC(7~11 또는 없음)가 상충 — **창 결정은 사용자 몫**, 데이터는 5를 지지하지 않는다.
    4. 3-class에서 유의한 양의 Δ가 median5·SG11·airpls1e6·SG7·minmax·arpls1e7 6건으로 이진보다 많다 — 3군 판별이 전처리에 더 민감하다는
       PL-3 원 관찰과 일치. 단 median5(0.704)는 이진에서 중립.
  - 산출물: `results/preprocessing_lab/guide_pipeline_20260907/aecd/ps_*/` (28 dir), `summary.csv` Δ 기준에 `ps_despike` 추가.
    DB: `experiment.runs` guide_aecd_ps_*_20260907 28건 적재(run_measurements 각 13,552). 코드: `PS_LADDER`, `--conditions PS_LADDER`.

- [x] PL-3 보충 3 (2026-09-09): 날짜 효과 vs lot 효과 — `measurement.runs`의 reagent/strip lot으로 분리 시도
  - **시약 lot은 하나**(BCCP0922, 6 run 전부) → 시약 lot은 이 코호트 안에서 어떤 차이도 설명할 수 없다.
  - **strip lot이 암/비암과 완전히 겹친다**: 비암 전부 lot 1(SK20260804B01, run 1~4, 8/10~13), 암 전부 lot 6(SK20260804C01, run 5~6, 8/13~14).
    8/13에도 비암은 lot 1(run 4), 암은 lot 6(run 5). → 암 신호와 strip lot 효과는 이 데이터로 분리 불가. "보충 2"의 같은 날(8/13) 평가도 lot은 못 걷어낸다.
  - **같은 strip lot·같은 시약 lot 안에서 OOF 암 확률이 날짜순으로 오른다** (production, 5시드 평균):

    | lot | 날짜 | 측정자 | 정상 | 질환대조 |
    |---|---|---|---|---|
    | 1 | 8/10 | 엄찬호 | 0.15 (n=2) | 0.05 (3) |
    | 1 | 8/11 | 엄찬호 | 0.23 (5) | 0.22 (17) |
    | 1 | 8/12 | 엄찬호 | 0.40 (9) | 0.41 (14) |
    | 1 | 8/13 | 고은혜 | 0.46 (4) | 0.41 (15) |
    | 6 | 8/13 | 고은혜 | — | 암 0.55 (8) |
    | 6 | 8/14 | 고은혜 | — | 암 0.56 (35) |

    정상과 질환대조가 같은 궤적을 그리므로 군 구성 차이가 아니다. 측정자 교체(8/12→13)에서 점프가 없으므로 측정자도 아니다.
    PS 표준물질 축 shift는 0.02~0.20 cm⁻¹로 날짜별 차이가 없어 파수축 drift도 아니다. → lot·시약·측정자·축과 무관한 **일 단위 drift**가 실재한다.
    4일간 이동 폭(0.05→0.46)이 암-비암 간격(비암 0.41→암 0.55)보다 크다.
  - 후보(미검증): strip 개봉 후 경과일(lot 제조 8/4), 검체 보관/해동 순서, 장비 강도 drift(OV-2: PS 강도 일간 CV 26%), 환경(온습도는 24.3~24.8℃/55~61%로 패턴 없음).
  - 분리 실험(제안): ① 8/10 검체 일부를 같은 lot 1로 다시 측정 → 점수가 오르면 drift 확정. ② 암을 lot 1로, 정상을 lot 6으로 교차 측정 → lot 효과 크기.
    ③ 같은 QC 검체(소변 pooled)를 매일 측정해 drift 추적. ④ PS 강도(축 아님)를 날짜별로 확인.

- [x] PL-3 보충 4 (2026-09-09): 표준물질(PS/Si)로 날짜 효과를 확인·제거할 수 있는가
  - 데이터: `data/mapping/Thermo Reference` PS·Si 각 5회/일, 보정일 8/4~8/7, 8/10~8/14. 검체와 같은 production 전처리(cal→SG5→rolling_min→SNV) 적용.
  - **(a) PS 스펙트럼만으로 측정일이 완벽히 구분된다**: 8/10~14 PS 25개, SNV 후 nearest-centroid LOO → 25/25 정답(우연 20%). PC1 설명력 29%.
    화학적으로 동일한 표준물질에도 날짜 지문이 찍혀 있다 = 장비(레이저/분광기) 상태가 날마다 다르다.
  - **(b) 그런데 그 지문은 검체의 날짜 이동과 방향이 다르다**: 날짜별 편차 벡터(그날 평균 − 전체 평균)를 비암 검체와 PS 사이에 상관 →
    대각선 −0.07 / 0.03 / −0.05 / 0.08 (전부 |r|<0.2). PS의 8/10→8/13 변화 방향에 검체를 투영해도 비암 0.27 → −0.85 → 0.40 → 0.38로 단조 추세 없음
    (검체 점수는 0.05→0.46 단조 증가). → **PS가 잡는 장비 광학 상태 변화는 검체 drift를 설명하지 못하고, 따라서 PS로 보정할 수도 없다.**
  - (c) 검체로 학습한 LR에 PS/Si를 넣은 "암 확률"은 PS 0.91~0.99(포화), Si 0.13~0.57(무작위) — 분포 밖 입력이라 해석 불가. 이 검정은 무효.
  - (d) 부수 발견: PS raw 강도가 8/6(40, peak 456) → 8/7(102, peak 1428)에서 2.5배 뛰고 8/10~14는 96~112로 안정. 8/7 전후에 장비 설정(레이저 출력/정렬)이
    바뀐 것으로 보이나 기록 없음 — 측정팀 확인 필요. 8/10~14 안에서는 강도 drift가 아니다.
  - **함의(사실 수준)**: 검체 drift는 PS가 보지 못하는 곳, 즉 SERS strip(기판)·검체-기판 상호작용·검체 취급 쪽에서 온다. PS는 일반 Raman 표준이라
    SERS 증강 상태를 반영하지 않는다. 날짜 효과를 추적·보정하려면 **strip 위에서 측정하는 QC 검체**(pooled 소변 또는 설계가이드 ⑥의 internal standard,
    예 4-MBA)를 매일 같이 측정해야 한다. 이 권고는 측정팀 보고서 3-4항과 같다.
  - 재현: 세션 내 계산(스크립트 없음). 입력 = `run_guide_pipeline.py`의 `load_aecd`/`run_condition(production)` + Thermo Reference 파일.

- [x] PL-3 확인 실험 (2026-09-09): **SG 창·baseline 유무 20시드 재확인** (사용자 질문: "SG 없애는 게 제일 괜찮아 보이는데 다시 확인", "baseline 보정 없는 게 좋다는 건가?") ✅
  - 조건 7개를 5→20시드로 재계산(`--force`, seeds 42,7,123,2024,31337,1~6,8~16). 새 조건 `ps_bl_none`(baseline 보정 없음), `ps_sm_none_bl_none`(SG·baseline 둘 다 없음).
    Δ는 같은 seed(같은 fold 분할)끼리 짝지은 20개 차이의 평균과 2.5~97.5 백분위, wins = Δ>0인 seed 수. 산출물 `sg_baseline_confirm_20seeds.csv`.

    | 조건 | spectra(QC 후) | 이진 20시드 | Δ vs SG5 [seed 범위] | wins | 3-class 20시드 | Δ3 [범위] | wins |
    |---|---|---|---|---|---|---|---|
    | ps_despike (SG 5, 기준) | 10,829 | 0.753±0.039 | 0 | | 0.668±0.033 | 0 | |
    | ps_sm_none (SG 없음) | 10,475 | **0.790±0.033** | +0.037 [−0.039, +0.109] | 14/20 | 0.671±0.022 | +0.003 | 14/20 |
    | ps_sm_sg7 | 11,078 | 0.779±0.029 | +0.026 [−0.054, +0.110] | 14/20 | 0.675±0.029 | +0.007 | 13/20 |
    | ps_sm_sg9 | 11,222 | 0.781±0.033 | +0.028 [−0.030, +0.090] | 14/20 | **0.684±0.025** | +0.016 | 12/20 |
    | ps_sm_sg11 | 11,335 | 0.774±0.022 | +0.021 [−0.057, +0.068] | 14/20 | 0.678±0.025 | +0.010 | 10/20 |
    | ps_bl_none (baseline 없음, SG 5) | **13,242** | 0.762±0.038 | +0.009 [−0.073, +0.094] | 10/20 | 0.679±0.029 | +0.011 | 11/20 |
    | ps_sm_none_bl_none | 13,238 | 0.787±0.034 | +0.034 [−0.047, +0.098] | 15/20 | 0.668±0.023 | −0.000 | 10/20 |

  - **관찰 (채택은 사용자 몫)**:
    1. **SG**: 이진에서 SG 5 이외의 모든 선택(없음/7/9/11)이 +0.02~+0.04, 14/20 seed에서 우세로 방향은 일관되나 seed 범위는 0을 포함.
       "SG 없음"이 수치상 최고(0.790)이지만 7/9/11과의 차이(≤0.016)는 잡음 안. **데이터가 기각하는 것은 SG 5뿐이고, 없음·7·9·11은 구분 불가.**
       3-class는 20시드에서 smoothing 효과가 사라짐(Δ ≤ 0.016, wins 10~13/20) — 5시드 때 "유의"했던 SG7/SG11 Δ3(+0.06/+0.09)는 seed 42 우연이었다.
    2. **baseline**: 없애도 이진 +0.009 / 3-class +0.011, wins 10~11/20 → 효과 없음. 단 QC 통과가 10,829→13,242(탈락 2,723→310)로 급증:
       baseline이 남으면 스펙트럼 간 상관이 높아져 Stage-2 corr QC가 사실상 꺼진다. "baseline 없음"은 QC 없음과 같이 읽어야 하며,
       rolling_min을 빼도 성능이 안 떨어지는 것은 SNV+LR이 baseline을 흡수한다는 뜻이지 baseline 보정이 해롭다는 뜻이 아니다.
       앞 사다리의 λ≤1e4 ALS/arPLS/airPLS 악화는 "baseline 보정" 자체가 아니라 과도한 보정+QC 탈락 폭증의 효과.
    3. **600–1800 절단**은 baseline이 아니라 파수 구간 조건(가이드 ③). 우리 규격 400–2200을 유지하기로 함(사용자 확인) — 참고값으로만 기록.
    4. SG 없음 채택 시 despike(WH z=6)가 우주선 제거의 유일한 단계가 되므로 despike는 켜 둬야 한다. 5시드 기준 0.766 vs 20시드 0.753처럼
       기준값 자체가 seed 수에 따라 0.01 움직인다 — 이 코호트에서 0.02 이하 차이는 어떤 seed 수로도 확정 불가.
  - DB: 7개 run에 20시드 metric 행 추가 적재(guide_aecd_ps_*_20260907; ps_bl_none·ps_sm_none_bl_none 신규).

- [x] PL-3 보충 5 (2026-09-09): 날짜 변동의 정체 — raw 강도·QC·보관기간·채취시기 확인
  - **raw 강도는 날짜 추세가 없다** (비암, 검체 중앙값): 평균 강도 8/10 2,286 → 8/11 1,687 → 8/12 1,718 → 8/13 1,911 a.u. 1001 피크도 같음. 검체 내 반복 CV 0.15~0.19,
    corr-QC 통과 점 수 93~101/121로 날짜 차이 없음. → 증강 크기·잡음·QC의 문제가 아니라 **형태**의 문제.
  - **SNV 후 이동 밴드**: 비암 8/13 − 8/10 차이는 1421~1427 cm⁻¹(−0.65)와 1600~1613 cm⁻¹(+0.67)에 집중. 1001/1450 비 1.11 → 1.00. 단 8/11의 이동 방향은
    8/13 방향과 무관(cos 0.03)하고 8/12는 유사(0.74) — 단일 축 drift가 아니라 run마다 다른 이동.
  - **비암 검체는 채취일 순서대로 측정됐다** (측정일 = 보관기간): 8/10 3월 초 채취(보관 ~155일) → 8/11 3월 중~4월 초(~140일) → 8/12 4~5월(~100일) → 8/13 5~6월(~75일).
    암은 8/13이 3월 채취(140~157일), 8/14가 4~6월(52~135일).
  - **그러나 보관기간 자체는 원인이 아니다**: 비암 점수 vs 보관일수 Spearman r = −0.57(전체)이지만 **측정일을 고정하면 r ≈ 0**(8/11 −0.18, 8/12 −0.12, 8/13 +0.22, 모두 p>0.3).
    암은 보관 52~157일에 걸쳐 r = −0.02, 8/13(140~157일) 0.549 vs 8/14(52~135일) 0.562로 차이 없음. 채취 시기 효과도 암에서 보이지 않음.
  - **남는 것 = run(하루) 단위 요인**, lot 1의 run 1~4 사이에서 발생. 배제된 것: 시약 lot, strip lot, 측정자, 파수축, 레이저/분광기 광학 상태(PS), 강도, 잡음, QC, 보관기간, 채취시기, 온습도.
    후보: strip 준비 상태(개봉 후 시간, 환원 반응 시간/온도, 건조), 그날의 검체 처리 순서·해동 시간. 기록이 없어 데이터로는 더 못 좁힌다.
  - 줄이는 방법(제안, 미검증): ① 측정 설계 — 매 run에 세 군 교차 배치(편향→잡음으로 전환), strip QC 검체 매일 측정, strip 준비 시간·해동 시간 기록.
    ② 분석 — 각 run에 세 군이 있을 때만 run별 중심화/ComBat 가능. 현재 데이터는 불가. ③ 모델 — 1400~1440·1590~1620 밴드를 제외/감쇠한 뒤 날짜 구분력과 암 AUC를
    같이 재평가하는 검정은 지금 데이터로 가능(미실행). ④ leave-one-day-out CV로 다른 날 일반화 성능 측정.

- [x] PL-3 보충 6 (2026-09-09): 밴드 제외 검정 — 1425·1600 밴드를 빼면 날짜 구분력 / 암 AUC가 변하는가 ✅
  - 스크립트(보존): `scripts/analysis/preprocessing_lab/band_mask_day_test.py`. 산출물 `results/preprocessing_lab/band_mask_20260909/`
    (`band_mask_results.csv`, `random_bands.json`, `band_by_day.png`, `mask_results.png`).
  - 설계: production 전처리 X(935 feature)에서 (a) 좁은 제외 1400–1440·1590–1620, (b) 넓은 제외 1380–1460·1570–1640, (c) 같은 총 폭의 무작위 위치 제외 5회(대조).
    암 AUC = PL-3 규약(LR, 환자 OOF, 5시드). 날짜 구분력 = 비암 69명(lot 1, run 1~4)만으로 측정일 4클래스 multinomial LR, 같은 CV, 환자 macro OvR AUC.

    | 조건 | feature | 암 AUC | 비암 점수 8/13−8/10 | 날짜 macro AUC | 날짜 정확도 |
    |---|---|---|---|---|---|
    | 전체 | 935 | 0.795±0.028 | 0.33 | 0.792±0.019 | 0.57 |
    | 좁은 밴드 제외 | 899 | 0.800±0.026 | 0.32 | 0.796±0.017 | 0.57 |
    | 넓은 밴드 제외 | 857 | 0.795±0.027 | 0.33 | 0.799±0.017 | 0.58 |
    | 무작위 제외 ×5 | 898~899 | 0.794~0.801 | 0.32~0.33 | 0.790~0.795 | 0.56~0.58 |

  - **결과: 아무것도 변하지 않는다.** 두 밴드는 날짜별 평균 차이가 가장 큰 곳일 뿐, 날짜 정보는 스펙트럼 전체에 분산되어 있다(비암만으로 측정일을 macro AUC 0.79,
    정확도 0.57로 맞힘 — 우연 0.5 / 0.33). 암 신호도 이 밴드에 의존하지 않는다. → "밴드를 피하는 모델"은 답이 아니다.
  - 그림 `band_by_day.png`: 1400–1440에서 비암은 8/10·8/11(높음)과 8/12·8/13(낮음) 두 묶음으로 갈리고 암은 후자와 같다. 1590–1620에서는 8/13 > 8/12 > 8/10 > 8/11로 날짜 순서가 아니다.
    즉 날짜 이동은 단조 drift가 아니라 run 단위 이동이며, 암 검체는 그 이동의 어느 상태와도 겹친다.

- [x] PL-3 확인 실험 2 (2026-09-10): **despike 유무** (사용자 질문 "despike는 의미 없는데 왜 기준점에 들어가 있나") ✅
  - 배경: PL-3 기준점(cal_despike → ps_despike)에 despike가 켜진 것은 설계가이드 6단계를 모두 켠 상태에서 한 단계씩 바꾸는 요인 설계의
    편의였고 성능 근거는 없었다(PL-1 off 0.794 vs on 0.758; PL-3 urea off 0.796 vs on 0.798; PS+Si SG5 off 0.745 vs on 0.753/20시드).
    SG 없음 조건에서 despike까지 끈 조합은 미실행 상태였음.
  - 조건 `ps_sm_none_nodespike` 20시드(SG·despike 없음, PS+Si, rolling_min, SNV) vs `ps_sm_none`(despike ✓) 같은 seed 짝지음:

    | 조건 | spectra | 이진 20시드 | 3-class 20시드 |
    |---|---|---|---|
    | ps_sm_none (despike ✓) | 10,475 | 0.790±0.033 | 0.671±0.022 |
    | ps_sm_none_nodespike (despike ✗) | 10,475 | **0.794±0.033** | 0.672±0.023 |
    Δ(✗−✓) 이진 +0.004 [+0.002, +0.009] wins 20/20 · 3-class +0.001 [−0.002, +0.004] wins 13/20.
  - **관찰**: QC 통과 수가 동일 → despike가 고친 점은 QC·성능에 무영향. 20/20 seed에서 미세하게 꺼진 쪽이 높은 것은 실제 신호점을
    이웃 평균으로 덮어쓴 손실로 해석 가능하나 크기(0.004)는 무의미. "SG 없음이면 despike를 켜 둬야 한다"는 2026-09-09 추론은 **철회**.
  - **현 시점 최고 조건(보라매 112, 20시드)**: PS+Si 축보정 · despike ✗ · trim 400–2200 · smoothing ✗ · rolling_min 101 · SNV · LR →
    이진 0.794±0.033 / 3-class 0.672±0.023. SG 7/9/11과는 잡음 안에서 동급. 채택 여부는 사용자 결정.

## Phase OV — 측정자(operator) 변동성 분석 (2026-09-04)

- [x] OV-1: 측정자에 따른 스펙트럼 변동성 통계 분석
  - **결론: 측정자 효과는 주효과로 추정 불가능하다.** aecd_platform 6개 run에서 측정자는
    날짜가 바뀔 때만 바뀐다 (엄찬호 8/10·8/11·8/12, 고은혜 8/13·8/13·8/14). 두 측정자가
    같은 날 측정한 적도, 같은 검체를 재측정한 적도 없다 (samples_in_multiple_runs = 0).
    측정자 = 날짜 = strip unit = 그날의 보정 상태가 완전히 aliasing.
  - 데이터: 13,673 스펙트럼 (113 검체 × 121 point), `measurement.raw_spectra`
  - 측정자 비교는 **BNOR 70검체로만** 수행 (BPRO 43건은 고은혜 단독 측정이라 질환 효과와 혼입).
    엄찬호 50 / 고은혜 20.
  - **run(=날짜) 수준 분산 비율** — 원본 지표에서 유의, SNV 후 소멸:

    | 지표 | run 분산 비율 | F(3,66) | p(ANOVA) | p(Kruskal) |
    |---|---|---|---|---|
    | noise_sd | **39.5%** | 11.77 | 3e-6 | 4e-5 |
    | median_intensity | 11.1% | 3.05 | 0.034 | 0.029 |
    | baseline_level | 10.2% | 2.88 | 0.043 | 0.031 |
    | AUC | 10.1% | 2.86 | 0.043 | 0.033 |
    | mean_point_corr | 4.4% | 1.77 | 0.162 | 0.117 |
    | SNR | 0% | 0.97 | 0.410 | 0.186 |
    | corr_to_global_mean (SNV 후 형태) | **0%** | 0.47 | 0.702 | 0.575 |

  - **고은혜 run(8/13)은 엄찬호 3일치 변동 폭 안에 있다.** 엄찬호 날짜 간 SD 기준 표준화 편차:
    AUC +0.22, median_intensity +0.32, baseline +0.37, noise_sd -0.19, SNR +0.87,
    corr_to_global_mean -0.08, mean_point_corr +0.17 — 전부 |1 SD| 미만.
    한쪽 측정자의 날짜가 1개뿐이라 p-value는 계산하지 않았다.
  - PCA (검체 평균 SNV 스펙트럼, PC1 29.3% / PC2 19.2%): run·측정자별 군집 없음.
  - 환경 변수는 배제됨: 24.3~24.8°C, 55~61% RH, laser 1 mW / 0.05 s 전 run 동일, reagent lot 동일.
  - 보정 데이터에 날짜 효과의 독립 증거: `replicate_shift_std_cm1`이 8/10~8/12 0.0024~0.0060에서
    8/13·8/14 0.0227로 약 5배 상승. 측정자 교대 시점과 일치하지만 이는 장비/일자 상태이지
    사람이 아니다.
  - **실무적 함의**: 원본 강도·노이즈에는 날짜 간 차이가 있으나 표준 전처리
    (SG smoothing → rolling-min baseline → SNV) 후 형태 지표의 run 분산이 0이 된다.
    현재 파이프라인이 이 변동을 흡수하고 있다.
  - **설계 권고**: 측정자 효과를 실제로 재려면 같은 날 두 측정자가 동일 검체를 교차 측정하는
    설계가 필요하다 (예: QC 표준물질을 매일 두 사람이 각각 측정).
  - 스크립트: `scripts/analysis/operator_variability_analysis.py`
  - 산출물: `results/operator_variability/` (spectrum_metrics.csv, sample_metrics.csv,
    variance_components.csv, operator_contrast.csv, pca_by_run_and_operator.png)

- [x] OV-2: 측정자 교차 설계 검정력 계산 (OV-1 후속, 2026-09-04)
  - **목적**: OV-1에서 측정자 효과가 추정 불가능하다고 결론냈으므로, 실제로 재려면
    몇 일이 필요한지 계산.
  - **핵심**: 교차 설계(같은 날 두 측정자가 같은 대상 측정)에서는 날짜 효과가 쌍 안에서
    상쇄되므로, 가장 큰 분산 성분인 **날짜 간 변동이 표본 계산에 들어가지 않는다.**
    필요 표본을 정하는 것은 일내 반복성이다.
  - **PS 표준물질 실측 반복성** (`data/mapping/Thermo Reference`, 2026-08-04~14, 9일 60 스펙트럼):
    | 지표 | 일내 CV | 일간 CV |
    |---|---|---|
    | AUC | 10.67% | 26.19% |
    | peak | 9.14% | 26.76% |
    | noise_sd | 11.18% | 30.93% |
    고정된 물질인데도 일간 CV가 일내의 2~3배 — OV-1의 날짜 효과를 독립적으로 재확인.
  - **설계 A (PS 교차 측정, AUC 기준, α=0.05 power=0.80)** — 필요 일수:
    | 1인당 반복수/일 | Δ=15% | Δ=10% | Δ=5% | Δ=3% |
    |---|---|---|---|---|
    | 3회 | 3일 | 6일 | 24일 | 67일 |
    | 5회 | 2일 | 4일 | 15일 | 40일 |
    | 10회 | 1일 | 2일 | 8일 | 20일 |
  - **설계 B (임상 검체 분주 교차)**: 탐지목표를 관측된 날짜 간 SD로 잡을 때 필요 쌍 수는
    기술적 반복성 SD 가정에 좌우된다 (0.2×검체간SD → 6쌍 / 0.5× → 35쌍 / 1× → 140쌍).
    **이 기술적 반복성은 현재 데이터로 추정 불가** — 같은 검체를 두 번 측정한 적이 없다.
    20쌍/일 기준 1~7일 범위.
  - **권고**: ① 파일럿 1일 — 같은 소변 10~15개를 분주해 두 측정자가 각각 측정, 기술적
    반복성부터 추정. ② 본 시험은 PS 표준물질 교차 측정을 일상 업무에 얹어 진행(검체 소모 0).
    ③ 어느 설계든 **측정자×날짜 상호작용**(측정자 차이가 날마다 일정한가)을 보려면 자유도상
    최소 5~6일이 필요하다. 따라서 실무적 답은 **최소 1주, 권장 2주**.
  - 스크립트: `scripts/analysis/operator_study_design.py`
  - 산출물: `results/operator_variability/design_a_qc_standard.csv`, `design_b_split_aliquot.csv`

- [x] OV-3: 검체당 측정점 수 수렴 분석 (매핑의 원래 목적, 2026-09-04)
  - **동기**: 측정 기간 단축 방법은 둘 — 측정자를 늘리거나(새 변수 발생), 검체당 점 수를
    줄이거나(새 변수 없음). 후자가 어디까지 가능한지 확인.
  - 대상: 보라매 113검체 × 121점, SNV 후. k점 평균과 121점 전체 평균의 상관/RMSE.
    k마다 40회 무작위 추출 (seed 20260904).

    | 점 수 | 상관(중앙값) | 상관(하위5%) | RMSE | RMSE/검체간차이 |
    |---|---|---|---|---|
    | 10 | 0.9943 | 0.9906 | 0.1001 | 19.5% |
    | 20 | 0.9974 | 0.9958 | 0.0685 | 13.4% |
    | 30 | 0.9984 | 0.9976 | 0.0532 | 10.4% |
    | **40** | **0.9990** | **0.9984** | **0.0425** | **8.3%** |
    | 60 | 0.9995 | 0.9992 | 0.0303 | 5.9% |
    | 121 | 1.0 | 1.0 | 0 | 0% |

  - 자: 같은 run 내 두 검체 평균 스펙트럼 간 RMSE 중앙값 = 0.5131 (모델이 쓰는 신호 크기),
    날짜 간 차이(검체 단위 보정) = 0.9524.
  - **결론: 40점이면 121점 평균을 상관 0.999로 재현하고, 축소 오차는 검체 간 차이의 8.3%.
    측정 시간 1/3로 줄이면서 새 변수는 0개.**
  - ⚠️ 한계: 이는 **평균 스펙트럼 재현** 기준이지 **모델 성능** 기준이 아니다. 확정하려면
    k점으로 재학습해 AUC를 비교해야 하는데, 현재 코호트(113명)로는 PL-1에서 확인했듯
    0.03 수준 차이를 판별할 수 없다. 점 수 축소는 재현 기준으로 먼저 정하고, 코호트가
    커진 뒤 성능 기준으로 재확인하는 순서를 권한다.
  - 스크립트: `scripts/analysis/mapping_point_count_convergence.py`
  - 산출물: `results/operator_variability/point_count_convergence.csv`,
    `point_count_convergence_per_sample.csv`, `point_count_convergence.png`

## Phase LC — Subject 수 학습곡선 / 암종별 최소 검체 수 (2026-09-09)

> 질문: "암종별 최소 검체 수는?", "전립선 450명 달성 시 예상 AUC는?"에 데이터로 답하기 위한 사후 분석.
> 재학습이 아닌 서브샘플링 분석. 스크립트 `scripts/analysis/learning_curve_sample_size.py`,
> 산출물 `results/learning_curve_20260909_v1/` (learning_curve_raw.csv / _summary.csv / _fits.json / learning_curve.png / run_metadata.json).

- [x] LC-1: 후향 7암종 코호트 학습곡선 (KAO 2026-06-10, 1,630명, 각 암종 vs 대조군 430명) ✅
  - 설계: subject당 replicate 평균 1행, StandardScaler+LR(C=0.1). 반복 30회: 30% stratified hold-out 고정 → 학습쪽에서 암 n명 + 대조군 n명(1:1) 서브샘플 → hold-out AUC.
  - AUC ≥ 0.95 최초 도달 n(암 subject): BLC 10 / CRC 15 / BRE 15 / LUN 20 / PAN 30 / PRO 50 / OVA 미도달(n=40에서 0.946, 가용 70명이라 50 이상 측정 불가)
  - Inverse power law `AUC(n)=a−b·n^−c` 적합 후 plateau−0.01 도달 n: PAN 50 / CRC 100 / BLC 100 / LUN 160 (PRO·OVA·BRE는 가용 n 안에서 미도달)
  - 곡선(mean AUC): PRO 10:0.829 → 20:0.914 → 30:0.935 → 50:0.954 → 70:0.970; PAN 10:0.853 → 30:0.955 → 70:0.973; OVA 10:0.852 → 40:0.946; BRE 10:0.911 → 20:0.975; LUN 20:0.950 → 100:0.985 → 210:0.993; CRC 15:0.955 → 100:0.994; BLC 10:0.968 → 100:0.993
  - ⚠️ 해석 한계: 암종별 단일 병원(hospital confound) 그대로인 in-distribution 곡선. cancer-vs-control screening 과제만이며 cancer-type ID(F1)는 별도 — STK-V2 held-out에서 BRE(n=30) F1 0.71, PAN(100) 0.65로 type ID는 더 많은 검체 필요.
- [x] LC-2: 전립선 450명 외삽 ✅
  - 후향 CBNUH PRO(100명) 곡선 외삽 → n=450 예측 AUC **0.99** (repeat별 적합 90% 구간 0.95–0.997). 이미 n=70에서 0.970이라 추가 검체의 한계효용 작음.
  - 전향 보라매 PRO 43 / 비암 69 (control 20 + PDC 49, Drop 제외 112명, `mean_representative_spectra.csv`) 곡선: n=8→30에서 0.611→0.627로 **거의 평탄**. 외삽 n=450 예측 AUC **0.65** (90% 구간 0.55–0.81). 적합된 c≈0.03이라 AUC 0.80 도달 필요 n은 1만 명 초과(=검체 수로 해결 불가).
  - 결론: 전립선 450명의 예상 AUC는 데이터 조건에 따라 0.65(현재 보라매 전향 측정 조건) ~ 0.99(후향 동일 분포)로 갈린다. 보라매 곡선의 평탄성은 병목이 표본 수가 아니라 신호 대 노이즈·라벨 정의(PDC vs 암)·측정 조건임을 뜻한다 (Phase AS-1 baseline 0.629, legacy STK-V2 feature 0.75와 정합).
  - 참고: aecd_platform DB에 cohort_group=prostate는 이미 453명(CBNUH 400 + BORAMAE 53)이나 스펙트럼 보유는 143명(CBNUH Thermo 100 + BORAMAE 43)뿐 — "450명 목표"는 측정 목표로 읽어야 함.

## Phase FM — 스펙트럼 파운데이션 모델(DSCF) 인코더 표현 probe (2026-09-15~16)

> 질문: "Raman/SERS 사전학습 파운데이션 모델의 표현을 쓰면 우리 데이터에서 LR을 넘을 수 있나?"
> 대상: Xue B, et al. Nat Mach Intell 2025;7(5):743–57 — Deep Spectral Component Filtering (DSCF).
> 공개 가중치 `SiT_PPS_tiny.pt`(figshare 28130648, md5 aee403eb46487f12d48f0b1d11ec3e8e, 2.5 GB, CC BY 4.0),
> 코드 GitHub streamflowmaster/Deep-Spectral-Component-Filtering-DSCF- commit 501fe4d (GPL-3.0, scratch clone, 소스 미수정).
> 스크립트 `scripts/analysis/dscf_encoder_probe_kao1630.py`, 산출물 `results/dscf_encoder_probe_kao1630_20260915/`
> (runs.csv / summary.json / run_metadata.json / run.log / run_metadata_stop1.json).

- [x] FM-1: DSCF 인코더 표현 → LR probe, 후향 KAO 코호트 (2026-09-16) ✅
  - 데이터: `results/kao_20260610_updated_cohort/processed_spectra.csv` 1,630명(암 1,200 / 대조 430), 8,150행(전원 5 replicate),
    933점 401.89–2198.69 cm⁻¹. subject 평균 1행, subject key = (group, sample_id), CPAN/YPAN→PAN, YNOR→NOR.
  - 조건 (모두 같은 fold·같은 시드): **A** 원 격자 933점 LR / **B** 512점 선형보간 + 샘플별 min-max LR (C의 주 비교 기준) /
    **C** DSCF 사전학습 인코더 stage 출력 길이축 평균 pooling → LR (enc1 256 / enc2 512 / enc3 1024 / enc4 2048차원) /
    **D** 같은 구조 무작위 초기화 인코더(초기화 시드 = CV 시드) → LR — 사전학습 효과 판별용 대조군.
  - 검증: 환자 단위 StratifiedGroupKFold(5), 시드 42/7/123/2024/31337, StandardScaler+LR(L2),
    C는 학습 fold 안 3-fold GridSearch {0.01, 0.1, 1}에서 선택. 지표는 시드별 OOF pooled.
  - **과제 1 암/비암 (AUC)**: A 0.9765±0.0028 · B 0.9737±0.0026 · C_enc1 0.8740 / enc2 0.9168 / enc3 0.9429 / **enc4 0.9535±0.0015** ·
    D_enc1 0.8731 / enc2 0.9296 / enc3 0.9564 / **enc4 0.9670±0.0026**
  - **과제 2 암종 7종 (macro-F1)**: A 0.8650±0.0078 · B 0.8711±0.0077 · C_enc1 0.5126 / enc2 0.5998 / enc3 0.6636 / **enc4 0.6916±0.0058** ·
    D_enc1 0.5014 / enc2 0.6466 / enc3 0.7466 / **enc4 0.7615±0.0145**
  - **짝 차이 (같은 시드, 평균±sd, C가 이긴 시드 수)**
    - C−B: 과제1 enc4 −0.0202±0.0036 (0/5), enc3 −0.0308, enc2 −0.0569, enc1 −0.0996 / 과제2 enc4 −0.1795±0.0047 (0/5), enc3 −0.2075, enc2 −0.2713, enc1 −0.3584
    - C−D: 과제1 enc1 +0.0009±0.0058 (3/5), enc2 −0.0127, enc3 −0.0135, enc4 −0.0135 (0/5) / 과제2 enc1 +0.0112±0.0121 (5/5), enc2 −0.0468, enc3 −0.0830 (0/5), enc4 −0.0699 (0/5)
    - B−A: 과제1 −0.0028±0.0024 (0/5), 과제2 +0.0061±0.0134 (4/5) → 512점 다운샘플 자체의 손실은 작음
  - 관측: 모든 stage에서 C < B. enc2 이상에서는 C < D(무작위 초기화)이고 시드 5개 모두 같은 방향. enc1만 C ≳ D.
  - 가중치 로드: 1차 시도는 공개 finetuning 설정(output_channels=1)으로 만들어 strict 로드 실패(unpatchy.proj 400 vs 4) → 중단 기록 `run_metadata_stop1.json`.
    사용자 승인 후 체크포인트에 저장된 `model_args`(outplanes=100)로 재구성해 strict 로드 성공(missing 0 / unexpected 0 / shape mismatch 0).
  - ⚠️ 한계
    - 체크포인트 기록이 `iter_num=159`, `best_val_loss=1e10` — 사전학습이 끝까지 진행된 가중치인지 불명확.
    - 사전학습 도메인(혈청 SERS, 638 nm, Ag/Au nanostar + 합성 IR/Raman/UV 혼합)과 우리 데이터(소변 SERS, 785 nm)가 다름.
    - 입력이 이미 전처리된 스펙트럼(음수 포함)이라 사전학습 전처리(최댓값 나눗셈)와 다른 min-max를 적용.
    - 인코더 특징 linear probe는 논문이 검증한 경로가 아님(논문은 DSCF를 분류기로 쓰지 않음).
    - 후향 코호트는 암종별 수집처가 달라 수집처 교란 포함(특히 과제 2). BRE 30명이라 macro-F1 변동 큼.
    - 탐색적 분석이며 외부 시험셋 없음.

## Phase DN — Denoising autoencoder 잡음 제거 후 분류 (2026-09-17)

> 질문: "반복 측정 짝으로 학습한 denoising AE로 잡음을 걸러내면 LR 분류가 오르는가?" (사용자 요청, Han 2024 CDAE의 짝 학습을 우리 실제 반복 측정 짝으로 대체)
> 스크립트 `scripts/analysis/denoising_ae_probe_kao1630.py`, 산출물 `results/denoising_ae_probe_kao1630_20260917/`
> (runs.csv / summary.json / run_metadata.json / run.log / ae_diagnostics.csv / ae_fit_info.csv / ae_training_curves.csv / residual_by_wavenumber.csv).

- [x] DN-1: 반복 측정 짝 denoising AE → subject 평균 → LR, 후향 KAO 코호트 (2026-09-17) ✅
  - 데이터: FM-1과 동일 (`results/kao_20260610_updated_cohort/processed_spectra.csv`, 1,630명 암 1,200 / 대조 430, 8,150행, 933점).
  - AE: 1D conv denoising AE, 채널 [16, 32, 32, 32], kernel 7, stride 2, 잠재 48, 파라미터 219,441. 입력 = replicate 1개, 목표 = 같은 subject의 **나머지 4개 replicate 평균(leave-one-out)**.
    MSE, Adam lr 1e-3, batch 128, 최대 100 epoch, 학습 fold subject 10%를 검증으로 떼어 patience 10 early stopping(평균 45.7 epoch, best 35.7, fold당 4.7 s, GPU 144 MB).
    스케일: 학습 fold replicate 전체의 평균·SD 스칼라 표준화, 출력은 원 스케일로 복원. 구조·하이퍼파라미터 고정(미튜닝).
  - **누수 방지**: AE는 각 (시드, fold)의 학습 fold subject replicate로만 학습(25회). 테스트 fold subject는 AE 학습에 미포함.
  - 조건 (같은 fold·같은 시드 42/7/123/2024/31337, 환자 단위 StratifiedGroupKFold 5, StandardScaler+LR, C∈{0.01,0.1,1} 학습 fold 안 3-fold 선택):
    **A** 원 replicate subject 평균 → LR / **B** replicate별 AE 복원 → subject 평균 → LR / **C** A의 subject 평균을 AE 1회 통과 → LR
  - **잡음 추정 진단 (테스트 fold subject, 시드 평균 ± SD)**: within-subject RMSE(replicate vs LOO 평균) 0.2292 → 0.1837, **−19.9% ± 0.1%** (replicate의 85.8%에서 감소).
    between-subject 분산(subject 평균 스펙트럼의 파장별 분산 평균) 유지율 **B 0.879 ± 0.009, C 0.895 ± 0.009** — 검체 간 차이의 약 11~12%도 함께 제거됨. subject 평균의 평균 절대 이동 0.099.
  - **과제 1 암/비암 (AUC)**: A 0.9781 ± 0.0019 · B 0.9594 ± 0.0023 · C 0.9583 ± 0.0020 → B−A **−0.0187 ± 0.0041 (0/5)**, C−A −0.0198 ± 0.0037 (0/5)
  - **과제 2 암종 7종 (macro-F1)**: A 0.8737 ± 0.0087 · B 0.6824 ± 0.0126 · C 0.6757 ± 0.0117 → B−A **−0.1912 ± 0.0103 (0/5)**, C−A −0.1980 ± 0.0104 (0/5)
  - 관측: 잡음(반복 편차)은 20% 줄었으나 두 과제 모두 시드 5개 전부에서 AE 복원 조건이 원 스펙트럼보다 낮음. 암종 구분 하락 폭(−0.19)이 암/비암(−0.02)보다 훨씬 큼. B와 C의 차이는 작음(평균 절대 차 0.013).
  - 참고: 기준 A(0.9781 / 0.8737)가 FM-1의 A(0.9765 / 0.8650)와 소폭 다름 — 같은 데이터·시드·규약이나 구현이 별도 스크립트라 fold 구성 또는 LR 세부가 다를 수 있음(미확인). 각 실험 안에서의 짝 비교에는 영향 없음.
  - ⚠️ 한계: 후향 코호트 수집처 교란(특히 과제 2) · 이미 전처리된 입력 · AE 구조 1개 고정(미튜닝, 잠재 차원·손실·목표 정의에 따라 달라질 수 있음) · 5회 반복 평균 자체가 이미 잡음을 상당히 제거하는 조건이라 AE의 추가 여지가 작았을 가능성 · 외부 시험셋 없음.
  - **DN-1 보충 (2026-09-17)**: 기준 A가 FM-1의 A와 다른 원인 확인 — DN-1은 두 과제가 같은 fold를 쓰도록 **11개 model_group으로 층화**(FM-1은 과제별 label로 층화). 같은 시드라도 fold 구성이 다르므로 실험 간 A 직접 비교는 하지 말 것. 각 실험 안의 짝 비교(B−A 등)에는 영향 없음. 파장별 잔차 RMS 비(후/전) 중앙값 0.756, 933점 모두 <1, 1900–2200 cm⁻¹ 대역에서 가장 큰 감소(0.692).

## 데이터 정리 기록 — data/ 폴더 이름 통일·임상 테이블 분리 (2026-09-15 ~ 09-18)

> 실험이 아니라 데이터 위치 변경 기록. **이 문서의 이전 기록에 나오는 data/ 경로는 모두 옛 경로다.**
> 옛 경로 → 새 경로 대응표: `data/99_manifests/path_rename_map.csv`, 폴더별 내용: `data/00_README/DATASET_INVENTORY.md`.
> 계획·실행 기록: `/home/user/workspace/_scratch/2026-09-15-data-folder-rename/README.md` (되돌리기 파일은 9/24~10/1 삭제 예정).

- **폴더 이름 통일 (2026-09-17 실행, commit 07d634e)**: `data/` 30개 항목을 sers_transfer 연구설계 분류(00~06, 99) 아래
  `{장비}_{측정종류}_{군}_{기간|undated}`로 이동. 기간은 DB `measurement.runs.measurement_date`로 확정된 경우만 표기.
  주요 대응: `raw_data`→`02_sers_primary_pooled_acquisition/thermo_retro_12groups_undated`,
  `raw_data_medical`→`02_.../ramcheck_retro_12groups_undated`, `임상데이터`→`03_sers_date_lot_balanced_acquisition/thermo_retest_12groups_20260416-20260519`,
  `mapping`→`03_.../thermo_mapping_BNOR-BPRO_20260810-20260814`, `Thermo`·`Thermo 1`·`Thermo 260911~260915`→`03_.../thermo_mapping_multi_*`.
  - **BLC 두 폴더 구분**: 같은 충북대 299명(#241 결측)을 두 번 측정한 것으로 확인. `BLC_1st_20260319-20260320`(옛 `11 BLC (299개)`, 3~4월 초 모델이 사용),
    `BLC_2nd_20260407-20260409`(옛 `20260407_Bladder_…`, 현재 config `folder_to_group` BLC). 파일명은 같고 내용은 전부 다름.
  - **후향 본세트 로드 범위 변화 1건**: 보라매 0709 액체 세트가 후향 본세트 밖으로 분리되어, 기본 로더의 thermo 후향 로드가 8,605→8,500 스펙트럼(BNOR 105, 21명 제외). 다른 군 수는 동일.
  - DB: `measurement.raw_spectra.source_uri` 106,809행·`measurement.runs.notes` 82행을 새 경로로 UPDATE(사용자 1회 예외 승인, 트랜잭션+검증, 백업 `pg_dump` 보관). 검증: 파일 누락 0, 옛 경로 잔존 0.
  - 중복 삭제(해시 검증): `Thermo (2)`(1.5GB), 풀려 있던 zip 6개. 기록 `data/99_manifests/deleted_duplicates_hash_check_20260915.json`.
- **임상 테이블 분리 (2026-09-17, commit 2911987)**: `data/clinical_data`(→`01_clinical_metadata/hospital_clinical_tables`) 202MB가 같은 1,782명의 파생본 8종+복사된 Python 환경이라
  역할별로 분리: `00_master_normalized_workbook`(기준 표 = `전체환자_임상정보_정규화_v8.xlsx` 2,899행 + 과거 버전 19개),
  `01_raw_hospital`, `02_standardized_1782`, `03_exclusion_and_clean`, `04_stage_lifestyle_overrides`(clinical_unified — 병기·흡연·음주 정답, 5개 암종 840명),
  `05_dictionaries_and_governance`, `06_sers_linkage`, `07_scripts_and_sql`, `_archive`. 20MB로 축소(.venv·PowerBI 파생본 삭제).
  `sample_exclusions/phase_x_exclusions.csv`(99건)는 사용자 실수 삭제 후 2026-09-18 재생성, `phase_x_plus_hs_exclusions.csv`는 생성 근거가 없어 복구 불가.
- **9/16~9/17 Thermo mapping 배치 적재 확인 (2026-09-18)**: run 192(9/16 SK20260806B01)·193(9/17 SK20260806C01)·194(9/17 SK20260810A01),
  임상 5,688 + control 60 = 5,748 스펙트럼, 158검체×36점, Si 교정 양일 pass. `measurement.raw_spectra` 합계 112,557.
  로더 `scripts/db/aecd_mapping_batch_20260916_0917_load.py`(commit 1c74eec, 이전 배치와 같은 규칙, 재적재 방지 가드 포함).

- [x] DN-2: 같은 denoising AE를 **조건 변경 후 전립선 112명(121점 mapping)**에 적용 (2026-09-18) ✅
  - 데이터: DB `measurement.raw_spectra` 변경 후 팔(frozen 13,552 id, 7월 측정분 제외 로더 = AS-25e), ver2 전처리(WH despike → trim 600–1800 → SG(5,3) → rolling-min(101) → SNV, 625점), 2단계 QC 후 **10,517점**(AS-13 보충 1의 "개선 8월 10,517"과 일치). 112명 = 암 43 / PDC 49 / 대조 20, subject당 QC 후 측정점 중앙값 96(51~117).
  - 학습 짝: 측정점 1개 → 같은 subject 나머지 측정점의 LOO 평균. AE 구조·HP DN-1과 동일(encoder 길이만 625 적응, 파라미터 160,465). (시드, fold)별 학습 fold subject로만 25회 학습(평균 22.8 epoch, fold당 12 s).
  - 조건·규약 DN-1과 동일(A 원 측정점 subject 평균 LR / B 측정점별 AE 복원 → 평균 / C 평균 → AE 1회). fold는 환자 단위 StratifiedGroupKFold(5), 3군 라벨 층화, 시드 42/7/123/2024/31337.
  - **진단 (테스트 fold, 시드 평균 ± SD)**: within-subject RMSE 0.2646 → 0.1848 (**−30.2% ± 0.9%**, 25/25 fold 감소, 측정점의 90%에서 감소). between-subject 분산 유지율 **B 0.641 ± 0.005 / C 0.670 ± 0.005** (DN-1 후향은 0.879 / 0.895). 파장별 잔차 RMS 후/전 비 중앙값 0.675, 비>1인 점 6/625(최대 1.68 @1031 cm⁻¹).
  - **과제 1 암 vs 비암 (AUC)**: A 0.772 ± 0.035 · B 0.616 ± 0.031 · C 0.597 ± 0.025 → B−A **−0.156 ± 0.032 (0/5)**, C−A −0.175 ± 0.027 (0/5). bal.acc A 0.695 / B 0.573 / C 0.548.
  - **과제 2 3군 (macro-F1)**: A 0.431 ± 0.026 · B 0.382 ± 0.026 · C 0.370 ± 0.015 → B−A **−0.049 ± 0.024 (0/5)**, C−A −0.061 ± 0.037 (0/5).
  - 관측: 반복 짝이 많아(5 → 약 94) 잡음 감소율은 커졌으나(20% → 30%) 검체 간 분산도 더 많이 제거됐고(유지율 0.88 → 0.64), 분류 하락 폭도 커짐(암/비암 −0.019 → −0.156). 시드 5개 모두 하락.
  - **A 정합성 (AS-25e ver2/deck 5시드 0.815, 20시드 0.804 ± 0.024)**: 본 A 0.772 ± 0.035. 시드 42(0.805)·123(0.771)은 AS-25e와 0.001 이내 일치, 시드 7(0.721)·2024·31337은 낮음. 같은 행렬로 분해: LR만 AS-25e 규약(스펙트럼 단위 C=1 balanced, 환자 = 점 확률 평균)으로 바꾸면 **0.810**, 층화만 2군으로 바꾸면 0.786, 둘 다 바꾸면 0.832 → 차이의 주원인은 **LR 규약(subject 평균 입력 vs 스펙트럼 단위 학습)**이고 fold 배정이 나머지. DN-1·DN-2의 A는 같은 규약이라 두 실험 간 비교 가능, AS-25e 값과는 직접 비교 금지.
  - 환경 메모: `sers-analysis`에 pydantic이 없어 로더(`sers.aecd_api`)는 base python 서브프로세스로 실행해 `data_cache.npz`에 저장, AE/LR은 `sers-analysis`. 로더 모듈의 JULY_DIR(9/17 폴더 개명으로 부재)은 런타임에 `data/03_sers_date_lot_balanced_acquisition/thermo_boramae_liquid_BNOR-BPRO_20260709-20260710`로 덮어씀(모듈 미수정). 7월 파일에만 있고 DB 미포함 8명(43, 58, 65, 96, 104, 110, 111, 113).
  - 스크립트: `/home/user/SERS-AI-merge/scripts/analysis/denoising_ae_probe_boramae112.py` (로더가 feat/boramae-api-contract worktree에만 있어 그곳에 둠, **미커밋**). 산출물 `results/denoising_ae_probe_boramae112_20260918/` (runs.csv, summary.json[A_consistency_vs_AS25e 포함], run_metadata.json, ae_diagnostics.csv, ae_fit_info.csv, ae_training_curves.csv, residual_by_wavenumber.csv, fold_assignment.csv, data_cache.npz).
  - ⚠️ 한계: 112명(fold 배정만으로 시드 SD 0.035, fold별 A AUC 0.56~0.95) · 단일 수집처·단일 측정 기간 · 전처리·QC 후 입력 · AE 구조 미튜닝 · 과제 2 대조 20명(fold당 4명) · 외부 시험셋 없음.

## Phase BE — 변경 후 측정 조건(측정일·스트립 lot·측정자) 배치 효과 분석 (2026-09-18)

> 질문: "시약 lot을 통일한 뒤 남은 측정 메타데이터가 스펙트럼 변동과 분류 성능에 얼마나 기여하나?" (사용자 요청)
> 스크립트 `/home/user/SERS-AI-ci-tiered/scripts/analysis/batch_effect_postchange_926.py` (로더가 있는 worktree, 미커밋), 산출물 `results/batch_effect_postchange_926_20260918/`
> (cohort.csv / design_table.csv / variance_decomposition.csv / variance_decomposition_pc.csv / day_eta2_by_wavenumber.csv / predictability.csv / effect_size_day_vs_group.csv / classification_by_split.csv / summary.json / run_metadata.json / run.log).

- [x] BE-1: 변경 후 임상 측정 926검체 배치 효과 (2026-09-18) ✅
  - 데이터: DB `measurement.measurements`(role=clinical) ⋈ `runs`, measurement_date ≥ 2026-08-19. 926검체(암 612 / 비암 314; NOR 191·PAN 178·PRO 149·LUN 89·CRC 85·BLC 56·BRE 55·HBP 43·DIA 41·H.D. 39), 19 측정일, **reagent_lot 1개(검정 불가)**, strip_lot 7종(날짜에 nested), operator 2명(8/27 교체, 날짜에 nested), 기기 1대. 121점 측정분은 36점 무작위 추출(seed 고정) → raw 33,336 → stage-1 33,333 → corr QC 27,025점, 검체 평균 1행. ver2 전처리(despike → 600–1800 → SG(5,3) → rolling-min(101) → SNV, 625점).
  - 설계: 매일 여러 군 혼합 측정(8월: NOR·BRE·PRO·PAN·대조군, 9/8부터 LUN·CRC·BLC 추가) → 군과 날짜 완전 교락 아님. 같은 검체가 두 날짜에 측정된 경우 0.
  - **분산 분해 (일원 η², 파장별 평균)**: 군 0.118 · 측정일 0.071 · strip lot 0.054 · 암/비암 0.024 · 측정자 0.017. 순차 OLS: 군 R² 0.118 → 군+측정일 0.181 (**측정일 증분 0.064**), 측정일 R² 0.071 → +군 증분 0.111. PCA 90%(14 PC) 가중도 같은 순서(군 0.158 / 측정일 0.073).
  - **예측 가능성 (검체 StratifiedKFold 5, StandardScaler+LR, C∈{0.01,0.1,1} 내부 3-fold, 시드 5개, 실제 vs 라벨 permuted)**: 측정자 AUC 0.915 ± 0.008 (perm 0.498) · strip lot macro-OvR AUC 0.878 ± 0.005 (0.501) · 측정일(검체 ≥20인 12일, 809검체) 0.858 ± 0.003 (0.501) · 암/비암 AUC 0.838 ± 0.005 (0.496) · 군 macro-F1 0.350 ± 0.007 (0.102).
  - **효과 크기 (검체 평균 스펙트럼 RMSE)**: 같은 군·날짜 간 0.224(중앙값 0.197, 549쌍) · 같은 날·군 간 0.265(0.217, 274쌍) · 같은 군·같은 날 반분 0.191(0.169, 53).
  - **분할 방식별 암/비암 AUC (같은 시드 5개, 검체 OOF)**: 무작위 0.838 ± 0.005 · **측정일 GroupKFold 0.832 ± 0.004 (−0.006)** · strip lot GroupKFold 0.830 ± 0.005 (−0.008) · 무작위+날짜 보정(학습 fold 날짜 평균 중심화 + one-hot) 0.840 ± 0.004 (+0.002).
  - 관측: 측정 요인은 스펙트럼에서 식별될 만큼 흔적을 남기지만(AUC 0.86~0.92) 설명 분산은 군의 절반 수준이고, 날짜·lot을 학습에서 제외해도 암/비암 AUC 변화가 시드 SD 이내.
  - ⚠️ 한계: strip lot·측정자는 측정일에 nested라 날짜 효과와 분리 불가 · 날짜 효과에 군 구성 변화(9/8 이후) 포함 · 군 η²에 수집처 교란(군 = 수집처 1:1) 포함 · η²는 일원(미조정), 순차 R²만 조정 · 날짜/lot 단위 fold는 테스트 군 구성이 무작위 fold와 다름 · 시약 lot 단일 수준(검정 안 됨) · 이 926검체는 후향 검체 재측정이라 AUC 0.84는 전향 성능 아님.
  - 덱: `/home/user/workspace/_scratch/2026-09-18-batch-effect-deck/2026-09-18_측정조건_변동_분석_대표님보고.pptx` (3장, 대표님 보고).

- [x] BE-2: 표준물질 보정·정규화 방식이 측정일 효과와 분류에 미치는 영향 (2026-09-18) ✅
  - 스크립트 `/home/user/SERS-AI-ci-tiered/scripts/analysis/standard_material_norm_postchange.py`(미커밋), 산출물 `results/standard_material_norm_postchange_20260918/` (condition_metric_table.csv, variance/predictability/classification_by_condition.csv, si_shift_by_run.csv, control_by_strip_unit.csv, control_by_date.csv, sample_strip_unit.csv, summary.json, run_metadata.json).
  - 설계: BE-1과 같은 926검체·같은 QC 통과 점(27,025)·같은 시드. 전처리 앞단(despike → 600–1800 → SG → rolling-min) 공통, **마지막 정규화만 교체**. N0 SNV(BE-1 재현: η² 측정일 0.0709, 무작위 AUC 0.838±0.005) / N1 Si 축 보정+SNV / N2 없음 / N3 같은 strip unit의 MB_40uM 1621 cm⁻¹ 피크로 나눔 / N4 같은 strip unit의 모사 소변(N4p 1000 cm⁻¹ 피크, N4a 면적) / N5 PQN. urea 기준 축 보정은 사용 안 함.
  - **control 측정 실태**: control은 날짜가 아니라 **strip unit(플레이트) 단위**로 측정됨. MB_40uM 8/19~8/26(8 unit, unit당 121점), **8/27~9/7 control 없음(디스크에도 없음)**, 모사 소변 9/8~9/17(48 unit, unit당 5점). DB에 measurement→strip_unit 키가 없어 `raw_spectra.source_uri` 폴더명으로 unit 매칭(926검체 전부, 62 unit). `Thermo Reference/MB&SU/2026080x cali`(lot 제조일 기준 측정)는 미사용.
  - **Si shift**: 23 run 전부 기록, observed−520.7 = 0.31~0.59 cm⁻¹(중앙 0.43), 격자 간격 1.92 cm⁻¹ → N1은 N0와 사실상 동일(η² 측정일 0.0710, AUC 0.838).
  - **control 세기 재현성**: MB 1621 피크 unit 간 CV **0.66**(최대/최소 14.8배), 날짜 간 0.53, unit 내부 0.15. 모사 소변 1000 피크 unit 간 CV 0.19, 날짜 간 0.13, unit 내부 0.09.
  - **전체 926**: η² 측정일 N0 0.071 / N2 0.115 / N5 0.077; η² 군 0.118 / 0.046 / 0.106; 암·비암 AUC 무작위 0.838 / 0.821 / 0.837, 측정일 단위 0.832 / 0.812 / 0.828.
  - **MB unit 부분집합(137검체, 8 unit, lot 1개)**: η² 측정일 N0 0.063 / N2 0.195 / **N3 0.475** / N5 0.067; AUC 무작위 0.887 / 0.873 / 0.862 / 0.899; 측정일 단위 0.902 / 0.868 / 0.845±0.058 / 0.901.
  - **모사 소변 unit 부분집합(690검체, 48 unit)**: η² 측정일 N0 0.0386 / N2 0.0505 / **N4p 0.0257** / N4a 0.0341 / N5 0.0412; η² 군 0.124 / 0.054 / 0.048 / 0.053 / 0.111; AUC 무작위 0.798 / 0.783 / 0.777 / 0.777 / 0.796.
  - 관측: (i) Si 축 보정은 효과 없음(shift < 격자). (ii) 정규화를 빼면 측정일 효과가 커짐 — 날짜 변동의 상당 부분이 세기. (iii) MB로 나누면 측정일 효과가 7배로 커짐 — MB 자체의 unit 간 세기 변동(CV 0.66)이 검체에 주입됨. (iv) 모사 소변으로 나누면 측정일 η²는 1/3 줄지만 군·암/비암 η²와 AUC도 함께 낮아짐. (v) PQN ≈ SNV.
  - ⚠️ 한계: control은 unit당 1회 측정이라 unit 고유 차이와 control 재현성 분리 불가 · N3/N4는 기간·군·lot이 달라 서로 비교 불가 · N2~N4는 절대 세기 유지 조건 · `paired_analysis_data.build_arm`이 같은 날 15:48경 stage-2b(최소 25점)로 변경돼 BE-1 당시 QC 규칙을 스크립트 안에 고정해 재현(현재 BE-1 스크립트 재실행 불가).

- [x] BE-3: 같은 라벨·다른 병원의 스펙트럼 차이 (2026-09-18) ✅
  - 스크립트 `/home/user/SERS-AI-ci-tiered/scripts/analysis/same_label_hospital_effect_postchange.py`(미커밋), 산출물 `results/same_label_hospital_effect_postchange_20260918/` (cohort_site.csv, confound_*.csv, shared_dates_by_site_pair.csv, distance_between_sites.csv, distance_within_site.csv, site_predictability.csv, site_eta2_summary.csv, site_eta2_by_wavenumber.csv).
  - 대상(DB site): LUN 2개 site(30/59), NOR 3개 site(100/77/14), PAN 4개 site(66/55/30/27). 나머지 7군은 site 1개.
  - **교락**: LUN 두 site와 NOR site2–site3는 같은 날 측정 0일 → 날짜·lot과 분리 불가. NOR site3–site1 11일 겹침, PAN 쌍 1~5일 겹침.
  - **site 예측 AUC (N0, 5시드; 전체 / 공유 날짜만 / 측정일 단위 분할)**: LUN 0.971 / — / 0.971 · NOR site1–site2 0.931 / 0.862(n=14) / 0.891 · NOR site3–site1 0.680 / 0.699(n=70, CI 0.49–0.87) / 0.680 · PAN site2–site4 0.875 / 0.824(n=81) / 0.831 · PAN site4–site1' 0.925 / 0.913 / **0.599** · PAN site2–site3 0.702 / **0.460**(n=61) / 0.704. permuted 0.42~0.60.
  - **거리(RMSE)**: LUN site 간 0.574(같은 site 날짜 간 0.17~0.19) — 단 공유일 0. NOR site3–site1 같은 날 0.184(반분 0.24/0.06). PAN 쌍 같은 날 0.10~0.24(반분 0.11~0.34).
  - **파장별 site η²**: LUN 0.253(대조 0.012), NOR 0.042(0.011), PAN 0.047(0.017). 군 η² 곡선과 상관: LUN 0.63, NOR 0.09, PAN −0.19.
  - 관측: 날짜를 통제해도 site가 식별되는 쌍(NOR site1–site2, PAN site2–site4)과 식별되지 않는 쌍(PAN site2–site3 0.46, NOR site3–site1 CI가 0.5 포함)이 공존. LUN은 차이가 가장 크지만 날짜와 완전 교락.
  - ⚠️ 한계: site가 원래 라벨(YNOR·YPAN·KPAN)에 묶여 모집·보관 차이와 분리 불가 · 소수 site n=14(fold당 양성 2~3) · 공유일 1~3일인 쌍은 CI 무의미 · 비교한 군 η² 곡선 자체가 site와 교락.
  - **BE-2 보충 (2026-09-18): 기준 물질이 플레이트 상태를 반영하는가** — 사용자 질문("QC를 인공소변으로 하면 어떤가"). 플레이트(strip unit)별 control 기준 피크 중앙값 vs 같은 플레이트 임상 검체의 정규화 전 신호(N2 양수 면적, 검체별 점 중앙값 → 플레이트 중앙값). **모사 소변: Spearman +0.65 (p=6e-7, 48 플레이트, 690검체), log-Pearson +0.70** / 메틸렌블루: +0.29 (p=0.49, 8 플레이트, 137검체). 같은 플레이트들에서 검체 신호 CV 0.15(SU 기간)·0.18(MB 기간)인데 control CV는 SU 0.19 / MB 0.66. 산출물 `results/standard_material_norm_postchange_20260918/control_vs_unit_sample_signal.csv`(덱 그림 스크립트 `workspace/_scratch/2026-09-18-batch-effect-deck/make_figs.py`가 생성). 한계: control은 플레이트당 1회(SU 5점), 두 물질의 측정 기간·측정자·lot·검체 구성이 달라 물질 간 직접 비교는 교락.
