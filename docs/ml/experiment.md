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
- [x] Phase AS-11: 보라매 publication 파이프라인을 AECD API(mapping, 환원제 변경 후)로 재실행 ✅ (2026-09-09)
  - **성격: 조건표(dataset_conditions.md) 8번 세트(mapping, Sigma-Aldrich 226904 Lot BCCP0922)에 대한 결과 1건.** 7월 액상 세트(5번, lot 미확인)로 만든 기존 논문 산출물을 대체하는 것이 아니라 환원제 변경 전·후 비교표의 한 행이다.
  - 데이터: BORAMAE 112명 (Drop 1 제외) = 정상 20 / 비암(prostate disease control) 49 / 암 43. subject 스펙트럼 = 121 mapping 점 전체 평균. 로더 `publications/전향검체/보라매병원/src/boramae_data.py` (PR #14에서 실제 API 계약으로 정합, `grade_group` API 노출)
  - 파이프라인: 논문 원본과 동일 — trim(model grid) → SG(11,3) → rolling-min baseline(101) → SNV → 935-pt grid → StandardScaler+LR(class_weight=balanced), StratifiedKFold 5 outer × 4 inner GridSearch (`prostate_comparison_model.py`). ⚠️ 이 파이프라인은 **StratifiedKFold**이며 Phase AS-1~5의 subject-level GroupKFold와 다르다(단, 입력이 subject 평균 1행/명이라 누수는 없음)
  - 결과 (bootstrap 95% CI):

    | 과제 | 7월 액상 세트 (기존 커밋, n=109) | **8월 mapping (n=112)** |
    |---|---|---|
    | Screening (정상+비암 vs 암) ROC-AUC | 0.714 [0.611–0.814] | **0.789 [0.703–0.866]** |
    | Screening balanced acc / sens / spec | 0.704 / 0.659 / 0.750 | 0.681 / 0.651 / 0.710 |
    | 3-group macro OvR ROC-AUC | 0.811 [0.748–0.874] | **0.670 [0.592–0.743]** |
    | 3-group balanced acc / macro-F1 | 0.668 / 0.674 | 0.511 / 0.508 |

    3-group confusion (행=true): Control 7/9/4, Biopsy-neg 14/26/9, Cancer 5/10/28. Screening: Non-cancer 49/20, Cancer 15/28.
  - 같은 mapping 세트 다른 파이프라인과의 자리: AS-1 baseline 0.629 < AS-5 raw subject-mean 0.7145 < legacy STK-V2 0.7506 < **AS-11 0.789**. 차이는 전처리(SG+rolling-min+SNV)와 CV 설계 차이를 포함하므로 동일 조건 비교가 아니다.
  - ⚠️ 7월 vs 8월 비교의 해석 한계: (1) 7월 세트는 **파일 라벨(BPRO/BNOR)이 병리 확정 전 번호**라 기존 0.714/0.811 자체의 정답표가 DB v7과 다르다(`project_prospective_label_mismatch`); (2) 7월 센서 lot 미확인; (3) replicate 5 vs mapping 121 차이. **환원제 효과로 귀속하려면 5번↔8번 동일 환자를 DB 라벨로 재짝지어 비교해야 한다.** 2026-09-09 `solum_label` 번호 대조 결과 mapping 113명 **전원**이 7월 액상 세트에 있음(라벨까지 같은 64명 + 7월 BPRO→DB BNOR로 바뀐 49명; 7월에만 있는 7명은 DB 미등록). 즉 짝 비교 가능 n=112(Drop 제외) — AS-12에서 수행.
  - `generate_final_publication_outputs.py`는 산출물 생성 후 `validate_report_snapshot()`의 `ReportDriftError`로 정지 (손으로 쓴 PROSTATE_COMPARISON.md가 새 값을 포함하지 않음 — 설계된 가드). 재생성된 figures/tables는 **커밋하지 않음** (조건표 완성 후 결정)
  - 산출물(미커밋, worktree `/home/user/SERS-AI-ci-tiered/publications/전향검체/보라매병원/{figures,tables}`): fig01/02/03/04a/04b/05/06, `prostate_classification_metrics.csv`, `*_oof_predictions.csv`, `*_confusion_matrix.csv`
  - ⚠️ **정정 (AS-12에서 확인)**: 위 0.789는 단일 fold 배정 값이다. `nested_oof()`가 fold seed를 고정해 subject 순서가 fold를 결정하는데, 같은 데이터·같은 파이프라인에서 순서만 바꿔 10회 반복하면 screening AUC **0.727 ± 0.035 (범위 0.676–0.770)**, 3군 macro AUC **0.604 ± 0.032**이다. 0.789는 그 분포의 최대치 0.770보다도 높아 10회 어느 반복에서도 재현되지 않았다(범위 밖) — 대표값으로 인용하지 말 것.
- [x] Phase AS-12: 환원제 변경 전(7월 액상) vs 후(8월 mapping) — **동일 환자 112명 짝 비교**, fold 반복 10회 ✅ (2026-09-09)
  - 설계: 7월 07-09 액상(5회 점 측정, Ave100, 로컬 CSV, lot 미기록) ↔ 8월 mapping(121점, aecd_platform, Sigma-Aldrich 226904 Lot BCCP0922). `solum_label` 번호로 짝지음 — mapping 113명 전원이 7월 세트에 있음(라벨 동일 64 + 7월 BPRO→DB BNOR 49), Drop 1 제외 → **112명**. 7월에만 있는 7명(43·58·65·96·104·111·113)은 DB 미등록으로 제외. **라벨은 세 조건 모두 DB clinical v7 cohort_group** (정상 20 / 비암 49 / 암 43). 반복 수 통제용 3번째 조건: mapping 121점 중 무작위 5점(seed 20260909).
  - 파이프라인 AS-11과 동일 (trim → SG(11,3) → rolling-min(101) → SNV → 환자 평균 → StandardScaler+LR balanced, nested 5×4 StratifiedKFold). 단 subject 순서를 10회 무작위 치환해 fold 배정을 바꿈; 같은 반복 안에서는 세 조건이 같은 치환을 써 짝 검정(동일 환자 DeLong, `powder_comparison/delong.py`)이 성립.
  - **결과 (평균 ± SD, 10회; 범위)**

    | 조건 | Screening AUC | 3군 macro OvR AUC | Screening BAcc | 3군 BAcc |
    |---|---|---|---|---|
    | 7월 액상 (변경 전) | **0.737 ± 0.022** (0.703–0.771) | **0.806 ± 0.023** (0.776–0.844) | 0.694 | 0.658 |
    | 8월 mapping 121점 (변경 후) | **0.727 ± 0.035** (0.676–0.770) | **0.604 ± 0.032** (0.561–0.666) | 0.653 | 0.413 |
    | 8월 mapping 5점 추출 | 0.601 ± 0.041 | 0.550 ± 0.026 | 0.571 | 0.366 |

    짝 검정 (Screening, 동일 환자 DeLong): 7월→8월(121점) ΔAUC **−0.011 ± 0.027** (범위 −0.048~+0.027), p<0.05 **0/10회**, p 중앙값 0.68 → **차이 없음**. 7월→8월(5점) −0.137, 6/10 유의. 8월 121점→5점 −0.126, 7/10 유의.
  - **3군 하락은 fold 노이즈 밖**: 7월 최저 0.776 > 8월 최고 0.666, 10/10 반복 모두 하락. 혼동행렬(repeat 0, 행=실제): 7월 Control 16/2/2, Biopsy-neg 6/31/12, Cancer 2/18/23 → 8월 Control **7**/10/3, Biopsy-neg **16**/24/9, Cancer **8**/8/27. 즉 **Control 군 구분이 사라짐**(Control 정답 16→7명, 비암·암이 Control로 오인 8→24명). 암→암은 23→27로 유지 → Screening AUC 불변과 정합.
  - 환자 수준: 같은 환자의 OOF P(암)은 두 조건 사이에서 거의 상관 없음 — 0.5 기준 판정 뒤집힘 **45/112명**, ΔP 중앙값 −0.006 (Wilcoxon p=0.87). 전처리 후 스펙트럼의 환자별 피어슨 상관 중앙값 **0.55** (5–95% 0.42–0.73 — 처음 적은 0.31–0.78은 오기, summary.json 기준 정정; 군별 Control 0.56 / 비암 0.60 / 암 0.51).
  - **반복 수 통제 결과**: mapping 단일 점 5개 평균은 121점 평균보다 뚜렷이 나쁨(−0.126, 7/10 유의). 7월 "1회"는 Ave100 누적이라 mapping 1점과 품질이 다르다 — 7월 5회 ≈ 8월 121점 수준. 조건 간 비교는 121점 평균 기준이 맞다.
  - **해석 한계 (환원제 단독 효과로 못 봄)**: 7월 센서/환원제 lot 미기록, 측정일 1개월 차, 점 측정 vs mapping 방식 차, 7명 제외. 결론은 "Screening 불변 / Control 구분 하락"까지이고 원인 귀속은 없음. 다음: 7월 lot 기록 복원, 같은 날 같은 분주를 구·신 환원제로 측정하는 한-변수 실험(07-15 Sigma 1~5 설계 형태).
  - 스크립트: `scripts/analysis/boramae_paired_reducing_agent_comparison.py`(분석, ~16분) → `_figures.py` → `_deck.py`(대표님 보고 덱, 수치는 summary.json에서 자동 생성). 산출물 `results/boramae_paired_reducing_agent/` (summary.json, repeat_summary.csv, paired_summary.csv, paired_tests.csv, subject_oof.csv, confusion_*.csv, fig_*.png), 덱 `publications/전향검체/보라매병원/slides/환원제 변경전후 동일환자 비교.html`
  - 조건표 `workspace/_scratch/2026-09-09-performance-recovery-assessment/dataset_conditions.md` 8번 행·C항 갱신 대상

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
