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
- [x] Phase AS-1: Baseline 노트북 (screening/3-class/legacy STK-V2 비교 + repeat-count sweep) ✅ (2026-08-25)
  - Screening AUC 0.586, 3-class macro AUC 0.575; legacy STK-V2 screening AUC 0.7405(BAcc 0.6621)로 baseline 대비 높음
  - → notebooks/aecd_api_model_baseline_outputs/
- [x] Phase AS-2: 3-class 5조건 비교 (direct/DWT비교/특허DWT/all-QC spectra/legacy) ✅ (2026-08-25)
  - macro OvR AUC: DWT비교(raw subject mean) 0.6595(최고) > legacy STK-V2 0.6053 > all-QC 0.5547 > direct 0.5459 > 특허DWT 0.4523(최저)
  - → notebooks/aecd_api_model_3class_outputs/
- [x] Phase AS-3: All QC-passed spectra(12,926개) 직접 입력 screening 모델 ✅ (2026-08-25)
  - subject OOF AUC 0.5957, spectrum OOF AUC 0.5763
  - → notebooks/aecd_api_model_all_qc_spectra_outputs/
- [x] Phase AS-4: 특허 "반복측정 평균스펙트럼생성" 검증 — direct mean-spectrum vs legacy STK-V2 ✅ (2026-08-28)
  - Direct raw mean-spectrum OOF AUC 0.6698 vs STK-V2 0.7316 (legacy가 더 높음); 3-class도 STK-V2가 소폭 우위(0.6210 vs 0.5616)
  - → notebooks/aecd_api_model_mean_spectrum_outputs/
- [x] Phase AS-5: 특허 "DWT 노이즈제거" 검증 — DWT 전처리 vs raw subject-mean ✅ (2026-08-25)
  - Raw subject-mean OOF AUC 0.7163 vs 특허 DWT 0.5233 — "이 데이터/설계에서는 DWT가 raw보다 높은 AUC를 만들지 않음"(원본 결론 그대로 인용)
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
