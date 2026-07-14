# 연세세브란스 병원 전향검체 SERS 분석 산출물

## Cohort

- primary unique-subject 분석 대상: 59명, 295 spectra
- 전향 췌장암: `YPAN` 30명, 150 spectra
- 전향 정상/대조: `YNOR` 29명, 145 spectra
- 각 subject당 5 replicate 사용, subject-level mean spectrum으로 aggregation
- 임상정보 매칭: `YPAN` 30/30, `YNOR` 29/29
- `YNOR_21`은 독립 피험자가 아니라 `YNOR_48` 검체의 2차 측정본이므로 primary 분석에서 제외
- Cancer Type ID는 적용 불가: 이 subset에는 암종이 `PAN` 하나만 존재

## Clinical data mapping

- 원본 mapping table: 60 rows, 300 spectra
- Clinical row mapped: 59명, 295 spectra
- Repeat acquisition without independent clinical row: `YNOR_21`, 5 spectra
- Primary model/figure cohort: 59 unique subjects, 295 spectra
- Mapping key: `source_group + sample_id` → `subject_id` → `clinical_patient_id`

| Mapping table | Unit | Rows | Use |
|---|---:|---:|---|
| `tables/yonsei_clinical_data_mapping.csv` | subject | 60 | 검체별 임상정보 전체 매핑 |
| `tables/yonsei_specimen_clinical_mapping.csv` | spectrum replicate | 300 | 각 replicate spectrum과 임상정보 매핑 |
| `tables/yonsei_sample_clinical_mapping_compact.csv` | subject | 60 | 리뷰용 compact mapping |

### YNOR·YPAN 전체 inventory

- 전체 정리: `YNOR_YPAN_DATA_SUMMARY.md`
- 임상 레코드: YNOR 30명, YPAN 30명
- Primary spectrum 독립 피험자: YNOR 29명, YPAN 30명
- Primary/reacquired acquisition inventory: `tables/yonsei_ynor_ypan_acquisition_inventory.csv` (검체 단위, Git 비추적)
- Pseudonymized full clinical records: `tables/yonsei_ynor_ypan_clinical_records.csv` (검체 단위, Git 비추적)
- Clinical field completeness: `tables/yonsei_ynor_ypan_clinical_completeness.csv`
- Processed acquisition 포함·제외 상태: `tables/yonsei_ynor_ypan_processed_inventory.csv` (검체 단위, Git 비추적)

### SharePoint 업로드본

- 아래 패키지는 생성 후 SharePoint에 보관하며 Git에는 포함하지 않는다.
- 통합 Excel: `sharepoint_upload/2026_SERS-AI_Yonsei_Prospective_Cohort_v1.0/2026_SERS-AI_Yonsei_Prospective_Cohort_v1.0.xlsx`
- 업로드용 ZIP: `sharepoint_upload/2026_SERS-AI_Yonsei_Prospective_Cohort_v1.0.zip`
- 내용: cohort 요약, 측정 일정, 비식별 subject manifest, 1차·2차 acquisition inventory, YNOR ID mapping, 임상정보 completeness
- 제외 항목: 직접식별자, 정확한 임상·측정 일자, 자유서술 과거력, 원본 경로, raw spectrum

## Figures

- Fig01: `figures/fig01_severance_preprocessing_pipeline.png` / `.pdf` — 전처리 단계 audit
- Fig02: `figures/fig02_severance_vs_legacy_pan_processed.png` / `.pdf` — clean retrospective CPAN 54명 vs Severance prospective YPAN 30명
- Fig03: `figures/fig03_screening_binary_auc_confusion_matrix.png` / `.pdf` — optimized LR repeated-CV confusion matrix, ROC, metrics
- Fig04: `figures/fig04_screening_peak_sets_common_differential.png` / `.pdf` — YNOR vs YPAN 공통·차등 peak set
- Fig05: `figures/fig05_pancreatic_stage_group_difference.png` / `.pdf` — YPAN 임상 stage 1-2 vs 3-4 내부 비교

기존 분리형 성능 그림(`fig01_yonsei_cancer_screening_roc`, `fig02_yonsei_screening_metrics`,
`fig03_yonsei_confusion_matrix`)도 보존했다.

## Descriptive spectrum 결과

- unique-subject cohort 기준: YPAN 30명, YNOR 29명, clean retrospective CPAN 54명
- Fig01 raw audit와 processed descriptive analysis 모두 동일한 59명 unique-subject 기준
- Fig04 peak criteria: 그룹 평균 SNV spectrum, prominence `max(0.08, group mean range의 10%)`, 최소 간격 18 cm⁻¹, 공통 허용 범위 ±12 cm⁻¹
- Fig04 결과: 공통 peak 8개, 차등 peak cluster 8개
- Fig05 stage mapping 보유 YPAN: stage 1-2 4명, stage 3-4 19명; stage missing 7명은 stage 비교에서 제외
- Fig02의 cohort 차이, Fig04의 차등 peak, Fig05의 stage 차이는 descriptive/exploratory 결과이며 질병 특이성 또는 임상적 유의성을 단독으로 입증하지 않음

## Cancer Screening 결과

Primary setting: production preprocessing → subject mean → 32-point block mean → StandardScaler → Logistic Regression (`C=3`). Unique-subject stratified 5-fold CV를 100회 반복했다.

> 주의: 이 Cancer Screening 성능은 병원별 모집군·측정 조건 차이의 영향을 받을 수 있으므로 cross-hospital 일반화 성능으로 해석할 수 없다. 독립 병원 검증이 필요하다.

| Metric | Estimate | Interval / variability |
|---|---:|---:|
| Repeated-CV mean AUROC | 0.720 ± 0.040 | split variability 2.5–97.5%: 0.647–0.792 |
| Consensus OOF AUROC | 0.747 | 0.613–0.862 |
| PR-AUC | 0.787 | 0.646–0.896 |
| Accuracy | 0.627 | 0.508–0.746 |
| Sensitivity | 0.600 | 0.419–0.778 |
| Specificity | 0.655 | 0.472–0.821 |
| PPV | 0.643 | 0.464–0.815 |
| NPV | 0.613 | 0.435–0.783 |
| F1 | 0.621 | 0.458–0.759 |

## Model comparison

| Model | Evaluation | AUROC |
|---|---|---:|
| 이전 10-model LR meta ensemble | 60-row single 5-fold pooled OOF | 0.493 |
| 과거 full-spectrum LR | 59명 single 5-fold fold mean | 0.692 |
| 재평가 full-spectrum LR | 59명 repeated 5-fold 100회 | 0.629 ± 0.044 |
| **Optimized bin32 LR** | **59명 repeated 5-fold 100회** | **0.720 ± 0.040** |

## 해석

- 32-point binning으로 feature 수를 935개에서 29개로 줄인 LR이 동일 repeated split의 full-spectrum LR보다 AUROC +0.091 높았다.
- consensus AUROC 0.747은 반복 OOF probability 평균 결과이며, primary estimate는 repeated-CV mean 0.720이다.
- 59명 단일기관 내부 CV 결과이므로 외부 검증 성능 또는 확정적 publication claim으로 해석하지 않는다.

## 사용한 파일

- `results/kao_20260610_updated_cohort/processed_spectra.csv`
- `results/kao_20260610_updated_cohort/cohort_subject_manifest.csv`
- `results/training/stacking_v2_kao_20260610_updated/fixed_split_results.json`
- `scripts/training/train_usersnet.py`
- `models/stacking_utils.py`
- `data/raw_data/10-3. Y-Pancreatic cancer (YPAN)`
- `data/raw_data/12. Y-Normal (YNOR)`
- `results/raw_data_vs_clinical_reference/ynor_id_remapping_trace.csv`

## Tables

- `tables/yonsei_clinical_data_mapping.csv`
- `tables/yonsei_specimen_clinical_mapping.csv`
- `tables/yonsei_sample_clinical_mapping_compact.csv`
- `tables/yonsei_cohort_manifest.csv`
- `tables/yonsei_oof_predictions.csv`
- `tables/yonsei_cancer_screening_metrics_ci.csv`
- `tables/yonsei_base_model_metrics.csv`
- `tables/yonsei_clinical_summary.csv`
- `tables/yonsei_confusion_counts.csv`
- `tables/yonsei_cv_folds.csv`
- `tables/yonsei_roc_curve.csv`
- `tables/fig04_screening_peak_detection_criteria.csv`
- `tables/fig04_screening_group_peak_sets.csv`
- `tables/fig04_screening_common_and_differential_peaks.csv`
- `tables/fig05_pancreatic_stage_group_counts.csv`
- `tables/severance_lr_oof_predictions.csv`
- `tables/severance_lr_repeated_cv_auc.csv`
- `tables/severance_lr_full_spectrum_baseline_auc.csv`
- `tables/severance_lr_cv_folds.csv`
- `tables/severance_lr_metrics_ci.csv`
- `tables/severance_lr_optimization_leaderboard.csv`

## Reproducibility

- 분석 스크립트: `src/yonsei_prospective_current_model.py`
- 원본 실행 위치: `scripts/analysis/yonsei_prospective_current_model.py`
- 보라매 대응 figure suite: `src/generate_severance_figures.py`
- AUROC 최적화: `src/optimize_severance_auc.py`
- 상세 실험 기록: `AUC_OPTIMIZATION.md`
- SharePoint cohort 패키지: `src/generate_sharepoint_cohort_package.py`
