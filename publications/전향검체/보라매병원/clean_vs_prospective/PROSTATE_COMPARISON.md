# 보라매 전향검체 Screening 및 3군 성능

## 분석 대상

- 총 109명: Control 21명, Biopsy-negative (PSA 상승·전립선 생검 음성) 47명, 전립선암 41명
- `_ave` 제외, `_1.._5` replicate를 subject mean으로 집계
- 성능은 subject-level nested 5-fold OOF, 내부 4-fold에서 Logistic Regression `C` 선택
- 95% CI는 OOF 예측의 2,000회 bootstrap

## 2군 성능

| 정의 | N | ROC-AUC | Balanced accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|
| Screening: Control+Biopsy-negative vs Cancer | 109 | 0.714 (0.611-0.814) | 0.704 (0.617-0.794) | 0.659 (0.514-0.804) | 0.750 (0.642-0.848) |
| Screening + urea alignment | 109 | 0.752 (0.653-0.849) | 0.707 (0.622-0.797) | 0.634 (0.486-0.786) | 0.779 (0.679-0.873) |
| Strict: Control vs Cancer | 62 | 0.928 (0.851-0.981) | 0.832 (0.724-0.927) | 0.902 (0.795-0.977) | 0.762 (0.560-0.933) |
| Strict + urea alignment | 62 | 0.918 (0.831-0.984) | 0.820 (0.708-0.916) | 0.878 (0.773-0.973) | 0.762 (0.571-0.933) |

- Screening은 Biopsy-negative 47명을 non-cancer에 포함한다.
- Strict는 Biopsy-negative를 제외한 Control 21명 vs Cancer 41명 비교다.
- Figure: `figures/fig03_screening_binary_auc_confusion_matrix.png` / `.pdf`

## 3-group 성능

| 정의 | N | Macro OVR ROC-AUC | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| 3-group: Control/Biopsy-negative/Cancer | 109 | 0.811 (0.748-0.874) | 0.668 (0.576-0.758) | 0.674 (0.583-0.759) |
| 3-group + urea alignment | 109 | 0.834 (0.779-0.893) | 0.702 (0.618-0.791) | 0.691 (0.603-0.776) |

- Figure: `figures/fig06_three_group_auc_confusion_matrix.png` / `.pdf`

## Train overfitting 처리

- 기존 single split은 train accuracy 1.000, test accuracy 0.697로 과적합 신호가 명확했다.
- 새 주 결과는 outer fold에 한 번도 학습되지 않은 nested OOF 예측만 사용한다.

## Clean PRO와 전향 암군 차이 및 peak alignment

- Clean retrospective PRO 91명(CBNUH) vs 보라매 전향 전립선암 41명
- 평균 전처리 스펙트럼 Pearson r: 0.826
- Cohort-source ROC-AUC: 0.999; urea alignment 후 0.998
- 높은 source AUC는 병원·채취시점·batch 차이이며 생물학적 차이로 해석하지 않는다.

| 단계 | Clean peaks | Prospective peaks | Matched | ±5 cm⁻¹ | Median abs shift | Max abs shift |
|---|---:|---:|---:|---:|---:|---:|
| Common grid only | 10 | 11 | 7 | 7 | 1.92 | 3.85 |
| Urea aligned | 10 | 11 | 7 | 7 | 1.92 | 1.92 |

- Urea anchor: Clean 455/455, prospective cancer 188/205 replicates
- Figure: `figures/fig07_legacy_vs_prospective_peak_alignment.png` / `.pdf`

## 사용한 파일

- `data/clinical_data/보라매 병원 임상정보.xlsx`
- `results/clean_cohort_20260605/clean_cohort_manifest.csv`
- `data/raw_data/1. Prostate cancer (100개)/PRO *_1..5.CSV`
- `data/raw_data/20260709_BPRO,BNOR_1mW_0.05s_Ave100/*_1..5.CSV`
- `artifacts/usersnet/v1.0.0/common_grid.npy`

## 해석 주의

- 결과는 exploratory nested OOF 성능이며 외부검증 성능이 아니다.
- Cancer Screening AUC를 cross-hospital 일반화 근거로 사용하면 안 된다.
