# 보라매 전향검체 Screening 및 3군 성능

## 분석 대상

- 총 109명: Control 21명, Biopsy-negative (PSA 상승·전립선 생검 음성) 47명, 전립선암 41명
- `_ave` 파일은 제외하고 `_1.._5` replicate만 사용, subject mean으로 집계
- 모든 성능은 subject-level nested 5-fold OOF 예측으로 계산
- 내부 4-fold에서 Logistic Regression `C`를 선택하고 외부 fold는 모델 선택에 사용하지 않음
- 95% CI는 OOF 예측의 2,000회 bootstrap

## Screening 성능

Biopsy-negative 47명은 non-cancer 쪽에 포함했다.

| 정의 | N | ROC-AUC | Balanced accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|
| Screening: Control+Biopsy-negative vs Cancer | 109 | 0.714 (0.611-0.814) | 0.704 (0.617-0.794) | 0.659 (0.514-0.804) | 0.750 (0.642-0.848) |

- Figure: `figures/fig03_screening_binary_auc_confusion_matrix.png` / `.pdf`
- Confusion matrix: `tables/screening_binary_confusion_matrix.csv`

## 3-group 성능

| 정의 | N | Macro OVR ROC-AUC | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| 3-group: Control/Biopsy-negative/Cancer | 109 | 0.811 (0.748-0.874) | 0.668 (0.576-0.758) | 0.674 (0.583-0.759) |

- Figure: `figures/fig06_three_group_auc_confusion_matrix.png` / `.pdf`
- Confusion matrix: `tables/three_group_confusion_matrix.csv`

## 해석 주의

- 현재 결과는 표본 수가 작아 validation split 없이 nested OOF로만 산출한 exploratory 성능이다.
- Cancer Screening AUC는 외부검증 성능이나 cross-hospital 일반화 근거로 사용하면 안 된다.
