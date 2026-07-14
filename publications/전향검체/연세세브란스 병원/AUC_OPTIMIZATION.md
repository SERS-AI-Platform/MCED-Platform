# Severance Cancer Screening AUROC 최적화

## 결론

- 최종 모델: production preprocessing → subject mean → 32-point block mean → StandardScaler → Logistic Regression (`C=3`)
- 평가: unique subject 기준 stratified 5-fold CV를 100회 반복
- 반복별 pooled OOF AUROC: **0.720 ± 0.040**
- 반복 OOF probability 평균의 consensus AUROC: **0.747** (bootstrap 95% CI 0.613–0.862)
- 동일 split의 full-spectrum LR baseline: **0.629 ± 0.044**
- paired 개선폭: **+0.091 AUROC**

## 데이터 교정

기존 60명 publication cohort의 `YNOR_21`은 독립 피험자가 아니라 `YNOR_48` 검체의 2차 측정본이다. 근거는 `results/raw_data_vs_clinical_reference/ynor_id_remapping_trace.csv`와 `ynor_original_rename_manifest_20260710.csv`이다. 동일 피험자의 측정본이 서로 다른 fold에 들어갈 수 있는 누수를 막기 위해 모델 평가는 59 unique subjects(`YPAN` 30, `YNOR` 29)로 수행했다.

## 실험 요약

| Setting | 평가 | AUROC |
|---|---|---:|
| 기존 10-model LR meta ensemble | 60-row 단일 5-fold pooled OOF | 0.493 |
| 과거 full-spectrum LR | 59명 단일 5-fold fold mean | 0.692 |
| 재평가 full-spectrum LR | 59명, repeated 5-fold 100회 | 0.629 ± 0.044 |
| **최적 bin32 LR (`C=3`)** | **59명, repeated 5-fold 100회** | **0.720 ± 0.040** |

탐색 범위는 full spectrum LR, PCA-LR, univariate feature selection, derivative spectra, 2–128 point spectral binning, LR regularization이었다. 개발 split과 분리한 seed 검증 및 nested CV에서 32-point binning이 반복 선택되어 최종 설정으로 고정했다.

## 해석 제한

- 59명 단일기관 내부 CV이므로 외부 검증 성능이 아니다.
- consensus AUROC 0.747은 100개 OOF 예측을 평균한 값이며, primary estimate는 반복별 AUROC 평균 0.720이다.
- stage·성별·연령 같은 임상 변수는 AUC 최적화에 사용하지 않았다.
- threshold 0.5 기반 accuracy와 sensitivity/specificity는 별도 threshold 최적화 없이 계산했다.
