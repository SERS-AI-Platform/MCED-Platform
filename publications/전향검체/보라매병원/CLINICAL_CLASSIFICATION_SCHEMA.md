# Clinical Classification Analysis Schema

## 목적

보라매 전향검체의 SERS 분류 결과를 임상정보와 연결하여, 어떤 임상군·Grade Group·소변 조건에서 Cancer가 `Biopsy-negative` 또는 `Control`로 오분류되는지 분석하기 위한 subject-level schema이다.

분석의 기본 단위는 환자 1명(subject)이다. 한 환자에 여러 Raman replicate가 있더라도 먼저 subject-level mean spectrum으로 집계하고, 모델 예측은 nested 5-fold OOF 결과만 연결한다.

## 데이터 흐름

```text
Clinical Excel (internal)
        |
        |  internal patient_code로 결합
        v
Subject-level spectrum summary
        |
        |  nested 5-fold OOF prediction
        v
Classification result
        |
        |  clinical covariate + grade derivation
        v
Clinical-classification analysis table
```

## 공개·내부 식별자 원칙

| 구분 | 필드 | 사용 범위 |
|---|---|---|
| 공개용 | `case_id` | `BPRO 1` 같은 원본 sample code가 아닌 비식별 분석 ID만 사용 |
| 내부 결합용 | `patient_code`, `solum_label`, `source_row_number` | 임상 Excel과 spectrum을 결합할 때만 사용; Figure·논문 table·외부 CSV에 포함하지 않음 |
| 원본 보존용 | `raw_group` | Excel 원본값을 내부 입력에서만 보존 |
| 논문용 | `publication_group` | `Control`, `Biopsy-negative`, `Prostate cancer` |

## Logical schema

| 영역 | 필드 | 자료형 | 정의 및 분석 용도 | 공개 여부 |
|---|---|---|---|---|
| Identity | `case_id` | string | 비식별 subject-level ID | 공개 |
| Identity | `patient_code` | string | 병원 내부 환자 식별/결합 키 | 내부 전용 |
| Cohort | `publication_group` | enum | `Control`, `Biopsy-negative`, `Prostate cancer` | 공개 |
| Cohort | `raw_group` | string | 원본 Excel group 값 | 내부 전용 |
| Cohort | `cancer_type` | string/null | 암종; 현재 prostate cohort에서는 prostate | 공개 가능 |
| Demographic | `age` | numeric/null | 채취 시점 연령 | 제한 공개 |
| Demographic | `sex` | enum/null | 성별 | 제한 공개 |
| Prostate | `psa` | numeric/null | PSA; Cancer와 Biopsy-negative 혼동의 핵심 변수 | 제한 공개 |
| Prostate | `gleason_score` | string/null | 원본 Gleason Score | 제한 공개 |
| Prostate | `grade_group` | integer/null | ISUP Grade Group 1-5 | 제한 공개 |
| Prostate | `grade_band` | enum | `GG1-2`, `GG3-5`, `Unknown`; 분석 파생 변수 | 공개 가능 |
| Prostate | `pathology_result` | string/null | 병리 결과 | 제한 공개 |
| Prostate | `stage` | string/null | 임상/병리 stage | 제한 공개 |
| Urine | `ua_sg` | numeric/null | urine specific gravity; 희석도 보정 | 제한 공개 |
| Urine | `ua_ph` | numeric/null | 소변 pH | 제한 공개 |
| Urine | `ua_protein` | categorical/numeric/null | 단백뇨 여부/값 | 제한 공개 |
| Urine | `ua_blood` | categorical/numeric/null | 혈뇨 여부/값 | 제한 공개 |
| Urine | `ua_leukocyte` | categorical/numeric/null | 백혈구 또는 염증 신호 | 제한 공개 |
| Blood/renal | `creatinine` | numeric/null | 신기능 및 urea-like band 해석 보정 | 제한 공개 |
| Blood/renal | `bun` | numeric/null | urea 관련 생리 변수 | 제한 공개 |
| Blood/renal | `glucose`, `hba1c` | numeric/null | 대사성 혼동 변수 | 제한 공개 |
| Pre-analytic | `collection_date` | date/null | 채뇨일; batch/time effect | 내부 또는 날짜 비식별화 후 공개 |
| Pre-analytic | `receipt_date` | date/null | 접수일; 보관 지연 추정 | 내부 전용 |
| Pre-analytic | `sample_type` | string/null | 검체 종류 | 공개 가능 |
| Pre-analytic | `has_post_treatment_sample` | boolean/null | 치료 후 검체 여부 | 공개 가능 |
| Pre-analytic | `treatment_info` | string/null | 치료 상태; 치료 전 검체 분석 여부 확인 | 제한 공개 |
| Model | `task` | enum | `screening_binary` 또는 `three_group` | 공개 |
| Model | `fold` | integer | 외부 OOF fold 1-5 | 공개 |
| Model | `true_label` | enum | 모델 입력 기준 실제 class | 공개 |
| Model | `pred_label` | enum | OOF 예측 class | 공개 |
| Model | `prob_control` | numeric | Control 예측 확률 | 공개 |
| Model | `prob_biopsy_negative` | numeric | Biopsy-negative 예측 확률 | 공개 |
| Model | `prob_cancer` | numeric | Cancer 예측 확률 | 공개 |
| Derived | `is_correct` | boolean | `true_label == pred_label` | 공개 |
| Derived | `error_direction` | enum/null | `correct`, `cancer_to_biopsy_negative`, `cancer_to_control`, `non_cancer_to_cancer`, 기타 | 공개 |
| Derived | `cancer_misclassified` | boolean | 실제 Cancer인데 Cancer로 예측되지 않았는지 | 공개 |
| Derived | `grade_adjustment_set` | enum | `GG1-2`, `GG3-5`, `Unknown` | 공개 |
| Derived | `clinical_missingness_class` | enum | 핵심 임상정보 complete/partial/missing | 공개 |

## 모델별 필드 사용

| 분석 질문 | 대상 | 필수 분류 필드 | 임상정보 | 결과 변수 |
|---|---|---|---|---|
| Cancer Screening 오분류 | 전체 109명 | `true_label`, `pred_label`, `prob_cancer` | PSA, age, `ua_sg`, creatinine, blood/leukocyte | Cancer miss 여부와 임상 변수의 관계 |
| 3-group 혼동 | 전체 109명 | 세 class 확률과 `error_direction` | PSA, `ua_sg`, renal/urine variables | Cancer→Biopsy-negative와 Cancer→Control 분리 |
| Grade별 Cancer miss | Cancer 41명 중 Grade 확인군 | `cancer_misclassified`, `grade_group`, `grade_band` | PSA, age, urine dilution, batch | GG1-2 vs GG3-5 오분류율 |
| PSA 독립성 | Cancer + Biopsy-negative | `prob_cancer`, `pred_label` | PSA, age, `ua_sg`, prostate volume 가능 시 PSAD | PSA 보정 후 Cancer association |
| Urea-like band 해석 | 전체 또는 Cancer 내부 | peak intensity/area와 분류 결과 | `ua_sg`, creatinine, BUN, pH | 보정 전후 peak effect |
| Batch robustness | 전체 109명 | OOF fold, prediction | collection/receipt date, substrate lot, instrument ID | batch-out 성능과 prediction stability |

## 파생 변수 규칙

```text
grade_band:
  Grade Group 1 or 2 -> GG1-2
  Grade Group 3, 4, or 5 -> GG3-5
  missing -> Unknown

error_direction:
  true_label == pred_label -> correct
  true_label == Cancer and pred_label == Biopsy-negative
    -> cancer_to_biopsy_negative
  true_label == Cancer and pred_label == Control
    -> cancer_to_control
  true_label != Cancer and pred_label == Cancer
    -> non_cancer_to_cancer
  otherwise -> other_error
```

## 해석 순서

1. 먼저 `true_label`, `pred_label`, `prob_cancer`로 실제 오분류 방향을 고정한다.
2. Cancer 오분류를 `grade_band`와 교차하여 GG1-2와 GG3-5의 miss rate를 비교한다.
3. 같은 비교를 PSA, urine specific gravity, creatinine, 혈뇨·백혈구, batch로 층화하거나 보정한다.
4. `Cancer -> Biopsy-negative`와 `Cancer -> Control`을 합치지 않고 별도 error direction으로 유지한다.
5. 임상정보가 없는 환자는 제거하지 말고 `clinical_missingness_class`로 표시한 뒤 complete-case와 available-case 결과를 함께 보고한다.

## 현재 데이터에서의 적용 범위

- 실제 결합 표: `tables/clinical_classification_analysis.csv` (109 subjects, 35 fields, UTF-8 BOM)
- 내부 매칭 표: `internal/clinical_classification_analysis_bpro.csv` (원본 `BPRO n` 라벨 기준, 109 subjects, 35 fields)
- 현재 모델 결과는 subject-level nested 5-fold OOF prediction이다.
- 현재 보라매 cohort는 Control 21명, Biopsy-negative 47명, Prostate cancer 41명이다.
- Grade Group은 Cancer 내부의 일부 환자에서만 확인되므로 `Unknown`을 별도 상태로 유지한다.
- 이 schema는 임상정보를 모델 입력에 자동으로 넣는다는 뜻이 아니다. 우선 spectrum-only 분류 결과에 임상정보를 연결해 혼동 요인을 평가하고, 이후 spectrum+clinical multimodal 모델을 별도 비교한다.
