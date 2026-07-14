# 환자별 Peak/Spectrum 기반 Explainability — 설계안

- 상태: 설계 문서 (코드 미구현)
- 관련: IFU/사양서 1.5.1 "Explainability (Contributed Peak or Spectrum Difference)"
- 전제: [[threshold 단일화]] 완료(threshold=0.60, artifacts/usersnet/current→v1.0.0), [[SSI risk stratification]] 완료(3단계 LOW/MODERATE/HIGH)

## 1. 현재 존재하는 것 — Cohort-level SHAP (AACR 파이프라인)

`publications/aacr/src/fig_spectra_shap.py` + `src/sers/visualization/shap.py`에 이미 구현되어 있는 방법론:

```
SHAP_i = coef_i × (x_i − E[x_i])
```

- `coef_i`: Cancer Screening **Fusion LR**(binary, cancer vs non-cancer) 계수, 5-fold 평균
- `x_i`: **암종별 평균 스펙트럼**의 i번째 wavenumber 강도 (cohort mean, 환자 개인 아님)
- `E[x_i]`: 전체(non-cancer 포함) 평균 스펙트럼의 i번째 강도
- Top-5 피크를 `PEAK_GROUP_RADIUS=15 cm⁻¹` 내에서 병합해 대표 피크로 표시
- 표시 규칙(메모리 확정): `|SHAP|` 절대값 막대 + ↑/↓ 방향 텍스트 (부호 있는 막대 금지)

**한계**: 이 값은 "암종 A 환자군 평균이 전체 평균과 어떻게 다른가"이지, **"이 환자 1명이 왜 이 결과가 나왔는가"가 아님.** IFU가 요구하는 건 후자(per-patient attribution).

## 2. 설계안: Per-patient Peak Attribution

### 2.1 계산식 (기존 공식 재사용, cohort mean → 환자 값으로 교체)

```
contribution_i(patient) = coef_i × (x_i(patient) − x̄_i(predicted_type_cohort))
```

- `coef_i`: 기존과 동일, Cancer Screening Fusion LR 계수 (재학습 불필요, 이미 아티팩트에 존재)
- `x_i(patient)`: **해당 환자**의 전처리된 스펙트럼 i번째 값 (추론 시점에 이미 계산됨 — `sers_predict.py`의 `features`)
- `x̄_i(predicted_type_cohort)`: Cancer Type ID가 예측한 암종의 학습 코호트 평균 스펙트럼 (오프라인 사전계산, 아티팩트로 저장)
- 935 wavenumber 전체에 대해 계산 후, `known_peaks`(17개, `peak_config.json`) 반경 내로 병합해 대표 피크 단위로 집계 → 기존 AACR 로직과 동일한 병합 함수 재사용 가능

이 방식은 **재학습이 필요 없다** — Cancer Screening Fusion LR 계수와 암종별 코호트 평균 스펙트럼(오프라인 1회 계산 후 아티팩트에 저장)만 있으면 추론 시점에 환자 벡터와의 내적 연산 한 번으로 끝남.

### 2.2 출력 스펙 (물리량 초안)

| 필드 | 설명 |
|---|---|
| `peak_name` | known_peaks의 이름 (예: `creatinine`, `amide_I`) |
| `wavenumber` | 중심 파수 (cm⁻¹) |
| `contribution` | `\|coef_i × (x_i − x̄_i)\|` — 절대값 (기존 [[feedback_feature_importance_positive]] 규칙 준수) |
| `direction` | `↑` (환자 강도가 코호트 평균보다 높음, cancer 방향 기여) / `↓` (낮음) |
| `patient_value` / `cohort_mean` | 환자 강도 vs 코호트 평균 강도 (참고용 raw 값) |
| `rank` | 기여도 순위, Top-N (AACR과 동일하게 N=5 권장) |

### 2.3 Spectrum Difference 시각화

환자 스펙트럼(raw 또는 1차 미분) 위에:
1. 예측 암종의 코호트 평균 스펙트럼 (참조선)
2. Non-cancer 코호트 평균 스펙트럼 (대조선)
3. Top-N 기여 피크 위치에 마커/음영

`fig_spectra_shap.py`의 플롯 함수를 "cohort mean vs cohort mean" → "patient vs cohort mean(×2)"로 바꾸면 재사용 가능. 신규 플로팅 로직 불필요.

## 3. 미해결 질문 (구현 전 결정 필요)

1. **Cancer Type ID 기여도는 다룰지 여부.** 위 설계는 "왜 cancer로 판정됐는가"(Cancer Screening)만 설명한다. "왜 하필 이 암종인가"(Cancer Type ID)는 [[feedback_shap_model]]에 따르면 multiclass 계수가 "덜 해석 가능"하다고 이미 결론남 — Cancer Type ID 설명은 이 설계로 커버되지 않으며 별도 방법론 검토 필요.
2. **코호트 평균 스펙트럼의 기준 코호트가 무엇인가.** production 모델(v1.0.0, n=1569) 학습 코호트 기준으로 고정할지, 최신 데이터로 주기적 재계산할지 — 재계산 정책 필요.
3. **StackingPredictor(운영 모델)와의 정합성.** 위 설계는 Cancer Screening **Fusion LR**(단일 base model) 계수를 사용한다. 그러나 실제 운영 모델은 StackingPredictor(10 base model + ElasticNet meta)로, meta-learner 단계에 선형 해석이 그대로 적용되지 않는다. 두 가지 선택지:
   - (a) 설명용으로는 base 10개 중 `lr_raw`/`lr_d1`(선형) 계수만 사용 — meta 단계는 설명에서 제외, "참고용 근사"임을 명시
   - (b) meta-learner(ElasticNet, 선형)까지 포함해 base model SHAP을 체인으로 합성 — 수학적으로는 가능(선형 조합의 선형 조합은 선형)하나 검증 필요
4. **임상 검증 요구.** Peak attribution이 실제로 알려진 대사체 마커(creatinine, hippuric acid 등)와 일치하는지 소수 샘플로 sanity check 후 배포해야 함 — AACR 코호트 레벨 결과와 방향이 일치하는지가 최소 기준.

## 4. 구현 범위 (별도 작업)

이번 세션에서는 코드 구현하지 않음. 구현 시 필요한 아티팩트:
- 암종별 코호트 평균 스펙트럼 (오프라인 계산, `artifacts/usersnet/v1.0.0/`에 신규 파일로 저장)
- `sers_predict.py`에 `explain_patient()` 메서드 추가 (predict_patient 이후 선택적 호출)
- 리포트 템플릿에 Top-N 피크 표 + 스펙트럼 diff 플롯 추가
