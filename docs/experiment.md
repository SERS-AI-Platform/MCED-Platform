# SERS-AI Experiment Tracker

> **Single source of truth** for experiment planning and status.
> 기계용 이력은 `logs/experiment_registry.json` 참조.

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
