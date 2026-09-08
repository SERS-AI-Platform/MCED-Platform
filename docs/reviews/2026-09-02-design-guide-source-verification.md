# 설계 가이드 출처 검증 — `preprocessing_design_guide.md`

**날짜**: 2026-09-02
**대상**: `docs/ml/preprocessing_design_guide.md` (원본
`SERS_Raman_전처리_설계가이드.docx`)
**방법**: 참조 24편 중 합법적으로 전문 확보 가능한 것을 받아 가이드의 주장과
1:1 대조. 유료 장벽은 우회하지 않음.

## 요약

가이드의 전처리 단계 서술(§2)은 확보 가능한 출처들과 **대체로 잘 맞는다**.
다만 §3(peak 정의 관점)의 **물리/기전 항목에서, 인용된 논문이 가이드의 주장을
뒷받침하지 않는다.** 오히려 반대 방향을 전제한다.

---

## 🔴 FINDING 1 — [21] Zhao가 가이드의 규범적 주장을 뒷받침하지 않음

**가이드 §3 (표 2, 물리/기전 행)의 주장:**
> "baseline이 물리 모델의 한 항이므로 무차별 제거는 곧 정보 손실"

**출처 [21] Zhao 2023 (Nanomaterials 13(23):2998, CC-BY, 전문 확보)의 실제 서술:**

Zhao는 전처리에 대한 권고를 **하지 않는다.** 그러나 그 문제가 등장하는 유일한
지점(§5, 정량화 전제조건)에서 **정반대를 전제한다** — 정량화가 성립하려면:
> "(3) any interfering spectral features, **such as baseline signals or
> background medium, have been removed** from the measured SERS spectrum"

즉 Zhao는 baseline을 **제거해야 할 간섭 성분(interfering feature)** 으로 분류한다.
또한 ML 적용에 대해서도 정규화된 스펙트럼 사용을 지지한다:
> "This forms the theoretical basis for using normalized SERS spectra in
> machine learning and deep learning… models."

**판정**: "무차별 제거는 정보 손실" 은 **가이드 작성자의 해석**이며 [21]의
주장이 아니다. 문헌 근거로 제시하면 부정확하다.

**Zhao가 실제로 뒷받침하는 것** (이건 유효하다): baseline의 진폭·형태가 기판의
국소 광학 특성에 따라 크게 변한다는 점(단조 → 포물선형까지), 그리고 hot spot
수 `N_H`와 `n_iH` 변동이 상대강도 변동을 만든다는 점 (Eq. 2, 4–6).

## 🟡 FINDING 2 — [21] 수식 표기가 원문과 다름

**가이드**: `I_SERS = R_in · (hot-spot 항 + non-hot-spot 항) + fluorescence`

**원문 Eq.(1)**:
> `I_SERS(Δv) = R_in(Δv)(I_AH + I_AR + I_BH + I_BR + I_MH + I_MR + I_BS + I_FLU + I_bk) + I_noise`

차이 두 가지:
1. 형광이 가이드에서는 `R_in` **밖**에 있으나 원문에서는 **안**에 있다.
   `R_in` 밖에 있는 것은 `I_noise`뿐이다. 이건 사소하지 않다 — baseline도
   peak과 **동일한 장비 응답·광학 감쇠를 겪는다**는 뜻이기 때문이다.
2. `I_FLU`, `I_bk` 항이 가이드에서 누락됐다.

## 🟡 FINDING 3 — internal standard 처방의 출처가 [21]이 아님

가이드 §3의 "internal standard 대비 상대강도 안정성" 문구는 [21]이 아니라
**[18] Bell 2020**(Angew Chem, CC-BY, 전문 확보)이 명시하는 내용이다.
인용 번호를 정정해야 한다.

## 🟡 FINDING 4 — EMSC 수식의 인용 사슬이 끊어져 있음

가이드 §2 ⑧단계의 EMSC 서술은 [16,17,18]을 인용하는데:
- [18] Bell: EMSC 언급 **0회**
- [17] Butler: "EMSC"가 소프트웨어 표의 MATLAB 툴박스 이름으로만 2회 등장
- 따라서 수식의 실질적 근거는 **[16] Bocklitz 2011뿐**이며, 이 논문은 유료라
  확보하지 못했다 → **미검증**
- 정작 EMSC 원 논문 격인 [19] Guo 2018은 이 줄에 인용돼 있지 않다 (유료, 미확보)

또한 가이드 §2에서 [19]를 근거로 단 "GA/30,000 전처리 조합" 취지의 주장은
[19] 초록에서 확인되지 않는다 → **미검증**.

## ✅ 검증된 것 (반대로 잘 맞는 부분)

[17] Butler 2016 (Nat Protoc, 저자 원고 확보)은 가이드의 **파이프라인 순서**를
직접 뒷받침한다: QC 스크리닝 → 노이즈 저감 → baseline → normalization → 축소.
추가로 두 문장이 우리 파이프라인 선택에 직접 관련된다:
> "Wherever possible we advise the use of **derivative baseline correction and
> vector normalisation**"

> "we note that these processes also degrade spectral features and **recommend
> limited and cautious use of smoothing**."

두 번째는 현재 우리 프로덕션 기본값(SG smoothing 상시 on)과 긴장 관계에 있다.

## 조치 권고 (문서 수정은 사용자 판단)

1. §3 물리/기전 행의 "무차별 제거는 정보 손실" 문장을 **가이드 저자의 해석**으로
   표시하거나, 실제로 그렇게 주장하는 다른 출처를 찾아 재인용.
2. Eq.(1) 표기를 원문대로 정정 (형광을 `R_in` 안으로, `I_FLU`/`I_bk` 복원).
3. internal standard 문구의 인용을 [21] → [18]로 정정.
4. EMSC 수식 인용에 [19]를 추가하거나, [16] 확보 전까지 미검증으로 표시.

## 확보 실패 목록 (우회하지 않음)

[1] ASTM(유료 표준) · [2][5][6][7][11][12][15][16][19][20] 유료 · [13] Eilers
보고서(무료 사본 없음) · [22] Picot(OA인데 Cloudflare로 도구 접근 불가) ·
[23][24] 초록만. **[24] Xue는 코드가 공개**돼 있다 (Zenodo 10.5281/zenodo.15013288).
