# SERS / Raman 스펙트럼 전처리 설계 가이드

> **출처**: `SERS_Raman_전처리_설계가이드.docx` (2026-09-02 수신)를 git 추적
> 가능한 마크다운으로 옮긴 것. 원본 docx는 Downloads에 있어 유실 위험이 있고
> diff·grep·링크가 되지 않아 여기로 옮겼다.
>
> **이 문서의 역할**: Preprocessing Lab이 **무엇을 왜** 구현할지의 근거.
> **어떻게** 추가하는지의 절차는 `preprocessing_lab_plugin_guide.md`에 있다.
> §5의 레퍼런스 24편과 §2의 단계별 method는 `aecd_platform`의
> `experiment.papers` / `experiment.preprocessing_methods`에 등록되어 실험
> 이력과 FK로 연결된다.
단계별 정의 · 근거 · 실제 method 수식과 각 논문의 Peak 정의 연계

## 1. 개요 및 목적
전처리는 "raw spectrum → 분석 가능한 spectrum"으로 가는 변환 과정이지만, 동시에 peak 정보를 깎아낼 수 있는 최대 위험 지점이기도 하다. 과도한 smoothing은 좁은 Raman peak을 왜곡시키고, 공격적인 baseline correction은 생물학적으로 의미 있는 저주파 변동을 억제하며, normalization은 상대 peak 강도를 인위적으로 바꿀 수 있다.
따라서 각 단계마다 (a) 왜 필요한가(rationale), (b) 실제 문헌이 사용한 method와 수식(how), (c) 그 단계가 어떤 peak 정의를 전제하는가를 함께 명시해야 한다. 본 문서는 반복측정 기반 SERS 스펙트럼 품질평가 체계와 짝을 이루는 전처리 측 문서이다.

## 2. 전처리 단계별 통합표 (정의 · 근거 · 실제 method)

| 단계 | 왜 필요한가 (rationale) | 실제 method 및 수식 (how) | 근거 |
|---|---|---|---|
| ① 파수축 · 강도 보정 (calibration) | 장비·날짜별 파수 shift가 있으면 "±10 cm⁻¹ 적분" 같은 band 기반 peak 정의가 무너짐. 다장비·다기관 배포의 전제 조건 | 표준물질(ASTM E1840 등)로 파수축을 보정한 뒤, white-light source로 강도(감도) 보정. 서로 다른 여기파장으로 측정한 스펙트럼에 대한 spectrometer calibration protocol이 제시되어 있음. 실제로 26개 서로 다른 Raman 시스템 간 Raman shift 일관성이 낮으며, 그 원인이 시료가 아니라 장비 차이임이 보고됨. | [1,2,3] |
| ② Cosmic ray / spike 제거 (despiking) | CCD에 우주선이 충돌하면 좁고 강한 가짜 peak 발생 → PCA 주성분 방향 왜곡 등 후속 분석 오염 | Whitaker–Hayes: 1차 차분 detrended 스펙트럼의 modified Z-score로 spike 위치를 찾고 단순 이동평균으로 대체 ∇x(i) = x(i) − x(i−1) z_mod(i) = 0.6745 · (∇x(i) − median(∇x)) / MAD(∇x) \|z_mod\| > threshold(약 6~8) → 해당 지점을 국소 평균으로 대체 대안 변환: Shannon entropy, Laplacian(Ryabchykov), modified Z-score, Mollifier. 반복측정 기반이면 median filter 차분 threshold 방식도 사용. | [4,5] |
| ③ 구간 절단 (truncation) | 관심 영역 밖은 정보 대비 noise·형광 기여가 큼 | 생물학적 진동 모드가 집중된 fingerprint 영역(예: 600–1800 cm⁻¹)으로 절단. 절단 경계가 baseline fitting 결과를 바꾸므로, baseline correction 이전에 고정해야 함. | [6] |
| ④ Denoising / smoothing | shot noise를 줄여 SNR을 확보. 단, peak 강도·폭을 깎으면 안 됨 | Savitzky–Golay: 창 길이 m, 다항식 차수 p의 국소 최소자승 적합(계수는 convolution으로 사전계산). 실무 원칙 — 창 길이 m은 대상 peak의 최소 FWHM보다 작아야 함. Wavelet threshold denoising (WTD): 소파 계수 임계처리 후 역변환. Wiener estimation: 수치 calibration dataset으로 noise 통계를 추정한 뒤 최소 MSE 복원. CDAE (딥러닝): convolutional autoencoder의 bottleneck에 추가 conv layer와 비교 함수를 결합해 denoising과 baseline correction을 통합 수행 → 파라미터 선택이 운용자 경험에 의존하는 고전 알고리즘의 문제를 회피. | [7,8,9,10] |
| ⑤ Baseline correction (형광 배경 제거) | 형광 배경은 Raman 신호보다 수 자릿수 강할 수 있고 peak 강도비를 왜곡시킴 | ModPoly: 최소자승 다항식 적합의 변형에 기반한 자동 형광 차감. 반복마다 x⁽ᵏ⁺¹⁾ = min(x⁽ᵏ⁾, p⁽ᵏ⁾). I-ModPoly (Zhao 2007): 잔차 표준편차 σ를 종료조건에 도입해 저SNR에서의 과적합 억제. asLS (Eilers): min Σ wᵢ(yᵢ − zᵢ)² + λ Σ (Δ²zᵢ)²,  wᵢ = p (yᵢ > zᵢ), 1 − p (그 외). airPLS: 적합 baseline과 원신호 간 SSE의 가중치를 직전 반복의 baseline–원신호 차이로부터 적응적으로 갱신 → peak detection 같은 사전정보나 사용자 개입 불필요. arPLS: 비대칭 재가중을 로지스틱 함수로 완화. SNIP: LLS 변환 후 반복 clipping. | [11,12,13,14,15] |
| ⑥ Normalization | 레이저 출력·초점·hot spot 수 변동에서 오는 배율(multiplicative) 변동 제거 | vector norm x/‖x‖₂ · SNV (x − x̄)/sd(x) · min–max band normalization: 특성 band 적분강도로 나눔 (예: 1004 cm⁻¹ phenylalanine, 1441 cm⁻¹ lipid). EMSC: x = a + b·m + Σₖ dₖ νᵏ + e  →  x_corr = (x − a − Σₖ dₖ νᵏ) / b. SERS 전용: EF 변동을 target과 동일하게 반응하는 internal standard 첨가로 완화하고, target 신호를 IS 신호로 정규화 (4-MBA, d5-pyridine 등). | [16,17,18] |
| ⑦ 순서 결정 · 조합 최적화 | 같은 알고리즘도 적용 순서가 바뀌면 결과가 달라짐 → 파이프라인 자체가 하이퍼파라미터 | baseline correction을 normalization보다 먼저 수행해야 하며, 충분한 baseline correction 후 total intensity normalization을 적용했을 때 견고한 calibration model이 얻어짐. SNV·MSC·EMSC를 기본형 그대로 쓰면 여러 데이터셋의 baseline 특성을 감당하지 못함. Bocklitz 등은 30,000개 이상의 전처리 전략을 비교했고, grid search가 약 6일 걸린 반면 유전 알고리즘(GA)은 7분 만에 준최적 전략을 반환. | [16,19] |
| ⑧ Model transfer / 표준화 | 장비·기관이 바뀌면 전처리만으로는 부족 | EMSC 기반 model transfer로 device-specific 성분을 보정. 전 과정을 experimental design – data preprocessing – data learning – model transfer 4부로 구성한 프로토콜이 표준 참조. | [3,20] |
핵심 순서: calibration → despike → truncation → (denoise) → baseline → normalization → feature/peak 추출. smoothing과 baseline의 선후는 문헌상 논쟁적이므로, 두 순서를 모두 실행한 뒤 peak fidelity 지표(ΔI_rel, Δν)로 결정하는 것이 안전하다.

## 3. Peak 정의 방식이 전처리 선택을 규정한다
품질평가 문서에서 정리한 3가지 peak 정의 관점은, 각각 어떤 전처리를 허용하고 어떤 전처리를 금지하는지로 직결된다.

| Peak 정의 관점 | 정의 | 전처리에 부과되는 제약 | 근거 |
|---|---|---|---|
| 파라메트릭 (Lorentzian) | peak = (ν₀, FWHM, 면적 A) 세 파라미터를 갖는 Lorentzian의 중첩 | 각 peak을 중심으로 폭 n = FWHM인 창 안에서 SNR과 RMSE를 계산해 peak 영역의 변화만 집중 평가. → smoothing 창 길이는 반드시 최소 FWHM보다 작아야 하고, baseline의 곡률 반경은 FWHM보다 훨씬 커야 함. | [9] |
| 물리 / 기전 (hot spot) | I_SERS = R_in · (hot-spot 항 + non-hot-spot 항) + fluorescence(baseline) + noise | baseline이 물리 모델의 한 항이므로 무차별 제거는 곧 정보 손실. hot spot 수 N_H와 hot spot 내 분자 수 n_iH의 변동이 상대강도를 흔들기 때문에, normalization은 internal standard 또는 특정 band 기준이 타당. | [21] |
| 통계 / 특성-band | peak = 알려진 분자 band의 적분강도(및 그 분산). 예: 1441(lipid), 1001(Phe), 1659(amide I)를 ±10 cm⁻¹ 적분 | 적분창이 ±10 cm⁻¹이므로 파수축 오차가 이보다 작아야 함 → ① calibration 단계가 필수 전제. 적분값의 RSD로 재현성을 보므로, normalization 기준으로 삼는 band 자체가 안정적이어야 함. | [22,23,24] |

### 실무적 함의 — 정의에 따라 검증 지표가 달라진다
Lorentzian 정의 → FWHM 창 내 SNR / RMSE, Δν, ΔI_rel
Band 적분 정의 → 적분강도 RSD, 파수축 drift (cm⁻¹)
Hot-spot 정의 → baseline 분산 Var(b(ν)), internal standard 대비 상대강도 안정성

## 4. 적용 시 체크포인트
Ground truth 부재 문제. 임상 SERS에는 clean spectrum이 없으므로 MSE / PSNR을 쓸 수 없다. 반복측정 residual σ̂(ν)를 noise 참조로 삼고, 전처리 전후 σ̂ 감소량 대비 peak 면적 손실량의 trade-off 곡선을 그리는 방식이 현실적이다.
파라미터 의존성. 고전 알고리즘의 효과는 파라미터 설정에 좌우되고, 파라미터 선택은 대체로 운용자 경험에 의존한다. λ, p, 다항식 차수, SG 창 길이는 grid search 또는 GA로 고정한 뒤 SOP에 명시해야 재현성 QC를 통과한다.
순서 고정. baseline → normalization 순서는 문헌적으로 합의되어 있다 [16]. 반대로 하면 배경이 정규화 배율에 섞여 들어간다.
딥러닝 전처리 도입 시. CDAE류는 denoising과 baseline correction을 통합하지만, 학습 데이터에 없는 peak을 지워버릴 위험이 있다. FWHM 창 기반 peak fidelity 검증 [9] 없이는 임상 적용이 불가하다.

## 5. 참조 (Vancouver style)
ASTM E1840-96(2014). Standard guide for Raman shift standards for spectrometer calibration. West Conshohocken (PA): ASTM International; 2014.
Bocklitz TW, Dörfer T, Heinke R, Schmitt M, Popp J. Spectrometer calibration protocol for Raman spectra recorded with different excitation wavelengths. Spectrochim Acta A Mol Biomol Spectrosc. 2015;149:544–9.
Guo S, Popp J, Bocklitz T. Chemometric analysis in Raman spectroscopy from experimental design to machine learning–based modeling. Nat Protoc. 2021;16(12):5426–59.
Whitaker DA, Hayes K. A simple algorithm for despiking Raman spectra. Chemom Intell Lab Syst. 2018;179:82–4.
Ryabchykov O, Bocklitz T, Ramoji A, Neugebauer U, Förster M, Kroegel C, et al. Automatization of spike correction in Raman spectra of biological samples. Chemom Intell Lab Syst. 2016;155:1–6.
Afseth NK, Segtnan VH, Wold JP. Raman spectra of biological samples: a study of preprocessing methods. Appl Spectrosc. 2006;60(12):1358–67.
Savitzky A, Golay MJE. Smoothing and differentiation of data by simplified least squares procedures. Anal Chem. 1964;36(8):1627–39.
Barton SJ, Ward TE, Hennelly BM. Algorithm for optimal denoising of Raman spectra. Anal Methods. 2018;10(30):3759–69.
Han M, Dang Y, Han J. Denoising and baseline correction methods for Raman spectroscopy based on convolutional autoencoder: a unified solution. Sensors (Basel). 2024;24(10):3161.
Bai Y, Liu Q. Denoising Raman spectra by Wiener estimation with a numerical calibration dataset. Biomed Opt Express. 2020;11(1):200–14.
Lieber CA, Mahadevan-Jansen A. Automated method for subtraction of fluorescence from biological Raman spectra. Appl Spectrosc. 2003;57(11):1363–7.
Zhao J, Lui H, McLean DI, Zeng H. Automated autofluorescence background subtraction algorithm for biomedical Raman spectroscopy. Appl Spectrosc. 2007;61(11):1225–32.
Eilers PHC, Boelens HFM. Baseline correction with asymmetric least squares smoothing. Leiden: Leiden University Medical Centre Report; 2005.
Zhang ZM, Chen S, Liang YZ. Baseline correction using adaptive iteratively reweighted penalized least squares. Analyst. 2010;135(5):1138–46.
Baek SJ, Park A, Ahn YJ, Choo J. Baseline correction using asymmetrically reweighted penalized least squares smoothing. Analyst. 2015;140(1):250–7.
Bocklitz T, Walter A, Hartmann K, Rösch P, Popp J. How to pre-process Raman spectra for reliable and stable models? Anal Chim Acta. 2011;704(1–2):47–56.
Butler HJ, Ashton L, Bird B, Cinque G, Curtis K, Dorney J, et al. Using Raman spectroscopy to characterize biological materials. Nat Protoc. 2016;11(4):664–87.
Bell SEJ, Charron G, Cortés E, Kneipp J, Lamy de la Chapelle M, Langer J, et al. Towards reliable and quantitative surface-enhanced Raman scattering (SERS): from key parameters to good analytical practice. Angew Chem Int Ed. 2020;59(14):5454–62.
Guo S, Kohler A, Zimmermann B, Heinke R, Stöckel S, Rösch P, et al. Extended multiplicative signal correction based model transfer for Raman spectroscopy in biological applications. Anal Chem. 2018;90(16):9787–95.
Guo S, Heinke R, Stöckel S, Rösch P, Bocklitz T, Popp J. Towards an improvement of model transferability for Raman spectroscopy in biological applications. Vib Spectrosc. 2017;91:111–8.
Zhao Y. On the measurements of the surface-enhanced Raman scattering spectrum: effective enhancement factor, optical configuration, spectral distortion, and baseline variation. Nanomaterials (Basel). 2023;13(23):2998.
Picot F, Dallaire F, Daoust F, Chaikho L, Sheehy G, Bégin T, et al. Data consistency and classification model transferability across biomedical Raman spectroscopy systems. Transl Biophotonics. 2021;3(4):e202000019.
Liu Q, Azziz A, Cucuiet V, Majdinasab M, Arib C, Yang X, et al. Investigating the reproducibility and repeatability of commercial SERS substrates using a new methodological approach. Anal Methods. 2026;18(9):1917–27.
Xue B, Bi X, Dong Z, Xu Y, Liang M, Fang X, et al. Deep spectral component filtering as a foundation model for spectral analysis demonstrated in metabolic profiling. Nat Mach Intell. 2025;7(5):743–57.

---

## ⚠️ 출처 검증 결과 (2026-09-02 추가, 원문 대조)

이 문서의 참조 24편 중 합법적으로 전문을 확보할 수 있는 것을 받아 주장과
대조했다. 전체 결과: `docs/reviews/2026-09-02-design-guide-source-verification.md`

**본문은 원본 그대로 두고, 확인된 불일치만 아래에 기록한다:**

| 위치 | 가이드 주장 | 원문 확인 결과 |
|---|---|---|
| §3 물리/기전 | "baseline이 물리 모델의 한 항이므로 무차별 제거는 곧 정보 손실" | ❌ [21] Zhao는 이를 주장하지 않는다. 오히려 §5에서 정량화 전제조건으로 "baseline signals … have been removed"를 든다 — baseline을 **제거 대상 간섭 성분**으로 본다. 이 문장은 **가이드 작성자의 해석**이다. |
| §3 수식 | `I_SERS = R_in·(hot-spot + non-hot-spot) + fluorescence` | ⚠️ 원문 Eq.(1)은 형광이 `R_in` **안**에 있고(`R_in` 밖은 `I_noise`뿐), `I_FLU`/`I_bk` 항이 있다. baseline도 peak과 같은 장비 응답을 겪는다는 의미가 달라진다. |
| §3 internal standard | [21] 인용 | ⚠️ 실제 출처는 **[18] Bell 2020**이다. |
| §2 ⑧ EMSC | [16,17,18] 인용 | ⚠️ [18]은 EMSC 언급 0회, [17]은 툴박스 이름으로만 등장. 실질 근거는 [16]뿐이며 미확보(유료) → **미검증**. |

**반대로 검증된 것**: [17] Butler 2016은 §2의 파이프라인 순서를 직접 뒷받침한다.
다만 같은 논문이 "**smoothing은 제한적·신중하게 쓰라**"고 권고하는데, 이는 현재
프로덕션 기본값(SG smoothing 상시 on)과 긴장 관계에 있다.
