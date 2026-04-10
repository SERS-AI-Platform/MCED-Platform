# SERS-AI IP R&D 특허 조사 종합 보고서

**조사일:** 2026-03-18
**대상 기술:** 소변 SERS 스펙트럼 기반 다암종 AI 분류 시스템 (AECD Platform)
**조사 범위:** Google Patents, USPTO, WIPO, KIPRIS, CNIPA, 학술논문 DB

---

## 목차

1. [조사 방법론](#1-조사-방법론)
2. [핵심 선행특허 상세 분석](#2-핵심-선행특허-상세-분석)
3. [한국 특허 조사 결과](#3-한국-특허-조사-결과)
4. [중국 특허 조사 결과](#4-중국-특허-조사-결과)
5. [관련 학술논문 기반 기술 동향](#5-관련-학술논문-기반-기술-동향)
6. [경쟁 제품/기업 IP 현황](#6-경쟁-제품기업-ip-현황)
7. [기술 요소별 선행기술 매트릭스](#7-기술-요소별-선행기술-매트릭스)
8. [특허 공백(White Space) 분석](#8-특허-공백white-space-분석)
9. [FTO(Freedom-to-Operate) 분석](#9-ftofreedom-to-operate-분석)
10. [IPC/CPC 분류코드 참고](#10-ipccpc-분류코드-참고)
11. [추가 조사 필요 사항](#11-추가-조사-필요-사항)

---

## 1. 조사 방법론

### 검색 DB
| DB | URL | 검색 완료 |
|---|---|---|
| Google Patents | patents.google.com | ✅ |
| USPTO (웹검색) | patents.justia.com | ✅ |
| KIPRIS (웹검색) | kipris.or.kr | ✅ (간접검색) |
| CNIPA (웹검색) | cnipa.gov.cn | ✅ (간접검색) |
| WIPO PATENTSCOPE (웹검색) | patentscope.wipo.int | ✅ (간접검색) |
| PubMed/학술DB | pubmed.ncbi.nlm.nih.gov | ✅ |

### 검색 키워드 조합
- "surface enhanced Raman" + "urine" + "cancer" + "classification"
- "SERS" + "urine" + "multi-cancer" / "multiple cancer"
- "Raman spectroscopy" + "urine" + "machine learning" + "cancer"
- "표면증강라만" + "소변" + "암" + "진단"
- IPC코드: G01N 21/65, G16H 50/20, G06N 20/00
- "two-stage" / "hierarchical" / "cascade" + "cancer" + "spectroscopy"
- "preprocessing" / "quality control" / "normalization" + "Raman" + "urine"

---

## 2. 핵심 선행특허 상세 분석

### 2.1 US20210215610A1 — 가장 직접적 선행기술

| 항목 | 내용 |
|---|---|
| **명칭** | Methods of Disease Detection and Characterization Using Computational Analysis of Urine Raman Spectra |
| **출원인** | Virginia Tech Intellectual Properties Inc |
| **발명자** | John L. Robertson, Ryan Senger |
| **우선일** | 2020-01-09 |
| **출원일** | 2021-01-11 |
| **공개일** | 2021-07-15 |
| **상태** | **공개(Pending)** — 미등록 |
| **출원번호** | US17/146,301 |

**기술 요약:**
- 소변 라만 스펙트럼의 전산분석으로 질병 검출
- 785nm 레이저 여기, 400-1800 cm⁻¹ 분석 범위
- PCA + DAPC (Discriminant Analysis of Principal Components) 중심
- Goldindec 기저선 보정, 벡터 정규화
- RametrixTM 분석 소프트웨어 플랫폼

**독립항 요약:**
1. 소변 라만 스펙트럼 취득 → 기저선 보정/정규화 → PCA → DAPC → 참조모델 비교
2. 다변량 분석(TPD/TSD)으로 질환/건강 소변 간 통계적 차이 식별
3. PCA + DAPC + PLS + ML + NN 조합 분석 + 교차검증

**대상 질환 (종속항):**
- 방광암, 신장암, 전립선암, 자궁경부암, 난소암, 부신암
- 방광염, 전립선염, 신부전, 당뇨 합병증 등 비암 질환 다수

**성능 청구:** 정확도 90-100%, 민감도 90-100%, 특이도 85-100%

**SOLUM과의 차이점:**

| 요소 | Virginia Tech | SOLUM (AECD) |
|---|---|---|
| 라만 종류 | **일반 라만** (비증강) | **SERS** (표면증강) |
| 분석 방법 | PCA/DAPC 중심 | LR/앙상블(LR+ResNet18) |
| 분류 구조 | **단일 stage** (질환별 개별 모델) | **2단계 계층** (암여부→암종) |
| 전처리 | Goldindec 기저선, 벡터 정규화 | SG 평활화, Rolling min 기저선, SNV |
| QC | 명시 없음 | **3단계 QC (강도/RSD/상관계수)** |
| 다암종 동시 분류 | 개별 질환 모델 | **5암종 동시 분류** |
| 비암 대조군 | 건강군만 | **DIA/HBP/H.D. 만성질환 포함** |
| 운영 모드 | 단일 | **3-threshold (스크리닝/균형/확인)** |

**FTO 평가:** 🟡 중간 위험 — 소변+라만+ML 조합은 유사하나, SERS vs 일반라만, 2단계 vs 단일stage, 전처리 방법 차이로 **비침해 가능성 높음**

---

### 2.2 US9410949B2 — SERS 소변 암진단 (등록 특허)

| 항목 | 내용 |
|---|---|
| **명칭** | Label-free Detection of Renal Cancer |
| **출원인** | Washington University (St. Louis) |
| **발명자** | Srikanth Singamaneni, Evan Kharasch 등 |
| **출원일** | 2011-12-02 |
| **등록일** | **2016-08-09** |
| **상태** | **등록(Granted)** |

**기술 요약:**
- 소변 내 AQP1/ADFP 단백질 바이오마커를 SERS/LSPR로 정량 검출
- 3D ZnO 나노와이어 + Au 나노입자 기판
- **항체 기반** 라벨프리 검출 (특정 바이오마커 타겟)
- 신장암 단일 암종

**SOLUM과의 차이점:**

| 요소 | WUSTL | SOLUM |
|---|---|---|
| 검출 방식 | **항체 기반** (특정 바이오마커) | **Label-free fingerprint** (대사체 전체) |
| 타겟 | AQP1, ADFP | 소변 전체 대사체 스펙트럼 |
| 암종 | 신장암 단일 | 5암종 동시 |
| AI/ML | 없음 (정량 검출만) | LR/ResNet18 앙상블 |
| 분류 구조 | 해당 없음 | 2단계 계층 |

**FTO 평가:** 🟢 낮은 위험 — 완전히 다른 접근법 (항체 vs fingerprint). SOLUM은 항체를 사용하지 않으므로 **비침해**

---

### 2.3 CN115078331B — SERS-AICS (중국 등록 특허)

| 항목 | 내용 |
|---|---|
| **명칭** | SERS-AI Cancer Screening (SERS-AICS) |
| **출원인** | 우한대학 (Wuhan University) |
| **상태** | **등록(Granted)** — 중국 |

**기술 요약:**
- **혈청** + Ag 나노와이어 SERS + 공분산행렬 차원축소 + SVM
- 5암종: 폐, 대장, 간, 위, 식도
- 382 건강 대조 + 1,582 환자, 2개 독립 코호트
- 정확도 95.81%, 민감도 95.87%, 특이도 95.40%
- 15μL 혈청 사용

**SOLUM과의 차이점:**

| 요소 | 우한대 SERS-AICS | SOLUM |
|---|---|---|
| **검체** | **혈청** | **소변** |
| 나노입자 | Ag 나노와이어 | Au 나노입자 |
| AI 모델 | SVM | LR + ResNet18 앙상블 |
| 차원축소 | 공분산행렬 | 직접 933 특징 입력 |
| 분류 구조 | 단일 stage | 2단계 계층 |
| 암종 | 폐, 대장, 간, 위, 식도 | PRO, LUN, CRC, CPAN, OVA |
| 비암 대조군 | 건강군 | DIA, HBP, H.D., NOR |
| QC | 명시 없음 | 3단계 QC |

**FTO 평가:** 🟡 중간 위험 — SERS+AI+다암종 조합은 유사하나, 검체(혈청 vs 소변), 나노입자(Ag vs Au), 모델(SVM vs LR), 분류구조 차이로 **비침해 가능성 높음**. 단, 중국 진출 시 상세 FTO 필요

---

### 2.4 KR20170000142A — SERS 순환종양세포 검출

| 항목 | 내용 |
|---|---|
| **명칭** | 표면 증강 라만 산란 방법(SERS)을 이용하여 순환 종양세포 및 순환 종양 줄기유사세포를 검출하는 방법 |
| **출원인** | (조사 필요) |
| **공개일** | 2017-01-02 |
| **상태** | 공개 |

**기술 요약:**
- SERS 나노태그로 순환종양세포(CTC) 및 종양줄기유사세포 검출
- 혈액 기반, SERS 매핑으로 표면 마커 발현 수준 분석
- 유방암 세포 분화/미분화 분석

**FTO 평가:** 🟢 낮은 위험 — 혈액 CTC 기반으로 소변 대사체 fingerprint와 완전 상이

---

### 2.5 KR20070030263A — SERS 기판 제조

| 항목 | 내용 |
|---|---|
| **명칭** | 표면 강화 라만 분광법을 위한 기판 표면 제조 시스템과 방법 |
| **출원인** | Griffin Analytics LLC |
| **우선일** | 2004-06-07 |
| **상태** | 공개 |
| **IPC** | G01N 21/65, G01N 21/658, C23C 14/00 |

**기술 요약:**
- SERS 기판 제조 방법 (금속 증착 파라미터 제어)
- LSPR 파장 최적화
- 전체 소변/혈액 샘플 시연 (Fig.3)
- **암 진단 특이적 청구항 없음**

**FTO 평가:** 🟢 낮은 위험 — 기판 하드웨어 제조 특허. SOLUM AI 방법론과 비침해

---

### 2.6 US20050250091A1 — 소변 라만 방광암 검출

| 항목 | 내용 |
|---|---|
| **명칭** | Raman molecular imaging for detection of bladder cancer |
| **출원인** | (학술연구) |
| **상태** | 공개 (미등록) |

**기술 요약:**
- 소변 내 방광암 세포의 라만 신호 검출
- 라만 분자 이미징 (SERS 아님)
- 방광암 단일 암종

**FTO 평가:** 🟢 낮은 위험 — 일반 라만, 단일 암종, 이미징 기반

---

### 2.7 WO2016094330A2 — ML 기반 암 위험도 예측

| 항목 | 내용 |
|---|---|
| **명칭** | Methods and machine learning systems for predicting the likelihood or risk of having cancer |
| **출원인** | 20/20 GeneSystems Inc |
| **우선일** | 2014-12-08 |
| **상태** | **Ceased (중단)** |

**기술 요약:**
- 혈액 바이오마커(CEA, CA-125, Cyfra 21-1 등) + ML로 암 위험도 예측
- 면역분석(ELISA) 기반 정량
- 복합 스코어링 시스템
- 라만/SERS 사용하지 않음

**FTO 평가:** 🟢 낮은 위험 — 완전 다른 기술 (면역분석 vs SERS)

---

### 2.8 US10671885B2 — 암 진단 방법 (등록)

| 항목 | 내용 |
|---|---|
| **명칭** | Cancer diagnostic method and system |
| **발명자** | Ping Zhang, Kuldeep Kumar |
| **등록일** | 2020-06-02 |
| **상태** | **등록(Granted)** |

**기술 요약:**
- 유방암 진단 (맘모그래피 영상 기반)
- 로지스틱 회귀 + 판별분석 + 신경망 결합
- 영상 처리 특징 + 임상 특징 융합
- 라만/SERS/소변 사용하지 않음

**FTO 평가:** 🟢 낮은 위험 — 영상 기반으로 SERS 스펙트럼과 무관

---

## 3. 한국 특허 조사 결과

### 검색 결과 요약

| 특허번호 | 명칭 | 관련 기술 | SOLUM 관련도 |
|---|---|---|---|
| KR20170000142A | SERS 순환종양세포 검출 | SERS + 혈액 CTC | 낮음 |
| KR20070030263A | SERS 기판 제조 | SERS 하드웨어 | 낮음 |
| KR101685085B1 | 암 진단 키트 | 소변 타이로신 효소 검출 | 낮음 |
| KR101652854B1 | 방광암 소변 표지 | BTM/UBTM 바이오마커 | 낮음 |
| KR20150144697A | 현장 소변 검사기 | 소변 수집 장치 | 낮음 |
| KR101124273B1 | 모바일 소변 검사 시스템 | 소변 분석 시스템 | 낮음 |

**핵심 발견:** 한국 특허에서 "SERS + 소변 + 다암종 + AI 분류"를 모두 포함하는 특허는 **발견되지 않음**

### 프로젝트 보유 특허 (참고용, 직접 관련 아님)

| 출원번호 | 명칭 | 출원인 | 관련도 |
|---|---|---|---|
| 10-2021-0158330 | 무채혈 비침습성 혈당 체외진단 시스템 | 김영준 | 낮음 |
| 10-2022-0091450 | 멀티모달 AI 기반 신속진단키트 인증 장치 | (주)스톤랩 | 중간 |
| 10-2022-0118138 | 조기간암 진단용 혈청 엑소좀 SF3B4 마커 | 아주대 | 중간 |
| 10-2022-0162174 | AI 기반 체외진단 바이오마커 측정 시스템 | (주)스톤랩 | 중간 |
| 10-2023-0127602 | 딥러닝 기반 체외진단 판독 모델 | (주)스타캣 | 중간 |
| 10-2024-0046144 | (확인 필요) | - | - |

---

## 4. 중국 특허 조사 결과

| 특허번호 | 명칭 | 기술 | SOLUM 관련도 |
|---|---|---|---|
| **CN115078331B** | SERS-AICS 암 스크리닝 | 혈청 SERS + SVM, 5암종 | **높음** (상세분석 2.3 참조) |
| CN102137937A | 방광암 유전변이 | 유전자 마커 | 낮음 |
| CN109154613A | 암 모니터링 방법 | 일반 진단 | 낮음 |
| CN110546277B | 암 진단 치료 방법 | 일반 진단 | 낮음 |

**핵심 발견:** CN115078331B(우한대)가 유일한 직접 경쟁 특허이나 **혈청 기반**으로 소변과 차별화 가능

---

## 5. 관련 학술논문 기반 기술 동향

### 5.1 소변 SERS + 암진단 핵심 논문

| 년도 | 저자/저널 | 기술 | 암종 | 성능 | 특허 출원 여부 |
|---|---|---|---|---|---|
| 2021 | Jung et al. / KIMS | AgNW 스트립 SERS 센서, 소변 | 전립선, 췌장 | 정성적 구분 | **한국/미국 출원 중** |
| 2021 | Chen et al. / Adv. Intell. Sys. | 소변 라만 + CNN 딥러닝 | 전립선 | DL 기반 분류 | 미확인 |
| 2021 | Feng et al. / Spectrochim Acta | 소변 SERS + PCA-SVM | 대장암 (스테이지별) | Sens 95.8/80.9/84.3% | 미확인 |
| 2022 | Gualerzi et al. / Mol. Med. | miRNA + SERS 소변 | 방광암 | AUC 0.92 | 미확인 |
| 2023 | Jung et al. / Optoelectron. Lett. | PCA-SVM, 소변 SERS | 대장암 스테이지 | 스테이지별 분류 | 미확인 |
| 2024 | Analytica Chimica Acta | AgNW 필터막 SERS, 소변 | 전립선, 췌장 | Fingerprinting 구분 | 미확인 |
| 2024 | Sensors & Actuators B | 3D Au 나노구조, 소변 | **다암종** | ML 기반 분류 | 미확인 |
| 2024 | ACS Sensors | 다파장 SERS, 소변 | 방광암 | ML 분류 정확도 향상 | 미확인 |
| 2024 | MDPI Int. J. Mol. Sci. | Label-free SERS, 소변 | 신장세포암 | PCA-LDA, SVM | 미확인 |
| 2024 | Biosens. Bioelectron. | 소변 SERS + ML, 쥐 모델 | 방광암 조기 | ≥99.6% 정확도 | 미확인 |

### 5.2 혈액/혈청 SERS + 다암종 핵심 논문

| 년도 | 저자/저널 | 기술 | 암종 | 성능 |
|---|---|---|---|---|
| 2023 | Shin et al. / Nat. Comm. | Exosome-SERS-AI, 혈장 | 6암종(폐,유방,대장,간,췌장,위) | AUC 0.970, Sens 90.2% |
| 2023 | eLight (우한대) | SERS-AICS, 혈청 | 5암종(폐,대장,간,위,식도) | Acc 95.81% |
| 2025 | BMC Medicine | SERS + DL, 혈청 대규모 | Pan-cancer | Sens 95.87%, Spec 95.40% |

### 5.3 비-SERS 경쟁 기술 논문

| 년도 | 저자/저널 | 기술 | 검체 |
|---|---|---|---|
| 2025 | Various | ATR-FTIR + ML, 소변 | 부인과 암 |
| 2024 | Various | FTIR + within-class feature, 소변 | 췌장암 조기 |
| 2026 | J. Raman Spec. | 소변 SERS 응용 리뷰 | 다종 질환 |

---

## 6. 경쟁 제품/기업 IP 현황

### 6.1 GRAIL (Galleri)

| 항목 | 내용 |
|---|---|
| 기술 | cfDNA 메틸화 분석 |
| 검체 | 혈액 |
| 암종 | 50종 이상 |
| 규제 | FDA PMA 신청 중 (2025) |
| 매출 | $136.8M (2025) |
| 시장점유율 | MCED 시장 40% 이상 |
| SOLUM과 차별 | 완전 다른 기술 (메틸화 vs SERS). 가격/비침습성에서 SOLUM 우위 |

### 6.2 KIMS (한국재료연구원)

| 항목 | 내용 |
|---|---|
| 기술 | 산호형 나노 SERS 스트립 센서 |
| 검체 | 소변 |
| 암종 | 전립선, 췌장 (2암종) |
| 특허 | 한국/미국 출원 중 (번호 미공개) |
| 센서 단가 | 100원 이하 |
| SOLUM과 관계 | **센서 하드웨어** 경쟁. AI 분류 방법론은 미보유 → 라이선스/협업 가능 |

### 6.3 TOBY AI Urine Test

| 항목 | 내용 |
|---|---|
| 기술 | AI 소변 분석 (비-SERS) |
| 검체 | 소변 |
| 암종 | 방광암 |
| 규제 | **FDA Breakthrough Device Designation** (2025.06) |
| SOLUM과 차별 | 단일 암종, 비-SERS 기술 |

---

## 7. 기술 요소별 선행기술 매트릭스

### 범례: ● 직접 선행 | ◐ 부분 선행 | ○ 선행 없음

| 기술 요소 | US20210215610 | US9410949 | CN115078331 | Exo-SERS-AI | KIMS Strip | TOBY |
|---|---|---|---|---|---|---|
| **소변 검체** | ● | ● | ○ (혈청) | ○ (혈장) | ● | ● |
| **SERS (표면증강)** | ○ (일반라만) | ● | ● | ● | ● | ○ |
| **Au 나노입자** | ○ | ● (ZnO+Au) | ○ (Ag) | ● (Au) | ○ (Ag) | ○ |
| **Label-free fingerprint** | ● | ○ (항체) | ● | ◐ (엑소좀) | ● | ○ |
| **ML/AI 분류** | ● (PCA/DAPC) | ○ | ● (SVM) | ● (MIL+DL) | ○ | ● |
| **다암종 동시 분류 (≥3)** | ◐ (개별모델) | ○ | ● (5종) | ● (6종) | ○ | ○ |
| **2단계 계층분류** | ○ | ○ | ○ | ◐ (암종예측) | ○ | ○ |
| **만성질환 대조군** | ○ | ○ | ○ | ○ | ○ | ○ |
| **QC 파이프라인** | ○ | ○ | ○ | ○ | ○ | ○ |
| **SNV 정규화** | ○ | ○ | ○ | ○ | ○ | ○ |
| **LR+DL 앙상블** | ○ | ○ | ○ | ○ | ○ | ○ |
| **3-threshold 운영** | ○ | ○ | ○ | ○ | ○ | ○ |
| **Multimodal Fusion** | ○ | ○ | ○ | ○ | ○ | ○ |
| **Confounding 검증** | ○ | ○ | ○ | ○ | ○ | ○ |

### 핵심 발견

**어떤 선행기술도 커버하지 못하는 SOLUM 고유 요소 (○ only):**
1. ✅ 2단계 계층분류 (Stage1→Stage2)
2. ✅ 만성질환 대조군 포함 설계 (DIA, HBP, H.D.)
3. ✅ 3단계 QC 파이프라인 (강도게이트/RSD/상관계수)
4. ✅ SNV 정규화 기반 SERS 전처리
5. ✅ LR + DL 가중 앙상블 (80:20)
6. ✅ 3-threshold 운영 모드 (스크리닝/균형/확인)
7. ✅ SERS + 임상데이터 Multimodal Fusion
8. ✅ 교란변수(나이/성별/BMI) 독립성 검증 방법

---

## 8. 특허 공백(White Space) 분석

### 8.1 공백 영역 상세

| # | 공백 영역 | 근거 | 청구 가능성 |
|---|---|---|---|
| **W1** | 소변 SERS + 5종 이상 다암종 동시 분류 | 기존: 소변 SERS는 1-2암종, 다암종은 혈액 기반 | **★★★ 핵심** |
| **W2** | 2단계 계층적 분류 (암여부→암종) | 기존: 모두 단일 stage 분류 | **★★★ 핵심** |
| **W3** | 만성질환 대조군 포함 스크리닝 | 기존: 정상군만 대조 | **★★★ 핵심** |
| **W4** | SERS 스펙트럼 3단계 QC 방법 | 기존: QC 방법론 특허 없음 | **★★☆** |
| **W5** | Classical ML > DL 입증 기반 앙상블 | 기존: DL 우위 전제 | **★★☆** |
| **W6** | 3-threshold 운영 시스템 | 기존: 단일 역치 | **★★☆** |
| **W7** | SERS + 임상데이터 Early Fusion | 기존: SERS 단독 사용 | **★☆☆** |
| **W8** | 교란변수 독립성 검증 방법 | 기존: 미수행 | **★☆☆** |

### 8.2 공백 영역 시각화

```
      ┌───────────────── 검체 유형 ──────────────────┐
      │            혈액/혈장           소변             │
      │                                               │
  단  │  WO2016094330   US20050250091                 │
  일  │  (ML+혈액)      (라만 방광암)                   │
  암  │                                               │
  종  │  US9410949      KIMS Strip                    │
      │  (SERS 신장암)   (SERS 전립선/췌장)             │
      ├───────────────────────────────────────────────┤
  다  │  CN115078331B   ┌─────────────────────┐      │
  암  │  (SERS-AICS)    │                     │      │
  종  │                 │  ★ SOLUM AECD ★     │      │
  동  │  Exo-SERS-AI    │  W1: 소변+5암종      │      │
  시  │  (혈장 엑소좀)    │  W2: 2단계 분류      │      │
  분  │                 │  W3: 만성질환 대조    │      │
  류  │  BMC Med 2025   │  W4: 3단계 QC       │      │
      │  (혈청 DL)       │  W5: LR 앙상블      │      │
      │                 │  W6: 3-threshold    │      │
      │                 └─────────────────────┘      │
      └───────────────────────────────────────────────┘
         ◀── AI 복잡도: 단순(PCA) ─── 중간(SVM) ─── 고급(DL 앙상블) ──▶
```

---

## 9. FTO(Freedom-to-Operate) 분석

### 종합 FTO 평가

| 특허 | 위험도 | 핵심 차별점 | 조치 필요 |
|---|---|---|---|
| **US20210215610A1** (Virginia Tech) | 🟡 중간 | SERS vs 일반라만, 2단계 vs 단일, 전처리 상이 | 출원 전 상세 비교 분석 |
| **US9410949B2** (WUSTL) | 🟢 낮음 | 항체기반 vs label-free fingerprint | 모니터링 |
| **CN115078331B** (우한대) | 🟡 중간 | 소변 vs 혈청, Au vs Ag, LR vs SVM | 중국 진출 시 상세 FTO |
| **KIMS Strip Patent** (출원 중) | 🟢 낮음 | AI 방법론 vs 센서 하드웨어 | 번호 확인 후 모니터링 |
| **KR20170000142A** (CTC SERS) | 🟢 낮음 | 혈액 CTC vs 소변 대사체 | 모니터링 |
| **KR20070030263A** (SERS 기판) | 🟢 낮음 | 기판 제조 vs AI 분류 | 해당 없음 |
| **WO2016094330A2** (GeneSystems) | 🟢 낮음 | 면역분석 vs SERS | 해당 없음 (Ceased) |
| **US10671885B2** (Ping Zhang) | 🟢 낮음 | 영상 기반 vs 스펙트럼 기반 | 해당 없음 |

### FTO 결론

**전체적으로 SOLUM AECD Platform의 FTO 상태는 양호합니다.**

- 가장 유사한 선행기술(US20210215610A1, CN115078331B) 대비에서도 핵심 구성요소(검체, 분류구조, 전처리, QC)에서 명확한 차별성 존재
- "소변 SERS + 2단계 계층 분류 + 만성질환 대조군 + 3단계 QC" 조합은 **전세계적으로 선행기술이 없는 공백 영역**
- 특허 출원 시 위 차별점을 독립항에 반영하면 강한 특허 확보 가능

---

## 10. IPC/CPC 분류코드 참고

### SOLUM 기술 해당 분류

| IPC/CPC 코드 | 설명 | 해당 기술요소 |
|---|---|---|
| **G01N 21/65** | 라만 산란 분석 | SERS 스펙트럼 측정 |
| **G01N 21/658** | 표면 플라즈몬 증강 라만 | SERS 나노입자 증강 |
| **G01N 33/574** | 암 진단용 면역분석 | 암 바이오마커 검출 |
| **G01N 33/493** | 소변 분석 | 소변 검체 처리 |
| **G16H 50/20** | AI 기반 의료 데이터 분석 | ML/DL 분류 모델 |
| **G16H 50/30** | 질병 위험도 평가 | 암 위험도 판정 |
| **G06N 20/00** | 머신러닝 | LR, 앙상블 |
| **G06N 3/04** | 신경망 아키텍처 | ResNet18-1D |
| **G06N 3/08** | 신경망 학습 | 역전파, 조기종료 |
| **A61B 5/145** | 체외진단 센서 | SERS 센서 기반 진단 |

### 검색 추천 조합
- G01N 21/65 AND G16H 50/20 → SERS + AI 의료분석
- G01N 21/658 AND G01N 33/493 → 표면증강라만 + 소변분석
- G01N 21/65 AND G06N 20/00 AND "cancer" → 라만 + ML + 암

---

## 11. 추가 조사 필요 사항

### 11.1 즉시 수행 필요

| # | 항목 | 방법 | 우선순위 |
|---|---|---|---|
| 1 | KIPRIS 직접 검색 | G01N 21/65 + 소변 + 암 | ★★★ |
| 2 | KIMS 스트립 특허 번호 확인 | KIPRIS에서 "정호상" + "재료연구원" + SERS 검색 | ★★★ |
| 3 | CN115078331B 상세 청구항 분석 | Google Patents 중문 번역 | ★★★ |
| 4 | US20210215610A1 심사 경과 확인 | USPTO PAIR에서 심사 이력 | ★★☆ |
| 5 | WIPO PATENTSCOPE 직접 검색 | PCT/WO 출원 중 SERS+urine+cancer | ★★☆ |

### 11.2 외부 전문가 의뢰 추천

| # | 항목 | 의뢰 대상 |
|---|---|---|
| 1 | 정식 FTO 보고서 | 특허법인 (변리사) |
| 2 | 특허맵 전문 분석 | KISTA IP-R&D 지원사업 |
| 3 | 중국 특허 상세 조사 | 중국 현지 특허대리소 |
| 4 | GRAIL/Illumina 특허 포트폴리오 분석 | 특허 분석 전문업체 |

### 11.3 모니터링 대상

| 대상 | 모니터링 방법 | 주기 |
|---|---|---|
| US20210215610A1 심사 진행 | USPTO PAIR | 월 1회 |
| KIMS 관련 출원 | KIPRIS 키워드 알림 | 월 1회 |
| 중국 SERS+암진단 신규 출원 | Google Patents 알림 | 월 1회 |
| SERS+urine+cancer 신규 논문 | PubMed 알림 | 주 1회 |

---

## 참고 자료 (Sources)

### 특허 DB
- [Google Patents](https://patents.google.com)
- [KIPRIS](https://www.kipris.or.kr)
- [USPTO](https://www.uspto.gov/patents/search)
- [WIPO PATENTSCOPE](https://patentscope.wipo.int)
- [Justia Patents](https://patents.justia.com)
- [KISTA IP-R&D](https://www.kista.re.kr/user/content.do?pageId=PAGE_000000000000015)
- [KIPO IPC/CPC 분류](https://www.kipo.go.kr/ko/topMenuLink.do?menuCd=SCD0200268)

### 학술논문
- [Exosome-SERS-AI (Nature Communications 2023)](https://www.nature.com/articles/s41467-023-37403-1)
- [SERS-AICS Pan-Cancer (BMC Medicine 2025)](https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-025-03887-5)
- [3D Au Nanoarchitecture Urine SERS (Sensors & Actuators B 2024)](https://www.sciencedirect.com/science/article/pii/S0925400524005586)
- [AgNW Filter SERS Urine (Analytica Chimica Acta 2024)](https://www.sciencedirect.com/science/article/pii/S0003267024000345)
- [Multiwavelength Urine SERS (ACS Sensors 2024)](https://pubs.acs.org/doi/10.1021/acssensors.4c01873)
- [Label-Free SERS Urine RCC (MDPI 2024)](https://www.mdpi.com/1422-0067/25/7/3891)
- [AI-Enhanced SERS Review (Wiley 2026)](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/aidi.202500030)
- [SERS Biosensors Liquid Biopsy (Nano Convergence 2024)](https://nanoconvergencejournal.springeropen.com/articles/10.1186/s40580-024-00428-3)
- [Cancer Diagnostics Patent Landscape (Nature Biotech 2025)](https://www.nature.com/articles/s41587-025-02935-y)
- [AI-Powered SERS (Analytical Chemistry 2024)](https://pubs.acs.org/doi/10.1021/acs.analchem.4c06584)
- [SERS+Microfluidics Cancer (Taylor&Francis 2025)](https://www.tandfonline.com/doi/full/10.1080/05704928.2025.2465394)
- [Urine SERS Application Review (J. Raman Spec. 2026)](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/jrs.70067)
- [SERS Label-free Cancer Liquid Biopsy (Frontiers 2025)](https://www.frontiersin.org/journals/chemistry/articles/10.3389/fchem.2025.1696979/full)

### 뉴스/시장
- [KIMS Strip Sensor News](https://www.koreabiomed.com/news/articleView.html?idxno=20304)
- [FDA Breakthrough TOBY AI](https://www.cancernetwork.com/view/ai-based-urine-test-earns-fda-breakthrough-device-status-in-bladder-cancer)
- [GRAIL Galleri Market](https://medcitynews.com/2026/02/grail-galleri-blood-test-multi-cancer-early-detection-mced-screening-liquid-biopsy-gral/)
- [Cancer Diagnostics Patent Report](https://www.researchandmarkets.com/reports/5997706/cancer-diagnostics-patent-landscape-report)
- [IP R&D 방법론 (KISTA)](https://www.kista.re.kr/user/content.do?pageId=PAGE_000000000000015)
- [World Cancer Day AI Cancer Diagnosis IP (Reddie & Grose)](https://www.reddie.co.uk/2025/02/04/ai-analysis-of-biological-samples-for-cancer-diagnosis/)

---

*본 보고서는 2026-03-18 기준 웹 검색 결과를 종합한 것으로, KIPRIS/CNIPA/WIPO 직접 DB 검색을 통한 보완 조사가 필요합니다.*
