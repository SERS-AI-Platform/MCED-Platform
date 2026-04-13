# SERS-AI 특허 포트폴리오 재구성 전략서

> **작성일:** 2026-04-06
> **용도:** 변리사 협의용 Discussion Document
> **버전:** v2.1 -- IP-2 폐기, IP-4~7 청구항 초안 추가
> **상태:** 내부 검토 완료, 변리사 미팅 전 사전 자료

---

## 1. 왜 재구성이 필요한가

기존 7건 발명신고서(A, C, D, E, F, G, M/SSI)는 기술 구상 단계에서 작성되어 **미검증 기술을 검증된 것처럼 기술**하고 있습니다.

### 1.1 주요 문제점

| 기존 발명신고서 | 문제 | 심각도 |
|---|---|---|
| **(D) MIL 품질 가중 통합** | 코드 구현 전무. 프로젝트 전체 검색에서 MIL 관련 코드 없음. | **치명적** |
| **(G) 대사체 클러스터 분류** | "enterotype vs protein metabolism" 클러스터링의 실험 증거 없음 | **치명적** |
| **(F) Blind NMF/MCR-ALS** | 구현은 있으나 분류 AUC 0.827로 production 수준(0.95+) 대비 현저히 부족 | 높음 |
| **(E) 고정 파수 그리드** | Grid 구현 완료, but cross-instrument 성능 AUC 0.48-0.62로 **실패** | 중간 |
| **(A) 시스템 특허** | "MIL 어텐션 가중 통합"이 핵심 기술로 기재되어 있으나 미구현 | 중간 |

### 1.2 이대로 출원할 경우의 위험

1. **실시 가능성(enablement) 요건 불충족**: (D) MIL, (G) 클러스터 분류의 실시예를 작성할 데이터가 없음
2. **명세서 허위 기재**: (E)에서 "성능 저하 없이 다기관 통합 가능"이라 기재하나, 실제 cross-instrument AUC는 무작위 수준
3. **무효 소송 취약**: 경쟁사가 실시예 재현 불가를 주장할 경우 방어 곤란
4. **투자 DD 리스크**: 기술-특허 괴리가 발견될 경우 기업 가치 평가에 부정적 영향

---

## 2. 기술별 검증 상태 (Evidence Audit)

### 2.1 검증 기준

- **코드 존재**: 해당 기술이 실제 코드로 구현되어 있는가?
- **실험 수행**: 실제 데이터로 학습/평가가 실행되었는가?
- **결과 보존**: 재현 가능한 결과 파일(metrics, predictions)이 존재하는가?
- **성능 유의**: 기존 방법 대비 통계적으로 유의미한 개선을 보이는가?

### 2.2 카테고리 A: 즉시 출원 가능 (코드 + 데이터 + 결과 완비)

| # | 기술 | 코드 위치 | 핵심 성능 수치 | 검증 Phase |
|---|---|---|---|---|
| T1 | **2-Stage 계층 분류** (Det + TypeID) | `models/train.py`, `models/model.py` | Det AUC 0.966, Type F1 0.914 | Phase CF-G (7-cancer) |
| T2 | **Sex Constraint** (생물학적 제약 마스킹) | `models/run_sersnet.py:125-146` | +4.0pp F1 (0.852 -> 0.892) | Phase Q (held-out) |
| T3 | **계층적 Late Fusion** (Stage1: SERS only, Stage2: SERS+clinical) | `models/run_train_val_test.py`, `models/clinical_utils.py` | +10.5pp F1 (0.818 -> 0.923) | Phase W (held-out) |
| T4 | **Multi-view Feature** (raw + 1st deriv + 2nd deriv) | `models/train_multichannel.py:68-94` | +4.6pp F1 (0.823 -> 0.869) | Phase MV-E2L |
| T5 | **3-Level QC Pipeline** (intensity gate + RSD + correlation) | `src/sers/qc/qc.py` | 전체 데이터셋 적용 완료 | Production |
| T6 | **Fixed 933-point Grid** (동일 기기 내 표준화) | `src/sers/preprocessing.py`, `config/config.yaml` | 동일 장비 내 재현성 확보 | Phase V |
| T7 | **SSI 점수 변환** (piecewise linear, 0-10) | `src/sers/scoring.py` | Rank-preserving, invertible (수학적 증명) | Production |
| T8 | **Mean Aggregation 우월성** | `models/train.py:409-418` | mean >> medoid (+14pp F1) | Phase B vs A |

### 2.3 카테고리 B: 추가 개발 후 출원 가능

| # | 기술 | 현재 상태 | 부족한 것 | 예상 소요 |
|---|---|---|---|---|
| T9 | **Cross-instrument 표준화** | Grid 구현 완료, AUC 0.48-0.62 (실패) | Domain adaptation / transfer calibration | 2-3개월 |
| T10 | **Peak-based 대사체 해석** (P-11) | 73종 참조 라이브러리 구축, cross-reference 코드 존재 | Production 통합, 분류 성능 기여 검증 | 1-2개월 |
| T11 | **Stacking Ensemble** (9 base models) | 코드 + 결과 존재 (`scripts/analysis/stacking_ensemble.py`) | LR 대비 유의미한 우위 불확실 (Type F1 0.877 vs LR 0.914) | 1개월 |
| T12 | **Contrastive Learning** | 코드 + 결과 존재 (`models/contrastive/`) | F1 0.66으로 LR(0.87+) 대비 열위 | 3개월+ |

### 2.4 카테고리 C: 폐기 또는 장기 보류

| # | 기술 | 기존 발명신고서 | 폐기/보류 사유 |
|---|---|---|---|
| T13 | **MIL 품질 가중 통합** | (D) | **코드 구현 전무**. 실제로는 mean aggregation만 사용. |
| T14 | **Blind NMF/MCR-ALS** | (F) | R²=0.885, 분류 AUC=0.827. LR(0.97+) 대비 현저히 낮음. |
| T15 | **대사체 클러스터 분류** | (G) | "enterotype vs protein metabolism" 분류 실험이 수행된 적 없음. |

---

## 3. 재구성된 특허 포트폴리오

### 3.1 즉시 출원: 2건 (2026년 4월 내 가출원)

---

#### IP-1: 계층적 다질환 스크리닝 시스템 (Umbrella Patent)

**기존 (A) 대비 변경:**
- "MIL 어텐션 가중 통합" 전면 삭제 -> mean aggregation으로 정정
- "품질 가중치 학습" 삭제 -> threshold-based QC filtering으로 정정
- 검증된 기술만으로 재구성

**독립항 (권고):**

> 생체 시료로부터 획득한 대사체 프로파일 데이터를 분석하는 방법으로서,
> (a) 복수 회 반복 측정된 스펙트럼 데이터에 대해 품질 기준에 따라 부적합 데이터를 제거하는 단계;
> (b) 적합 데이터를 통합하여 피검자별 대표 특징 벡터를 생성하는 단계;
> (c) 제1 분류 모델을 이용하여 질환 존재 여부를 판별하는 단계;
> (d) 양성 판별 시, 제2 분류 모델을 이용하여 k종(k >= 2) 질환 아형을 식별하는 단계;
> 를 포함하는 방법.

**종속항 그룹:**

| 종속항 | 내용 | 근거 기술 |
|---|---|---|
| 종속항 1 | 제1 분류기는 스펙트럼 데이터 단독 입력, 제2 분류기는 스펙트럼 + 임상 메타데이터 융합 입력 | T3 (Late Fusion) |
| 종속항 2 | 피검자의 생물학적 속성에 기초하여 제2 분류기의 출력 확률을 보정하는 단계를 더 포함 | T2 (Sex Constraint) |
| 종속항 3 | 대표 특징 벡터는 적합 스펙트럼의 산술 평균으로 생성 | T8 (Mean Agg) |
| 종속항 4 | 스펙트럼 데이터를 사전 정의된 N개 파수 지점 그리드로 보간하는 전처리 단계를 더 포함 | T6 (Grid) |
| 종속항 5 | 원시 스펙트럼, 1차 미분 스펙트럼, 2차 미분 스펙트럼을 연결(concatenate)하여 다중 뷰 특징을 구성 | T4 (Multi-view) |
| 종속항 6 | 대조군이 건강 정상군 및 만성질환군을 포함 | 데이터 구성 |
| 종속항 7 | k >= 5이고, 질환 아형이 전립선암, 유방암, 난소암, 폐암, 대장암, 췌장암, 방광암 중 복수를 포함 | 7-cancer 검증 |

**실시예 데이터:**
- 7암종, 1,628 subjects, 8,140 spectra, 5개 병원
- Det AUC: 0.966 (CV) / Type ID F1: 0.914 (CV, 3-view + clinical + sex constraint)
- Held-out test: Det AUC 0.979, Type F1 0.923 (Phase W)

**전략적 의미:**
- 가장 넓은 보호 범위. "대사체 프로파일 데이터"로 SERS에 한정되지 않음
- 종속항으로 IP-4~6 기술을 커버하되, 추후 분할출원(divisional) 가능
- 경쟁사 침해 탐지 용이: "2단계 계층 분류" 여부만 확인

---

#### ~~IP-2: 생물학적 제약 기반 확률 보정 방법~~ — **폐기**

**폐기 사유:** 성별에 따라 불가능한 암종의 확률을 0으로 마스킹하는 것은 기술적으로 자명(trivial)하여 독립 특허로서의 가치가 부족. IP-1의 종속항 2에서 커버하는 것으로 충분.

**기존 초안:** `docs/patent/PATENT_02_biological_constraint_probability_correction.md` — 참고용 보관

---

#### IP-3: SSI (SERS Screening Index) 임상 점수 변환

**코드:** `src/sers/scoring.py` — 완전한 production 코드 존재

**독립항 (권고):**

> AI 분류 모델의 질환 확률 출력(0 ~ 1)을 임상 점수(0 ~ N)로 변환하는 방법으로서,
> (a) 분류 모델의 운영 임계값(operating threshold)을 수신하는 단계;
> (b) 상기 운영 임계값이 사전 정의된 고정 cutoff 점수에 매핑되도록 구간별 선형 변환(piecewise linear transformation)을 정의하는 단계;
> (c) 질환 확률을 상기 변환에 의해 0 ~ N 범위의 임상 점수로 변환하는 단계;
> 를 포함하되, 상기 변환은 단조 증가(monotonic) 및 순위 보존(rank-preserving)인 점수 변환 방법.

**종속항:**
1. N = 10, cutoff = 4.0 (OVA1 precedent)
2. 5단계 위험도 층화 (LOW / LOW_MODERATE / MODERATE / HIGH / VERY_HIGH)
3. 역변환 함수 제공으로 원래 확률 복원 가능
4. 3종 운영 모드(Screening/Balanced/Confirmation)에서 임계값만 변경하되 동일 변환 함수 사용

**실시예 데이터:**
- 수학적 증명: monotonic, invertible, rank-preserving -> AUC/Sens/Spec 동등
- FDA 510(k) Predicate: OVA1 (Vermillion, 0-10 복합지수, 난소암)

**전략적 의미:**
- MFDS 인허가 시 직접 활용 (점수 해석 체계)
- 분류 방법과 독립적인 post-processing -> 라이선싱 단독 가치
- OVA1 predicate 전략의 핵심 요소

---

### 3.2 추가 개발 후 출원: 4건 (2026년 5-6월)

#### IP-4: 3단계 자동 QC Pipeline

**현재 상태:** `src/sers/qc/qc.py`에 완전 구현. Level 0 (intensity gate) + Level 1 (RSD < 5%) + Level 1 (correlation > 0.95).

**추가 필요:**
- [ ] 의도적 불량 검체(노이즈 주입, 기포, 기판 불량) 포함 시 QC 탈락 시연 데이터 생성
- [ ] QC 적용/미적용 시 분류 성능 차이 정량화 (현재 QC 이슈로 인해 비교 데이터 미정리)
- [ ] 타 기관(Medical instrument) 데이터에서 QC 효과 별도 검증

**예상 소요:** 2-3주

**독립항 초안:**

> 분광 분석 기반 진단 시스템에서 반복 측정 스펙트럼의 품질을 자동으로 평가하는 방법으로서,
> (a) 동일 시료에 대해 복수 회(n ≥ 2) 반복 측정된 스펙트럼 데이터를 수신하는 단계;
> (b) 각 스펙트럼의 총 신호 강도가 사전 정의된 하한 임계값 이상인지 판정하여 무효 스펙트럼을 제거하는 제1 품질 검사 단계;
> (c) 상기 제1 품질 검사를 통과한 스펙트럼들 간의 파수별 상대표준편차(RSD)를 산출하고, 상기 RSD가 사전 정의된 상한 이하인 경우에만 적합으로 판정하는 제2 품질 검사 단계;
> (d) 상기 제2 품질 검사를 통과한 스펙트럼들 간의 쌍별 상관계수를 산출하고, 최소 상관계수가 사전 정의된 하한 이상인 경우에만 적합으로 판정하는 제3 품질 검사 단계;
> (e) 상기 제1 내지 제3 품질 검사를 모두 통과한 스펙트럼만을 후속 분석에 사용하는 단계;
> 를 포함하는 방법.

**종속항 후보:**
1. 제1 품질 검사의 하한 임계값은 동일 기판에서 측정된 기준 스펙트럼의 중앙값 대비 비율로 정의
2. 제2 품질 검사의 RSD 상한은 5%이고, 제3 품질 검사의 상관계수 하한은 0.95
3. 부적합 판정 시 해당 시료의 재측정을 자동 요청하는 단계를 더 포함
4. 상기 분광 분석은 SERS(표면증강라만분광)이고, 반복 측정은 동일 기판의 상이한 위치에서 수행

---

#### IP-5: 계층적 Multimodal Late Fusion

**현재 상태:** `models/run_train_val_test.py`에 구현 완료. Phase W에서 +10.5pp 검증. Cross-attention (`models/cross_attention/`) 및 FiLM (`models/film/`) 변형도 구현 완료.

**추가 필요:**
- [ ] Early fusion vs Late fusion 체계적 비교표 정리
- [ ] FiLM / Cross-Attention 실험 결과 정리 (DLF-1~6 결과 존재하나 비교 분석 미완)
- [ ] Stage 1에서 clinical을 제외하는 것이 confounding 방지에 효과적이라는 분석 보강

**예상 소요:** 2-3주

**참고:** IP-1의 종속항 1에서도 커버되지만, 독립 출원 시 "계층별로 서로 다른 모달리티를 사용하는 전략" 자체에 대한 넓은 청구 가능.

**독립항 초안:**

> 복수 종류의 데이터 모달리티를 이용하여 질환을 분류하는 방법으로서,
> (a) 제1 모달리티 데이터(분광 데이터)를 입력으로 하는 제1 분류 모델을 이용하여 질환 유무를 판별하는 단계;
> (b) 상기 제1 분류 모델에서 양성으로 판별된 피검자에 대해, 상기 제1 모달리티 데이터와 제2 모달리티 데이터(임상 메타데이터)를 결합한 융합 입력을 생성하는 단계;
> (c) 상기 융합 입력을 제2 분류 모델에 입력하여 질환 아형을 식별하는 단계;
> 를 포함하되, 상기 제1 분류 모델은 제1 모달리티 단독으로 학습되고, 상기 제2 분류 모델은 제1 및 제2 모달리티의 융합으로 학습되는 계층적 다중모달 분류 방법.

**종속항 후보:**
1. 제2 모달리티 데이터는 연령, 성별, BMI 중 하나 이상을 포함하는 임상 정보
2. 제1 분류 모델의 학습 시 제2 모달리티를 의도적으로 배제함으로써 교란 변수(confounding factor)에 의한 과적합을 방지
3. 융합 입력은 제1 모달리티의 특징 벡터와 제2 모달리티의 특징 벡터를 연결(concatenate)하여 생성
4. 제2 분류 모델은 cross-attention 메커니즘 또는 FiLM(Feature-wise Linear Modulation)을 이용하여 모달리티 간 상호작용을 학습
5. 제1 모달리티는 SERS 스펙트럼이고, 제2 모달리티는 전자의무기록(EMR)에서 추출된 임상 변수

---

#### IP-6: Multi-view Spectral Feature Representation

**현재 상태:** `models/train_multichannel.py`에 구현. raw + 1st derivative + 2nd derivative concatenation으로 +4.6pp F1.

**추가 필요:**
- [ ] 개별 view ablation 결과 체계적 정리 (Phase FR-1, FR-2, MV-E0~E2 데이터 존재)
- [ ] Derivative가 왜 도움이 되는지 분광학적 근거 보강 (피크 위치 vs 피크 형상 정보 분리)
- [ ] 4-view 이상 (peak ratio 추가) 확장 실험

**예상 소요:** 2-3주

**독립항 초안:**

> 분광 스펙트럼 데이터로부터 질환 분류를 위한 특징 벡터를 생성하는 방법으로서,
> (a) 원시 스펙트럼 데이터를 제1 뷰로 획득하는 단계;
> (b) 상기 원시 스펙트럼 데이터의 파수에 대한 1차 미분을 산출하여 제2 뷰를 생성하는 단계;
> (c) 상기 원시 스펙트럼 데이터의 파수에 대한 2차 미분을 산출하여 제3 뷰를 생성하는 단계;
> (d) 상기 제1 뷰, 제2 뷰, 및 제3 뷰를 연결(concatenate)하여 다중 뷰 특징 벡터를 구성하는 단계;
> (e) 상기 다중 뷰 특징 벡터를 분류 모델에 입력하여 질환을 분류하는 단계;
> 를 포함하되, 상기 제1 뷰는 피크 강도 정보를, 상기 제2 뷰는 피크 위치 및 기울기 정보를, 상기 제3 뷰는 피크 형상 및 곡률 정보를 각각 인코딩하는 다중 뷰 스펙트럼 특징 생성 방법.

**종속항 후보:**
1. 상기 다중 뷰 특징 벡터에 피크 피팅(peak fitting)으로부터 추출된 피크 파라미터(위치, 높이, 폭)를 추가 뷰로 연결하는 단계를 더 포함
2. 각 뷰에 대해 독립적으로 정규화(normalization)를 수행한 후 연결
3. 상기 분광 스펙트럼은 SERS 스펙트럼, IR 흡수 스펙트럼, 또는 형광 스펙트럼 중 하나
4. 상기 분류 모델은 선형 모델이며, 상기 다중 뷰 구성에 의해 비선형 스펙트럼 특성이 선형 분류 가능한 특징 공간으로 변환

---

#### IP-7: Peak-based 대사체 해석 하이브리드

**현재 상태:** `docs/patent/PATENT_11_peak_metabolite_interpretation.md` 초안 존재. 73종 실측 참조 라이브러리 구축 완료. Cross-reference 코드 (`metabolite_profiling/cross_reference_peaks.py`) 존재.

**추가 필요:**
- [ ] Production pipeline 통합 (`scripts/deployment/sers_predict.py`에 해석 모듈 추가)
- [ ] Peak ratio 바이오마커의 암종별 유의성 검증 (통계 검정)
- [ ] 참조 라이브러리 매칭의 정확도 정량화

**예상 소요:** 1-2개월

**전략적 의미:** MFDS/FDA의 AI 해석가능성 요구사항에 직접 대응. 73종 실측(문헌이 아닌) 참조 라이브러리는 상당한 IP 자산.

**독립항 초안:**

> AI 분류 모델의 질환 판별 결과에 대해 분자 수준의 해석을 제공하는 방법으로서,
> (a) 피검자의 분광 스펙트럼에서 통계적으로 유의미한 감별 피크(discriminative peak)를 식별하는 단계;
> (b) 실측 표준물질의 스펙트럼으로 구축된 참조 라이브러리를 이용하여, 상기 감별 피크를 대응하는 대사체(metabolite)에 매핑하는 단계;
> (c) 상기 매핑된 대사체 정보와 상기 AI 분류 모델의 특징 중요도(feature importance)를 결합하여, 질환 판별에 기여한 주요 대사 경로를 보고하는 단계;
> 를 포함하는 AI 분류 결과의 분자 해석 방법.

**종속항 후보:**
1. 참조 라이브러리는 N종(N ≥ 50) 이상의 표준물질에 대해 동일 분석 조건에서 실측한 스펙트럼으로 구성
2. 감별 피크 식별은 질환군과 대조군 간 통계 검정(t-test, Mann-Whitney U 등) 및 다중비교 보정을 포함
3. 주요 대사 경로 보고는 KEGG, HMDB 등 공개 대사체 데이터베이스와의 교차 참조를 포함
4. 피크 강도 비율(peak ratio)을 바이오마커로 산출하고, 상기 비율의 질환군 간 차이를 시각화하여 제공
5. 상기 방법은 분류 모델의 재학습 없이 추론 결과에 대한 사후(post-hoc) 해석으로 적용

---

### 3.3 폐기 또는 보류: 3건

#### 기존 (D) MIL 품질 가중 통합 — 폐기

**사유:** 프로젝트 전체를 `MIL`, `multiple_instance`, `attention_pool` 등으로 검색한 결과 구현 코드가 없음. `models/train.py`의 aggregation 함수는 `aggregate_mean()`과 `aggregate_medoid()`만 존재.

**변리사 안내:** 이 발명신고서의 핵심 청구("MIL 어텐션으로 반복 스펙트럼의 품질 가중치를 학습하여 통합")가 미구현 상태입니다. 실시예를 작성할 수 없으므로 출원하지 않는 것을 권고합니다. 추후 MIL을 실제 구현하고, mean aggregation 대비 통계적으로 유의미한 개선이 확인되면 새 출원을 검토합니다.

---

#### 기존 (F) Blind NMF/MCR-ALS — 보류

**사유:**
- NMF 코드 존재 (`metabolite_profiling/experiments/`)
- 복원 정확도: R² = 0.885 (blind MCR-ALS, k=5)  /  R² = 0.965 (NMF Frobenius, k=30)
- 분류 성능: AUC = 0.827 (blind MCR) — LR(0.97+) 대비 현저히 낮음
- Semi-supervised NMF(참조 스펙트럼 초기화)가 blind보다 우수 -> "blind" 주장의 강점 약화

**재평가 조건:** Blind 분해 성분을 LR 특징으로 추가했을 때 Type ID F1이 유의미하게(>= 2pp) 향상되면 재검토.

---

#### 기존 (G) 대사체 클러스터 분류 — 폐기

**사유:** "장내미생물체 우세형 vs 단백질 대사 우세형" 클러스터링을 기술하나, 이를 수행한 실험이 존재하지 않음. 대사체 프로파일링 결과(`metabolite_profiling/`)에서 스펙트럼 유사도 분석만 수행되었고, 환자 수준 클러스터 분류는 시도된 적 없음.

---

## 4. 출원 구조 전략

### 4.1 권고 구조

```
[IP-1] 계층적 다질환 스크리닝 시스템 ──── Umbrella Patent (가장 넓은 범위)
  |
  ├── 종속항 → T3 (Late Fusion) ←─── 추후 IP-5로 분할출원 가능
  ├── 종속항 → T2 (Sex Constraint) ── IP-1 종속항으로만 유지 (독립출원 가치 없음)
  ├── 종속항 → T4 (Multi-view) ←───── 추후 IP-6으로 분할출원 가능
  ├── 종속항 → T6 (Grid) ──────────── 동일 기기 내 표준화로 한정
  └── 종속항 → T8 (Mean Agg)

[IP-3] SSI 점수 변환 ───────────────── 독립 출원 (규제 대응)
```

### 4.2 Divisional/CIP 전략

IP-1 명세서에 **미청구 기술을 상세히 기재**하여 추후 분할출원 기반 확보:

| 명세서 기재 (청구 안 함) | 추후 분할출원 대상 |
|---|---|
| QC pipeline 상세 알고리즘 | IP-4 |
| Early vs Late fusion 비교 | IP-5 |
| Multi-view (raw + d1 + d2) 상세 | IP-6 |
| Peak-based 해석 개념 | IP-7 |

### 4.3 IP-3을 분리하는 이유

- **IP-3 (SSI):** 분류 방법과 무관한 post-processing. FDA 510(k) predicate (OVA1) 대응. 규제 전략의 핵심.
- **IP-2 (Sex Constraint) 폐기 사유:** 성별 기반 확률 마스킹은 기술적으로 자명하여 독립 출원 가치 없음. IP-1 종속항 2로 충분히 커버.

---

## 5. 기존 발명신고서 -> 재구성 매핑

| 기존 | 상태 | 재구성 | 비고 |
|---|---|---|---|
| **(A)** 다중암 스크리닝 시스템 | 수정 필요 | **IP-1** (umbrella) | MIL 삭제, mean agg 정정 |
| **(B)** 성별 분류 최적화 | **폐기** (독립출원 가치 없음) | IP-1 종속항 2 | 기술 자명, IP-1 종속항으로 충분 |
| **(C)** Multimodal fusion | 유효, 과도한 청구 수정 | **IP-5** 또는 IP-1 종속항 | 추가 정리 후 출원 |
| **(D)** MIL 품질 가중 통합 | **폐기** | -- | 코드 미구현 |
| **(E)** 고정 파수 그리드 | 부분 유효 | IP-1 종속항 | cross-instrument 주장 삭제 |
| **(F)** Blind NMF/MCR-ALS | **보류** | -- | 성능 부족 |
| **(G)** 대사체 클러스터 분류 | **폐기** | -- | 증거 없음 |
| **(M/SSI)** SSI 점수 변환 | 유효 | **IP-3** (독립 출원) | 즉시 출원 가능 |

**주목:** 기존에 "(B) 성별 기반 분류 최적화"는 "후처리 트릭 수준"으로 제거되었으나, 실험 검증 결과 +4.0pp F1의 유의미한 효과가 있으며, 재학습 없이 적용 가능한 범용 방법이므로 **독립 출원 가치가 충분**합니다.

---

## 6. MFDS 규제 타임라인 연계

| 규제 마일스톤 | 필요 특허 | 시기 |
|---|---|---|
| 혁신의료기기 지정 신청 | IP-1 출원 사실 필요 | 2026 Q2 |
| MFDS 인허가 심사 (AI 해석가능성) | IP-7 (peak-based 해석) | 2026 Q3 |
| 다기관 임상시험 개시 (보라매병원) | IP-1 + QC (IP-4) | 2026 Q3 |
| FDA 510(k) Predicate 비교 | IP-3 (SSI - OVA1 대응) | 2027 |
| 단일암 MFDS 허가 (전립선암) | IP-1 | 2027 |

---

## 7. 변리사 미팅 시 확인 사항

### 7.1 청구 범위 관련

1. **IP-1 독립항의 범위**: "대사체 프로파일 데이터"로 넓게 잡아서 SERS에 한정하지 않는 전략이 유효한지? 기존 SERS 특허(등록 3건)와의 중복 범위는?
2. **IP-2의 범용성**: "다중 질환 분류기"에 대한 확률 보정 방법으로 출원 시, SERS 외 cfDNA/단백질 기반 classifier에도 적용 주장 가능한지?
3. **(E) 고정 그리드의 청구 범위**: "동일 기기 내 배치 표준화"로 한정해도 특허 가치가 있는지? Cross-instrument 주장을 완전히 삭제해야 하는지?

### 7.2 출원 전략 관련

4. **미국 Provisional 구성**: IP-1, IP-2, IP-3을 각각 별도 Provisional vs 하나의 Provisional에 3건 포함?
5. **IP-1 명세서에 미청구 기술 기재 범위**: Divisional/CIP 대비 어디까지 상세히 기재해야 하는지?
6. **국내 조기심사**: SOLUM Healthcare 벤처기업 요건 충족 여부, 조기심사 대상 해당 여부

### 7.3 리스크 관련

7. **Cross-instrument 실패 공개**: IP-1 명세서에 "그리드 표준화만으로는 장비 간 호환이 불충분하며 추가 보정이 필요함"을 기재하는 것이 바람직한지?
8. **기존 등록 3건과의 관계**: 신규 출원 시 자기 충돌(self-collision) 검토 필요
9. **(D), (G) 폐기 시**: 기존 발명신고서가 외부에 공개된 적 있는지? 공개되었다면 novelty에 영향?

### 7.4 타이밍 관련

10. **P-11 (Peak-based 해석)**: 73종 참조 라이브러리만으로 선출원 vs production 통합 후 출원? 참조 라이브러리 자체의 특허성은?
11. **AACR 포스터 발표 (2026-04)**: 포스터 내용이 공개되면 grace period 시작. 가출원 시기와의 관계는?

---

## 8. 실행 일정

```
2026-04 W2              2026-04 W3-4            2026-05              2026-06
    |                       |                       |                    |
    |--- 변리사 미팅 --------|                       |                    |
    |    (본 문서 기반)       |                       |                    |
    |                       |--- IP-1 가출원 --------|                    |
    |                       |--- IP-3 가출원 --------|                    |
    |                       |                       |--- IP-4 (QC) ------|
    |                       |                       |--- IP-5 (Fusion) --|
    |                       |                       |                    |--- IP-6 (Multi-view)
    |                       |                       |                    |--- IP-7 (Peak 해석)
```

---

## 9. 핵심 원칙

> **검증된 것만 출원한다.**
> **미검증 기술은 구현 + 검증 후 새로 출원한다.**
> **명세서에는 실제 실험 결과만 기재한다.**

---

## 부록: 검증 데이터 요약

### A. 데이터셋 규모

| 항목 | 수치 |
|---|---|
| 피검자 수 | 1,628명 (clean cohort) / 1,700명 (전체) |
| 스펙트럼 수 | ~8,140 (clean) / ~8,500 (전체) |
| 암종 수 | 7 (PRO, BRE, OVA, LUN, CRC, PAN, BLC) |
| 대조군 | 4 (NOR, DIA, HBP, H.D.) |
| 수집 기관 | 5개 병원 |
| 반복 측정 | 환자당 5회 |

### B. Best 성능 (Phase CF-G, 2026-04-03)

| 지표 | 수치 | 조건 |
|---|---|---|
| Detection AUC | 0.9662 | 7-cancer, 5-fold CV, mean agg |
| Type ID F1 | 0.9136 +/- 0.023 | 3-view + clinical + sex constraint |
| 입력 특징 | 1,948개 | full spectrum(935) + 1st deriv(935) + Voigt peaks(75) + clinical(3) |
| 모델 | Logistic Regression | 해석 가능, 규제 적합 |

### C. Held-out Test (Phase W, 2026-03-23)

| 지표 | SERS-only | Fusion (SERS + clinical) |
|---|---|---|
| Detection AUC | 0.929 | 0.979 |
| Type ID F1 | 0.756 | 0.923 |
| 조건 | 60/20/20 split, 5 repeats | + age/sex/BMI + sex constraint |

### D. 실험 이력

- **총 실험 Phase 수:** 75+ (Phase A ~ CF-H)
- **총 MLflow 기록:** 1 run (나머지는 파일 시스템 기반 추적)
- **최신 실험:** 2026-04-06 (weekend experiments)
- **주력 모델:** Logistic Regression (DL 대비 Type ID F1이 일관되게 우수)

---

*작성: SOLUM Healthcare AI팀*
*문서 버전: v2.0*
*다음 업데이트: 변리사 미팅 후 피드백 반영*
