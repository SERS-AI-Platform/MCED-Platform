# 보라매병원 전립선 생검 음성군 라벨링 기준

> **Memory decision, 2026-07-11:** 보라매 전향검체의 논문·그림·모델 class label은 `Biopsy-negative`로 고정한다. `PNB`로 변경하지 않는다.

## 결정

| 사용 위치 | 표준 표현 |
|---|---|
| Figure, table, model class label | `Biopsy-negative` |
| Methods 최초 정의 | `Men with elevated PSA and a histologically negative prostate biopsy (biopsy-negative group)` |
| 한국어 설명 | `PSA 상승·전립선 생검 음성군` 또는 `생검 음성군` |
| 원본 Excel 입력값 | `Elevated PSA, biopsy-negative (PSA↑/Bx−)`를 그대로 보존하고 분석 경계에서 `Biopsy-negative`로 정규화 |

`PSA/Bx-`와 `PSA-Bx-`는 내부 메모 형태에 가깝고 논문 라벨로 사용하지 않는다. `PNB`와 단독 `BN`도 사용하지 않는다.

## 문헌 조사 결과

1. 전립선암 소변·대사체 바이오마커 연구에서는 비교군을 `biopsy-negative controls`, `biopsy-negative patients`, 또는 `negative prostate biopsy`로 직접 표현하는 사례가 반복된다.
2. `PNB`는 문헌에서 `previous negative biopsy`를 뜻한다. 이는 과거 음성 생검 후 재생검을 받는 환자군이며, 현재 생검 결과가 음성인 보라매군과 같은 정의가 아니다.
3. `PNB`는 다른 병리 문헌에서 `prostate needle biopsy`를 뜻하기도 하므로 군 라벨로 사용하면 중의적이다.
4. `BN`은 일부 연구에서 `biopsy negative`로 정의되지만, 다른 전립선 생검 연구에서는 `biopsy-naïve`의 약어로 사용된다. 그림만 따로 볼 때 오해될 수 있어 약어 없는 `Biopsy-negative`가 가장 명확하다.
5. `NED`는 `no evidence of disease` 또는 `no evidence of prostate cancer/high-grade PIN on biopsy`처럼 연구별 정의가 필요하다. 단일 음성 생검만으로 암이 완전히 없다고 확정할 수 없으므로 현재 cohort의 기본 라벨로 사용하지 않는다.

## 2026-07-11 문헌 재확인

- `PNB`는 연구 문맥에서 주로 `previous negative biopsy`를 의미한다. 즉, 과거 음성 생검 이후에도 임상적 의심이 남아 재평가 또는 재생검을 받는 cohort를 가리키며, 단순히 현재 생검 결과가 음성이라는 뜻으로 사용할 수 없다. [Mischinger et al., Frontiers in Surgery 2022](https://doi.org/10.3389/fsurg.2022.1013389)은 `biopsy-naive (BN)`와 `previous-negative biopsy (PNB)`를 별도 cohort로 정의한다.
- 전립선암 소변·대사체 연구에서는 `biopsy-negative controls`, `biopsy-negative men`, `negative prostate biopsy`처럼 생검 결과를 직접 기술하는 표현이 사용된다. 따라서 본 데이터의 공개 라벨에는 약어보다 `Biopsy-negative`가 명확하다.
- 음성 생검 뒤에도 발견되지 않은 암이 남을 수 있으므로 `healthy`, `normal`, `cancer-free`, `NED`로 확장 해석하지 않는다. 음성 생검 cohort의 후속 암 위험을 다룬 연구도 이 집단을 `negative baseline biopsy`로 기술한다. [Moreira et al., Prostate Cancer and Prostatic Diseases 2016](https://doi.org/10.1038/pcan.2015.66)

### 최종 라벨 규칙

| 산출물 | 사용할 표현 |
|---|---|
| Figure / confusion matrix / model class | `Biopsy-negative` |
| 첫 Methods 정의 | `Men with elevated PSA and a histologically negative prostate biopsy (Biopsy-negative group)` |
| Figure legend의 설명 | `Biopsy-negative (elevated PSA, negative prostate biopsy)` |
| 한국어 발표자료 | `PSA 상승·전립선 생검 음성군` |
| `PNB` 사용 조건 | 해당 환자들이 실제로 과거 음성 생검 후 재생검·재평가 cohort라는 사실이 확인된 경우에만 사용 |

## 적용 원칙

- `Control`과 `Biopsy-negative`는 별도 군으로 유지한다.
- Cancer Screening에서는 두 군을 non-cancer로 합칠 수 있지만, Methods와 Figure legend에 구성 인원과 생검 음성 정의를 명시한다.
- 3-group 분석은 `Control / Biopsy-negative / Prostate cancer` 순서를 사용한다.
- 생검 음성은 암 부재를 완전히 보장하지 않으므로 `healthy`, `normal`, `cancer-free`로 치환하지 않는다.
- 과거 음성 생검 후 재생검 cohort임이 확인된 경우에만 `previous negative biopsy`를 사용한다.

## 근거 문헌

- Sreekumar A, et al. *Metabolomic profiles delineate potential role for sarcosine in prostate cancer progression*. Nature. 2009. Figure와 본문에서 `prostate biopsy negative controls`를 사용한다. https://pmc.ncbi.nlm.nih.gov/articles/PMC2724746/
- Theodorescu D, et al. *Discovery and validation of urinary biomarkers for prostate cancer*. Proteomics Clin Appl. 2009. `negative prostate biopsy`, `biopsy-negative men`을 사용하며, 별도 정의가 있을 때 `NED`를 사용한다. https://pmc.ncbi.nlm.nih.gov/articles/PMC2744126/
- Barocas DA, et al. *Oxidative stress measured by urine F2-isoprostane level is associated with prostate cancer*. J Urol. 2011. `biopsy-negative controls`를 사용한다. https://pmc.ncbi.nlm.nih.gov/articles/PMC3093434/
- Mischinger J, et al. *Combining targeted and systematic prostate biopsy...*. Front Surg. 2022. `PNB`를 `previous negative biopsy`로 명시한다. https://pubmed.ncbi.nlm.nih.gov/36277287/
- Bonkhoff H. *Significance of prostate cancer missed on needle biopsy...*. Prostate. 2016. `PNB`를 `prostate needle biopsy`로 사용한다. https://pubmed.ncbi.nlm.nih.gov/26616257/

## 해석 주의

현재 47명의 원본 정의는 `Elevated PSA, biopsy-negative`이다. 생검 시점, 생검 방식, core 수, mpMRI 및 추적 생검 여부가 모두 확인된 것은 아니므로 `previous negative biopsy`, `benign biopsy`, `NED`보다 관찰 사실에 가까운 `Biopsy-negative`를 표준 라벨로 채택한다.
