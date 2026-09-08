# SOLUM Healthcare Analysis Presentation Design System

## 1. Purpose and authority

이 문서는 SERS-AI에서 새로 만드는 모든 분석·기술·의사결정 장표의 시각적 기준이다. 결과물은 PowerPoint를 직접 생성하는 대신 HTML/CSS로 작성하고 브라우저 또는 PDF로 렌더링한다.

- 기준 자료: `C:\Users\user\Downloads\시스템 환경 구성 협의 결과 및 확인 요청.pptx`
- 기준 자료에서 확인한 범위: 8개 슬라이드, 16:9, SOLUM Healthcare 기업 장표
- 적용 대상: 분석 결과, 모델 성능, 데이터 흐름, 시스템 아키텍처, 보안 협의, 경영진 보고
- 비적용 대상: 임상 운영 웹 애플리케이션 UI. 해당 화면은 저장소 루트의 `DESIGN.md`를 따른다.
- 충돌 시 우선순위: 데이터 정확성 및 필수 고지 > 이 문서 > 개별 장표의 장식적 요구

## 2. Visual identity

장표는 `clinical precision`과 `corporate restraint`를 지향한다. 흰 배경 위에 딥네이비 구조선을 사용하고, 한 장표에 하나의 결론만 분명하게 전달한다. 장식보다 정보 계층, 수치, 흐름, 비교가 먼저 보여야 한다.

핵심 인상은 다음과 같다.

- 넓은 흰 여백
- 상단의 강한 네이비 챕터 바
- 좌측의 큰 두 자리 섹션 번호
- 좌측 정렬 제목과 한 줄 결론
- 네이비 중심의 제한된 색상
- 얇고 정확한 선, 직사각형 카드, 절제된 모서리
- 도표와 수치가 중심이며 장식 이미지는 최소화

## 3. Canvas and coordinate system

모든 장표는 16:9 고정 캔버스를 사용한다.

| 항목 | 기준 |
|---|---:|
| 논리 캔버스 | `1600 × 900px` |
| 출력 비율 | `16:9` |
| 인쇄 크기 | `13.333in × 7.5in` |
| 안전 여백 | 좌우 `64px`, 하단 `40px` |
| 일반 본문 시작점 | `y = 168px` |
| 챕터 헤더 높이 | `152px` |
| 기본 컬럼 간격 | `32px` |
| 기본 요소 간격 | `16px`, `24px`, `32px` |

HTML 미리보기는 캔버스 자체를 반응형으로 재배치하지 않는다. 브라우저 폭에 맞춰 장표 전체를 비례 축소한다. PDF와 이미지 렌더링에서도 요소의 위치와 줄바꿈이 동일해야 한다.

## 4. Color system

### 4.1 Core tokens

| Token | Value | Use |
|---|---|---|
| `--navy-950` | `#001F3C` | 표지 배경, 최상위 브랜드 면 |
| `--navy-800` | `#0A306D` | 챕터 바, 주요 블록, 화살표, 핵심 텍스트 |
| `--navy-700` | `#123B78` | 보조 네이비 면 |
| `--ink-900` | `#111827` | 제목과 본문 |
| `--ink-700` | `#333333` | 일반 설명 |
| `--ink-500` | `#6B7280` | 캡션과 보조 정보 |
| `--line-300` | `#D6E0EF` | 카드와 표의 경계선 |
| `--surface-100` | `#F4F6FA` | KPI 카드와 옅은 보조 면 |
| `--surface-200` | `#E0E4ED` | 표의 교차 행과 구획 |
| `--white` | `#FFFFFF` | 기본 배경과 네이비 위 글자 |

### 4.2 Analytical accents

분석 강조색은 의미가 있을 때만 사용한다.

| Token | Value | Meaning |
|---|---|---|
| `--purple-700` | `#391D8D` | Cancer Screening, 첫 번째 모델 단계, AUC |
| `--purple-100` | `#E0DDF0` | 보라 KPI 카드 경계 또는 옅은 면 |
| `--green-700` | `#029567` | Cancer Type ID, 두 번째 모델 단계, Macro F1 |
| `--green-100` | `#D0EDE4` | 초록 KPI 카드 경계 또는 옅은 면 |
| `--warning-600` | `#B7791F` | 검토 필요, 제한, 주의 |
| `--danger-700` | `#B42318` | 오류와 실패. 성능 저하를 과장하는 장식에는 사용 금지 |

한 장표에서 브랜드 네이비를 제외한 의미색은 최대 2개만 사용한다. 범주형 차트에서 여러 색상이 필요한 경우에도 색상은 범주 구분에만 사용하고 제목, 축, 카드에는 반복하지 않는다.

### 4.3 Brand assets

HTML 장표에서는 원본 PPTX에서 추출한 다음 자산을 사용한다.

| Asset | Use |
|---|---|
| `assets/solum-healthcare-navy.png` | 흰 배경, 마감 장표 |
| `assets/solum-healthcare-white.png` | 네이비 배경, 표지와 챕터 헤더 |

- 이미지 비율을 변경하거나 로고를 다시 조판하지 않는다.
- 로고 주변에는 로고 전체 높이의 최소 `25%`에 해당하는 빈 공간을 둔다.
- 색상 반전 필터로 다른 버전을 만들지 말고 배경에 맞는 원본 자산을 선택한다.
- 장표당 로고는 1개만 사용한다.

## 5. Typography

### 5.1 Font family

원본 장표의 주 글꼴은 `나눔스퀘어`이며, 일부 분석 요소에 `Pretendard`와 `Segoe UI`가 사용됐다. 새 HTML 장표는 다음 순서를 따른다.

```css
font-family: "NanumSquare", "Pretendard", "Noto Sans KR", "Segoe UI", sans-serif;
```

- 제목: `NanumSquare ExtraBold` 또는 `NanumSquare`, `font-weight: 800`
- 본문: `NanumSquare`, `font-weight: 400–700`
- 수치와 영문 차트: `Pretendard` 또는 `Segoe UI`
- 코드, 경로, 식별자만 고정폭 글꼴 사용
- 장표 한 장에서 서로 다른 글꼴 계열을 2개 넘게 섞지 않는다.

### 5.2 Type scale

| Role | Size | Weight | Line height |
|---|---:|---:|---:|
| Cover brand/title | `64–76px` | `700–800` | `1.05` |
| Section number | `64px` | `800` | `1` |
| Header title | `30px` | `700` | `1.2` |
| Insight headline | `34–40px` | `700` | `1.25` |
| Block title | `24–28px` | `700` | `1.25` |
| KPI value | `54–64px` | `700–800` | `1` |
| Body | `19–22px` | `400–600` | `1.45` |
| Chart caption | `17–19px` | `400–600` | `1.35` |
| Footnote/source | `12–14px` | `400` | `1.35` |

한글 본문은 `18px` 미만으로 축소하지 않는다. 내용이 넘치면 글자를 줄이지 말고 문장을 줄이거나 장표를 나눈다.

## 6. Master layouts

### 6.1 Cover

- 왼쪽 `58–62%`는 `--navy-950` 배경으로 사용한다.
- 오른쪽은 연구·검사·실험실 관련 고해상도 사진을 사용한다.
- 사진 경계는 1개의 강한 사선 또는 평행한 사선 마스크로 처리한다.
- 브랜드 로고는 왼쪽 중앙에 흰색 버전을 사용한다.
- 제목은 로고 아래에 1줄, 한글 부제는 그 아래에 작게 배치한다.
- 저작권은 좌측 하단 안전 여백 안에 둔다.
- 표지에는 차트, 카드, 긴 설명을 넣지 않는다.

### 6.2 Standard content slide

상단에 고정된 챕터 헤더를 둔다.

```text
┌───────────────┬──────────────────────────────────────────────┐
│      01       │ 장표 제목                         LOGO       │
│               │ Ch 02 Core Competency                       │
└───────────────┴──────────────────────────────────────────────┘
```

- 헤더 배경: `--navy-950`
- 번호 영역: 흰색 사선 패널 안에 네이비 두 자리 번호
- 제목: 흰색, 좌측 정렬
- 영문 챕터 캡션: 흰색 또는 `#D6E0EF`, italic, 작은 크기
- 로고: 우측 상단 흰색 버전
- 번호는 `01`, `02`처럼 항상 두 자리로 표기한다.

### 6.3 Analysis result slide

- 헤더 아래 첫 줄에 분석 결론을 큰 문장으로 배치한다.
- 차트는 2개 또는 3개의 동일한 시각적 컬럼으로 정렬한다.
- 각 차트의 캡션은 차트 아래 중앙 정렬한다.
- 핵심 모델 지표는 하단 KPI 카드에 분리한다.
- 여러 분석 단계를 비교할 때만 중앙 화살표를 사용한다.
- 수치가 장표의 주인공이어야 하며 설명문은 보조 역할에 머문다.

### 6.4 Architecture and data-flow slide

- 시스템 경계는 `2px` 네이비 점선으로 표시한다.
- 시스템 노드는 네이비 채움 또는 흰색 바탕·네이비 테두리를 사용한다.
- 데이터 흐름은 굵기 `3–4px`의 직선 또는 직각 연결선으로 표현한다.
- 단방향과 양방향은 화살촉으로 구분하고 반드시 텍스트 레이블을 붙인다.
- 망, 위치, 보안 영역은 점선 그룹 박스로 묶는다.
- 핵심 원칙은 오른쪽에 짧은 불릿 블록으로 분리한다.
- 흐름선이 노드나 텍스트를 가로지르지 않게 한다.

### 6.5 Decision and request table

- 표 헤더는 `--navy-950` 또는 `--navy-800`, 글자는 흰색을 사용한다.
- 본문 행은 `#D1D3D8`과 `#E5E7EB` 수준의 중립 회색을 교차 사용한다.
- 첫 열 번호는 `①`, `②` 또는 `01`, `02` 중 하나로 통일한다.
- 요청, 담당, 결정 등 열의 목적을 명확히 분리한다.
- 미결 사항은 색으로만 표시하지 않고 `안내 필요`, `결정 필요`처럼 텍스트로 쓴다.
- 한 셀 안에 5줄이 넘으면 표를 나누거나 별도 장표로 이동한다.

### 6.6 Closing slide

- 흰 배경 중앙에 네이비 로고만 배치한다.
- 회사 URL은 하단 중앙에 연한 회색으로 표시한다.
- 추가 메시지, 연락처, 차트는 요청이 있을 때만 넣는다.

## 7. Components

### 7.1 Insight headline

장표가 답하는 질문에 대한 결론을 한 문장으로 작성한다. 제목을 반복하지 않는다.

- 길이: 한글 기준 36자 이내 권장
- 위치: 챕터 헤더 바로 아래
- 정렬: 중앙 또는 좌측. 같은 덱 안에서는 한 방식으로 통일
- 금지: `분석 결과`, `현황`, `기타`처럼 결론이 없는 표현

### 7.2 KPI card

- 흰색 또는 매우 옅은 의미색 배경
- `1px` 의미색 테두리
- 모서리 반경 `8px`
- 상단: 단계 또는 지표 라벨
- 중앙: 큰 값
- 하단: `AUC`, `Macro F1`, `Sensitivity` 등 정확한 지표명
- 지표마다 평가 데이터셋과 표본 수를 근처 각주에 표기한다.

### 7.3 Process step

- 단계 번호는 검은색 또는 네이비 정사각형 안에 흰색으로 표시한다.
- 단계 이름은 번호 오른쪽 또는 아래에 굵게 배치한다.
- 프로세스 이미지는 동일한 시각 높이로 맞춘다.
- 단계 사이에는 한 방향 화살표를 사용한다.
- 전체 소요시간은 하단의 긴 네이비 화살표에 표시할 수 있다.

### 7.4 System node

- 핵심 저장소와 서버: 네이비 면, 흰색 제목, 옅은 파란색 세부정보
- 일반 워크스테이션: 흰색 면, 네이비 점선 테두리
- 카드 내부 텍스트는 제목 1줄과 사양 1–2줄로 제한한다.
- 그림자와 입체 효과는 사용하지 않는다.

### 7.5 Source and disclaimer

- 외부 자료와 문헌은 슬라이드 하단에 출처를 표시한다.
- 내부 분석은 데이터셋, run 또는 산출물 경로를 식별할 수 있어야 한다.
- 가정, 제한, 미검증 정보는 본문 또는 각주에서 명시한다.
- 중요한 제한은 글자 크기를 줄여 숨기지 않는다.

## 8. Chart rules

### 8.1 General

- 한 차트는 하나의 질문에 답해야 한다.
- 범례보다 직접 레이블을 우선한다.
- 축, 단위, 표본 수, 평가 기준을 생략하지 않는다.
- 3D 차트, 과도한 그라디언트, 그림자, 장식적 격자선을 사용하지 않는다.
- 차트 배경은 흰색으로 유지한다.
- 격자선은 필요한 경우 `--line-300` 이하의 대비로 사용한다.
- 소수 자릿수는 비교에 필요한 수준으로 통일한다.

### 8.2 Model performance

- Cancer Screening은 보라, Cancer Type ID는 초록을 기본 의미색으로 사용한다.
- 혼동행렬은 단일 순차 색상 스케일을 사용하고 셀 안에 값을 직접 표기한다.
- ROC, PR, calibration plot은 기준선과 신뢰구간을 명확히 구분한다.
- 성능값만 크게 표시하지 말고 검증 방식, 표본 수, 데이터 분할 기준을 함께 표기한다.

### 8.3 SERS and cohort plots

- 스펙트럼 축에는 Raman shift 단위와 intensity 정의를 표기한다.
- 평균 스펙트럼은 집단별 표본 수와 집계 방식(mean 또는 medoid)을 명시한다.
- 코호트 색상은 동일한 덱에서 일관되게 유지한다.
- 병원 차이가 성능에 영향을 줄 수 있는 분석에는 병원 confound 고지를 포함한다.

## 9. Data integrity and terminology

이 절은 시각 규칙보다 우선한다.

- `PAN = CPAN + YPAN = 100명`이며 SPAN을 PAN에 포함하지 않는다.
- Cancer Screening 결과에는 cross-hospital 일반화의 hospital-confound 가능성을 명시한다.
- 서로 다른 run은 aggregation mode, cancer set, non-cancer set, sample count가 모두 같을 때만 직접 비교한다.
- Stage 1 대신 `Cancer Screening`, Stage 2 대신 `Cancer Type ID`를 사용한다.
- 차트와 KPI는 분석 산출물에서 확인된 값만 사용한다.
- 문헌 피크는 보조 해석으로만 표시하고 내부 모델의 피크처럼 표현하지 않는다.
- 개인의 확진 확률과 집단 관측 비율을 혼동하지 않는다.

## 10. HTML implementation contract

### 10.1 Required structure

```html
<main class="deck">
  <section class="slide slide--content" aria-label="슬라이드 1">
    <header class="chapter-header">
      <div class="chapter-number">01</div>
      <div class="chapter-copy">
        <h1>분석 제목</h1>
        <p>Ch 01 Analysis</p>
      </div>
      <img class="brand-logo" src="assets/solum-healthcare-white.png" alt="SOLUM Healthcare">
    </header>
    <div class="slide-body">
      <h2 class="insight-headline">장표가 전달할 하나의 결론</h2>
    </div>
  </section>
</main>
```

### 10.2 Required CSS foundation

```css
:root {
  --slide-width: 1600px;
  --slide-height: 900px;
  --navy-950: #001f3c;
  --navy-800: #0a306d;
  --navy-700: #123b78;
  --ink-900: #111827;
  --ink-700: #333333;
  --ink-500: #6b7280;
  --line-300: #d6e0ef;
  --surface-100: #f4f6fa;
  --surface-200: #e0e4ed;
  --purple-700: #391d8d;
  --purple-100: #e0ddf0;
  --green-700: #029567;
  --green-100: #d0ede4;
  --white: #ffffff;
  --font-sans: "NanumSquare", "Pretendard", "Noto Sans KR", "Segoe UI", sans-serif;
}

* {
  box-sizing: border-box;
}

html,
body {
  margin: 0;
  background: #d9dde5;
  font-family: var(--font-sans);
  color: var(--ink-900);
}

.slide {
  position: relative;
  width: var(--slide-width);
  height: var(--slide-height);
  overflow: hidden;
  background: var(--white);
  page-break-after: always;
}

.chapter-header {
  height: 152px;
  display: grid;
  grid-template-columns: 220px 1fr 220px;
  align-items: center;
  padding-right: 64px;
  color: var(--white);
  background: var(--navy-950);
}

.slide-body {
  height: calc(var(--slide-height) - 152px);
  padding: 24px 64px 40px;
}

@page {
  size: 13.333in 7.5in;
  margin: 0;
}

@media print {
  html,
  body {
    background: #ffffff;
    print-color-adjust: exact;
    -webkit-print-color-adjust: exact;
  }
}
```

### 10.3 Rendering rules

- 장표마다 하나의 `<section class="slide">`를 사용한다.
- 스크롤 가능한 카드나 접히는 콘텐츠를 만들지 않는다.
- 차트와 다이어그램은 가능한 한 SVG로 생성한다.
- 텍스트를 이미지 안에 굽지 않는다.
- 웹 애니메이션, hover 의존 정보, 자동 재생 효과를 사용하지 않는다.
- 이미지에는 목적을 설명하는 `alt`를 제공한다.
- PDF 출력 시 배경 그래픽과 브랜드 색상이 유지되어야 한다.
- 결과물은 Chrome 계열 브라우저에서 동일하게 렌더링되어야 한다.

## 11. Density limits

- 제목: 최대 2줄
- Insight headline: 최대 2줄
- 본문 불릿: 장표당 최대 5개, 불릿당 최대 2줄
- 차트: 최대 3개
- KPI: 최대 4개
- 아키텍처 노드: 최대 9개 권장
- 표: 최대 8행 권장
- 본문과 차트가 동시에 복잡하면 장표를 분리한다.

## 12. Do and do not

### Do

- 결론을 먼저 쓰고 근거를 아래에 배치한다.
- 좌우 정렬선과 카드 높이를 맞춘다.
- 동일 의미는 동일한 색과 도형으로 반복한다.
- 정보 그룹 사이에 충분한 흰 여백을 둔다.
- 네이비를 구조색으로, 보조색을 의미색으로 사용한다.

### Do not

- 범용 SaaS 대시보드처럼 둥근 카드와 그림자를 남발하지 않는다.
- 모든 요소에 색을 넣지 않는다.
- 사진을 본문 배경으로 깔아 데이터 대비를 낮추지 않는다.
- 장표를 문서처럼 긴 문단으로 채우지 않는다.
- 숫자만 크게 만들고 평가 조건을 생략하지 않는다.
- 원본 PPTX의 저해상도 캡처를 HTML 장표 배경으로 사용하지 않는다.

## 13. QA checklist

완료 전 다음 항목을 모두 확인한다.

- `1600 × 900px` 캔버스에서 잘림이 없다.
- PDF 출력 결과가 브라우저 화면과 동일하다.
- 모든 텍스트가 최소 크기 기준을 충족한다.
- 제목, 숫자, 로고, 본문의 기준선이 맞는다.
- 네이비 외 의미색이 2개를 넘지 않는다.
- 차트의 축, 단위, 표본 수, 출처가 있다.
- 성능 비교 조건과 제한이 명시됐다.
- Cancer Screening hospital-confound 고지가 필요한 장표에 포함됐다.
- 로고 비율과 안전 여백이 유지됐다.
- 외부 공유 전에 개인정보, 환자 식별자, 비밀정보가 제거됐다.

## 14. Reference slide mapping

| Reference slide | Pattern to reuse |
|---|---|
| 1 | 네이비 표지, 사선 사진 마스크, 흰색 로고 |
| 2 | 긴 고지문과 하단 기준선 |
| 3 | 챕터 헤더와 의사결정 표 |
| 4 | 단계형 프로세스와 전체 소요시간 화살표 |
| 5 | 3열 분석 차트와 하단 KPI 비교 |
| 6 | 보안 영역을 포함한 상세 시스템 아키텍처 |
| 7 | 네이비 시스템 노드와 단방향·양방향 데이터 흐름 |
| 8 | 흰 배경의 중앙 로고 마감 |
