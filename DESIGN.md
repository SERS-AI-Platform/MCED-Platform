# AECD Software Design System

## 1. Atmosphere & Identity

차분하고 명확한 임상 운영 화면을 지향한다. 장식보다 환자·QC·결과 상태와 다음 행동을 빠르게 구분하는 것이 우선이며, 흰색 카드와 파란색 주요 행동 버튼을 시각적 기준으로 사용한다.

## 2. Color

화면 색상은 `scripts/deployment/static/css/clinical.css`의 `:root` 토큰을 기준으로 한다. 기본 배경은 `--bg`, 카드는 `--bg-card`, 본문은 `--text-primary`, 보조 설명은 `--text-secondary`, 주요 행동은 `--accent`, 상태는 `--positive-bg`, `--negative-bg`, `--warning-bg`를 사용한다. 최종 판정 카드는 평균 SSI 3단계 기준에 따라 `SSI < 1.0`은 `--negative-surface`, `1.0 ≤ SSI ≤ 4.0`은 `--warning-surface`, `SSI > 4.0`은 `--positive-surface`를 사용한다. 게이지와 판정 배지는 같은 세 구간을 일관되게 표현하고 AA 대비를 확보한 `--negative-ink`, `--warning-ink`, `--positive-ink`를 사용한다. 새 원시 색상은 추가하지 않는다.

## 3. Typography

기본 글꼴은 `--font-sans`, 데이터 식별자는 `--font-mono`를 사용한다. 본문은 14~16px, 카드 제목은 20px을 기준으로 하며 기존 CSS 크기 체계를 유지한다.

## 4. Spacing & Layout

4px 배수 간격을 사용한다. 기본 화면은 `.app-container`, `.main-content`, `.card` 구조를 따르며 작업 버튼은 `.action-row`에 배치한다. 이력 화면의 넓은 콘텐츠 상한은 `--content-wide` 토큰을 사용하고, 기존 반응형 규칙을 모바일·태블릿·데스크톱에서 동일하게 적용한다.

## 5. Components

### Card

- 구조: `.card` 안에 `.card-title`과 본문을 배치한다.
- 상태: 기본·오류·성공 상태는 기존 alert/status 클래스를 재사용한다.
- 접근성: 제목 계층을 유지하고 정보는 색상만으로 전달하지 않는다.

### Action button

- 구조: 실제 이동은 `<a class="btn">`, 제출은 `<button class="btn">`을 사용한다.
- 변형: 주요 행동은 `.btn-primary`, 보조 행동은 `.btn-secondary`를 사용한다.
- 상태: 기존 hover·active·disabled 규칙을 유지하며 링크 목적을 텍스트로 명확히 표시한다.
- 접근성: 키보드로 접근 가능해야 하며 최소 44px 터치 영역을 유지한다.

### History filters and table

- 구조: 검색 조건은 `.history-filters`, 결과는 `.history-table`, 행별 동작은 `.history-actions`를 사용한다.
- 상태: QC 통과·실패는 기존 `.qc-badge`를 재사용하고, 대기는 중립 배경으로 표시한다.
- 반응형: 데스크톱에서는 표, 640px 이하에서는 각 행을 레이블이 있는 카드형 목록으로 전환한다.
- 권한: 관리자에게만 담당 계정 필터와 열을 표시하며, 일반 사용자는 자신의 기록만 본다.

### Retest BMI card

- 구조: 과거 BMI 누락 기록의 재검에서만 `.retest-bmi-card`를 표시한다.
- 상태: 10.0–60.0 범위 오류는 기존 `.inline-alert`를 사용한다.
- 접근성: 숫자 입력은 명시적 레이블, `required`, 범위 속성과 44px 조작 영역을 유지한다.

### Administrator account management card

- 구조: 관리자에게만 표시되는 `[계정]` 화면에서 대상 계정 선택, 임시 비밀번호, 비밀번호 확인 필드를 한 카드에 배치한다.
- 상태: 초기화 성공은 `.success-message`, 입력 오류는 `.inline-alert`를 재사용한다.
- 권한: 일반 계정에는 헤더 링크를 표시하지 않으며 직접 URL 접근도 서버에서 차단한다.
- 접근성: 모든 입력에 명시적 레이블과 `required`, 비밀번호 최소 길이, 안내문 연결을 적용한다.

### Mean SSI reference

- 구조: 결과 페이지에서 `.mean-ssi-reference` 안에 현재 평균 SSI 신호 수준, SSI 구간, 같은 구간 검증군의 관측 암 비율과 표본 수를 배치한다.
- 상태: 평균 SSI 판정 구간은 기존 `.risk-badge` 상태를 재사용하며 최종 판정 카드와 동일한 3단계 경계를 표시한다.
- 주의: 관측 암 비율은 검증 데이터의 구간별 집단 관측치이며 개인의 확진 확률이나 최종 판정으로 표현하지 않는다. 표본 수가 100명 미만인 구간에는 불확실성 안내를 표시한다.
- 상세 표: QC 요약과 반복측정 상세는 `.result-detail-table`을 사용하며 모바일에서 각 열 이름을 표시하는 2열 카드로 전환한다.
- 접근성: 의미 있는 제목 계층을 사용하고 수치의 의미를 색상 외 텍스트로 함께 제공한다.

## 6. Motion & Interaction

버튼의 기존 100~150ms 배경·transform 전환만 사용한다. 다운로드처럼 결과가 분리되는 행동은 별도 버튼과 명확한 파일 형식으로 표시한다.

## 7. Depth & Surface

기존 혼합 전략을 유지한다. 카드는 `--border`와 `--shadow`, 강조 화면은 상태별 배경 토큰을 사용한다.

## 8. Accessibility Constraints & Accepted Debt

WCAG 2.2 AA를 목표로 하고, 키보드 접근·명확한 링크 이름·44px 최소 조작 영역을 유지한다. 기존 전체 CSS의 토큰 정리와 독립 컴포넌트 쇼케이스는 이번 기능 변경 범위 밖의 기존 부채로 남긴다.
