# Clinical Use Documentation Package

본 폴더는 SERS-AI 임상용 웹앱의 사용설명서와 사용적합성 평가 자료를 한 묶음으로 관리하기 위한 작업본이다.

현재 문서의 기준 용어는 [../TERMINOLOGY_STANDARD.md](../TERMINOLOGY_STANDARD.md)를 따른다.

## 문서 목록

| 문서 | 목적 |
|---|---|
| [USER_MANUAL.md](USER_MANUAL.md) | 검사자/임상의용 사용설명서 |
| [USABILITY_EVALUATION_PROTOCOL.md](USABILITY_EVALUATION_PROTOCOL.md) | 사용적합성 평가 계획서 |
| [USABILITY_EVALUATION_FORMS.md](USABILITY_EVALUATION_FORMS.md) | 관찰 기록지, 사후 설문, 이슈 기록 양식 |
| [TRACEABILITY_MATRIX.md](TRACEABILITY_MATRIX.md) | 주요 사용 요구사항과 평가 태스크의 추적표 |

## 고정 기준

- 모델: STK-V2 앙상블 모델
- 판정 기준: 표준 판정 기준
- 내부 기준값 프로파일: `balanced`
- 사용자 표시 기준값: SSI 4.0
- 결과 용어:
  - `SSI 점수`
  - `추가 확인 권고`
  - `기준 미만`
  - `암종 분류 확률`

## 금지/주의 용어

아래 표현은 별도 정의와 검증 전까지 사용자 화면, 보고서, 사용설명서, 사용적합성 자료에서 사용하지 않는다.

- 강한 양성
- 약한 양성
- 암 진단
- 정상 확정
- 확진검사
- 운영 모드 선택

## 문서 관리 메모

이 패키지는 인허가 제출본이 아니라 내부 정리본이다. 실제 제출 전에는 제품명, 모델 버전, 소프트웨어 버전, 사용환경, 평가 대상자 수, IRB/윤리 검토 범위, 법규 문구를 품질문서 체계에 맞춰 고정해야 한다.
