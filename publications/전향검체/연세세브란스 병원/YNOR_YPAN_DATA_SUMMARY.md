# YNOR·YPAN 전체 데이터 및 임상정보 정리

## 핵심 구분

- 임상 레코드: YNOR 30명 + YPAN 30명 = 60명
- primary spectrum 독립 피험자: YNOR 29명 + YPAN 30명 = 59명, 각 5 replicate
- reacquired spectrum: 동일 59명에 대해 각 5 replicate가 별도로 존재
- processed acquisition은 60건이지만 `YNOR 21`은 `YNOR 48`의 재측정본이므로 primary 분석은 59명

## 데이터 층별 수량

| 군 | 임상 레코드 | Primary subjects / spectra | Reacquired subjects / spectra | Processed acquisitions | Primary model |
|---|---:|---:|---:|---:|---:|
| YNOR | 30 | 29 / 145 | 29 / 145 | 30 | 29 |
| YPAN | 30 | 30 / 150 | 30 / 150 | 30 | 30 |

## 임상 구성

- YNOR 진단 구성: 건강한 자 10명, 췌장낭종 10명, 기타 양성 질환 6명, 만성 췌장염 4명
- YNOR age: 51.8 ± 15.2; median 51.0; range 22.0-71.0; BMI: 24.0 ± 3.1; median 24.2; range 16.8-28.7
- YPAN age: 65.7 ± 10.9; median 67.5; range 44.0-86.0; BMI: 21.9 ± 3.1; median 22.5; range 15.8-28.4
- 성별: YNOR F 18 / M 12, YPAN F 14 / M 16
- YPAN analysis stage: 1-2 4명, 3-4 19명, missing 7명
- YNOR은 암 병기 대상이 아니며, 15/30명에서 sample timing이 post-op으로 기록되고 15명은 결측
- CA19-9: YNOR 14/30, YPAN 24/30에서 값 보유

## 해석상 중요사항

- YNOR은 순수 건강인군이 아니라 건강인·췌장낭종·만성 췌장염·기타 양성 질환이 섞인 non-cancer 군이다.
- YPAN이 YNOR보다 연령이 높고 BMI가 낮아 임상 구성 차이가 spectrum 분류에 영향을 줄 수 있다.
- `YNOR 21`과 `YNOR 48`을 독립 fold로 분리하면 동일 피험자 누수가 생기므로 반드시 subject 단위로 묶거나 repeat를 제외해야 한다.
- 연세 subset은 단일기관 내부 자료다. Cancer Screening 결과를 외부검증 또는 cross-hospital 일반화 근거로 사용하면 안 된다.

## 생성 표

- `tables/yonsei_ynor_ypan_group_inventory.csv`
- `tables/yonsei_ynor_ypan_acquisition_inventory.csv` (검체 단위, Git 비추적)
- `tables/yonsei_ynor_ypan_processed_inventory.csv` (검체 단위, Git 비추적)
- `tables/yonsei_ynor_ypan_clinical_records.csv` (검체 단위, Git 비추적)
- `tables/yonsei_ynor_ypan_clinical_completeness.csv`

직접 환자 ID, 임상 원본 파일명, exact date, 자유서술 과거력은 새 표에서 제외했다. 날짜는 연도만 보존했다.
