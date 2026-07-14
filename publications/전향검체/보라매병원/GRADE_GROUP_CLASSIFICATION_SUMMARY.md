# Grade Group별 Cancer 분류 결과

보라매 Prostate cancer 41명을 대상으로 현재 OOF 결과를 Grade Group별로 정리했다. `three-group`은 Control / Biopsy-negative / Cancer를 구분하고, `Screening`은 Non-cancer / Cancer를 구분한다.

| Grade Group | Cancer n | three-group 정확 | three-group 오분류 | 오분류율 | Cancer→Biopsy-negative | Cancer→Control | Screening 정확 | Screening 오분류 | 오분류율 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GG1 | 9 | 7 | 2 | 22.2% | 2 | 0 | 9 | 0 | 0.0% |
| GG2 | 6 | 4 | 2 | 33.3% | 2 | 0 | 4 | 2 | 33.3% |
| GG3 | 11 | 5 | 6 | 54.5% | 4 | 2 | 6 | 5 | 45.5% |
| GG4 | 10 | 6 | 4 | 40.0% | 3 | 1 | 6 | 4 | 40.0% |
| GG5 | 5 | 4 | 1 | 20.0% | 1 | 0 | 2 | 3 | 60.0% |
| **전체** | **41** | **26** | **15** | **36.6%** | **12** | **3** | **31** | **10** | **24.4%** |

## 해석

- `three-group`에서는 GG3가 6/11명으로 가장 많이 오분류되었다.
- `three-group`의 Cancer 오분류 15명 중 12명은 `Biopsy-negative`, 3명은 `Control`로 분류되었다.
- `GG1-2`를 합치면 three-group 오분류는 4/15명(26.7%)이다.
- `GG3-5`를 합치면 three-group 오분류는 11/26명(42.3%)이다.
- Screening에서는 GG5가 3/5명(60.0%)으로 가장 높은 오분류율을 보였지만, 각 Grade의 표본 수가 작아 Grade별 성능 차이를 확정할 수 없다.
