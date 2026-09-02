# Paper–Code Audit: `whitaker_hayes_despike`

**날짜**: 2026-09-02
**대상 구현**: `src/sers/preprocessing.py:whitaker_hayes_despike`
**대상 논문**: Whitaker DA, Hayes K. A simple algorithm for despiking Raman
spectra. Chemom Intell Lab Syst. 2018;179:82–4. (`experiment.papers` [4])

## 근거 자료의 한계 (먼저 밝힘)

**논문 본문은 읽지 못했다.** Elsevier 유료이고, ChemRxiv 프리프린트는
Cloudflare 챌린지로 접근 불가였다(유료 장벽이 아니라 봇 차단). 우회는 하지
않았다.

대신 **저자들이 직접 공개한 참조 구현**을 확보했다:
Mendeley Data `sxjgbgg95y` v1 — 저자 R 스크립트 + 보충자료.
`docs/ml/papers/ref04_whitaker2018_script.R`, `ref04_whitaker2018_supp.pdf`.

프로즈 요약보다 참조 구현이 대조 근거로 더 낫다. 다만 **본문에만 있고
deposit에 없는 내용은 확인 불가**로 남는다 (아래 명시).

## 대조표

| 저자 R 참조구현 | 우리 구현 (감사 전) | 판정 | 조치 |
|---|---|---|---|
| `ModifiedZscore(diff(y))`, R `mad()` (기본 constant=1.4826) | `0.6745*(grad−median)/median(\|grad−median\|)` | ✅ **수학적 동등** (1/1.4826 = 0.67449) | 없음 (주석으로 등가성 명시) |
| `z = rbind(0, z)` — 선두 0 padding | `z_mod[0] = 0` | ✅ 일치 | 없음 |
| `ma = 5` 이웃 창 | `window = 5` | ✅ 일치 | 없음 |
| `w = w[z[w]==0]` — flag된 이웃 제외 | `clean[lo:hi]` 마스크 | ✅ 일치 | 없음 |
| `mean(y[w])` — 원본 y 기준 | flag된 지점만 갱신, 이웃은 clean만 읽음 | ✅ 동등 (순차 오염 없음) | 없음 |
| **`threshold = 6`** | 기본값 **7.0** | ❌ **불일치** | 기본값 6.0으로 정정 |
| **`z[1] = z[n] = 1`** — 양 끝점 무조건 spike 처리 | 없음 | ❌ **불일치** | `force_endpoints=True` 추가 |

## 확인 불가로 남은 것

- 논문 본문이 threshold 6을 어떤 근거로 제시하는지 (deposit은 값만 하드코딩,
  주석으로 "높은 값에서 시작해 낮춰가라"고만 안내)
- 문자열 "0.6745"가 본문에 실제로 등장하는지 — deposit에는 없다. 1.4826과의
  등가성은 **우리 쪽 유도**이지 논문 인용이 아니다.

## 파생 관찰 (논문 주장 아님, 우리 관찰)

단일 지점 spike는 차분에서 두 개의 큰 값을 만들므로 `z[i]`와 `z[i+1]`이 함께
flag된다. 즉 flag 개수가 순진한 탐지기의 약 2배가 된다. 동작상 무해하지만
flag 수를 품질 지표로 쓸 때 오해할 수 있다.

## 기존 QC와의 관계

`src/sers/qc/qc.py:detect_cosmic_ray`와 중복이 아니다:

| | `detect_cosmic_ray` | `whitaker_hayes_despike` |
|---|---|---|
| 검사 범위 | 전역 최대점 1개 | 전 구간 |
| 조치 | 스펙트럼 전체 폐기 | 해당 지점만 복구 |
| 결과 | 데이터 손실 | 데이터 보존 |

단, Stage-1 QC가 먼저 스펙트럼을 버리면 despike가 볼 기회가 없다. 순서
상호작용은 벤치마크에서 확인할 것.

## 판정

**REQUEST_CHANGES → 수정 완료.** 불일치 2건을 참조구현에 맞게 정정했고
테스트 116개가 통과한다. 다만 `audit_status`는 **`pending` 유지** —
승인은 사용자 몫이며, 본문 미확인 항목이 남아 있다.

## 검증 기록

| 항목 | 방법 | 결과 |
|---|---|---|
| 저자 R 코드 확보 | Mendeley Data 직접 다운로드 | ✅ `ref04_whitaker2018_script.R` |
| 논문 본문 | Elsevier / ChemRxiv | ❌ 접근 불가 (우회 안 함) |
| 정정 후 테스트 | `pytest tests/test_preprocessing.py` | ✅ 116 passed |
| 린트 | `ruff check` | ✅ 통과 |
