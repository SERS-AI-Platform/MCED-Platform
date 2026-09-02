# Paper–Code Audit: `airpls_baseline`

**날짜**: 2026-09-02
**대상 구현**: `src/sers/preprocessing.py:airpls_baseline`
**대상 논문**: Zhang ZM, Chen S, Liang YZ. Baseline correction using adaptive
iteratively reweighted penalized least squares. Analyst. 2010;135(5):1138–46.
(`experiment.papers` [14])
**근거 자료**: 저자 자가보관 원고 —
`github.com/zmzhang/airPLS/airPLS_manuscript.pdf` → `docs/ml/papers/ref14_zhang2010.pdf`

## 대조표

| 논문 스펙 | 우리 구현 | 판정 |
|---|---|---|
| 목적함수 (eq.8) `Qᵗ = Σwᵢᵗ\|xᵢ−zᵢᵗ\|² + λΣ\|zⱼ−z_{j−1}\|²`, `z=(W+λD'D)⁻¹Wx` | 동일 (2차 차분 행렬 + spsolve) | ✅ 일치 |
| 가중치 (eq.9) `w=0` if `x≥z` | `weights[residual >= 0] = 0.0` | ✅ 일치 |
| 가중치 (eq.9) `w=exp(t·\|x−z\|/\|dᵗ\|)` if `x<z`, `\|dᵗ\|`=음수 잔차의 L1 노름 | `exp(iteration·\|res_neg\|/negative_sum)`, `negative_sum=\|Σres_neg\|` | ✅ **동등** (잔차가 모두 음수이므로 `\|Σ\|=Σ\|·\|`) |
| 종료조건 (eq.10) `\|dᵗ\| < 0.001·\|x\|` | `negative_sum < tol(1e-3) * Σ\|y\|` | ✅ 일치 |
| **가장자리 처리** | 양 끝 2점을 `max(weights[negative])`로 고정 | ⚠️ **논문에 없는 내용** |
| **λ 기본값** | `1e5` | ⚠️ 논문은 보편 기본값을 제시하지 않음 |
| 반복 횟수 | `niter=15` | ⚠️ 논문 명시 없음 |

## 핵심 발견: 우리 구현은 "출판된 알고리즘"이 아니라 제3자 번역본 계열

논문은 가장자리 처리를 **전혀 규정하지 않는다.** 그런데 널리 쓰이는 두 구현이
서로 다르게 처리한다:

| 구현 | 가장자리 | 기본값 |
|---|---|---|
| 저자 MATLAB `airPLS.m` (Zhimin Zhang, 2011) | 양 끝 **10%** 구간을 가중치 `p=0.05`로 고정 (`wep`/`p` — 논문에 없는 파라미터) | `lambda=10e7, order=2, itermax=20` |
| `airPLS.py` (Renato Lombardo 2014, **저자 코드 아님**, R 소스 번역) | 양 **끝 2점만** `exp(t·max\|d\|/dssn)`으로 고정 | `lambda_=100, porder=1, itermax=15` |
| **우리 구현** | 양 **끝 2점**을 `max(w_neg)`로 고정 | `lam=1e5, niter=15` |

우리 것은 **Lombardo 계열**(끝 2점 고정, itermax 15)이며, λ 기본값만 독자적이다
(1e5 — 저자 1e8도, Lombardo 100도 아님).

## 판정

**CLEAR (핵심 알고리즘) / WATCH (가장자리·기본값)**

- 목적함수·가중치 갱신·종료조건은 논문과 일치하므로 **알고리즘 자체는 올바르다**.
- 다만 "우리 airPLS는 Zhang 2010 그대로"라고 말하면 **부정확하다**. 가장자리
  처리는 논문 밖의 선택이고, 하필 저자 구현과도 다르다.
- λ=1e5는 근거가 확인되지 않은 값이다. 논문은 λ를 유일한 튜닝 대상으로 보고
  로그 그리드 탐색을 권한다 (예시값: 10 / 30 / 500).

## 조치 권고 (코드 변경은 하지 않음 — 사용자 판단 대상)

1. 프로덕션 기본 baseline은 `rolling_min`이라 이 구현이 기본 경로에 있지는
   않다. 그래서 즉시 위험은 아니다.
2. airPLS를 실제 실험에 쓸 때는 **λ를 로그 그리드로 탐색**해야 한다. 1e5를
   그대로 쓰는 것은 논문 근거가 없다.
3. 가장자리 처리를 저자 MATLAB 방식으로 맞출지 여부는 별도 결정 사항.
   두 방식이 가장자리에서 다른 결과를 낸다.

## 확인 불가

- 논문 본문의 가장자리 관련 서술 유무: 저자 원고 PDF에서 확인한 범위에서는
  없음. 출판본(Analyst)은 유료라 대조하지 못함.
