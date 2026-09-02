# sers-data-integrity-checker — QC Threshold 라우트 백테스트

**날짜**: 2026-08-31
**목적**: 신규 리뷰 에이전트가 이미 정답을 아는 사고(2026-04-29 dashboard v3
가짜 데이터 감사에서 지목된 QC Threshold 라우트)를 실제로 잡아내는지 검증.

## 감사 대상
- `/home/user/workspace/solum-dashboard/spa/src/routes/research/qc/Threshold.tsx` (주 대상)
- 같은 디렉터리 비교 대상: `DataValidation.tsx`, `Diagnostic.tsx`, `PreprocessingExplorer.tsx`
- 대조에 사용한 실제 데이터: `SERS-AI/results/qc/qc_experiment/sweep_results.csv`/`.json`,
  `SERS-AI/results/qc/qc_validation/QC_VALIDATION_REPORT.md`,
  `SERS-AI/scripts/pipeline/run_qc_threshold_experiment.py`, `SERS-AI/src/sers/config.py`

## 가짜 숫자 의심 항목 (최우선)

**1. `Threshold.tsx:8-12` — `Math.random()`으로 AUC를 매 렌더마다 생성**
```js
auc: 0.96 + Math.random() * 0.03 - (rsd - 5) * 0.005 - (0.95 - corr) * 0.4
```
과거 사고(`feedback_data_fabrication.md`)와 동일 패턴. 차트에는 `simulated`
배지가 있어 라벨링은 됐으나, RouteHero 상단 stats와 "주요 발견" 박스에는
이 라벨이 없음.

**2. Threshold sweep grid 자체가 실행된 적 없는 조합**
- 화면 grid: RSD `[3,4,5,6,7]` × Corr `[0.85,0.9,0.92,0.95,0.97,0.98,0.99]`
- 실제 실행 grid(`run_qc_threshold_experiment.py`, `sweep_results.csv`):
  RSD `[3,5,7,10,15,20,inf]` × Corr `[0.95,0.9,0.85,0.8,0.0]`
- RSD=4,6 / Corr=0.92,0.97,0.98,0.99 조합은 실행된 적 없음 — x축 자체가 허구.

**3. "QC 탈락 19.5%, RSD<6·Corr>0.92로 완화 권장" — 실제 내부 검증 결과와 정반대**
- `QC_VALIDATION_REPORT.md` 결론: 현재 기준(RSD<5, Corr>0.95) **유지 권고**,
  실제 rejection rate **69.3%** (19.5%가 아님).
- 숫자 크기와 권고 방향 모두 실제 자료와 반대 — 추적 불가, 발명된 서사로 판단.

**4. `Diagnostic.tsx` — `FAILURE_BY_GROUP` 하드코딩**
- BLC fail rate 19.4%로 표시되나 실제(`QC_VALIDATION_REPORT.md` per-group
  summary)는 pass_rate 7%(즉 fail ~93%, 전체 그룹 중 최악)로 정반대.
- BRE/OVA/PRO 등 그룹별 표본 수도 실제 `group_qc_summary`와 불일치.

**5. `DataValidation.tsx` — 8,140 스펙트럼/92.2% pass 등 하드코딩, fetch 없음, 출처 추적 불가.**

## 비교: 문제 없는 파일
- `PreprocessingExplorer.tsx` — `/data/preprocessing_stages.json`을 실제 fetch,
  실제 파일 값과 정확히 일치. **PASS.**

## 도메인 규칙 위반
- Hospital confound: FAIL(부분) — 가설만 나열, 통제된 분석 없음. vault 접근 불가로 최종 확인은 "확인 필요".
- QC 필터링: FAIL — 실제 파이프라인 산출물이 있는데도 참조하지 않고 대체 수치 사용.
- CSV 인코딩: 확인불가(해당 라우트는 CSV 미작성).
- BLC 299개 하드코딩: PASS(부분, total은 맞으나 fail rate는 틀림).

## 종합 판정
**BLOCK** (외부 공유/merge 전 반드시 해결)

## 조치 필요 목록
1. `Threshold.tsx`의 `Math.random()` sweep을 실제 데이터 fetch로 교체, 존재하지 않는 grid 조합 제거.
2. RouteHero/주요 발견 문구를 `QC_VALIDATION_REPORT.md` 실제 결론에 맞게 수정 또는 확정 전까지 placeholder화 — 어느 결론이 맞는지 사용자 확인 필요.
3. `Diagnostic.tsx`의 `FAILURE_BY_GROUP`을 실제 파일 기반으로 교체, BLC fail rate 즉시 정정.
4. `DataValidation.tsx` 수치 출처 확인 후 교체 또는 라벨링.
5. Hospital confound 정식 분석 승격 여부 결정.
6. 위 수정 전까지 세 라우트 모두 외부 공유/데모 제외 권장.

---

## 메타: 이 리뷰 자체에 대한 프로세스 노트

이 리뷰의 1차 실행 시, 에이전트에게 "결과를 이 파일에도 저장하라"고
지시했으나 **실행되지 않았다** (보고서 텍스트만 반환하고 파일 쓰기는
생략함). 이 파일은 호출한 세션(Claude)이 사후에 수동으로 작성한 것이다.
이후 설계를 변경: `sers-*` 리뷰 에이전트는 Write 도구를 아예 갖지 않고
(완전 읽기 전용), 결과를 `docs/reviews/`에 남기는 책임은 호출한 세션이
진다. 자세한 내용은 `/home/user/AGENTS.md`의 "Review log" 절 참고.
