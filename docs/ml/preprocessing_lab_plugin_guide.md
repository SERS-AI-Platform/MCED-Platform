# Preprocessing Lab — 새 방법(논문 출처) 추가 가이드

이 문서는 논문에서 가져온 전처리 방법(baseline correction, smoothing,
normalization)을 SERS-AI 파이프라인에 안전하게 추가하는 표준 절차다.
`sers-preprocessing-comparator`와 `paper-code-audit` 스킬 워크플로우가 이
절차를 전제로 한다.

**무엇을 왜 구현할지**의 근거는 `preprocessing_design_guide.md`에 있다 (8단계
전처리 정의·근거·수식 + 논문 24편). 그 문서의 method 목록은 이미
`experiment.preprocessing_methods`에 `audit_status='pending'`으로 등록돼 있으니,
새 구현을 시작할 때 거기서 대상을 고르면 된다.

## 원칙

- **`src/sers/preprocessing.py`의 dispatcher에 등록한다.** `src/sers/signal.py`나
  `scripts/analysis/run_mapping_aligned_baseline_area_dwt.py`처럼 primitive를
  직접 재구현해서 우회하지 않는다 — 그러면 새 방법이 QC/config/테스트 인프라
  전체를 다시 잃는다.
- **기본값은 항상 opt-in이다.** 새 config 필드의 기본값은 현재 프로덕션 동작과
  동일해야 한다 — 명시적으로 그 방법을 선택하지 않는 한 기존 파이프라인은
  1비트도 바뀌지 않는다.
- **논문 인용을 코드에 남긴다.** 나중에 `preprocessing_lab_methods.reference_id`가
  가리키는 근거가 코드 자체에도 있어야 한다.

## 절차 (baseline 방법 추가 예시)

1. **구현.** `src/sers/preprocessing.py`의 baseline 계열 함수들 근처
   (`estimate_baseline`이 dispatch하는 지점, `preprocessing.py:764` 부근)에
   새 함수를 추가한다:

   ```python
   def my_paper_baseline(y: np.ndarray, *, param1: float = ...) -> np.ndarray:
       """<Paper Title>, <Authors>, <Year>. DOI/URL: <link>.

       <한두 줄로 핵심 아이디어 요약 — 논문의 알고리즘을 그대로 옮긴 것인지,
       어떤 부분을 우리 데이터에 맞게 조정했는지 명시>
       """
       ...
       return baseline
   ```

2. **Dispatcher에 분기 추가.** `estimate_baseline`(또는 해당하는
   `smooth_spectrum`/`normalize_spectrum`)의 elif 체인에 새 method 문자열을
   추가한다 — 이 문자열이 곧 `preprocessing_lab_methods.method_key`가 된다:

   ```python
   elif method == "my_paper_baseline":
       return my_paper_baseline(y, param1=kwargs.get("param1", ...))
   ```

3. **Config 필드 추가.** `src/sers/config.py`의 `PreprocessingConfig`에 flat
   필드를 추가한다 (기존 명명 규칙 `<method>_<param>` 그대로):

   ```python
   my_paper_baseline_param1: float = ...  # 논문 기본값, 프로덕션엔 영향 없음 (opt-in)
   ```

4. **테스트 추가.** `tests/test_preprocessing.py`의 관련 dispatch parametrize
   리스트(`test_baseline_dispatch_methods` 등)에 새 method 문자열을 추가하고,
   필요하면 그 방법 전용 `_is_finite`류 sanity 테스트를 하나 추가한다.

5. **DB에 등록.** `aecd_platform`의 `experiment.papers`에 논문을 먼저
   등록(없으면), `experiment.preprocessing_methods`에 `method_key`/`paper_id`/
   `implementation_path`/`audit_status='pending'`으로 행 추가
   (`src/sers/preprocessing_lab/db.py`의 `ExperimentTracker` 사용).

6. **사람 검수.** `paper-code-audit` 스킬을 논문 + 이번 diff에 대해 실행 →
   사용자가 결과를 보고 핵심 아이디어와 일치하는지 확인 → 승인 시
   `audit_status='approved'`로 갱신.

7. **실행.** `@preprocess-lab`을 DB 모드로 호출해 `my_paper_baseline`을
   `aecd_platform` 데이터에 대해 실행 (평가는 `@model-bench`에 위임), 결과가
   `experiment.preprocessing_runs`/`run_metrics`와 (`@experiment-runner`를 통해)
   `docs/ml/experiment.md` + `logs/experiment_registry.json`에 기록되는지 확인.

## 체크리스트

- [ ] `preprocessing.py`의 dispatcher에 등록했다 (signal.py나 독자 스크립트에
      숨기지 않았다)
- [ ] 새 config 필드 기본값이 프로덕션 동작을 바꾸지 않는다 (opt-in)
- [ ] 함수 docstring에 논문 인용이 있다
- [ ] `test_preprocessing.py`의 parametrize 리스트에 추가했다
- [ ] `experiment.papers`/`experiment.preprocessing_methods`에 등록했다
- [ ] `paper-code-audit` 결과를 사람이 검수하고 `audit_status`를 갱신했다
