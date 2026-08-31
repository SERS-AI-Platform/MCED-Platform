# AECD 전체 원천 데이터 Screening 기록

## 목적

환원제 변경 전·후 데이터를 동일한 source inventory 규칙으로 확인하고, 이후
학습·평가에 재사용할 후보군과 calibration/control 자료를 분리한다. 이 기록은
개별 환자 식별자, 원본 파일명, 원본 URI, 스펙트럼 배열을 포함하지 않는다.

## 판정 기준

- 현재 `mapping` 임상 반복 측정은 관측된 DB reagent lot와 측정일로 확인된
  `post_change_verified`로 분류한다.
- 과거 Thermo, Medical Raman, remeasurement 및 기타 임상 후보 source는
  시간상 이전 자료이지만 변경 전 환원제의 화학명이나 lot가 source metadata에
  남아 있지 않아 `pre_change_unverified`로 분류한다.
- 유한값을 갖는 raw/replicate 파일은 보존하며, 이 screening 단계에서는
  spectral QC를 적용하지 않는다.
- `_ave` 파생 평균, reference/calibration, background, equipment test,
  metabolite reference는 inventory에는 기록하되 임상 screening 후보군에서
  제외한다.
- Cancer Screening AUC는 기관·측정 배치와 질환군이 얽힌 hospital/measurement
  confounding 가능성을 포함해 해석한다.

## 집계 결과

| 항목 | 값 |
|---|---:|
| 확인 파일 | 77,722 |
| 유한값 valid 파일 | 69,735 |
| reject 파일 | 12 |
| 파생 평균 제외 | 7,975 |
| screening 후보 파일 | 45,199 |
| 총 spectrum point | 127,712,605 |
| manifest SHA-256 | `c47345f5d37a70c94ba73979edde2ab1ed518d0807e598497818c75521d45292` |

주요 source family는 다음과 같다.

| source family | 환원제 phase | 역할 | valid 파일 |
|---|---|---|---:|
| mapping clinical | post_change_verified | 임상 반복 측정 후보 | 13,673 |
| historical Thermo | pre_change_unverified | 임상 반복 측정 후보 | 11,785 |
| Medical Raman | pre_change_unverified | 임상 raw 후보 | 11,599 |
| remeasurement | pre_change_unverified | 임상 반복 측정 후보 | 8,142 |

`mapping` archive는 별도 신규 dataset으로 세지 않았다. sidecar를 제외한 archive
13,937개와 filesystem 13,937개가 모두 일치했고 size mismatch는 0건이므로
현재 mapping post-change source의 duplicate alias로 기록했다.

Reference, background, equipment, metabolite 자료는 임상 후보군에 섞지 않고
별도 artifact role로 보존한다. 자세한 source family별 집계와 reject 사유는
`results/data_source_screening_20260827_v1/`의 aggregate CSV/JSON을 사용한다.

## MLflow 기록

- Experiment: `aecd-all-source-screening`
- Run ID: `f19485cb393d4ca4928305adb2f57df7`
- Lineage key: `aecd-all-source-screening:c47345f5d37a70c94ba73979edde2ab1ed518d0807e598497818c75521d45292:cf87e2e0ae272b97b0759aa68091dd54bdade578eff14f16831e7634786b6b77`
- UI: `http://127.0.0.1:5000/#/experiments/3/runs/f19485cb393d4ca4928305adb2f57df7`

MLflow에는 aggregate counts, phase policy, manifest/archive hash와 screening
disclaimer만 기록한다. patient ID, source URI, raw array는 기록하지 않는다.

실험 history registry는 별도 experiment로 등록했다.

- Experiment: `aecd-experiment-registry` (experiment `4`)
- 등록 후보/완료: `107 / 107`
- phase: `post_change_verified` 13건, `pre_change_unverified` 94건
- 증거 수준: 주 분석 11건, 참고 2건, historical summary 8건,
  historical log metadata-only 86건
- UI: `http://127.0.0.1:5000/#/experiments/4`

이 registry의 historical log 86건은 결과 artifact가 현재 경로에서 확인되지
않은 metadata-only 기록이다. direct comparison 대상이 아니며, run tag에
`direct_comparison=not_allowed`를 고정했다.

## 코드 정리 screening

| 후보 | 확인 결과 | 현재 조치 |
|---|---|---|
| `scripts/legacy/analysis/backfill_experiment_registry.py` | 참조하는 `models/manifest.json`이 없고, 기존 artifact 경로도 전부 확인되지 않음 | 삭제 후보. 정확한 삭제 승인 후 제거 |
| `notebooks/build_stkv2_dataset.py` | TODO/미구현 부분이 있으나 두 notebook에서 import되고 산출물도 존재 | 보류. canonical feature extractor로 대체한 뒤 정리 |
| `scripts/legacy/analysis/explore_data.py` | `sers analyze explore`가 직접 실행 | 유지 |
| `models/legacy/scripts/train.py` | CLI, benchmark, legacy model builder에서 사용 | 유지 |
| `scripts/training/_legacy/train_resnet.py` | legacy analysis 도구 여러 곳에서 동적 import | 유지 |
| `models/legacy/scripts/test.py` | CLI와 benchmark에서 사용되며 `--help` 실행 확인 | 유지 |
| `models/legacy/model.py` | architecture base class의 abstract method | 유지 |

따라서 현재 증거만으로 즉시 삭제할 수 있는 파일은 확정하지 않았다. 삭제 시에는
위 표의 `backfill_experiment_registry.py`처럼 대상과 참조 범위를 먼저 고정하고,
문서의 stale reference도 함께 처리한 후 targeted test를 실행한다.

## 재현 명령

```bash
uv run python scripts/db/aecd_source_inventory.py
uv run python scripts/db/aecd_source_inventory.py --log-mlflow
uv run python scripts/db/aecd_experiment_registry.py
```

source inventory 명령은 inventory를 다시 만들며, MLflow 기록은 같은 lineage
key가 이미 있으면 중복 생성하지 않는다. experiment registry도 동일한 lineage
key를 재사용하므로 재실행 시 기존 run을 보존한다.

## Historical model artifact backfill

기존 결과 폴더와 legacy production artifact에서 checkpoint를 다시 찾아 별도
MLflow experiment와 Model Registry에 등록했다.

- Experiment: `aecd-model-artifact-backfill` (experiment `5`)
- Backfilled bundles: `95`
- Model artifact files: `667` (`.pt` 329개, `.joblib` 338개)
- Registered model families: `13`
- Model Registry versions: `95`
- Metadata가 확인된 bundle: `93`
- Checkpoint-only bundle: `2`
- Inventory: `results/mlflow_historical_model_backfill_20260827_v1/backfill_inventory.csv`
- Report: `results/mlflow_historical_model_backfill_20260827_v1/REPORT.md`

각 run은 `source_model_version`과 함께 `config_*`, `model_params_*`를 MLflow
parameter로 기록하고, 직전 chronological bundle 대비 변경값을
`lineage/parameter_changes.json`에 저장한다. 이 기록은 과거 checkpoint 형식을
그대로 보존한 retrospective bundle이며, serving 가능한 MLflow flavor model로
주장하지 않는다. 환원제 phase는 source identity가 확인되지 않은
`pre_change_unverified`로 유지한다.

재현 명령:

```bash
uv run scripts/db/aecd_model_artifact_backfill.py
```
