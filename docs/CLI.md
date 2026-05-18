# SERS CLI Reference

통합 CLI로 모든 파이프라인 단계를 실행합니다. 기존 `python scripts/...` 경로 기반 호출도 그대로 동작합니다.

```
pip install -e .    # 최초 1회
sers --help         # 전체 명령어 확인
```

---

## Workflow

```
preprocess → train → test → predict
```

---

## preprocess

스펙트럼 전처리: 로드 → 파싱 → QC → 정규화 → 저장

```bash
sers preprocess                                    # Thermo + QC v2 (기본)
sers preprocess --source medical                   # Medical instrument
sers preprocess -n snv --qc-policy strict          # SNV 정규화 + strict QC
sers preprocess -n minmax -o ./output              # MinMax + 출력 경로 지정
sers preprocess --skip-qc                          # QC 스킵
sers preprocess --raw-only                         # main.py만 (QC pipeline 없이)
sers preprocess --raw-only --source medical        # Medical raw 전처리만
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `-s, --source` | 장비 소스 (`thermo` / `medical`) | `thermo` |
| `-c, --config` | config.yaml 경로 | `config/config.yaml` |
| `-n, --normalization` | 정규화 (`snv` / `minmax` / `l2` / `area` / `none`) | config 따름 |
| `--qc-policy` | QC 정책 (`v2` / `strict` / `none`) | `v2` |
| `-o, --output-dir` | 출력 디렉토리 | config 따름 |
| `--no-trim` | fingerprint region 트리밍 스킵 | |
| `--skip-qc` | QC 스킵, 전체 스펙트럼 사용 | |
| `--raw-only` | main.py만 실행 (QC pipeline 없이) | |

---

## train

모델 학습. 모델별 서브커맨드 제공.

### resnet18 (기본)

```bash
sers train resnet18
sers train resnet18 --epochs 200 --lr 3e-4
sers train resnet18 --use-focal-loss --focal-gamma 2.0
sers train resnet18 --aggregate none --n-splits 10
```

### lr (Logistic Regression)

```bash
sers train lr
sers train lr --logreg-c 0.1 --logreg-max-iter 5000
```

### rf (Random Forest)

```bash
sers train rf
sers train rf --rf-n-estimators 500 --rf-max-depth 10
```

### xgboost

```bash
sers train xgboost
sers train xgboost --xgb-n-estimators 300 --xgb-max-depth 5 --xgb-learning-rate 0.1
```

### cnn1d

```bash
sers train cnn1d
sers train cnn1d --epochs 100 --lr 1e-3
```

### stacking (Stacking Ensemble)

```bash
sers train stacking
sers train stacking --dry-run                      # 빠른 테스트 (1 fold, 3 base, 2 meta)
sers train stacking --val-group SPAN --meta-learner elasticnet
sers train stacking --no-shap --no-cm
```

| 옵션 | 설명 |
|------|------|
| `--dry-run` | 빠른 테스트 모드 |
| `--val-group` | 외부 검증 그룹 (예: `SPAN`) |
| `--meta-learner` | 메타 러너 (`elasticnet` 등) |
| `--no-shap` | SHAP 스킵 |
| `--no-cm` | Confusion matrix 스킵 |
| `--shap-samples` | SHAP background 샘플 수 |

### multichannel

```bash
sers train multichannel
sers train multichannel --epochs 200 --lr 1e-4
sers train multichannel --aggregate none --cancer-types PRO LUN CRC
```

### tvt (Train/Val/Test Split)

```bash
sers train tvt                                     # 60/20/20 split × 5회
sers train tvt --n-repeats 10 --seed 123
sers train tvt --exclude-patients exclusions.csv
```

### 공통 학습 옵션

resnet18, lr, rf, xgboost, cnn1d에 공통 적용:

| 옵션 | 설명 |
|------|------|
| `-i, --input` | 입력 스펙트럼 CSV |
| `-o, --output` | 출력 디렉토리 |
| `-c, --config` | Config YAML |
| `--experiment` | 실험 이름 |
| `--version` | 버전 태그 |
| `--cancer-types` | 암종 (반복 가능) |
| `--non-cancer-groups` | 비암 그룹 (반복 가능) |
| `-a, --aggregate` | 집계 방법 (`medoid` / `mean` / `none`) |
| `--n-splits` | CV fold 수 |
| `--epochs` | 학습 에포크 |
| `--batch-size` | 배치 크기 |
| `--lr` | 학습률 |
| `--device` | 디바이스 (`auto` / `cuda` / `cpu`) |
| `--no-amp` | Mixed precision 비활성화 |
| `--no-mlflow` | MLflow 로깅 비활성화 |
| `--phase` | Phase 라벨 |
| `--tags` | 태그 (반복 가능) |
| `--exclude-patients` | 제외 환자 CSV |

---

## benchmark

여러 모델을 동시에 비교 학습.

```bash
sers benchmark resnet18 lr xgboost
sers benchmark resnet18 lr rf --n-splits 10
sers benchmark resnet18 lr xgboost --no-mlflow
```

모델 약어: `resnet18`, `lr`, `rf`, `xgboost`, `cnn1d`

---

## test

학습된 모델 평가 (ROC, Confusion Matrix, SHAP, t-SNE, GradCAM).

```bash
sers test
sers test -i results/training
sers test -i results/training --no-shap --tsne-dim 2
sers test --top-k-features 50 --shap-samples 200
sers test --val-group SPAN --aggregate mean
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `-i, --input` | 모델 디렉토리 | `models/results/_archive/resnet18_medoid_v1` |
| `--processed-csv` | 스펙트럼 CSV | `results/processed_spectra.csv` |
| `--device` | 디바이스 | `auto` |
| `--tsne-dim` | t-SNE 차원 (`2` / `3`) | `3` |
| `--no-shap` | SHAP 스킵 | |
| `--no-feature-selection` | Feature selection 스킵 | |
| `--no-gradcam` | GradCAM 스킵 | |
| `--top-k-features` | 표시할 상위 feature 수 | `30` |
| `--shap-samples` | SHAP background 샘플 | `100` |
| `--shap-explain` | SHAP explain 샘플 | `200` |
| `--val-group` | 외부 검증 그룹 | |
| `-a, --aggregate` | 집계 방법 | `mean` |

---

## predict

스펙트럼 CSV에서 암 스크리닝 추론.

```bash
sers predict spectrum.csv
sers predict *.CSV --age 55 --sex M --bmi 24.3
sers predict sample.csv --mode screening --instrument medical
sers predict batch/*.CSV -o results.json -q
```

| 옵션 | 설명 |
|------|------|
| `SPECTRA` | 스펙트럼 CSV 파일 (1개 이상, 필수) |
| `--age` | 환자 나이 |
| `--sex` | 환자 성별 (`M` / `F`) |
| `--bmi` | 환자 BMI |
| `-m, --mode` | 운영 모드 (`screening` / `balanced` / `confirmatory`) |
| `-i, --instrument` | 장비 (`thermo` / `medical`) |
| `--model-dir` | 모델 아티팩트 디렉토리 |
| `-o, --output` | 출력 JSON 경로 |
| `-q, --quiet` | 텍스트 출력 억제, JSON만 |

---

## serve

SERS 웹 애플리케이션 실행.

```bash
sers serve                                         # 기본 webapp :8000
sers serve -p 9000                                 # 포트 변경
sers serve --clinical                              # IEC 62366 임상용 :8080
sers serve --clinical -p 8080 --reload             # 개발 모드
sers serve --host 127.0.0.1 -p 5000               # localhost만
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--clinical` | 임상용 인터페이스 | 기본 webapp |
| `--host` | 호스트 주소 | `0.0.0.0` |
| `-p, --port` | 포트 | `8000` (basic) / `8080` (clinical) |
| `--reload` | 코드 변경 시 자동 리로드 (clinical만) | |

---

## qc

QC 검증 파이프라인 (13개 phase).

```bash
sers qc list                                       # phase 목록 확인
sers qc phase 01                                   # 개별 phase 실행
sers qc phase 06b                                  # Calibration diagnostics
sers qc all                                        # 전체 순차 실행
sers qc all --stop-on-error                        # 실패 시 중단
```

### Phase 목록

| ID | 설명 |
|----|------|
| `01` | Literature standard validation |
| `02` | Statistical optimization |
| `03` | Clinical impact analysis |
| `04` | Report generation |
| `05` | Cross-instrument variance |
| `06` | Preprocessing validation |
| `06b` | Calibration diagnostics |
| `06c` | Batch leakage detection |
| `06d` | Two-stage QC testing |
| `07` | Threshold optimization |
| `08` | Hospital confounding analysis |
| `09` | Hospital stratification |
| `10` | Within-hospital cancer type |

---

## data

데이터 관리: 표준화, 검증, 업로드.

```bash
sers data standardize                              # 임상 데이터 표준화
sers data standardize --dry-run                    # 미리보기만
sers data validate                                 # 스펙트럼 파일 검증
sers data validate -i ./data/raw_data              # 특정 디렉토리
sers data upload                                   # Supabase 업로드 (기본)
sers data upload --target postgres --drop          # PostgreSQL (테이블 재생성)
sers data upload -t postgres --host localhost      # PostgreSQL 커스텀 호스트
```

---

## production

Production 모델 아티팩트 빌드 (전체 데이터로 학습).

```bash
sers production                                    # 기본 모델
sers production --stacking                         # Stacking 모델
sers production -o models/production               # 출력 경로 지정
sers production --fit-pds --grid models/production/common_grid.npy  # PDS 캘리브레이션 포함
```

| 옵션 | 설명 |
|------|------|
| `--stacking` | Stacking production 모델 빌드 |
| `-o, --output-dir` | 출력 디렉토리 |
| `--fit-pds` | PDS 캘리브레이션 아티팩트도 생성 |
| `--grid` | common_grid.npy 경로 (`--fit-pds` 용) |
| `--pds-out` | PDS 출력 .npz 경로 (`--fit-pds` 용) |

---

## analyze

분석 및 시각화 도구.

### tsne

선택한 그룹들의 t-SNE 시각화.

```bash
sers analyze tsne -g YPAN -g SPAN -g CPAN -g NOR -g DIA -g HBP -g YNOR -g H.D.
sers analyze tsne -g PRO -g CRC -g NOR --dim 3
sers analyze tsne -g PRO -g NOR --aggregate medoid --perplexity 50
sers analyze tsne -g YPAN -g SPAN -g CPAN -g NOR -o figures/tsne_pancreatitis.png
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `-g, --groups` | 포함할 그룹 (반복, 필수) | |
| `-i, --input` | 스펙트럼 CSV | `results/processed_spectra.csv` |
| `-o, --output` | 출력 이미지 경로 (자동 생성) | |
| `--dim` | t-SNE 차원 (`2` / `3`) | `2` |
| `--perplexity` | t-SNE perplexity | `30` |
| `-a, --aggregate` | 환자별 집계 (`none` / `mean` / `medoid`) | `none` |
| `--pca-init` | PCA 사전 축소 차원 (0=스킵) | `50` |

**사용 가능한 그룹:** PRO, BRE, OVA, LUN, CRC, BLC, CPAN, SPAN, YPAN, NOR, DIA, HBP, H.D., YNOR

### 기타

```bash
sers analyze explore                  # 데이터 탐색
sers analyze cross-inst               # 장비간 비교 분석
sers analyze equipment                # 장비 QC 분석
```

---

## 기존 스크립트 호환

통합 CLI와 기존 경로 기반 호출 모두 동작합니다:

```bash
# 동일한 동작
sers train resnet18 --epochs 200
python models/train.py --model resnet18 --epochs 200

sers preprocess -n snv
python scripts/pipeline/run_qc_preprocess.py --normalization snv

sers predict sample.csv
python scripts/deployment/sers_predict.py sample.csv
```
