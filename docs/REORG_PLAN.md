# SERS-AI 프로젝트 재구성 계획 (확정판)

**목표 레이아웃**: PyPA `src-layout` (a) + Cookiecutter Data Science (b) 혼합
**작성일**: 2026-04-15 (개정)
**상태**: 📋 계획 확정 — 단계별 이동 실행 대기

**참조 문서**:
- `docs/DIRECTORY_CONVENTIONS.md` — 디렉토리 정의 및 분별 규칙 (정본)
- `docs/MODEL_STATUS.md` — 모델별 상태 및 이동 목적지 (정본)

---

## 사용자 확정 사항 (Q1~Q4)

| 질문 | 답 |
|---|---|
| Q1. Clinical fusion 통합 범위 | **뼈대만** — `clinical_fusion.py` 파일 생성 + ResNet 의존 제거, meta 통합은 별도 task (v1.1.0) |
| Q2. Legacy CLI 노출 | **유지** — `sers train --model resnet18`, `--model lr-fusion`, `--model sersnet-ensemble` 지원 |
| Q3. `sers compare` 명령어 | **신규 작성** — 여러 모델 비교 리포트 자동화 |
| Q4. 버전 번호 체계 | **동의** — `v1.0.0` = 현 STK-V2 spectral only, `v1.1.0` = clinical fusion 추가 |

### 추가 확정
- `run_sersnet.py` (LR+ResNet+Clinical 블렌딩 앙상블) → **baseline-legacy 승격** (uSERS-Net v1.1.0 공정 비교 baseline)
- `MODEL_STATUS.md` 도 Obsidian vault 동기화 대상에 포함

---

## 최종 목표 트리

```
SERS-AI/
├── src/sers/                            # 라이브러리
│   ├── io/                              # io.py, ingest, reingest
│   ├── preprocessing/                   # core, signal, calibration_transfer
│   ├── qc/
│   ├── features/                        # scoring, analysis
│   ├── models/
│   │   ├── _registry.py                 # ★ 정본 Registry
│   │   ├── usersnet/                    # 🟢 Production
│   │   │   ├── __init__.py              # public API
│   │   │   ├── stacking.py              # (구) stacking_utils
│   │   │   └── clinical_fusion.py       # 뼈대 (v1.1.0 통합 예정)
│   │   ├── _legacy/                     # 📘 Baseline
│   │   │   ├── resnet_v1/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── model.py
│   │   │   │   └── dataset.py
│   │   │   └── lr_fusion_v1/
│   │   │       └── fusion.py
│   │   └── experimental/                # 🧪 연구
│   │       ├── contrastive/
│   │       ├── cross_attention/
│   │       ├── film/
│   │       ├── transformer/
│   │       └── blc_mfds/
│   ├── validation/                      # 패키지로 통합
│   ├── visualization/                   # 패키지로 통합
│   ├── cli/
│   │   ├── train.py                     # registry 기반 리팩토링
│   │   ├── compare.py                   # ★ 신규 (sers compare)
│   │   └── ...
│   ├── config.py
│   └── logging_config.py
│
├── scripts/
│   ├── pipeline/
│   ├── training/
│   │   ├── train_usersnet.py
│   │   ├── build_usersnet_production.py
│   │   └── _legacy/
│   │       ├── train_resnet.py
│   │       └── build_lr_fusion.py
│   ├── evaluation/
│   │   ├── eval_usersnet_holdout.py
│   │   └── _legacy/
│   │       └── eval_resnet.py
│   ├── experiments/
│   │   └── _archive/                    # 완료된 ablation
│   ├── analysis/
│   ├── figures/
│   │   └── generate_architecture.py
│   ├── deployment/
│   ├── db/
│   │   └── sql_explorer.py              # ← scripts/ 루트에서 이동
│   └── qc_validation/
│
├── artifacts/                           # ★ 신규 최상위
│   ├── usersnet/
│   │   ├── v1.0.0/
│   │   └── current → v1.0.0
│   ├── baselines/
│   │   ├── resnet18/v1.0.0/
│   │   └── lr-fusion/v1.0.0/
│   └── _legacy/
│
├── results/                             # ★ 최상위로 승격 (models/results 에서)
│   ├── runs/
│   │   └── {YYYY-MM-DD}_{model}_v{ver}_{mode}/
│   ├── comparisons/                     # sers compare 출력
│   └── benchmarks/                      # 구 01_benchmarks
│
├── figures/
│   ├── paper/
│   ├── poster/
│   └── slides/
│
├── config/  data/  docs/  tests/  notebooks/  logs/  infra/
└── pyproject.toml
```

---

## Phase 계획 (bottom-up, 저영향 → 고영향)

### ✅ Phase 0 — 사전 준비 (문서화, 현재 완료됨)

- [x] `docs/DIRECTORY_CONVENTIONS.md` 작성
- [x] `docs/MODEL_STATUS.md` 작성
- [x] `docs/REORG_PLAN.md` 개정 (이 문서)
- [x] `src/sers/models/_registry.py` + `__init__.py` 작성 (2026-04-15)
- [x] `tests/test_model_registry.py` 작성 — registry ↔ MODEL_STATUS.md 키 동기화 검증

---

### 🟢 Phase 1 — 저영향 이동 (import 참조 0-2)

목표: 삭제 대상 + 라이브러리 성격 명확한 저영향 파일부터.

#### 1-A. 일회성 스크립트 삭제 (산출물 확인 완료)
```bash
git rm src/sers/ingest_ypan.py
git rm src/sers/reingest_staging.py
git rm src/sers/update_lun_dates.py
```
근거: 모든 산출물이 `data/clinical_data/standardized/`, `AACR/data/`, `publications/aacr/data/` 에 존재 확인.

#### 1-B. `src/sers/` 평면 라이브러리 모듈 → 서브패키지화

| 현재 | 이동 후 | refs |
|---|---|---|
| `src/sers/io.py` | `src/sers/io/core.py` | 3 |
| `src/sers/preprocessing.py` | `src/sers/preprocessing/core.py` | 2 |
| `src/sers/signal.py` | `src/sers/preprocessing/signal.py` | 1 |
| `src/sers/calibration_transfer.py` | `src/sers/preprocessing/calibration_transfer.py` | 0 |
| `src/sers/scoring.py` | `src/sers/features/scoring.py` | 0 |
| `src/sers/analysis.py` | `src/sers/features/analysis.py` | 0 |
| `src/sers/validation.py` | `src/sers/validation/core.py` (기존 `validation/` 와 통합) | 2 |
| `src/sers/visualization.py` | `src/sers/visualization/legacy.py` (기존 `visualization/` 와 통합) | 0 |

각 서브패키지 `__init__.py` 에서 기존 import 경로 유지용 re-export:
```python
# src/sers/io/__init__.py
from .core import *  # noqa
```

#### 1-C. `scripts/` 루트 평면 파일 정리

| 현재 | 이동 후 |
|---|---|
| `scripts/migrate_figures.py` | `scripts/figures/migrate_figures.py` |
| `scripts/sql_explorer.py` | `scripts/db/sql_explorer.py` |
| `scripts/run_benchmark.ps1` | `scripts/training/run_benchmark.ps1` |
| `scripts/setup_repo.bat` | `infra/setup_repo.bat` |
| `scripts/__init__.py` | 삭제 |

#### 1-D. `figures/` 정리

| 현재 | 이동 후 |
|---|---|
| `figures/usersnet_architecture.py` | `scripts/figures/generate_architecture.py` |
| `figures/usersnet_architecture.{pdf,png,svg}` | `figures/paper/architecture/` |
| `figures/table1_demographics.{tex,pdf,png}` | `figures/paper/` |
| `figures/table1_demographics.{aux,log}` | 삭제 + `.gitignore` |
| `figures/multimodal_slide.html`, `multimodal_staircase.html` | `figures/slides/` |
| `figures/training/stacking_v2_holdout/*` | `results/runs/2026-04-14_usersnet_v1.0.0_holdout/figures/` |
| `figures/training/stacking_v2_holdout.zip` | 삭제 (원본 풀려있음) |

---

### 🟡 Phase 2 — Artifacts / Results 분리 (import 영향 없음, 경로 문자열만)

#### 2-A. 학습된 아티팩트 이동
```bash
git mv models/production_stacking/ artifacts/usersnet/v1.0.0/
git mv models/production/          artifacts/baselines/lr-fusion/v1.0.0/
ln -s v1.0.0 artifacts/usersnet/current
```

#### 2-B. 결과 이동
```bash
git mv models/results/ results-tmp/
# 01_benchmarks 등을 results/benchmarks/ 로 재배치
mkdir -p results/benchmarks
mv results-tmp/01_benchmarks/* results/benchmarks/
mv results-tmp/02_tuning results/tuning/
mv results-tmp/03_learning_curves results/learning_curves/
mv results-tmp/04_comparisons results/comparisons_legacy/
mv results-tmp/05_subset_analysis results/subset_analysis/
mv results-tmp/README.md results/
mv results-tmp/_archive results/_archive
rmdir results-tmp
```

#### 2-C. 경로 문자열 업데이트 (grep 스캔 필요)
- `config/config.yaml` 내 `models/production*` → `artifacts/usersnet/current`
- `config/config.yaml` 내 `models/results/*` → `results/*`
- `scripts/deployment/sers_predict.py`
- `src/sers/cli/production.py`
- `models/manifest.json` 내 상대경로 (이 파일은 삭제 예정)
- `.gitignore` — `artifacts/`, `results/runs/`, `logs/`, `*.aux`, `*.log` 추가

---

### 🟡 Phase 3 — Experimental/Legacy 격리

#### 3-A. Experimental 모델 이동
```bash
git mv models/contrastive/      src/sers/models/experimental/contrastive/
git mv models/cross_attention/  src/sers/models/experimental/cross_attention/
git mv models/film/             src/sers/models/experimental/film/
git mv models/transformer/      src/sers/models/experimental/transformer/
git mv models/blc_mfds/         src/sers/models/experimental/blc_mfds/
```

각 폴더에 `README.md` 생성 (목적, 가설, 마지막 실험, 현 상태).

#### 3-B. Archived 실험 이동 (시기 태그 추가)

`MODEL_STATUS.md` 의 "📦 Archived" 표 그대로 실행:
- `models/run_*.py` (실험성) → `scripts/experiments/_archive/YYYY-MM_{name}.py`
- `models/train_ensemble.py`, `train_multichannel.py`, `run_step4_normalization.py` → `_archive/`
- `models/tune_torch.py`, `pancreatic_experiment.py`, `run_pan_improvement.py` → `_archive/`
- `models/learning_curve_comparison.py`, `plot_overall_comparison.py` → `scripts/analysis/` 또는 `scripts/figures/`
- `models/run_clinical_analysis.py`, `run_confounding_analysis.py` → `scripts/analysis/`
- `models/xai_fusion_analysis.py` → `scripts/analysis/xai_fusion_analysis.py`

`scripts/experiments/_archive/README.md` 생성: 각 실험의 결과 위치 + 결론 요약.

---

### 🔴 Phase 4 — 고영향 이동 (import 27, 30)

Phase 4 는 이전 Phase 가 끝나고 **shim 없이** 한 번에 처리.

#### 4-A. uSERS-Net (Production) 이동
```bash
git mv models/stacking_utils.py   src/sers/models/usersnet/stacking.py
git mv models/clinical_utils.py   src/sers/models/usersnet/clinical_fusion.py
# clinical_fusion.py: ResNet 의존 (from models.model import SERSDataset) 제거
# 뼈대만 유지 — 실제 stacking meta 통합은 v1.1.0 task
```

```bash
git mv models/train_stacking.py              scripts/training/train_usersnet.py
git mv models/build_production_stacking.py   scripts/training/build_usersnet_production.py
git mv models/eval_stacking_holdout.py       scripts/evaluation/eval_usersnet_holdout.py
```

Import 일괄 수정 (sed 스크립트):
```
from models.stacking_utils → from sers.models.usersnet.stacking
from models.clinical_utils → from sers.models.usersnet.clinical_fusion
```

#### 4-B. Legacy ResNet 이동 + 격리
```bash
git mv models/model.py       src/sers/models/_legacy/resnet_v1/model.py
git mv models/train.py       scripts/training/_legacy/train_resnet.py
git mv models/test.py        scripts/evaluation/_legacy/eval_resnet.py
git mv models/build_production_model.py  scripts/training/_legacy/build_lr_fusion.py
```

Import 일괄 수정:
```
from models.model → from sers.models._legacy.resnet_v1.model
```

`test.py` 인코딩 복구 (docstring 의 `??` → `—`).

#### 4-C. Registry 및 CLI 완성
- `src/sers/models/_registry.py` — 모든 모델 ModelSpec 정의
- `src/sers/cli/train.py` — `--model {key}` 플래그 지원, registry 참조
- `src/sers/cli/compare.py` — **신규 `sers compare` 명령어**
  ```bash
  sers compare --models usersnet,resnet18,lr-fusion \
               --report results/comparisons/usersnet_vs_baselines/
  ```
- `src/sers/cli/evaluate.py` — `--model {key}` 지원

---

### 🧹 Phase 5 — Cleanup

- [ ] `models/` 폴더 제거 (빈 껍데기만 남음)
- [ ] `models/__init__.py`, `models/__pycache__/`, `models/manifest.json` 삭제
- [ ] `.gitignore` 최종 업데이트
- [ ] `pyproject.toml` packages 경로 업데이트
- [ ] `docs/PROJECT_STRUCTURE.md` 업데이트
- [ ] `CLAUDE.md` 내 폴더 설명 업데이트
- [ ] 테스트 실행 (`pytest tests/`) 로 회귀 확인

---

## 영향 받는 외부 참조 (사전 스캔)

### config/config.yaml
- [ ] `models/production*` 경로 → `artifacts/usersnet/current`
- [ ] `models/results/*` → `results/*`

### pyproject.toml
- [ ] `[tool.setuptools.packages.find]` — `src/` 기반 설정 확인

### .gitignore (추가 필요)
```
artifacts/
results/runs/
results/_archive/
*.aux
*.log
*.zip
__pycache__/
```

### 문서
- [ ] `docs/PROJECT_STRUCTURE.md`
- [ ] `CLAUDE.md` (프로젝트 구조 섹션)
- [ ] `docs/CLI.md`
- [ ] `README.md` (있다면)

---

## 실행 순서 (체크리스트)

```
[ ] Phase 0 — 문서 확정
    [x] DIRECTORY_CONVENTIONS.md
    [x] MODEL_STATUS.md
    [x] REORG_PLAN.md (이 문서)
    [ ] _registry.py 초안

[x] Phase 1 — 저영향 이동 (2026-04-15 완료, 194 tests passed)
    [x] 1-A 일회성 삭제 (ingest_ypan, reingest_staging, update_lun_dates)
    [x] 1-B src/sers/ 패키지화 (io/, preprocessing/, features/, validation/ 통합)
        - sers.signal shim 유지 (deprecated, 이후 제거 예정)
        - visualization/legacy.py 보존 (기존 shadow된 dead code)
    [x] 1-C scripts/ 루트 정리 (migrate_figures→figures/, sql_explorer→db/, setup_repo→infra/)
    [x] 1-D figures/ 정리 (paper/, slides/ 분리 + LaTeX 중간산출 삭제 + training→results/runs/)

[x] Phase 2 — Artifacts/Results 분리 (2026-04-15 완료, 194 tests pass)
    [x] 2-A production_stacking → artifacts/usersnet/v1.0.0/ + current symlink
    [x] 2-B production → artifacts/baselines/lr-fusion/v1.0.0/
    [x] 2-C models/results → results/ (benchmarks, tuning, learning_curves, etc.)
    [x] 2-D 경로 문자열 업데이트 (sers_predict, cli/production, BUILD_GUIDE, etc.)

[x] Phase 3 — Experimental/Legacy 격리 (2026-04-15 완료, 194 tests pass)
    [x] 3-A Experimental → src/sers/models/experimental/{contrastive,cross_attention,film,transformer,blc_mfds}/ + README 5종
    [x] 3-B Archived → scripts/experiments/_archive/ (YYYY-MM_ prefix) + README
    [x] 3-B2 사후 분석 → scripts/analysis/ (clinical_breakdown, confounding_analysis, xai_fusion_analysis, learning_curve_comparison)

[x] Phase 4 — 고영향 이동 (2026-04-15 완료, 194 tests pass)
    [x] 4-A uSERS-Net 이동: stacking_utils/clinical_utils → sers.models.usersnet/*
        train_stacking → scripts/training/train_usersnet.py
        build_production_stacking → scripts/training/build_usersnet_production.py
        eval_stacking_holdout → scripts/evaluation/eval_usersnet_holdout.py
    [x] 4-B Legacy 격리:
        model.py → sers.models._legacy.resnet_v1.model
        train.py → scripts/training/_legacy/train_resnet.py
        test.py → scripts/evaluation/_legacy/eval_resnet.py
        build_production_model.py → scripts/training/_legacy/build_lr_fusion.py
        run_sersnet.py → scripts/training/_legacy/run_sersnet_ensemble.py
    [x] 4-B2 Import 일괄 수정 (sed): models.model/stacking_utils/clinical_utils/experimental/*
    [x] 4-B3 models/ compat shim (train, train_stacking, run_train_val_test) — deprecation warning
    [x] 4-C Registry + CLI
        - src/sers/models/_registry.py (9 models: 1 production + 3 baseline + 5 experimental)
        - src/sers/cli/compare.py (sers compare —models usersnet,resnet18,...)
        - src/sers/models/usersnet/__init__.py (public API)
        - src/sers/models/_legacy/__init__.py, _legacy/resnet_v1/__init__.py
        - src/sers/models/experimental/__init__.py

[x] Phase 5 — Cleanup (2026-04-15)
    [x] models/ 실질적 제거 (compat shim 4 파일 + MODEL_STATUS.md는 docs/ 로 이동)
    [x] pyproject.toml packages = src/ 기반 유지 (수정 불필요)
    [x] tests/test_model_registry.py 경로 갱신 (models/MODEL_STATUS.md → docs/MODEL_STATUS.md)
    [x] 전체 회귀 테스트 194/194 pass
```

**재구성 완료 — 총 소요: 약 1.5시간 (집중 작업)**

**총 예상 작업량**: 6~8시간 (집중 작업 시)
