# SERS-AI Directory Conventions

**목적**: 모든 파일이 단 하나의 디렉토리로 귀속되도록 하는 결정 규칙.
**작성일**: 2026-04-15
**상태**: ✅ 승인됨 (사용자 확정)

---

## 판정 프레임워크

애매한 파일을 만나면 다음 **4가지 속성** 으로 판정:

1. **생산자** — 누가 만드는가? (개발자 / 학습 스크립트 / 평가 스크립트 / 외부 수집 등)
2. **소비자** — 누가 쓰는가? (`import` / CLI 실행자 / production inference / 사람)
3. **재생성 가능성** — 코드+데이터로 다시 만들 수 있는가?
4. **버저닝 필요성** — 이전 버전을 보존해야 하는가?

---

## 7가지 최상위 디렉토리의 정식 정의

| 디렉토리 | 생산자 | 주 소비자 | 재생성 | 버저닝 | git |
|---|---|---|---|---|---|
| `src/sers/` | 개발자 | `import` (런타임) | — | git tag | ✅ |
| `scripts/` | 개발자 | 사람이 CLI 실행 | — | git tag | ✅ |
| `config/` | 개발자 | scripts 가 읽음 | — | git tag | ✅ |
| `data/` | 외부(수집) | preprocessing | ❌ (원본) | snapshot hash | ❌ |
| `artifacts/` | training script | production inference | ✅ | **semver** | LFS/❌ |
| `results/` | training/eval script | 사람이 비교·리포트 | ✅ | run-id | 선택 |
| `figures/` | figure script | 사람이 봄 (논문/슬라이드) | ✅ | paper version | ✅ |

---

## 7가지 분별 질문 (한 번에 결정)

```
Q1. 다른 파이썬 코드가 import 하는가? ──────────► src/sers/
Q2. CLI 에서 사람이 실행하는 파이썬 파일인가? ──► scripts/
Q3. production inference 가 로드하는가? ────────► artifacts/
Q4. 실험별로 비교하는 수치/로그/예측인가? ─────► results/
Q5. 논문·포스터·슬라이드에 들어가는 최종 이미지? ► figures/
Q6. 설정값 (YAML/JSON/TOML) 인가? ──────────────► config/
Q7. 입력 원본/중간 데이터인가? ─────────────────► data/
```

**한 파일이 두 곳에 해당되지 않아야 함.** 해당된다면 파일을 쪼개야 한다는 신호.

---

## 자주 틀리는 경계 케이스 (고정 판정)

### 1. `*.joblib` — artifact vs result
```
"이 파일이 없으면 production 이 멈추는가?"
  YES → artifacts/usersnet/v1.0.0/meta_s1.joblib
  NO  → results/runs/{run-id}/fold_3_model.joblib
```

### 2. 혼동행렬 PNG — figure vs result
```
"누가 보는가?"
  실험 품질 점검   → results/runs/{run-id}/figures/confusion_matrix.png
  논문·포스터 수록 → figures/paper/figure_3_confusion_matrix.pdf
```
동일 내용이라도 용도가 다르면 **두 벌 유지**. 스타일·해상도·legend 가 다름.

### 3. `fold_predictions.npz` — artifact vs result
```
production 이 로드하지 않음 → results/runs/{run-id}/predictions/
```

### 4. `build_production_*.py`, `train_*.py`, `eval_*.py` — script vs src/
```
__main__ + argparse 있음 → scripts/
```

### 5. `*_utils.py` — script vs src/
```
다른 파일이 import 함 → src/sers/
```

### 6. Jupyter notebook — 어디?
```
탐색 (재현 목적 아님) → notebooks/
재현 가능 분석       → 코드는 scripts/figures/, 결과는 results/
```

### 7. Manifest — 어떤 manifest?
```
artifacts/usersnet/v1.0.0/manifest.json  ← 이 버전의 "무엇" (아키텍처, 학습일, 데이터 hash)
results/runs/{run-id}/manifest.json      ← 이 실행의 "어떻게" (명령, config, metrics, git commit)
```

---

## `src/sers/` 내부 구조 (Layer 기반)

```
src/sers/
├── io/                       # 파일 I/O, ingest
├── preprocessing/            # QC 이외 전처리, signal, calibration
├── qc/                       # QC 로직
├── features/                 # scoring, analysis
├── models/                   # 모델 정의 (import 대상)
│   ├── usersnet/             # 🟢 Active production
│   │   ├── __init__.py       # public API: uSERSNet, load_production
│   │   ├── stacking.py       # Level-0 base + Level-1 meta
│   │   └── clinical_fusion.py # 뼈대 (v1.1.0에서 통합 예정)
│   ├── _legacy/              # 📘 Baseline 비교용
│   │   ├── resnet_v1/
│   │   │   ├── model.py
│   │   │   └── dataset.py
│   │   └── lr_fusion_v1/
│   │       └── fusion.py
│   └── experimental/         # 🧪 연구 모델
│       ├── contrastive/
│       ├── cross_attention/
│       ├── film/
│       ├── transformer/
│       └── blc_mfds/
├── validation/               # validation.py + validation/ 통합
├── visualization/            # visualization.py + visualization/ 통합
├── cli/                      # CLI 엔트리포인트 (얇음)
├── config.py                 # 설정 로더 (루트 유지)
└── logging_config.py         # 로깅 설정 (루트 유지)
```

---

## `scripts/` 내부 구조 (동사 기반)

```
scripts/
├── pipeline/                 # 데이터 변환 (QC, 전처리, 표준화)
├── training/                 # 모델 학습 (train_*, build_*, tune_*)
│   └── _legacy/              # legacy 모델 학습 (baseline 비교용)
├── evaluation/               # 모델 평가 (eval_*, test_*)
│   └── _legacy/
├── experiments/              # 일회성 실험·ablation (run_*, *_experiment)
│   └── _archive/             # 완료된 실험 (결과만 results/ 에 남음)
├── analysis/                 # 사후 분석 (subgroup, confounding, interpretation)
├── figures/                  # 논문/포스터 figure 생성 코드
├── deployment/               # sers_predict, webapp, API
├── db/                       # DB 업로드/쿼리, SQL
└── qc_validation/            # QC 검증 4단계
```

**이동 규칙**:
- `train_*.py` → `training/`
- `build_production_*.py` → `training/`
- `eval_*.py`, `test_*.py` → `evaluation/`
- `run_*.py` (실험적) → `experiments/`
- `run_qc_*.py` (파이프라인) → `pipeline/`
- `plot_*.py`, `*_figures.py` → `figures/`
- `*_analysis.py` (사후 해석) → `analysis/`

---

## `artifacts/` 내부 구조 (Semver)

```
artifacts/
├── usersnet/
│   ├── v1.0.0/              # 첫 릴리즈 (STK-V2, spectral only)
│   │   ├── manifest.json
│   │   ├── base_models/
│   │   ├── meta_s1.joblib
│   │   ├── meta_s2.joblib
│   │   ├── preprocessing.json
│   │   └── common_grid.npy
│   ├── v1.1.0/              # Clinical Fusion 추가 (예정)
│   │   └── clinical_scaler.joblib (추가)
│   └── current → v1.0.0     # symlink — production 포인터
├── baselines/
│   ├── resnet18/v1.0.0/     # 📘 Legacy ResNet 유지
│   └── lr-fusion/v1.0.0/    # 📘 구 production (ResNet+LR)
└── _legacy/                 # registry 에서 완전 은퇴한 모델
```

### Semver 규칙 (SERS-AI 특화)

| 버전 변경 | 조건 |
|---|---|
| **MAJOR** v1→v2 | 아키텍처 변경 (stacking → transformer 등) |
| **MINOR** v1.0→v1.1 | 피처 추가 (clinical fusion 통합), 새 입력 변수 |
| **PATCH** v1.1.0→v1.1.1 | 동일 설정 재학습 (데이터 추가, 동일 아키텍처) |

### Manifest 스키마

```json
{
  "model_name": "uSERS-Net",
  "model_key": "usersnet",
  "version": "1.0.0",
  "parent_version": null,
  "internal_architecture": "stacking_v2",
  "trained_at": "2026-04-15T14:30:00+09:00",
  "training_script": "scripts/training/train_usersnet.py",
  "training_commit": "abc1234",
  "data_snapshot": {
    "clinical_csv_hash": "sha256:...",
    "spectra_count": 2847,
    "subject_count": 299
  },
  "config": { "meta_learner": "ElasticNet", "C": 0.5, "l1_ratio": 0.5 },
  "metrics": {
    "holdout": {"auc_binary": 0.996, "auc_multi": 0.92},
    "nested_cv": {"auc_binary_mean": 0.983, "auc_binary_std": 0.011}
  },
  "changes": ["initial release"]
}
```

### Production 포인터

`artifacts/usersnet/current` symlink 가 현 production.
`scripts/deployment/sers_predict.py` 는 `artifacts/usersnet/current/` 만 참조.
롤백 = symlink 재지정.

---

## `results/` 내부 구조 (Run-id)

```
results/
├── runs/                                    # 개별 실행 (append-only)
│   ├── 2026-04-15_usersnet_v1.0.0_holdout/
│   │   ├── manifest.json                    # 명령, config, git commit
│   │   ├── metrics.json                     # AUC, F1 등
│   │   ├── predictions/
│   │   │   ├── fold_predictions.npz
│   │   │   └── per_subject.csv
│   │   ├── figures/                         # 진단용 (논문용 아님)
│   │   │   ├── confusion_matrix.png
│   │   │   └── roc_curve.png
│   │   └── logs/
│   └── 2026-04-16_resnet18_v1.0.0_holdout/
├── comparisons/                             # sers compare 출력
│   └── 2026-04-16_usersnet_vs_baselines/
│       ├── report.md
│       ├── metrics_table.csv
│       └── figures/
└── benchmarks/                              # 표준 벤치마크 추적
    └── README.md
```

**Run-id 규칙**: `YYYY-MM-DD_{model_key}_v{version}_{split_mode}[_{tag}]`
예: `2026-04-15_usersnet_v1.1.0_holdout_with_clinical`

---

## `figures/` 내부 구조 (Consumer 기반)

```
figures/
├── paper/                    # 논문 최종 figure
│   ├── architecture/
│   ├── figure_1_pipeline.pdf
│   ├── figure_3_confusion_matrix.pdf
│   └── table1_demographics.pdf
├── poster/                   # 포스터 (AACR 등)
├── slides/                   # 발표용 (HTML, keynote)
│   ├── multimodal_slide.html
│   └── multimodal_staircase.html
└── training/                 # ⚠️ 실제로는 results/runs/{run-id}/figures/ 로 이동 예정
```

**삭제 대상**: LaTeX 중간산출물 `*.aux`, `*.log` → `.gitignore` 에 추가.

---

## 모델 카테고리 (4-tier)

| 분류 | 기준 | 처리 |
|---|---|---|
| 🟢 **production** | 현 운영 모델 | `src/sers/models/usersnet/` |
| 📘 **baseline-legacy** | 비교용 유지 (구 파이프라인) | `src/sers/models/_legacy/` |
| 🧪 **experimental** | 연구·실험 모델 | `src/sers/models/experimental/` |
| 📦 **archived** | 완료된 ablation, 결과만 남김 | `scripts/experiments/_archive/` |

---

## 명명 통일

### 모델 공개 이름 = "uSERS-Net"

- 내부 구현이 stacking/transformer/etc. 로 바뀌어도 공개 이름은 유지.
- Registry 에 `display_name: "uSERS-Net"` 으로 고정.
- 논문·리포트·CLI 출력·SHAP 라벨 모두 "uSERS-Net" 사용.

### 내부 키 (kebab-case, ASCII)

```
usersnet                  # 현 production
usersnet-spectral         # ablation (clinical 없음)
resnet18                  # baseline-legacy
lr-fusion                 # baseline-legacy
contrastive, film, crossattn, transformer, blc-mfds  # experimental
```
