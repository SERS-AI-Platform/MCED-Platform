# Model Status Registry

**목적**: `models/` 내부 모든 파일·폴더의 현재 상태와 사유를 고정 기록.
**작성일**: 2026-04-15
**Registry 카테고리**: 🟢 production / 📘 baseline-legacy / 🧪 experimental / 📦 archived / 🗑️ delete

이 문서와 `src/sers/models/_registry.py` 는 **한 쌍** 으로 동기화되어야 함.

---

## 🟢 Production — uSERS-Net (현 운영 모델)

현재 공개 이름: **uSERS-Net**, 내부 아키텍처: Stacking V2, 버전: v1.0.0

| 파일 | 역할 | 이동 목적지 | 비고 |
|---|---|---|---|
| `stacking_utils.py` | Level-0(10 base) + Level-1(meta-LR) 구현 | `src/sers/models/usersnet/stacking.py` | 8곳 import |
| `clinical_utils.py` | 임상 변수 로더 (현재 ResNet 의존) | `src/sers/models/usersnet/clinical_fusion.py` | **뼈대만** — ResNet 의존 제거. v1.1.0 통합은 별도 task |
| `train_stacking.py` | STK-V2 학습 (nested CV) | `scripts/training/train_usersnet.py` | |
| `build_production_stacking.py` | Production 빌더 | `scripts/training/build_usersnet_production.py` | |
| `eval_stacking_holdout.py` | Train/Val/Test 60/20/20 평가 | `scripts/evaluation/eval_usersnet_holdout.py` | |
| `production_stacking/` | 학습된 아티팩트 (joblib×22, json×4, npy×1) | `artifacts/usersnet/v1.0.0/` | `current → v1.0.0` symlink |

### v1.1.0 계획 (이후 별도 task)
- Meta-learner 입력에 clinical features 추가 (Late Fusion)
- `clinical_fusion.py` 본격 구현
- 새 artifact `artifacts/usersnet/v1.1.0/` 생성, `current` 포인터 이동

---

## 📘 Baseline-Legacy — ResNet18 & LR-Fusion (비교용 유지)

CLI 에서 `sers train --model resnet18`, `--model lr-fusion` 으로 계속 실행 가능.

### ResNet18-1D Two-Stage

| 파일 | 역할 | 이동 목적지 | 사유 |
|---|---|---|---|
| `model.py` | ResNet18-1D Two-Stage 모델 정의 | `src/sers/models/_legacy/resnet_v1/model.py` | 27곳 import — baseline 비교용 |
| `train.py` | ResNet18 학습 파이프라인 | `scripts/training/_legacy/train_resnet.py` | 25곳 import |
| `test.py` | ResNet18 평가/시각화 | `scripts/evaluation/_legacy/eval_resnet.py` | ⚠️ docstring 인코딩 깨짐 복구 필요 |

### LR-Fusion (구 production)

| 파일 | 역할 | 이동 목적지 | 사유 |
|---|---|---|---|
| `build_production_model.py` | 구 production 빌더 (LR+clinical fusion) | `scripts/training/_legacy/build_lr_fusion.py` | baseline 비교용 |
| `production/` | 구 production 아티팩트 | `artifacts/baselines/lr-fusion/v1.0.0/` | 유지 |

### SERS-Net Ensemble (LR+ResNet+Clinical)

| 파일 | 역할 | 이동 목적지 | 사유 |
|---|---|---|---|
| `run_sersnet.py` | Stage 1/2 블렌딩 앙상블 (0.8·LR_fusion + 0.2·ResNet) + Sex constraint | `scripts/training/_legacy/run_sersnet_ensemble.py` | **uSERS-Net v1.1.0 (stacking+clinical) 비교의 공정 baseline** — clinical 포함한 유일한 legacy 앙상블 |

**참고**: `run_sersnet.py` 의 clinical 은 LR 쪽에만 들어가고 ResNet 쪽에는 없음 (진짜 삼중 결합 아님).
`train_ensemble.py` (LR+ResNet, clinical 없음) 과 `run_multimodal.py` (multimodal 초기 탐색, ResNet 없음) 는 archived 유지.

---

## 🧪 Experimental — 연구 모델 (별도 격리)

| 폴더 | 역할 | 이동 목적지 | 상태 |
|---|---|---|---|
| `contrastive/` | Contrastive pretraining (model.py, pretrain.py) | `src/sers/models/experimental/contrastive/` | 연구중 (2026-04-10 수정) |
| `cross_attention/` | Cross-attention fusion (model.py, train.py) | `src/sers/models/experimental/cross_attention/` | 연구중 |
| `film/` | FiLM conditioning (model.py, train.py) | `src/sers/models/experimental/film/` | 연구중 |
| `transformer/` | Transformer encoder (model.py, train.py) | `src/sers/models/experimental/transformer/` | 연구중 |
| `blc_mfds/` | BLC MFDS 전용 모델 (train.py) | `src/sers/models/experimental/blc_mfds/` | 상태 미확정 — README 필요 |
| `xai_fusion_analysis.py` | FiLM/CrossAttn SHAP 분석 | `scripts/analysis/xai_fusion_analysis.py` | 실험 모델과 쌍 |

**이동 시 각 폴더에 README.md 필수 생성**:
- 목적 / 가설 / 마지막 실험 결과 / 현재 상태 (active/paused/failed)

---

## 📦 Archived — 완료된 ablation·실험 (코드 보존, 결과는 `results/`)

"Step 3/4/5" 처럼 한 번 돌리고 끝난 실험. 파일명 앞에 시기 태그 추가.

| 파일 | 역할 | 이동 목적지 |
|---|---|---|
| `train_multichannel.py` | Step 3: 3-channel ResNet ablation | `scripts/experiments/_archive/2026-03_step3_multichannel.py` |
| `run_step4_normalization.py` | Step 4: SNV normalization ablation | `scripts/experiments/_archive/2026-03_step4_normalization.py` |
| `train_ensemble.py` | Step 5: LR+ResNet 단순 앙상블 | `scripts/experiments/_archive/2026-03_step5_ensemble.py` |
| `run_multimodal.py` | 초기 multimodal fusion 탐색 | `scripts/experiments/_archive/2026-03_multimodal_early.py` |
| `run_clinical_analysis.py` | Per-cancer 민감도 + LUN stage 분석 | `scripts/analysis/clinical_breakdown.py` (사후 분석으로 재분류) |
| `run_confounding_analysis.py` | 연령·성별 교란 분석 | `scripts/analysis/confounding_analysis.py` |
| `run_pan_improvement.py` | PAN Lasso + threshold 실험 | `scripts/experiments/_archive/2026-04_pan_improvement.py` |
| `pancreatic_experiment.py` | CPAN+YPAN binary 실험 | `scripts/experiments/_archive/2026-04_pancreatic_binary.py` |
| `run_train_val_test.py` | 구 train/val/test split 실험 | `scripts/experiments/_archive/2026-04_train_val_test_split.py` |
| `learning_curve_comparison.py` | Learning curve 비교 | `scripts/analysis/learning_curve_comparison.py` |
| `plot_overall_comparison.py` | 전체 모델 비교 플롯 | `scripts/figures/plot_overall_comparison.py` |
| `tune_torch.py` | Torch HP tuning | `scripts/experiments/_archive/2026-03_torch_hp_tune.py` |
| `results/` | 실험 결과 (`01_benchmarks` ~ `05_subset_analysis`) | `results/` (루트로 승격) + `results/benchmarks/` 하위 재배치 |

**`scripts/experiments/_archive/README.md` 생성 필수**:
- 각 실험의 결과 위치 링크
- 결론 한 줄 요약

---

## 🗑️ Delete

| 파일 | 사유 |
|---|---|
| `models/__init__.py` | 재구성 후 `models/` 는 디렉토리 아님 |
| `models/__pycache__/` | 캐시 |
| `models/manifest.json` | 루트 manifest 불필요 (각 artifact 버전에 포함됨) |

---

## Registry (정본)

이 표와 동기화되어야 할 코드 파일: `src/sers/models/_registry.py`

```python
# src/sers/models/_registry.py (예정)
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    category: str                # production | baseline-legacy | experimental
    module: str                  # 파이썬 import 경로
    train_script: str
    eval_script: str
    artifact_dir: str            # artifacts/<...>/current
    paper_name: str

MODEL_REGISTRY = {
    "usersnet": ModelSpec(
        key="usersnet",
        display_name="uSERS-Net",
        category="production",
        module="sers.models.usersnet",
        train_script="scripts/training/train_usersnet.py",
        eval_script="scripts/evaluation/eval_usersnet_holdout.py",
        artifact_dir="artifacts/usersnet/current",
        paper_name="uSERS-Net",
    ),
    "resnet18": ModelSpec(
        key="resnet18",
        display_name="ResNet18-1D",
        category="baseline-legacy",
        module="sers.models._legacy.resnet_v1",
        train_script="scripts/training/_legacy/train_resnet.py",
        eval_script="scripts/evaluation/_legacy/eval_resnet.py",
        artifact_dir="artifacts/baselines/resnet18/v1.0.0",
        paper_name="ResNet18-1D (baseline)",
    ),
    "lr-fusion": ModelSpec(
        key="lr-fusion",
        display_name="LR-Fusion (SERS+Clinical)",
        category="baseline-legacy",
        module="sers.models._legacy.lr_fusion_v1",
        train_script="scripts/training/_legacy/build_lr_fusion.py",
        eval_script=None,
        artifact_dir="artifacts/baselines/lr-fusion/v1.0.0",
        paper_name="LR-Fusion (baseline)",
    ),
    "sersnet-ensemble": ModelSpec(
        key="sersnet-ensemble",
        display_name="SERS-Net Ensemble",
        category="baseline-legacy",
        module="sers.models._legacy.sersnet_ensemble",
        train_script="scripts/training/_legacy/run_sersnet_ensemble.py",
        eval_script=None,
        artifact_dir="artifacts/baselines/sersnet-ensemble/v1.0.0",
        paper_name="SERS-Net Ensemble (LR+ResNet+Clinical)",
    ),
    "contrastive": ModelSpec(...),      # experimental 5종
    "cross-attention": ModelSpec(...),
    "film": ModelSpec(...),
    "transformer": ModelSpec(...),
    "blc-mfds": ModelSpec(...),
}
```

---

## 변경 이력

| 날짜 | 변경 | 사유 |
|---|---|---|
| 2026-04-15 | 초안 작성 | `models/` 디렉토리 감사 결과 반영 |
