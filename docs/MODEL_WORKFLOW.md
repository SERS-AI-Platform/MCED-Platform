# Model Workflow

This document describes the current model-development workflow for
MCED-Platform. The active entry points are `sers` CLI commands and scripts under
`scripts/training/`; `models/legacy/` remains available only for compatibility
and historical reproduction.

## 1. Prepare Spectra

Run preprocessing and QC before model training:

```bash
sers preprocess --config config/config.yaml
```

Common variants:

```bash
sers preprocess --source medical
sers preprocess --normalization snv --qc-policy strict
sers preprocess --skip-qc
```

The preprocessing command wraps `scripts/pipeline/run_qc_preprocess.py`.
Outputs are controlled by `config/config.yaml` unless `--output-dir` is passed.

## 2. Train the Active STK-V2 / uSERS-Net Model

Use the stacking command for the current production-path model:

```bash
sers train stacking --dry-run
sers train stacking
```

Useful options:

```bash
sers train stacking --val-group SPAN --meta-learner elasticnet
sers train stacking --no-shap --no-cm
```

The command wraps `scripts/training/train_usersnet.py`.

## 3. Build Production Artifacts

Build the active production artifact package:

```bash
sers production --stacking -o artifacts/usersnet/current
```

Fit the PDS calibration artifact when needed:

```bash
sers production --stacking --fit-pds \
  --grid artifacts/usersnet/current/common_grid.npy \
  --pds-out artifacts/usersnet/current/calibration/pds.npz
```

Production artifact manifests under `artifacts/usersnet/` are the SSOT for
deployed model thresholds and preprocessing parameters.

## 4. Run Baseline / Legacy Comparisons

Legacy model families are still exposed through the CLI for reproducibility:

```bash
sers train xgboost --aggregate none --n-splits 10
sers train resnet18 --epochs 200 --lr 3e-4
sers benchmark resnet18 lr xgboost --no-mlflow
```

These commands wrap scripts under `models/legacy/scripts/`. Treat them as
benchmark or reproduction paths, not the primary production workflow.

## 5. Evaluate Models and Transfer Behavior

General model evaluation:

```bash
sers test -i results/training --processed-csv results/processed_spectra.csv --no-shap
```

Current SERS transfer and acquisition-set analyses live under:

```text
scripts/analysis/sers_transfer/
scripts/evaluation/
artifacts/sers_transfer/
```

The current transfer package summary starts at:

```text
artifacts/sers_transfer/TRANSFER_MANIFEST.md
```

## 6. Reporting Rules

When reporting model results, always state:

- aggregation mode: `mean`, `medoid`, or `none`
- cancer set
- non-cancer set
- sample count
- model artifact version or commit
- decision profile: `screening`, `balanced`, or `confirmatory`

Do not compare runs unless aggregation, cancer set, non-cancer set, and sample
count match. Use the terminology in `docs/TERMINOLOGY_STANDARD.md`.

## 7. Verification Before Sharing Results

Run the same quality gates used by CI:

```bash
ruff check src/ tests/
mypy
pytest --cov=sers --cov-report=term-missing --cov-fail-under=35
```

The current mypy gate intentionally covers the CLI/config/scoring surface first.
Full-repository type checking remains a separate migration task.
