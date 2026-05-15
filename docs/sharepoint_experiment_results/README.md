# SERS-AI 02_Experiment_Results Governance

SharePoint destination:

```text
/01_Projects/SERS-AI/02_Experiment_Results/
```

Purpose: store review-ready model performance results, result tables, and final figures.

## Required Contents

| Content | Required | Notes |
| --- | --- | --- |
| Result Excel workbook | yes | one workbook per major run/result package |
| ROC curves | yes | final PNG/PDF only |
| Step 1 performance | yes | cancer vs non-cancer |
| Step 1 confusion matrix | yes | binary confusion matrix |
| Step 2 performance | yes | cancer type identification |
| Step 2 confusion matrix | yes | type-level confusion matrix |
| Per-cancer performance table | yes | AUC, sensitivity, precision, recall, specificity where available |
| Run metadata | yes | run ID, sample count, dataset definition, Git commit |

## Strict Rules

- Do not upload raw spectrum data.
- Do not upload processed spectrum matrices such as `processed_spectra.csv`.
- Do not upload training logs, cache folders, model binaries, or temporary experiment outputs.
- Store only final or review-ready results.
- Every result Excel must include:
  - `run_id`
  - sample count
  - dataset definition
  - Git commit reference
  - source result document or artifact path
- Missing metrics must be marked `TBD` or `not available`, not silently omitted.

## Recommended Folder Structure

```text
02_Experiment_Results/
  README.md
  2026-03_PhaseP_Q_5Cancer_HeldOut/
    2026_SERS-AI_ExperimentResults_PhaseP_Q_5Cancer_v0.1.xlsx
    Figures/
      ROC/
      Confusion_Matrix/
      Performance/
  2026-03_PhaseW_7Cancer_HeldOut/
    2026_SERS-AI_ExperimentResults_PhaseW_7Cancer_v0.1.xlsx
    Figures/
      ROC/
      Confusion_Matrix/
      Performance/
  VERSION_LOG.md
```

## Result Excel Required Sheet Schema

| Sheet | Required fields |
| --- | --- |
| `Run_Summary` | run ID, date, model variant, step, Git commit, source document |
| `Dataset_Definition` | cancer types, non-cancer groups, sample count, spectrum count, split definition |
| `Step1_Performance` | AUC, sensitivity, specificity, precision, recall, threshold, CI/std if available |
| `Step1_Confusion_Matrix` | TN, FP, FN, TP or matrix table |
| `Step2_Performance` | macro/micro metrics, per-class AUC if available |
| `Step2_Confusion_Matrix` | class-by-class confusion matrix |
| `Per_Cancer_Performance` | cancer type, N, AUC, sensitivity, precision, recall, specificity |
| `Figure_Index` | figure filename, figure type, source path, notes |
| `Commit_References` | commit hash, scope, relevance |

## Primary Internal Sources

| Source | Use |
| --- | --- |
| `docs/EXPERIMENT_CONTEXT.md` | detailed Phase P/Q performance and per-cancer sensitivity |
| `docs/experiment.md` | phase-level run history |
| `docs/CHANGELOG.md` | milestone result log |
| `publications/aacr/figures/` | final AACR-style ROC/performance figures |
| `publications/bumbucheo/figures/` | 7-cancer proposal figures |
| `models/production/manifest.json` | production model run metadata |
