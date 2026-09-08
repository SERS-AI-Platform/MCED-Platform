# SERS-AI Experiment Results Index

| Field | Value |
| --- | --- |
| SharePoint area | `02_Experiment_Results` |
| Version | v0.1 |
| Last updated | 2026-05-15 |
| Rule | final/review-ready results only; raw spectrum data excluded |

## 1. Purpose

This folder stores model performance results for SERS-AI. It is the SharePoint-facing location for result Excel files, ROC curves, confusion matrices, and per-cancer performance tables.

Raw spectrum files, processed spectrum matrices, training logs, and model binaries are explicitly excluded.

## 2. Required Metadata For Every Result Excel

Every result workbook must include the following fields.

| Required field | Definition |
| --- | --- |
| `run_id` | stable identifier for the experiment run or result package |
| sample count | subject/patient count and spectrum count where available |
| dataset definition | cancer classes, non-cancer groups, inclusion/exclusion rules, split definition |
| Git commit | exact commit hash used for code/result traceability |
| source path | internal source document or artifact path |
| model variant | SERS-only, fusion, sex constraint, stacking, etc. |
| step | Step 1 detection or Step 2 cancer type identification |

## 3. v0.1 Result Packages

| Run ID | Scope | Dataset | Required output |
| --- | --- | --- | --- |
| `PhaseP_Q_5Cancer_HeldOut_v0.1` | 5-cancer held-out validation and sex-constraint result | PRO, LUN, CRC, CPAN, OVA + NOR/DIA/HBP/H.D. | Excel, ROC, confusion matrix, per-cancer table |
| `PhaseW_7Cancer_HeldOut_v0.1` | 7-cancer held-out/fusion milestone | 7-cancer setting tracked in `docs/ml/experiment.md` | Excel, ROC, confusion matrix, sensitivity/performance figures |

## 4. Current KPI Snapshot

| Run ID | Step | Model | AUC | Sensitivity | Specificity | Precision | Recall | Type ID metric |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `PhaseP_5Cancer_HeldOut` | Step 1 | SERS-only | 0.977 +/- 0.005 | 0.926 +/- 0.004 | 0.929 +/- 0.019 | TBD | 0.926 +/- 0.004 | F1 0.852 +/- 0.026 |
| `PhaseP_5Cancer_HeldOut` | Step 1 | Fusion | 0.986 +/- 0.004 | 0.939 +/- 0.011 | 0.949 +/- 0.019 | TBD | 0.939 +/- 0.011 | F1 0.877 +/- 0.018 |
| `PhaseQ_5Cancer_SexConstraint` | Step 2 | SERS + sex constraint | 0.977 +/- 0.005 | 0.926 +/- 0.004 | 0.929 +/- 0.019 | TBD | 0.926 +/- 0.004 | F1 0.892 +/- 0.026 |
| `PhaseW_7Cancer_HeldOut` | Step 1/2 | Fusion | 0.979 +/- 0.005 | TBD | TBD | TBD | TBD | Type ID F1 0.923 +/- 0.022 |

## 5. Figure Sources For v0.1 Staging

| Figure type | Source path | SharePoint handling |
| --- | --- | --- |
| 7-cancer ROC curves | `publications/bumbucheo/figures/fig_roc_curves_7c.png` | upload to `PhaseW_7Cancer_HeldOut/Figures/ROC/` |
| 7-cancer confusion matrix | `publications/bumbucheo/figures/fig_confusion_matrices_7c.png` | upload to `PhaseW_7Cancer_HeldOut/Figures/Confusion_Matrix/` |
| 7-cancer sensitivity bar | `publications/bumbucheo/figures/fig_sensitivity_bar_7c.png` | upload to `PhaseW_7Cancer_HeldOut/Figures/Performance/` |
| AACR ROC curves | `publications/aacr/figures/fig3_roc_curves.png` and `.pdf` | upload to review package if AACR result is referenced |
| AACR stage performance | `publications/aacr/figures/fig4_stage_performance.png` and `.pdf` | upload to review package if AACR result is referenced |

## 6. Raw Data Exclusion

The following are not allowed in SharePoint `02_Experiment_Results`:

- `data/raw_data/`
- `data/raw_data_medical/`
- `results/processed_spectra.csv`
- `results/data/*.csv` containing processed spectrum matrices
- model `.joblib`, `.npy`, `.npz` artifacts
- logs and cache folders

## 7. Git Commit References

| Commit | Relevance |
| --- | --- |
| `45bbc72` | initial imported model code, experiment context, production manifests |
| `1cd527f` | merged OneDrive development materials |
| `cc3a998` | created SharePoint Project Overview and Model Development docs |
| `fae245a` | recorded SharePoint document commit references |
| `578dcbf` | created `02_Experiment_Results` governance, result index, and SharePoint remap entries |

For Excel workbooks, include `578dcbf` or a later run-specific commit hash in the `Commit_References` sheet.
