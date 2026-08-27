# MLOps Run Card: Mapping preprocessing ablation

## Run identification

| Field | Value |
|---|---|
| Run ID | `mapping_aligned_baseline_area_dwt_20260826_v1` |
| Run date | 2026-08-26 |
| Purpose | Compare preprocessing conditions for the mapping Multi-scale 1D ResNet and LR reference |
| Source commit | `46b0f0e17f1ca192c77b34daffa37ed3eac34519` |
| MLflow experiment | `aecd-mapping-preprocessing-ablation` (experiment `2`) |
| MLflow run | `c3d3eb76cf184acca55d8273de2cf418` |
| Notebook input | None; this locked run was script-driven |
| Local result root | `results/mapping_aligned_baseline_area_dwt_20260826_v1/` |

## Repository layout and lineage

This run separates source, exploratory work, and generated evidence:

| Layer | Location | Rule |
|---|---|---|
| Executable experiment source | `scripts/analysis/` | Entrypoints and reusable model/evaluation/output modules are versioned in Git |
| Direct spectrum helpers | `scripts/patent/` | Mapping loader and QC/peak helpers required by the executable source are versioned with the run |
| Exploratory analysis | `notebooks/` | Not an input to this locked run; existing notebooks and embedded outputs are not staged automatically |
| Derived run outputs | `results/mapping_aligned_baseline_area_dwt_20260826_v1/` | Kept in the local ignored result store and linked by run ID, source commit, and MLflow artifacts |
| Aggregate evidence | MLflow `comparison/` and `conditions/` artifacts | Reports, aggregate CSVs, figures, metadata, and training histories are retained without patient-level arrays |

The previous monolithic ResNet runner is now a compatibility CLI facade. Its implementation is split into `mapping_resnet_data.py`, `mapping_resnet_model.py`, `mapping_resnet_evaluation.py`, `mapping_resnet_repeat.py`, `mapping_resnet_outputs.py`, and `mapping_resnet_reference.py`. The preprocessing ablation, comparison, and plotting entrypoints remain separate scripts so that a result can be traced from an executed source file to its generated comparison artifact.

The MLflow run deliberately excludes `patient_oof_predictions.csv`, `oof_prediction_arrays.npz`, `preprocessed_*_arrays.npz`, `repeat_metrics_mc.csv`, notebook outputs, and raw clinical workbooks. These are local or regenerable artifacts and may contain sensitive or patient-linked information. No patient identifier is stored in MLflow parameters, tags, metrics, or the run card.

## Dataset and fixed protocol

- Dataset root: `data/mapping/`
- Cohort: Control 21, Prostate Disease Control 49, Prostate Cancer 43; 113 patients total
- Repeats: 121 finite spectra per patient, 13,673 rows total; no non-finite repeats removed
- Input grid: 402–2198 cm⁻¹, 933 points
- Outer validation: 5-fold `StratifiedGroupKFold`, grouped by patient
- Inner model selection: 4-fold grouped CV for the LR reference and validation monitoring for ResNet
- ResNet: 3 residual blocks total, multi-scale kernels 3/5/7, channels 32→64→128
- Training: 30 epochs maximum, early stopping, CUDA, seed `20260826`
- Repeat study: candidate counts 1, 3, 5, 9, 16, 25, 36, 49, 64, 81, 100, 121; 100 Monte-Carlo draws per candidate
- Patient aggregation: mean for the primary comparison; median, majority, and trimmed mean retained in the detailed outputs

All finite repeats remain in the locked evaluation. Robust QC values are recorded for audit and are not used to silently remove the 121-repeat evaluation rows.

## Preprocessing conditions

All conditions trim 400–2200 cm⁻¹ and linearly align raw intensity to the fixed 402–2198 cm⁻¹ grid. The conditions then differ as follows:

| Condition | Additional operation |
|---|---|
| `raw_aligned` | No baseline correction, normalization, or DWT |
| `baseline` | Centered rolling-minimum baseline, window 101 |
| `baseline_area` | Baseline correction followed by area normalization, `y / sum(abs(y))` |
| `baseline_dwt` | Baseline correction followed by `db4`, symmetric level-7 soft-threshold DWT |
| `baseline_area_dwt` | Baseline correction, area normalization, then the same DWT |

## Patient-level OOF summary

Values below are the mean-aggregated Multi-scale ResNet OOF results. The LR reference is included to show the representation-learning comparison under the same patient split and aggregation protocol.

| Preprocessing | ResNet Cancer Screening AUC | ResNet Cancer Screening BA | ResNet Cancer Type ID macro AUC | ResNet Cancer Type ID macro F1 | LR Cancer Screening AUC |
|---|---:|---:|---:|---:|---:|
| `raw_aligned` | 0.653 | 0.570 | 0.556 | 0.251 | 0.696 |
| `baseline` | 0.506 | 0.507 | 0.499 | 0.234 | 0.704 |
| `baseline_area` | 0.523 | 0.500 | 0.522 | 0.104 | 0.696 |
| `baseline_dwt` | 0.489 | 0.496 | 0.502 | 0.262 | 0.659 |
| `baseline_area_dwt` | 0.512 | 0.500 | 0.543 | 0.104 | 0.654 |

The raw-aligned condition is the strongest ResNet condition in this run. The baseline and baseline+DWT conditions show a train/validation AUC gap of approximately 0.202 and 0.206 respectively, consistent with overfitting or loss of usable representation after those transforms. The LR reference remains a separate clinical baseline and does not prove that the CNN extracted the same features.

The repeat-count result is relative to each condition's own 121-repeat reference. ResNet mean aggregation produced Screening / Cancer Type ID minimum candidates of 36 / 49 for `raw_aligned`, 1 / 9 for `baseline`, 1 / 1 for `baseline_area`, 1 / 9 for `baseline_dwt`, and 1 / 1 for `baseline_area_dwt`. The chance-level conditions make these low relative thresholds unsuitable as evidence of a more efficient measurement protocol.

## Interpretation boundary

Cancer Screening AUC may be confounded by hospital and measurement source because the non-cancer and cancer groups are not guaranteed to be cross-hospital balanced. These results should not be interpreted as external cross-hospital generalization. Cancer Type ID is the more biologically interpretable task in this cohort, but it still requires independent external validation.

## Data fingerprints

The hashes below identify the data files used without copying their contents into Git or MLflow:

| File | SHA-256 |
|---|---|
| `data/mapping/clinical_df.xlsx` | `c49eb12812faa807f4b81ce747058d0e1d65fde1b7205925354e9838ccb1dbdb` |
| `artifacts/usersnet/v1.0.0/common_grid.npy` | `9b4fc501edff7e1e268fc6f167f6fab9b271a19f96e59fcc1bf4ad08c7b7b8e8` |

## Change history

| Date | Change | Evidence |
|---|---|---|
| 2026-08-26 | Completed five-condition preprocessing ablation with patient-level OOF, repeat-count Monte Carlo evaluation, and comparison figures | Local result root and `REPORT.md` |
| 2026-08-26 | Corrected comparison bar-chart y-axis so sensitivity values of 1.0 are fully visible | `figure_preprocessing_metrics_bar.png` |
| 2026-08-27 | Split the ResNet runner into reusable source modules while preserving the CLI/import compatibility surface | Git commit `46b0f0e17f1ca192c77b34daffa37ed3eac34519` |
| 2026-08-27 | Registered aggregate metrics and selected evidence artifacts in local MLflow | Run `c3d3eb76cf184acca55d8273de2cf418` |

## Reproduction entrypoints

The locked condition runs use the following source entrypoint pattern:

```bash
python scripts/analysis/run_mapping_aligned_baseline_area_dwt.py \
  --condition raw_aligned \
  --output results/mapping_aligned_baseline_area_dwt_20260826_v1/raw_aligned \
  --epochs 30 --mc-iterations 100 --n-blocks 3 --device cuda
```

After all five conditions complete, use `compare_mapping_aligned_baseline_area_dwt.py` and `plot_mapping_aligned_baseline_area_dwt.py` to regenerate the aggregate comparison CSVs and figures. The MLflow registration records the exact source commit and data fingerprints used for this run.
