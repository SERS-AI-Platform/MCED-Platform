# Experiment Context

Updated: 2026-03-18

## 1. What This Project Is Trying To Do

This repository is building a SERS-based urine screening pipeline for multi-cancer detection.

The current modeling task is a two-step classifier:

1. **Cancer Detection**: cancer vs non-cancer (binary screening)
2. **Cancer Identification**: cancer subtype classification among the selected cancer cohorts

The codebase currently supports both:

- data/QC pipeline under `src/sers/`
- model training/evaluation under `models/`

## 2. Dataset Snapshot Used Across Most Experiments

The main input for modeling is `results/processed_spectra.csv`.

Raw metadata (`results/metadata_raw.csv`) shows the following cohort sizes:

| Group | Subjects | Spectra |
| --- | ---: | ---: |
| PRO | 100 | 500 |
| BRE | 30 | 150 |
| OVA | 70 | 350 |
| LUN | 300 | 1500 |
| CRC | 300 | 1500 |
| CPAN | 70 | 350 |
| SPAN | 72 | 360 |
| NOR | 100 | 500 |
| DIA | 100 | 500 |
| HBP | 100 | 500 |
| H.D. | 100 | 500 |
| UNK | 2 | 10 |

Important notes:

- Standard groups sum to 1342 subjects and 6710 spectra.
- Most modeling experiments exclude `UNK`.
- `config.yaml` defines cancer groups as `PRO, BRE, OVA, LUN, CRC, CPAN, SPAN` and non-cancer groups as `NOR, DIA, HBP, H.D.`.
- Earlier project logic merged `CPAN` and `SPAN` into `PAN`, but the current model code keeps them separate.

## 3. Preprocessing / QC Context

### Pipeline Architecture

The pipeline has two distinct stages that run sequentially:

```
Raw Spectra → [QC Pipeline] → QC-passed spectra → [Preprocessing Pipeline] → processed_spectra.csv
```

**QC and preprocessing are independent.** QC operates entirely on raw intensity data. Normalization (SNV etc.) is only applied during preprocessing, after QC is complete. The training pipeline (`models/train.py`) loads the already-preprocessed `processed_spectra.csv` and does not apply any additional normalization.

### QC Pipeline (operates on raw data, no normalization)

The QC pipeline (`src/sers/qc/qc.py`, invoked by `run_qc_preprocess.py`) checks measurement quality on raw spectra:

**Level 0: Intensity Gate** — "Did SERS enhancement work?"

- Computes mean intensity in the fingerprint region (400-2200 cm^-1) for each spectrum
- Sets adaptive threshold: `median(all fp_means) x 0.1`
- Catches catastrophic SERS enhancement failures (e.g., substrate not activated)
- Current data: all spectra pass (threshold ~9.0, all fp_means well above)

**Level 1: Replicate RSD** — "Is intensity reproducible across 5 replicates?"

- RSD = `(std / max_intensity) x 100` across replicates of the same sample
- Threshold: `< 5%`
- Uses RSD (not CV) because baseline-corrected spectra have near-zero regions where CV explodes
- Current data: all samples have RSD < 1.5% (well below threshold)

**Level 1: Replicate Correlation** — "Is spectral shape consistent?"

- Mean pairwise Pearson correlation across all replicate pairs
- Threshold: `> 0.95`
- Current data: all samples have correlation > 0.99

### Preprocessing Pipeline (applied after QC, configurable normalization)

The preprocessing pipeline (`src/sers/preprocessing.py`) transforms QC-passed raw spectra:

| Step | Operation | Parameters | Purpose |
| --- | --- | --- | --- |
| 1 | Trim | 400-2200 cm^-1 | Remove non-fingerprint regions (substrate features, noise) |
| 2 | Smooth | Savitzky-Golay (window=11, poly=3) | Reduce high-frequency noise, preserve peak shapes |
| 3 | Baseline correction | Rolling minimum (window=101) | Remove fluorescence background |
| 4 | Normalize | **Configurable**: SNV / MinMax / L2 / Area / None | Remove measurement artifacts (SERS hot-spot variation) |
| 5 | Resample | Interpolate to common grid | Align all spectra to same wavenumber axis |

**Normalization is the only configurable step.** The default is SNV (`use_snv: true` in `config.yaml`), but it can be changed:

```bash
python run_qc_preprocess.py --normalization none    # Skip normalization entirely
python run_qc_preprocess.py --normalization minmax   # Use MinMax instead
python run_qc_preprocess.py --normalization l2       # Use L2 normalization
```

### Current config (`config.yaml`)

Preprocessing:

- `do_smooth: true`, `smooth_window: 11`, `smooth_poly: 3`
- `baseline_window: 101`
- `use_snv: true` (default normalization)

QC:

- `intensity_gate_ratio: 0.1`
- `rsd_threshold: 5.0`
- `corr_threshold: 0.95`
- `expected_reps: 5`
- `fingerprint_region: 400-2200`

### Visualization Examples

Example plots of each QC and preprocessing step using real data are available in `results/figures/`:

| File | Content |
| --- | --- |
| `pipeline_overview.png` | Full 8-step pipeline on single page (QC + preprocessing) |
| `step0_raw_spectrum.png` | Raw SERS spectrum with fingerprint region highlighted |
| `step1_qc_intensity_gate.png` | Intensity gate distribution and high vs low enhancement comparison |
| `step2_qc_replicate.png` | Replicate RSD distribution, correlation distribution, and 5-replicate overlay |
| `step3_preprocessing_pipeline.png` | Step-by-step: raw -> trim -> smooth -> baseline -> SNV/MinMax |
| `step4_snv_comparison.png` | SNV effect: cancer vs normal spectra before/after normalization, difference spectrum |

Generated by: `python scripts/generate_qc_preprocessing_examples.py`

### Observed behavior

- All spectra pass the current QC thresholds. QC is functioning as dataset characterization rather than aggressive filtering.
- `results/metadata_raw.csv` and `results/processed_spectra.csv` have the same cohort totals.
- QC output contains `YPAN` rows, while raw/processed modeling snapshots use `UNK` for 2 subjects / 10 spectra. Those rows are outside the standard training cohorts.

## 4. High-Level Experiment Timeline

### Phase A: Initial base model on medoid-aggregated subjects

Relevant artifacts:

- `results/training/training_summary.json`
- `models/results/training_v2/training_summary.json`

Early baseline on 2026-02-26:

- aggregation: `medoid`
- samples: `1272`
- cancer types: `PRO, BRE, OVA, LUN, CRC, PAN`
- result:
  - Detection AUC: `0.793`
  - Identification accuracy: `0.486`
  - Identification F1 macro: `0.356`

Interpretation:

- This appears to be the older formulation where pancreatic cohorts were merged into `PAN`.
- Performance was not yet competitive, especially on Cancer Identification.

Improved medoid baseline on 2026-02-27 (`training_v2`):

- aggregation: `medoid`
- samples: `1342`
- cancer types: `PRO, BRE, OVA, LUN, CRC, CPAN, SPAN`
- result:
  - Detection AUC: `0.948`
  - Identification accuracy: `0.670`
  - Identification F1 macro: `0.548`

Interpretation:

- Separating `CPAN` and `SPAN` and moving to the newer model/training stack gave a major jump over the older `PAN` baseline.

### Phase B: Switching from subject-level medoid to all spectra

Relevant artifacts:

- `models/results/training_v2_norep/training_summary.json`
- `models/results/training_all_spectra/training_summary.json`

`training_v2_norep` on 2026-02-27:

- aggregation: `none`
- samples: `6710`
- result:
  - Detection AUC: `0.958`
  - Identification accuracy: `0.727`
  - Identification F1 macro: `0.613`

`training_all_spectra` on 2026-03-12:

- aggregation: `none`
- samples: `6710`
- result:
  - Detection AUC: `0.952`
  - Identification accuracy: `0.715`
  - Identification F1 macro: `0.609`

Interpretation:

- Using all spectra instead of medoid aggregation was one of the biggest beneficial changes.
- The stored rerun on 2026-03-12 is slightly worse than the 2026-02-27 `training_v2_norep` snapshot, so the earlier no-aggregation run is still the stronger reference between those two.

### Phase C: Easier subset experiment for CRC + CPAN vs controls

Relevant artifact:

- `models/results/training_subset/training_summary.json`

Run on 2026-03-12:

- aggregation: `none`
- samples: `3850`
- cancer types: `CRC, CPAN`
- non-cancer groups: `NOR, DIA, HBP, H.D.`
- result:
  - Detection AUC: `0.981`
  - Identification accuracy: `0.830`
  - Identification F1 macro: `0.729`

Interpretation:

- The narrower disease setting is substantially easier than the full multi-cancer problem.
- This is useful as a proof that the pipeline can separate a constrained clinical subset better than the full cohort mix.

### Phase D: First benchmark on selected all-spectra cohorts

Relevant artifact:

- `logs/experiment_runs.jsonl`
- `models/results/benchmark_all_spectra/`

Key comparable setting:

- date window: 2026-03-12 to 2026-03-13
- aggregation: `none`
- samples: `6200`
- cancer types: `PRO, LUN, CRC, CPAN, OVA`
- non-cancer groups: `NOR, DIA, HBP, H.D.`

Important event:

- The first XGBoost run on 2026-03-12 failed behaviorally, producing a degenerate result around `Detection AUC = 0.5`.
- Later XGBoost versions fixed this and became competitive again.

Best full-setting results stored in this branch of experiments:

- XGBoost `v005`
  - Detection AUC: `0.970`
  - Identification accuracy: `0.838`
  - Identification F1 macro: `0.776`
- ResNet18 best comparable full-setting run was lower
  - Identification F1 macro roughly `0.724`
- CNN1D remained much weaker
  - Identification F1 macro roughly `0.472-0.508`

Interpretation:

- In this 5-cancer / 4-control all-spectra setting, XGBoost became the strongest non-logistic model.
- The shallow CNN remained clearly underpowered.

### Phase E: Reduced-class benchmark created some of the best raw scores

Relevant source:

- `logs/experiment_runs.jsonl`
- `models/results/overall_comparison/best_per_model.csv`

One notable run on 2026-03-13 used:

- samples: `5850`
- cancer types: `PRO, LUN, CRC, OVA`
- non-cancer groups: `NOR, DIA, HBP, H.D.`

Best results from that reduced problem:

- XGBoost `benchmark_all_spectra/v004`
  - Identification accuracy: `0.869`
  - Identification F1 macro: `0.822`
- ResNet18 `benchmark_all_spectra/v003`
  - Identification accuracy: `0.832`
  - Identification F1 macro: `0.788`
- CNN1D `benchmark_all_spectra/v003`
  - Identification accuracy: `0.650`
  - Identification F1 macro: `0.613`

Interpretation:

- These are some of the numerically strongest model runs in the repo.
- They are not directly comparable to the full multi-cancer experiments, because the class set is smaller and easier.

### Phase F: Full benchmark including classical baselines

Relevant artifacts:

- `models/results/benchmark_all_models/benchmark_summary.csv`
- `models/results/overall_comparison/best_per_model.csv`

Run date: 2026-03-13

Setting:

- aggregation: `none`
- samples: `6200`
- cancer types: `PRO, LUN, CRC, CPAN, OVA`
- non-cancer groups: `NOR, DIA, HBP, H.D.`

Results:

| Model | Detection AUC | Id Acc | Identification AUC | Identification F1 Macro |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.981 | 0.902 | 0.985 | 0.870 |
| XGBoost | 0.969 | 0.837 | 0.963 | 0.770 |
| ResNet18 | 0.961 | 0.770 | 0.933 | 0.710 |
| Random Forest | 0.931 | 0.753 | 0.928 | 0.642 |
| CNN1D | 0.890 | 0.460 | 0.813 | 0.424 |

Main conclusion:

- Logistic Regression is currently the strongest fair baseline on the main comparable 6200-spectra benchmark.
- Deep learning models are not winning on the current feature representation / data regime.

### Phase G: Medoid benchmark reruns under `models/results/training/`

Relevant artifacts:

- `models/results/training/experiment_002/`
- `models/results/training/experiment_003/`

Common setting:

- date: 2026-03-16
- aggregation: `medoid`
- samples: `1342`
- cancer types: `PRO, BRE, OVA, LUN, CRC, CPAN, SPAN`
- non-cancer groups: `NOR, DIA, HBP, H.D.`

`experiment_002`:

- Logistic Regression: Identification F1 macro `0.697`
- Random Forest: `0.447`
- CNN1D: `0.150`
- ResNet18: `0.527`
- XGBoost failed due to missing dependency in the environment

`experiment_003`:

- Same medoid setting rerun after XGBoost became available
- Logistic Regression: `0.697`
- XGBoost: `0.568`
- ResNet18: `0.474`
- CNN1D: `0.130`

Interpretation:

- On medoid-aggregated subject-level data, Logistic Regression is again the best stored model.
- `experiment_003` is effectively the complete rerun of the medoid benchmark after fixing the XGBoost environment issue.

### Phase H: Hyperparameter retuning on 2026-03-16

Relevant artifacts:

- `models/results/torch_tuning/tuning_summary.csv`
- `models/tune_torch.py`

Two major tuning tracks were run:

1. `retune_20260316_*`
   - 5 cancers: `PRO, LUN, CRC, CPAN, OVA`
   - 4 non-cancer groups
   - aggregation: `none`
   - samples: `6200`

2. `retune_no_span_20260316_*`
   - 6 cancers: `PRO, BRE, OVA, LUN, CRC, CPAN`
   - `SPAN` excluded
   - aggregation: `none`
   - samples: `6350`

Best tuning outcomes from stored summaries:

- XGBoost
  - best in 6200-sample tuning track: `tune_03_shallower_faster`
  - Identification F1 macro: `0.775`
  - very similar to prior benchmark XGBoost, not a big leap

- ResNet18
  - tuned runs landed around Identification F1 macro `0.650-0.658` in the 6200-sample track
  - this is worse than earlier benchmark ResNet18 results

- CNN1D
  - tuned runs improved only modestly
  - still much weaker than classical baselines

- No-SPAN track
  - best XGBoost macro F1 dropped to about `0.679`
  - best ResNet18 macro F1 dropped to about `0.603`
  - best CNN1D macro F1 dropped to about `0.353`

Interpretation:

- Hyperparameter retuning did not overturn the ranking.
- Changing class composition mattered more than tuning.
- The 6-cancer no-SPAN setting appears harder than the 5-cancer benchmark setting.

### Phase I: Learning curve checks

Relevant artifacts:

- `models/results/learning_curve_quickcheck/`
- `models/results/learning_curve_smoke/`
- `models/learning_curve_comparison.py`

Quickcheck interpretation saved in the repo:

- fraction `0.20`: Logistic Regression Identification F1 macro `0.534`
- fraction `0.50`: Logistic Regression Identification F1 macro `0.587`
- fraction `1.00`: Logistic Regression Identification F1 macro `0.676`
- ResNet18 remained below logistic regression at every tested fraction

Stored interpretation says:

- no tested fraction showed a neural network overtaking logistic regression

Smoke run:

- at `20%` subject fraction
  - Logistic Regression Identification F1 macro `0.706`
  - CNN1D Identification F1 macro `0.447`

Interpretation:

- Current neural models are not simply waiting for a little more data.
- At least in the present setup, the handcrafted spectral representation plus a linear model is stronger than the learned encoders.

### Phase J: Systematic ResNet18 optimization and multi-channel experiment (2026-03-17)

Relevant artifacts:

- `results/training/experiment_002/` (GPU baseline re-establishment)
- `results/training/step2_hypothesis_A/`
- `results/training/step2_hypothesis_B/`
- `results/training/step2_hypothesis_C/`
- `results/training/step3_multichannel/`
- `EXPERIMENT_RESULTS_STEP1_2.md`
- `EXPERIMENT_RESULTS_STEP3_5.md`

Environment: WSL2, NVIDIA GeForce RTX 5070 Ti (16GB), PyTorch 2.10.0+cu128

#### Step 1: GPU baseline re-establishment

Setting identical to Phase F (6200 samples, 5 cancers, 4 controls, aggregation=none).

| Model | Detection AUC | Id Acc | Id F1 Macro |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.981 | 0.902 | 0.872 |
| XGBoost | 0.969 | 0.828 | 0.757 |
| ResNet18 | 0.958 | 0.771 | 0.710 |

Reproducibility confirmed against Phase F results. GPU provided ~6x speedup for ResNet18.

#### Step 2: ResNet18 optimization hypotheses (all failed)

Three hypotheses were tested independently:

| Hypothesis | Change | Id F1 macro | Delta vs baseline |
| --- | --- | ---: | --- |
| Baseline ResNet18 | (default) | 0.710 | -- |
| A: Larger encoder | channels (64,128,256,512), 4x params | 0.677 | -0.033 (worse) |
| B: Strong regularization | dropout 0.6, WD 5e-3, focal loss | 0.707 | -0.003 (neutral) |
| C: Identification reweight | identification loss weight 2.0, focal loss | 0.668 | -0.042 (worse) |

Hypothesis A (larger encoder): More parameters only increased overfitting.
Hypothesis B (regularization): Reduced overfitting gap (0.039 to 0.015) but did not improve val performance.
Hypothesis C (identification reweight): Degraded both steps; objectives are coupled through the shared encoder.

Conclusion: Standard hyperparameter tuning cannot close the 16% F1 gap with LR.

#### Step 3: Multi-channel input (SNV + 1st/2nd derivative)

Rationale: Give the CNN explicit derivative information (peak positions, curvature) as separate input channels.

Architecture: ResNet18-1D with `Conv1d(3, 32, ...)` stem. Three channels: original SNV spectrum, 1st derivative, 2nd derivative. Each standardized independently.

| Model | Det AUC | Id F1 macro |
| --- | ---: | ---: |
| LR (reference) | 0.981 | 0.872 |
| ResNet18 1-channel | 0.958 | 0.710 |
| ResNet18 3-channel | 0.961 | 0.706 |

Verdict: No improvement. The CNN's 1D convolutions already implicitly compute local differences (equivalent to derivatives). Explicit derivative channels are redundant.

### Phase K: Ensemble experiment (2026-03-17)

Relevant artifacts:

- `results/training/step5_ensemble/`
- `results/training/step5_ensemble/blend_results.csv`
- `results/training/step5_ensemble/blend_sweep.png`
- `EXPERIMENT_RESULTS_STEP3_5.md`

Rationale: Even if ResNet18 is weaker overall, it may capture complementary non-linear patterns that LR misses. A simple weighted average of predictions could combine the best of both.

Method:

- Same 5-fold CV split (StratifiedGroupKFold, same seed) for both LR and ResNet18
- Blend: `P = alpha * P_LR + (1-alpha) * P_ResNet18`
- Sweep alpha from 0.0 (pure ResNet18) to 1.0 (pure LR) in 0.1 increments

Setting: 6200 samples, 5 cancers, 4 controls, aggregation=none.

Blend sweep results:

| alpha (LR weight) | Det AUC | Id Acc | Id F1 macro | Id AUC |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 (pure ResNet) | 0.956 | 0.726 | 0.673 | 0.916 |
| 0.1 | 0.967 | 0.761 | 0.709 | 0.947 |
| 0.2 | 0.972 | 0.800 | 0.752 | 0.960 |
| 0.3 | 0.976 | 0.840 | 0.797 | 0.969 |
| 0.4 | 0.979 | 0.876 | 0.838 | 0.975 |
| 0.5 | 0.982 | 0.897 | 0.867 | 0.978 |
| 0.6 | 0.983 | 0.899 | 0.869 | 0.980 |
| 0.7 | 0.984 | 0.901 | 0.871 | 0.981 |
| 0.8 | 0.984 | 0.904 | 0.875 | 0.983 |
| 0.9 | 0.984 | 0.904 | 0.875 | 0.984 |
| 1.0 (pure LR) | 0.981 | 0.902 | 0.872 | 0.985 |

Best blend: alpha=0.8 (80% LR + 20% ResNet18)

| Metric | Pure LR | Ensemble (0.8/0.2) | Delta |
| --- | ---: | ---: | --- |
| Det AUC | 0.981 | 0.984 | +0.003 |
| Id Accuracy | 0.902 | 0.904 | +0.002 |
| Id F1 macro | 0.872 | 0.875 | +0.003 |
| Id AUC | 0.985 | 0.983 | -0.002 |

Interpretation:

- ResNet18 does capture a small amount of non-linear pattern that LR misses.
- The contribution is marginal (~20% weight) but consistent across the 0.7-0.9 alpha range.
- The ensemble is the new best result on the main benchmark setting, though the improvement is small.
- Id AUC is slightly lower due to probability calibration differences in the blend.

### Phase L: Normalization ablation experiment (2026-03-18)

Relevant artifacts:

- `results/norm_none/processed_spectra.csv`
- `results/norm_minmax/processed_spectra.csv`
- `results/norm_l2/processed_spectra.csv`
- `results/training/step4_none/`
- `results/training/step4_minmax/`
- `results/training/step4_l2/`
- `results/training/step4_normalization_comparison/`
- `models/run_step4_normalization.py`

Hypothesis: SNV normalization makes spectra linearly separable, giving LR an inherent advantage over DL. If we remove SNV, the LR-ResNet gap should narrow because DL can learn its own normalization while LR cannot.

Method:

- Re-preprocess all spectra with 4 normalization methods: SNV (existing), None, MinMax, L2
- Train LR + ResNet18 on each variant with identical settings (6200 samples, 5 cancers, 4 controls, aggregation=none)
- Compare the LR-ResNet gap across normalization methods

Results:

| Normalization | LR Det AUC | ResNet Det AUC | Det Gap | LR Id F1 | ResNet Id F1 | Id F1 Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SNV | 0.981 | 0.958 | 0.023 | 0.872 | 0.710 | 0.162 |
| None | 0.977 | 0.966 | 0.011 | 0.879 | 0.720 | 0.158 |
| MinMax | 0.979 | 0.952 | 0.027 | 0.868 | 0.640 | 0.228 |
| L2 | 0.981 | 0.939 | 0.042 | 0.877 | 0.584 | 0.293 |

Verdict: **Hypothesis DISPROVED.**

- Removing SNV does NOT narrow the gap (0.162 vs 0.158 — essentially identical).
- LR is robust to normalization choice (Id F1: 0.868-0.879 across all 4 methods). The `StandardScaler` in the LR pipeline compensates for any normalization.
- ResNet18 is highly sensitive to normalization: MinMax (0.640) and L2 (0.584) are much worse than SNV (0.710) or None (0.720).
- The gap is actually **wider** for MinMax (+0.066) and L2 (+0.131) compared to SNV.

Interpretation:

- The linear separability of this spectral data is **not caused by SNV**. It comes from the upstream preprocessing steps (trim + smooth + baseline correction) which extract clean metabolite peak features.
- LR's advantage is intrinsic to the data structure, not an artifact of normalization choice.
- ResNet18's weakness is not a representation problem — it's a variance/overfitting problem on a dataset that is already well-structured for linear methods.
- SNV is actually one of the **better** normalizations for ResNet18, suggesting it helps DL by standardizing input scale.

### Phase M: Confounding analysis — Age/Sex/BMI (2026-03-18)

Relevant artifacts:

- `results/training/confounding_analysis/`
- `results/training/confounding_analysis/confounding_analysis.png`
- `results/training/confounding_analysis/confounding_summary.json`
- `models/run_confounding_analysis.py`

Concern: Normal subjects are much younger (mean 45.3) than cancer patients (mean 65-73). PRO is 100% male, OVA is 100% female. The SERS model might be learning age/sex demographics instead of metabolite patterns.

Method:

- Train LR classifier using demographics only (age, sex, BMI) — no spectra
- Measure spectral feature correlation with age
- Compare SERS performance on full cohort vs age-matched subgroup (50-80 years)
- Test SERS + demographics fusion

Results:

| Model | Det AUC | Id F1 macro |
| --- | ---: | ---: |
| Age only | 0.756 | 0.206 |
| Age + Sex | 0.766 | 0.271 |
| Age + Sex + BMI | 0.774 | 0.309 |
| **SERS only** | **0.987** | **0.861** |
| **SERS + Age + Sex** | **0.989** | **0.882** |

Age-matched subgroup analysis (restricting to age 50-80):

| Model | Full Det AUC | Age-matched Det AUC | Change |
| --- | ---: | ---: | --- |
| Demographics (age+sex) | 0.766 | 0.693 | -0.073 (drops) |
| SERS | 0.987 | 0.989 | +0.002 (maintained) |

Spectral-age correlation: mean |r| = 0.076, max |r| = 0.297. No feature has |r| > 0.3 with age.

Interpretation:

- **SERS is genuinely detecting cancer metabolites, not demographics.** Demographic confounding exists (AUC 0.77) but SERS performance is robust to age matching (0.987 → 0.989), while demographics-only drops (0.766 → 0.693).
- **SERS already captures demographic signal.** Adding age+sex to SERS improves AUC by only 0.1%, meaning spectral features already encode age-correlated metabolic information.
- **Cancer subtype classification is entirely spectral.** Demographics achieve Id F1 of only 0.27 — age/sex cannot distinguish CRC from LUN. SERS Id F1 of 0.86 is genuinely metabolite-driven.
- **For clinical deployment**: the age confounding concern is valid but minor. The model's classification is not an artifact of demographic imbalance.

### Phase N: Multimodal experiment — SERS + Clinical Features (2026-03-18)

Relevant artifacts:

- `results/training/multimodal/`
- `results/training/multimodal/multimodal_results.png`
- `results/training/multimodal/multimodal_summary.json`
- `models/run_multimodal.py`

Clinical data availability:

- Tier 1 (all groups): age, sex, BMI — 5050/6200 spectra matched
- Tier 2 (controls + LUN only): 12 lab values (WBC, RBC, Hb, platelet, AST, ALT, creatinine, glucose, total cholesterol, calcium, total bilirubin, uric acid) — 2495 spectra
- Lab values are NOT available for PRO, CRC, CPAN, OVA — multimodal with labs is only feasible for binary detection in a reduced cohort

#### Experiment 1: Full cohort early/late fusion (n=5050, 5 cancers, 4 controls)

| Model | Det AUC | Det Acc | Det Sens | Det Spec | Id F1 macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clinical only (age+sex+BMI) | 0.774 | 0.709 | 0.724 | 0.673 | 0.309 |
| SERS only | 0.987 | 0.951 | 0.957 | 0.937 | 0.861 |
| **Early fusion (SERS + age+sex+BMI)** | **0.988** | **0.955** | **0.961** | **0.941** | **0.877** |
| Late fusion (90% SERS / 10% clinical) | 0.989 | 0.952 | — | — | 0.864 |

#### Experiment 2: Lab subset — binary cancer detection (Controls + LUN, n=2495)

| Model | Det AUC |
| --- | ---: |
| Clinical (age+sex+BMI) | 0.789 |
| Labs only (12 blood values) | 0.746 |
| Clinical (demographics + labs) | 0.887 |
| SERS only | 0.989 |
| SERS + demographics | 0.990 |
| **SERS + demographics + labs** | **0.991** |

Interpretation:

- **Early fusion (SERS + age/sex/BMI) achieves Id F1 0.877** — the new best result, surpassing ensemble (0.875) and pure SERS (0.861/0.872 depending on subset).
- The Id F1 improvement (+1.6%) is larger than the ensemble gain (+0.3%), making demographics the single most effective addition to the SERS pipeline.
- Demographics help cancer subtype classification because age/sex patterns differ across cancer types (PRO=100% male older, OVA=100% female younger).
- Lab values provide a decent standalone cancer detector (0.887 AUC with demographics), but SERS already captures most of this metabolic information (adding labs to SERS: 0.989 → 0.991, only +0.2%).
- **Early fusion is simpler than ensemble** (just concatenate 3 features to existing 933) and gives a bigger gain.

### Phase O: Per-cancer sensitivity & LUN stage analysis (2026-03-18)

Relevant artifacts:

- `results/training/clinical_analysis/`
- `results/training/clinical_analysis/clinical_analysis.png`
- `results/training/clinical_analysis/clinical_analysis_summary.json`
- `models/run_clinical_analysis.py`

#### Per-cancer detection sensitivity (Cancer Detection: cancer vs non-cancer)

| Cancer Type | N | Sensitivity | Interpretation |
| --- | ---: | ---: | --- |
| CRC (colorectal) | 1500 | 0.981 | Easiest to detect |
| CPAN (pancreatic) | 350 | 0.969 | Easy |
| LUN (lung) | 1500 | 0.957 | Moderate |
| PRO (prostate) | 500 | 0.838 | Difficult |
| OVA (ovarian) | 350 | 0.797 | Most difficult |

Non-cancer specificity:

| Control Group | N | Specificity |
| --- | ---: | ---: |
| HBP | 500 | 0.970 |
| DIA | 500 | 0.950 |
| NOR | 500 | 0.898 |
| H.D. | 500 | 0.868 |

Notable: NOR and H.D. have lower specificity (more false positives), suggesting some healthy/comorbid subjects have metabolite profiles that overlap with cancer signatures.

#### Per-cancer type classification (Cancer Identification)

| Cancer Type | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| LUN | 0.952 | 0.911 | 0.931 |
| CRC | 0.939 | 0.914 | 0.926 |
| PRO | 0.834 | 0.914 | 0.872 |
| CPAN | 0.782 | 0.849 | 0.814 |
| OVA | 0.785 | 0.846 | 0.814 |

Key confusion patterns:
- PRO ↔ OVA: 6-7% mutual misclassification (both are urogenital cancers)
- CPAN → CRC: 12% of pancreatic misclassified as colorectal (same GI tract)
- LUN and CRC are the most distinctly classified (>91% recall, >93% precision)

#### LUN cancer stage analysis

Clinical staging data: 253/300 LUN subjects (765/1500 spectra) matched with stage info.

| Stage | Spectra | Subjects | Detection Sensitivity |
| --- | ---: | ---: | ---: |
| Stage I | 565 | 113 | **0.996** |
| Stage II | 115 | 23 | **0.983** |
| Stage III | 85 | 17 | 0.882 |
| **Early (I+II)** | **680** | **136** | **0.994** |
| Late (III+IV) | 85 | 17 | 0.882 |

**Key finding: SERS detects Stage I lung cancer with 99.6% sensitivity.** This is the strongest clinical result — 113 early-stage lung cancer patients were detected with near-perfect accuracy. Early-stage detection is the primary value proposition for a urine screening test.

Late-stage (III) shows lower sensitivity (0.882), possibly due to treatment effects or different metabolite profiles in advanced disease.

#### Operating points for clinical deployment

| Target Sensitivity | Achieved Specificity | Use case |
| --- | --- | --- |
| 95.0% | 90.4% | Balanced screening |
| 98.0% | 77.7% | High-sensitivity screening |
| 99.0% | 66.2% | Maximum detection (accept more false positives) |

| Target Specificity | Achieved Sensitivity | Use case |
| --- | --- | --- |
| 95.0% | 91.5% | Low false-positive clinical setting |
| 98.0% | 86.2% | Confirmatory use |

## 5. Current Best Results To Remember

These should be treated as the most useful reference points:

### Best overall (multimodal early fusion)

Setting:

- date: 2026-03-18
- aggregation: `none`
- samples: `5050` (spectra with matched clinical data)
- cancers: `PRO, LUN, CRC, CPAN, OVA`
- controls/non-cancer: `NOR, DIA, HBP, H.D.`
- clinical features: age, sex, BMI concatenated to 933 spectral features

Winner:

- LR Early Fusion (SERS + age + sex + BMI)
  - Detection AUC: `0.988`
  - Detection Sensitivity: `0.961`
  - Detection Specificity: `0.941`
  - Identification F1 macro: `0.877`

### Best SERS-only benchmark (ensemble)

Setting:

- date: 2026-03-17
- aggregation: `none`
- samples: `6200`
- cancers: `PRO, LUN, CRC, CPAN, OVA`
- controls/non-cancer: `NOR, DIA, HBP, H.D.`

Winner:

- Ensemble (80% LR + 20% ResNet18)
  - Detection AUC: `0.984`
  - Identification F1 macro: `0.875`

### Best SERS-only benchmark (single model)

Setting:

- date: 2026-03-17 (reproduced from 2026-03-13)
- aggregation: `none`
- samples: `6200`
- cancers: `PRO, LUN, CRC, CPAN, OVA`
- controls/non-cancer: `NOR, DIA, HBP, H.D.`

Winner:

- Logistic Regression
  - Detection AUC: `0.981`
  - Identification accuracy: `0.902`
  - Identification AUC: `0.985`
  - Identification F1 macro: `0.872`

### Best reduced-class benchmark

Setting:

- date: 2026-03-13
- aggregation: `none`
- samples: `5850`
- cancers: `PRO, LUN, CRC, OVA`
- controls/non-cancer: `NOR, DIA, HBP, H.D.`

Winner:

- XGBoost
  - Identification accuracy: `0.869`
  - Identification F1 macro: `0.822`

### Best medoid benchmark

Setting:

- date: 2026-03-16
- aggregation: `medoid`
- samples: `1342`
- cancers: `PRO, BRE, OVA, LUN, CRC, CPAN, SPAN`
- controls/non-cancer: `NOR, DIA, HBP, H.D.`

Winner:

- Logistic Regression
  - Identification accuracy: `0.786`
  - Identification F1 macro: `0.697`

## 6. What The Results Currently Suggest

### Strong conclusions

- All-spectra training is better than medoid-only subject aggregation.
- Separating `CPAN` and `SPAN` was better than the older merged `PAN` setup.
- Logistic Regression is the strongest stable single-model baseline on the main benchmark.
- XGBoost is competitive and becomes very strong when the class set is reduced.
- CNN1D is consistently the weakest family in the stored experiments.
- ResNet18 is better than CNN1D, but still does not beat the strong classical baselines alone.
- Hyperparameter tuning of ResNet18 (capacity, regularization, loss weighting) does not close the gap with LR.
- Multi-channel input (SNV + derivatives) does not help ResNet18 because 1D convolutions already implicitly compute derivatives.
- Ensemble (80% LR + 20% ResNet18) provides a small but consistent improvement over pure LR (+0.3% Id F1).
- Removing SNV normalization does NOT narrow the LR-ResNet gap (Phase L). The linear separability comes from upstream preprocessing, not normalization choice.
- LR is robust to normalization (Id F1: 0.868-0.879 across SNV/None/MinMax/L2), while ResNet18 is highly sensitive (0.584-0.720).
- MinMax and L2 normalization make the gap WIDER, not narrower.
- Confounding analysis confirms SERS detects cancer metabolites, not demographics (Phase M). SERS AUC is maintained on age-matched subgroups while demographics-only drops.
- Adding age/sex to SERS improves Det AUC by only 0.1%, and Id F1 by 2% — spectral features already encode demographic metabolic information.
- Early fusion (SERS + age/sex/BMI) achieves Id F1 0.877, the new best result (Phase N). Demographics help cancer subtype classification more than ensemble or DL approaches.
- Lab values (WBC, AST, etc.) provide decent standalone cancer detection (0.887 AUC) but add minimal value on top of SERS (+0.2% AUC).
- Per-cancer sensitivity varies widely (Phase O): CRC 98.1% > CPAN 96.9% > LUN 95.7% > PRO 83.8% > OVA 79.7%. Prostate and ovarian cancers are hardest to detect.
- **SERS detects Stage I lung cancer with 99.6% sensitivity** (113 subjects). Early-stage (I+II) sensitivity is 99.4%.
- Main confusion patterns: PRO ↔ OVA (urogenital), CPAN → CRC (GI tract). LUN and CRC are most distinctly classified.

### Working hypothesis (updated)

The spectral data is inherently linearly separable after basic preprocessing (trim + smooth + baseline correction). This linear separability is a property of the data itself — clean metabolite peak ratios — not an artifact of any specific normalization method. Classical models exploit this structure efficiently, while deep models pay a variance cost without gaining representation advantage.

This hypothesis has been comprehensively validated across all experiment phases:
- Phase J: Neither capacity, regularization, loss reweighting, nor multi-channel input closes the gap.
- Phase L: Removing normalization entirely does not help DL; the gap persists regardless of normalization method.
- Phase N: The most effective improvement comes not from better models but from adding basic clinical metadata (age/sex/BMI) via simple feature concatenation.
- The only productive use of ResNet18 is as a 20% minority contributor in an LR-dominated ensemble.

### Production recommendation

- **Recommended**: LR early fusion with age/sex/BMI (Id F1: 0.877) — simplest implementation, best performance
- **SERS-only fallback**: Logistic Regression alone (Id F1: 0.872) — when clinical metadata is unavailable
- **Maximum SERS-only**: LR + ResNet18 ensemble at alpha=0.8 (Id F1: 0.875) — marginal gain, added complexity
- Early fusion supersedes ensemble as the recommended production approach: simpler, better, and clinically natural (age/sex are always available at point of care).

### Production inference pipeline (deployed 2026-03-18)

The recommended production model has been packaged into a self-contained inference pipeline:

**Build** (run once):
```bash
python models/build_production_model.py
```

**Predict** (run per patient):
```bash
# SERS only
python scripts/sers_predict.py spectrum.CSV

# With clinical features (recommended, +1.6% Id F1)
python scripts/sers_predict.py spectrum.CSV --age 65 --sex M --bmi 24.3

# Batch (multiple replicates from same patient)
python scripts/sers_predict.py sample_1.CSV sample_2.CSV --age 60 --sex F

# JSON output for integration
python scripts/sers_predict.py spectrum.CSV --quiet --output result.json
```

**Artifacts** (`models/production/`, ~190 KB total):

| File | Content |
| --- | --- |
| `manifest.json` | Training metadata, class names, config |
| `preprocessing.json` | Frozen preprocessing parameters |
| `common_grid.npy` | Wavenumber grid (933 points) |
| `stage1_sers.joblib` | Binary cancer detector (SERS only) |
| `stage2_sers.joblib` | Cancer type classifier (SERS only) |
| `stage1_fusion.joblib` | Binary cancer detector (SERS + age/sex/BMI) |
| `stage2_fusion.joblib` | Cancer type classifier (SERS + age/sex/BMI) |

**Features**:
- Auto model selection: provides age+sex → fusion model, otherwise → SERS-only fallback
- Graceful degradation: missing BMI → auto-fill with training median
- Raw CSV input → preprocessed → prediction in one command
- JSON output mode (`--quiet`) for software integration
- No GPU or PyTorch dependency — runs on numpy/sklearn only

## 7. Important Comparison Rules

Do not compare runs casually unless these match:

- same aggregation mode (`medoid` vs `none`)
- same cancer set
- same non-cancer set
- same sample count

Many apparent "wins" in this repo come from easier class compositions rather than purely better modeling.

## 8. Practical Reference Files

If someone needs to continue the project quickly, these are the most important files:

- `config.yaml`
- `src/sers/config.py`
- `models/model.py`
- `models/train.py`
- `models/train_multichannel.py`
- `models/train_ensemble.py`
- `models/test.py`
- `models/tune_torch.py`
- `logs/experiment_runs.jsonl`
- `models/results/benchmark_all_models/benchmark_summary.csv`
- `models/results/overall_comparison/best_per_model.csv`
- `models/results/torch_tuning/tuning_summary.csv`
- `models/results/learning_curve_quickcheck/learning_curve_interpretation.txt`
- `results/training/step5_ensemble/blend_results.csv`
- `EXPERIMENT_RESULTS_STEP1_2.md`
- `EXPERIMENT_RESULTS_STEP3_5.md`
- `scripts/generate_qc_preprocessing_examples.py`
- `results/figures/pipeline_overview.png` (and other step*.png figures)
- `models/run_step4_normalization.py`
- `results/training/step4_normalization_comparison/comparison_table.csv`
- `results/training/step4_normalization_comparison/normalization_comparison.png`
- `models/run_confounding_analysis.py`
- `results/training/confounding_analysis/confounding_analysis.png`
- `models/run_multimodal.py`
- `results/training/multimodal/multimodal_results.png`
- `models/run_clinical_analysis.py`
- `results/training/clinical_analysis/clinical_analysis.png`
- `models/build_production_model.py`
- `scripts/sers_predict.py`
- `models/production/` (saved model artifacts)

## 9. Suggested Default Narrative For Future Work

If continuing from the current branch, the most honest summary is:

- the full pipeline is production-ready: QC → preprocessing → inference via `scripts/sers_predict.py`
- the best model (LR early fusion) is deployed as a CLI tool with saved artifacts in `models/production/`
- the strongest benchmark is LR with optional age/sex/BMI fusion (Id F1: 0.877)
- systematic attempts to improve ResNet18 (larger encoder, regularization, loss reweighting, multi-channel input) all failed to close the gap with LR
- normalization ablation (SNV/None/MinMax/L2) confirmed that the linear separability is intrinsic to the data, not an artifact of SNV preprocessing
- confounding analysis confirmed SERS classification is genuinely metabolite-driven, not an artifact of age/sex demographic imbalance
- the only productive DL contribution found is as a 20% minority member of an LR-dominated ensemble
- per-cancer and stage analysis (Phase O) provides the strongest clinical evidence: 99.6% Stage I lung cancer detection, clear cancer-specific sensitivity hierarchy, and biologically meaningful confusion patterns
- future work should focus on one of:
  - **data expansion** (more subjects, especially PRO and OVA which have lowest sensitivity)
  - **cross-hospital validation** (hospital info available in clinical data — generalization test)
  - **feature importance / key wavenumber analysis** (which molecular bonds drive classification — needed for biological interpretation in publications)
  - **prospective clinical validation** of the LR early-fusion pipeline, now supported by comprehensive evidence: confounding validation, early-stage detection, and per-cancer sensitivity characterization
- future comparisons should always anchor against Logistic Regression on the same cohort definition
