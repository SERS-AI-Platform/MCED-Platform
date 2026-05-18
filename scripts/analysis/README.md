# scripts/analysis triage

Last reviewed: 2026-05-11

This directory is not the model source of truth. It contains post-hoc analysis,
figure generation, one-off experiments, and legacy experiment drivers. The
current model source of truth is:

- Registry: `src/sers/models/_registry.py`
- Production model: `usersnet`
- Public name: `uSERS-Net`
- Internal architecture: Stacking V2, 10 base models + ElasticNet meta learner
- Artifact pointer: `artifacts/usersnet/current -> artifacts/usersnet/v1.0.0`
- Production builder: `scripts/training/build_usersnet_production.py`
- Stacking trainer: `scripts/training/train_usersnet.py`

Important naming rule:

- `uSERS-Net` means the Stacking V2 production model above.
- `SERS-Net Ensemble` means the legacy `0.8 * LR_fusion + 0.2 * ResNet18`
  baseline in `scripts/training/_legacy/run_sersnet_ensemble.py`.
- Any analysis that calls the alpha-blend baseline `uSERS-Net` should be renamed
  or annotated before being used in reports.

## Active / keep

These scripts are still connected to current model interpretation, validation,
or documented result folders.

| Script | Keep reason | Main result connection |
|---|---|---|
| `stk_v2_full_eval.py` | Re-evaluates STK-V2 and peak interpretation | `results/training/stacking_optimization_v2/`, legacy weekend STK outputs |
| `stk_v2_meta_shap.py` | Meta-learner input attribution | `results/training/stacking_optimization_v2/` |
| `plot_stk_v2_confusion_matrices.py` | Publication confusion matrices | `results/training/stacking_v2/oof_predictions.npz` |
| `plot_stk_v2_nested_cv_cm.py` | Nested-CV confusion matrices | `results/training/stacking_v2/oof_predictions.npz` |
| `validate_pds_stacking.py` | PDS calibration validation against stacking artifacts | `artifacts/usersnet/current`, `results/cross_instrument/calibration/` |
| `confounding_analysis.py` | Age/sex/BMI confounding check | `results/training/confounding_analysis/` |
| `clinical_breakdown.py` | Per-cancer and clinical subgroup breakdown | `results/training/clinical_analysis/` |
| `xai_fusion_analysis.py` | FiLM/Cross-Attention interpretation | `results/training/film_xattn_comparison/` |
| `phase_ab_crc_pan_xai.py` | CRC/PAN confusion interpretation | `results/training/phase_ab_xai/` |
| `cross_instrument_analysis.py` | Thermo vs medical instrument analysis | `results/training/cross_instrument/` |
| `cross_instrument_calibration.py` | PDS calibration transfer experiment | `results/cross_instrument/calibration/` |
| `cross_instrument_sweep.py` | Calibration sweep | `results/cross_instrument/calibration/sweep/` |
| `lr_coefficient_peak_analysis.py` | LR coefficient and peak overlay | `results/training/peak_analysis/` |
| `peak_ratio_feature_experiment.py` | Peak-ratio feature experiment | `results/training/peak_ratio_experiment/` |
| `peak_fitting_ablation.py` | Voigt peak fitting ablation | `results/training/peak_validation/` or run-specific output |
| `multiview_derivative_viz.py` | 3-channel spectrum validation figures | `results/training/multiview_derivative/` |
| `multiview_ensemble.py` | Pre-STK multiview ensemble benchmark | `results/training/multiview_ensemble/` |
| `multiview_ablation.py` | Multiview ablation | `results/training/multiview_ablation/` |
| `optimize_7cancer_full.py` | Historical full optimization record | `results/training/optimize_7cancer_full/` |
| `zopt_comprehensive_figures.py` | Figures for full optimization | `results/training/optimize_7cancer_full/` |

## Support / keep but not model code

These are data preparation, dashboard export, or publication figure scripts.
Keep them separate from model evaluation.

| Script/group | Role |
|---|---|
| `build_master_clinical.py`, `generate_exclusion_registry.py`, `generate_phase_x_exclusions.py`, `trace_missing_clean_vs_spectral.py` | Clinical/spectral cohort bookkeeping |
| `build_cancer_xlsx.py`, `create_rds.py` | Clinical export formats |
| `analyze_equipment.py`, `analyze_equipment_qc.py`, `prepare_medical_eval_data.py`, `prepare_peak_comparison.py` | Equipment/dashboard data generation |
| `export_group_spectra.py`, `extract_dashboard_spectra.py`, `extract_preprocessing_stages.py`, `generate_qc_preprocessing_examples.py` | Visualization/export helpers |
| `poster_figures.py`, `poster_figures_r.R`, `poster_latex_tables.R`, `qc_diagnostic_dashboard.R` | Publication or dashboard output |
| `tsne_groups.py`, `unified_subgroup_interactive.py`, `blc_subgroup_clinical.py`, `blc_subgroup_interactive.py`, `all_subgroup_interactive.py` | Exploratory subgroup visualization |

## Archive candidates

Move these under `scripts/experiments/_archive/` after confirming the linked
result folder is preserved.

| Candidate | Reason |
|---|---|
| `weekend_experiments/` | Its STK-V2 work has mostly moved into `scripts/training/train_usersnet.py` and `results/training/stacking_v2/`; keep only as dated experiment archive. |
| `clinical_fusion_experiment.py`, `clinical_fusion_full.py` | v1.1.0 clinical fusion exploration, not current production. |
| `feature_representation_ablation.py`, `plot_ablation_results.py` | Completed ablation/figure workflow. |
| `learning_curve_comparison.py` | Historical comparison; keep only if still needed for paper supplement. |
| `compare_baseline_methods.py`, `compare_baseline_phaseQ.py`, `compare_grid_performance.py` | Completed preprocessing/grid ablations. |
| `run_cpan_binary_experiment.py`, `generate_pancreatic_reports.py`, `pan_subgroup_peak_diff.py` | Pancreatic-specific one-off analyses. |
| `two_device_visualization.py`, `weekend_results_analysis.py` | Derived summary/figure scripts from older experiment batches. |

## Needs naming review

| Candidate | Issue |
|---|---|
| `usersnet_ovr_roc.py` | Uses alpha-blend LR_fusion + ResNet18 outputs but labels them `uSERS-Net`; this conflicts with the registry definition of `uSERS-Net` as Stacking V2. |
| `usersnet_resnet18_ig.py` | ResNet18 attribution script writes into the same alpha-blend run folder; it should be labeled as ResNet18/legacy attribution unless it is explicitly tied to Stacking V2. |

## Remove or relocate candidates

Do not delete these blindly. Remove or move after checking they are not the only
copy of a result artifact.

| Candidate | Action |
|---|---|
| `__pycache__/` | Delete; generated Python cache. |
| `medical_eval_data.json`, `medical_detailed_eval.json`, `offset_analysis.json`, `peak_comparison.json`, `peak_comparison_data.json` | Move to `results/analysis/dashboard_data/` or the dashboard/publication folder that consumes them. |
| `test_transform.py` | Convert to a real test or archive; it is exploratory top-level code. |
| Scripts with no CLI args and top-level execution | Add `argparse`/README entry or archive if not reproducible. |

## Cleanup rule

Before deleting a script:

1. Record its latest known output folder.
2. Confirm whether it is referenced by CLI, PyInstaller spec, docs, or figure exporters.
3. If it produced a published figure/table, archive instead of deleting.
4. If it is only generated cache or duplicate JSON output, delete or relocate.
