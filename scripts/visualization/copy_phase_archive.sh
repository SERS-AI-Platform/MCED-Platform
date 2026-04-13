#!/usr/bin/env bash
# Copy curated phase archive figures into the dashboard, grouped by category.
set -e
SRC=/home/user/SERS-AI/results/figures
DST=/home/user/workspace/solum-dashboard/figures/phase_archive
mkdir -p "$DST"/{01_baseline,02_normalization,03_clinical_fusion,04_held_out,05_single_cancer,06_cross_instrument,07_multiview,08_dl_models,09_ensemble_stacking,10_zopt_uSERS,11_interpretation,12_weekend_advanced,13_stk_v2_R}

# 01 — Baseline & 5-cancer benchmark (F, K)
cp "$SRC/training/phase_F_lr/stage1_cancer_vs_non_cancer.png"     "$DST/01_baseline/F_lr_stage1.png"
cp "$SRC/training/phase_F_benchmark/benchmark_comparison.png"     "$DST/01_baseline/F_benchmark_comparison.png"
cp "$SRC/training/phase_F_benchmark/benchmark_train_val_gap.png"  "$DST/01_baseline/F_benchmark_overfit_gap.png"
cp "$SRC/training/phase_K_ensemble/blend_sweep.png"               "$DST/01_baseline/K_ensemble_blend_sweep.png"

# 02 — Normalization ablation (L)
cp "$SRC/training/phase_L_comparison/normalization_comparison.png" "$DST/02_normalization/L_normalization.png"

# 03 — Clinical fusion (M, N)
cp "$SRC/training/phase_M_confounding/confounding_analysis.png"   "$DST/03_clinical_fusion/M_confounding.png"
cp "$SRC/training/phase_M_clinical/clinical_analysis.png"         "$DST/03_clinical_fusion/M_clinical.png"
cp "$SRC/training/phase_N_multimodal/multimodal_results.png"      "$DST/03_clinical_fusion/N_multimodal_fusion.png"

# 04 — Held-out validation (W, X)
cp "$SRC/training/phase_W_r_heldout/train_val_test_results.png"    "$DST/04_held_out/W_5cancer_heldout.png"
cp "$SRC/training/phase_W_r_heldout_7c/train_val_test_results.png" "$DST/04_held_out/W_7cancer_heldout.png"
cp "$SRC/training/phase_X_r_heldout/train_val_test_results.png"    "$DST/04_held_out/X_clean_cohort_heldout.png"
cp "$SRC/training/phase_X_clean/stage1_cancer_vs_non_cancer.png"   "$DST/04_held_out/X_clean_stage1.png"

# 05 — Single cancer (BLC, PAN)
cp "$SRC/training/phase_BLC_M/blc_mfds_comparison.png"  "$DST/05_single_cancer/BLC_MFDS_comparison.png"
cp "$SRC/training/phase_BLC_M/blc_mfds_roc.png"         "$DST/05_single_cancer/BLC_MFDS_roc.png"
cp "$SRC/training/phase_PAN_experiment/roc_curve_test.png"          "$DST/05_single_cancer/PAN_roc.png"
cp "$SRC/training/phase_PAN_experiment/confusion_matrix_test.png"   "$DST/05_single_cancer/PAN_confusion.png"
cp "$SRC/training/phase_T_blc/stage1_cancer_vs_non_cancer.png"      "$DST/05_single_cancer/T_BLC_8class_stage1.png"
cp "$SRC/training/phase_U_bre/stage1_cancer_vs_non_cancer.png"      "$DST/05_single_cancer/U_BRE_7class_stage1.png"

# 06 — Cross-instrument (XI, two_device)
cp "$SRC/training/phase_XI/cross_instrument_correlation.png"   "$DST/06_cross_instrument/XI_correlation.png"
cp "$SRC/training/phase_XI/cross_instrument_summary.png"       "$DST/06_cross_instrument/XI_summary.png"
cp "$SRC/training/phase_XI/spectral_comparison_by_group.png"   "$DST/06_cross_instrument/XI_spectral_by_group.png"
cp "$SRC/two_device/confusion_matrices.png"                    "$DST/06_cross_instrument/two_device_confusion.png"
cp "$SRC/two_device/feature_importance.png"                    "$DST/06_cross_instrument/two_device_importance.png"
cp "$SRC/two_device/mean_spectra_comparison.png"               "$DST/06_cross_instrument/two_device_mean_spectra.png"

# 07 — Multi-view / feature representation (MV, FR, AB_xai)
cp "$SRC/training/phase_MV/ensemble_results_thermo.png"          "$DST/07_multiview/MV_ensemble_thermo.png"
cp "$SRC/training/phase_MV_ablation/ablation_summary.png"        "$DST/07_multiview/MV_ablation_summary.png"
cp "$SRC/training/phase_MV_derivative/group_mean_3ch.png"        "$DST/07_multiview/MV_derivative_group_mean.png"
cp "$SRC/training/phase_MV_derivative/channel_correlation.png"   "$DST/07_multiview/MV_channel_correlation.png"
cp "$SRC/training/phase_FR_pk_ratio/comparison_chart.png"        "$DST/07_multiview/FR_peak_ratio_comparison.png"
cp "$SRC/training/phase_FR_pk_ratio/peak_detection.png"          "$DST/07_multiview/FR_peak_detection.png"
cp "$SRC/training/phase_AB_xai/analysis1_lr_coefficient_overlay_v1.png" "$DST/07_multiview/AB_lr_coefficient_overlay.png"
cp "$SRC/training/phase_AB_xai/analysis2_shap_spectrum_overlay_v1.png"  "$DST/07_multiview/AB_shap_spectrum_overlay.png"
cp "$SRC/training/phase_AB_xai/supplementary_tsne_embedding_v1.png"     "$DST/07_multiview/AB_tsne_embedding.png"

# 08 — Deep learning models (DLF, J ResNet, CON)
cp "$SRC/training/phase_DLF_comparison/xai_model_comparison.png"          "$DST/08_dl_models/DLF_model_comparison.png"
cp "$SRC/training/phase_DLF_comparison/xai_attention_radar.png"           "$DST/08_dl_models/DLF_attention_radar.png"
cp "$SRC/training/phase_DLF_comparison/xai_film_gamma_beta_heatmap.png"   "$DST/08_dl_models/DLF_film_gamma_beta.png"
cp "$SRC/training/phase_DLF_comparison/xai_cross_attention_weights.png"   "$DST/08_dl_models/DLF_cross_attention_weights.png"
cp "$SRC/training/phase_J_resnet18/training_curves.png"                   "$DST/08_dl_models/J_resnet18_training.png"
cp "$SRC/training/phase_F_resnet18/training_curves.png"                   "$DST/08_dl_models/F_resnet18_training.png"
cp "$SRC/training/phase_CON/pretrain_loss.png"                            "$DST/08_dl_models/CON_pretrain_loss.png"

# 09 — Ensemble & stacking (STK, K, weekend stacking)
cp "$SRC/training/phase_STK/stacking_results_thermo.png"           "$DST/09_ensemble_stacking/STK_thermo.png"
cp "$SRC/training/phase_STK/stacking_results_medical.png"          "$DST/09_ensemble_stacking/STK_medical.png"
cp "$SRC/weekend/_summary/01_stacking_confusion_matrices.png"      "$DST/09_ensemble_stacking/weekend_confusion.png"
cp "$SRC/weekend/_summary/02_stacking_roc_curves.png"              "$DST/09_ensemble_stacking/weekend_roc.png"
cp "$SRC/weekend/_summary/06_meta_learner_comparison.png"          "$DST/09_ensemble_stacking/weekend_meta_compare.png"
cp "$SRC/weekend/_summary/07_base_model_analysis.png"              "$DST/09_ensemble_stacking/weekend_base_analysis.png"
cp "$SRC/weekend/_summary/00_weekend_summary.png"                  "$DST/09_ensemble_stacking/weekend_summary.png"

# 10 — Z-opt uSERS-Net comprehensive
cp "$SRC/zopt_comprehensive/fig1_wavenumber_feature_ablation.png"  "$DST/10_zopt_uSERS/zopt_fig1_wavenumber.png"
cp "$SRC/zopt_comprehensive/fig2_aggregation_comparison.png"       "$DST/10_zopt_uSERS/zopt_fig2_aggregation.png"
cp "$SRC/zopt_comprehensive/fig3_architecture_search.png"          "$DST/10_zopt_uSERS/zopt_fig3_architecture.png"
cp "$SRC/zopt_comprehensive/fig4_held_out_robustness.png"          "$DST/10_zopt_uSERS/zopt_fig4_held_out.png"
cp "$SRC/zopt_comprehensive/fig5_grand_summary.png"                "$DST/10_zopt_uSERS/zopt_fig5_grand_summary.png"
cp "$SRC/zopt_comprehensive/fig6_per_cancer_accuracy.png"          "$DST/10_zopt_uSERS/zopt_fig6_per_cancer.png"

# 11 — Interpretation (spectral_interpretation_7cancer, PI_peaks)
cp "$SRC/spectral_interpretation_7cancer/00_combined_spectral_interpretation.png" "$DST/11_interpretation/SI7_combined.png"
cp "$SRC/spectral_interpretation_7cancer/01_feature_importance_lr_coefficients.png" "$DST/11_interpretation/SI7_lr_coefficients.png"
cp "$SRC/spectral_interpretation_7cancer/02_mean_spectra_overlay.png"   "$DST/11_interpretation/SI7_mean_spectra.png"
cp "$SRC/spectral_interpretation_7cancer/03_difference_spectra.png"     "$DST/11_interpretation/SI7_difference_spectra.png"
cp "$SRC/spectral_interpretation_7cancer/04_top_discriminating_wavenumbers.png" "$DST/11_interpretation/SI7_top_wavenumbers.png"
cp "$SRC/spectral_interpretation_7cancer/05_coefficient_heatmap_7class.png"     "$DST/11_interpretation/SI7_coefficient_heatmap.png"
cp "$SRC/spectral_interpretation_7cancer/06_bre_focused_analysis.png"           "$DST/11_interpretation/SI7_BRE_focus.png"
cp "$SRC/training/phase_PI_peaks/cancer_peak_discrimination_heatmap.png"        "$DST/11_interpretation/PI_peaks_heatmap.png"
cp "$SRC/training/phase_PI_peaks/stage1_binary_coefficient_spectrum.png"        "$DST/11_interpretation/PI_peaks_stage1.png"
cp "$SRC/training/phase_PI_peaks/stage2_per_cancer_coefficient_spectra.png"     "$DST/11_interpretation/PI_peaks_stage2.png"

# 12 — Weekend advanced experiments (CON-SW, LOHO, NFS, PI)
cp "$SRC/weekend/_summary/08_contrastive_sweep.png"           "$DST/12_weekend_advanced/CON_SW_sweep.png"
cp "$SRC/weekend/_summary/09_loho_validation.png"             "$DST/12_weekend_advanced/LOHO_validation.png"
cp "$SRC/weekend/_summary/10_permutation_importance.png"      "$DST/12_weekend_advanced/PI_permutation.png"
cp "$SRC/weekend/_summary/11_norm_feature_search.png"         "$DST/12_weekend_advanced/NFS_sweep.png"
cp "$SRC/weekend/_summary/12_norm_feature_top10.png"          "$DST/12_weekend_advanced/NFS_top10.png"

# 13 — STK-V2 R-redrawn (Korean labels)
cp "$SRC/stk_v2_r/"*.png "$DST/13_stk_v2_R/"

echo "[done] phase_archive populated:"
find "$DST" -type f -name "*.png" | wc -l
echo "files total"
