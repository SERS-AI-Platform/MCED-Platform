#!/usr/bin/env Rscript
# ============================================================================
# FINAL remaining R figures — DL internals, cross-instrument spectra,
# stacking v1, AB-XAI, confounding
# ============================================================================
`%||%` <- function(a, b) if (!is.null(a)) a else b
.get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  idx <- grep("--file=", args)
  if (length(idx)) return(dirname(normalizePath(sub("--file=", "", args[idx]))))
  "."
}
source(file.path(.get_script_dir(), "theme_sers.R"))

ROOT    <- "/home/user/SERS-AI"
REM     <- file.path(ROOT, "results/r_export/remaining")
OUTROOT <- file.path(ROOT, "results/figures/phase_archive_r")
safe_read <- function(path) { if (file.exists(path)) read_csv(path, show_col_types = FALSE) else NULL }

# ============================================================================
# 1. FiLM gamma/beta heatmap
# ============================================================================
cat("\n=== FiLM gamma/beta ===\n")
dir.create(file.path(OUTROOT, "dl_internals"), recursive = TRUE, showWarnings = FALSE)

film <- safe_read(file.path(REM, "dl_internals/film_gamma_beta_summary.csv"))
if (!is.null(film)) {
  # Top 20 features with largest gamma deviation from 1.0
  film$gamma_dev <- abs(film$gamma_mean - 1.0)
  top_feat <- film %>% arrange(desc(gamma_dev)) %>% slice_head(n = 24)

  p <- ggplot(top_feat, aes(factor(feature_idx), layer)) +
    geom_tile(aes(fill = gamma_mean), color = "white", linewidth = 0.8) +
    geom_text(aes(label = sprintf("%.3f", gamma_mean)), size = 2.8, family = KFONT) +
    scale_fill_gradient2(low = "#2563eb", mid = "#f8fafc", high = "#dc2626",
                         midpoint = 1.0, name = expression(gamma)) +
    labs(title = "FiLM — gamma parameter (상위 24 feature)",
         subtitle = "gamma > 1.0 = 해당 feature 증폭, < 1.0 = 억제. Clinical(age/sex/BMI)에 의한 modulation",
         x = "Feature index (channel)", y = "Layer",
         caption = "대부분 gamma ≈ 1.0 (거의 modulation 없음) → clinical 3개 변수만으로는 modulation 한계.") +
    theme_sers(11) + coord_equal()
  save_fig(p, file.path(OUTROOT, "dl_internals/DLF_film_gamma.png"), w = 11, h = 5)

  # Beta heatmap (same features)
  p2 <- ggplot(top_feat, aes(factor(feature_idx), layer)) +
    geom_tile(aes(fill = beta_mean), color = "white", linewidth = 0.8) +
    geom_text(aes(label = sprintf("%.3f", beta_mean)), size = 2.8, family = KFONT) +
    scale_fill_gradient2(low = "#2563eb", mid = "#f8fafc", high = "#dc2626",
                         midpoint = 0, name = expression(beta)) +
    labs(title = "FiLM — beta parameter (상위 24 feature)",
         subtitle = "beta = additive shift. 양수 = activation 증가, 음수 = 감소",
         x = "Feature index (channel)", y = "Layer") +
    theme_sers(11) + coord_equal()
  save_fig(p2, file.path(OUTROOT, "dl_internals/DLF_film_beta.png"), w = 11, h = 5)
}

# CrossAttention weights
xattn <- safe_read(file.path(REM, "dl_internals/xattn_layer_weights.csv"))
if (!is.null(xattn) && "q_norm" %in% names(xattn)) {
  xattn_avg <- xattn %>%
    filter(!is.na(q_norm)) %>%
    group_by(layer) %>%
    summarise(Q = mean(q_norm), K = mean(k_norm), V = mean(v_norm), Out = mean(out_norm),
              .groups = "drop") %>%
    pivot_longer(-layer, names_to = "component", values_to = "norm")
  p <- ggplot(xattn_avg, aes(component, factor(layer), fill = norm)) +
    geom_tile(color = "white", linewidth = 1) +
    geom_text(aes(label = sprintf("%.1f", norm)), size = 3.5, family = KFONT) +
    scale_fill_viridis_c(option = "mako", direction = -1, end = 0.85, name = "L2 norm") +
    labs(title = "CrossAttention — Layer × Component weight norms",
         subtitle = "Q/K/V/Out 각 projection의 L2 norm (5-fold 평균)",
         x = "Component", y = "Layer") +
    theme_sers() + coord_equal()
  save_fig(p, file.path(OUTROOT, "dl_internals/DLF_xattn_weights.png"), w = 8, h = 5)
}

# ============================================================================
# 2. Cross-instrument — mean spectra comparison (Thermo vs Medical)
# ============================================================================
cat("\n=== Cross-instrument spectra ===\n")
dir.create(file.path(OUTROOT, "cross_instrument"), recursive = TRUE, showWarnings = FALSE)

twodev <- safe_read(file.path(REM, "cross_instrument/two_device_mean_spectra_comparison.csv"))
if (!is.null(twodev)) {
  # Subsample for speed
  twodev_sub <- twodev %>% filter(wavenumber %% 8 < 2)

  p <- ggplot(twodev_sub, aes(wavenumber, mean_intensity, color = instrument)) +
    geom_line(linewidth = 0.4, alpha = 0.7) +
    facet_wrap(~ group, scales = "free_y", ncol = 3) +
    scale_color_manual(values = c(Thermo = "#1d4ed8", Medical = "#dc2626"), name = "기기") +
    labs(title = "Two-device 평균 스펙트럼 비교 (Thermo vs Medical)",
         subtitle = "동일 환자 그룹을 두 기기로 측정 — 전체 형태는 유사하나 intensity 차이 큼",
         x = "Wavenumber (cm-1)", y = "Mean intensity",
         caption = "파란색=Thermo(연구용), 빨간색=Medical(휴대용).\n기기 간 intensity scale 차이 → cross-training 실패 원인.") +
    theme_sers(10) + theme(legend.position = "top")
  save_fig(p, file.path(OUTROOT, "cross_instrument/XI_two_device_spectra.png"), w = 13, h = 9)
}

# Cross-instrument summary bar
xi_sum <- safe_read(file.path(REM, "cross_instrument/cross_instrument_summary.csv"))
if (!is.null(xi_sum)) {
  xi_sum$scenario_label <- c("Thermo 자체 CV", "Medical 자체 CV",
                              "Thermo→Medical", "Medical→Thermo")[1:nrow(xi_sum)]
  p <- ggplot(xi_sum, aes(reorder(scenario_label, s1_auc), s1_auc,
                           fill = ifelse(grepl("→", scenario_label), "cross", "self"))) +
    geom_col(width = 0.6, alpha = 0.85) +
    geom_text(aes(label = sprintf("%.3f", s1_auc)), hjust = -0.1, size = 4, family = KFONT) +
    coord_flip() +
    scale_fill_manual(values = c(self = "#0d9488", cross = "#dc2626"),
                      labels = c(self = "자체 학습/평가", cross = "교차 평가"), name = "") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(title = "Cross-Instrument 요약 — 자체 vs 교차 AUC",
         subtitle = "동일 1,628명: 자체 학습 시 AUC 0.90+, 교차 전이 시 0.48~0.62",
         x = NULL, y = "Stage 1 AUC",
         caption = "결론: 기기 간 transfer 불가. 기기별 독립 모델 필수.\nPDS calibration 적용 시 0.92까지 개선 가능 (Phase CIC 참조).") +
    theme_sers()
  save_fig(p, file.path(OUTROOT, "cross_instrument/XI_cross_summary.png"), w = 10, h = 5.5)
}

# ============================================================================
# 3. Stacking V1 (Thermo + Medical per-model)
# ============================================================================
cat("\n=== Stacking V1 ===\n")
dir.create(file.path(OUTROOT, "stacking_v1"), recursive = TRUE, showWarnings = FALSE)

stk1 <- safe_read(file.path(REM, "stacking_v1/stacking_v1_combined_comparison.csv"))
if (!is.null(stk1)) {
  p <- ggplot(stk1, aes(auc, f1_type, color = device, shape = type)) +
    geom_point(size = 4, alpha = 0.8) +
    geom_text(aes(label = name), size = 2.8, vjust = -1.2, family = KFONT, check_overlap = TRUE) +
    scale_color_manual(values = c(thermo = "#1d4ed8", medical = "#dc2626"), name = "기기") +
    scale_shape_manual(values = c(base = 16, ensemble = 17), name = "종류") +
    labs(title = "Stacking V1 — 기기별 base model + ensemble 비교",
         subtitle = "Thermo(파랑) vs Medical(빨강), 9 base + ensemble",
         x = "Stage 1 AUC", y = "Type ID F1",
         caption = "Thermo가 전반적으로 Medical보다 우수.\nEnsemble(▲)이 single base(●) 대비 개선.") +
    theme_sers()
  save_fig(p, file.path(OUTROOT, "stacking_v1/STK_v1_comparison.png"), w = 10, h = 7)
}

# ============================================================================
# 4. AB-XAI — SHAP profile
# ============================================================================
cat("\n=== AB-XAI ===\n")
dir.create(file.path(OUTROOT, "ab_xai"), recursive = TRUE, showWarnings = FALSE)

shap_prof <- safe_read(file.path(REM, "ab_xai/ab_xai_shap_profiles.csv"))
if (!is.null(shap_prof)) {
  shap_long <- shap_prof %>%
    pivot_longer(-wavenumber, names_to = "profile", values_to = "shap") %>%
    filter(wavenumber %% 6 < 2)
  p <- ggplot(shap_long, aes(wavenumber, shap, color = profile)) +
    geom_line(linewidth = 0.5, alpha = 0.8) +
    geom_hline(yintercept = 0, color = "#94a3b8", linewidth = 0.3) +
    scale_color_viridis_d(option = "turbo", end = 0.9, name = "Profile") +
    labs(title = "AB-XAI — CRC↔PAN 혼동 분석 SHAP profile",
         subtitle = "정분류 vs 오분류 샘플의 SHAP 차이 — 어디서 혼동이 발생하는지",
         x = "Wavenumber (cm-1)", y = "SHAP value",
         caption = "CRC_correct vs PAN_confused_as_CRC:\nSHAP 차이가 큰 파장 = 모델이 혼동하는 원인 영역.") +
    theme_sers() + theme(legend.position = "top")
  save_fig(p, file.path(OUTROOT, "ab_xai/AB_shap_profiles.png"), w = 12, h = 6)
}

# ============================================================================
# 5. Confounding analysis
# ============================================================================
cat("\n=== Confounding ===\n")
conf <- safe_read(file.path(REM, "bonus/confounding_analysis_summary.csv"))
if (!is.null(conf) && nrow(conf) > 3) {
  # Parse key-value if needed
  metric_rows <- conf %>% filter(grepl("auc|f1|confounder", key, ignore.case = TRUE))
  if (nrow(metric_rows) > 0) {
    p <- ggplot(metric_rows, aes(reorder(key, as.numeric(value)), as.numeric(value))) +
      geom_col(fill = "#0d9488", width = 0.6) +
      coord_flip() +
      labs(title = "Phase M — Confounding 변수 분석",
           x = NULL, y = "Score") +
      theme_sers()
    save_fig(p, file.path(OUTROOT, "03_clinical_fusion/M_confounding_R.png"), w = 9, h = 6)
  }
}

# ============================================================================
# 6. Multiview channel statistics
# ============================================================================
cat("\n=== Multiview channels ===\n")
mv_ch <- safe_read(file.path(REM, "bonus/multiview_channel_statistics.csv"))
if (!is.null(mv_ch) && nrow(mv_ch) > 0) {
  ch_col <- intersect(names(mv_ch), c("channel", "view", "name"))[1]
  val_col <- intersect(names(mv_ch), c("correlation", "mean_corr", "value"))[1]
  if (!is.na(ch_col) && !is.na(val_col)) {
    p <- ggplot(mv_ch %>% slice_head(n = 20),
                aes(reorder(.data[[ch_col]], .data[[val_col]]), .data[[val_col]])) +
      geom_col(fill = "#6d28d9", width = 0.6, alpha = 0.8) +
      coord_flip() +
      labs(title = "Multi-view channel 간 상관/통계",
           x = NULL, y = val_col) +
      theme_sers()
    save_fig(p, file.path(OUTROOT, "07_multiview/MV_channel_correlation_R.png"), w = 10, h = 6)
  }
}

cat("\n[done] final remaining figures complete.\n")
cat("output:", OUTROOT, "\n")
