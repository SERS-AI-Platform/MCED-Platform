#!/usr/bin/env Rscript
# ============================================================================
# Phase Archive EXTENDED — remaining categories (④⑤⑧⑩⑪⑫)
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
RDATA   <- file.path(ROOT, "results/r_export")
OUTROOT <- file.path(ROOT, "results/figures/phase_archive_r")

safe_read <- function(path) {
  if (file.exists(path)) read_csv(path, show_col_types = FALSE) else NULL
}

# ============================================================================
# 04 — HELD-OUT VALIDATION
# ============================================================================
cat("\n=== 04 Held-out ===\n")
dir.create(file.path(OUTROOT, "04_heldout"), recursive = TRUE, showWarnings = FALSE)

ho7 <- safe_read(file.path(RDATA, "04_heldout/heldout_7cancer_splits.csv"))
if (!is.null(ho7)) {
  ho_val <- ho7 %>% filter(split == "test") %>%
    mutate(repeat_label = paste("Repeat", `repeat` + 1))

  ho_summary <- ho_val %>%
    summarise(mean_auc = mean(det_auc, na.rm=T), sd_auc = sd(det_auc, na.rm=T),
              mean_f1 = mean(id_f1_macro, na.rm=T), sd_f1 = sd(id_f1_macro, na.rm=T),
              n = n())

  ho_long <- ho_val %>%
    select(repeat_label, det_auc, id_f1_macro) %>%
    pivot_longer(-repeat_label, names_to = "metric", values_to = "value") %>%
    mutate(metric = recode(metric,
                           det_auc = "Detection AUC",
                           id_f1_macro = "Type ID F1 macro"))

  p <- ggplot(ho_long, aes(repeat_label, value, fill = metric)) +
    geom_col(position = "dodge", width = 0.6, alpha = 0.85) +
    geom_text(aes(label = sprintf("%.3f", value)),
              position = position_dodge(0.6), vjust = -0.3, size = 3.2, family = KFONT) +
    scale_fill_manual(values = c("Detection AUC" = "#0d9488", "Type ID F1 macro" = "#6d28d9"), name = "") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15)), limits = c(0, 1)) +
    labs(title = "Phase W — 7-cancer Held-out 검증 (60/20/20 split, 5 repeats)",
         subtitle = sprintf("Test set 평균: AUC %.3f ± %.3f, F1 %.3f ± %.3f",
                           ho_summary$mean_auc, ho_summary$sd_auc,
                           ho_summary$mean_f1, ho_summary$sd_f1),
         x = NULL, y = "Score",
         caption = "각 repeat는 무작위 split → 안정적으로 AUC 0.93+, F1 0.68+.\nheld-out에서 CV 대비 약간 하락은 정상 (데이터 분할 효과).") +
    theme_sers()
  save_fig(p, file.path(OUTROOT, "04_heldout/W_7cancer_heldout.png"), w = 10, h = 6)
}

# Per-group held-out
pg7 <- safe_read(file.path(RDATA, "04_heldout/heldout_7cancer_per_group.csv"))
if (!is.null(pg7)) {
  pg_mean <- pg7 %>%
    filter(split == "test") %>%
    group_by(group) %>%
    summarise(mean_sens = mean(sensitivity, na.rm = T), .groups = "drop") %>%
    filter(!is.na(mean_sens)) %>%
    arrange(desc(mean_sens)) %>%
    mutate(mean_f1 = mean_sens)  # use sensitivity as proxy
  pg_mean$group <- factor(pg_mean$group, levels = rev(pg_mean$group))

  p <- ggplot(pg_mean, aes(mean_sens, group, fill = mean_sens)) +
    geom_col(width = 0.6) +
    geom_text(aes(label = sprintf("%.3f", mean_sens)), hjust = -0.15, size = 3.5, family = KFONT) +
    scale_fill_viridis_c(option = "mako", direction = -1, end = 0.85, guide = "none") +
    scale_x_continuous(expand = expansion(mult = c(0, 0.2))) +
    labs(title = "Phase W — 암종/그룹별 Held-out Sensitivity",
         subtitle = "7-cancer, test split 평균",
         x = "Mean Sensitivity", y = NULL,
         caption = "암종별 sensitivity 차이가 있음.\n그룹 크기 및 스펙트럼 특성에 따라 변동.") +
    theme_sers()
  save_fig(p, file.path(OUTROOT, "04_heldout/W_per_group.png"), w = 10, h = 7)
}

# ============================================================================
# 05 — SINGLE CANCER (BLC MFDS)
# ============================================================================
cat("\n=== 05 Single Cancer ===\n")
dir.create(file.path(OUTROOT, "05_single_cancer"), recursive = TRUE, showWarnings = FALSE)

blc <- safe_read(file.path(RDATA, "05_single_cancer/blc_model_comparison.csv"))
if (!is.null(blc)) {
  blc_long <- blc %>%
    select(model, auc, f1, sensitivity, specificity) %>%
    pivot_longer(-model, names_to = "metric", values_to = "value") %>%
    mutate(metric = recode(metric, auc = "AUC", f1 = "F1",
                           sensitivity = "Sensitivity", specificity = "Specificity"))

  p <- ggplot(blc_long, aes(metric, value, fill = model)) +
    geom_col(position = "dodge", width = 0.6, alpha = 0.85) +
    geom_text(aes(label = sprintf("%.3f", value)),
              position = position_dodge(0.6), vjust = -0.3, size = 3.2, family = KFONT) +
    scale_fill_manual(values = c(baseline = "#1d4ed8", film = "#dc2626", xattn = "#059669"),
                      name = "Model") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.12)), limits = c(0, 1.05)) +
    labs(title = "Phase BLC-M — 방광암 단일진단 모델 비교",
         subtitle = "SERS 3ch baseline vs DL(FiLM, CrossAttn), n=699 (BLC 299 + non-cancer 400)",
         x = NULL, y = "Score",
         caption = "SERS baseline만으로 AUC 0.9999, F1 0.993 — 거의 완벽 분류.\nDL 추가 불필요한 것이 방광암의 특징.") +
    theme_sers()
  save_fig(p, file.path(OUTROOT, "05_single_cancer/BLC_MFDS_comparison.png"), w = 10, h = 6)
}

# ============================================================================
# 08 — DL MODELS (training curves)
# ============================================================================
cat("\n=== 08 DL Models ===\n")
dir.create(file.path(OUTROOT, "08_dl_models"), recursive = TRUE, showWarnings = FALSE)

# Collect all epoch metrics into one comparison
dl_files <- list.files(file.path(RDATA, "08_dl_models"), pattern = "epoch_metrics_.*\\.csv", full.names = TRUE)
if (length(dl_files) > 0) {
  all_epochs <- do.call(rbind, lapply(dl_files, function(f) {
    df <- read_csv(f, show_col_types = FALSE)
    df$source <- gsub("epoch_metrics_|_v00[0-9]", "", tools::file_path_sans_ext(basename(f)))
    df
  }))

  # Show val_auc_s1 curves for key models
  key_models <- c("experiment_002_resnet18", "film_xattn_comparison_film_resnet18",
                   "film_xattn_comparison_xattn_resnet18")
  avail <- intersect(key_models, unique(all_epochs$source))

  if (length(avail) > 0 && "val_auc_s1" %in% names(all_epochs)) {
    ep_key <- all_epochs %>%
      filter(source %in% avail, !is.na(val_auc_s1)) %>%
      group_by(source, epoch) %>%
      summarise(mean_auc = mean(val_auc_s1, na.rm = T), .groups = "drop") %>%
      mutate(model = recode(source,
                            experiment_002_resnet18 = "ResNet18 baseline",
                            film_xattn_comparison_film_resnet18 = "FiLM (clinical fusion)",
                            film_xattn_comparison_xattn_resnet18 = "CrossAttn (clinical fusion)"))

    p <- ggplot(ep_key, aes(epoch, mean_auc, color = model)) +
      geom_line(linewidth = 1.1, alpha = 0.85) +
      geom_point(size = 1.5) +
      scale_color_manual(values = c("ResNet18 baseline" = "#dc2626",
                                     "FiLM (clinical fusion)" = "#c2410c",
                                     "CrossAttn (clinical fusion)" = "#4338ca"),
                         name = "Model") +
      scale_y_continuous(labels = number_format(accuracy = 0.01)) +
      labs(title = "DL 모델 학습 곡선 — Val Stage 1 AUC",
           subtitle = "ResNet18 baseline vs FiLM vs CrossAttention (clinical fusion)",
           x = "Epoch", y = "Val AUC (Stage 1)",
           caption = "DL은 빠르게 수렴하지만 val AUC plateau ≈ 0.95~0.97.\nLR baseline의 CV AUC(0.98)를 못 넘김 → 데이터 규모 한계.") +
      theme_sers()
    save_fig(p, file.path(OUTROOT, "08_dl_models/DL_training_curves.png"), w = 10, h = 6)
  }

  # Train vs val loss
  if ("train_loss" %in% names(all_epochs) && length(avail) > 0) {
    loss_df <- all_epochs %>%
      filter(source %in% avail) %>%
      group_by(source, epoch) %>%
      summarise(train = mean(train_loss, na.rm = T), val = mean(val_loss, na.rm = T), .groups = "drop") %>%
      pivot_longer(c(train, val), names_to = "split", values_to = "loss") %>%
      mutate(model = recode(source,
                            experiment_002_resnet18 = "ResNet18",
                            film_xattn_comparison_film_resnet18 = "FiLM",
                            film_xattn_comparison_xattn_resnet18 = "CrossAttn"))

    p <- ggplot(loss_df, aes(epoch, loss, color = model, linetype = split)) +
      geom_line(linewidth = 0.9) +
      scale_linetype_manual(values = c(train = "solid", val = "dashed"), name = "Split") +
      scale_color_manual(values = c(ResNet18 = "#dc2626", FiLM = "#c2410c", CrossAttn = "#4338ca"),
                         name = "Model") +
      labs(title = "DL 모델 학습/검증 Loss 비교",
           subtitle = "실선 = train, 점선 = validation — gap이 클수록 과적합",
           x = "Epoch", y = "Loss",
           caption = "모든 DL에서 train loss↓ val loss→plateau 패턴 = 과적합 경향.") +
      theme_sers()
    save_fig(p, file.path(OUTROOT, "08_dl_models/DL_loss_curves.png"), w = 10, h = 6)
  }
}

# ============================================================================
# 10 — Z-OPT OPTIMIZATION PHASES
# ============================================================================
cat("\n=== 10 Z-opt ===\n")
dir.create(file.path(OUTROOT, "10_optimization"), recursive = TRUE, showWarnings = FALSE)

# Wavenumber ablation
wn <- safe_read(file.path(RDATA, "10_optimization/phase1_wavenumber_ablation.csv"))
if (!is.null(wn)) {
  p <- ggplot(wn, aes(reorder(config, s2_f1), s2_f1, fill = s2_f1)) +
    geom_col(width = 0.6) +
    geom_text(aes(label = sprintf("%.3f", s2_f1)), hjust = -0.15, size = 3.5, family = KFONT) +
    coord_flip() +
    scale_fill_viridis_c(option = "mako", direction = -1, end = 0.85, guide = "none") +
    scale_x_discrete() +
    labs(title = "Z-opt Phase 1 — 파장 범위 × Feature Transform",
         subtitle = "Full range(402~2198)이 fingerprint보다 일관되게 높음",
         x = NULL, y = "Type ID F1 macro",
         caption = "1st derivative(d1)가 raw보다 F1 +6.6pp.\nfull_range + d1 = 최적 조합.") +
    theme_sers()
  save_fig(p, file.path(OUTROOT, "10_optimization/zopt_wavenumber.png"), w = 10, h = 7)
}

# Aggregation comparison
agg <- safe_read(file.path(RDATA, "10_optimization/phase2_aggregation_ablation.csv"))
if (!is.null(agg)) {
  config_col <- intersect(names(agg), c("config", "aggregate"))[1]
  agg_long <- agg %>%
    select(all_of(config_col), s1_auc, s2_f1) %>%
    pivot_longer(-all_of(config_col), names_to = "metric", values_to = "value") %>%
    mutate(metric = recode(metric, s1_auc = "Det AUC", s2_f1 = "Type ID F1"))

  p <- ggplot(agg_long, aes(.data[[config_col]], value, fill = metric)) +
    geom_col(position = "dodge", width = 0.5, alpha = 0.85) +
    geom_text(aes(label = sprintf("%.3f", value)),
              position = position_dodge(0.5), vjust = -0.3, size = 3.8, family = KFONT) +
    scale_fill_manual(values = c("Det AUC" = "#0d9488", "Type ID F1" = "#6d28d9"), name = "") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(title = "Z-opt Phase 2 — Aggregation 방법 비교",
         subtitle = "mean >> medoid (F1 +14pp) — 핵심 발견",
         x = "Aggregation method", y = "Score",
         caption = "Mean: 환자당 5 replicate 평균. 노이즈 감소 효과가 극적.\nMedoid: 대표 spectrum 1개 선택 → 정보 손실.") +
    theme_sers()
  save_fig(p, file.path(OUTROOT, "10_optimization/zopt_aggregation.png"), w = 9, h = 6)
}

# Architecture HP search
arch <- safe_read(file.path(RDATA, "10_optimization/phase3_architecture_hp_search.csv"))
if (!is.null(arch)) {
  f1_col <- intersect(names(arch), c("s2_f1", "ens_f1"))[1]
  if (!is.na(f1_col)) {
    arch <- arch %>% arrange(desc(.data[[f1_col]])) %>% slice_head(n = 15)
    arch$config_f <- factor(arch$config, levels = rev(arch$config))
    p <- ggplot(arch, aes(.data[[f1_col]], config_f, fill = .data[[f1_col]])) +
      geom_col(width = 0.6) +
      geom_text(aes(label = sprintf("%.3f", .data[[f1_col]])), hjust = -0.15, size = 3.3, family = KFONT) +
      scale_fill_viridis_c(option = "mako", direction = -1, end = 0.85, guide = "none") +
      scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
      labs(title = "Z-opt Phase 3 — Architecture HP Search Top-15",
           subtitle = "lr / dropout / channel width sweep",
           x = "Type ID F1 macro", y = NULL,
           caption = "lr=3e-4, dropout=0.3, channels=64-512이 최적.") +
      theme_sers()
    save_fig(p, file.path(OUTROOT, "10_optimization/zopt_architecture.png"), w = 10, h = 7)
  }
}

# ============================================================================
# 11 — INTERPRETATION (spectral)
# ============================================================================
cat("\n=== 11 Interpretation ===\n")
dir.create(file.path(OUTROOT, "11_interpretation"), recursive = TRUE, showWarnings = FALSE)

# Mean spectra overlay (sampled wavenumbers for performance)
ms <- safe_read(file.path(RDATA, "17_spectral_interpretation/mean_spectra_overlay.csv"))
if (!is.null(ms)) {
  # Subsample every 5th wavenumber for plotting speed
  ms_sub <- ms %>% filter(wavenumber %% 10 < 2)
  p <- ggplot(ms_sub, aes(wavenumber, mean_intensity, color = diagnosis)) +
    geom_line(linewidth = 0.5, alpha = 0.75) +
    scale_color_manual(values = PAL_CANCER, name = "그룹") +
    labs(title = "암종별 평균 SERS 스펙트럼 overlay",
         subtitle = "각 그룹의 mean spectrum (n=1,628, Thermo, SNV 정규화)",
         x = "Wavenumber (cm-1)", y = "Intensity (SNV)",
         caption = "전체적 형태는 유사하나 600~700, 1500~1600 cm-1 영역에서 그룹 간 차이 관찰.\n이 미세한 차이를 LR이 학습하여 분류에 활용.") +
    theme_sers() + theme(legend.position = "top")
  save_fig(p, file.path(OUTROOT, "11_interpretation/SI_mean_spectra_overlay.png"), w = 12, h = 6)
}

# Peak discrimination heatmap
pd_matrix <- safe_read(file.path(RDATA, "11_interpretation/peak_discrimination_matrix.csv"))
if (!is.null(pd_matrix)) {
  pk_col <- names(pd_matrix)[1]
  pd_long <- pd_matrix %>%
    rename(peak = all_of(pk_col)) %>%
    pivot_longer(-peak, names_to = "cancer", values_to = "discrimination")

  p <- ggplot(pd_long, aes(cancer, peak, fill = discrimination)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(aes(label = ifelse(abs(discrimination) > 0.1, sprintf("%.2f", discrimination), "")),
              size = 2.8, family = KFONT) +
    scale_fill_gradient2(low = "#2563eb", mid = "white", high = "#dc2626",
                          midpoint = 0, name = "Discrimination\nscore") +
    labs(title = "암종별 Peak discrimination matrix",
         subtitle = "각 셀 = 해당 peak의 해당 암종 분류 기여도",
         x = "Cancer type", y = "Peak",
         caption = "양수(빨강) = 해당 암종 양성 방향, 음수(파랑) = 음성 방향.\n암종마다 반응하는 peak 조합이 다름 → multiclass 분류의 근거.") +
    theme_sers() + theme(axis.text.y = element_text(size = 8))
  save_fig(p, file.path(OUTROOT, "11_interpretation/SI_peak_discrimination.png"), w = 10, h = 9)
}

# Peak intensity by diagnosis (dot+box)
pid <- safe_read(file.path(RDATA, "17_spectral_interpretation/peak_intensity_by_diagnosis.csv"))
if (!is.null(pid)) {
  # Already long format: diagnosis, feature, peak_intensity
  top_peaks <- pid %>%
    group_by(feature) %>%
    summarise(var = var(peak_intensity, na.rm = T), .groups = "drop") %>%
    arrange(desc(var)) %>% slice_head(n = 8) %>% pull(feature)
  pid_top <- pid %>% filter(feature %in% top_peaks)

  p <- ggplot(pid_top, aes(diagnosis, peak_intensity, fill = diagnosis)) +
    geom_col(stat = "identity", width = 0.6, alpha = 0.8) +
    facet_wrap(~ feature, scales = "free_y", ncol = 4) +
    scale_fill_manual(values = PAL_CANCER, guide = "none") +
    labs(title = "Top-8 Peak별 암종 간 intensity 비교",
         subtitle = "가장 변동이 큰 8개 peak — 암종별 특이 패턴 확인",
         x = NULL, y = "Peak intensity",
         caption = "같은 peak이라도 암종에 따라 intensity 분포가 다름 → SERS 분류의 생물학적 근거.") +
    theme_sers(10) + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7))
  save_fig(p, file.path(OUTROOT, "11_interpretation/SI_peak_intensity_boxplot.png"), w = 14, h = 10)
}

cat("\n[done] extended figures complete.\n")
cat("output:", OUTROOT, "\n")
