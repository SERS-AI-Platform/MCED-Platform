#!/usr/bin/env Rscript
# ============================================================================
# Phase Archive REMAINING — all figures that can be drawn from existing CSVs
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
safe_read <- function(path) { if (file.exists(path)) read_csv(path, show_col_types = FALSE) else NULL }

# ============================================================================
# 01b — Overfit gap (Phase F)
# ============================================================================
cat("\n=== 01b Overfit gap ===\n")
dir.create(file.path(OUTROOT, "01_baseline"), recursive = TRUE, showWarnings = FALSE)
gap <- safe_read(file.path(RDATA, "01_baseline/benchmark_train_val_gaps.csv"))
if (!is.null(gap)) {
  gap_cols <- names(gap)
  train_col <- grep("train.*auc|train_auc", gap_cols, value = TRUE, ignore.case = TRUE)[1]
  val_col <- grep("val.*auc|val_auc|^auc$", gap_cols, value = TRUE, ignore.case = TRUE)[1]
  if (!is.na(train_col) && !is.na(val_col)) {
    gap$gap_val <- gap[[train_col]] - gap[[val_col]]
  } else {
    # Try named columns
    gap$gap_val <- gap$train_s1_auc - gap$val_s1_auc
  }
  gap <- gap %>% filter(!is.na(gap_val))
  if (nrow(gap) > 0) {
    gap$model_f <- factor(gap$model, levels = gap$model[order(gap$gap_val)])
    p <- ggplot(gap, aes(model_f, gap_val, fill = gap_val)) +
      geom_col(width = 0.6) +
      geom_text(aes(label = sprintf("%.3f", gap_val)), hjust = -0.2, size = 3.8, family = KFONT) +
      coord_flip() +
      scale_fill_viridis_c(option = "rocket", direction = -1, end = 0.85, guide = "none") +
      scale_y_continuous(expand = expansion(mult = c(0, 0.25))) +
      labs(title = "Phase F — Train-Val 과적합 갭 (AUC)",
           subtitle = "DL(ResNet18)일수록 갭이 큼 → 과적합 경향",
           x = NULL, y = "AUC Gap (Train - Val)",
           caption = "LR은 갭 < 0.02, DL은 > 0.10.\n데이터 규모(1,240명)에서 DL은 과적합 리스크 높음.") +
      theme_sers()
    save_fig(p, file.path(OUTROOT, "01_baseline/F_overfit_gap.png"), w = 9, h = 5)
  }
}

# ============================================================================
# 03b — Confounding (Phase M) — feature_representation has model comparison
# ============================================================================
cat("\n=== 03b Feature representation detail ===\n")
dir.create(file.path(OUTROOT, "03_clinical_fusion"), recursive = TRUE, showWarnings = FALSE)
fr <- safe_read(file.path(RDATA, "03_clinical_fusion/fs_feature_representation.csv"))
if (!is.null(fr)) {
  # Parse "0.9852±0.0097" format
  parse_pm <- function(x) {
    parts <- strsplit(as.character(x), "±")
    sapply(parts, function(p) as.numeric(p[1]))
  }
  parse_sd <- function(x) {
    parts <- strsplit(as.character(x), "±")
    sapply(parts, function(p) if(length(p)>1) as.numeric(p[2]) else 0)
  }

  if ("Type_F1" %in% names(fr)) {
    fr$f1_mean <- parse_pm(fr$Type_F1)
    fr$f1_std <- parse_sd(fr$Type_F1)
  } else if ("Type_F1_mean" %in% names(fr)) {
    fr$f1_mean <- fr$Type_F1_mean
    fr$f1_std <- fr$Type_F1_std
  }

  if ("f1_mean" %in% names(fr) && "Model" %in% names(fr)) {
    p <- ggplot(fr, aes(reorder(Feature, f1_mean), f1_mean, fill = Model)) +
      geom_col(position = "dodge", width = 0.6, alpha = 0.85) +
      geom_errorbar(aes(ymin = f1_mean - f1_std, ymax = f1_mean + f1_std),
                    width = 0.2, position = position_dodge(0.6)) +
      coord_flip() +
      scale_fill_viridis_d(option = "turbo", end = 0.85) +
      labs(title = "Phase FR — Feature Representation 비교 (LR / RF / XGB)",
           subtitle = "full_spectrum / 1st_derivative / 2nd_derivative / peak_only / concat",
           x = NULL, y = "Type ID F1 macro",
           caption = "1st derivative > full spectrum > peak_only.\nLR이 전 feature에서 RF/XGB 대비 우위.") +
      theme_sers()
    save_fig(p, file.path(OUTROOT, "03_clinical_fusion/FR_feature_repr.png"), w = 10, h = 7)
  }
}

# ============================================================================
# 07 — Multiview derivative channel stats
# ============================================================================
cat("\n=== 07 Multiview derivative ===\n")
dir.create(file.path(OUTROOT, "07_multiview"), recursive = TRUE, showWarnings = FALSE)

mv_ens <- safe_read(file.path(RDATA, "07_multiview/multiview_ensemble.csv"))
if (!is.null(mv_ens)) {
  f1_col <- intersect(names(mv_ens), c("s2_f1", "id_f1_macro", "f1"))[1]
  cfg_col <- intersect(names(mv_ens), c("config", "name", "model"))[1]
  if (!is.na(f1_col) && !is.na(cfg_col)) {
    mv_ens <- mv_ens %>% arrange(desc(.data[[f1_col]])) %>% slice_head(n = 15)
    mv_ens$cfg_f <- factor(mv_ens[[cfg_col]], levels = rev(mv_ens[[cfg_col]]))
    p <- ggplot(mv_ens, aes(.data[[f1_col]], cfg_f, fill = .data[[f1_col]])) +
      geom_col(width = 0.6) +
      geom_text(aes(label = sprintf("%.3f", .data[[f1_col]])), hjust = -0.1, size = 3.3, family = KFONT) +
      scale_fill_viridis_c(option = "mako", direction = -1, end = 0.85, guide = "none") +
      scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
      labs(title = "Phase MV — Multi-view ensemble 결과",
           subtitle = "raw + d1 + d2 조합 방식별 F1 비교",
           x = "Type ID F1", y = NULL,
           caption = "raw+d1+d2 concat이 raw 단독 대비 F1 +4.6pp 향상.") +
      theme_sers()
    save_fig(p, file.path(OUTROOT, "07_multiview/MV_ensemble.png"), w = 10, h = 7)
  }
}

# Channel derivative stats — skip (complex nested structure)

# ============================================================================
# 10b — Z-opt held-out repeats + final eval
# ============================================================================
cat("\n=== 10b Z-opt held-out ===\n")
dir.create(file.path(OUTROOT, "10_optimization"), recursive = TRUE, showWarnings = FALSE)

zho <- safe_read(file.path(RDATA, "10_optimization/phase4_held_out_repeats.csv"))
if (!is.null(zho)) {
  f1_col <- intersect(names(zho), c("ens_s2_f1", "s2_f1", "lr_s2_f1"))[1]
  auc_col <- intersect(names(zho), c("ens_s1_auc", "s1_auc", "lr_s1_auc"))[1]
  if (!is.na(f1_col) && !is.na(auc_col)) {
    zho_long <- zho %>%
      mutate(rep = row_number()) %>%
      select(rep, all_of(c(auc_col, f1_col))) %>%
      pivot_longer(-rep, names_to = "metric", values_to = "value") %>%
      mutate(metric = ifelse(grepl("auc", metric), "Det AUC", "Type ID F1"))

    p <- ggplot(zho_long, aes(factor(rep), value, fill = metric)) +
      geom_col(position = "dodge", width = 0.6, alpha = 0.85) +
      geom_text(aes(label = sprintf("%.3f", value)),
                position = position_dodge(0.6), vjust = -0.3, size = 3.2, family = KFONT) +
      scale_fill_manual(values = c("Det AUC" = "#0d9488", "Type ID F1" = "#6d28d9"), name = "") +
      scale_y_continuous(expand = expansion(mult = c(0, 0.12)), limits = c(0, 1.05)) +
      labs(title = "Z-opt Phase 4 — Held-out robustness (5 repeats)",
           subtitle = "60/20/20 random split × 5, 안정성 확인",
           x = "Repeat", y = "Score",
           caption = "repeat별 AUC·F1 편차가 작음 → 결과 안정적.") +
      theme_sers()
    save_fig(p, file.path(OUTROOT, "10_optimization/zopt_held_out.png"), w = 10, h = 6)
  }
}

# ============================================================================
# 11b — Spectral interpretation: difference spectra + coefficient heatmap
# ============================================================================
cat("\n=== 11b Spectral interpretation extended ===\n")
dir.create(file.path(OUTROOT, "11_interpretation"), recursive = TRUE, showWarnings = FALSE)

# Difference spectra (all_peak_differences.csv)
pdiff <- safe_read(file.path(RDATA, "17_spectral_interpretation/all_peak_differences.csv"))
if (!is.null(pdiff)) {
  # Top differences by absolute value
  top_diff <- pdiff %>%
    group_by(diagnosis, comparison) %>%
    arrange(desc(abs_difference)) %>%
    slice_head(n = 5) %>% ungroup() %>%
    slice_head(n = 40)

  p <- ggplot(top_diff, aes(wavenumber, difference, color = diagnosis)) +
    geom_segment(aes(xend = wavenumber, y = 0, yend = difference), alpha = 0.7) +
    geom_point(size = 1.5) +
    facet_wrap(~ diagnosis, scales = "free_y", ncol = 2) +
    scale_color_manual(values = PAL_CANCER, guide = "none") +
    labs(title = "암종별 Top peak difference (target - reference)",
         subtitle = "각 암종에서 confused class 대비 가장 차이가 큰 wavenumber",
         x = "Wavenumber (cm-1)", y = "Intensity difference",
         caption = "양수 = 해당 암종이 더 높은 peak, 음수 = 더 낮은 peak.") +
    theme_sers()
  save_fig(p, file.path(OUTROOT, "11_interpretation/SI_difference_spectra.png"), w = 11, h = 8)
}

# All mean spectra per group (raw individual spectra view)
ams <- safe_read(file.path(RDATA, "17_spectral_interpretation/all_mean_spectra.csv"))
if (!is.null(ams)) {
  # Subsample for plot speed
  ams_sub <- ams %>% filter(wavenumber %% 8 < 2)
  p <- ggplot(ams_sub, aes(wavenumber, mean_intensity, color = diagnosis)) +
    geom_line(linewidth = 0.4, alpha = 0.7) +
    facet_wrap(~ diagnosis, scales = "free_y", ncol = 3) +
    scale_color_manual(values = PAL_CANCER, guide = "none") +
    labs(title = "암종별 평균 스펙트럼 (faceted)",
         subtitle = "각 그룹의 mean ± shade 없이 개별 패널로",
         x = "Wavenumber (cm-1)", y = "Mean intensity") +
    theme_sers(10) + theme(strip.text = element_text(size = 10))
  save_fig(p, file.path(OUTROOT, "11_interpretation/SI_mean_spectra_faceted.png"), w = 12, h = 8)
}

# Coefficient-peak crossref
cpx <- safe_read(file.path(RDATA, "11_interpretation/coefficient_peak_crossref.csv"))
if (!is.null(cpx)) {
  wn_col <- intersect(names(cpx), c("wavenumber", "peak_region", "peak"))[1]
  val_cols <- intersect(names(cpx), c("PRO","LUN","CRC","PAN","OVA","BRE","BLC",
                                       "s1_coef","mean_abs_coef"))
  if (!is.na(wn_col) && length(val_cols) > 0) {
    cpx_long <- cpx %>%
      select(all_of(c(wn_col, val_cols))) %>%
      pivot_longer(-all_of(wn_col), names_to = "target", values_to = "coef")
    names(cpx_long)[1] <- "peak"

    p <- ggplot(cpx_long, aes(target, peak, fill = coef)) +
      geom_tile(color = "white", linewidth = 0.5) +
      scale_fill_gradient2(low = "#2563eb", mid = "white", high = "#dc2626",
                           midpoint = 0, name = "Coefficient") +
      labs(title = "LR Coefficient-Peak 교차 참조",
           subtitle = "Stage 2 per-cancer LR coefficient의 peak 영역별 값",
           x = "Target", y = "Peak region") +
      theme_sers() + theme(axis.text.y = element_text(size = 8))
    save_fig(p, file.path(OUTROOT, "11_interpretation/SI_coefficient_crossref.png"), w = 10, h = 9)
  }
}

# ============================================================================
# 12b — NFS heatmap (norm × feature × model grid)
# ============================================================================
cat("\n=== 12b NFS heatmap ===\n")
dir.create(file.path(OUTROOT, "09_nfs"), recursive = TRUE, showWarnings = FALSE)
nfs <- safe_read(file.path(RDATA, "09_nfs/nfs_results.csv"))
if (!is.null(nfs)) {
  nfs_lr <- nfs %>% filter(model == "LR")
  if (nrow(nfs_lr) > 1) {
    p <- ggplot(nfs_lr, aes(feature, norm, fill = mean_f1)) +
      geom_tile(color = "white", linewidth = 1) +
      geom_text(aes(label = sprintf("%.3f", mean_f1)), size = 3.5, family = KFONT, fontface = "bold") +
      scale_fill_viridis_c(option = "mako", direction = -1, end = 0.9, name = "F1") +
      labs(title = "Phase NFS — Norm × Feature heatmap (LR only)",
           subtitle = "F1 = 20-seed 평균, norm = 정규화 방법, feature = derivative view",
           x = "Feature transform", y = "Normalization",
           caption = "l2 / d1이 F1 0.917로 1위.\nSNV가 default이나 l2도 경쟁력 있음.") +
      theme_sers() + coord_equal()
    save_fig(p, file.path(OUTROOT, "09_nfs/NFS_heatmap.png"), w = 9, h = 6)
  }
}

# ============================================================================
# Contrastive pretrain loss (if epoch data exists)
# ============================================================================
cat("\n=== Contrastive pretrain ===\n")
con_ep <- safe_read(file.path(RDATA, "08_dl_models/contrastive_methods.csv"))
if (!is.null(con_ep) && nrow(con_ep) > 0) {
  f1c <- intersect(names(con_ep), c("finetune_f1", "f1_type", "f1"))[1]
  p <- ggplot(con_ep, aes(reorder(method, -.data[[f1c]]), .data[[f1c]], fill = .data[[f1c]])) +
    geom_col(width = 0.6) +
    geom_text(aes(label = sprintf("%.3f", .data[[f1c]])), vjust = -0.3, size = 3.5, family = KFONT) +
    scale_fill_viridis_c(option = "mako", direction = -1, end = 0.85, guide = "none") +
    labs(title = "Contrastive pretrain — 방법별 finetune F1",
         x = "Method", y = "Finetune F1") +
    theme_sers()
  save_fig(p, file.path(OUTROOT, "06_contrastive/CON_methods.png"), w = 9, h = 6)
}

cat("\n[done] remaining figures complete.\n")
