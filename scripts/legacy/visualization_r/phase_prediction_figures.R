#!/usr/bin/env Rscript
# ============================================================================
# Prediction-based figures: ROC, Confusion, Probability dist, LR coefficients
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
PRED    <- file.path(ROOT, "results/r_export/predictions")
OUTROOT <- file.path(ROOT, "results/figures/phase_archive_r")
safe_read <- function(path) { if (file.exists(path)) read_csv(path, show_col_types = FALSE) else NULL }

phases <- list(
  list(dir = "phase_F_baseline", label = "Phase F — 5-cancer Baseline", short = "F"),
  list(dir = "phase_T_8class", label = "Phase T — 8-class (BLC 추가)", short = "T"),
  list(dir = "phase_U_7class", label = "Phase U — 7-class (BRE 재추가)", short = "U"),
  list(dir = "phase_X_clean", label = "Phase X — Clean cohort", short = "X")
)

for (ph in phases) {
  ph_dir <- file.path(PRED, ph$dir)
  out_dir <- file.path(OUTROOT, "predictions")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  cat(sprintf("\n=== %s ===\n", ph$label))

  # --- ROC Stage 1 ---
  roc1 <- safe_read(file.path(ph_dir, "roc_stage1.csv"))
  if (!is.null(roc1)) {
    auc_val <- round(max(cumsum(diff(roc1$fpr) * (roc1$tpr[-1] + roc1$tpr[-nrow(roc1)])/2), na.rm=T), 4)
    # Approximate AUC from trapezoid
    p <- ggplot(roc1, aes(fpr, tpr)) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#94a3b8") +
      geom_line(color = "#1d4ed8", linewidth = 1.2) +
      annotate("text", x = 0.55, y = 0.15, label = sprintf("AUC = %.4f", auc_val),
               family = KFONT, size = 5, color = "#1e3a8a", fontface = "bold") +
      coord_equal() +
      labs(title = sprintf("%s — Stage 1 ROC Curve", ph$label),
           subtitle = "Cancer vs Non-cancer (5-fold CV 전체 OOF)",
           x = "1 - Specificity (FPR)", y = "Sensitivity (TPR)") +
      theme_sers()
    save_fig(p, file.path(out_dir, sprintf("%s_roc_stage1.png", ph$short)), w = 7, h = 7)
  }

  # --- Confusion Stage 2 ---
  cm2 <- safe_read(file.path(ph_dir, "confusion_stage2.csv"))
  if (!is.null(cm2)) {
    labels <- names(cm2)[-1]
    cm2_long <- cm2 %>%
      rename(True = 1) %>%
      pivot_longer(-True, names_to = "Predicted", values_to = "Count") %>%
      mutate(True = factor(True, levels = rev(labels)),
             Predicted = factor(Predicted, levels = labels))

    # Normalize per row for fill
    cm2_long <- cm2_long %>%
      group_by(True) %>%
      mutate(pct = Count / sum(Count)) %>%
      ungroup()

    p <- ggplot(cm2_long, aes(Predicted, True, fill = pct)) +
      geom_tile(color = "white", linewidth = 1.2) +
      geom_text(aes(label = Count), family = KFONT, size = 3.8, fontface = "bold",
                color = ifelse(cm2_long$pct > 0.5, "white", "#1e293b")) +
      scale_fill_viridis_c(option = "mako", direction = -1, end = 0.9,
                           name = "Rate", labels = percent_format()) +
      labs(title = sprintf("%s — Stage 2 Confusion Matrix", ph$label),
           subtitle = "7-class cancer type classification (OOF predictions)",
           x = "Predicted", y = "True") +
      theme_sers() + coord_equal()
    save_fig(p, file.path(out_dir, sprintf("%s_confusion_stage2.png", ph$short)), w = 9, h = 8)
  }

  # --- Stage 1 Probability Distribution ---
  prob <- safe_read(file.path(ph_dir, "stage1_prob_distribution.csv"))
  if (!is.null(prob)) {
    prob$label <- ifelse(prob$binary_label == 1, "Cancer", "Non-cancer")
    p <- ggplot(prob, aes(pred_prob_cancer, fill = label, color = label)) +
      geom_density(alpha = 0.4, linewidth = 0.8) +
      geom_vline(xintercept = 0.5, linetype = "dashed", color = "#94a3b8") +
      scale_fill_manual(values = c(Cancer = "#dc2626", `Non-cancer` = "#2563eb"), name = "") +
      scale_color_manual(values = c(Cancer = "#dc2626", `Non-cancer` = "#2563eb"), name = "") +
      labs(title = sprintf("%s — Stage 1 예측 확률 분포", ph$label),
           subtitle = "Cancer(빨강) vs Non-cancer(파랑) 밀도 분포, 점선 = threshold 0.5",
           x = "P(Cancer)", y = "Density",
           caption = "두 분포가 잘 분리될수록 좋음. 겹치는 영역 = 불확실 구간.") +
      theme_sers() + theme(legend.position = "top")
    save_fig(p, file.path(out_dir, sprintf("%s_prob_dist.png", ph$short)), w = 9, h = 5.5)
  }

  # --- Per-class metrics ---
  pcm <- safe_read(file.path(ph_dir, "per_class_metrics.csv"))
  if (!is.null(pcm)) {
    name_col <- intersect(names(pcm), c("class_name", "cancer", "class"))[1]
    if (is.na(name_col)) name_col <- names(pcm)[1]
    pcm_long <- pcm %>%
      select(all_of(name_col), any_of(c("f1", "sensitivity", "specificity", "precision"))) %>%
      pivot_longer(-all_of(name_col), names_to = "metric", values_to = "value")
    names(pcm_long)[1] <- "cancer"

    p <- ggplot(pcm_long, aes(metric, cancer, fill = value)) +
      geom_tile(color = "white", linewidth = 1) +
      geom_text(aes(label = sprintf("%.2f", value)), family = KFONT, size = 3.5) +
      scale_fill_viridis_c(option = "mako", direction = -1, end = 0.9, name = "Score") +
      labs(title = sprintf("%s — Per-cancer type 성능", ph$label),
           subtitle = "F1 / Sensitivity / Specificity / Precision (OOF)",
           x = NULL, y = NULL) +
      theme_sers() + coord_equal()
    save_fig(p, file.path(out_dir, sprintf("%s_per_class.png", ph$short)), w = 9, h = 7)
  }
}

# ============================================================================
# LR Coefficient overlay (Stage 1 binary)
# ============================================================================
cat("\n=== LR Coefficients ===\n")
dir.create(file.path(OUTROOT, "lr_coefficients"), recursive = TRUE, showWarnings = FALSE)

# Load common grid
grid_csv <- safe_read(file.path(ROOT, "results/preprocessing/common_grid.csv"))
grid_vec <- NULL
if (!is.null(grid_csv)) grid_vec <- grid_csv[[1]]

for (ph in phases) {
  coef_f <- file.path(PRED, sprintf("lr_coefficients/%s_binary_coefs.csv", ph$dir))
  coef <- safe_read(coef_f)
  if (!is.null(coef)) {
    if (!is.null(grid_vec) && nrow(coef) == length(grid_vec)) {
      coef$wavenumber <- grid_vec
    } else {
      coef$wavenumber <- seq(402, by = (2198-402)/(nrow(coef)-1), length.out = nrow(coef))
    }
    p <- ggplot(coef, aes(wavenumber, coef_mean)) +
      geom_ribbon(aes(ymin = coef_mean - coef_std, ymax = coef_mean + coef_std),
                  fill = "#6366f1", alpha = 0.2) +
      geom_line(color = "#4338ca", linewidth = 0.5) +
      geom_hline(yintercept = 0, color = "#94a3b8", linewidth = 0.3) +
      labs(title = sprintf("%s — LR Stage 1 coefficient spectrum", ph$label),
           subtitle = "mean ± std across 5 folds, 양수 = cancer 방향, 음수 = non-cancer 방향",
           x = "Wavenumber (cm-1)", y = "Coefficient",
           caption = "ribbon = fold 간 편차. 좁을수록 안정적인 특성.") +
      theme_sers()
    save_fig(p, file.path(OUTROOT, "lr_coefficients",
                          sprintf("%s_binary_coef.png", ph$short)), w = 12, h = 5)
  }

  # Stage 2 per-cancer coefficients
  coef2_f <- file.path(PRED, sprintf("lr_coefficients/%s_stage2_coefs.csv", ph$dir))
  coef2 <- safe_read(coef2_f)
  if (!is.null(coef2)) {
    # Columns: feature_idx, class_0_mean, class_0_std, class_1_mean, ...
    class_cols <- grep("class_.*_mean$", names(coef2), value = TRUE)
    if (length(class_cols) > 0) {
      if (!is.null(grid_vec) && nrow(coef2) == length(grid_vec)) {
        coef2$wavenumber <- grid_vec
      } else {
        coef2$wavenumber <- seq(402, by = (2198-402)/(nrow(coef2)-1), length.out = nrow(coef2))
      }
      coef2_long <- coef2 %>%
        select(wavenumber, all_of(class_cols)) %>%
        pivot_longer(-wavenumber, names_to = "class", values_to = "coef") %>%
        mutate(class = gsub("class_|_mean", "", class))

      p <- ggplot(coef2_long, aes(wavenumber, coef, color = class)) +
        geom_line(linewidth = 0.4, alpha = 0.7) +
        scale_color_viridis_d(option = "turbo", end = 0.92, name = "Cancer class") +
        geom_hline(yintercept = 0, color = "#94a3b8", linewidth = 0.3) +
        labs(title = sprintf("%s — LR Stage 2 per-cancer coefficient spectra", ph$label),
             subtitle = "각 암종의 LR coefficient를 wavenumber 축에 overlay — 암종마다 반응 peak 다름",
             x = "Wavenumber (cm-1)", y = "Coefficient") +
        theme_sers() + theme(legend.position = "top")
      save_fig(p, file.path(OUTROOT, "lr_coefficients",
                            sprintf("%s_stage2_coef.png", ph$short)), w = 12, h = 5.5)
    }
  }
}

# ============================================================================
# Stacking predictions — ROC for base models
# ============================================================================
cat("\n=== Stacking ROC ===\n")
stk_roc <- safe_read(file.path(PRED, "stacking_v2/stacking_roc_base_models.csv"))
if (!is.null(stk_roc)) {
  mcol <- intersect(names(stk_roc), c("base_model", "model"))[1]
  p <- ggplot(stk_roc, aes(fpr, tpr, color = .data[[mcol]])) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#94a3b8") +
    geom_line(linewidth = 0.7, alpha = 0.8) +
    scale_color_viridis_d(option = "turbo", end = 0.92, name = "Base model") +
    coord_equal() +
    labs(title = "STK-V2 — Base model별 ROC (outer-fold OOF)",
         subtitle = "10개 base 각각의 Stage 1 ROC, 동일 1,628명",
         x = "1 - Specificity (FPR)", y = "Sensitivity (TPR)",
         caption = "lr_d2/lr_concat/ridge_concat가 AUC 0.99+ 영역.\nRF는 AUC 0.95~0.97 — meta에 보완 기여만.") +
    theme_sers()
  save_fig(p, file.path(OUTROOT, "predictions/STK_V2_base_roc.png"), w = 9, h = 8)
}

# Stacking confusion from stage2 predictions
stk_s2 <- safe_read(file.path(PRED, "stacking_v2/stacking_stage2_predictions.csv"))
if (!is.null(stk_s2)) {
  # Build confusion matrix
  if ("true_label" %in% names(stk_s2) && "pred_label" %in% names(stk_s2)) {
    cm <- table(True = stk_s2$true_label, Predicted = stk_s2$pred_label)
    cm_df <- as.data.frame(cm) %>%
      mutate(True = factor(True, levels = rev(levels(True))))
    cm_df <- cm_df %>%
      group_by(True) %>% mutate(pct = Freq / sum(Freq)) %>% ungroup()

    p <- ggplot(cm_df, aes(Predicted, True, fill = pct)) +
      geom_tile(color = "white", linewidth = 1.2) +
      geom_text(aes(label = Freq), family = KFONT, size = 3.5, fontface = "bold",
                color = ifelse(cm_df$pct > 0.5, "white", "#1e293b")) +
      scale_fill_viridis_c(option = "mako", direction = -1, end = 0.9,
                           name = "Rate", labels = percent_format()) +
      labs(title = "STK-V2 — Stage 2 Confusion Matrix (ElasticNet meta)",
           subtitle = "7-class cancer type, nested CV outer-fold OOF",
           x = "Predicted cancer type", y = "True cancer type") +
      theme_sers() + coord_equal()
    save_fig(p, file.path(OUTROOT, "predictions/STK_V2_confusion.png"), w = 9, h = 8)
  }
}

cat("\n[done] prediction figures complete.\n")
cat(sprintf("total new: %d\n", length(list.files(file.path(OUTROOT, "predictions"), "\\.png$")) +
                                 length(list.files(file.path(OUTROOT, "lr_coefficients"), "\\.png$"))))
