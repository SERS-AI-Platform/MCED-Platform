#!/usr/bin/env Rscript
# ============================================================================
# SERS-AI — Shared R theme & reusable plotting functions
# ============================================================================
# All Phase Archive figures share this theme for visual consistency.
# Usage: source("theme_sers.R") at the top of each category script.
# ============================================================================

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr)
  library(scales); library(showtext); library(sysfonts)
  library(viridis); library(patchwork)
})

# ── Korean font ──
.setup_font <- function() {
  candidates <- c(
    "/home/user/.fonts/NotoSansKR.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
  )
  for (p in candidates) {
    if (file.exists(p)) {
      sysfonts::font_add("KR", regular = p)
      return("KR")
    }
  }
  "sans"
}
KFONT <- .setup_font()
showtext_auto(); showtext_opts(dpi = 300)

# ── Palette ──
PAL_CANCER <- c(
  PRO = "#e63946", LUN = "#457b9d", CRC = "#2a9d8f", PAN = "#e9c46a",
  OVA = "#f4a261", BRE = "#264653", BLC = "#6d597a",
  NOR = "#94a3b8", DIA = "#cbd5e1", HBP = "#a8a29e", `H.D.` = "#78716c"
)
PAL_MODEL <- c(
  LR = "#1d4ed8", XGBoost = "#059669", RF = "#7c3aed", Ridge = "#0891b2",
  ResNet18 = "#dc2626", CNN = "#ea580c", SVM = "#db2777",
  Ensemble = "#be185d", FiLM = "#c2410c", CrossAttn = "#4338ca",
  Contrastive = "#0f766e", Transformer = "#7e22ce"
)

# ── Theme ──
theme_sers <- function(base_size = 12) {
  theme_minimal(base_size = base_size, base_family = KFONT) +
    theme(
      plot.title       = element_text(face = "bold", size = base_size + 3, color = "#1e3a8a"),
      plot.subtitle    = element_text(color = "#475569", size = base_size, margin = margin(b = 8)),
      plot.caption     = element_text(color = "#64748b", size = base_size - 2, hjust = 0,
                                      margin = margin(t = 8), lineheight = 1.2),
      plot.caption.position = "plot",
      plot.margin      = margin(14, 18, 12, 14),
      axis.title       = element_text(face = "bold", size = base_size),
      axis.text        = element_text(color = "#334155", size = base_size - 1),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "#e2e8f0", linewidth = 0.4),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background  = element_rect(fill = "white", color = NA),
      legend.position  = "right",
      legend.title     = element_text(face = "bold", size = base_size - 1),
      legend.text      = element_text(size = base_size - 1),
      strip.text       = element_text(face = "bold", size = base_size, color = "#1e293b")
    )
}

# ── Save helper ──
save_fig <- function(p, path, w = 10, h = 6) {
  ggsave(path, p, width = w, height = h, dpi = 300, bg = "white")
  cat("  [saved]", basename(path), "\n")
}

# ── Reusable plot builders ──

# Bar comparison (horizontal)
plot_bar_h <- function(df, x_col, y_col, fill_col = NULL,
                       title = "", subtitle = "", caption = "",
                       xlab = "", ylab = "") {
  aes_map <- if (!is.null(fill_col)) {
    aes(.data[[x_col]], .data[[y_col]], fill = .data[[fill_col]])
  } else {
    aes(.data[[x_col]], .data[[y_col]], fill = .data[[x_col]])
  }
  ggplot(df, aes_map) +
    geom_col(width = 0.7, color = "white") +
    coord_flip() +
    scale_fill_viridis_d(option = "mako", end = 0.85, guide = "none") +
    labs(title = title, subtitle = subtitle, caption = caption,
         x = xlab, y = ylab) +
    theme_sers()
}

# ROC curve from TPR/FPR data
plot_roc <- function(fpr, tpr, auc_val, title = "ROC Curve", subtitle = "") {
  df <- data.frame(fpr = fpr, tpr = tpr)
  ggplot(df, aes(fpr, tpr)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#94a3b8") +
    geom_line(color = "#1d4ed8", linewidth = 1.2) +
    annotate("text", x = 0.6, y = 0.15, family = KFONT, size = 4.5, color = "#1e3a8a",
             label = sprintf("AUC = %.4f", auc_val), fontface = "bold") +
    scale_x_continuous(limits = c(0, 1)) + scale_y_continuous(limits = c(0, 1)) +
    labs(title = title, subtitle = subtitle, x = "1 - Specificity (FPR)", y = "Sensitivity (TPR)") +
    theme_sers() + coord_equal()
}

# Confusion matrix heatmap
plot_confusion <- function(mat, labels, title = "Confusion Matrix", subtitle = "") {
  df <- expand.grid(Predicted = labels, True = rev(labels))
  df$Count <- as.vector(t(mat[nrow(mat):1, ]))
  ggplot(df, aes(Predicted, True, fill = Count)) +
    geom_tile(color = "white", linewidth = 1) +
    geom_text(aes(label = Count), family = KFONT, size = 4.5, fontface = "bold") +
    scale_fill_viridis_c(option = "mako", direction = -1, end = 0.9, guide = "none") +
    labs(title = title, subtitle = subtitle) +
    theme_sers() + theme(axis.title = element_blank()) +
    coord_equal()
}

# Spectrum overlay
plot_spectra_overlay <- function(df_long, title = "", subtitle = "", caption = "",
                                 color_col = "group") {
  ggplot(df_long, aes(wavenumber, intensity, color = .data[[color_col]])) +
    geom_line(alpha = 0.7, linewidth = 0.5) +
    scale_color_manual(values = PAL_CANCER, name = "") +
    labs(title = title, subtitle = subtitle, caption = caption,
         x = "Wavenumber (cm-1)", y = "Intensity (SNV)") +
    theme_sers() + theme(legend.position = "top")
}

cat("[theme_sers.R] loaded\n")
