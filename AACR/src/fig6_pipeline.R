#!/usr/bin/env Rscript
# =============================================================================
# AACR Figure 6 — Two-Stage Classification Pipeline (standalone)
# =============================================================================
suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
})

out_dir <- "/home/user/SERS-AI/AACR"

# --- canvas ---
fig <- ggplot() +
  coord_cartesian(xlim = c(0, 24), ylim = c(0, 16), expand = FALSE) +
  theme_void() +
  theme(plot.background = element_rect(fill = "white", color = NA),
        plot.margin = margin(8, 8, 8, 8))

# --- helpers -----------------------------------------------------------------
box <- function(xc, yc, w, h, fill, border = NULL, r = 0, lw = 0.4) {
  bc <- if (is.null(border)) fill else border
  annotate("rect", xmin = xc - w/2, xmax = xc + w/2,
           ymin = yc - h/2, ymax = yc + h/2,
           fill = fill, color = bc, linewidth = lw)
}

lab <- function(x, y, text, sz = 3, col = "#222222", face = "plain", ...) {
  annotate("text", x = x, y = y, label = text,
           size = sz, color = col, fontface = face, ...)
}

arrow_r <- function(x1, y1, x2, y2, col = "#aaaaaa", lw = 0.4) {
  annotate("segment", x = x1, y = y1, xend = x2, yend = y2,
           arrow = arrow(length = unit(0.15, "cm"), type = "closed"),
           color = col, linewidth = lw)
}

line_s <- function(x1, y1, x2, y2, col = "#aaaaaa", lw = 0.4, lt = "solid") {
  annotate("segment", x = x1, y = y1, xend = x2, yend = y2,
           color = col, linewidth = lw, linetype = lt)
}

# --- palette -----------------------------------------------------------------
bg_blue   <- "#eef4fb"
bd_blue   <- "#4a90d9"
tx_blue   <- "#2b6cb0"
bg_orange <- "#fef6ee"
bd_orange <- "#e8792b"
tx_orange <- "#c05621"
bg_green  <- "#edf7ed"
bd_green  <- "#38a169"
bg_red    <- "#fde8e8"
bd_red    <- "#c53030"
tx_red    <- "#c53030"
bg_purple <- "#f5f0fa"
bd_purple <- "#805ad5"
tx_purple <- "#6b46c1"
gray1     <- "#f7f7f7"
gray2     <- "#e2e2e2"
gray3     <- "#718096"
gray4     <- "#4a5568"
gray5     <- "#2d3748"

# =============================================================================
# TITLE
# =============================================================================
fig <- fig +
  lab(12, 15.5, "Two-Stage SERS-Based Multi-Cancer Classification Pipeline",
      5.5, gray5, "bold") +
  lab(12, 14.95, "AECD Platform   |   Urine-based screening   |   Logistic Regression",
      3, gray3, "plain")

# =============================================================================
# ROW 1 — INPUT & PREPROCESSING (y ~ 13.5)
# =============================================================================
y1 <- 13.5

# Input
fig <- fig +
  box(1.8, y1, 2.8, 1.1, gray1, gray2) +
  lab(1.8, y1 + 0.2, "Urine SERS", 3, gray5, "bold") +
  lab(1.8, y1 - 0.2, "1,240 patients", 2.2, gray3)

fig <- fig + arrow_r(3.2, y1, 4.0, y1)

# Preprocessing steps
pre_labels <- c("Spectral\nTrim", "Savitzky-\nGolay", "Baseline\nCorrection",
                "SNV\nNormalization", "Fixed\nGrid", "Medoid\nSelection")
pre_params <- c("400-2200 cm-1", "w=11, p=3", "Rolling min", "Mean=0, SD=1",
                "935 points", "1 per patient")
pre_x <- seq(5.2, 22, length.out = 6)
shades <- colorRampPalette(c("#90a4ae", "#37474f"))(6)

for (i in seq_along(pre_labels)) {
  fig <- fig +
    box(pre_x[i], y1, 2.4, 1.1, shades[i]) +
    lab(pre_x[i], y1 + 0.18, pre_labels[i], 2.2, "white", "bold", lineheight = 0.8) +
    lab(pre_x[i], y1 - 0.3, pre_params[i], 1.7, "#cfd8dc", "plain")
  if (i < length(pre_labels))
    fig <- fig + arrow_r(pre_x[i] + 1.2, y1, pre_x[i+1] - 1.2, y1, "#bbbbbb")
}

# =============================================================================
# FEATURE VECTOR (y ~ 12)
# =============================================================================
y_fv <- 11.9
fig <- fig +
  arrow_r(12, y1 - 0.55, 12, y_fv + 0.25, "#bbbbbb") +
  box(12, y_fv, 6, 0.35, gray1, gray2, lw = 0.3) +
  lab(12, y_fv, "Feature vector  x  in  R^935   (preprocessed spectral intensities)",
      2.5, gray4, "italic")

# =============================================================================
# STAGE 1 — CANCER SCREENING (y ~ 8.5 - 10.5)
# =============================================================================
s1_top <- 11.2
s1_bot <- 7.6
s1_mid <- (s1_top + s1_bot) / 2

# Background
fig <- fig +
  box(8, s1_mid, 14.5, s1_top - s1_bot, bg_blue, bd_blue, lw = 0.6) +
  lab(1.5, s1_top - 0.4, "STAGE 1", 4, tx_blue, "bold", hjust = 0) +
  lab(1.5, s1_top - 0.9, "Cancer Screening", 3, tx_blue, "bold", hjust = 0) +
  lab(1.5, s1_top - 1.35, "Binary classification", 2.2, "#90caf9", "plain", hjust = 0)

# Arrow into Stage 1
fig <- fig + arrow_r(12, y_fv - 0.2, 6, s1_top - 0.3, "#bbbbbb")

# Model box
fig <- fig +
  box(6, s1_mid, 4.5, 2.2, "white", bd_blue, lw = 0.5) +
  lab(6, s1_mid + 0.7, "Logistic Regression", 3, tx_blue, "bold") +
  line_s(3.8, s1_mid + 0.35, 8.2, s1_mid + 0.35, "#d0e3f7", 0.3) +
  lab(6, s1_mid, "L2 regularization  (C = 1.0)", 2, gray3, "plain") +
  lab(6, s1_mid - 0.35, "Balanced class weights", 2, gray3, "plain") +
  lab(6, s1_mid - 0.7, "SAGA solver, max_iter=1000", 2, gray3, "plain")

# Decision split
fig <- fig +
  line_s(8.25, s1_mid, 9.2, s1_mid, "#bbbbbb") +
  line_s(9.2, s1_mid, 9.2, s1_mid + 0.7, "#bbbbbb") +
  line_s(9.2, s1_mid, 9.2, s1_mid - 0.7, "#bbbbbb") +
  arrow_r(9.2, s1_mid + 0.7, 10.0, s1_mid + 0.7, tx_red) +
  arrow_r(9.2, s1_mid - 0.7, 10.0, s1_mid - 0.7, bd_green)

# Output labels
fig <- fig +
  box(11.2, s1_mid + 0.7, 1.8, 0.6, bd_red) +
  lab(11.2, s1_mid + 0.7, "Cancer", 2.8, "white", "bold") +
  lab(12.4, s1_mid + 0.7, "p >= 0.5", 1.8, tx_red, "italic", hjust = 0) +
  box(11.5, s1_mid - 0.7, 2.2, 0.6, bd_green) +
  lab(11.5, s1_mid - 0.7, "Non-Cancer", 2.5, "white", "bold") +
  lab(12.9, s1_mid - 0.7, "p < 0.5", 1.8, bd_green, "italic", hjust = 0)

# Performance box
fig <- fig +
  box(6, s1_bot + 0.4, 5.5, 0.5, "white", bd_blue, lw = 0.3) +
  lab(6, s1_bot + 0.4,
      "AUC 0.981   |   Sensitivity 93.9%   |   Specificity 92.2%",
      2.2, tx_blue, "bold")

# =============================================================================
# CONNECTING ARROW: Stage 1 Cancer -> Stage 2
# =============================================================================
fig <- fig +
  arrow_r(11.2, s1_mid + 0.4, 11.2, 5.8, tx_red, 0.5) +
  lab(11.9, 6.5, "Cancer\nsamples", 2, tx_red, "italic", hjust = 0, lineheight = 0.85)

# =============================================================================
# STAGE 2 — CANCER TYPE ID (y ~ 3.5 - 5.5)
# =============================================================================
s2_top <- 5.7
s2_bot <- 2.2
s2_mid <- (s2_top + s2_bot) / 2

fig <- fig +
  box(13, s2_mid, 20, s2_top - s2_bot, bg_orange, bd_orange, lw = 0.6) +
  lab(3.5, s2_top - 0.4, "STAGE 2", 4, tx_orange, "bold", hjust = 0) +
  lab(3.5, s2_top - 0.9, "Cancer Type Identification", 3, tx_orange, "bold", hjust = 0) +
  lab(3.5, s2_top - 1.35, "Multi-class, One-vs-Rest", 2.2, "#ffcc80", "plain", hjust = 0)

# Model box
fig <- fig +
  box(9, s2_mid + 0.1, 4.5, 2.2, "white", bd_orange, lw = 0.5) +
  lab(9, s2_mid + 0.8, "Logistic Regression", 3, tx_orange, "bold") +
  line_s(6.8, s2_mid + 0.45, 11.2, s2_mid + 0.45, "#fde0c2", 0.3) +
  lab(9, s2_mid + 0.1, "One-vs-Rest  (5 sub-classifiers)", 2, gray3, "plain") +
  lab(9, s2_mid - 0.25, "argmax probability assignment", 2, gray3, "plain") +
  lab(9, s2_mid - 0.6, "Same L2 / balanced / SAGA", 2, gray3, "plain")

# Fan-out to 5 cancer types
cancer_full  <- c("Prostate", "Ovarian", "Lung", "Pancreatic", "Colorectal")
cancer_short <- c("PRO", "OVA", "LUN", "PAN", "CRC")
cancer_n     <- c("n=100", "n=70", "n=300", "n=70", "n=300")
cc <- c("#2b6cb0", "#805ad5", "#38a169", "#dd6b20", "#c53030")

fan_x <- 12.5
y_out <- seq(s2_top - 0.6, s2_bot + 0.5, length.out = 5)

fig <- fig +
  line_s(11.25, s2_mid + 0.1, fan_x, s2_mid + 0.1, "#bbbbbb")

for (i in seq_along(cancer_short)) {
  fig <- fig +
    line_s(fan_x, s2_mid + 0.1, fan_x + 0.8, y_out[i], cc[i], 0.35) +
    arrow_r(fan_x + 0.8, y_out[i], fan_x + 1.3, y_out[i], cc[i], 0.35) +
    box(fan_x + 2.2, y_out[i], 1.4, 0.45, cc[i]) +
    lab(fan_x + 2.2, y_out[i], cancer_short[i], 2.5, "white", "bold") +
    lab(fan_x + 3.2, y_out[i] + 0.05, cancer_full[i], 2.2, cc[i], "plain", hjust = 0) +
    lab(fan_x + 3.2, y_out[i] - 0.25, cancer_n[i], 1.7, gray3, "plain", hjust = 0)
}

# Performance box
fig <- fig +
  box(9, s2_bot + 0.35, 5.5, 0.5, "white", bd_orange, lw = 0.3) +
  lab(9, s2_bot + 0.35,
      "F1 macro 0.871   |   AUC 0.985   |   Accuracy 90.2%",
      2.2, tx_orange, "bold")

# =============================================================================
# BOTTOM — Optional extensions
# =============================================================================
yb <- 1.2

# Ensemble
fig <- fig +
  box(6.5, yb, 11, 1.1, bg_red, bd_red, lw = 0.4) +
  lab(6.5, yb + 0.3,
      "Optional Ensemble:  80% LR  +  20% ResNet18-1D", 2.5, tx_red, "bold") +
  lab(6.5, yb - 0.15,
      "Screening AUC 0.984 (+0.003)   |   Type ID F1 0.875 (+0.004)", 2, gray3, "plain")

# Fusion
fig <- fig +
  box(17.5, yb, 11, 1.1, bg_purple, bd_purple, lw = 0.4) +
  lab(17.5, yb + 0.3,
      "Optional Fusion:  Age + Sex + BMI", 2.5, tx_purple, "bold") +
  lab(17.5, yb - 0.15,
      "Sex constraint eliminates PRO / OVA cross-classification", 2, gray3, "plain")

# =============================================================================
# SAVE
# =============================================================================
ggsave(file.path(out_dir, "fig6_pipeline_architecture.pdf"), fig,
       width = 13, height = 8.5, dpi = 300)
ggsave(file.path(out_dir, "fig6_pipeline_architecture.png"), fig,
       width = 13, height = 8.5, dpi = 300)
cat("Saved fig6_pipeline_architecture.pdf/png\n")
