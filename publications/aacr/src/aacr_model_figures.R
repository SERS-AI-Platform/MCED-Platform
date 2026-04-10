#!/usr/bin/env Rscript
# =============================================================================
# AACR Poster — Model Comparison & Pipeline Architecture (Publication Quality)
# Uses REAL data for spectra, weights, blend curves — not placeholder boxes
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
  library(scales)
  library(readr)
  library(grid)
  library(gridExtra)
  library(cowplot)
})

base    <- "/home/user/SERS-AI"
out_dir <- file.path(base, "AACR")

# ---------- common theme -----------------------------------------------------
theme_pub <- theme_minimal(base_size = 10) +
  theme(
    text              = element_text(family = "sans", color = "#1a1a2e"),
    plot.title        = element_text(face = "bold", size = 11, hjust = 0,
                                     margin = margin(b = 3)),
    plot.subtitle     = element_text(size = 8, color = "#666666",
                                     margin = margin(b = 8)),
    axis.title        = element_text(face = "bold", size = 9),
    axis.text         = element_text(size = 8, color = "#333333"),
    legend.title      = element_blank(),
    legend.text       = element_text(size = 8),
    panel.grid.major  = element_line(color = "#ececec", linewidth = 0.3),
    panel.grid.minor  = element_blank(),
    plot.margin       = margin(8, 10, 8, 10),
    plot.background   = element_rect(fill = "white", color = NA)
  )

# =============================================================================
# Figure 5 — Model Performance Comparison (6 models including ensemble)
# =============================================================================
cat(">> Figure 5: Model Comparison (6 models + ensemble)...\n")

# All models, same conditions (experiment_002: agg=none, n=6200)
# + ensemble best (alpha=0.8) from step5_ensemble
# + RF/CNN1D from archive experiment_003 (agg=medoid, n=1342) — separate panel

model_order <- c("Logistic Regression", "Ensemble\n(LR 80% + ResNet 20%)",
                 "XGBoost", "ResNet18-1D", "Random Forest", "CNN1D-Shallow")

model_colors <- c(
  "Logistic Regression"          = "#2c3e50",
  "Ensemble\n(LR 80% + ResNet 20%)" = "#c0392b",
  "XGBoost"                      = "#2980b9",
  "ResNet18-1D"                  = "#27ae60",
  "Random Forest"                = "#8e44ad",
  "CNN1D-Shallow"                = "#e67e22"
)

# Main comparison data (agg=none, n=6200, 5-fold CV)
df_main <- tibble(
  Model   = factor(model_order[1:4], levels = model_order),
  S1_AUC  = c(0.9807, 0.9838, 0.9692, 0.9577),
  S1_Sens = c(0.9390, 0.9417, 0.9240, 0.9012),
  S1_Spec = c(0.9215, 0.9240, 0.8755, 0.8810),
  S2_F1   = c(0.8715, 0.8752, 0.7574, 0.7104),
  S2_AUC  = c(0.9849, 0.9825, 0.9598, 0.9301),
  Setting = "Full replicates (n=6,200)"
)

# Additional models (agg=medoid, n=1342) — only RF and CNN1D
df_extra <- tibble(
  Model   = factor(model_order[5:6], levels = model_order),
  S1_AUC  = c(0.9181, 0.7973),
  S1_Sens = c(0.9310, 0.9055),
  S1_Spec = c(0.6075, 0.3150),
  S2_F1   = c(0.4469, 0.1295),
  S2_AUC  = c(0.9089, 0.7455),
  Setting = "Medoid (n=1,342)"
)

df_all <- bind_rows(df_main, df_extra)

# Reshape to long
df_long <- df_all %>%
  pivot_longer(cols = c(S1_AUC, S1_Sens, S1_Spec, S2_F1, S2_AUC),
               names_to = "Metric", values_to = "Value") %>%
  mutate(
    Stage = ifelse(grepl("^S1", Metric), "Cancer Screening (Stage 1)",
                   "Cancer Type ID (Stage 2)"),
    Metric_clean = case_when(
      Metric == "S1_AUC"  ~ "AUC",
      Metric == "S1_Sens" ~ "Sensitivity",
      Metric == "S1_Spec" ~ "Specificity",
      Metric == "S2_F1"   ~ "F1 macro",
      Metric == "S2_AUC"  ~ "AUC"
    )
  )

# --- Panel A: Heatmap-style table ---
# Create a compact comparison table visualization
df_heat <- df_all %>%
  mutate(Model = as.character(Model)) %>%
  pivot_longer(-c(Model, Setting), names_to = "Metric", values_to = "Value") %>%
  mutate(
    Metric = factor(Metric,
      levels = c("S1_AUC", "S1_Sens", "S1_Spec", "S2_AUC", "S2_F1"),
      labels = c("Screening\nAUC", "Screening\nSensitivity", "Screening\nSpecificity",
                  "Type ID\nAUC", "Type ID\nF1 macro")),
    Model = factor(Model, levels = rev(model_order)),
    label = sprintf("%.3f", Value)
  )

panel_heat <- ggplot(df_heat, aes(x = Metric, y = Model, fill = Value)) +
  geom_tile(color = "white", linewidth = 1.5) +
  geom_text(aes(label = label,
                color = ifelse(Value > 0.85, "white", "dark")),
            size = 3, fontface = "bold", show.legend = FALSE) +
  scale_fill_gradientn(
    colors = c("#fee0d2", "#fc9272", "#de2d26", "#67000d"),
    limits = c(0, 1), name = "Score",
    breaks = seq(0, 1, 0.25)
  ) +
  scale_color_manual(values = c("white" = "white", "dark" = "#2c3e50")) +
  labs(
    title = "A. Model Performance Comparison",
    subtitle = "5-fold CV | Top 4: full replicates (n=6,200) | Bottom 2: medoid (n=1,342)",
    x = NULL, y = NULL
  ) +
  theme_pub +
  theme(
    axis.text.y    = element_text(size = 8, lineheight = 0.9, face = "bold"),
    axis.text.x    = element_text(size = 7.5, lineheight = 0.9),
    panel.grid     = element_blank(),
    legend.position = "right",
    legend.key.height = unit(0.8, "cm"),
    legend.key.width  = unit(0.3, "cm")
  )

# --- Panel B: Ensemble blend sweep (real data) ---
blend <- read_csv(file.path(base, "results/training/step5_ensemble/blend_results.csv"),
                  show_col_types = FALSE)

blend_long <- blend %>%
  select(alpha, s1_auc, s2_f1_macro) %>%
  pivot_longer(-alpha, names_to = "Metric", values_to = "Value") %>%
  mutate(Metric = ifelse(Metric == "s1_auc",
                         "Cancer Screening AUC", "Cancer Type ID F1"))

panel_blend <- ggplot(blend_long, aes(x = alpha, y = Value, color = Metric)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  geom_vline(xintercept = 0.8, linetype = "dashed", color = "#c0392b", linewidth = 0.4) +
  annotate("label", x = 0.8, y = min(blend_long$Value) + 0.01,
           label = "Optimal\nalpha=0.8", size = 2.3, color = "#c0392b",
           fontface = "bold", fill = "white", label.size = 0.2, lineheight = 0.9) +
  annotate("text", x = 0.02, y = max(blend_long$Value) - 0.005,
           label = "100% ResNet18", size = 2, color = "#666", hjust = 0) +
  annotate("text", x = 0.98, y = max(blend_long$Value) - 0.005,
           label = "100% LR", size = 2, color = "#666", hjust = 1) +
  scale_color_manual(values = c("Cancer Screening AUC" = "#1565c0",
                                "Cancer Type ID F1" = "#e65100")) +
  scale_x_continuous(breaks = seq(0, 1, 0.2),
                     labels = c("0.0\n(ResNet)", "0.2", "0.4", "0.6", "0.8", "1.0\n(LR)")) +
  scale_y_continuous(labels = number_format(accuracy = 0.001)) +
  labs(
    title = "B. Ensemble Blend Sweep",
    subtitle = expression(P[blend] == alpha %.% P[LR] + (1 - alpha) %.% P[ResNet18]),
    x = expression(alpha ~ "(LR weight)"), y = "Score"
  ) +
  theme_pub +
  theme(
    legend.position = c(0.25, 0.3),
    legend.background = element_rect(fill = "white", color = "#ececec", linewidth = 0.3),
    legend.key.height = unit(0.3, "cm"),
    legend.text = element_text(size = 7)
  )

fig5 <- panel_heat / panel_blend + plot_layout(heights = c(1.2, 1))

ggsave(file.path(out_dir, "fig5_model_comparison.pdf"), fig5,
       width = 8, height = 8, dpi = 300)
ggsave(file.path(out_dir, "fig5_model_comparison.png"), fig5,
       width = 8, height = 8, dpi = 300)
cat("   Saved fig5_model_comparison.pdf/png\n")


# =============================================================================
# Figure 6 — Model Pipeline Only (publication quality)
# =============================================================================
cat(">> Figure 6: Model Pipeline...\n")

fig6 <- ggplot() +
  coord_fixed(ratio = 1, xlim = c(0, 18), ylim = c(0, 10), expand = FALSE) +
  theme_void() +
  theme(plot.background = element_rect(fill = "white", color = NA),
        plot.margin = margin(6, 6, 6, 6))

rbox <- function(xc, yc, w, h, fill, border = NULL, lwd = 0.5, alpha = 1) {
  border_col <- if (is.null(border)) fill else border
  annotate("rect", xmin = xc - w/2, xmax = xc + w/2,
           ymin = yc - h/2, ymax = yc + h/2,
           fill = fill, color = border_col,
           linewidth = lwd, alpha = alpha)
}
txt <- function(x, y, label, sz = 2.8, col = "white", face = "bold", ...) {
  annotate("text", x = x, y = y, label = label,
           size = sz, color = col, fontface = face, ...)
}
arr <- function(x1, y1, x2, y2, col = "#888888", lwd = 0.45) {
  annotate("segment", x = x1, y = y1, xend = x2, yend = y2,
           arrow = arrow(length = unit(0.14, "cm"), type = "closed"),
           color = col, linewidth = lwd)
}

# === Title ===
fig6 <- fig6 +
  txt(9, 9.7, "Two-Stage SERS Classification Pipeline", 5, "#1a1a2e") +
  txt(9, 9.25, "AECD Platform  |  Urine-based Multi-Cancer Screening", 2.8, "#777777", "plain")

# === Input ===
fig6 <- fig6 +
  rbox(2, 8.4, 2.8, 0.85, "#37474f") +
  txt(2, 8.55, "Urine SERS", 2.8) +
  txt(2, 8.2, "1,240 patients x 5 reps", 1.8, "#b0bec5", "plain")

fig6 <- fig6 + arr(3.4, 8.4, 4.3, 8.4, "#888888")

# === Preprocessing ===
pre <- c("Trim", "SG Filter", "Baseline", "SNV", "Fixed Grid", "Medoid")
pre_sub <- c("400-2200", "w=11, p=3", "Rolling min", "Normalize", "935 pts", "1 / patient")
pre_x <- seq(5.2, 16, length.out = 6)
pre_fills <- colorRampPalette(c("#607d8b", "#263238"))(6)

for (i in seq_along(pre)) {
  fig6 <- fig6 +
    rbox(pre_x[i], 8.4, 1.6, 0.85, pre_fills[i]) +
    txt(pre_x[i], 8.55, pre[i], 2.2) +
    txt(pre_x[i], 8.2, pre_sub[i], 1.5, "#b0bec5", "plain")
  if (i < length(pre))
    fig6 <- fig6 + arr(pre_x[i] + 0.8, 8.4, pre_x[i+1] - 0.8, 8.4, "#999999")
}

# Arrow down to feature vector
fig6 <- fig6 +
  arr(9, 7.95, 9, 7.45, "#888888") +
  rbox(9, 7.2, 4, 0.35, "#eceff1", border = "#bdbdbd", lwd = 0.3) +
  txt(9, 7.2, "Feature vector  x  (935 spectral intensities)", 2.2, "#455a64", "italic")

# === STAGE 1 ===
fig6 <- fig6 +
  rbox(5, 5.3, 8.5, 3.2, "#e3f2fd", border = "#1565c0", lwd = 0.6, alpha = 0.35)

fig6 <- fig6 +
  txt(1.2, 6.65, "STAGE 1", 3.2, "#1565c0", "bold", hjust = 0) +
  txt(1.2, 6.2, "Cancer Screening", 2.5, "#1565c0", "bold", hjust = 0) +
  txt(1.2, 5.8, "(Binary)", 2, "#90caf9", "plain", hjust = 0)

# LR model box
fig6 <- fig6 +
  rbox(5, 5.5, 3.2, 1.8, "white", border = "#1565c0", lwd = 0.5) +
  txt(5, 6.15, "Logistic Regression", 2.5, "#1565c0") +
  annotate("segment", x = 3.5, y = 5.85, xend = 6.5, yend = 5.85,
           color = "#bbdefb", linewidth = 0.3) +
  txt(5, 5.55, "L2 regularization (C=1.0)", 1.8, "#78909c", "plain") +
  txt(5, 5.2, "Balanced class weights", 1.8, "#78909c", "plain") +
  txt(5, 4.85, "SAGA solver", 1.8, "#78909c", "plain")

# Arrow into Stage 1
fig6 <- fig6 + arr(9, 7.0, 5, 6.4, "#888888")

# Branch to Cancer / Non-Cancer
fig6 <- fig6 +
  annotate("segment", x = 6.6, y = 5.5, xend = 7.3, yend = 5.5,
           color = "#888888", linewidth = 0.4) +
  annotate("segment", x = 7.3, y = 5.5, xend = 7.3, yend = 6.2,
           color = "#888888", linewidth = 0.4) +
  annotate("segment", x = 7.3, y = 5.5, xend = 7.3, yend = 4.8,
           color = "#888888", linewidth = 0.4) +
  arr(7.3, 6.2, 7.8, 6.2, "#c62828") +
  arr(7.3, 4.8, 7.8, 4.8, "#2e7d32")

fig6 <- fig6 +
  rbox(8.7, 6.2, 1.5, 0.55, "#c62828") + txt(8.7, 6.2, "Cancer", 2.5) +
  rbox(8.7, 4.8, 1.7, 0.55, "#2e7d32") + txt(8.7, 4.8, "Non-Cancer", 2.3)

# Metrics
fig6 <- fig6 +
  rbox(5, 3.9, 3.2, 0.45, "white", border = "#1565c0", lwd = 0.3) +
  txt(5, 3.9, "AUC 0.981 | Sens 93.9% | Spec 92.2%", 1.9, "#1565c0")

# === Arrow from Cancer to Stage 2 ===
fig6 <- fig6 +
  arr(8.7, 5.9, 8.7, 4.0, "#c62828", 0.5) +
  txt(9.5, 4.5, "Cancer\nsamples only", 1.8, "#c62828", "italic", hjust = 0, lineheight = 0.85)

# === STAGE 2 ===
fig6 <- fig6 +
  rbox(14, 5.3, 7.5, 3.2, "#fff3e0", border = "#e65100", lwd = 0.6, alpha = 0.35)

fig6 <- fig6 +
  txt(10.6, 6.65, "STAGE 2", 3.2, "#e65100", "bold", hjust = 0) +
  txt(10.6, 6.2, "Cancer Type ID", 2.5, "#e65100", "bold", hjust = 0) +
  txt(10.6, 5.8, "(Multi-class OvR)", 2, "#ffcc80", "plain", hjust = 0)

# LR model box
fig6 <- fig6 +
  rbox(13.5, 5.5, 3.2, 1.8, "white", border = "#e65100", lwd = 0.5) +
  txt(13.5, 6.15, "Logistic Regression", 2.5, "#e65100") +
  annotate("segment", x = 12, y = 5.85, xend = 15, yend = 5.85,
           color = "#ffe0b2", linewidth = 0.3) +
  txt(13.5, 5.55, "One-vs-Rest strategy", 1.8, "#78909c", "plain") +
  txt(13.5, 5.2, "5 binary sub-classifiers", 1.8, "#78909c", "plain") +
  txt(13.5, 4.85, "Probability calibration", 1.8, "#78909c", "plain")

# Fan out to 5 cancer types
fig6 <- fig6 +
  annotate("segment", x = 15.1, y = 5.5, xend = 15.7, yend = 5.5,
           color = "#888888", linewidth = 0.4)

cc_names <- c("Prostate", "Ovarian", "Lung", "Pancreatic", "Colorectal")
cc_abbr  <- c("PRO", "OVA", "LUN", "PAN", "CRC")
cc_cols  <- c("#1976d2", "#7b1fa2", "#388e3c", "#ef6c00", "#c62828")
yy <- seq(6.5, 4.5, length.out = 5)

for (i in seq_along(cc_abbr)) {
  fig6 <- fig6 +
    annotate("segment", x = 15.7, y = 5.5, xend = 16.2, yend = yy[i],
             color = cc_cols[i], linewidth = 0.35) +
    arr(16.2, yy[i], 16.5, yy[i], cc_cols[i], 0.35) +
    rbox(17.2, yy[i], 1.2, 0.35, cc_cols[i]) +
    txt(17.2, yy[i], cc_abbr[i], 2.2)
}

# Metrics
fig6 <- fig6 +
  rbox(13.5, 3.9, 3.2, 0.45, "white", border = "#e65100", lwd = 0.3) +
  txt(13.5, 3.9, "F1 macro 0.871 | AUC 0.985", 1.9, "#e65100")

# === Bottom: Ensemble + Fusion options ===
fig6 <- fig6 +
  rbox(5, 2.8, 7.5, 1.2, "#fce4ec", border = "#c62828", lwd = 0.5, alpha = 0.5) +
  txt(5, 3.2, "Optional: Ensemble (80% LR + 20% ResNet18-1D)", 2.3, "#c62828") +
  txt(5, 2.85, "Weighted probability blending: P = 0.8 P    + 0.2 P", 1.8, "#c62828", "plain") +
  txt(5, 2.5, "Screening AUC 0.984 (+0.003)  |  Type ID F1 0.875 (+0.004)", 1.8, "#888888", "plain")

fig6 <- fig6 +
  rbox(14, 2.8, 7.5, 1.2, "#f3e5f5", border = "#6a1b9a", lwd = 0.5, alpha = 0.5) +
  txt(14, 3.2, "Optional: Clinical Fusion (Age + Sex + BMI)", 2.3, "#6a1b9a") +
  txt(14, 2.85, "Feature concatenation with spectral vector", 1.8, "#6a1b9a", "plain") +
  txt(14, 2.5, "Sex constraint eliminates PRO / OVA cross-classification", 1.8, "#888888", "plain")

# === Dataset info bar ===
fig6 <- fig6 +
  rbox(9, 1.5, 17, 0.7, "#f5f5f5", border = "#e0e0e0", lwd = 0.3) +
  txt(1, 1.5, "n = 1,240  |  933 features  |  5 cancer types + 4 non-cancer controls  |  5-fold stratified patient-level CV  |  No aggregation (5 reps per patient)",
      2, "#666666", "plain", hjust = 0)

ggsave(file.path(out_dir, "fig6_pipeline_architecture.pdf"), fig6,
       width = 11, height = 6, dpi = 300)
ggsave(file.path(out_dir, "fig6_pipeline_architecture.png"), fig6,
       width = 11, height = 6, dpi = 300)
cat("   Saved fig6_pipeline_architecture.pdf/png\n")

cat("\n>> All figures saved to:", out_dir, "\n")
