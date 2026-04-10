#!/usr/bin/env Rscript
# =============================================================================
# AACR Poster Figures — SERS-based Multi-Cancer Screening
# SOLUM Healthcare / AECD Platform
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(scales)
  library(patchwork)
  library(RColorBrewer)
  library(cowplot)
  library(stringr)
})

# ---------- paths ------------------------------------------------------------
base      <- "/home/user/SERS-AI"
spec_path <- file.path(base, "results/processed_spectra.csv")
overlay   <- file.path(base, "results/training/R_BLC_added/logistic_regression/v001/evaluation/stage2/mean_spectra_overlay.csv")
confpairs <- file.path(base, "AACR/confusion_pairs_exp002.csv")
out_dir   <- file.path(base, "AACR")

# ---------- theme ------------------------------------------------------------
theme_poster <- theme_minimal(base_size = 22) +
  theme(
    text              = element_text(family = "sans", color = "#1a1a2e"),
    plot.title        = element_text(face = "bold", size = 26, hjust = 0,
                                     margin = margin(b = 10)),
    plot.subtitle     = element_text(size = 18, color = "#555555",
                                     margin = margin(b = 14)),
    axis.title        = element_text(face = "bold", size = 20),
    axis.text         = element_text(size = 18, color = "#333333"),
    legend.title      = element_text(face = "bold", size = 18),
    legend.text       = element_text(size = 16),
    legend.key.height = unit(0.8, "cm"),
    legend.key.width  = unit(1.2, "cm"),
    panel.grid.major  = element_line(color = "#e8e8e8", linewidth = 0.3),
    panel.grid.minor  = element_blank(),
    strip.text        = element_text(face = "bold", size = 20),
    plot.margin       = margin(20, 24, 20, 24),
    plot.background   = element_rect(fill = "white", color = NA)
  )
theme_set(theme_poster)

# ---------- color palette (6 groups) -----------------------------------------
group_colors <- c(
  "Non-Cancer"  = "#1b3a5c",
  "PRC"         = "#e8733a",
  "OVC"         = "#7b2d8e",
  "LC"          = "#d4a088",
  "PAC"         = "#a01020",
  "CRC"         = "#2980b9"
)

group_order <- names(group_colors)

# =============================================================================
# Figure 2 — Mean SERS spectra by cancer type
# =============================================================================
cat(">> Figure 2: Mean SERS Spectra...\n")

spec <- read_csv(spec_path, show_col_types = FALSE)

# Map groups
spec <- spec %>%
  mutate(cancer_group = case_when(
    group %in% c("NOR", "DIA", "HBP", "H.D.") ~ "Non-Cancer",
    group == "PRO"  ~ "PRC",
    group == "OVA"  ~ "OVC",
    group == "LUN"  ~ "LC",
    group == "CPAN" ~ "PAC",
    group == "CRC"  ~ "CRC",
    TRUE ~ NA_character_
  )) %>%
  filter(!is.na(cancer_group))

# Pivot to long: wavenumber columns start with "x_"
wn_cols <- names(spec)[grepl("^x_", names(spec))]
wavenumbers <- as.numeric(sub("^x_", "", wn_cols))

# Compute mean ± SEM per group per wavenumber (medoid per patient first)
# For speed, sample 1 replicate per sample_id (medoid is replicate=1 in processed)
spec_medoid <- spec %>%
  group_by(cancer_group, group, sample_id) %>%
  slice(1) %>%
  ungroup()

spec_long <- spec_medoid %>%
  select(cancer_group, all_of(wn_cols)) %>%
  pivot_longer(cols = all_of(wn_cols), names_to = "feature", values_to = "intensity") %>%
  mutate(wavenumber = as.numeric(sub("^x_", "", feature)))

spec_summary <- spec_long %>%
  group_by(cancer_group, wavenumber) %>%
  summarise(
    mean_int = mean(intensity, na.rm = TRUE),
    sd_int   = sd(intensity, na.rm = TRUE),
    n        = n(),
    sem      = sd_int / sqrt(n),
    .groups  = "drop"
  ) %>%
  mutate(cancer_group = factor(cancer_group, levels = group_order))

# Downsample for cleaner plot (every 3rd point)
spec_plot <- spec_summary %>%
  group_by(cancer_group) %>%
  filter(row_number() %% 3 == 1) %>%
  ungroup()

fig2 <- ggplot(spec_plot, aes(x = wavenumber, y = mean_int, color = cancer_group)) +
  geom_ribbon(aes(ymin = mean_int - sem, ymax = mean_int + sem,
                  fill = cancer_group), alpha = 0.12, linewidth = 0) +
  geom_line(linewidth = 0.55, alpha = 0.9) +
  scale_color_manual(values = group_colors, name = NULL) +
  scale_fill_manual(values = group_colors, guide = "none") +
  scale_x_continuous(
    breaks = seq(400, 2200, 200),
    labels = seq(400, 2200, 200),
    expand = c(0.01, 0)
  ) +
  labs(
    title    = "Mean SERS Spectra by Cancer Type",
    subtitle = "SNV-normalized, baseline-corrected | shaded region = SEM",
    x = expression("Raman Shift (cm"^{-1}*")"),
    y = "Intensity (a.u.)"
  ) +
  theme(
    legend.position  = "top",
    legend.direction = "horizontal",
    legend.justification = "left"
  ) +
  guides(color = guide_legend(nrow = 1, override.aes = list(linewidth = 1.5)))

ggsave(file.path(out_dir, "fig2_sers_spectra.png"), fig2,
       width = 16, height = 9, dpi = 150)
cat("   Saved fig2_sers_spectra.pdf/png\n")


# =============================================================================
# Figure 3 — AI Model Performance (Stage 1 + Stage 2 per-class)
# =============================================================================
cat(">> Figure 3: AI Model Performance...\n")

# --- Panel A: Stage 1 (Cancer Screening) metrics ---
s1_metrics <- tibble(
  Metric = c("AUC", "Sensitivity", "Specificity", "F1 Score"),
  Value  = c(0.9807, 0.9390, 0.9215, 0.9502)
) %>%
  mutate(Metric = factor(Metric, levels = rev(Metric)))

panel_a <- ggplot(s1_metrics, aes(x = Metric, y = Value)) +
  geom_col(fill = "#2c3e50", width = 0.6, alpha = 0.85) +
  geom_text(aes(label = sprintf("%.3f", Value)),
            hjust = -0.15, size = 7, fontface = "bold", color = "#2c3e50") +
  coord_flip(ylim = c(0, 1.08)) +
  scale_y_continuous(breaks = seq(0, 1, 0.2), labels = scales::number_format(accuracy = 0.1)) +
  labs(title = "A. Cancer Screening (Stage 1)",
       subtitle = "Binary: Cancer vs Non-Cancer | 5-fold CV",
       x = NULL, y = NULL) +
  theme(panel.grid.major.y = element_blank())

# --- Panel B: Stage 2 per-class performance ---
# Compute per-class accuracy from confusion_pairs
conf <- read_csv(confpairs, show_col_types = FALSE)

# Total per true class
class_totals <- conf %>%
  select(true_class, true_total) %>%
  distinct()

# Sum of misclassified per true_class
misclassified <- conf %>%
  group_by(true_class) %>%
  summarise(n_wrong = sum(count), .groups = "drop")

per_class <- class_totals %>%
  left_join(misclassified, by = "true_class") %>%
  mutate(
    n_wrong  = replace_na(n_wrong, 0),
    n_correct = true_total - n_wrong,
    sensitivity = n_correct / true_total
  ) %>%
  # Map to poster names, keep only demographics groups
  mutate(cancer_group = case_when(
    true_class == "PRO"  ~ "PRC",
    true_class == "OVA"  ~ "OVC",
    true_class == "LUN"  ~ "LC",
    true_class == "CPAN" ~ "PAC",
    true_class == "CRC"  ~ "CRC",
    TRUE ~ NA_character_
  )) %>%
  filter(!is.na(cancer_group)) %>%
  mutate(cancer_group = factor(cancer_group, levels = rev(group_order[-1])))

panel_b <- ggplot(per_class, aes(x = cancer_group, y = sensitivity, fill = cancer_group)) +
  geom_col(width = 0.6, alpha = 0.85, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.1f%%", sensitivity * 100)),
            hjust = -0.15, size = 7, fontface = "bold", color = "#2c3e50") +
  coord_flip(ylim = c(0, 1.12)) +
  scale_fill_manual(values = group_colors) +
  scale_y_continuous(breaks = seq(0, 1, 0.2),
                     labels = scales::percent_format(accuracy = 1)) +
  labs(title = "B. Cancer Type Identification (Stage 2)",
       subtitle = "Per-class sensitivity | 5-fold CV",
       x = NULL, y = NULL) +
  theme(panel.grid.major.y = element_blank())

fig3 <- panel_a / panel_b + plot_layout(heights = c(1, 1.2))

ggsave(file.path(out_dir, "fig3_model_performance.png"), fig3,
       width = 12, height = 13, dpi = 150)
cat("   Saved fig3_model_performance.pdf/png\n")


# =============================================================================
# NOTE: Figure 4 (Confusion Matrix) is now generated by fig4_confusion_matrices.py
# =============================================================================

cat("\n>> All figures saved to: ", out_dir, "\n")
cat("   - fig2_sers_spectra.pdf/png\n")
cat("   - fig3_model_performance.pdf/png\n")
cat("   - fig4_confusion_matrices.png (generated by Python script)\n")
