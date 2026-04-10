#!/usr/bin/env Rscript
# =============================================================================
# SERS QC Diagnostic Dashboard — BLC Focus
# Beautiful R visualizations for QC analysis
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(patchwork)
  library(ggridges)
  library(scales)
  library(viridis)
  library(ggrepel)
  library(showtext)
})

# --- Fonts ---
font_add_google("Noto Sans KR", "noto")
font_add_google("Inter", "inter")
showtext_auto()

# --- Config ---
BASE_DIR   <- "/home/user/SERS-AI"
OUT_DIR    <- file.path(BASE_DIR, "results", "qc_diagnostic")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

RSD_THRESH  <- 5.0
CORR_THRESH <- 0.95

# --- Color palette (cancer-type aware, publication quality) ---
cancer_colors <- c(
  "PRO" = "#2563EB",   # blue
  "BRE" = "#DB2777",   # pink
  "OVA" = "#7C3AED",   # violet
  "LUN" = "#059669",   # emerald
  "CRC" = "#D97706",   # amber
  "PAN" = "#DC2626",   # red (CPAN+YPAN+SPAN)
  "CPAN" = "#EF4444",  # red-400
  "YPAN" = "#F87171",  # red-300
  "SPAN" = "#FCA5A5",  # red-200
  "BLC" = "#0D9488",   # teal
  "NOR" = "#6B7280",   # gray
  "YNOR" = "#9CA3AF",  # gray-400
  "H.D." = "#A3A3A3",  # neutral
  "HBP" = "#78716C",   # stone
  "DIA" = "#B45309"    # amber-dark
)

# Cancer vs non-cancer grouping
cancer_types <- c("PRO", "BRE", "OVA", "LUN", "CRC", "CPAN", "YPAN", "SPAN", "BLC")
noncancer_types <- c("NOR", "YNOR", "H.D.", "HBP", "DIA")

# --- Theme ---
theme_sers <- function(base_size = 13) {
  theme_minimal(base_size = base_size, base_family = "inter") +
    theme(
      plot.title = element_text(face = "bold", size = base_size + 4,
                                margin = margin(b = 8), color = "#1a1a2e"),
      plot.subtitle = element_text(size = base_size, color = "#64748b",
                                   margin = margin(b = 12)),
      plot.caption = element_text(size = base_size - 3, color = "#94a3b8",
                                  hjust = 0, margin = margin(t = 12)),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "#f1f5f9", linewidth = 0.4),
      axis.title = element_text(size = base_size, color = "#475569"),
      axis.text = element_text(size = base_size - 1, color = "#64748b"),
      legend.position = "right",
      legend.title = element_text(face = "bold", size = base_size - 1),
      legend.text = element_text(size = base_size - 2),
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      strip.text = element_text(face = "bold", size = base_size)
    )
}

# =============================================================================
# Load Data
# =============================================================================
cat("Loading data...\n")

qc_stats <- read_csv(file.path(BASE_DIR, "results", "qc_stats.csv"),
                      show_col_types = FALSE) %>%
  mutate(
    qc_pass = (mean_rsd <= RSD_THRESH) & (mean_corr >= CORR_THRESH),
    category = ifelse(group %in% cancer_types, "Cancer", "Non-Cancer"),
    is_blc = (group == "BLC")
  )

qc_failures <- read_csv(file.path(BASE_DIR, "results", "qc_failures.csv"),
                         show_col_types = FALSE)

intensity_gate <- read_csv(file.path(BASE_DIR, "results", "qc_intensity_gate.csv"),
                           show_col_types = FALSE)

# Load a subset of spectra for BLC spectral overlay
spectra_raw <- read_csv(file.path(BASE_DIR, "results", "processed_spectra.csv"),
                        show_col_types = FALSE, n_max = 10000)

# =============================================================================
# FIGURE 1: QC Failure Rate by Group (Bar + Annotation)
# =============================================================================
cat("Figure 1: Failure rates...\n")

fail_summary <- qc_stats %>%
  group_by(group) %>%
  summarise(
    total = n(),
    fail = sum(!qc_pass),
    pass = sum(qc_pass),
    fail_rate = fail / total * 100,
    .groups = "drop"
  ) %>%
  mutate(
    group = factor(group, levels = group[order(fail_rate)]),
    category = ifelse(group %in% cancer_types, "Cancer", "Non-Cancer")
  )

p1 <- ggplot(fail_summary, aes(x = group, y = fail_rate, fill = as.character(group))) +
  geom_col(width = 0.7, show.legend = FALSE) +
  geom_hline(yintercept = 50, linetype = "dashed", color = "#ef4444", alpha = 0.6) +
  geom_text(aes(label = sprintf("%.0f%%\n(%d/%d)", fail_rate, fail, total)),
            hjust = -0.1, size = 3.2, color = "#475569", family = "inter") +
  scale_fill_manual(values = cancer_colors) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.25)),
                     labels = label_percent(scale = 1)) +
  coord_flip() +
  labs(
    title = "QC Failure Rate by Cancer Type",
    subtitle = sprintf("Thresholds: RSD < %.0f%%, Correlation > %.2f  |  Overall: %d/%d (%.1f%%) fail",
                       RSD_THRESH, CORR_THRESH,
                       sum(fail_summary$fail), sum(fail_summary$total),
                       sum(fail_summary$fail) / sum(fail_summary$total) * 100),
    x = NULL, y = "Failure Rate",
    caption = "Red dashed line = 50% threshold. BLC shows 90%+ failure rate."
  ) +
  theme_sers() +
  theme(panel.grid.major.y = element_blank())

ggsave(file.path(OUT_DIR, "01_failure_rate_by_group.png"), p1,
       width = 10, height = 7, dpi = 300, bg = "white")

# =============================================================================
# FIGURE 2: RSD Distribution Ridge Plot (all groups)
# =============================================================================
cat("Figure 2: RSD ridge plot...\n")

group_order <- fail_summary %>% arrange(fail_rate) %>% pull(group) %>% as.character()
qc_stats$group_f <- factor(qc_stats$group, levels = group_order)

p2 <- ggplot(qc_stats, aes(x = mean_rsd, y = group_f, fill = group)) +
  geom_density_ridges(
    alpha = 0.7, scale = 1.8, rel_min_height = 0.005,
    quantile_lines = TRUE, quantiles = 2,
    show.legend = FALSE
  ) +
  geom_vline(xintercept = RSD_THRESH, linetype = "dashed", color = "#ef4444", linewidth = 0.8) +
  scale_fill_manual(values = cancer_colors) +
  scale_x_continuous(limits = c(0, 15), labels = function(x) paste0(x, "%")) +
  annotate("text", x = RSD_THRESH + 0.3, y = Inf, label = sprintf("Threshold = %g%%", RSD_THRESH),
           hjust = 0, vjust = 1.5, size = 3.5, color = "#ef4444", fontface = "bold", family = "inter") +
  labs(
    title = "Replicate RSD Distribution by Group",
    subtitle = "Ridge density with median line  |  BLC distribution shifted far right",
    x = "Mean RSD (%)", y = NULL,
    caption = "RSD = (SD / max intensity) x 100  |  Fingerprint region 400-2200 cm\u207b\u00b9"
  ) +
  theme_sers() +
  theme(panel.grid.major.y = element_blank())

ggsave(file.path(OUT_DIR, "02_rsd_ridge_plot.png"), p2,
       width = 10, height = 8, dpi = 300, bg = "white")

# =============================================================================
# FIGURE 3: Correlation Distribution Ridge Plot
# =============================================================================
cat("Figure 3: Correlation ridge plot...\n")

p3 <- ggplot(qc_stats, aes(x = mean_corr, y = group_f, fill = group)) +
  geom_density_ridges(
    alpha = 0.7, scale = 1.8, rel_min_height = 0.005,
    quantile_lines = TRUE, quantiles = 2,
    show.legend = FALSE
  ) +
  geom_vline(xintercept = CORR_THRESH, linetype = "dashed", color = "#ef4444", linewidth = 0.8) +
  scale_fill_manual(values = cancer_colors) +
  scale_x_continuous(limits = c(0.6, 1.0)) +
  annotate("text", x = CORR_THRESH - 0.01, y = Inf,
           label = sprintf("Threshold = %.2f", CORR_THRESH),
           hjust = 1, vjust = 1.5, size = 3.5, color = "#ef4444", fontface = "bold", family = "inter") +
  labs(
    title = "Replicate Correlation Distribution by Group",
    subtitle = "Pairwise Pearson correlation  |  BLC spans the widest range",
    x = "Mean Correlation", y = NULL,
    caption = "Correlation computed on preprocessed spectra (baseline-corrected, normalized)"
  ) +
  theme_sers() +
  theme(panel.grid.major.y = element_blank())

ggsave(file.path(OUT_DIR, "03_corr_ridge_plot.png"), p3,
       width = 10, height = 8, dpi = 300, bg = "white")

# =============================================================================
# FIGURE 4: RSD vs Correlation Scatter (BLC highlighted)
# =============================================================================
cat("Figure 4: RSD vs Correlation scatter...\n")

p4 <- ggplot(qc_stats, aes(x = mean_rsd, y = mean_corr)) +
  # Background: all non-BLC
  geom_point(data = filter(qc_stats, group != "BLC"),
             aes(color = group), alpha = 0.35, size = 1.5) +
  # Foreground: BLC
  geom_point(data = filter(qc_stats, group == "BLC"),
             color = "#0D9488", alpha = 0.7, size = 2.2, shape = 17) +
  # Threshold lines
  geom_vline(xintercept = RSD_THRESH, linetype = "dashed", color = "#ef4444", alpha = 0.7) +
  geom_hline(yintercept = CORR_THRESH, linetype = "dashed", color = "#ef4444", alpha = 0.7) +
  # Quadrant labels
  annotate("rect", xmin = 0, xmax = RSD_THRESH, ymin = CORR_THRESH, ymax = 1,
           fill = "#22c55e", alpha = 0.06) +
  annotate("rect", xmin = RSD_THRESH, xmax = 15, ymin = 0.5, ymax = CORR_THRESH,
           fill = "#ef4444", alpha = 0.06) +
  annotate("text", x = 1.5, y = 0.99, label = "PASS", color = "#16a34a",
           fontface = "bold", size = 4, family = "inter") +
  annotate("text", x = 12, y = 0.65, label = "FAIL\n(both)", color = "#dc2626",
           fontface = "bold", size = 4, family = "inter") +
  scale_color_manual(values = cancer_colors) +
  scale_x_continuous(limits = c(0, 15), labels = function(x) paste0(x, "%")) +
  scale_y_continuous(limits = c(0.5, 1.0)) +
  labs(
    title = "RSD vs Correlation: BLC Samples (triangles) vs Others",
    subtitle = "Most BLC samples cluster in the high-RSD / low-correlation region",
    x = "Mean RSD (%)", y = "Mean Correlation",
    color = "Group",
    caption = "Green zone = QC pass  |  BLC shown as teal triangles"
  ) +
  theme_sers() +
  guides(color = guide_legend(override.aes = list(alpha = 1, size = 3)))

ggsave(file.path(OUT_DIR, "04_rsd_vs_corr_scatter.png"), p4,
       width = 11, height = 8, dpi = 300, bg = "white")

# =============================================================================
# FIGURE 5: BLC vs All Others — Boxplot Comparison
# =============================================================================
cat("Figure 5: BLC vs others boxplot...\n")

qc_compare <- qc_stats %>%
  mutate(highlight = ifelse(group == "BLC", "BLC", "Others"))

p5a <- ggplot(qc_compare, aes(x = highlight, y = mean_rsd, fill = highlight)) +
  geom_boxplot(width = 0.5, outlier.alpha = 0.3, alpha = 0.8) +
  geom_jitter(width = 0.15, alpha = 0.15, size = 0.8) +
  geom_hline(yintercept = RSD_THRESH, linetype = "dashed", color = "#ef4444") +
  scale_fill_manual(values = c("BLC" = "#0D9488", "Others" = "#94a3b8")) +
  labs(title = "RSD", x = NULL, y = "Mean RSD (%)") +
  theme_sers(base_size = 12) +
  theme(legend.position = "none")

p5b <- ggplot(qc_compare, aes(x = highlight, y = mean_corr, fill = highlight)) +
  geom_boxplot(width = 0.5, outlier.alpha = 0.3, alpha = 0.8) +
  geom_jitter(width = 0.15, alpha = 0.15, size = 0.8) +
  geom_hline(yintercept = CORR_THRESH, linetype = "dashed", color = "#ef4444") +
  scale_fill_manual(values = c("BLC" = "#0D9488", "Others" = "#94a3b8")) +
  labs(title = "Correlation", x = NULL, y = "Mean Correlation") +
  theme_sers(base_size = 12) +
  theme(legend.position = "none")

p5 <- (p5a | p5b) +
  plot_annotation(
    title = "BLC vs All Other Groups: QC Metric Distributions",
    subtitle = "BLC median RSD and correlation are substantially worse than other groups",
    caption = "Red dashed = QC threshold  |  Jittered points show individual samples",
    theme = theme_sers()
  )

ggsave(file.path(OUT_DIR, "05_blc_vs_others_boxplot.png"), p5,
       width = 10, height = 6, dpi = 300, bg = "white")

# =============================================================================
# FIGURE 6: Failure Reason Breakdown (Stacked)
# =============================================================================
cat("Figure 6: Failure reasons...\n")

failure_reasons <- qc_failures %>%
  mutate(
    reason_cat = case_when(
      grepl("High RSD.*Low corr", failure_reason) ~ "Both (RSD + Corr)",
      grepl("High RSD", failure_reason)            ~ "RSD only",
      grepl("Low corr", failure_reason)             ~ "Correlation only",
      TRUE                                          ~ "Other"
    )
  ) %>%
  group_by(group, reason_cat) %>%
  summarise(n = n(), .groups = "drop") %>%
  left_join(fail_summary %>% select(group, total), by = c("group" = "group"))

# Order by failure rate
failure_reasons$group <- factor(failure_reasons$group, levels = group_order)

p6 <- ggplot(failure_reasons, aes(x = group, y = n, fill = reason_cat)) +
  geom_col(width = 0.7, alpha = 0.9) +
  scale_fill_manual(values = c(
    "RSD only" = "#f59e0b",
    "Correlation only" = "#6366f1",
    "Both (RSD + Corr)" = "#ef4444",
    "Other" = "#a3a3a3"
  )) +
  coord_flip() +
  labs(
    title = "QC Failure Reasons by Group",
    subtitle = "BLC failures are predominantly driven by correlation, not RSD alone",
    x = NULL, y = "Number of Failed Samples",
    fill = "Failure Reason",
    caption = "\"Both\" = sample fails both RSD and correlation thresholds simultaneously"
  ) +
  theme_sers() +
  theme(panel.grid.major.y = element_blank())

ggsave(file.path(OUT_DIR, "06_failure_reasons_stacked.png"), p6,
       width = 10, height = 7, dpi = 300, bg = "white")

# =============================================================================
# FIGURE 7: Intensity Gate — fp_mean distribution by group
# =============================================================================
cat("Figure 7: Intensity gate...\n")

ig_summary <- intensity_gate %>%
  group_by(group) %>%
  summarise(
    median_fp = median(fp_mean),
    gate_fail_rate = sum(!gate_pass) / n() * 100,
    .groups = "drop"
  )

p7 <- ggplot(intensity_gate, aes(x = fp_mean, y = factor(group, levels = group_order),
                                  fill = group)) +
  geom_density_ridges(alpha = 0.7, scale = 1.5, show.legend = FALSE,
                      rel_min_height = 0.005) +
  geom_vline(xintercept = unique(intensity_gate$gate_threshold)[1],
             linetype = "dashed", color = "#ef4444", linewidth = 0.6) +
  scale_fill_manual(values = cancer_colors) +
  scale_x_log10(labels = label_comma()) +
  labs(
    title = "Raw Intensity (fp_mean) Distribution by Group",
    subtitle = "Log scale  |  Intensity gate catches SERS enhancement failure",
    x = "Fingerprint Mean Intensity (log scale)", y = NULL,
    caption = "Red dashed = adaptive intensity gate threshold"
  ) +
  theme_sers() +
  theme(panel.grid.major.y = element_blank())

ggsave(file.path(OUT_DIR, "07_intensity_gate_ridge.png"), p7,
       width = 10, height = 8, dpi = 300, bg = "white")

# =============================================================================
# FIGURE 8: BLC Spectral Overlay — Pass vs Fail
# =============================================================================
cat("Figure 8: BLC spectral overlay...\n")

blc_spectra <- spectra_raw %>% filter(group == "BLC")

if (nrow(blc_spectra) > 0) {
  # Get pass/fail status
  blc_qc <- qc_stats %>%
    filter(group == "BLC") %>%
    select(sample_id, qc_pass)

  blc_long <- blc_spectra %>%
    left_join(blc_qc, by = "sample_id") %>%
    pivot_longer(cols = starts_with("x_"), names_to = "wavenumber", values_to = "intensity") %>%
    mutate(
      wavenumber = as.numeric(gsub("x_", "", wavenumber)),
      qc_status = ifelse(qc_pass, "QC Pass", "QC Fail")
    )

  # Sample max 20 spectra per group for readability
  set.seed(42)
  sample_ids_pass <- blc_qc %>% filter(qc_pass) %>% pull(sample_id)
  sample_ids_fail <- blc_qc %>% filter(!qc_pass) %>% pull(sample_id)

  sel_pass <- if(length(sample_ids_pass) > 10) sample(sample_ids_pass, 10) else sample_ids_pass
  sel_fail <- if(length(sample_ids_fail) > 10) sample(sample_ids_fail, 10) else sample_ids_fail

  blc_subset <- blc_long %>%
    filter(sample_id %in% c(sel_pass, sel_fail), replicate == 1)

  p8 <- ggplot(blc_subset, aes(x = wavenumber, y = intensity,
                                group = interaction(sample_id, replicate),
                                color = qc_status)) +
    geom_line(alpha = 0.5, linewidth = 0.4) +
    scale_color_manual(values = c("QC Pass" = "#22c55e", "QC Fail" = "#ef4444")) +
    scale_x_reverse() +
    labs(
      title = "BLC Spectra: QC Pass vs Fail (Rep 1)",
      subtitle = sprintf("10 pass + 10 fail samples shown  |  Pass: %d, Fail: %d total",
                         length(sample_ids_pass), length(sample_ids_fail)),
      x = expression("Wavenumber (cm"^-1*")"), y = "Intensity (a.u.)",
      color = "QC Status",
      caption = "Single replicate per sample shown for clarity"
    ) +
    theme_sers() +
    theme(legend.position = c(0.85, 0.85),
          legend.background = element_rect(fill = "white", color = "#e2e8f0"))

  ggsave(file.path(OUT_DIR, "08_blc_spectra_pass_vs_fail.png"), p8,
         width = 12, height = 6, dpi = 300, bg = "white")
}

# =============================================================================
# FIGURE 9: BLC Replicate Variance Heatmap (wavenumber-resolved)
# =============================================================================
cat("Figure 9: BLC wavenumber-resolved variance...\n")

if (nrow(blc_spectra) > 0) {
  # For a few BLC samples, compute per-wavenumber SD across replicates
  blc_var_samples <- blc_spectra %>%
    group_by(sample_id) %>%
    filter(n() >= 3) %>%  # need at least 3 reps
    ungroup()

  # Pick 5 pass and 5 fail
  sel_var_pass <- head(intersect(unique(blc_var_samples$sample_id), sample_ids_pass), 5)
  sel_var_fail <- head(intersect(unique(blc_var_samples$sample_id), sample_ids_fail), 5)

  blc_var_data <- blc_var_samples %>%
    filter(sample_id %in% c(sel_var_pass, sel_var_fail)) %>%
    left_join(blc_qc, by = "sample_id") %>%
    pivot_longer(cols = starts_with("x_"), names_to = "wavenumber", values_to = "intensity") %>%
    mutate(wavenumber = as.numeric(gsub("x_", "", wavenumber))) %>%
    group_by(sample_id, qc_pass, wavenumber) %>%
    summarise(sd_intensity = sd(intensity, na.rm = TRUE), .groups = "drop") %>%
    mutate(
      qc_label = ifelse(qc_pass, "Pass", "Fail"),
      sample_label = paste0(ifelse(qc_pass, "P", "F"), "_", sample_id)
    )

  p9 <- ggplot(blc_var_data, aes(x = wavenumber, y = sample_label, fill = sd_intensity)) +
    geom_tile() +
    scale_fill_viridis(option = "inferno", name = "SD") +
    scale_x_reverse() +
    facet_grid(qc_label ~ ., scales = "free_y", space = "free_y") +
    labs(
      title = "BLC Per-Wavenumber Replicate Variance",
      subtitle = "Pass vs Fail samples  |  Which spectral regions drive variability?",
      x = expression("Wavenumber (cm"^-1*")"), y = NULL,
      caption = "Color = SD of intensity across replicates  |  Inferno colormap"
    ) +
    theme_sers(base_size = 11) +
    theme(
      axis.text.y = element_text(size = 9),
      strip.text = element_text(face = "bold", size = 12)
    )

  ggsave(file.path(OUT_DIR, "09_blc_wavenumber_variance_heatmap.png"), p9,
         width = 14, height = 6, dpi = 300, bg = "white")
}

# =============================================================================
# FIGURE 10: Threshold Sensitivity — How failure rate changes
# =============================================================================
cat("Figure 10: Threshold sensitivity...\n")

rsd_range  <- seq(2, 15, by = 0.5)
corr_range <- seq(0.70, 0.99, by = 0.01)

# RSD sensitivity (fix corr at 0.95)
rsd_sens <- tibble(rsd_thresh = rsd_range) %>%
  rowwise() %>%
  mutate(
    blc_fail = sum(qc_stats$group == "BLC" &
                     (qc_stats$mean_rsd > rsd_thresh | qc_stats$mean_corr < CORR_THRESH)) /
               sum(qc_stats$group == "BLC") * 100,
    all_fail = sum(qc_stats$mean_rsd > rsd_thresh | qc_stats$mean_corr < CORR_THRESH) /
               nrow(qc_stats) * 100
  ) %>% ungroup()

# Corr sensitivity (fix RSD at 5)
corr_sens <- tibble(corr_thresh = corr_range) %>%
  rowwise() %>%
  mutate(
    blc_fail = sum(qc_stats$group == "BLC" &
                     (qc_stats$mean_rsd > RSD_THRESH | qc_stats$mean_corr < corr_thresh)) /
               sum(qc_stats$group == "BLC") * 100,
    all_fail = sum(qc_stats$mean_rsd > RSD_THRESH | qc_stats$mean_corr < corr_thresh) /
               nrow(qc_stats) * 100
  ) %>% ungroup()

p10a <- ggplot(rsd_sens) +
  geom_line(aes(x = rsd_thresh, y = blc_fail, color = "BLC"), linewidth = 1.2) +
  geom_line(aes(x = rsd_thresh, y = all_fail, color = "All Groups"), linewidth = 1.2) +
  geom_vline(xintercept = RSD_THRESH, linetype = "dashed", color = "#94a3b8") +
  scale_color_manual(values = c("BLC" = "#0D9488", "All Groups" = "#6366f1")) +
  scale_x_continuous(labels = function(x) paste0(x, "%")) +
  scale_y_continuous(labels = function(y) paste0(y, "%")) +
  labs(title = "RSD Threshold Sensitivity", subtitle = "Corr fixed at 0.95",
       x = "RSD Threshold", y = "Failure Rate", color = NULL) +
  theme_sers(base_size = 11) +
  theme(legend.position = c(0.8, 0.8))

p10b <- ggplot(corr_sens) +
  geom_line(aes(x = corr_thresh, y = blc_fail, color = "BLC"), linewidth = 1.2) +
  geom_line(aes(x = corr_thresh, y = all_fail, color = "All Groups"), linewidth = 1.2) +
  geom_vline(xintercept = CORR_THRESH, linetype = "dashed", color = "#94a3b8") +
  scale_color_manual(values = c("BLC" = "#0D9488", "All Groups" = "#6366f1")) +
  scale_y_continuous(labels = function(y) paste0(y, "%")) +
  labs(title = "Correlation Threshold Sensitivity", subtitle = "RSD fixed at 5%",
       x = "Correlation Threshold", y = "Failure Rate", color = NULL) +
  theme_sers(base_size = 11) +
  theme(legend.position = c(0.2, 0.8))

p10 <- (p10a | p10b) +
  plot_annotation(
    title = "QC Threshold Sensitivity Analysis",
    subtitle = "BLC failure rate remains high even with very relaxed thresholds",
    caption = "Gray dashed = current threshold  |  BLC curve stays above 60% regardless of RSD relaxation",
    theme = theme_sers()
  )

ggsave(file.path(OUT_DIR, "10_threshold_sensitivity.png"), p10,
       width = 12, height = 6, dpi = 300, bg = "white")

# =============================================================================
# FIGURE 11: Per-Group QC Metric Summary (Heatmap-style tile)
# =============================================================================
cat("Figure 11: Summary heatmap...\n")

group_summary <- qc_stats %>%
  group_by(group) %>%
  summarise(
    n = n(),
    median_rsd = median(mean_rsd),
    median_corr = median(mean_corr),
    q75_rsd = quantile(mean_rsd, 0.75),
    q25_corr = quantile(mean_corr, 0.25),
    fail_rate = sum(!qc_pass) / n() * 100,
    .groups = "drop"
  ) %>%
  mutate(group = factor(group, levels = group_order))

gs_long <- group_summary %>%
  select(group, median_rsd, median_corr, fail_rate) %>%
  pivot_longer(-group, names_to = "metric", values_to = "value") %>%
  mutate(
    metric_label = case_when(
      metric == "median_rsd" ~ "Median RSD (%)",
      metric == "median_corr" ~ "Median Corr",
      metric == "fail_rate" ~ "Failure Rate (%)"
    ),
    # Normalize for color
    norm_val = case_when(
      metric == "median_rsd" ~ scales::rescale(value, from = c(0, 10)),
      metric == "median_corr" ~ scales::rescale(1 - value, from = c(0, 0.3)),  # invert
      metric == "fail_rate" ~ value / 100
    )
  )

p11 <- ggplot(gs_long, aes(x = metric_label, y = group, fill = norm_val)) +
  geom_tile(color = "white", linewidth = 1.5) +
  geom_text(aes(label = case_when(
    metric == "median_rsd" ~ sprintf("%.1f", value),
    metric == "median_corr" ~ sprintf("%.3f", value),
    metric == "fail_rate" ~ sprintf("%.0f%%", value)
  )), size = 3.8, fontface = "bold", color = ifelse(gs_long$norm_val > 0.5, "white", "#1e293b"),
  family = "inter") +
  scale_fill_gradient2(low = "#22c55e", mid = "#fbbf24", high = "#ef4444",
                       midpoint = 0.4, limits = c(0, 1), guide = "none") +
  labs(
    title = "QC Summary by Group",
    subtitle = "Green = good  |  Red = poor quality  |  BLC stands out across all metrics",
    x = NULL, y = NULL
  ) +
  theme_sers(base_size = 12) +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(face = "bold", size = 12)
  )

ggsave(file.path(OUT_DIR, "11_qc_summary_heatmap.png"), p11,
       width = 8, height = 7, dpi = 300, bg = "white")

# =============================================================================
# Summary stats
# =============================================================================
cat("\n=== QC DIAGNOSTIC SUMMARY ===\n")
cat(sprintf("Total samples: %d\n", nrow(qc_stats)))
cat(sprintf("Total failures: %d (%.1f%%)\n", sum(!qc_stats$qc_pass),
            sum(!qc_stats$qc_pass)/nrow(qc_stats)*100))
cat("\nPer-group failure rates:\n")
fail_summary %>% arrange(desc(fail_rate)) %>%
  mutate(line = sprintf("  %-6s %3d/%3d = %5.1f%%", group, fail, total, fail_rate)) %>%
  pull(line) %>% cat(sep = "\n")

cat(sprintf("\n\nBLC specifics:\n"))
blc_stats <- qc_stats %>% filter(group == "BLC")
cat(sprintf("  Median RSD:  %.2f%% (others: %.2f%%)\n",
            median(blc_stats$mean_rsd),
            median(filter(qc_stats, group != "BLC")$mean_rsd)))
cat(sprintf("  Median Corr: %.4f (others: %.4f)\n",
            median(blc_stats$mean_corr),
            median(filter(qc_stats, group != "BLC")$mean_corr)))

cat(sprintf("\nFigures saved to: %s\n", OUT_DIR))
cat("Done!\n")
