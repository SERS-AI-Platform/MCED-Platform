#!/usr/bin/env Rscript

# Plot baseline-correction validation results as individual R figures.
#
# Each metric is exported in two forms:
#   *_titled.* : internal/review version with title, subtitle, and caption
#   *_paper.*  : paper-ready version without title/subtitle/caption
#
# The underlying experiment is a paired preprocessing ablation that keeps the
# downstream evaluator fixed as subject-aggregated logistic regression.

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
})

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
project_root <- Sys.getenv(
  "SERS_PROJECT_ROOT",
  unset = normalizePath(file.path(dirname(script_path), "..", ".."), mustWork = TRUE)
)
summary_path <- file.path(
  project_root,
  "results",
  "baseline_method_validation",
  "baseline_method_validation_summary.csv"
)
out_dir <- file.path(
  project_root,
  "results",
  "figures",
  "baseline_method_validation_r"
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

df <- read_csv(summary_path, show_col_types = FALSE) %>%
  mutate(
    method_label = factor(
      method,
      levels = c("rolling_minimum", "asls_1e6_p001", "asls_1e7_p0001", "arpls_1e6"),
      labels = c("Rolling min\nw=101", "AsLS\n1e6 / 0.01", "AsLS\n1e7 / 0.001", "arPLS\n1e6")
    ),
    method_short = factor(
      method,
      levels = c("rolling_minimum", "asls_1e6_p001", "asls_1e7_p0001", "arpls_1e6"),
      labels = c("Rolling", "AsLS-1e6", "AsLS-1e7", "arPLS")
    )
  )

method_colors <- c(
  "Rolling min\nw=101" = "#4E79A7",
  "AsLS\n1e6 / 0.01" = "#F28E2B",
  "AsLS\n1e7 / 0.001" = "#B07AA1",
  "arPLS\n1e6" = "#59A14F"
)

theme_metric <- function() {
  theme_bw(base_size = 11) +
    theme(
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "#333333", linewidth = 0.45),
      axis.text.x = element_text(size = 10),
      axis.text.y = element_text(size = 10),
      axis.title = element_text(size = 11),
      plot.title = element_text(size = 11.5, face = "bold", hjust = 0),
      plot.subtitle = element_text(size = 9.2, color = "#555555", hjust = 0),
      plot.caption = element_text(size = 7.4, color = "#777777", hjust = 0),
      legend.position = "none"
    )
}

save_plot <- function(plot, stem, width = 4.8, height = 3.6) {
  png_path <- file.path(out_dir, paste0(stem, ".png"))
  pdf_path <- file.path(out_dir, paste0(stem, ".pdf"))
  ggsave(png_path, plot, width = width, height = height, dpi = 320, bg = "white")
  ggsave(pdf_path, plot, width = width, height = height, bg = "white")
}

metric_plot <- function(
  data,
  y,
  ymin = NULL,
  ymax = NULL,
  y_label,
  title,
  subtitle,
  caption,
  ylim,
  value_digits = 3,
  paper = FALSE
) {
  y_sym <- rlang::sym(y)
  ymin_sym <- if (!is.null(ymin)) rlang::sym(ymin) else NULL
  ymax_sym <- if (!is.null(ymax)) rlang::sym(ymax) else NULL

  p <- ggplot(data, aes(x = method_label, y = !!y_sym, color = method_label)) +
    geom_hline(yintercept = 0, color = "#DDDDDD", linewidth = 0.3) +
    geom_point(size = 3.2) +
    scale_color_manual(values = method_colors) +
    scale_y_continuous(limits = ylim, expand = expansion(mult = c(0.04, 0.08))) +
    labs(x = NULL, y = y_label) +
    theme_metric()

  if (!is.null(ymin_sym) && !is.null(ymax_sym)) {
    p <- p + geom_errorbar(aes(ymin = !!ymin_sym, ymax = !!ymax_sym), width = 0.16, linewidth = 0.75)
  }

  label_fmt <- paste0("%.", value_digits, "f")
  p <- p + geom_text(
    aes(label = sprintf(label_fmt, !!y_sym)),
    vjust = -1.05,
    size = 3.1,
    color = "#222222"
  )

  if (!paper) {
    p <- p + labs(title = title, subtitle = subtitle, caption = caption)
  }

  p
}

caption_common <- "Fixed grid 935 + v2 QC; fixed subject-level LR evaluator."

plot_specs <- list(
  list(
    stem = "01_screening_auc",
    y = "s1_auc_mean",
    ymin = "s1_auc_mean_minus_sd",
    ymax = "s1_auc_mean_plus_sd",
    y_label = "Screening AUC (mean +/- SD)",
    title = "Screening AUC",
    subtitle = "Baseline correction ablation with fixed subject-level LR evaluator",
    caption = caption_common,
    ylim = c(0.94, 0.98),
    digits = 3
  ),
  list(
    stem = "02_cancer_type_macro_f1",
    y = "s2_type_macro_f1_mean",
    ymin = "s2_type_macro_f1_mean_minus_sd",
    ymax = "s2_type_macro_f1_mean_plus_sd",
    y_label = "Cancer-type macro-F1 (mean +/- SD)",
    title = "Cancer-Type Macro-F1",
    subtitle = "Baseline correction ablation with fixed subject-level LR evaluator",
    caption = caption_common,
    ylim = c(0.78, 0.89),
    digits = 3
  ),
  list(
    stem = "03_qc_subject_availability",
    y = "stage2_final_subject_retention_pct",
    ymin = NULL,
    ymax = NULL,
    y_label = "Subjects available after fixed v2 QC (%)",
    title = "Subject Availability After Fixed QC",
    subtitle = "Subjects remaining after applying the same v2 QC rule",
    caption = "Operational metric, not a primary accuracy metric.",
    ylim = c(88, 99),
    digits = 1
  ),
  list(
    stem = "04_corr_to_mean_p05",
    y = "corr_to_mean_p05",
    ymin = NULL,
    ymax = NULL,
    y_label = "Replicate corr-to-mean, 5th percentile",
    title = "Replicate Consistency Diagnostic",
    subtitle = "Corr-to-mean distribution before fixed v2 QC thresholding",
    caption = "Operational diagnostic, not a standalone accuracy metric.",
    ylim = c(0.89, 0.94),
    digits = 3
  ),
  list(
    stem = "05_runtime",
    y = "preprocess_time_s",
    ymin = NULL,
    ymax = NULL,
    y_label = "Preprocessing runtime (s)",
    title = "Runtime Cost",
    subtitle = "Same spectra and same preprocessing/QC context",
    caption = caption_common,
    ylim = c(0, 75),
    digits = 1
  )
)

df_plot <- df %>%
  mutate(
    s1_auc_mean_minus_sd = s1_auc_mean - s1_auc_std,
    s1_auc_mean_plus_sd = s1_auc_mean + s1_auc_std,
    s2_type_macro_f1_mean_minus_sd = s2_type_macro_f1_mean - s2_type_macro_f1_std,
    s2_type_macro_f1_mean_plus_sd = s2_type_macro_f1_mean + s2_type_macro_f1_std
  )

for (spec in plot_specs) {
  p_titled <- metric_plot(
    df_plot,
    y = spec$y,
    ymin = spec$ymin,
    ymax = spec$ymax,
    y_label = spec$y_label,
    title = spec$title,
    subtitle = spec$subtitle,
    caption = spec$caption,
    ylim = spec$ylim,
    value_digits = spec$digits,
    paper = FALSE
  )
  save_plot(p_titled, paste0(spec$stem, "_titled"), width = 5.8, height = 3.8)

  p_paper <- metric_plot(
    df_plot,
    y = spec$y,
    ymin = spec$ymin,
    ymax = spec$ymax,
    y_label = spec$y_label,
    title = spec$title,
    subtitle = spec$subtitle,
    caption = spec$caption,
    ylim = spec$ylim,
    value_digits = spec$digits,
    paper = TRUE
  )
  save_plot(p_paper, paste0(spec$stem, "_paper"), width = 4.8, height = 3.6)
}

write_csv(
  df %>%
    select(
      method,
      reference_family,
      stage2_final_subjects,
      stage2_final_subject_retention_pct,
      corr_to_mean_p05,
      preprocess_time_s,
      s1_auc_mean,
      s1_auc_std,
      s2_type_macro_f1_mean,
      s2_type_macro_f1_std
    ),
  file.path(out_dir, "baseline_method_validation_plot_values.csv")
)

cat("Wrote figures to", out_dir, "\n")
