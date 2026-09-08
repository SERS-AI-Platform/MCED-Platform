#!/usr/bin/env Rscript

# Replicate-level baseline-correction figures.
#
# These plots show whether baseline correction reduces replicate-specific
# broad background differences and produces more consistent replicate spectra.

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
data_dir <- file.path(project_root, "results", "baseline_replicate_effect")
out_dir <- file.path(project_root, "results", "figures", "baseline_replicate_effect_r")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

long_df <- read_csv(
  file.path(data_dir, "baseline_replicate_effect_long.csv"),
  show_col_types = FALSE
)
summary_df <- read_csv(
  file.path(data_dir, "baseline_replicate_effect_summary.csv"),
  show_col_types = FALSE
)

component_order <- c(
  "sg_smoothed_raw",
  "estimated_background",
  "baseline_corrected_raw",
  "baseline_corrected_snv"
)
component_labels <- c(
  "sg_smoothed_raw" = "Before baseline correction\nSG-smoothed, original intensity",
  "estimated_background" = "Estimated background\nrolling minimum",
  "baseline_corrected_raw" = "After baseline correction\nbefore SNV",
  "baseline_corrected_snv" = "After baseline correction + SNV\npipeline output scale"
)

long_df <- long_df %>%
  mutate(
    component = factor(component, levels = component_order),
    component_label = factor(component, levels = component_order, labels = component_labels[component_order]),
    replicate = factor(replicate)
  )

example_label <- sprintf(
  "%s %s, %d replicates",
  summary_df$group[1],
  summary_df$sample_id[1],
  summary_df$n_replicates[1]
)

metric_label <- sprintf(
  "Mean pairwise corr: %.3f -> %.3f -> %.3f",
  summary_df$raw_mean_pairwise_corr[1],
  summary_df$baseline_corrected_mean_pairwise_corr[1],
  summary_df$corrected_snv_mean_pairwise_corr[1]
)

theme_signal <- function() {
  theme_bw(base_size = 10.5) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "#EAEAEA", linewidth = 0.35),
      panel.border = element_rect(color = "#333333", linewidth = 0.45),
      axis.text = element_text(color = "#333333"),
      axis.title = element_text(color = "#111111"),
      plot.title = element_text(size = 12.5, face = "bold", hjust = 0),
      plot.subtitle = element_text(size = 9.2, color = "#555555", hjust = 0),
      plot.caption = element_text(size = 7.5, color = "#777777", hjust = 0),
      strip.background = element_rect(fill = "#F2F4F7", color = "#D0D5DD"),
      strip.text = element_text(size = 9.2, face = "bold"),
      legend.title = element_blank(),
      legend.position = "bottom"
    )
}

save_plot <- function(plot, stem, width = 6.2, height = 5.4) {
  ggsave(file.path(out_dir, paste0(stem, ".png")), plot, width = width, height = height, dpi = 320, bg = "white")
  ggsave(file.path(out_dir, paste0(stem, ".pdf")), plot, width = width, height = height, bg = "white")
}

make_replicate_plot <- function(paper = FALSE) {
  p <- ggplot(long_df, aes(x = wavenumber, y = intensity, color = replicate, group = replicate)) +
    geom_line(linewidth = 0.38, alpha = 0.92) +
    facet_wrap(~component_label, ncol = 1, scales = "free_y") +
    scale_color_brewer(palette = "Dark2") +
    labs(
      x = expression("Raman shift (cm"^-1*")"),
      y = "Intensity (a.u.)"
    ) +
    theme_signal()

  if (!paper) {
    p <- p +
      labs(
        title = "Baseline Correction: Replicate Consistency",
        subtitle = paste0(example_label, " | ", metric_label),
    caption = "Baseline correction precedes SNV, matching the preprocessing pipeline."
      )
  }
  p
}

make_corr_plot <- function(paper = FALSE) {
  plot_df <- tibble::tibble(
    stage = factor(
      c("Before baseline\ncorrection", "After baseline\ncorrection", "After baseline\ncorrection + SNV"),
      levels = c("Before baseline\ncorrection", "After baseline\ncorrection", "After baseline\ncorrection + SNV")
    ),
    mean_pairwise_corr = c(
      summary_df$raw_mean_pairwise_corr[1],
      summary_df$baseline_corrected_mean_pairwise_corr[1],
      summary_df$corrected_snv_mean_pairwise_corr[1]
    )
  )

  p <- ggplot(plot_df, aes(x = stage, y = mean_pairwise_corr, fill = stage)) +
    geom_col(width = 0.56, color = "#333333", linewidth = 0.3) +
    geom_text(aes(label = sprintf("%.3f", mean_pairwise_corr)), vjust = -0.55, size = 3.4) +
    scale_fill_manual(values = c("#9E9E9E", "#76B7B2", "#4E79A7")) +
    scale_y_continuous(limits = c(0, 1.05), expand = expansion(mult = c(0, 0.02))) +
    labs(
      x = NULL,
      y = "Mean pairwise replicate correlation"
    ) +
    theme_signal() +
    theme(legend.position = "none")

  if (!paper) {
    p <- p +
      labs(
        title = "Replicate Similarity",
        subtitle = example_label,
        caption = "Diagnostic summary for the selected representative subject."
      )
  }
  p
}

save_plot(make_replicate_plot(FALSE), "01_replicate_consistency_titled", width = 6.4, height = 7.1)
save_plot(make_replicate_plot(TRUE), "01_replicate_consistency_paper", width = 5.8, height = 6.4)

save_plot(make_corr_plot(FALSE), "02_replicate_correlation_titled", width = 5.6, height = 3.9)
save_plot(make_corr_plot(TRUE), "02_replicate_correlation_paper", width = 5.0, height = 3.5)

write_excel_csv(summary_df, file.path(out_dir, "baseline_replicate_effect_plot_values.csv"))

cat("Wrote replicate-level baseline-correction figures to", out_dir, "\n")
