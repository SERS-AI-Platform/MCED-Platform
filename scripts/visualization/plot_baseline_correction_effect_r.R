#!/usr/bin/env Rscript

# Signal-level baseline-correction figures.
#
# These plots explain the purpose of baseline correction itself:
# broad background estimation/subtraction and preservation of narrower
# Raman/SERS peak structure. They are intentionally not performance figures.

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
data_dir <- file.path(project_root, "results", "baseline_correction_effect")
out_dir <- file.path(project_root, "results", "figures", "baseline_correction_effect_r")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

long_df <- read_csv(
  file.path(data_dir, "baseline_correction_effect_example_long.csv"),
  show_col_types = FALSE
)
meta <- read_csv(
  file.path(data_dir, "baseline_correction_effect_example_metadata.csv"),
  show_col_types = FALSE
)
method_summary <- read_csv(
  file.path(data_dir, "baseline_correction_effect_method_summary.csv"),
  show_col_types = FALSE
)

method_order <- c("rolling_minimum", "asls_1e6", "arpls_1e6")
method_labels <- c(
  "rolling_minimum" = "Rolling minimum\nw=101",
  "asls_1e6" = "AsLS\nlam=1e6, p=0.01",
  "arpls_1e6" = "arPLS\nlam=1e6"
)
method_colors <- c(
  "rolling_minimum" = "#4E79A7",
  "asls_1e6" = "#F28E2B",
  "arpls_1e6" = "#59A14F"
)

long_df <- long_df %>%
  mutate(
    method = factor(method, levels = c("raw_native", "raw_smoothed", method_order)),
    method_plot = factor(method, levels = method_order, labels = method_labels[method_order])
  )

example_label <- sprintf(
  "%s %s, replicate %s",
  meta$group[1],
  meta$sample_id[1],
  meta$replicate[1]
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
      legend.title = element_blank(),
      legend.position = "bottom"
    )
}

save_plot <- function(plot, stem, width = 5.8, height = 3.7) {
  ggsave(file.path(out_dir, paste0(stem, ".png")), plot, width = width, height = height, dpi = 320, bg = "white")
  ggsave(file.path(out_dir, paste0(stem, ".pdf")), plot, width = width, height = height, bg = "white")
}

make_background_plot <- function(paper = FALSE) {
  raw_native_df <- long_df %>%
    filter(component == "raw_native")
  raw_smooth_df <- long_df %>%
    filter(component == "raw_smoothed")
  baseline_df <- long_df %>%
    filter(component == "estimated_baseline") %>%
    mutate(method_plot = factor(method, levels = method_order, labels = method_labels[method_order]))

  p <- ggplot() +
    geom_line(
      data = raw_native_df,
      aes(x = wavenumber, y = intensity, color = "Raw\n(interp.)"),
      linewidth = 0.28,
      alpha = 0.65
    ) +
    geom_line(
      data = raw_smooth_df,
      aes(x = wavenumber, y = intensity, color = "SG-smoothed\ninput"),
      linewidth = 0.48
    ) +
    geom_line(
      data = baseline_df,
      aes(x = wavenumber, y = intensity, color = method),
      linewidth = 0.75
    ) +
    scale_color_manual(
      values = c(
        "Raw\n(interp.)" = "#9E9E9E",
        "SG-smoothed\ninput" = "#222222",
        method_colors
      ),
      breaks = c("Raw\n(interp.)", "SG-smoothed\ninput", method_order),
      labels = c(
        "Raw\n(interp.)",
        "SG-smoothed\ninput",
        method_labels[method_order]
      )
    ) +
    guides(color = guide_legend(nrow = 2, byrow = TRUE)) +
    labs(
      x = expression("Raman shift (cm"^-1*")"),
      y = "Intensity (a.u.)",
      color = NULL
    ) +
    theme_signal()

  if (!paper) {
    p <- p +
      labs(
        title = "Baseline Correction: Background Estimation",
        subtitle = paste0("Representative spectrum with estimated broad background (", example_label, ")"),
        caption = "Original intensity scale; Savitzky-Golay smoothing precedes baseline estimation."
      )
  }
  p
}

make_corrected_plot <- function(paper = FALSE) {
  corrected_df <- long_df %>%
    filter(component == "corrected_snv") %>%
    mutate(method_plot = factor(method, levels = method_order, labels = method_labels[method_order]))

  p <- ggplot(corrected_df, aes(x = wavenumber, y = intensity, color = method)) +
    geom_line(linewidth = 0.45) +
    facet_wrap(~method_plot, ncol = 1) +
    scale_color_manual(values = method_colors, breaks = method_order, labels = method_labels[method_order]) +
    labs(
      x = expression("Raman shift (cm"^-1*")"),
      y = "Corrected intensity after SNV"
    ) +
    theme_signal() +
    theme(
      strip.background = element_rect(fill = "#F2F4F7", color = "#D0D5DD"),
      strip.text = element_text(size = 9.5, face = "bold"),
      legend.position = "none"
    )

  if (!paper) {
    p <- p +
      labs(
        title = "Corrected Peak Structure",
        subtitle = paste0("Baseline-subtracted spectra after SNV normalization (", example_label, ")"),
        caption = "Scaling is applied only after baseline subtraction, matching the preprocessing order."
      )
  }
  p
}

make_summary_plot <- function(paper = FALSE) {
  plot_df <- method_summary %>%
    mutate(
      method_plot = factor(method, levels = method_order, labels = method_labels[method_order])
    )

  p <- ggplot(plot_df, aes(x = method_plot, y = baseline_area_fraction, color = method)) +
    geom_point(size = 3.3) +
    geom_segment(aes(xend = method_plot, y = 0, yend = baseline_area_fraction), linewidth = 0.65) +
    scale_color_manual(values = method_colors, breaks = method_order, labels = method_labels[method_order]) +
    scale_y_continuous(limits = c(0, max(plot_df$baseline_area_fraction) * 1.18)) +
    labs(
      x = NULL,
      y = "Estimated background area fraction"
    ) +
    theme_signal() +
    theme(legend.position = "none")

  if (!paper) {
    p <- p +
      labs(
        title = "Estimated Background Burden",
        subtitle = "Area of estimated background relative to the smoothed raw spectrum",
        caption = "Representative-spectrum diagnostic; not a cohort-level accuracy metric."
      )
  }
  p
}

save_plot(make_background_plot(FALSE), "01_background_estimation_titled", width = 6.2, height = 4.0)
save_plot(make_background_plot(TRUE), "01_background_estimation_paper", width = 5.8, height = 3.5)

save_plot(make_corrected_plot(FALSE), "02_corrected_peak_structure_titled", width = 6.2, height = 5.8)
save_plot(make_corrected_plot(TRUE), "02_corrected_peak_structure_paper", width = 5.8, height = 5.2)

save_plot(make_summary_plot(FALSE), "03_background_burden_titled", width = 5.6, height = 3.7)
save_plot(make_summary_plot(TRUE), "03_background_burden_paper", width = 4.8, height = 3.4)

write_excel_csv(method_summary, file.path(out_dir, "baseline_correction_effect_plot_values.csv"))

cat("Wrote signal-level baseline-correction figures to", out_dir, "\n")
