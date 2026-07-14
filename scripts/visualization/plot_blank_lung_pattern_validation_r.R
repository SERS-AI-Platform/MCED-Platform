#!/usr/bin/env Rscript

# Blank/background residual and LUN class-pattern validation figures.

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
data_dir <- file.path(project_root, "results", "blank_lung_pattern_validation")
out_dir <- file.path(project_root, "results", "figures", "blank_lung_pattern_validation_r")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

blank_df <- read_csv(file.path(data_dir, "blank_pattern_validation_long.csv"), show_col_types = FALSE)
lung_df <- read_csv(file.path(data_dir, "lung_pattern_validation_long.csv"), show_col_types = FALSE)
lung_summary <- read_csv(file.path(data_dir, "lung_pattern_validation_summary.csv"), show_col_types = FALSE)
affinity_df <- read_csv(file.path(data_dir, "lung_subject_class_affinity.csv"), show_col_types = FALSE)
reference_group <- lung_summary$reference_group[1]

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

save_plot <- function(plot, stem, width = 5.8, height = 3.6) {
  ggsave(file.path(out_dir, paste0(stem, ".png")), plot, width = width, height = height, dpi = 320, bg = "white")
  ggsave(file.path(out_dir, paste0(stem, ".pdf")), plot, width = width, height = height, bg = "white")
}

blank_summary <- blank_df %>%
  filter(component %in% c("blank_sg_smoothed", "blank_baseline_corrected")) %>%
  group_by(component, wavenumber) %>%
  summarise(
    mean_intensity = mean(intensity, na.rm = TRUE),
    p10 = quantile(intensity, 0.10, na.rm = TRUE),
    p90 = quantile(intensity, 0.90, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    component_label = case_when(
      component == "blank_sg_smoothed" ~ "Blank before baseline correction",
      component == "blank_baseline_corrected" ~ "Blank after baseline correction",
      TRUE ~ component
    ),
    component_label = factor(
      component_label,
      levels = c("Blank before baseline correction", "Blank after baseline correction")
    )
  )

make_blank_plot <- function(paper = FALSE) {
  p <- ggplot(blank_summary, aes(x = wavenumber, y = mean_intensity)) +
    geom_ribbon(aes(ymin = p10, ymax = p90), fill = "#A0CBE8", alpha = 0.38) +
    geom_line(color = "#4E79A7", linewidth = 0.55) +
    facet_wrap(~component_label, ncol = 1, scales = "free_y") +
    labs(
      x = expression("Raman shift (cm"^-1*")"),
      y = "Intensity (a.u.)"
    ) +
    theme_signal()

  if (!paper) {
    p <- p +
      labs(
        title = "Blank Spectrum Background Check",
        subtitle = "Mean blank spectrum with 10th-90th percentile band across blank replicates",
        caption = "Broad blank background is reduced; residual structure remains."
      )
  }
  p
}

lung_rep <- lung_df %>%
  filter(panel == "lung_subject_replicates") %>%
  mutate(replicate = factor(replicate))

class_overlay <- lung_df %>%
  filter(panel == "class_mean_overlay") %>%
  mutate(
    series = gsub("subject mean", "class mean", series),
    series = factor(series, levels = c("LUN class mean", paste0(reference_group, " class mean")))
  )

class_diff <- lung_df %>%
  filter(panel == "class_mean_difference")

affinity_long <- bind_rows(
  affinity_df %>%
    transmute(
      sample_id,
      comparison = "LUN mean\n(leave-one-out)",
      correlation = corr_to_lun_leave_one_subject_out_mean
    ),
  affinity_df %>%
    transmute(
      sample_id,
      comparison = paste0(reference_group, " mean"),
      correlation = corr_to_reference_mean
    )
) %>%
  mutate(comparison = factor(comparison, levels = c("LUN mean\n(leave-one-out)", paste0(reference_group, " mean"))))

make_lung_overlay_plot <- function(paper = FALSE) {
  p <- ggplot() +
    geom_line(
      data = lung_rep,
      aes(x = wavenumber, y = intensity, group = replicate),
      color = "#B7B7B7",
      linewidth = 0.32,
      alpha = 0.80
    ) +
    geom_line(
      data = class_overlay,
      aes(x = wavenumber, y = intensity, color = series),
      linewidth = 0.78
    ) +
    scale_color_manual(values = c("LUN class mean" = "#E15759", "NOR class mean" = "#4E79A7", "YNOR class mean" = "#4E79A7")) +
    labs(
      x = expression("Raman shift (cm"^-1*")"),
      y = "Normalized intensity (a.u.)"
    ) +
    theme_signal()

  if (!paper) {
    p <- p +
      labs(
        title = "LUN Pattern After Preprocessing",
        subtitle = sprintf(
          "LUN %s vs class means | corr: LUN %.3f, %s %.3f",
          lung_summary$lung_subject_id[1],
          lung_summary$lung_subject_mean_corr_to_lun_mean[1],
          reference_group,
          lung_summary$lung_subject_mean_corr_to_reference_mean[1]
        ),
        caption = "Gray lines: selected LUN subject replicates. Colored lines: class means."
      )
  }
  p
}

make_lung_difference_plot <- function(paper = FALSE) {
  p <- ggplot(class_diff, aes(x = wavenumber, y = intensity)) +
    geom_hline(yintercept = 0, color = "#777777", linewidth = 0.35) +
    geom_line(color = "#E15759", linewidth = 0.65) +
    labs(
      x = expression("Raman shift (cm"^-1*")"),
      y = "Mean difference (a.u.)"
    ) +
    theme_signal()

  if (!paper) {
    p <- p +
      labs(
        title = "Class-Level LUN Difference",
        subtitle = sprintf("LUN mean minus %s mean after preprocessing", reference_group),
        caption = "Disease pattern should be interpreted at the class/subject-mean level, not from one replicate alone."
      )
  }
  p
}

make_lung_affinity_plot <- function(paper = FALSE) {
  p <- ggplot(affinity_long, aes(x = comparison, y = correlation, group = sample_id)) +
    geom_line(color = "#B7B7B7", linewidth = 0.35, alpha = 0.65) +
    geom_point(aes(color = comparison), size = 1.55, alpha = 0.82) +
    scale_color_manual(values = c("LUN mean\n(leave-one-out)" = "#E15759", "NOR mean" = "#4E79A7", "YNOR mean" = "#4E79A7")) +
    labs(
      x = NULL,
      y = "Correlation to class mean"
    ) +
    theme_signal() +
    theme(legend.position = "none")

  if (!paper) {
    p <- p +
      labs(
        title = "LUN Subject Affinity Check",
        subtitle = sprintf(
          "%d/%d LUN subjects closer to LUN mean; median delta %.3f",
          lung_summary$n_lun_subjects_closer_to_lun_loo_mean[1],
          lung_summary$n_lun_subjects[1],
          lung_summary$median_lun_corr_delta_lun_minus_reference[1]
        ),
        caption = "The LUN class mean excludes the evaluated subject to avoid circularity."
      )
  }
  p
}

save_plot(make_blank_plot(FALSE), "01_blank_background_check_titled", width = 6.0, height = 5.0)
save_plot(make_blank_plot(TRUE), "01_blank_background_check_paper", width = 5.6, height = 4.5)

save_plot(make_lung_overlay_plot(FALSE), "02_lun_pattern_overlay_titled", width = 6.1, height = 3.9)
save_plot(make_lung_overlay_plot(TRUE), "02_lun_pattern_overlay_paper", width = 5.7, height = 3.4)

save_plot(make_lung_difference_plot(FALSE), "03_lun_class_difference_titled", width = 5.8, height = 3.6)
save_plot(make_lung_difference_plot(TRUE), "03_lun_class_difference_paper", width = 5.4, height = 3.2)

save_plot(make_lung_affinity_plot(FALSE), "04_lun_subject_affinity_titled", width = 5.5, height = 3.8)
save_plot(make_lung_affinity_plot(TRUE), "04_lun_subject_affinity_paper", width = 5.1, height = 3.3)

write_excel_csv(lung_summary, file.path(out_dir, "lung_pattern_plot_values.csv"))

cat("Wrote blank/LUN pattern validation figures to", out_dir, "\n")
