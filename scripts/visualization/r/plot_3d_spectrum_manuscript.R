#!/usr/bin/env Rscript
# ============================================================================
# Manuscript-ready 3D-style SERS spectrum figures
#
# Outputs:
#   1) Raman shift x disease/model group x subject-mean spectrum waterfall
#   2) Subject-level PC1 x PC2 x PC3 spectral space projection
#   3) Combined two-panel figure
#   4) Subject-mean spectra stacked in one row per subject
#   5) Subject-mean spectra in 3D row-stack projection
#
# This script intentionally uses a static 2D isometric projection instead of
# plotly/rgl so that the outputs are stable in PDF/TIFF manuscript workflows.
# ============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(scales)
  library(patchwork)
  library(grid)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  hit <- grep(paste0("^", flag, "="), args, value = TRUE)
  if (length(hit) == 0) return(default)
  sub(paste0("^", flag, "="), "", hit[[1]])
}

script_args <- commandArgs(trailingOnly = FALSE)
script_file <- sub("^--file=", "", script_args[grep("^--file=", script_args)][1])
if (is.na(script_file)) {
  root_dir <- normalizePath(getwd())
} else {
  root_dir <- normalizePath(file.path(dirname(script_file), "../../.."))
}

cohort_dir <- get_arg(
  "--cohort-dir",
  file.path(root_dir, "publications", "대한암학회_20260610_pan_combined_split")
)
cohort_dir <- normalizePath(cohort_dir)
csv_dir <- file.path(cohort_dir, "csv")
out_dir <- get_arg("--out-dir", file.path(root_dir, "publications", "cancers"))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

processed_spectra_path <- file.path(root_dir, "results", "processed_spectra.csv")
cohort_manifest_path <- file.path(csv_dir, "cohort_subject_manifest.csv")
pca_scores_path <- file.path(csv_dir, "pca_subject_scores.csv")
pca_var_path <- file.path(csv_dir, "pca_explained_variance.csv")

required <- c(processed_spectra_path, cohort_manifest_path, pca_scores_path, pca_var_path)
missing <- required[!file.exists(required)]
if (length(missing) > 0) {
  stop("Missing required input file(s):\n", paste(missing, collapse = "\n"))
}

group_order <- c("CONTROL", "PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "BLC")
group_labels <- c(
  CONTROL = "Control",
  PRO = "Prostate",
  BRE = "Breast",
  OVA = "Ovarian",
  LUN = "Lung",
  CRC = "Colorectal",
  PAN = "Pancreatic",
  BLC = "Bladder"
)
group_palette <- c(
  CONTROL = "#6b7280",
  PRO = "#d55e00",
  BRE = "#cc79a7",
  OVA = "#e69f00",
  LUN = "#009e73",
  CRC = "#0072b2",
  PAN = "#8f6f00",
  BLC = "#7e57c2"
)

theme_manuscript <- function(base_size = 11) {
  theme_minimal(base_size = base_size, base_family = "sans") +
    theme(
      plot.title = element_text(face = "bold", color = "#111827", size = base_size + 2),
      plot.subtitle = element_text(color = "#4b5563", size = base_size - 1, margin = margin(b = 6)),
      axis.title = element_blank(),
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      panel.grid = element_blank(),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      legend.title = element_blank(),
      legend.text = element_text(size = base_size - 1, color = "#374151"),
      legend.key.height = unit(0.38, "cm"),
      plot.margin = margin(10, 18, 10, 12)
    )
}

scale_to <- function(x, to = c(0, 1), from = range(x, na.rm = TRUE)) {
  scales::rescale(x, to = to, from = from)
}

save_all <- function(plot, stem, width, height, save_pdf = TRUE) {
  png_path <- file.path(out_dir, paste0(stem, ".png"))
  pdf_path <- file.path(out_dir, paste0(stem, ".pdf"))
  tif_path <- file.path(out_dir, paste0(stem, ".tiff"))

  ggsave(png_path, plot, width = width, height = height, dpi = 600, bg = "white")
  if (isTRUE(save_pdf)) {
    ggsave(pdf_path, plot, width = width, height = height, device = cairo_pdf, bg = "white")
  }
  ggsave(tif_path, plot, width = width, height = height, dpi = 600,
         compression = "lzw", bg = "white")

  cat("[saved]", png_path, "\n")
  if (isTRUE(save_pdf)) {
    cat("[saved]", pdf_path, "\n")
  }
  cat("[saved]", tif_path, "\n")
}

make_subject_mean_waterfall <- function() {
  spectra_wide <- read_csv(processed_spectra_path, show_col_types = FALSE)
  cohort <- read_csv(cohort_manifest_path, show_col_types = FALSE) %>%
    mutate(
      display_group = if_else(analysis_group == "Control", "CONTROL", model_group)
    ) %>%
    select(source_group, sample_id, subject_id, display_group)

  joined <- spectra_wide %>%
    inner_join(cohort, by = c("group" = "source_group", "sample_id" = "sample_id")) %>%
    filter(display_group %in% group_order) %>%
    mutate(
      display_group = factor(display_group, levels = group_order)
    )

  wn_cols <- grep("^x_", names(joined), value = TRUE)
  subject_mean <- joined %>%
    group_by(subject_id, display_group) %>%
    summarise(
      across(all_of(wn_cols), ~ mean(.x, na.rm = TRUE)),
      n_replicates = n(),
      .groups = "drop"
    ) %>%
    mutate(
      display_group = factor(display_group, levels = group_order),
      group_index = as.integer(display_group) - 1L,
      spectrum_no = row_number()
    )

  spectrum_counts <- subject_mean %>%
    count(display_group, name = "n_subjects") %>%
    mutate(group_label = unname(group_labels[as.character(display_group)]))
  write_excel_csv(spectrum_counts, file.path(out_dir, "fig_sers_3d_subject_mean_spectra_counts.csv"))

  wavenumbers <- as.numeric(sub("^x_", "", wn_cols))
  intensity_matrix <- as.matrix(subject_mean[, wn_cols])
  intensity_values <- as.vector(t(intensity_matrix))
  intensity_range <- range(intensity_values, na.rm = TRUE)

  n_spectra <- nrow(subject_mean)
  n_subjects <- n_distinct(subject_mean$subject_id)
  n_wn <- length(wavenumbers)
  wn_range <- range(wavenumbers, na.rm = TRUE)
  n_groups <- length(group_order)
  x_shift <- 0.042
  y_shift <- 0.185

  spectra_proj <- tibble(
    spectrum_no = rep(subject_mean$spectrum_no, each = n_wn),
    display_group = factor(rep(as.character(subject_mean$display_group), each = n_wn), levels = group_order),
    group_index = rep(subject_mean$group_index, each = n_wn),
    wavenumber = rep(wavenumbers, times = n_spectra),
    intensity = intensity_values
  ) %>%
    mutate(
      x_norm = scale_to(wavenumber, from = wn_range),
      z = scale_to(intensity, from = intensity_range),
      x_proj = x_norm + group_index * x_shift,
      y_offset = group_index * y_shift,
      y_proj = z + y_offset
    )

  label_df <- tibble(
    display_group = factor(group_order, levels = group_order),
    group_index = seq_along(group_order) - 1L,
    group_label = unname(group_labels[group_order]),
    x_proj = 1 + (seq_along(group_order) - 1L) * x_shift + 0.025,
    y_proj = 0.52 + (seq_along(group_order) - 1L) * y_shift
  )

  tick_values <- c(400, 800, 1200, 1600, 2000, 2200)
  tick_values <- tick_values[tick_values >= wn_range[1] & tick_values <= wn_range[2]]
  tick_df <- tibble(
    wavenumber = tick_values,
    x = scale_to(tick_values, from = wn_range),
    y = -0.095,
    label = tick_values
  )

  depth_end <- tibble(
    x = (n_groups - 1) * x_shift,
    y = (n_groups - 1) * y_shift - 0.035
  )

  ggplot() +
    annotate("segment", x = 0, y = -0.08, xend = 1.08, yend = -0.08,
             linewidth = 0.35, color = "#374151",
             arrow = arrow(length = unit(0.12, "inches"), type = "closed")) +
    annotate("segment", x = 0, y = -0.08, xend = depth_end$x,
             yend = depth_end$y, linewidth = 0.35, color = "#374151",
             arrow = arrow(length = unit(0.12, "inches"), type = "closed")) +
    annotate("segment", x = 0, y = -0.08, xend = 0, yend = 1.18,
             linewidth = 0.35, color = "#374151",
             arrow = arrow(length = unit(0.12, "inches"), type = "closed")) +
    geom_segment(data = tick_df, aes(x = x, xend = x, y = y + 0.010, yend = y - 0.010),
                 linewidth = 0.25, color = "#6b7280") +
    geom_text(data = tick_df, aes(x = x, y = y - 0.035, label = label),
              size = 2.6, color = "#4b5563") +
    annotate("text", x = 0.55, y = -0.175, label = "Raman shift (cm^-1)",
             size = 3.3, color = "#111827") +
    annotate("text", x = depth_end$x + 0.060, y = depth_end$y + 0.025,
             label = "Group", size = 3.3, color = "#111827", angle = 24) +
    annotate("text", x = -0.055, y = 0.58,
             label = "Intensity\n(global scale)", size = 3.1,
             color = "#111827", angle = 90, lineheight = 0.9) +
    geom_line(
      data = spectra_proj,
      aes(x = x_proj, y = y_proj, group = spectrum_no, color = display_group),
      linewidth = 0.085, alpha = 0.16, lineend = "round"
    ) +
    geom_text(
      data = label_df,
      aes(x = x_proj, y = y_proj, label = group_label, color = display_group),
      hjust = 0, size = 3.1, fontface = "bold"
    ) +
    scale_color_manual(values = group_palette, labels = group_labels, drop = FALSE) +
    scale_fill_manual(values = group_palette, labels = group_labels, drop = FALSE) +
    coord_cartesian(
      xlim = c(-0.08, 1.42),
      ylim = c(-0.22, (n_groups - 1) * y_shift + 1.22),
      clip = "off"
    ) +
    labs(
      title = "3D Waterfall View of Subject-Mean SERS Spectra",
      subtitle = sprintf(
        "Raman shift x disease group x intensity; n=%s subject-mean spectra from %s subjects; full global intensity range",
        comma(n_spectra), comma(n_subjects)
      )
    ) +
    guides(fill = "none", color = guide_legend(override.aes = list(linewidth = 1.4, alpha = 1))) +
    theme_manuscript(11) +
    theme(legend.position = "none")
}

make_subject_mean_row_stack <- function() {
  spectra_wide <- read_csv(processed_spectra_path, show_col_types = FALSE)
  cohort <- read_csv(cohort_manifest_path, show_col_types = FALSE) %>%
    mutate(
      display_group = if_else(analysis_group == "Control", "CONTROL", model_group)
    ) %>%
    select(source_group, sample_id, subject_id, display_group)

  joined <- spectra_wide %>%
    inner_join(cohort, by = c("group" = "source_group", "sample_id" = "sample_id")) %>%
    filter(display_group %in% group_order) %>%
    mutate(display_group = factor(display_group, levels = group_order))

  wn_cols <- grep("^x_", names(joined), value = TRUE)
  subject_mean <- joined %>%
    group_by(subject_id, display_group) %>%
    summarise(
      across(all_of(wn_cols), ~ mean(.x, na.rm = TRUE)),
      n_replicates = n(),
      .groups = "drop"
    ) %>%
    arrange(display_group, subject_id) %>%
    mutate(row_index = row_number())

  wavenumbers <- as.numeric(sub("^x_", "", wn_cols))
  intensity_matrix <- as.matrix(subject_mean[, wn_cols])

  # Per-subject scaling keeps every row legible while preserving each subject's
  # spectral shape. The 3D waterfall keeps the global intensity scale.
  row_min <- apply(intensity_matrix, 1, min, na.rm = TRUE)
  row_max <- apply(intensity_matrix, 1, max, na.rm = TRUE)
  row_range <- pmax(row_max - row_min, .Machine$double.eps)
  scaled_matrix <- sweep(intensity_matrix, 1, row_min, "-")
  scaled_matrix <- sweep(scaled_matrix, 1, row_range, "/")

  n_subjects <- nrow(subject_mean)
  n_wn <- length(wavenumbers)
  row_amplitude <- 0.84

  spectra_rows <- tibble(
    subject_id = rep(subject_mean$subject_id, each = n_wn),
    display_group = factor(rep(as.character(subject_mean$display_group), each = n_wn), levels = group_order),
    row_index = rep(subject_mean$row_index, each = n_wn),
    wavenumber = rep(wavenumbers, times = n_subjects),
    intensity_scaled = as.vector(t(scaled_matrix))
  ) %>%
    mutate(y = row_index + intensity_scaled * row_amplitude)

  group_spans <- subject_mean %>%
    group_by(display_group) %>%
    summarise(
      y_min = min(row_index),
      y_max = max(row_index),
      y_mid = mean(range(row_index)),
      n_subjects = n(),
      .groups = "drop"
    ) %>%
    mutate(group_label = unname(group_labels[as.character(display_group)]))

  ggplot() +
    geom_rect(
      data = group_spans,
      aes(xmin = -Inf, xmax = Inf, ymin = y_min - 0.45, ymax = y_max + 1.25,
          fill = display_group),
      alpha = 0.018, inherit.aes = FALSE
    ) +
    geom_line(
      data = spectra_rows,
      aes(x = wavenumber, y = y, group = subject_id, color = display_group),
      linewidth = 0.13, alpha = 0.62, lineend = "round"
    ) +
    geom_hline(
      data = group_spans,
      aes(yintercept = y_max + 1.45),
      linewidth = 0.25, color = "#d1d5db"
    ) +
    geom_text(
      data = group_spans,
      aes(x = max(wavenumbers) + 55, y = y_mid, label = sprintf("%s (n=%s)", group_label, n_subjects),
          color = display_group),
      hjust = 0, size = 3.0, fontface = "bold"
    ) +
    scale_color_manual(values = group_palette, labels = group_labels, drop = FALSE) +
    scale_fill_manual(values = group_palette, labels = group_labels, drop = FALSE) +
    scale_x_continuous(
      breaks = c(400, 800, 1200, 1600, 2000, 2200),
      expand = expansion(mult = c(0.01, 0.08))
    ) +
    scale_y_reverse(expand = expansion(mult = c(0.01, 0.01))) +
    labs(
      title = "Subject-Mean SERS Spectra Stacked by Patient",
      subtitle = sprintf(
        "Each row represents one subject-mean spectrum; n=%s subjects, ordered by disease group",
        comma(n_subjects)
      ),
      x = "Raman shift (cm^-1)",
      y = "Subjects ordered by group"
    ) +
    guides(color = "none", fill = "none") +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", color = "#111827", size = 14),
      plot.subtitle = element_text(color = "#4b5563", size = 11, margin = margin(b = 8)),
      axis.title = element_text(face = "bold", color = "#111827"),
      axis.text.x = element_text(color = "#4b5563"),
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(color = "#e5e7eb", linewidth = 0.3),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      plot.margin = margin(12, 72, 12, 12)
    ) +
    coord_cartesian(clip = "off")
}

make_subject_mean_3d_row_stack <- function() {
  spectra_wide <- read_csv(processed_spectra_path, show_col_types = FALSE)
  cohort <- read_csv(cohort_manifest_path, show_col_types = FALSE) %>%
    mutate(
      display_group = if_else(analysis_group == "Control", "CONTROL", model_group)
    ) %>%
    select(source_group, sample_id, subject_id, display_group)

  joined <- spectra_wide %>%
    inner_join(cohort, by = c("group" = "source_group", "sample_id" = "sample_id")) %>%
    filter(display_group %in% group_order) %>%
    mutate(display_group = factor(display_group, levels = group_order))

  wn_cols <- grep("^x_", names(joined), value = TRUE)
  subject_mean <- joined %>%
    group_by(subject_id, display_group) %>%
    summarise(
      across(all_of(wn_cols), ~ mean(.x, na.rm = TRUE)),
      n_replicates = n(),
      .groups = "drop"
    ) %>%
    arrange(display_group, subject_id) %>%
    mutate(row_index = row_number())

  wavenumbers <- as.numeric(sub("^x_", "", wn_cols))
  intensity_matrix <- as.matrix(subject_mean[, wn_cols])

  # Each subject gets its own local z-scale so that all 1,628 spectra remain
  # visible. The grouped 3D waterfall above preserves global intensity scale.
  row_min <- apply(intensity_matrix, 1, min, na.rm = TRUE)
  row_max <- apply(intensity_matrix, 1, max, na.rm = TRUE)
  row_range <- pmax(row_max - row_min, .Machine$double.eps)
  scaled_matrix <- sweep(intensity_matrix, 1, row_min, "-")
  scaled_matrix <- sweep(scaled_matrix, 1, row_range, "/")

  n_subjects <- nrow(subject_mean)
  n_wn <- length(wavenumbers)
  wn_range <- range(wavenumbers, na.rm = TRUE)
  depth_shift <- 0.060
  row_amplitude <- 0.88

  spectra_rows <- tibble(
    subject_id = rep(subject_mean$subject_id, each = n_wn),
    display_group = factor(rep(as.character(subject_mean$display_group), each = n_wn), levels = group_order),
    row_index = rep(subject_mean$row_index, each = n_wn),
    wavenumber = rep(wavenumbers, times = n_subjects),
    intensity_scaled = as.vector(t(scaled_matrix))
  ) %>%
    mutate(
      x_proj = wavenumber + row_index * depth_shift,
      y_proj = row_index + intensity_scaled * row_amplitude
    )

  group_spans <- subject_mean %>%
    group_by(display_group) %>%
    summarise(
      y_min = min(row_index),
      y_max = max(row_index),
      y_mid = mean(range(row_index)),
      n_subjects = n(),
      .groups = "drop"
    ) %>%
    mutate(
      group_label = unname(group_labels[as.character(display_group)]),
      x_label = wn_range[2] + y_mid * depth_shift + 70
    )

  band_df <- group_spans %>%
    rowwise() %>%
    do({
      tibble(
        display_group = .$display_group,
        x = c(
          wn_range[1] + .$y_min * depth_shift,
          wn_range[2] + .$y_min * depth_shift,
          wn_range[2] + .$y_max * depth_shift,
          wn_range[1] + .$y_max * depth_shift
        ),
        y = c(.$y_min - 0.45, .$y_min - 0.45, .$y_max + 1.25, .$y_max + 1.25)
      )
    }) %>%
    ungroup()

  boundary_df <- group_spans %>%
    transmute(
      display_group,
      y = y_max + 1.45,
      x_start = wn_range[1] + y * depth_shift,
      x_end = wn_range[2] + y * depth_shift
    )

  axis_origin <- tibble(x = wn_range[1], y = 1)
  subject_axis_end <- tibble(
    x = wn_range[1] + n_subjects * depth_shift,
    y = n_subjects
  )

  ggplot() +
    geom_polygon(
      data = band_df,
      aes(x = x, y = y, group = display_group, fill = display_group),
      alpha = 0.018, color = NA
    ) +
    geom_line(
      data = spectra_rows,
      aes(x = x_proj, y = y_proj, group = subject_id, color = display_group),
      linewidth = 0.13, alpha = 0.62, lineend = "round"
    ) +
    geom_segment(
      data = boundary_df,
      aes(x = x_start, xend = x_end, y = y, yend = y),
      linewidth = 0.25, color = "#d1d5db"
    ) +
    geom_segment(
      aes(
        x = axis_origin$x,
        xend = wn_range[2] + 80,
        y = axis_origin$y - 1.2,
        yend = axis_origin$y - 1.2
      ),
      inherit.aes = FALSE, linewidth = 0.35, color = "#374151",
      arrow = arrow(length = unit(0.12, "inches"), type = "closed")
    ) +
    geom_segment(
      aes(
        x = axis_origin$x,
        xend = subject_axis_end$x,
        y = axis_origin$y,
        yend = subject_axis_end$y
      ),
      inherit.aes = FALSE, linewidth = 0.35, color = "#374151",
      arrow = arrow(length = unit(0.12, "inches"), type = "closed")
    ) +
    geom_segment(
      aes(
        x = axis_origin$x,
        xend = axis_origin$x,
        y = axis_origin$y,
        yend = axis_origin$y + 48
      ),
      inherit.aes = FALSE, linewidth = 0.35, color = "#374151",
      arrow = arrow(length = unit(0.12, "inches"), type = "closed")
    ) +
    annotate("text", x = mean(wn_range), y = -28, label = "Raman shift (cm^-1)",
             size = 3.2, color = "#111827") +
    annotate("text", x = subject_axis_end$x + 38, y = subject_axis_end$y * 0.50,
             label = "Subject axis", size = 3.1, color = "#111827", angle = 84) +
    annotate("text", x = wn_range[1] - 40, y = 30, label = "Intensity",
             size = 3.1, color = "#111827", angle = 90) +
    geom_text(
      data = group_spans,
      aes(x = x_label, y = y_mid, label = sprintf("%s (n=%s)", group_label, n_subjects),
          color = display_group),
      hjust = 0, size = 3.0, fontface = "bold"
    ) +
    scale_color_manual(values = group_palette, labels = group_labels, drop = FALSE) +
    scale_fill_manual(values = group_palette, labels = group_labels, drop = FALSE) +
    scale_x_continuous(
      breaks = c(400, 800, 1200, 1600, 2000, 2200),
      expand = expansion(mult = c(0.01, 0.12))
    ) +
    scale_y_reverse(expand = expansion(mult = c(0.015, 0.025))) +
    labs(
      title = "3D Row-Stack Projection of Subject-Mean SERS Spectra",
      subtitle = sprintf(
        "x=Raman shift, y=subject axis, z=per-subject normalized intensity; n=%s subject-mean spectra",
        comma(n_subjects)
      )
    ) +
    guides(color = "none", fill = "none") +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", color = "#111827", size = 14),
      plot.subtitle = element_text(color = "#4b5563", size = 11, margin = margin(b = 8)),
      axis.title = element_blank(),
      axis.text.x = element_text(color = "#4b5563"),
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(color = "#e5e7eb", linewidth = 0.3),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      plot.margin = margin(12, 92, 12, 12)
    ) +
    coord_cartesian(clip = "off")
}

project_3d <- function(coords, theta = 38 * pi / 180, phi = 24 * pi / 180) {
  rz <- matrix(c(
    cos(theta), -sin(theta), 0,
    sin(theta),  cos(theta), 0,
    0,           0,          1
  ), nrow = 3, byrow = TRUE)
  rx <- matrix(c(
    1, 0,        0,
    0, cos(phi), -sin(phi),
    0, sin(phi),  cos(phi)
  ), nrow = 3, byrow = TRUE)
  rotated <- as.matrix(coords) %*% t(rz) %*% t(rx)
  tibble(x_proj = rotated[, 1], y_proj = rotated[, 3])
}

make_pca_projection <- function() {
  pca <- read_csv(pca_scores_path, show_col_types = FALSE) %>%
    filter(display_group %in% group_order) %>%
    mutate(
      display_group = factor(display_group, levels = group_order),
      group_label = unname(group_labels[as.character(display_group)])
    )

  pca_var <- read_csv(pca_var_path, show_col_types = FALSE)
  pc_labels <- pca_var %>%
    filter(component %in% c("PC1", "PC2", "PC3")) %>%
    mutate(label = sprintf("%s %.1f%%", component, explained_variance_percent)) %>%
    pull(label)
  total_var <- pca_var %>%
    filter(component %in% c("PC1", "PC2", "PC3")) %>%
    summarise(total = sum(explained_variance_percent)) %>%
    pull(total)

  coords <- pca %>%
    transmute(
      x = scale_to(PC1, to = c(-1, 1)),
      y = scale_to(PC2, to = c(-1, 1)),
      z = scale_to(PC3, to = c(-1, 1))
    )
  proj <- project_3d(coords)
  pca_proj <- bind_cols(pca, proj)

  centroid_proj <- pca_proj %>%
    group_by(display_group, group_label) %>%
    summarise(x_proj = median(x_proj), y_proj = median(y_proj), .groups = "drop") %>%
    mutate(
      label_dx = case_when(
        display_group == "CONTROL" ~ 0.10,
        display_group == "PRO" ~ -0.22,
        display_group == "BRE" ~ 0.05,
        display_group == "OVA" ~ -0.14,
        display_group == "BLC" ~ -0.17,
        TRUE ~ 0.04
      ),
      label_dy = case_when(
        display_group == "CONTROL" ~ -0.08,
        display_group == "PRO" ~ -0.02,
        display_group == "BRE" ~ 0.08,
        display_group == "OVA" ~ 0.12,
        display_group == "BLC" ~ 0.10,
        display_group == "PAN" ~ 0.10,
        TRUE ~ 0.07
      ),
      hjust_label = case_when(
        display_group %in% c("PRO", "OVA", "BLC") ~ 1,
        TRUE ~ 0
      )
    )

  axis_ends <- tibble(
    axis = c(pc_labels[1], pc_labels[2], pc_labels[3]),
    x = c(1.25, 0, 0),
    y = c(0, 1.25, 0),
    z = c(0, 0, 1.25)
  )
  axis_proj <- bind_cols(axis_ends, project_3d(axis_ends[, c("x", "y", "z")]))
  origin <- project_3d(tibble(x = 0, y = 0, z = 0))

  ggplot() +
    geom_segment(
      data = axis_proj,
      aes(x = origin$x_proj, y = origin$y_proj, xend = x_proj, yend = y_proj),
      inherit.aes = FALSE, linewidth = 0.38, color = "#374151",
      arrow = arrow(length = unit(0.12, "inches"), type = "closed")
    ) +
    geom_text(
      data = axis_proj,
      aes(x = x_proj * 1.08, y = y_proj * 1.08, label = axis),
      inherit.aes = FALSE, size = 3.0, color = "#111827"
    ) +
    geom_point(
      data = pca_proj,
      aes(x = x_proj, y = y_proj, color = display_group),
      size = 1.35, alpha = 0.48
    ) +
    geom_point(
      data = centroid_proj,
      aes(x = x_proj, y = y_proj, fill = display_group),
      shape = 21, color = "white", stroke = 0.55, size = 4.1
    ) +
    geom_text(
      data = centroid_proj,
      aes(
        x = x_proj + label_dx,
        y = y_proj + label_dy,
        label = group_label,
        color = display_group,
        hjust = hjust_label
      ),
      size = 3.0, fontface = "bold"
    ) +
    scale_color_manual(values = group_palette, labels = group_labels, drop = FALSE) +
    scale_fill_manual(values = group_palette, labels = group_labels, drop = FALSE) +
    coord_equal(clip = "off") +
    labs(
      title = "Subject-Level 3D Spectral Space",
      subtitle = sprintf("Isometric projection of PC1-PC3 scores; cumulative variance %.1f%%", total_var)
    ) +
    guides(fill = "none", color = guide_legend(override.aes = list(size = 3, alpha = 1))) +
    theme_manuscript(11) +
    theme(legend.position = "right")
}

p_waterfall <- make_subject_mean_waterfall()
p_row_stack <- make_subject_mean_row_stack()
p_row_stack_3d <- make_subject_mean_3d_row_stack()
p_pca <- make_pca_projection()

p_combined <- (
  p_waterfall +
    labs(
      title = "A. 3D Waterfall View of Subject-Mean SERS Spectra",
      subtitle = "Raman shift x group x intensity; one mean spectrum per subject"
    )
) + (
  p_pca +
    labs(
      title = "B. Subject-Level 3D Spectral Space",
      subtitle = "PC1-PC3 isometric projection; cumulative variance 49.2%"
    )
) +
  plot_layout(widths = c(1.08, 1.00))

save_all(p_waterfall, "fig_sers_3d_waterfall_subject_mean_spectra", width = 9.0, height = 5.8, save_pdf = FALSE)
save_all(p_row_stack, "fig_sers_subject_mean_spectra_row_stack", width = 9.5, height = 22.0, save_pdf = FALSE)
save_all(p_row_stack_3d, "fig_sers_subject_mean_spectra_3d_row_stack", width = 10.0, height = 22.0, save_pdf = FALSE)
save_all(p_pca, "fig_sers_3d_pca_subject_space", width = 7.0, height = 5.8, save_pdf = TRUE)
save_all(p_combined, "fig_sers_3d_subject_mean_combined", width = 14.0, height = 5.9, save_pdf = FALSE)

cat("\nDone. Output directory:", out_dir, "\n")
