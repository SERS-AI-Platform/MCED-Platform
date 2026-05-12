#!/usr/bin/env Rscript
# ──────────────────────────────────────────────────────
# Supplementary Figure: Non-cancer → Early → Late Stage
# SERS Spectral Signatures (ggplot2, Nature-quality)
# ──────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(patchwork)
  library(scales)
})

# ── Paths ──
# Resolve script directory robustly
args <- commandArgs(trailingOnly = FALSE)
script_dir <- if (any(grepl("--file=", args))) {
  dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
} else {
  getwd()
}
SERS_ROOT  <- normalizePath(file.path(script_dir, "../.."))
AACR_DIR   <- file.path(SERS_ROOT, "AACR")
SPEC_PATH  <- file.path(SERS_ROOT, "results", "processed_spectra.csv")
FIG_DIR    <- file.path(AACR_DIR, "figures")

cat("SERS_ROOT:", SERS_ROOT, "\n")
cat("Loading spectra...\n")

# ── Load spectra ──
spec <- read_csv(SPEC_PATH, show_col_types = FALSE)
wn_cols <- grep("^x_", names(spec), value = TRUE)
wavenumbers <- as.numeric(sub("^x_", "", wn_cols))

# ── Staging helpers ──
classify_ajcc <- function(s) {
  s <- toupper(trimws(as.character(s)))
  ifelse(is.na(s) | s %in% c("", "UNKNOWN"), NA_character_,
  ifelse(grepl("^III|^IV", s), "Late",
  ifelse(grepl("^I|^II", s), "Early", NA_character_)))
}

classify_tstage <- function(s) {
  s <- toupper(trimws(as.character(s)))
  ifelse(is.na(s), NA_character_,
  ifelse(grepl("^T[12]", s), "Early",
  ifelse(grepl("^T[34]", s), "Late", NA_character_)))
}

# ── Load staging ──
pro <- read_csv(file.path(AACR_DIR, "data", "PRO_with_staging.csv"), show_col_types = FALSE) %>%
  mutate(sample_id = as.integer(sub("PRO ", "", patient_id)),
         stage_group = classify_tstage(t_stage)) %>%
  filter(!is.na(stage_group)) %>%
  select(sample_id, stage_group) %>%
  mutate(cancer = "PRO")

lun <- read_csv(file.path(SERS_ROOT, "data", "clinical_data", "standardized",
                           "LUN_clinical_standardized.csv"), show_col_types = FALSE) %>%
  mutate(sample_id = row_number(),
         stage_group = classify_ajcc(stage)) %>%
  filter(!is.na(stage_group)) %>%
  select(sample_id, stage_group) %>%
  mutate(cancer = "LUN")

pan <- read_csv(file.path(AACR_DIR, "data", "PAN_with_ajcc_stage.csv"), show_col_types = FALSE) %>%
  mutate(sample_id = row_number(),
         stage_group = classify_ajcc(ajcc_stage)) %>%
  filter(!is.na(stage_group)) %>%
  select(sample_id, stage_group) %>%
  mutate(cancer = "CPAN")

crc <- read_csv(file.path(AACR_DIR, "data", "CRC_with_ajcc_stage.csv"), show_col_types = FALSE) %>%
  mutate(sample_id = row_number(),
         stage_group = classify_ajcc(ajcc_stage)) %>%
  filter(!is.na(stage_group)) %>%
  select(sample_id, stage_group) %>%
  mutate(cancer = "CRC")

staging <- bind_rows(pro, lun, pan, crc)

# ── Cancer display info ──
cancer_info <- tribble(
  ~cancer, ~abbr,  ~full,          ~color,
  "PRO",   "PRC",  "Prostate",     "#8B4513",
  "LUN",   "LC",   "Lung",         "#2E8B57",
  "CPAN",  "PAC",  "Pancreatic",   "#CD5C5C",
  "CRC",   "CRC",  "Colorectal",   "#4682B4",
)

NORMAL_COLOR <- "#9E9E9E"
EARLY_COLOR  <- "#2980B9"
LATE_COLOR   <- "#E74C3C"

NON_CANCER <- c("NOR", "DIA", "HBP", "H.D.")

# ── Compute mean spectra per group ──
compute_mean_spectrum <- function(df, grp_filter, sid_filter = NULL) {
  sub <- df %>% filter(group %in% grp_filter)
  if (!is.null(sid_filter)) {
    sub <- sub %>% filter(sample_id %in% sid_filter)
  }
  if (nrow(sub) == 0) return(NULL)
  sub %>%
    select(all_of(wn_cols)) %>%
    summarise(across(everything(), mean, na.rm = TRUE)) %>%
    pivot_longer(everything(), names_to = "wn_col", values_to = "intensity") %>%
    mutate(wavenumber = as.numeric(sub("^x_", "", wn_col)))
}

# Non-cancer baseline
nc_mean <- compute_mean_spectrum(spec, NON_CANCER) %>%
  mutate(stage = "Non-cancer")

nc_n <- spec %>% filter(group %in% NON_CANCER) %>%
  distinct(group, sample_id) %>% nrow()

# ── Build plot data ──
plot_data <- list()

for (i in seq_len(nrow(cancer_info))) {
  ci <- cancer_info[i, ]
  st <- staging %>% filter(cancer == ci$cancer)

  early_ids <- st %>% filter(stage_group == "Early") %>% pull(sample_id)
  late_ids  <- st %>% filter(stage_group == "Late")  %>% pull(sample_id)

  early_mean <- compute_mean_spectrum(spec, ci$cancer, early_ids)
  late_mean  <- compute_mean_spectrum(spec, ci$cancer, late_ids)

  nc_panel <- nc_mean %>% mutate(
    cancer = ci$cancer, abbr = ci$abbr, full = ci$full,
    n_label = paste0("Non-cancer (n=", nc_n, ")")
  )

  if (!is.null(early_mean)) {
    early_panel <- early_mean %>% mutate(
      stage = "Early", cancer = ci$cancer, abbr = ci$abbr, full = ci$full,
      n_label = paste0("Early (n=", length(early_ids), ")")
    )
    plot_data <- c(plot_data, list(early_panel))
  }
  if (!is.null(late_mean)) {
    late_panel <- late_mean %>% mutate(
      stage = "Late", cancer = ci$cancer, abbr = ci$abbr, full = ci$full,
      n_label = paste0("Late (n=", length(late_ids), ")")
    )
    plot_data <- c(plot_data, list(late_panel))
  }
  plot_data <- c(plot_data, list(nc_panel))
}

df_plot <- bind_rows(plot_data) %>%
  mutate(
    stage = factor(stage, levels = c("Non-cancer", "Early", "Late")),
    facet_label = paste0(abbr, " \u2014 ", full),
    facet_label = factor(facet_label, levels = paste0(cancer_info$abbr, " \u2014 ", cancer_info$full))
  )

# ── Compute difference ribbon (|Early - Non-cancer|, top 5% highlighted) ──
diff_data <- list()
for (i in seq_len(nrow(cancer_info))) {
  ci <- cancer_info[i, ]
  fl <- paste0(ci$abbr, " \u2014 ", ci$full)

  nc_vals <- df_plot %>%
    filter(facet_label == fl, stage == "Non-cancer") %>%
    arrange(wavenumber)
  ea_vals <- df_plot %>%
    filter(facet_label == fl, stage == "Early") %>%
    arrange(wavenumber)

  if (nrow(nc_vals) == 0 || nrow(ea_vals) == 0) next

  diff_abs <- abs(ea_vals$intensity - nc_vals$intensity)
  thresh <- quantile(diff_abs, 0.95)
  is_diff <- diff_abs >= thresh

  diff_data[[i]] <- tibble(
    wavenumber = nc_vals$wavenumber,
    ymin = pmin(nc_vals$intensity, ea_vals$intensity),
    ymax = pmax(nc_vals$intensity, ea_vals$intensity),
    is_diff = is_diff,
    facet_label = fl,
    cancer_color = ci$color
  )
}
df_diff <- bind_rows(diff_data) %>%
  filter(is_diff) %>%
  mutate(facet_label = factor(facet_label,
    levels = paste0(cancer_info$abbr, " \u2014 ", cancer_info$full)))

# ── Build facet labels with sample sizes ──
facet_n_labels <- list()
for (i in seq_len(nrow(cancer_info))) {
  ci <- cancer_info[i, ]
  st <- staging %>% filter(cancer == ci$cancer)
  n_e <- sum(st$stage_group == "Early")
  n_l <- sum(st$stage_group == "Late")
  old_label <- paste0(ci$abbr, " \u2014 ", ci$full)
  new_label <- paste0(ci$abbr, " \u2014 ", ci$full,
                      "    (Early n=", n_e, ", Late n=", n_l, ")")
  facet_n_labels[[old_label]] <- new_label
}

df_plot <- df_plot %>%
  mutate(facet_display = facet_n_labels[as.character(facet_label)] %>% unlist(),
         facet_display = factor(facet_display,
           levels = unlist(facet_n_labels[paste0(cancer_info$abbr, " \u2014 ", cancer_info$full)])))
df_diff <- df_diff %>%
  mutate(facet_display = facet_n_labels[as.character(facet_label)] %>% unlist(),
         facet_display = factor(facet_display,
           levels = unlist(facet_n_labels[paste0(cancer_info$abbr, " \u2014 ", cancer_info$full)])))

# ── Plot ──
cat("Creating figure...\n")

p <- ggplot(df_plot, aes(x = wavenumber, y = intensity)) +
  # Difference highlight ribbon (top 5% regions)
  geom_rect(
    data = df_diff,
    aes(xmin = wavenumber - 1.5, xmax = wavenumber + 1.5,
        ymin = -Inf, ymax = Inf),
    fill = "#FFD700", alpha = 0.15, inherit.aes = FALSE
  ) +
  # Spectra lines
  geom_line(aes(color = stage, linetype = stage, linewidth = stage), alpha = 0.9) +
  facet_wrap(~ facet_display, ncol = 2, scales = "free_y") +
  scale_color_manual(
    values = c("Non-cancer" = NORMAL_COLOR, "Early" = EARLY_COLOR, "Late" = LATE_COLOR),
    labels = c("Non-cancer" = paste0("Non-cancer (n=", nc_n, ")"),
               "Early" = "Early stage (I\u2013II)",
               "Late"  = "Late stage (III\u2013IV)")
  ) +
  scale_linetype_manual(
    values = c("Non-cancer" = "solid", "Early" = "solid", "Late" = "longdash"),
    labels = c("Non-cancer" = paste0("Non-cancer (n=", nc_n, ")"),
               "Early" = "Early stage (I\u2013II)",
               "Late"  = "Late stage (III\u2013IV)")
  ) +
  scale_linewidth_manual(
    values = c("Non-cancer" = 0.5, "Early" = 0.8, "Late" = 0.6),
    guide = "none"
  ) +
  labs(
    title = "SERS Spectral Signatures: Non-cancer \u2192 Early \u2192 Late Stage",
    subtitle = "Mean spectra per stage | Yellow bands = top 5% difference regions (Early vs Non-cancer)",
    x = expression("Wavenumber (cm"^{-1}*")"),
    y = "Intensity (a.u.)",
    color = NULL, linetype = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid.major = element_line(color = "#F0F0F0", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold", size = 11, hjust = 0, color = "#2C3E50"),
    plot.title = element_text(face = "bold", size = 16, color = "#2C3E50", hjust = 0.5),
    plot.subtitle = element_text(size = 10, color = "#888888", hjust = 0.5),
    axis.title  = element_text(size = 10),
    axis.text   = element_text(size = 8, color = "#333333"),
    legend.position = "bottom",
    legend.text = element_text(size = 11),
    legend.key.width = unit(1.8, "cm"),
    plot.margin = margin(8, 14, 6, 8),
    panel.spacing = unit(1.5, "lines")
  )

# ── Save ──
out_png <- file.path(FIG_DIR, "suppl_spectra_normal_early_late.png")
out_pdf <- file.path(FIG_DIR, "suppl_spectra_normal_early_late.pdf")

ggsave(out_png, p, width = 14, height = 8, dpi = 300, bg = "white")
ggsave(out_pdf, p, width = 14, height = 8, bg = "white")

cat("Saved:", out_png, "\n")
cat("Saved:", out_pdf, "\n")
cat("Done.\n")
