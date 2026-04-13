#!/usr/bin/env Rscript
# ──────────────────────────────────────────────────────────────
# 각 암종별 Mean SERS Spectrum (cancer vs control)
# 통계적으로 유의한 피크 영역만 Up(빨강) / Down(파랑) shading
# 라벨: PRC, OVC, LC, PAC, CRC
# ──────────────────────────────────────────────────────────────

library(dplyr)
library(readr)
library(tidyr)
library(stringr)
library(ggplot2)
library(cowplot)

# ── Paths ──
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) > 0) {
  root <- normalizePath(file.path(dirname(script_path), ".."))
} else {
  root <- normalizePath("..")
}

out_dir <- file.path(root, "AACR")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ── extract_peaks.R 결과 로드 ──
peaks_path <- file.path(out_dir, "discriminative_peaks_all.csv")
if (!file.exists(peaks_path)) {
  stop("Run extract_peaks.R first to generate discriminative_peaks_all.csv")
}
peak_df <- read_csv(peaks_path, show_col_types = FALSE)

# ── Spectra 로드 ──
spec <- read_csv(file.path(root, "results", "processed_spectra.csv"),
                 show_col_types = FALSE)

wn_cols <- grep("^x_", names(spec), value = TRUE)
wavenumbers <- as.numeric(str_replace(wn_cols, "^x_", ""))

non_cancer  <- c("NOR", "DIA", "HBP", "H.D.")
cancer_types <- c("PRO", "OVA", "LUN", "CPAN", "CRC")

# ── 라벨: 사용자 지정 ──
short_labels <- c(PRO = "PRC", OVA = "OVC", LUN = "LC",
                  CPAN = "PAC", CRC = "CRC")

cancer_colors <- c(
  PRO  = "#8B4513",
  OVA  = "#6A5ACD",
  LUN  = "#2E8B57",
  CPAN = "#CD5C5C",
  CRC  = "#4682B4"
)

up_color   <- "#E74C3C"
down_color <- "#3498DB"

spec <- spec %>%
  mutate(label = case_when(
    group %in% non_cancer   ~ "Non-Cancer",
    group %in% cancer_types ~ group,
    TRUE ~ NA_character_
  )) %>%
  filter(!is.na(label))

# Per-patient medoid
spec_med <- spec %>%
  group_by(group, sample_id) %>%
  slice(1) %>%
  ungroup()

# ── Non-cancer mean ──
nc_mat  <- spec_med %>% filter(label == "Non-Cancer") %>%
  select(all_of(wn_cols)) %>% as.matrix()
nc_mean <- colMeans(nc_mat)
nc_sem  <- apply(nc_mat, 2, sd) / sqrt(nrow(nc_mat))

# ── 유의 영역 추출 (엄격 기준) ──
build_sig_regions <- function(dfc,
                              p_thresh  = 0.001,
                              fc_thresh = 0.3,
                              min_pts   = 5,
                              max_regions = 8) {

  sig <- dfc %>%
    filter(p_adj < p_thresh, abs(log2_fc) > fc_thresh) %>%
    arrange(wavenumber) %>%
    mutate(direction = ifelse(log2_fc > 0, "Up", "Down"))

  if (nrow(sig) == 0) return(tibble())

  sig <- sig %>%
    mutate(region_id = cumsum(c(TRUE, diff(wavenumber) > 15 |
                                      diff(as.numeric(factor(direction))) != 0)))

  regions <- sig %>%
    group_by(region_id, direction) %>%
    summarise(
      xmin    = min(wavenumber) - 2,
      xmax    = max(wavenumber) + 2,
      best_p  = min(p_adj),
      mean_fc = mean(log2_fc),
      n_pts   = n(),
      .groups = "drop"
    ) %>%
    filter(n_pts >= min_pts) %>%
    arrange(best_p) %>%
    slice_head(n = max_regions)

  return(regions)
}

# ── 플롯 생성 ──
plot_list <- list()

for (cancer in cancer_types) {

  ca_mat  <- spec_med %>% filter(label == cancer) %>%
    select(all_of(wn_cols)) %>% as.matrix()
  ca_mean <- colMeans(ca_mat)
  ca_sem  <- apply(ca_mat, 2, sd) / sqrt(nrow(ca_mat))
  n_ca    <- nrow(ca_mat)

  lbl <- short_labels[cancer]

  spec_long <- tibble(
    wavenumber = rep(wavenumbers, 2),
    intensity  = c(nc_mean, ca_mean),
    sem        = c(nc_sem, ca_sem),
    grp        = rep(c("Control", lbl), each = length(wavenumbers))
  )

  dfc     <- peak_df %>% filter(cancer == !!cancer)
  regions <- build_sig_regions(dfc)

  y_max <- max(c(nc_mean + nc_sem, ca_mean + ca_sem), na.rm = TRUE)

  p <- ggplot() +

    # ── Shading ──
    {if (nrow(regions) > 0) {
      geom_rect(
        data = regions,
        aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf,
            fill = direction),
        alpha = 0.13, inherit.aes = FALSE
      )
    }} +
    scale_fill_manual(
      values = c("Up" = up_color, "Down" = down_color),
      labels = c("Up"   = expression(""*uparrow*" Cancer > Control"),
                 "Down" = expression(""*downarrow*" Cancer < Control")),
      name = NULL, drop = FALSE
    ) +

    # ── SEM band ──
    geom_ribbon(
      data = spec_long %>% filter(grp == "Control"),
      aes(x = wavenumber, ymin = intensity - sem, ymax = intensity + sem),
      fill = "grey70", alpha = 0.18
    ) +
    geom_ribbon(
      data = spec_long %>% filter(grp == lbl),
      aes(x = wavenumber, ymin = intensity - sem, ymax = intensity + sem),
      fill = cancer_colors[cancer], alpha = 0.13
    ) +

    # ── Mean spectra ──
    geom_line(
      data = spec_long %>% filter(grp == "Control"),
      aes(x = wavenumber, y = intensity),
      colour = "grey50", linewidth = 0.3, alpha = 0.65
    ) +
    geom_line(
      data = spec_long %>% filter(grp == lbl),
      aes(x = wavenumber, y = intensity),
      colour = cancer_colors[cancer], linewidth = 0.45
    ) +

    # ── Wavenumber labels on significant regions ──
    {if (nrow(regions) > 0) {
      geom_text(
        data = regions %>% mutate(wn_mid = (xmin + xmax) / 2),
        aes(x = wn_mid, y = y_max * 1.02,
            label = sprintf("%.0f", wn_mid)),
        size = 1.8, colour = "grey30", inherit.aes = FALSE
      )
    }} +

    labs(
      x     = expression(Wavenumber~(cm^{-1})),
      y     = "Intensity (a.u.)"
    ) +

    theme_classic(base_size = 8, base_family = "sans") +
    theme(
      axis.title       = element_text(size = 7.5),
      axis.text        = element_text(size = 6.5),
      legend.position  = "none",
      plot.margin      = margin(2, 6, 2, 4)
    ) +

    # ── 인라인 legend ──
    annotate("segment", x = wavenumbers[1] + 15, xend = wavenumbers[1] + 65,
             y = y_max * 0.96, yend = y_max * 0.96,
             colour = "grey50", linewidth = 0.35) +
    annotate("text", x = wavenumbers[1] + 72, y = y_max * 0.96,
             label = "Control", hjust = 0, size = 2, colour = "grey50") +
    annotate("segment", x = wavenumbers[1] + 15, xend = wavenumbers[1] + 65,
             y = y_max * 0.88, yend = y_max * 0.88,
             colour = cancer_colors[cancer], linewidth = 0.5) +
    annotate("text", x = wavenumbers[1] + 72, y = y_max * 0.88,
             label = sprintf("%s (n=%d)", lbl, n_ca),
             hjust = 0, size = 2, colour = cancer_colors[cancer])

  plot_list[[cancer]] <- p

  # 개별 저장
  fname <- sprintf("spectrum_%s", tolower(lbl))
  ggsave(file.path(out_dir, paste0(fname, ".pdf")), p,
         width = 150, height = 55, units = "mm", dpi = 300)
  ggsave(file.path(out_dir, paste0(fname, ".png")), p,
         width = 150, height = 55, units = "mm", dpi = 300)
  cat(sprintf("  %s\n", fname))
}

# ── 공통 shading legend (별도 grob) ──
legend_plot <- ggplot() +
  geom_rect(aes(xmin = 0, xmax = 1, ymin = 0, ymax = 1, fill = "Up"),
            alpha = 0.15) +
  geom_rect(aes(xmin = 2, xmax = 3, ymin = 0, ymax = 1, fill = "Down"),
            alpha = 0.15) +
  scale_fill_manual(
    values = c("Up" = up_color, "Down" = down_color),
    labels = c("Up"   = "Cancer > Control",
               "Down" = "Cancer < Control"),
    name = NULL
  ) +
  theme_void(base_size = 8) +
  theme(
    legend.position  = "bottom",
    legend.direction = "horizontal",
    legend.text      = element_text(size = 7),
    legend.key.size  = unit(0.35, "cm")
  )

shared_legend <- get_legend(legend_plot)

# ── 5패널 합본 (세로 정렬) ──
panels <- plot_grid(
  plotlist = plot_list,
  ncol = 1, align = "v",
  labels = as.character(short_labels),
  label_size = 10, label_fontface = "bold"
)

combined <- plot_grid(
  panels, shared_legend,
  ncol = 1, rel_heights = c(1, 0.04)
)

ggsave(file.path(out_dir, "spectra_all_cancers_peaks.pdf"), combined,
       width = 170, height = 230, units = "mm", dpi = 300)
ggsave(file.path(out_dir, "spectra_all_cancers_peaks.png"), combined,
       width = 170, height = 230, units = "mm", dpi = 300)
cat("  spectra_all_cancers_peaks\n")

cat("\nDone\n")
