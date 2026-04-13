#!/usr/bin/env Rscript
# ──────────────────────────────────────────────────────────────
# Cancer-specific discriminative peak extraction
# Method: Wilcoxon rank-sum test (each wavenumber, cancer vs control)
#         + log2 fold change + BH p-value correction
# ──────────────────────────────────────────────────────────────

library(dplyr)
library(readr)
library(tidyr)
library(tibble)
library(stringr)
library(ggplot2)
library(purrr)

# ── Paths ──
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) > 0) {
  root <- normalizePath(file.path(dirname(script_path), ".."))
} else {
  root <- normalizePath("..")
}
spec_path <- file.path(root, "results", "processed_spectra.csv")
out_dir   <- file.path(root, "AACR")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cat("Loading spectra from:", spec_path, "\n")

# ── Load data ──
spec <- read_csv(spec_path, show_col_types = FALSE)

# Feature columns (x_402.00, x_404.xx, ...)
wn_cols <- grep("^x_", names(spec), value = TRUE)
wavenumbers <- as.numeric(str_replace(wn_cols, "^x_", ""))

cat(sprintf("Loaded %d samples, %d wavenumbers (%.0f – %.0f cm⁻¹)\n",
            nrow(spec), length(wn_cols), min(wavenumbers), max(wavenumbers)))

# ── Group mapping ──
non_cancer <- c("NOR", "DIA", "HBP", "H.D.")
cancer_types <- c("PRO", "OVA", "LUN", "CPAN", "CRC")
display_names <- c(PRO = "Prostate (PRC)", OVA = "Ovarian (OVC)",
                   LUN = "Lung (LC)", CPAN = "Pancreatic (PAC)",
                   CRC = "Colorectal (CRC)")

spec <- spec %>%
  mutate(
    label = case_when(
      group %in% non_cancer  ~ "Non-Cancer",
      group %in% cancer_types ~ group,
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(label))

# Per-patient medoid: take first replicate per (group, sample_id)
spec_med <- spec %>%
  group_by(group, sample_id) %>%
  slice(1) %>%
  ungroup()

cat(sprintf("After medoid: %d patients\n", nrow(spec_med)))

# ── Non-cancer reference ──
nc_mat <- spec_med %>%
  filter(label == "Non-Cancer") %>%
  select(all_of(wn_cols)) %>%
  as.matrix()

cat(sprintf("Non-Cancer patients: %d\n", nrow(nc_mat)))

# ── Per-cancer peak extraction ──
all_results <- list()

for (cancer in cancer_types) {

  cat(sprintf("\n══ %s (%s) ══\n", cancer, display_names[cancer]))

  ca_mat <- spec_med %>%
    filter(label == cancer) %>%
    select(all_of(wn_cols)) %>%
    as.matrix()

  n_cancer <- nrow(ca_mat)
  n_nc     <- nrow(nc_mat)
  cat(sprintf("  n_cancer = %d, n_control = %d\n", n_cancer, n_nc))

  # Wilcoxon test at each wavenumber
  results <- tibble(
    wavenumber = wavenumbers,
    col        = wn_cols,
    mean_cancer  = colMeans(ca_mat),
    mean_control = colMeans(nc_mat),
    sd_cancer    = apply(ca_mat, 2, sd),
    sd_control   = apply(nc_mat, 2, sd)
  )

  # Wilcoxon p-values (vectorised with suppressWarnings for ties)
  pvals <- map_dbl(seq_along(wn_cols), function(j) {
    suppressWarnings(
      wilcox.test(ca_mat[, j], nc_mat[, j],
                  exact = FALSE, correct = TRUE)$p.value
    )
  })

  results <- results %>%
    mutate(
      p_value    = pvals,
      p_adj      = p.adjust(p_value, method = "BH"),
      log2_fc    = log2((mean_cancer + 1e-6) / (mean_control + 1e-6)),
      abs_diff   = mean_cancer - mean_control,
      neg_log10p = -log10(p_adj),
      cancer     = cancer,
      display    = display_names[cancer]
    )

  # Significant peaks: BH-adjusted p < 0.01 & |log2FC| > 0.2
  sig <- results %>%
    filter(p_adj < 0.01, abs(log2_fc) > 0.2) %>%
    arrange(p_adj)

  cat(sprintf("  Significant wavenumbers (p_adj<0.01, |log2FC|>0.2): %d / %d\n",
              nrow(sig), nrow(results)))

  if (nrow(sig) > 0) {
    # Cluster adjacent significant peaks into regions
    sig_sorted <- sig %>% arrange(wavenumber)
    sig_sorted$region <- cumsum(c(TRUE, diff(sig_sorted$wavenumber) > 15))

    regions <- sig_sorted %>%
      group_by(region) %>%
      summarise(
        wn_start   = min(wavenumber),
        wn_end     = max(wavenumber),
        wn_center  = wavenumber[which.min(p_adj)],
        best_p_adj = min(p_adj),
        mean_fc    = mean(log2_fc),
        n_points   = n(),
        .groups    = "drop"
      ) %>%
      arrange(best_p_adj)

    cat("  Top peak regions:\n")
    for (i in seq_len(min(10, nrow(regions)))) {
      r <- regions[i, ]
      direction <- ifelse(r$mean_fc > 0, "↑ cancer", "↓ cancer")
      cat(sprintf("    %4.0f–%4.0f cm⁻¹ (center: %.0f) | p_adj = %.2e | log2FC = %+.3f (%s)\n",
                  r$wn_start, r$wn_end, r$wn_center, r$best_p_adj, r$mean_fc, direction))
    }
  }

  all_results[[cancer]] <- results
}

# ── Combine and save ──
df_all <- bind_rows(all_results)

csv_out <- file.path(out_dir, "discriminative_peaks_all.csv")
write_csv(df_all, csv_out)
cat(sprintf("\n✓ Full results saved: %s\n", csv_out))

# Save top peaks per cancer
top_peaks <- df_all %>%
  filter(p_adj < 0.01, abs(log2_fc) > 0.2) %>%
  group_by(cancer) %>%
  arrange(p_adj) %>%
  slice_head(n = 30) %>%
  ungroup() %>%
  select(cancer, display, wavenumber, mean_cancer, mean_control,
         log2_fc, abs_diff, p_value, p_adj, neg_log10p)

csv_top <- file.path(out_dir, "top_discriminative_peaks.csv")
write_csv(top_peaks, csv_top)
cat(sprintf("✓ Top peaks saved: %s\n", csv_top))

# ──────────────────────────────────────────────────────────────
# VOLCANO PLOT: log2FC vs -log10(p_adj) per cancer type
# ──────────────────────────────────────────────────────────────

cancer_colors <- c(
  PRO  = "#8B4513",
  OVA  = "#6A5ACD",
  LUN  = "#2E8B57",
  CPAN = "#CD5C5C",
  CRC  = "#4682B4"
)

for (cancer in cancer_types) {

  dfc <- df_all %>% filter(cancer == !!cancer)

  dfc <- dfc %>%
    mutate(
      sig_class = case_when(
        p_adj < 0.01 & log2_fc >  0.2 ~ "Up in cancer",
        p_adj < 0.01 & log2_fc < -0.2 ~ "Down in cancer",
        TRUE ~ "NS"
      )
    )

  # Top 5 most significant for labelling
  top5 <- dfc %>%
    filter(sig_class != "NS") %>%
    arrange(p_adj) %>%
    slice_head(n = 5)

  p <- ggplot(dfc, aes(x = log2_fc, y = neg_log10p)) +
    geom_point(aes(colour = sig_class), size = 0.6, alpha = 0.7) +
    scale_colour_manual(
      values = c("Up in cancer"   = cancer_colors[cancer],
                 "Down in cancer" = "#7F8C8D",
                 "NS"             = "#D5D5D5"),
      name = NULL
    ) +
    geom_hline(yintercept = -log10(0.01), linetype = "dashed",
               colour = "#BBBBBB", linewidth = 0.3) +
    geom_vline(xintercept = c(-0.2, 0.2), linetype = "dashed",
               colour = "#BBBBBB", linewidth = 0.3) +
    labs(
      title = display_names[cancer],
      x = expression(log[2]~fold~change~(cancer / control)),
      y = expression(-log[10]~adjusted~italic(p))
    ) +
    theme_minimal(base_size = 8, base_family = "sans") +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.2, colour = "#EEEEEE"),
      plot.title = element_text(face = "bold", size = 9,
                                colour = cancer_colors[cancer]),
      legend.position = "bottom",
      legend.key.size = unit(0.3, "cm")
    )

  # Label top peaks
  if (nrow(top5) > 0) {
    p <- p +
      ggrepel::geom_text_repel(
        data = top5,
        aes(label = sprintf("%.0f", wavenumber)),
        size = 2.2, colour = cancer_colors[cancer],
        segment.size = 0.2, segment.colour = "#AAAAAA",
        max.overlaps = 10, box.padding = 0.3
      )
  }

  fig_path <- file.path(out_dir, sprintf("volcano_%s.pdf", tolower(cancer)))
  ggsave(fig_path, p, width = 89, height = 75, units = "mm", dpi = 300)
  cat(sprintf("✓ Volcano plot: %s\n", fig_path))

  fig_path_png <- file.path(out_dir, sprintf("volcano_%s.png", tolower(cancer)))
  ggsave(fig_path_png, p, width = 89, height = 75, units = "mm", dpi = 300)
}

# ──────────────────────────────────────────────────────────────
# DIFFERENCE SPECTRUM PLOT: mean(cancer) - mean(control) per cancer
# with significant regions shaded
# ──────────────────────────────────────────────────────────────

diff_plots <- list()

for (cancer in cancer_types) {

  dfc <- df_all %>% filter(cancer == !!cancer)

  sig_regions <- dfc %>%
    filter(p_adj < 0.01, abs(log2_fc) > 0.2) %>%
    arrange(wavenumber) %>%
    mutate(region = cumsum(c(TRUE, diff(wavenumber) > 15))) %>%
    group_by(region) %>%
    summarise(xmin = min(wavenumber), xmax = max(wavenumber),
              direction = ifelse(mean(log2_fc) > 0, "up", "down"),
              .groups = "drop")

  p <- ggplot(dfc, aes(x = wavenumber, y = abs_diff)) +
    geom_line(colour = cancer_colors[cancer], linewidth = 0.4) +
    geom_hline(yintercept = 0, colour = "#999999", linewidth = 0.3)

  # Shade significant regions
  if (nrow(sig_regions) > 0) {
    p <- p +
      geom_rect(
        data = sig_regions,
        aes(xmin = xmin - 3, xmax = xmax + 3, ymin = -Inf, ymax = Inf),
        inherit.aes = FALSE,
        fill = cancer_colors[cancer], alpha = 0.15
      )
  }

  p <- p +
    labs(
      title = display_names[cancer],
      x = expression(Raman~shift~(cm^{-1})),
      y = expression(Delta~Intensity~(cancer - control))
    ) +
    theme_minimal(base_size = 8, base_family = "sans") +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.2, colour = "#EEEEEE"),
      plot.title = element_text(face = "bold", size = 9,
                                colour = cancer_colors[cancer])
    )

  diff_plots[[cancer]] <- p
}

# Combine into 5-panel figure
library(cowplot)
combined <- plot_grid(
  plotlist = diff_plots,
  ncol = 1, align = "v",
  labels = letters[1:5],
  label_size = 10, label_fontface = "bold"
)

fig_diff <- file.path(out_dir, "difference_spectra_significant.pdf")
ggsave(fig_diff, combined, width = 183, height = 240, units = "mm", dpi = 300)
cat(sprintf("✓ Combined difference spectra: %s\n", fig_diff))

fig_diff_png <- file.path(out_dir, "difference_spectra_significant.png")
ggsave(fig_diff_png, combined, width = 183, height = 240, units = "mm", dpi = 300)

cat("\n══ Done ══\n")
