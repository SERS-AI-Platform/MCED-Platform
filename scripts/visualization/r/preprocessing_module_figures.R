#!/usr/bin/env Rscript

# Generate one clean 300 dpi figure per preprocessing module.
# This mirrors the current project preprocessing defaults:
# calibration -> trim -> Savitzky-Golay smoothing -> rolling-minimum baseline
# -> SNV normalization -> fixed-grid resampling.

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default) {
  idx <- match(flag, args)
  if (!is.na(idx) && idx < length(args)) {
    return(args[[idx + 1]])
  }
  default
}

input_file <- get_arg(
  "--input",
  "/Users/ian/Desktop/임상데이터/20260508_Urine test/1. NOR/NOR 68_1.CSV"
)
output_dir <- get_arg("--output-dir", "figures/preprocessing_modules")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(ggplot2)
})

read_spectrum <- function(path) {
  if (!file.exists(path)) {
    set.seed(7)
    wn <- seq(50, 3300, length.out = 1700)
    baseline <- 12000 * exp(-wn / 1900) + 1200
    peaks <- 4500 * exp(-0.5 * ((wn - 1001.4) / 10)^2) +
      2500 * exp(-0.5 * ((wn - 720) / 18)^2) +
      3200 * exp(-0.5 * ((wn - 1450) / 28)^2)
    y <- baseline + peaks + rnorm(length(wn), sd = 120)
    return(data.frame(wn = wn, intensity = y))
  }
  df <- read.csv(path, header = FALSE)
  names(df) <- c("wn", "intensity")
  df
}

savgol_smooth <- function(y, window = 11, poly = 3) {
  if (window %% 2 == 0) window <- window + 1
  half <- floor(window / 2)
  n <- length(y)
  out <- numeric(n)
  for (i in seq_len(n)) {
    lo <- max(1, i - half)
    hi <- min(n, i + half)
    idx <- lo:hi
    x <- idx - i
    fit <- lm(y[idx] ~ poly(x, degree = poly, raw = TRUE))
    out[i] <- predict(fit, newdata = data.frame(x = 0))
  }
  out
}

rolling_min <- function(y, window = 101) {
  half <- floor(window / 2)
  n <- length(y)
  out <- numeric(n)
  for (i in seq_len(n)) {
    out[i] <- min(y[max(1, i - half):min(n, i + half)])
  }
  out
}

snv <- function(y) {
  sd_y <- sd(y)
  if (sd_y < 1e-10) {
    return(y - mean(y))
  }
  (y - mean(y)) / sd_y
}

module_theme <- function() {
  theme_minimal(base_size = 22) +
    theme(
      plot.title = element_text(size = 28, face = "bold", color = "#1f2937"),
      axis.title = element_text(size = 23, face = "bold"),
      axis.text = element_text(size = 20, color = "#374151"),
      legend.title = element_text(size = 21, face = "bold"),
      legend.text = element_text(size = 20),
      legend.position = "top",
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "#e5e7eb", linewidth = 0.45),
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      plot.margin = margin(18, 22, 18, 18)
    )
}

save_module <- function(plot, filename, width = 12, height = 7) {
  ggsave(
    file.path(output_dir, filename),
    plot,
    width = width,
    height = height,
    dpi = 300,
    bg = "white"
  )
}

raw <- read_spectrum(input_file)

target <- 1001.4
cal_window <- 10.0
cal_region <- raw[raw$wn >= target - cal_window & raw$wn <= target + cal_window, ]
cal_region$smooth <- savgol_smooth(cal_region$intensity, window = 7, poly = 2)
detected <- cal_region$wn[which.max(cal_region$smooth)]
shift <- target - detected
raw$wn_cal <- raw$wn + shift

trim_min <- 400
trim_max <- 2200
trimmed <- raw[raw$wn_cal >= trim_min & raw$wn_cal <= trim_max, ]
trimmed$smooth <- savgol_smooth(trimmed$intensity, window = 11, poly = 3)
trimmed$baseline <- rolling_min(trimmed$smooth, window = 101)
trimmed$corrected <- trimmed$smooth - trimmed$baseline
trimmed$snv <- snv(trimmed$corrected)

fixed_grid <- seq(402.0, 2198.0, length.out = 935)
fixed_intensity <- approx(trimmed$wn_cal, trimmed$snv, xout = fixed_grid)$y
fixed <- data.frame(wn = fixed_grid, intensity = fixed_intensity)

p1_df <- rbind(
  data.frame(
    wn = raw$wn,
    intensity = raw$intensity,
    Spectrum = "Before calibration"
  ),
  data.frame(
    wn = raw$wn_cal,
    intensity = raw$intensity,
    Spectrum = "After calibration"
  )
)
p1 <- ggplot(
  p1_df[p1_df$wn >= 970 & p1_df$wn <= 1035, ],
  aes(wn, intensity, color = Spectrum)
) +
  geom_line(linewidth = 1.05) +
  geom_vline(xintercept = target, linetype = "dashed", linewidth = 1.0, color = "#111827") +
  scale_color_manual(values = c("Before calibration" = "#6b7280", "After calibration" = "#2563eb")) +
  labs(title = "Wavenumber Calibration", x = "Raman shift (cm-1)", y = "Intensity", color = "Spectrum") +
  module_theme()
save_module(p1, "01_wavenumber_calibration.png")

p2_df <- raw
p2_df$Region <- ifelse(p2_df$wn_cal >= trim_min & p2_df$wn_cal <= trim_max, "Kept", "Excluded")
p2 <- ggplot(p2_df, aes(wn_cal, intensity, color = Region)) +
  geom_line(linewidth = 0.85) +
  geom_vline(xintercept = c(trim_min, trim_max), linetype = "dashed", linewidth = 1.0, color = "#111827") +
  scale_color_manual(values = c("Kept" = "#059669", "Excluded" = "#9ca3af")) +
  labs(title = "Fingerprint Trim", x = "Raman shift (cm-1)", y = "Intensity", color = "Region") +
  module_theme()
save_module(p2, "02_fingerprint_trim.png")

p3_df <- rbind(
  data.frame(wn = trimmed$wn_cal, intensity = trimmed$intensity, Spectrum = "Trimmed raw"),
  data.frame(wn = trimmed$wn_cal, intensity = trimmed$smooth, Spectrum = "Savitzky-Golay")
)
p3 <- ggplot(p3_df, aes(wn, intensity, color = Spectrum, linetype = Spectrum)) +
  geom_line(linewidth = 0.95) +
  scale_color_manual(values = c("Trimmed raw" = "#9ca3af", "Savitzky-Golay" = "#7c3aed")) +
  scale_linetype_manual(values = c("Trimmed raw" = "dotted", "Savitzky-Golay" = "solid")) +
  labs(title = "Smoothing", x = "Raman shift (cm-1)", y = "Intensity", color = "Spectrum") +
  module_theme()
save_module(p3, "03_smoothing.png")

p4_df <- data.frame(
  wn = trimmed$wn_cal,
  intensity = trimmed$corrected,
  Spectrum = "Baseline-corrected"
)
p4 <- ggplot(p4_df, aes(wn, intensity, color = Spectrum)) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 1.0, color = "#6b7280") +
  geom_line(linewidth = 0.95) +
  scale_color_manual(values = c("Baseline-corrected" = "#dc2626")) +
  scale_y_continuous(limits = c(0, NA)) +
  labs(title = "Baseline Correction", x = "Raman shift (cm-1)", y = "Corrected intensity", color = "Spectrum") +
  module_theme()
save_module(p4, "04_baseline_correction.png")

p5 <- ggplot(trimmed, aes(wn_cal, snv, color = "SNV")) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 1.0, color = "#6b7280") +
  geom_line(linewidth = 0.95) +
  scale_color_manual(values = c("SNV" = "#ea580c")) +
  labs(title = "SNV Normalization", x = "Raman shift (cm-1)", y = "SNV intensity", color = "Spectrum") +
  module_theme()
save_module(p5, "05_snv_normalization.png")

p6 <- ggplot(fixed, aes(wn, intensity, color = "Fixed grid")) +
  geom_line(linewidth = 0.95) +
  geom_point(data = fixed[seq(1, nrow(fixed), by = 25), ], size = 1.6, alpha = 0.8) +
  scale_color_manual(values = c("Fixed grid" = "#0891b2")) +
  labs(title = "Fixed Grid Resampling", x = "Raman shift (cm-1)", y = "SNV intensity", color = "Feature space") +
  module_theme()
save_module(p6, "06_fixed_grid_resampling.png")

cat("Saved preprocessing module figures to ", output_dir, "\n", sep = "")
