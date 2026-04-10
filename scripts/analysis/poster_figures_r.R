#!/usr/bin/env Rscript
# ── Publication-quality poster figures (ggplot2) ──

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
  library(scales)
  library(RColorBrewer)
})

out_dir <- "results/poster_figures"

# ═══════════════════════════════════════════════
# Theme & palette
# ═══════════════════════════════════════════════
theme_poster <- theme_minimal(base_size = 11) +
  theme(
    text             = element_text(family = "sans"),
    plot.title       = element_text(face = "bold", size = 13, hjust = 0),
    plot.subtitle    = element_text(size = 9, color = "grey40"),
    axis.title       = element_text(size = 11),
    axis.text        = element_text(size = 9),
    legend.title     = element_text(face = "bold", size = 9),
    legend.text      = element_text(size = 8),
    panel.grid.major = element_line(color = "grey92", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    strip.text       = element_text(face = "bold", size = 10),
    strip.background = element_rect(fill = "grey96", color = NA),
    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.margin      = margin(8, 12, 8, 8)
  )
theme_set(theme_poster)

# Axis labels (consistent everywhere)
xlab_wn <- expression("Wavenumber (cm"^{-1}*")")
ylab_int <- "Intensity (a.u.)"

# Fixed axis limits for spectra
Y_LIM   <- c(-3, 13)
X_BREAKS <- seq(500, 2200, 250)
X_LIM   <- c(400, 2200)

# Color palettes
cancer_pal <- c(
  LUN  = "#D94F4F", CRC  = "#E8813A", BLC  = "#C9A832",
  PRO  = "#47A76A", OVA  = "#3BA5A5", CPAN = "#4A90D9",
  YPAN = "#7E57C2", BRE  = "#C2185B"
)
noncancer_pal <- c(
  NOR  = "#90A4AE", YNOR = "#B0BEC5",
  DIA  = "#607D8B", HBP  = "#455A64", H.D. = "#37474F"
)
all_pal <- c(cancer_pal, noncancer_pal)

pan_pal <- c(CPAN = "#4A90D9", YPAN = "#7E57C2", SPAN = "#E8813A")

group_order <- c("LUN","CRC","BLC","PRO","OVA","CPAN","YPAN","BRE",
                 "NOR","YNOR","DIA","HBP","H.D.")

group_labels <- c(
  LUN="Lung", CRC="Colorectal", BLC="Bladder", PRO="Prostate",
  OVA="Ovarian", CPAN="Pancreatic\n(CBNU)", YPAN="Pancreatic\n(Severance)",
  BRE="Breast", NOR="Normal", YNOR="Normal\n(Severance)",
  DIA="Diabetes", HBP="Hypertension", H.D.="HTN+DM"
)

group_labels_short <- c(
  LUN="LUN", CRC="CRC", BLC="BLC", PRO="PRO",
  OVA="OVA", CPAN="CPAN", YPAN="YPAN",
  BRE="BRE", NOR="NOR", YNOR="YNOR",
  DIA="DIA", HBP="HBP", H.D.="H.D."
)

# ═══════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════
cat("[1] Loading data...\n")
cohort <- read.csv(file.path(out_dir, "cohort_clinical_for_r.csv"),
                   stringsAsFactors = FALSE) %>%
  filter(disease_group %in% group_order) %>%
  mutate(disease_group = factor(disease_group, levels = group_order),
         category = ifelse(disease_group %in% names(cancer_pal), "Cancer", "Non-cancer"))

span <- read.csv(file.path(out_dir, "span_clinical_for_r.csv"),
                 stringsAsFactors = FALSE)

mean_spec <- read.csv(file.path(out_dir, "mean_spectra_for_r.csv"),
                      stringsAsFactors = FALSE) %>%
  filter(!group %in% c("SPAN", "PS", "SENSOR", "UNK"))

ind_spec <- read.csv(file.path(out_dir, "individual_spectra_for_r.csv"),
                     stringsAsFactors = FALSE) %>%
  filter(!group %in% c("SPAN", "PS", "SENSOR", "UNK"))

pan_mean <- read.csv(file.path(out_dir, "pancreatic_mean_spectra_for_r.csv"),
                     stringsAsFactors = FALSE)

pan_ind <- read.csv(file.path(out_dir, "pancreatic_individual_spectra_for_r.csv"),
                    stringsAsFactors = FALSE)


# ═══════════════════════════════════════════════
# Fig 1: Demography Overview (4-panel)
# ═══════════════════════════════════════════════
cat("[2] Fig 1: Demography...\n")

counts <- cohort %>%
  count(disease_group, category) %>%
  mutate(disease_group = factor(disease_group, levels = rev(group_order)))

p1a <- ggplot(counts, aes(x = disease_group, y = n, fill = category)) +
  geom_col(width = 0.7, show.legend = FALSE) +
  geom_text(aes(label = n), hjust = -0.15, size = 3, color = "grey30") +
  scale_fill_manual(values = c("Cancer" = "#D94F4F", "Non-cancer" = "#4A90D9")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  coord_flip() +
  scale_x_discrete(labels = function(x) group_labels[x]) +
  labs(title = "A. Cohort Composition", subtitle = "N = 1,628 subjects",
       x = NULL, y = "Number of Subjects") +
  theme(panel.grid.major.y = element_blank())

p1b <- ggplot(cohort %>% filter(!is.na(age)),
              aes(x = disease_group, y = age, fill = category)) +
  geom_boxplot(width = 0.6, outlier.size = 0.8, outlier.alpha = 0.5,
               linewidth = 0.3, show.legend = FALSE) +
  scale_fill_manual(values = c("Cancer" = "#FADBD8", "Non-cancer" = "#D4E6F1")) +
  scale_x_discrete(labels = function(x) group_labels_short[x]) +
  labs(title = "B. Age Distribution", x = NULL, y = "Age (years)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

sex_data <- cohort %>%
  filter(!is.na(sex)) %>%
  count(disease_group, sex) %>%
  group_by(disease_group) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ungroup()

p1c <- ggplot(sex_data, aes(x = disease_group, y = n, fill = sex)) +
  geom_col(width = 0.65, position = "stack") +
  scale_fill_manual(values = c("M" = "#5DADE2", "F" = "#EC7063"),
                    labels = c("M" = "Male", "F" = "Female")) +
  scale_x_discrete(labels = function(x) group_labels_short[x]) +
  labs(title = "C. Sex Distribution", x = NULL, y = "Subjects", fill = NULL) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position.inside = c(0.85, 0.85),
        legend.position = "inside",
        legend.background = element_rect(fill = "white", color = "grey80", linewidth = 0.3))

p1d <- ggplot(cohort %>% filter(!is.na(bmi)),
              aes(x = disease_group, y = bmi, fill = category)) +
  geom_boxplot(width = 0.6, outlier.size = 0.8, outlier.alpha = 0.5,
               linewidth = 0.3, show.legend = FALSE) +
  scale_fill_manual(values = c("Cancer" = "#FADBD8", "Non-cancer" = "#D4E6F1")) +
  scale_x_discrete(labels = function(x) group_labels_short[x]) +
  labs(title = "D. BMI Distribution", x = NULL, y = expression("BMI (kg/m"^2*")")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

fig1 <- (p1a | p1b) / (p1c | p1d) +
  plot_annotation(
    title = "Cohort Demographics",
    theme = theme(plot.title = element_text(face = "bold", size = 16, hjust = 0.5))
  )

ggsave(file.path(out_dir, "fig1_demography_overview.png"), fig1,
       width = 14, height = 10, dpi = 300, bg = "white")
ggsave(file.path(out_dir, "fig1_demography_overview.pdf"), fig1,
       width = 14, height = 10, bg = "white")
cat("  Saved: fig1\n")


# ═══════════════════════════════════════════════
# Fig 2: Individual spectra by group (faceted, fixed axes)
# ═══════════════════════════════════════════════
cat("[3] Fig 2: Individual spectra...\n")

ind_spec <- ind_spec %>%
  mutate(group = factor(group, levels = group_order))

facet_labels <- setNames(
  paste0(group_labels_short[group_order], " (n=",
         cohort %>% count(disease_group) %>%
           arrange(match(disease_group, group_order)) %>% pull(n),
         ")"),
  group_order
)

fig2 <- ggplot() +
  geom_line(data = ind_spec,
            aes(x = wavenumber, y = intensity, group = spec_id, color = group),
            alpha = 0.12, linewidth = 0.25, show.legend = FALSE) +
  geom_line(data = mean_spec %>%
              filter(group %in% group_order) %>%
              mutate(group = factor(group, levels = group_order)),
            aes(x = wavenumber, y = intensity, color = group),
            linewidth = 0.7, show.legend = FALSE) +
  facet_wrap(~ group, ncol = 4,
             labeller = labeller(group = facet_labels)) +
  scale_color_manual(values = all_pal) +
  scale_x_continuous(limits = X_LIM, breaks = seq(500, 2000, 500)) +
  scale_y_continuous(limits = Y_LIM) +
  labs(title = "Individual SERS Spectra by Group",
       subtitle = "Thin lines = individual spectra (n=20 sampled), bold = group mean",
       x = xlab_wn, y = ylab_int) +
  theme(strip.text = element_text(size = 9))

ggsave(file.path(out_dir, "fig2_individual_spectra.png"), fig2,
       width = 16, height = 14, dpi = 300, bg = "white")
ggsave(file.path(out_dir, "fig2_individual_spectra.pdf"), fig2,
       width = 16, height = 14, bg = "white")
cat("  Saved: fig2\n")


# ═══════════════════════════════════════════════
# Fig 3: Mean spectra overlay (cancer / non-cancer, fixed axes)
# ═══════════════════════════════════════════════
cat("[4] Fig 3: Mean spectra overlay...\n")

cancer_groups   <- c("LUN","CRC","BLC","PRO","OVA","CPAN","YPAN","BRE")
noncancer_groups <- c("NOR","YNOR","DIA","HBP","H.D.")

n_labels <- cohort %>%
  count(disease_group) %>%
  mutate(label = paste0(disease_group, " (n=", n, ")"))
n_map <- setNames(n_labels$label, n_labels$disease_group)

# Y limits for mean spectra (tighter since these are means)
Y_LIM_MEAN <- c(-2, 6)

p3a <- ggplot(mean_spec %>% filter(group %in% cancer_groups) %>%
                mutate(group = factor(group, levels = cancer_groups)),
              aes(x = wavenumber, y = intensity, color = group)) +
  geom_ribbon(aes(ymin = intensity - sd, ymax = intensity + sd, fill = group),
              alpha = 0.08, color = NA) +
  geom_line(linewidth = 0.6) +
  scale_color_manual(values = cancer_pal, labels = function(x) n_map[x]) +
  scale_fill_manual(values = cancer_pal, guide = "none") +
  scale_x_continuous(limits = X_LIM, breaks = X_BREAKS) +
  scale_y_continuous(limits = Y_LIM_MEAN) +
  labs(title = "A. Cancer Groups", color = NULL,
       x = xlab_wn, y = ylab_int) +
  theme(legend.position = "right",
        legend.key.height = unit(0.4, "cm"))

p3b <- ggplot(mean_spec %>% filter(group %in% noncancer_groups) %>%
                mutate(group = factor(group, levels = noncancer_groups)),
              aes(x = wavenumber, y = intensity, color = group)) +
  geom_ribbon(aes(ymin = intensity - sd, ymax = intensity + sd, fill = group),
              alpha = 0.08, color = NA) +
  geom_line(linewidth = 0.6) +
  scale_color_manual(values = noncancer_pal, labels = function(x) n_map[x]) +
  scale_fill_manual(values = noncancer_pal, guide = "none") +
  scale_x_continuous(limits = X_LIM, breaks = X_BREAKS) +
  scale_y_continuous(limits = Y_LIM_MEAN) +
  labs(title = "B. Non-Cancer Groups", color = NULL,
       x = xlab_wn, y = ylab_int) +
  theme(legend.position = "right",
        legend.key.height = unit(0.4, "cm"))

fig3 <- p3a / p3b +
  plot_annotation(
    title = "Mean Preprocessed SERS Spectra (\u00b11 SD)",
    theme = theme(plot.title = element_text(face = "bold", size = 15, hjust = 0.5))
  )

ggsave(file.path(out_dir, "fig3_mean_spectra_all.png"), fig3,
       width = 14, height = 9, dpi = 300, bg = "white")
ggsave(file.path(out_dir, "fig3_mean_spectra_all.pdf"), fig3,
       width = 14, height = 9, bg = "white")
cat("  Saved: fig3\n")


# ═══════════════════════════════════════════════
# Fig 4: Pancreatic comparison (CPAN vs YPAN vs SPAN, fixed axes)
# ═══════════════════════════════════════════════
cat("[5] Fig 4: Pancreatic comparison...\n")

pan_order <- c("CPAN","YPAN","SPAN")
pan_labels_full <- c(
  CPAN = "CPAN (CBNU, pre-op, n=70)",
  YPAN = "YPAN (Severance, pre-op, n=30)",
  SPAN = "SPAN (Samsung, post-op, n=72)"
)

Y_LIM_PAN <- c(-2, 8)

# Panel A: Mean overlay with SD ribbon
p4a <- ggplot(pan_mean %>% mutate(group = factor(group, levels = pan_order)),
              aes(x = wavenumber, y = intensity, color = group)) +
  geom_ribbon(aes(ymin = intensity - sd, ymax = intensity + sd, fill = group),
              alpha = 0.12, color = NA) +
  geom_line(linewidth = 0.7) +
  scale_color_manual(values = pan_pal, labels = function(x) pan_labels_full[x]) +
  scale_fill_manual(values = pan_pal, guide = "none") +
  scale_x_continuous(limits = X_LIM, breaks = X_BREAKS) +
  scale_y_continuous(limits = Y_LIM_PAN) +
  labs(title = "A. Mean SERS Spectra \u2014 Pancreatic Cancer Subgroups (\u00b11 SD)",
       x = xlab_wn, y = ylab_int, color = NULL) +
  theme(legend.position.inside = c(0.75, 0.88),
        legend.position = "inside",
        legend.background = element_rect(fill = alpha("white", 0.9), color = "grey80", linewidth = 0.3),
        legend.key.height = unit(0.4, "cm"))

# Panels B-D: Individual spectra per subgroup (FIXED y-axis)
make_ind_panel <- function(grp, title_label) {
  sub_ind <- pan_ind %>% filter(group == grp)
  sub_mean <- pan_mean %>% filter(group == grp)

  ggplot() +
    geom_line(data = sub_ind,
              aes(x = wavenumber, y = intensity, group = spec_id),
              color = pan_pal[grp], alpha = 0.15, linewidth = 0.25) +
    geom_line(data = sub_mean,
              aes(x = wavenumber, y = intensity),
              color = pan_pal[grp], linewidth = 0.8) +
    scale_x_continuous(limits = X_LIM, breaks = seq(500, 2000, 500)) +
    scale_y_continuous(limits = Y_LIM_PAN) +
    labs(title = title_label, x = xlab_wn, y = ylab_int)
}

p4b <- make_ind_panel("CPAN", "B. CPAN \u2014 Individual (CBNU, n=70)")
p4c <- make_ind_panel("YPAN", "C. YPAN \u2014 Individual (Severance, n=30)")
p4d <- make_ind_panel("SPAN", "D. SPAN \u2014 Individual (Samsung, post-op, n=72)")

# Panel E: Difference spectra
cpan_mean_vals <- pan_mean %>% filter(group == "CPAN") %>% arrange(wavenumber)
ypan_mean_vals <- pan_mean %>% filter(group == "YPAN") %>% arrange(wavenumber)
span_mean_vals <- pan_mean %>% filter(group == "SPAN") %>% arrange(wavenumber)

diff_df <- bind_rows(
  data.frame(wavenumber = cpan_mean_vals$wavenumber,
             diff = cpan_mean_vals$intensity - span_mean_vals$intensity,
             comparison = "CPAN - SPAN"),
  data.frame(wavenumber = ypan_mean_vals$wavenumber,
             diff = ypan_mean_vals$intensity - span_mean_vals$intensity,
             comparison = "YPAN - SPAN")
)

p4e <- ggplot(diff_df, aes(x = wavenumber, y = diff, color = comparison)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey60", linewidth = 0.3) +
  geom_line(linewidth = 0.6) +
  scale_color_manual(values = c("CPAN - SPAN" = "#4A90D9",
                                 "YPAN - SPAN" = "#7E57C2")) +
  scale_x_continuous(limits = X_LIM, breaks = seq(500, 2000, 500)) +
  labs(title = "E. Difference Spectra (vs SPAN)",
       x = xlab_wn,
       y = expression(Delta ~ "Intensity (a.u.)"), color = NULL) +
  theme(legend.position.inside = c(0.8, 0.85),
        legend.position = "inside",
        legend.background = element_rect(fill = alpha("white", 0.9), color = "grey80", linewidth = 0.3))

fig4 <- p4a / (p4b | p4c) / (p4d | p4e) +
  plot_annotation(
    title = "Pancreatic Cancer Subgroup Comparison:\nPre-operative (CPAN, YPAN) vs Post-operative (SPAN)",
    theme = theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5))
  )

ggsave(file.path(out_dir, "fig4_pancreatic_comparison.png"), fig4,
       width = 14, height = 14, dpi = 300, bg = "white")
ggsave(file.path(out_dir, "fig4_pancreatic_comparison.pdf"), fig4,
       width = 14, height = 14, bg = "white")
cat("  Saved: fig4\n")


# ═══════════════════════════════════════════════
# Fig 5: Pancreatic demography comparison
# ═══════════════════════════════════════════════
cat("[6] Fig 5: Pancreatic demography...\n")

pan_clin <- bind_rows(
  cohort %>% filter(disease_group %in% c("CPAN","YPAN")) %>%
    mutate(disease_group = as.character(disease_group)),
  span %>% mutate(disease_group = "SPAN")
) %>%
  mutate(group = factor(disease_group, levels = pan_order))

pan_n <- pan_clin %>% count(group) %>%
  mutate(label = paste0(group, "\n(n=", n, ")"))
pan_n_map <- setNames(pan_n$label, pan_n$group)

p5a <- ggplot(pan_clin %>% filter(!is.na(age)),
              aes(x = group, y = age, fill = group)) +
  geom_boxplot(width = 0.5, outlier.size = 1, outlier.alpha = 0.5,
               linewidth = 0.4, show.legend = FALSE) +
  scale_fill_manual(values = pan_pal) +
  scale_x_discrete(labels = function(x) pan_n_map[x]) +
  labs(title = "A. Age", x = NULL, y = "Age (years)")

p5b <- ggplot(pan_clin %>% filter(!is.na(bmi)),
              aes(x = group, y = bmi, fill = group)) +
  geom_boxplot(width = 0.5, outlier.size = 1, outlier.alpha = 0.5,
               linewidth = 0.4, show.legend = FALSE) +
  scale_fill_manual(values = pan_pal) +
  scale_x_discrete(labels = function(x) pan_n_map[x]) +
  labs(title = "B. BMI", x = NULL, y = expression("BMI (kg/m"^2*")"))

pan_sex <- pan_clin %>%
  filter(!is.na(sex)) %>%
  count(group, sex) %>%
  group_by(group) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ungroup()

p5c <- ggplot(pan_sex, aes(x = group, y = n, fill = sex)) +
  geom_col(width = 0.5, position = "stack") +
  scale_fill_manual(values = c("M" = "#5DADE2", "F" = "#EC7063"),
                    labels = c("M" = "Male", "F" = "Female")) +
  scale_x_discrete(labels = function(x) pan_n_map[x]) +
  labs(title = "C. Sex", x = NULL, y = "Subjects", fill = NULL) +
  theme(legend.position.inside = c(0.85, 0.85),
        legend.position = "inside",
        legend.background = element_rect(fill = "white", color = "grey80", linewidth = 0.3))

fig5 <- p5a | p5b | p5c
fig5 <- fig5 + plot_annotation(
  title = "Pancreatic Subgroup Demographics",
  theme = theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5))
)

ggsave(file.path(out_dir, "fig5_pancreatic_demography.png"), fig5,
       width = 13, height = 5, dpi = 300, bg = "white")
ggsave(file.path(out_dir, "fig5_pancreatic_demography.pdf"), fig5,
       width = 13, height = 5, bg = "white")
cat("  Saved: fig5\n")

cat("\n=== All figures saved to:", out_dir, "===\n")
