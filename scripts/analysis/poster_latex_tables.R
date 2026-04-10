#!/usr/bin/env Rscript
# ── LaTeX Demography Tables for Poster ──

library(dplyr)
library(tidyr)
library(xtable)

out_dir <- "results/poster_figures"

cohort <- read.csv(file.path(out_dir, "cohort_clinical_for_r.csv"),
                   stringsAsFactors = FALSE)
span   <- read.csv(file.path(out_dir, "span_clinical_for_r.csv"),
                   stringsAsFactors = FALSE)

# ── Group order & labels ──
group_order <- c("LUN","CRC","BLC","PRO","OVA","CPAN","YPAN","BRE",
                 "NOR","YNOR","DIA","HBP","H.D.")
group_labels <- c(
  LUN="Lung", CRC="Colorectal", BLC="Bladder", PRO="Prostate",
  OVA="Ovarian", CPAN="Pancreatic (CBNU)", YPAN="Pancreatic (Severance)",
  BRE="Breast", NOR="Normal", YNOR="Normal (Severance)",
  DIA="Diabetes", HBP="Hypertension", `H.D.`="HTN + DM"
)
category <- c(
  LUN="Cancer", CRC="Cancer", BLC="Cancer", PRO="Cancer",
  OVA="Cancer", CPAN="Cancer", YPAN="Cancer", BRE="Cancer",
  NOR="Non-cancer", YNOR="Non-cancer", DIA="Non-cancer",
  HBP="Non-cancer", `H.D.`="Non-cancer"
)

# ── Build summary ──
build_summary <- function(df, groups) {
  rows <- list()
  for (g in groups) {
    sub <- df[df$disease_group == g, ]
    n <- nrow(sub)
    if (n == 0) next

    age_m  <- mean(sub$age, na.rm=TRUE)
    age_sd <- sd(sub$age, na.rm=TRUE)
    n_m    <- sum(sub$sex == "M", na.rm=TRUE)
    n_f    <- sum(sub$sex == "F", na.rm=TRUE)
    bmi_m  <- mean(sub$bmi, na.rm=TRUE)
    bmi_sd <- sd(sub$bmi, na.rm=TRUE)
    bmi_n  <- sum(!is.na(sub$bmi))

    rows[[length(rows)+1]] <- data.frame(
      Category = category[g],
      Group    = group_labels[g],
      N        = n,
      Age      = sprintf("%.1f \\pm %.1f", age_m, age_sd),
      Male     = sprintf("%d (%.0f\\%%)", n_m, 100*n_m/n),
      Female   = sprintf("%d (%.0f\\%%)", n_f, 100*n_f/n),
      BMI      = ifelse(bmi_n > 2,
                        sprintf("%.1f \\pm %.1f", bmi_m, bmi_sd),
                        "N/A"),
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

summary_df <- build_summary(cohort, group_order)

# Total row
total_n   <- nrow(cohort)
total_age <- sprintf("%.1f \\pm %.1f", mean(cohort$age, na.rm=TRUE), sd(cohort$age, na.rm=TRUE))
total_m   <- sum(cohort$sex == "M", na.rm=TRUE)
total_f   <- sum(cohort$sex == "F", na.rm=TRUE)
total_bmi <- sprintf("%.1f \\pm %.1f", mean(cohort$bmi, na.rm=TRUE), sd(cohort$bmi, na.rm=TRUE))

total_row <- data.frame(
  Category = "",
  Group    = "\\textbf{Total}",
  N        = total_n,
  Age      = sprintf("\\textbf{%s}", total_age),
  Male     = sprintf("\\textbf{%d (%.0f\\%%)}", total_m, 100*total_m/total_n),
  Female   = sprintf("\\textbf{%d (%.0f\\%%)}", total_f, 100*total_f/total_n),
  BMI      = sprintf("\\textbf{%s}", total_bmi),
  stringsAsFactors = FALSE
)
summary_df <- rbind(summary_df, total_row)

# ── Table 1: Full cohort ──
latex1 <- paste0(
"\\begin{table}[htbp]
\\centering
\\caption{Demographic characteristics of the study cohort (N = 1,628).}
\\label{tab:demography}
\\small
\\begin{tabular}{ll r c c c c}
\\toprule
\\textbf{Category} & \\textbf{Group} & \\textbf{N} & \\textbf{Age (mean $\\pm$ SD)} & \\textbf{Male} & \\textbf{Female} & \\textbf{BMI (mean $\\pm$ SD)} \\\\
\\midrule
")

prev_cat <- ""
for (i in seq_len(nrow(summary_df))) {
  row <- summary_df[i, ]
  cat_str <- ifelse(row$Category != prev_cat & row$Category != "", row$Category, "")
  if (row$Category != prev_cat & prev_cat != "" & row$Category != "") {
    latex1 <- paste0(latex1, "\\midrule\n")
  }
  prev_cat <- row$Category
  if (row$Group == "\\textbf{Total}") {
    latex1 <- paste0(latex1, "\\midrule\n")
  }
  latex1 <- paste0(latex1, sprintf("%s & %s & %d & $%s$ & %s & %s & $%s$ \\\\\n",
                                   cat_str, row$Group, row$N, row$Age,
                                   row$Male, row$Female, row$BMI))
}
latex1 <- paste0(latex1,
"\\bottomrule
\\end{tabular}
\\end{table}
")

writeLines(latex1, file.path(out_dir, "table1_cohort_demography.tex"))
cat("Saved: table1_cohort_demography.tex\n")

# ── Table 2: Pancreatic subgroup comparison ──
pan_groups <- c("CPAN", "YPAN")
pan_summary <- build_summary(cohort, pan_groups)

# Add SPAN
span_n   <- nrow(span)
span_age <- sprintf("%.1f \\pm %.1f", mean(span$age, na.rm=TRUE), sd(span$age, na.rm=TRUE))
span_m   <- sum(span$sex == "M", na.rm=TRUE)
span_f   <- sum(span$sex == "F", na.rm=TRUE)
span_bmi <- sprintf("%.1f \\pm %.1f", mean(span$bmi, na.rm=TRUE), sd(span$bmi, na.rm=TRUE))

span_row <- data.frame(
  Category = "Post-operative",
  Group    = "Pancreatic (Samsung)",
  N        = span_n,
  Age      = span_age,
  Male     = sprintf("%d (%.0f\\%%)", span_m, 100*span_m/span_n),
  Female   = sprintf("%d (%.0f\\%%)", span_f, 100*span_f/span_n),
  BMI      = span_bmi,
  stringsAsFactors = FALSE
)
pan_summary$Category <- "Pre-operative"
pan_all <- rbind(pan_summary, span_row)

latex2 <- paste0(
"\\begin{table}[htbp]
\\centering
\\caption{Demographic comparison of pancreatic cancer subgroups.}
\\label{tab:pancreatic_demography}
\\small
\\begin{tabular}{ll r c c c c}
\\toprule
\\textbf{Timing} & \\textbf{Subgroup} & \\textbf{N} & \\textbf{Age (mean $\\pm$ SD)} & \\textbf{Male} & \\textbf{Female} & \\textbf{BMI (mean $\\pm$ SD)} \\\\
\\midrule
")

prev_cat <- ""
for (i in seq_len(nrow(pan_all))) {
  row <- pan_all[i, ]
  cat_str <- ifelse(row$Category != prev_cat, row$Category, "")
  if (row$Category != prev_cat & prev_cat != "") {
    latex2 <- paste0(latex2, "\\midrule\n")
  }
  prev_cat <- row$Category
  latex2 <- paste0(latex2, sprintf("%s & %s & %d & $%s$ & %s & %s & $%s$ \\\\\n",
                                   cat_str, row$Group, row$N, row$Age,
                                   row$Male, row$Female, row$BMI))
}
latex2 <- paste0(latex2,
"\\bottomrule
\\end{tabular}
\\end{table}
")

writeLines(latex2, file.path(out_dir, "table2_pancreatic_demography.tex"))
cat("Saved: table2_pancreatic_demography.tex\n")

cat("\nDone. Check results/poster_figures/table*.tex\n")
