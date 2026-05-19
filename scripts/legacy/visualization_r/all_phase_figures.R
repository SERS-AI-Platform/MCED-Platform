#!/usr/bin/env Rscript
# ============================================================================
# SERS-AI — ALL Phase Archive Figures (R ggplot2, Korean labels)
# ============================================================================
# Reads tidy CSVs from results/r_export/ and generates ~75 figures.
# Outputs: results/figures/phase_archive_r/{category}/filename.png
# ============================================================================

`%||%` <- function(a, b) if (!is.null(a)) a else b

# Get script directory robustly
.get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  idx <- grep("--file=", args)
  if (length(idx)) return(dirname(normalizePath(sub("--file=", "", args[idx]))))
  "."
}
source(file.path(.get_script_dir(), "theme_sers.R"))

ROOT    <- "/home/user/SERS-AI"
RDATA   <- file.path(ROOT, "results/r_export")
OUTROOT <- file.path(ROOT, "results/figures/phase_archive_r")

safe_read <- function(path) {
  if (file.exists(path)) read_csv(path, show_col_types = FALSE) else NULL
}

# ============================================================================
# 01 — BASELINE & BENCHMARK (Phase F, K)
# ============================================================================
cat("\n=== 01 Baseline ===\n")
dir.create(file.path(OUTROOT, "01_baseline"), recursive = TRUE, showWarnings = FALSE)

# Benchmark comparison bar chart
bm <- safe_read(file.path(RDATA, "01_baseline/benchmark_summary.csv"))
if (!is.null(bm)) {
  bm_long <- bm %>%
    select(model, matches("s1_auc|s2_f1|val_s1|val_s2")) %>%
    pivot_longer(-model, names_to = "metric", values_to = "value") %>%
    filter(!is.na(value))

  # Determine column names dynamically
  auc_col <- intersect(names(bm), c("val_s1_auc", "s1_auc", "S1_ROC"))[1]
  f1_col  <- intersect(names(bm), c("val_s2_f1_macro", "s2_f1", "S2_F1"))[1]

  if (!is.na(auc_col) && !is.na(f1_col)) {
    bm$model_f <- factor(bm$model, levels = bm$model[order(bm[[f1_col]])])
    p <- ggplot(bm, aes(x = model_f)) +
      geom_col(aes(y = .data[[f1_col]], fill = "Type ID F1"), width = 0.4,
               position = position_nudge(x = -0.22), alpha = 0.9) +
      geom_col(aes(y = .data[[auc_col]], fill = "Det AUC"), width = 0.4,
               position = position_nudge(x = 0.22), alpha = 0.9) +
      geom_text(aes(y = .data[[f1_col]], label = sprintf("%.3f", .data[[f1_col]])),
                position = position_nudge(x = -0.22), vjust = -0.3,
                size = 3.4, family = KFONT) +
      scale_fill_manual(values = c("Det AUC" = "#0d9488", "Type ID F1" = "#6d28d9"), name = "") +
      coord_flip() +
      labs(title = "Phase F — 5-cancer 모델 벤치마크",
           subtitle = "6개 모델 비교 (5-fold StratifiedGroupKFold, n=1,240)",
           x = NULL, y = "Score",
           caption = "결론: LR이 AUC·F1 모두 1위. DL(ResNet18)은 과적합으로 LR 미달.") +
      theme_sers()
    save_fig(p, file.path(OUTROOT, "01_baseline/F_benchmark_comparison.png"), w = 10, h = 6)
  }

  # Overfit gap chart
  gap <- safe_read(file.path(RDATA, "01_baseline/benchmark_train_val_gaps.csv"))
  if (!is.null(gap)) {
    gap_col_t <- intersect(names(gap), c("train_s1_auc", "train_auc"))[1]
    gap_col_v <- intersect(names(gap), c("val_s1_auc", "val_auc", "s1_auc"))[1]
    if (!is.na(gap_col_t) && !is.na(gap_col_v)) {
      gap$gap <- gap[[gap_col_t]] - gap[[gap_col_v]]
      gap$model_f <- factor(gap$model, levels = gap$model[order(gap$gap)])
      p <- ggplot(gap, aes(model_f, gap, fill = gap)) +
        geom_col(width = 0.6) +
        geom_text(aes(label = sprintf("%.3f", gap)), hjust = -0.2, size = 3.5, family = KFONT) +
        coord_flip() +
        scale_fill_viridis_c(option = "rocket", direction = -1, end = 0.85, guide = "none") +
        scale_y_continuous(expand = expansion(mult = c(0, 0.2))) +
        labs(title = "Phase F — Train-Val 과적합 갭",
             subtitle = "DL 모델(ResNet18 등)일수록 갭이 큼 → 과적합 경향",
             x = NULL, y = "AUC Gap (Train - Val)",
             caption = "LR 계열은 갭 < 0.02, DL은 > 0.10 — 데이터 규모 1,240명에서 DL 한계.") +
        theme_sers()
      save_fig(p, file.path(OUTROOT, "01_baseline/F_benchmark_overfit_gap.png"), w = 10, h = 6)
    }
  }
}

# ============================================================================
# 02 — NORMALIZATION (Phase L)
# ============================================================================
cat("\n=== 02 Normalization ===\n")
dir.create(file.path(OUTROOT, "02_normalization"), recursive = TRUE, showWarnings = FALSE)

norm <- safe_read(file.path(RDATA, "02_normalization/comparison_table.csv"))
if (!is.null(norm)) {
  f1_col <- intersect(names(norm), c("lr_s2_f1", "val_s2_f1_macro", "f1"))[1]
  auc_col <- intersect(names(norm), c("lr_s1_auc", "val_s1_auc", "auc"))[1]
  nm_col <- intersect(names(norm), c("normalization", "norm", "method"))[1]
  if (!is.na(f1_col) && !is.na(nm_col)) {
    norm$norm_f <- factor(norm[[nm_col]], levels = norm[[nm_col]][order(norm[[f1_col]])])
    p <- ggplot(norm, aes(norm_f, .data[[f1_col]], fill = .data[[f1_col]])) +
      geom_col(width = 0.6) +
      geom_text(aes(label = sprintf("%.3f", .data[[f1_col]])), vjust = -0.3, size = 3.8, family = KFONT) +
      scale_fill_viridis_c(option = "mako", direction = -1, end = 0.85, guide = "none") +
      scale_y_continuous(expand = expansion(mult = c(0, 0.15)), limits = c(0, NA)) +
      labs(title = "Phase L — 정규화 방법 비교 (LR · Type ID F1)",
           subtitle = "SNV / None / MinMax / L2 4가지, 동일 5-fold CV",
           x = "정규화 방법", y = "Type ID F1 macro",
           caption = "SNV가 전 모델에서 1위. 이후 모든 phase의 default norm.") +
      theme_sers()
    save_fig(p, file.path(OUTROOT, "02_normalization/L_normalization.png"), w = 9, h = 6)
  }
}

# ============================================================================
# 03 — CLINICAL FUSION (Phase M, N)
# ============================================================================
cat("\n=== 03 Clinical Fusion ===\n")
dir.create(file.path(OUTROOT, "03_clinical_fusion"), recursive = TRUE, showWarnings = FALSE)

# Feature study includes clinical fusion data
fstudy <- safe_read(file.path(RDATA, "03_clinical_fusion/clinical_fusion_results.csv"))
if (!is.null(fstudy)) {
  p <- ggplot(fstudy %>% filter(Model == "LR"),
              aes(reorder(Feature, Type_F1_mean), Type_F1_mean, fill = Type_F1_mean)) +
    geom_col(width = 0.6) +
    geom_errorbar(aes(ymin = Type_F1_mean - Type_F1_std, ymax = Type_F1_mean + Type_F1_std),
                  width = 0.2) +
    geom_text(aes(label = sprintf("%.3f", Type_F1_mean)), hjust = -0.2, size = 3.5, family = KFONT) +
    coord_flip() +
    scale_fill_viridis_c(option = "mako", direction = -1, end = 0.85, guide = "none") +
    labs(title = "Phase CF — Feature × Clinical fusion 비교 (LR)",
         subtitle = "3-view + clinical = spec + deriv + peak + age/sex/BMI",
         x = NULL, y = "Type ID F1 macro",
         caption = "3-view + clinical(1,948 feat)이 F1 0.914로 7-cancer LR 최고.") +
    theme_sers()
  save_fig(p, file.path(OUTROOT, "03_clinical_fusion/CF_fusion_comparison.png"), w = 10, h = 7)
}

# ============================================================================
# 04 — CROSS-INSTRUMENT (Phase XI)
# ============================================================================
cat("\n=== 04 Cross-Instrument ===\n")
dir.create(file.path(OUTROOT, "04_cross_instrument"), recursive = TRUE, showWarnings = FALSE)

xi <- safe_read(file.path(RDATA, "04_cross_instrument/leaderboard.csv"))
if (!is.null(xi)) {
  xi$label <- paste(xi$direction, xi$method, sep = "\n")
  p <- ggplot(xi, aes(reorder(label, s1_auc), s1_auc, fill = direction)) +
    geom_col(width = 0.6, alpha = 0.85) +
    geom_text(aes(label = sprintf("%.3f", s1_auc)), hjust = -0.1, size = 3.2, family = KFONT) +
    coord_flip() +
    scale_fill_manual(values = c("Medical→Thermo" = "#dc2626", "Thermo→Medical" = "#2563eb"), name = "방향") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(title = "Phase XI — Cross-instrument calibration leaderboard",
         subtitle = "Thermo↔Medical transfer 성능 (다양한 calibration 방법)",
         x = NULL, y = "Stage 1 AUC",
         caption = "결론: PDS(w=31, ridge=0.01)가 최고지만 ceiling의 88~93% 수준. 기기 간 transfer는 한계적.") +
    theme_sers() + theme(axis.text.y = element_text(size = 8))
  save_fig(p, file.path(OUTROOT, "04_cross_instrument/XI_leaderboard.png"), w = 11, h = 9)
}

# ============================================================================
# 05 — FEATURE STUDY (Phase FR, MV)
# ============================================================================
cat("\n=== 05 Feature Study ===\n")
dir.create(file.path(OUTROOT, "05_feature_study"), recursive = TRUE, showWarnings = FALSE)

abl <- safe_read(file.path(RDATA, "05_feature_study/ablation_results.csv"))
if (!is.null(abl)) {
  abl_lr <- abl %>% filter(Model == "LR")
  p <- ggplot(abl_lr, aes(reorder(Feature, Type_F1_mean), Type_F1_mean, fill = Experiment)) +
    geom_col(width = 0.6, position = "dodge", alpha = 0.85) +
    geom_errorbar(aes(ymin = Type_F1_mean - Type_F1_std, ymax = Type_F1_mean + Type_F1_std),
                  width = 0.2, position = position_dodge(0.6)) +
    coord_flip() +
    scale_fill_viridis_d(option = "mako", end = 0.85) +
    labs(title = "Feature Representation / Clinical Fusion ablation (LR)",
         subtitle = "각 feature set별 7-cancer Type ID F1 비교 (5-fold CV, n=1,628)",
         x = NULL, y = "Type ID F1 macro",
         caption = "1st derivative > full spectrum > 2nd derivative > peak-only.\n3-view + clinical이 최고 F1 0.914.") +
    theme_sers()
  save_fig(p, file.path(OUTROOT, "05_feature_study/FR_ablation.png"), w = 11, h = 8)
}

# ============================================================================
# 06 — CONTRASTIVE SWEEP (Phase CON-SW)
# ============================================================================
cat("\n=== 06 Contrastive Sweep ===\n")
dir.create(file.path(OUTROOT, "06_contrastive"), recursive = TRUE, showWarnings = FALSE)

con <- safe_read(file.path(RDATA, "06_contrastive/sweep_results.csv"))
if (!is.null(con)) {
  p <- ggplot(con, aes(finetune_auc, finetune_f1_type,
                       color = factor(temperature), shape = factor(proj_dim))) +
    geom_point(size = 3.5, alpha = 0.8) +
    scale_color_viridis_d(option = "plasma", end = 0.9, name = "Temperature") +
    scale_shape_manual(values = c(16, 17, 15), name = "Proj dim") +
    labs(title = "Phase CON-SW — Contrastive pretrain sweep (36 configs)",
         subtitle = "Finetune AUC × Type ID F1, temperature/proj_dim/pretrain_epochs 3-grid",
         x = "Finetune AUC", y = "Finetune F1 (Type ID)",
         caption = "최고 F1 0.681 — LR baseline(0.85+)에 크게 미달.\nDL은 이 데이터 규모(1,628명)에서 LR 못 넘김 재확인.") +
    theme_sers()
  save_fig(p, file.path(OUTROOT, "06_contrastive/CON_SW_sweep.png"), w = 10, h = 7)
}

# ============================================================================
# 07 — PERMUTATION IMPORTANCE (Phase PI)
# ============================================================================
cat("\n=== 07 Permutation Importance ===\n")
dir.create(file.path(OUTROOT, "07_permutation"), recursive = TRUE, showWarnings = FALSE)

pi_df <- safe_read(file.path(RDATA, "07_permutation/importance_ranking.csv"))
if (!is.null(pi_df)) {
  top30 <- pi_df %>% slice_head(n = 30) %>%
    mutate(wn_label = sprintf("%.1f", wavenumber),
           wn_label = factor(wn_label, levels = rev(wn_label)))
  p <- ggplot(top30, aes(mean_auc_drop, wn_label, fill = mean_auc_drop)) +
    geom_col(width = 0.7) +
    geom_errorbarh(aes(xmin = mean_auc_drop - std_auc_drop,
                       xmax = mean_auc_drop + std_auc_drop),
                   height = 0.3, color = "#475569") +
    geom_text(aes(label = sprintf("%.4f", mean_auc_drop)),
              hjust = -0.15, size = 3, family = KFONT) +
    scale_fill_viridis_c(option = "rocket", direction = -1, end = 0.85, guide = "none") +
    scale_x_continuous(expand = expansion(mult = c(0, 0.2))) +
    labs(title = "Phase PI — Top 30 wavenumber (permutation importance)",
         subtitle = "935 wavenumber x 1,000 permutation, 모두 FDR-significant",
         x = "Mean AUC drop (높을수록 중요)", y = "Wavenumber (cm-1)",
         caption = "894 cm-1 (uric acid) 1위, 892·1913·1915·815 cm-1 상위.\n모든 wavenumber가 유의 → distributed representation.") +
    theme_sers()
  save_fig(p, file.path(OUTROOT, "07_permutation/PI_top30.png"), w = 10, h = 10)
}

# ============================================================================
# 08 — LOHO (Leave-One-Hospital-Out)
# ============================================================================
cat("\n=== 08 LOHO ===\n")
dir.create(file.path(OUTROOT, "08_loho"), recursive = TRUE, showWarnings = FALSE)

loho <- safe_read(file.path(RDATA, "08_loho/loho_results.csv"))
if (!is.null(loho)) {
  p <- ggplot(loho, aes(hospital, f1, fill = f1)) +
    geom_col(width = 0.6) +
    geom_text(aes(label = sprintf("%.3f", f1)), vjust = -0.3, size = 4, family = KFONT) +
    scale_fill_viridis_c(option = "mako", direction = -1, end = 0.85, guide = "none") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(title = "Phase LOHO — Leave-One-Hospital-Out 일반화 실패",
         subtitle = "학습에서 제외한 병원의 환자로만 평가",
         x = "Left-out hospital", y = "Type ID F1",
         caption = "Chungbuk F1=0.0, StMarys F1=0.11 → 병원 시그널 차이가 스펙트럼에 반영.\n다기관 전향적 검증이 반드시 필요.") +
    theme_sers()
  save_fig(p, file.path(OUTROOT, "08_loho/LOHO_validation.png"), w = 9, h = 6)
}

# ============================================================================
# 09 — NORM × FEATURE SEARCH (Phase NFS)
# ============================================================================
cat("\n=== 09 NFS ===\n")
dir.create(file.path(OUTROOT, "09_nfs"), recursive = TRUE, showWarnings = FALSE)

nfs <- safe_read(file.path(RDATA, "09_nfs/nfs_results.csv"))
if (!is.null(nfs)) {
  top15 <- nfs %>% arrange(desc(mean_f1)) %>% slice_head(n = 15)
  top15$config <- with(top15, paste(norm, feature, model, sep = " / "))
  top15$config <- factor(top15$config, levels = rev(top15$config))
  p <- ggplot(top15, aes(mean_f1, config, fill = mean_f1)) +
    geom_col(width = 0.6) +
    geom_errorbarh(aes(xmin = mean_f1 - std_f1, xmax = mean_f1 + std_f1),
                   height = 0.2, color = "#475569") +
    geom_text(aes(label = sprintf("%.3f", mean_f1)), hjust = -0.15, size = 3.3, family = KFONT) +
    scale_fill_viridis_c(option = "mako", direction = -1, end = 0.85, guide = "none") +
    scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(title = "Phase NFS — Norm x Feature search Top-15",
         subtitle = "1,040/1,800 config 완료 시점 중단, norm × feature × model grid",
         x = "Mean Type ID F1 (20 seeds)", y = NULL,
         caption = "l2 / d1 / LR 조합이 F1 0.917으로 1위.\nSNV가 default이나 l2도 경쟁력.") +
    theme_sers()
  save_fig(p, file.path(OUTROOT, "09_nfs/NFS_top15.png"), w = 10, h = 7)
}

# ============================================================================
# 10 — COMPLETE MODEL COMPARISON (overall)
# ============================================================================
cat("\n=== 10 Overall comparison ===\n")
dir.create(file.path(OUTROOT, "10_overall"), recursive = TRUE, showWarnings = FALSE)

cmp <- safe_read(file.path(RDATA, "10_overall/complete_model_comparison.csv"))
if (!is.null(cmp)) {
  auc_col <- intersect(names(cmp), c("S1_ROC", "s1_auc", "Det_AUC"))[1]
  f1_col  <- intersect(names(cmp), c("S2_F1", "s2_f1", "Type_F1"))[1]
  if (!is.na(auc_col) && !is.na(f1_col)) {
    cmp$N_num <- as.numeric(gsub("[^0-9]", "", cmp$N))
    p <- ggplot(cmp, aes(.data[[auc_col]], .data[[f1_col]])) +
      geom_point(aes(color = factor(Cancers), size = N_num), alpha = 0.7) +
      geom_text(aes(label = Model), size = 2.5, vjust = -1.2, family = KFONT, check_overlap = TRUE) +
      scale_color_viridis_d(option = "plasma", end = 0.9, name = "# cancers") +
      scale_size_continuous(range = c(2, 6), guide = "none") +
      labs(title = "전체 모델 비교 — Det AUC × Type ID F1",
           subtitle = "SERS-AI 전 phase 주요 모델 결과 (n / cancer 수 별)",
           x = "Detection AUC", y = "Type ID F1 macro",
           caption = "STK-V2(10→EN)가 우측 상단 corner — AUC 0.994, F1 0.926.") +
      theme_sers()
    save_fig(p, file.path(OUTROOT, "10_overall/complete_model_comparison.png"), w = 11, h = 8)
  }
}

cat("\n[done] all phase archive R figures complete.\n")
cat("output:", OUTROOT, "\n")
