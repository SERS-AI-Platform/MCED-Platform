#!/usr/bin/env Rscript
# STK-V2 — High-legibility Korean figures (R + ggplot2)
# Outputs PNG (300 dpi, white background) ready for embedding in dashboard.

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr)
  library(scales); library(showtext); library(sysfonts)
  library(viridis); library(patchwork); library(reshape2)
})

# ---- Korean font ----
font_add_google_or_local <- function() {
  if (file.exists("/home/user/.fonts/NotoSansKR.ttf")) {
    sysfonts::font_add("NotoKR", regular = "/home/user/.fonts/NotoSansKR.ttf")
    return("NotoKR")
  }
  paths <- system("fc-match -f '%{file}\\n' 'Noto Sans CJK KR'", intern = TRUE)
  if (length(paths) && file.exists(paths[1])) {
    sysfonts::font_add("NotoKR", regular = paths[1]); return("NotoKR")
  }
  "sans"
}
KFONT <- font_add_google_or_local()
showtext_auto(); showtext_opts(dpi = 300)

ROOT <- "/home/user/SERS-AI"
DAT  <- file.path(ROOT, "results/training/stacking_optimization_v2/r_data")
SHAP_CSV <- file.path(ROOT, "results/training/stacking_optimization_v2/meta_shap_values.csv")
OUT  <- file.path(ROOT, "results/figures/stk_v2_r")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

theme_kr <- function(base_size = 12) {
  theme_minimal(base_size = base_size, base_family = KFONT) +
    theme(
      plot.title       = element_text(face = "bold", size = base_size + 4, color = "#1e3a8a"),
      plot.subtitle    = element_text(color = "#475569", size = base_size, margin = margin(b = 8)),
      plot.caption     = element_text(color = "#64748b", size = base_size - 2, hjust = 0, margin = margin(t = 8), lineheight = 1.2),
      plot.caption.position = "plot",
      plot.margin      = margin(14, 18, 12, 14),
      axis.title       = element_text(face = "bold", size = base_size),
      axis.text        = element_text(color = "#334155", size = base_size - 1),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "#e2e8f0", linewidth = 0.4),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background  = element_rect(fill = "white", color = NA),
      legend.position  = "right",
      legend.title     = element_text(face = "bold", size = base_size - 1),
      legend.text      = element_text(size = base_size - 1),
      strip.text       = element_text(face = "bold", size = base_size, color = "#1e293b")
    )
}

save_png <- function(p, name, w = 9, h = 6) {
  ggsave(file.path(OUT, name), p, width = w, height = h, dpi = 300, bg = "white")
  cat("  wrote", name, "\n")
}

# ============================================================================
# Fig 1 — ElasticNet Meta SHAP summary (mean |SHAP| bar)
# ============================================================================
shap_df <- read_csv(SHAP_CSV, show_col_types = FALSE)

shap_summary <- shap_df %>%
  group_by(base_model) %>%
  summarise(mean_abs_shap = mean(abs(shap_value)),
            mean_shap = mean(shap_value), .groups = "drop") %>%
  arrange(desc(mean_abs_shap)) %>%
  mutate(base_model = factor(base_model, levels = base_model))

p1 <- ggplot(shap_summary, aes(x = base_model, y = mean_abs_shap,
                               fill = mean_abs_shap)) +
  geom_col(width = 0.7, color = "white") +
  geom_text(aes(label = sprintf("%.3f", mean_abs_shap)),
            vjust = -0.4, size = 3.6, family = KFONT, color = "#1e293b") +
  scale_fill_viridis_c(option = "mako", direction = -1, end = 0.85, guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(title = "ElasticNet Meta SHAP — 베이스 모델별 평균 기여도",
       subtitle = "Stage 1 (Cancer Screening) · production stacking meta · n=1,628",
       x = "Base model (10종)", y = "mean |SHAP value|",
       caption = paste0("해석: 값이 클수록 ElasticNet meta가 해당 base 예측에 더 의존함.\n",
                        "lr_d2 · lr_d1이 압도적으로 중요. RF 두 base는 meta weight 0 → SHAP도 0 (사실상 8-base 앙상블).")) +
  theme_kr(13)
save_png(p1, "fig1_meta_shap_summary.png", w = 10, h = 6)

# ============================================================================
# Fig 2 — Meta SHAP beeswarm (sample-level distribution)
# ============================================================================
shap_active <- shap_df %>%
  filter(base_model %in% (shap_summary %>% filter(mean_abs_shap > 0) %>% pull(base_model))) %>%
  mutate(base_model = factor(base_model, levels = rev(levels(shap_summary$base_model))))

p2 <- ggplot(shap_active, aes(x = shap_value, y = base_model, color = feature_value)) +
  geom_vline(xintercept = 0, color = "#94a3b8", linewidth = 0.4) +
  geom_jitter(height = 0.28, alpha = 0.5, size = 0.7) +
  scale_color_viridis_c(option = "plasma", end = 0.9,
                        name = "Base 예측값\n(0=정상, 1=암)") +
  scale_x_continuous(breaks = pretty_breaks(7)) +
  labs(title = "ElasticNet Meta SHAP — 샘플별 분포 (beeswarm)",
       subtitle = "각 점 = 1명 환자 · x축 = 해당 base가 meta 결정에 기여한 SHAP",
       x = "SHAP value (양수 = '암' 방향, 음수 = '정상' 방향)",
       y = NULL,
       caption = "해석: 베이스가 양성(노란) 예측을 낼 때 SHAP도 양수면 정합.\nlr_d2의 SHAP 폭이 가장 넓음 → meta가 가장 강하게 의존.") +
  theme_kr(13) + theme(legend.position = "right")
save_png(p2, "fig2_meta_shap_beeswarm.png", w = 11, h = 6.5)

# ============================================================================
# Fig 3 — Meta-learner per-fold comparison (boxplot)
# ============================================================================
nested <- read_csv(file.path(DAT, "nested_cv_results.csv"), show_col_types = FALSE)
ml_levels <- nested %>% group_by(meta_learner) %>%
  summarise(m = mean(s1_auc * 0.6 + s2_f1 * 0.4)) %>%
  arrange(desc(m)) %>% pull(meta_learner)
nested$meta_learner <- factor(nested$meta_learner, levels = ml_levels)

long <- nested %>%
  pivot_longer(c(s1_auc, s2_f1), names_to = "metric", values_to = "value") %>%
  mutate(metric = recode(metric,
                         s1_auc = "Stage 1 AUROC (Screening)",
                         s2_f1  = "Stage 2 F1 macro (Type ID)"))

p3 <- ggplot(long, aes(x = meta_learner, y = value, fill = meta_learner)) +
  geom_boxplot(width = 0.55, outlier.shape = NA, alpha = 0.85, color = "#1e293b") +
  geom_jitter(width = 0.12, size = 1.6, alpha = 0.85, color = "#0f172a") +
  facet_wrap(~ metric, scales = "free_y") +
  scale_fill_viridis_d(option = "mako", end = 0.85, guide = "none") +
  scale_y_continuous(labels = number_format(accuracy = 0.001)) +
  labs(title = "Meta-learner 5종 비교 — Outer 5-fold CV",
       subtitle = "ElasticNet이 combined score(0.6·AUC + 0.4·F1)에서 1위",
       x = "Meta-learner", y = "Score",
       caption = "각 점 = 1 outer fold · 박스 = 사분위 · ElasticNet은 sparsity로 RF 두 base 자동 제거 (해석성↑)") +
  theme_kr(13)
save_png(p3, "fig3_meta_learner_comparison.png", w = 11, h = 5.8)

# ============================================================================
# Fig 4 — Base model permutation contribution (horizontal bar)
# ============================================================================
mwc <- read_csv(file.path(DAT, "meta_weights_contributions.csv"), show_col_types = FALSE) %>%
  arrange(desc(contribution)) %>%
  mutate(base_model = factor(base_model, levels = rev(base_model)))

p4 <- ggplot(mwc, aes(x = contribution, y = base_model, fill = contribution)) +
  geom_col(width = 0.7, color = "white") +
  geom_text(aes(label = sprintf("%.4f", contribution)),
            hjust = -0.15, size = 3.5, family = KFONT, color = "#1e293b") +
  scale_fill_viridis_c(option = "rocket", direction = -1, end = 0.85, guide = "none") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(title = "Base model 기여도 — Permutation Importance",
       subtitle = "각 base의 OOF 예측을 셔플하고 ensemble AUC가 얼마나 떨어지는지 측정 (n_repeats=10)",
       x = "Mean AUC drop (높을수록 중요)", y = NULL,
       caption = "lr_concat이 1위 — view 다양성이 단일 view보다 강력함을 시사. RF는 사실상 무기여.") +
  theme_kr(13)
save_png(p4, "fig4_base_contribution.png", w = 10, h = 6)

# ============================================================================
# Fig 5 — Stacking aggregate Top-15 wavenumber bar
# ============================================================================
agg <- read_csv(file.path(DAT, "stacking_aggregate_top_peaks.csv"), show_col_types = FALSE) %>%
  arrange(desc(score_pct)) %>% slice_head(n = 15) %>%
  mutate(label = sprintf("%.1f", wavenumber),
         label = factor(label, levels = rev(label)),
         region = case_when(
           wavenumber > 1900 ~ "silent (1900~2200)",
           wavenumber >= 1700 & wavenumber <= 1800 ~ "C=O / urea (1700~1800)",
           wavenumber >= 1000 & wavenumber <= 1100 ~ "phenylalanine (1000~1100)",
           TRUE ~ "기타"
         ))

p5 <- ggplot(agg, aes(x = score_pct, y = label, fill = region)) +
  geom_col(width = 0.7, color = "white") +
  geom_text(aes(label = sprintf("%.3f%%", score_pct)),
            hjust = -0.15, size = 3.4, family = KFONT, color = "#1e293b") +
  scale_fill_manual(values = c("silent (1900~2200)" = "#94a3b8",
                                "C=O / urea (1700~1800)" = "#0d9488",
                                "phenylalanine (1000~1100)" = "#dc2626",
                                "기타" = "#a78bfa"),
                    name = "파장 영역") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.20))) +
  labs(title = "Stacking aggregate — Top 15 wavenumber",
       subtitle = "Σ |meta_weight| × contribution × normalized importance — concat은 raw/d1/d2 평균",
       x = "Score (%)", y = "Wavenumber (cm-1)",
       caption = paste0("해석: silent region(회색)이 다수 → derivative view가 grid edge baseline에 민감.\n",
                        "1021 cm-1(빨강) = phenylalanine, xgb_d1 단독 1위 피크.\nC=O 영역(녹색)은 대사체 carbonyl 시그널.")) +
  theme_kr(13) + theme(legend.position = "top")
save_png(p5, "fig5_stacking_aggregate_top15.png", w = 10, h = 7.5)

# ============================================================================
# Fig 6 — Per-base Top-10 features (faceted dotplot)
# ============================================================================
pb <- read_csv(file.path(DAT, "per_base_top_features.csv"), show_col_types = FALSE)

# normalize importance per base for visual comparability
pb <- pb %>% group_by(base_model) %>%
  mutate(imp_norm = importance / max(importance)) %>% ungroup() %>%
  mutate(base_model = factor(base_model,
           levels = c("lr_raw","lr_d1","lr_d2","lr_concat","ridge_concat",
                      "lr_peak","xgb_raw","xgb_d1","rf_raw","rf_d1")),
         feature_label = factor(feature_label, levels = rev(unique(feature_label))))

p6 <- ggplot(pb, aes(x = imp_norm, y = reorder(feature_label, imp_norm),
                     color = base_model)) +
  geom_segment(aes(x = 0, xend = imp_norm, yend = feature_label),
               linewidth = 0.6, alpha = 0.6) +
  geom_point(size = 2.6) +
  facet_wrap(~ base_model, scales = "free_y", ncol = 2) +
  scale_color_viridis_d(option = "turbo", end = 0.92, guide = "none") +
  scale_x_continuous(limits = c(0, 1.05), breaks = c(0, 0.5, 1)) +
  labs(title = "Base model별 Top-10 feature (정규화 importance)",
       subtitle = "LR/Ridge: |coef_| (StandardScaler 후) · XGB/RF: feature_importances_",
       x = "Importance (base 내 max로 정규화)", y = NULL,
       caption = "각 base의 Top-10이 서로 다름 → view 다양성이 stacking 이득의 본질.\n894 cm-1(uric_acid)가 lr_d1·rf_d1에서 공통, 1598~1604(purine ring)가 xgb_raw·rf_raw에서 공통.") +
  theme_kr(11) + theme(axis.text.y = element_text(size = 8),
                       strip.text = element_text(size = 10))
save_png(p6, "fig6_per_base_top10.png", w = 12, h = 11)

# ============================================================================
# Fig 7 — Single vs ensemble scatter
# ============================================================================
sve <- read_csv(file.path(DAT, "single_vs_ensemble.csv"), show_col_types = FALSE)

p7 <- ggplot(sve, aes(x = auc, y = f1_type, color = type, shape = type)) +
  geom_point(size = 5, stroke = 1.2) +
  ggrepel::geom_text_repel(aes(label = model), family = KFONT, size = 3.5,
                           max.overlaps = 20, box.padding = 0.4, color = "#1e293b") |> tryCatch(error = function(e) {
    geom_text(aes(label = model), family = KFONT, size = 3.4, vjust = -1.4, color = "#1e293b")
  }) -> repel_layer
p7 <- ggplot(sve, aes(x = auc, y = f1_type, color = type, shape = type)) +
  geom_point(size = 5, stroke = 1.2) +
  geom_text(aes(label = model), family = KFONT, size = 3.4, vjust = -1.3, color = "#1e293b") +
  scale_color_manual(values = c(single = "#475569", ensemble = "#dc2626"), name = "Type") +
  scale_shape_manual(values = c(single = 16, ensemble = 17), name = "Type") +
  scale_x_continuous(labels = number_format(accuracy = 0.001)) +
  scale_y_continuous(labels = number_format(accuracy = 0.001)) +
  labs(title = "Single base model vs Ensemble — Stage 1 AUC × Stage 2 F1",
       subtitle = "10개 base model 단독 성능과 simple-average ensemble 비교 (outer-test OOF)",
       x = "Stage 1 AUROC", y = "Stage 2 F1 macro",
       caption = "ridge_concat / lr_concat single이 0.99+ 도달 — 단일 모델만으로도 매우 강함.\nensemble_avg는 simple mean이라 stacking(ElasticNet meta)보다 낮음.") +
  theme_kr(13)
save_png(p7, "fig7_single_vs_ensemble.png", w = 10, h = 7)

cat("\n[done] all R figures saved to:", OUT, "\n")
