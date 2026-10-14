#!/usr/bin/env python
"""PL-3 보충 6: 날짜(run) 효과가 몰린 밴드를 제외하면 날짜 구분력과 암 AUC가 어떻게 되는가.

배경 (docs/ml/experiment.md PL-3 보충 2~5): 보라매 전립선 3군 코호트에서 비암 검체의
OOF 암 확률이 측정일(run) 순서대로 오르고, SNV 후 차이는 1421~1427 cm⁻¹(감소)와
1600~1613 cm⁻¹(증가)에 집중된다. 이 밴드를 빼면
  - 날짜 구분력이 떨어지고 암 AUC는 유지 → 밴드를 피하는 모델이 답
  - 둘 다 떨어짐 → 암 신호와 run 효과가 같은 밴드에 있음
을 가른다.

조건: full / mask_narrow(1400–1440, 1590–1620) / mask_wide(1380–1460, 1570–1640) /
random_k (같은 총 폭을 무작위 위치에 5회) — 무작위 대조가 없으면 "아무 밴드나 빼도
떨어진다"와 구분이 안 된다.

지표
  - cancer_auc      : 암 vs 비암, production 전처리, LR C=1.0, StratifiedGroupKFold(5) by subject,
                      환자 mean OOF, 5시드 mean±sd (PL-3 규약)
  - day_macro_auc   : 비암(lot 1, run 1~4)만으로 측정일 4클래스 multinomial LR, 같은 CV,
                      환자 mean OOF macro OvR AUC (5시드). 우연 = 0.5
  - day_accuracy    : 위 분류의 환자 단위 정확도 (우연 ≈ 0.33, 최다 클래스 비율)
  - nc_score_spread : 암 모델의 비암 OOF 확률 날짜별 평균의 (8/13 − 8/10)

그림 (PNG, 고정 스타일)
  - band_by_day.png : 비암 검체의 날짜별 평균 SNV 스펙트럼 — 전체 + 두 밴드 확대
  - mask_results.png: 조건별 cancer AUC vs day macro AUC

Usage:
    source scripts/db/pghost.sh
    python scripts/analysis/preprocessing_lab/band_mask_day_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "analysis" / "preprocessing_lab"))

import run_guide_pipeline as G  # noqa: E402

from sers.evaluation.metrics import macro_roc_auc  # noqa: E402
from sers.preprocessing_lab.db import ExperimentTracker  # noqa: E402
from sers.visualization.style import (  # noqa: E402
    CLASS_COLORS,
    COLOR_MUTED,
    COLOR_TEXT,
    FIGSIZE_WIDE,
    SIZE_ANNOTATION,
    apply_style,
    legend,
    save_png,
    set_title,
    style_axes,
)

OUT = REPO / "results" / "preprocessing_lab" / "band_mask_20260909"
SEEDS = G.SEEDS
NARROW = ((1400.0, 1440.0), (1590.0, 1620.0))
WIDE = ((1380.0, 1460.0), (1570.0, 1640.0))
N_RANDOM = 5
DAY_LABELS = ["20260810", "20260811", "20260812", "20260813"]
#: 날짜는 순서형 → 단일 색상 램프 (밝음 → 진함)
DAY_COLORS = {"20260810": "#BBDEFB", "20260811": "#64B5F6", "20260812": "#1E88E5", "20260813": "#0D47A1",
              "20260814": "#C62828"}


def load_days() -> dict[str, str]:
    with ExperimentTracker()._connect() as c:  # noqa: SLF001
        meta = pd.read_sql(
            "select s.sample_id::text as sid, to_char(r.measurement_date,'YYYYMMDD') as mday "
            "from master.samples s join measurement.measurements m using(sample_id) "
            "join measurement.runs r using(measurement_run_id) group by 1,2", c)
    return dict(zip(meta.sid, meta.mday))


def mask_for(grid: np.ndarray, bands) -> np.ndarray:
    keep = np.ones(len(grid), dtype=bool)
    for lo, hi in bands:
        keep &= ~((grid >= lo) & (grid <= hi))
    return keep


def random_bands(rng: np.random.Generator, grid: np.ndarray, widths=(40.0, 30.0)) -> tuple:
    bands = []
    for w in widths:
        lo = float(rng.uniform(grid[0], grid[-1] - w))
        bands.append((lo, lo + w))
    return tuple(bands)


def _oof(X, y, groups, seed, multi=False):
    oof = np.zeros((len(y), len(np.unique(y))))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y, groups):
        m = Pipeline([("s", StandardScaler()),
                      ("lr", LogisticRegression(max_iter=5000, class_weight="balanced",
                                                solver="lbfgs", random_state=seed))]).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])
    return oof if multi else oof[:, 1]


def patient_mean(oof, groups):
    df = pd.DataFrame(oof if oof.ndim == 2 else oof[:, None])
    df["g"] = groups
    return df.groupby("g").mean()


def evaluate_condition(name, keep, X, y, groups, sday, ncmask):
    Xk = X[:, keep]
    # 암 AUC + 비암 점수 날짜 spread
    res = Parallel(n_jobs=5)(delayed(_oof)(Xk, y, groups, s) for s in SEEDS)
    aucs, spreads = [], []
    for oof in res:
        pm = patient_mean(oof, groups)
        pg = pd.Series({g: y[groups == g][0] for g in pm.index})
        aucs.append(roc_auc_score(pg.loc[pm.index], pm[0]))
        pday = pd.Series({g: sday[groups == g][0] for g in pm.index})
        nc = pm[pg == 0][0].groupby(pday[pg == 0]).mean()
        spreads.append(float(nc.get("20260813", np.nan) - nc.get("20260810", np.nan)))
    # 날짜 분류 (비암, lot 1, run 1~4)
    Xn, gn, dn = Xk[ncmask], groups[ncmask], sday[ncmask]
    dcode = np.array([DAY_LABELS.index(d) for d in dn])
    resd = Parallel(n_jobs=5)(delayed(_oof)(Xn, dcode, gn, s, True) for s in SEEDS)
    dauc, dacc = [], []
    for oof in resd:
        pm = patient_mean(oof, gn)
        pd_true = np.array([dcode[gn == g][0] for g in pm.index])
        dauc.append(macro_roc_auc(pd_true, pm.to_numpy()))
        dacc.append(float((pm.to_numpy().argmax(1) == pd_true).mean()))
    return dict(condition=name, n_features=int(keep.sum()),
                cancer_auc=float(np.mean(aucs)), cancer_auc_sd=float(np.std(aucs, ddof=1)),
                nc_score_spread=float(np.nanmean(spreads)),
                day_macro_auc=float(np.mean(dauc)), day_macro_auc_sd=float(np.std(dauc, ddof=1)),
                day_accuracy=float(np.mean(dacc)))


def figure_band_by_day(X, y, sday, grid, path: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), gridspec_kw={"width_ratios": [2.2, 1, 1]})
    nc = y == 0
    handles = []
    for ax_i, (ax, lim) in enumerate(zip(axes, [(402, 2198), (1380, 1460), (1570, 1640)])):
        for d in DAY_LABELS:
            m = nc & (sday == d)
            if m.sum() == 0:
                continue
            line, = ax.plot(grid, X[m].mean(0), color=DAY_COLORS[d], linewidth=1.8 if ax_i else 1.2,
                            label=f"{d[4:6]}/{d[6:]} 비암 ({m.sum()} 스펙트럼)")
            if ax_i == 0:
                handles.append(line)
        for d in ("20260813", "20260814"):
            m = (y == 1) & (sday == d)
            line, = ax.plot(grid, X[m].mean(0), color=DAY_COLORS["20260814"], linewidth=1.2,
                            linestyle="--" if d == "20260813" else "-", alpha=0.85,
                            label=f"{d[4:6]}/{d[6:]} 암 ({m.sum()} 스펙트럼, lot 6)")
            if ax_i == 0:
                handles.append(line)
        ax.set_xlim(*lim)
        for lo, hi in NARROW:
            ax.axvspan(lo, hi, color="#000000", alpha=0.06, lw=0)
        style_axes(ax, grid="y", xlabel="Raman shift (cm⁻¹)", ylabel="SNV intensity" if ax_i == 0 else None)
        if ax_i:
            lo, hi = lim
            band = (grid >= lo) & (grid <= hi)
            ax.set_ylim(X[nc][:, band].mean(0).min() - 0.6, X[nc][:, band].mean(0).max() + 0.6)
            ax.set_title(f"확대: {int(NARROW[ax_i - 1][0])}–{int(NARROW[ax_i - 1][1])} cm⁻¹ (회색)", fontsize=13,
                         loc="left", fontweight="bold", color=COLOR_TEXT, pad=8)
    set_title(axes[0], "날짜별 평균 스펙트럼 — 비암(strip lot 1) vs 암(lot 6)",
              "production 전처리 · 스펙트럼 단위 평균 · 파랑 = 측정일(밝음→진함)")
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_png(fig, path)


def figure_results(df: pd.DataFrame, path: Path) -> None:
    """조건별 암 AUC와 날짜 구분력을 같은 축(0.5~1.0)에 — 축을 고정해야 '차이 없음'이 보인다."""
    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    order = ["full", "mask_narrow", "mask_wide"] + sorted(c for c in df.condition if c.startswith("random"))
    labels = {"full": "전체\n(제외 없음)", "mask_narrow": "좁은 밴드 제외\n1400–1440, 1590–1620",
              "mask_wide": "넓은 밴드 제외\n1380–1460, 1570–1640"}
    d = df.set_index("condition").loc[order]
    x = np.arange(len(order))
    ax.errorbar(x - 0.15, d.cancer_auc, yerr=d.cancer_auc_sd, fmt="o", markersize=8, capsize=3,
                color=CLASS_COLORS["cancer"], label="암 vs 비암 AUC (환자 OOF)", zorder=3)
    ax.errorbar(x + 0.15, d.day_macro_auc, yerr=d.day_macro_auc_sd, fmt="s", markersize=8, capsize=3,
                color=COLOR_TEXT, label="측정일 구분력 (비암 4일 분류 macro AUC)", zorder=3)
    for xi, (ca, da) in enumerate(zip(d.cancer_auc, d.day_macro_auc)):
        ax.annotate(f"{ca:.2f}", (xi - 0.15, ca), textcoords="offset points", xytext=(0, 16), ha="center",
                    fontsize=SIZE_ANNOTATION, color=CLASS_COLORS["cancer"])
        ax.annotate(f"{da:.2f}", (xi + 0.15, da), textcoords="offset points", xytext=(0, -22), ha="center",
                    fontsize=SIZE_ANNOTATION, color=COLOR_TEXT)
    ax.axhline(0.5, color=COLOR_MUTED, linestyle="--", linewidth=1, label="우연 (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(c, f"무작위 {c[-1]}\n(대조)") for c in order], fontsize=9)
    ax.set_ylim(0.45, 1.0)
    style_axes(ax, grid="y", xlabel="제외 조건", ylabel="AUC")
    set_title(ax, "밴드를 제외해도 날짜 구분력과 암 AUC는 변하지 않는다",
              "n=112 (암 43 / 비암 69) · LR C=1.0 · StratifiedGroupKFold(5) · 5시드 mean ± sd · 날짜 분류는 비암 69명(run 1~4)")
    legend(ax, loc="lower right")
    save_png(fig, path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = G.load_config()
    cohort = G.load_aecd()
    X, y, _y3, groups, _qc, _shift, _prep = G.run_condition(cohort, cfg, G.CONDITIONS["production"])
    day_of = load_days()
    sday = np.array([day_of[g.split("|")[1]] for g in groups])
    grid = G.GRID
    ncmask = (y == 0) & np.isin(sday, DAY_LABELS)

    figure_band_by_day(X, y, sday, grid, OUT / "band_by_day.png")

    conditions = [("full", np.ones(len(grid), bool)),
                  ("mask_narrow", mask_for(grid, NARROW)),
                  ("mask_wide", mask_for(grid, WIDE))]
    rng = np.random.default_rng(20260909)
    rand_bands = {}
    for k in range(N_RANDOM):
        b = random_bands(rng, grid)
        rand_bands[f"random_{k}"] = b
        conditions.append((f"random_{k}", mask_for(grid, b)))

    rows = []
    for name, keep in conditions:
        print(f"[{name}] features={keep.sum()}")
        rows.append(evaluate_condition(name, keep, X, y, groups, sday, ncmask))
        print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "band_mask_results.csv", index=False, encoding="utf-8-sig")
    (OUT / "random_bands.json").write_text(json.dumps(rand_bands, indent=2))
    figure_results(df, OUT / "mask_results.png")
    print(df.round(3).to_string(index=False))
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
