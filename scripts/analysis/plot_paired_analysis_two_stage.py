"""3-panel performance figures for the two-stage evaluation of the 427 overlap samples (36-point basis, AS-23 / AS-23b).

One figure per stage per condition, in the user's standard layout (confusion matrix + ROC + metric bars),
drawn only with ``sers.visualization.performance`` / ``sers.visualization.style`` and saved as PNG:

    stage 1 Cancer Screening : 2x2 confusion matrix (gate 0.5) · ROC · ROC-AUC / balanced acc. / sensitivity / specificity
    stage 2 Cancer Type ID   : 7x7 cascade confusion matrix (non-cancer + 6 types) · one-vs-rest ROC per type
                               (score = stage-1 cancer probability x type probability) · macro OvR AUC / macro recall / macro F1

Conditions: ① pre->pre, ② pre->post (pre-change model on post-change samples), ③ post->post.
Input: sample probabilities averaged over the 25 runs, written by the worktree analysis script
``SERS-AI-ci-tiered/scripts/analysis/paired_analysis_two_stage.py``.

Run from the repo root:
    PYTHONPATH=src python scripts/analysis/plot_paired_analysis_two_stage.py
Outputs -> results/paired_analysis_two_stage_figures/ (PNG only)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402

from sers.visualization.performance import plot_confusion_matrix, plot_roc  # noqa: E402
from sers.visualization.style import (  # noqa: E402
    CLASS_COLORS,
    COLOR_MUTED,
    COLOR_TEXT,
    SIZE_ANNOTATION,
    SIZE_SUBTITLE,
    SIZE_TITLE,
    apply_style,
    group_colors,
    legend,
    save_png,
    set_title,
    style_axes,
    subtitle_from,
)

REPO: Final = Path(__file__).resolve().parents[2]
SRC: Final = Path("/home/user/SERS-AI-ci-tiered/results/paired_analysis_two_stage/subject_probs_mean.csv")
OUT: Final = REPO / "results" / "paired_analysis_two_stage_figures"
TYPES: Final = ("PRO", "PAN", "LUN", "BRE", "CRC", "BLC")
TYPE_KO: Final = {"PRO": "전립선암", "PAN": "췌장암", "LUN": "폐암", "BRE": "유방암", "CRC": "대장암", "BLC": "방광암"}
CELLS: Final = (
    ("pre->pre", "① 변경 전 학습·평가", "1_pre_pre"),
    ("pre->post", "② 변경 전 모델 → 변경 후 검체", "2_pre_post"),
    ("post->post", "③ 변경 후 재학습·평가", "3_post_post"),
)
GATE: Final = 0.5
DEFAULT_GATE_LABEL: Final = "임계값 0.5"
GATE_LABEL = DEFAULT_GATE_LABEL
MODEL: Final = "LR, 전처리 ver1"
EVALUATION: Final = "검체 단위 5-fold · 검체당 5점 · 25회 평균 확률"
BAR_STAGE2: Final = "#455A64"


def metric_bars(ax, names, values, color, title: str, subtitle: str) -> None:
    pos = np.arange(len(values))
    ax.bar(pos, values, color=color, width=0.58, zorder=3)
    for i, v in enumerate(values):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=SIZE_ANNOTATION + 1, color=COLOR_TEXT)
    ax.set_xticks(pos)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.08)
    style_axes(ax, grid="y", ylabel="값")
    set_title(ax, title, subtitle)


def header(fig, title: str, subtitle: str, note: str | None = None) -> None:
    fig.text(0.005, 1.10, title, ha="left", va="bottom", fontsize=SIZE_TITLE + 2, fontweight="bold", color=COLOR_TEXT)
    fig.text(0.005, 1.055, subtitle, ha="left", va="bottom", fontsize=SIZE_SUBTITLE + 1, color=COLOR_MUTED)
    if note:
        fig.text(0.005, -0.06, note, ha="left", va="top", fontsize=SIZE_SUBTITLE, color=COLOR_MUTED)


def resubtitle(ax, title: str, subtitle: str) -> None:
    """Replace the subtitle that performance.py wrote (English terms, 2-digit AUC) with a Korean one."""
    for txt in list(ax.texts):
        if txt.get_transform() == ax.transAxes and txt.get_position()[1] >= 1.0:
            txt.remove()
    set_title(ax, title, subtitle)


def relabel_roc(ax, labels: list[str], colors: list[str] | None = None) -> None:
    """Relabel the ROC curves (3-digit AUC) and optionally recolor them; line 0 is the chance line."""
    lines = ax.get_lines()
    for i, (line, label) in enumerate(zip(lines[1:], labels)):
        line.set_label(label)
        if colors:
            line.set_color(colors[i])
    legend(ax, loc="lower right")


def gate_prob(df: pd.DataFrame) -> np.ndarray:
    """Probability used for the stage-1 decision: fold-threshold-rescaled column when present (Youden runs), else raw."""
    return (df.p_cancer_gate if "p_cancer_gate" in df else df.p_cancer).to_numpy()


def stage1_figure(df: pd.DataFrame, label: str, slug: str) -> dict[str, float]:
    y = df.y_cancer.to_numpy().astype(int)
    p = df.p_cancer.to_numpy()
    pred = (gate_prob(df) >= GATE).astype(int)
    auc = float(roc_auc_score(y, p))
    sens = float(((pred == 1) & (y == 1)).sum() / (y == 1).sum())
    spec = float(((pred == 0) & (y == 0)).sum() / (y == 0).sum())
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.4), gridspec_kw={"wspace": 0.32})
    plot_confusion_matrix(y, pred, title="혼동행렬", class_keys=("non_cancer", "cancer"), ax=axes[0])
    counts = f"n={len(y)} (비암 {int((y == 0).sum())} / 암 {int(y.sum())})"
    resubtitle(axes[0], "혼동행렬", f"{counts} · 정확도 {float((pred == y).mean()):.3f} · {GATE_LABEL}")
    plot_roc((y, p), title="ROC", ax=axes[1])
    resubtitle(axes[1], "ROC", f"{counts} · AUC {auc:.3f}")
    relabel_roc(axes[1], [f"AUC {auc:.3f}"])
    metric_bars(axes[2], ["ROC-AUC", "균형 정확도", "민감도", "특이도"], [auc, (sens + spec) / 2, sens, spec],
                CLASS_COLORS["cancer"], "지표", GATE_LABEL)
    header(fig, f"1단계 암 선별 (암 vs 비암) — {label}",
           subtitle_from(n=len(y), n_pos=int(y.sum()), n_neg=int((1 - y).sum()), model=MODEL, evaluation=EVALUATION),
           "12개 군은 각각 한 병원 검체라 AUC에는 병원 차이가 포함됨.")
    save_png(fig, OUT / f"fig{slug}_a_stage1_screening.png")
    return {"s1_auc": auc, "s1_sensitivity": sens, "s1_specificity": spec}


def stage2_figure(df: pd.DataFrame, label: str, slug: str) -> dict[str, float]:
    """Cancer-type identification among true cancers that stage 1 called cancer (6 x 6, no non-cancer class)."""
    y7 = df.y7.to_numpy().astype(int)
    p2 = df[[f"p_type_{t}" for t in TYPES]].to_numpy()
    gated = gate_prob(df) >= GATE
    target = gated & (y7 > 0)
    missed = int((~gated & (y7 > 0)).sum())
    false_pos = int((gated & (y7 == 0)).sum())
    yt = y7[target] - 1
    pt = p2[target] / p2[target].sum(axis=1, keepdims=True)
    pred = pt.argmax(axis=1)
    idx = list(range(len(TYPES)))
    ovr = [float(roc_auc_score((yt == i).astype(int), pt[:, i])) for i in idx]
    recalls = [float((pred[yt == i] == i).mean()) for i in idx]
    macro_f1 = float(f1_score(yt, pred, labels=idx, average="macro", zero_division=0))

    fig, axes = plt.subplots(1, 3, figsize=(22, 7.2), gridspec_kw={"wspace": 0.34, "width_ratios": [1.15, 1, 0.9]})
    entered = {t: int((target & (y7 == i + 1)).sum()) for i, t in enumerate(TYPES)}
    entered_txt = " / ".join(f"{TYPE_KO[t]} {v}" for t, v in entered.items())
    plot_confusion_matrix(yt, pred, title="혼동행렬 (암종)", classes=[TYPE_KO[t] for t in TYPES], class_keys=TYPES, ax=axes[0])
    resubtitle(axes[0], "혼동행렬 (암종)", f"n={int(target.sum())} ({entered_txt}) · 정확도 {float((pred == yt).mean()):.3f}")
    curves = [(TYPE_KO[t], (yt == i).astype(int), pt[:, i]) for i, t in zip(idx, TYPES)]
    plot_roc(curves, title="암종별 ROC", ax=axes[1])
    resubtitle(axes[1], "암종별 ROC (해당 암종 vs 나머지 암종)", f"점수 = 암종 확률, n={int(target.sum())}")
    colors = group_colors()
    relabel_roc(axes[1], [f"{TYPE_KO[t]} — AUC {a:.3f}" for t, a in zip(TYPES, ovr)], [colors.get(t, COLOR_TEXT) for t in TYPES])
    metric_bars(axes[2], ["암종별 AUC 평균", "균형 정확도", "macro-F1"], [float(np.mean(ovr)), float(np.mean(recalls)), macro_f1],
                BAR_STAGE2, "지표 (6개 암종)", "암 vs 비암 판정에서 암으로 판정된 실제 암 검체")
    header(fig, f"암종 구분 (6개 암종) — {label}",
           f"n={int(target.sum())} ({entered_txt}) · {MODEL} · {EVALUATION}",
           f"평가 대상: 암 vs 비암 판정({GATE_LABEL})에서 암으로 판정된 실제 암 검체. 제외: 비암으로 판정된 암 {missed}개, "
           f"암으로 판정된 비암 {false_pos}개.\n"
           "6개 암종 기준(난소암 없음, 췌장암 = CBNUH + 연세).")
    save_png(fig, OUT / f"fig{slug}_b_stage2_cancer_type.png")
    return {"s2_n": int(target.sum()), "s2_macro_ovr_auc": float(np.mean(ovr)), "s2_balanced_accuracy": float(np.mean(recalls)),
            "s2_macro_f1": macro_f1, "s1_missed_cancer": missed, "s1_false_positive": false_pos,
            **{f"s2_recall_{t}": r for t, r in zip(TYPES, recalls)}, **{f"s2_n_{t}": v for t, v in entered.items()}}


def main() -> None:
    global OUT, MODEL, EVALUATION, GATE_LABEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SRC, help="subject_probs_mean.csv from the analysis script")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--model-label", default=MODEL)
    parser.add_argument("--evaluation", default=EVALUATION)
    parser.add_argument("--gate-label", default=DEFAULT_GATE_LABEL, help="how the stage-1 threshold is described in the figures")
    args = parser.parse_args()
    OUT, MODEL, EVALUATION, GATE_LABEL = args.out, args.model_label, args.evaluation, args.gate_label
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    probs = pd.read_csv(args.source, encoding="utf-8-sig")
    rows = []
    for cell, label, slug in CELLS:
        df = probs[probs.cell == cell].reset_index(drop=True)
        r1 = stage1_figure(df, label, slug)
        r2 = stage2_figure(df, label, slug)
        rows.append({"cell": cell, "n": len(df), **r1, **r2})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "figure_metrics.csv", index=False, encoding="utf-8-sig")
    print(summary.round(3).to_string())
    print("wrote", OUT)


if __name__ == "__main__":
    main()
