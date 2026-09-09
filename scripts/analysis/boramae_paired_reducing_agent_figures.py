"""Figures for the paired July-liquid vs August-mapping comparison deck.

Reads results/boramae_paired_reducing_agent/ (from
boramae_paired_reducing_agent_comparison.py) and writes PNGs next to it.
Colours follow publications/전향검체/보라매병원/slides/DESIGN.md
(July liquid = blue, August mapping = orange, Biopsy-negative = purple).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

REPO: Final = Path(__file__).resolve().parents[2]
RES: Final = REPO / "results" / "boramae_paired_reducing_agent"
BLUE, ORANGE, PURPLE, GREEN, INK, MUTED = "#2C6E9B", "#A94712", "#72528F", "#496D57", "#202321", "#646862"
GROUP_COLOR = {"Control": BLUE, "Biopsy-negative": PURPLE, "Prostate cancer": ORANGE}
COND_LABEL = {"july_liquid": "7월 점측정 5회 (변경 전)", "aug_mapping": "8월 mapping 121점 (변경 후)", "aug_mapping_5": "8월 mapping 5점 추출"}
COND_COLOR = {"july_liquid": BLUE, "aug_mapping": ORANGE, "aug_mapping_5": "#D4884F"}

plt.rcParams.update({
    "font.family": ["Noto Sans CJK KR", "Noto Sans KR", "NanumGothic", "DejaVu Sans"],
    "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#D8D5CD", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "font.size": 12, "figure.dpi": 160, "savefig.dpi": 160, "figure.facecolor": "#FFFEFB", "axes.facecolor": "#FFFEFB",
})


def load_rows(name: str) -> list[dict[str, str]]:
    with (RES / name).open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def fig_roc(summary: dict, oof: list[dict[str, str]]) -> None:
    truth = np.array([row["label"] == "Prostate cancer" for row in oof], dtype=int)
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    dist = {(r["task"], r["metric"], r["condition"]): r for r in summary["repeat_summary"]}
    for cond, key in (("july_liquid", "p_cancer_july"), ("aug_mapping", "p_cancer_mapping"), ("aug_mapping_5", "p_cancer_mapping5")):
        scores = np.array([float(row[key]) for row in oof])
        fpr, tpr, _ = roc_curve(truth, scores)
        d = dist[("screening_binary", "roc_auc", cond)]
        ax.plot(fpr, tpr, color=COND_COLOR[cond], lw=2.6 if cond != "aug_mapping_5" else 1.6,
                ls="-" if cond != "aug_mapping_5" else "--",
                label=f"{COND_LABEL[cond]}  AUC {d['mean']:.3f} ± {d['sd']:.3f} ({summary['repeats']}회 fold 반복)")
    ax.plot([0, 1], [0, 1], color="#D8D5CD", lw=1)
    ax.set_xlabel("1 − 특이도 (False positive rate)")
    ax.set_ylabel("민감도 (True positive rate)")
    ax.set_title(f"Screening: 정상+비암 vs 암 — 동일 환자 {summary['n_paired']}명, OOF 확률 (곡선은 fold 반복 1회분)", fontsize=11.5, color=INK, loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(RES / "fig_roc_paired.png")
    plt.close(fig)


def fig_shift(oof: list[dict[str, str]], paired: list[dict]) -> None:
    labels = np.array([row["label"] for row in oof])
    pj = np.array([float(r["p_cancer_july"]) for r in oof])
    pm = np.array([float(r["p_cancer_mapping"]) for r in oof])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    for lab in ("Control", "Biopsy-negative", "Prostate cancer"):
        mask = labels == lab
        ax.scatter(pj[mask], pm[mask], s=34, color=GROUP_COLOR[lab], alpha=.85, edgecolor="white", lw=.6, label=f"{lab} (n={mask.sum()})")
    ax.plot([0, 1], [0, 1], color="#D8D5CD", lw=1)
    ax.axhline(.5, color="#D8D5CD", lw=.8, ls=":")
    ax.axvline(.5, color="#D8D5CD", lw=.8, ls=":")
    ax.set_xlabel("7월 점측정 5회: P(암)  (OOF, repeat 0)")
    ax.set_ylabel("8월 mapping 121점: P(암)  (OOF, repeat 0)")
    ax.set_title("환자별 암 확률 — 같은 환자를 두 조건에서", fontsize=12, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax = axes[1]
    order = np.argsort(pm - pj)
    delta = (pm - pj)[order]
    cols = [GROUP_COLOR[lab] for lab in labels[order]]
    ax.bar(np.arange(len(delta)), delta, color=cols, width=1.0)
    ax.axhline(0, color=INK, lw=.8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GROUP_COLOR[g], label=g) for g in ("Control", "Biopsy-negative", "Prostate cancer")], frameon=False, fontsize=9, loc="upper left")
    row = next(p for p in paired if p["comparison"] == "july_liquid -> aug_mapping" and p["repeat"] == 0)
    ax.set_title(f"확률 변화 (8월 − 7월, repeat 0): 중앙값 {row['prob_shift_median']:+.3f}, 판정 뒤집힘 {row['subjects_flipped']}명", fontsize=11.5, color=INK, loc="left")
    ax.set_xlabel("환자 (변화량 정렬)")
    ax.set_ylabel("ΔP(암)")
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(RES / "fig_prob_shift.png")
    plt.close(fig)


def fig_spectra(summary: dict) -> None:
    grid = np.array(summary["grid"])
    mbg = summary["mean_by_group"]
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.2), sharex=True)
    for ax, lab in zip(axes, ("Control", "Biopsy-negative", "Prostate cancer")):
        ax.plot(grid, mbg["july_liquid"][lab], color=BLUE, lw=1.6, label="7월 점측정 5회 (변경 전)")
        ax.plot(grid, mbg["aug_mapping"][lab], color=ORANGE, lw=1.6, label="8월 mapping 121점 (변경 후)")
        ax.set_ylabel("SNV 강도")
        ax.text(.995, .9, f"{lab} (n={summary['labels'][lab]})", transform=ax.transAxes, ha="right", color=GROUP_COLOR[lab], fontweight="bold")
    axes[0].legend(frameon=False, ncol=2, loc="upper left", fontsize=9.5)
    axes[0].set_title("임상군별 평균 스펙트럼 — 전처리 후 (SG → baseline → SNV), 동일 환자, 군 평균", fontsize=12, color=INK, loc="left")
    axes[-1].set_xlabel("Raman shift (cm⁻¹)")
    fig.tight_layout()
    fig.savefig(RES / "fig_group_mean_spectra.png")
    plt.close(fig)


def fig_metric_bars(summary: dict) -> None:
    dist = {(r["task"], r["metric"], r["condition"]): r for r in summary["repeat_summary"]}
    per_repeat = {}
    for m in summary["metrics"]:
        per_repeat.setdefault((m["task"], m["condition"]), []).append(m)
    conds = ["july_liquid", "aug_mapping", "aug_mapping_5"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1, .9]})
    for ax, (task, key, title) in zip(axes[:2], (("screening_binary", "roc_auc", "Screening ROC-AUC (정상+비암 vs 암)"), ("three_group", "macro_ovr_roc_auc", "3군 macro OvR ROC-AUC"))):
        means = [dist[(task, key, c)]["mean"] for c in conds]
        sds = [dist[(task, key, c)]["sd"] for c in conds]
        ax.bar(range(3), means, color=[COND_COLOR[c] for c in conds], width=.62, yerr=sds, capsize=5, ecolor=INK, error_kw={"lw": 1.2})
        for i, c in enumerate(conds):
            vals = [float(m[key]) for m in per_repeat[(task, c)]]
            ax.scatter(np.full(len(vals), i) + np.linspace(-.16, .16, len(vals)), vals, s=14, color=INK, alpha=.55, zorder=3)
            ax.text(i, means[i] + sds[i] + .015, f"{means[i]:.3f}", ha="center", fontsize=11, color=INK, fontweight="bold")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["7월 점측정 5회\n(변경 전)", "8월 mapping 121점\n(변경 후)", "8월 mapping\n5점 추출"], fontsize=9.5)
        ax.axhline(.5, color="#D8D5CD", lw=.8, ls=":")
        ax.text(2.45, .505, "0.5 = 우연", fontsize=8, color=MUTED, ha="right", va="bottom")
        ax.set_ylim(.3, 1.0)
        ax.set_ylabel(key.replace("_", " ").replace("roc auc", "ROC-AUC").replace("macro ovr ROC-AUC", "macro OvR ROC-AUC") + "  (축 0.3부터)")
        ax.set_title(f"{title}\n평균 ± SD, 점 = fold 반복 {summary['repeats']}회 각각 · n={summary['n_paired']} · OOF", fontsize=11, color=INK, loc="left")
    ax = axes[2]
    comps = ["july_liquid -> aug_mapping", "july_liquid -> aug_mapping_5", "aug_mapping -> aug_mapping_5"]
    names = ["7월 → 8월(121점)", "7월 → 8월(5점)", "8월 121점 → 5점"]
    for i, comp in enumerate(comps):
        deltas = [p["delta_auc"] for p in summary["paired"] if p["comparison"] == comp]
        ax.scatter(np.full(len(deltas), i) + np.linspace(-.14, .14, len(deltas)), deltas, s=18, color=INK, alpha=.6, zorder=3)
        ax.plot([i - .28, i + .28], [np.mean(deltas)] * 2, color=ORANGE, lw=2.4)
    ax.axhline(0, color="#D8D5CD", lw=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylabel("ΔAUC (뒤 − 앞), 동일 환자 DeLong")
    ax.set_title("Screening AUC 차이 — fold 반복별\n(선 = 평균)", fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(RES / "fig_metric_bars.png")
    plt.close(fig)


def main() -> None:
    summary = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    oof = load_rows("subject_oof.csv")
    fig_roc(summary, oof)
    fig_shift(oof, summary["paired"])
    fig_spectra(summary)
    fig_metric_bars(summary)
    print("figures written to", RES)


if __name__ == "__main__":
    main()
