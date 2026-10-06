"""성능 그림 — ROC, PR, confusion matrix. 스타일은 :mod:`sers.visualization.style`만 쓴다.

세 함수 모두 ``(fig, ax)``를 돌려주고, ``output=`` 경로를 주면 PNG로 저장한다.
제목은 호출자가 "무엇을" 한 줄로 주고, 부제목은 ``subtitle_from`` 규칙으로
자동 생성된다 (n · 군 구성 · 모델 · 평가 · 지표).

라벨 문자열(암/비암, True label 등)은 ``lang="ko"|"en"``으로만 바꾼다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from sers.visualization.style import (
    CLASS_COLORS,
    CLASS_LABELS_EN,
    CLASS_LABELS_KO,
    COLOR_MUTED,
    COLOR_REFERENCE,
    COLOR_TEXT,
    FIGSIZE_SINGLE,
    LINEWIDTH,
    LINEWIDTH_REFERENCE,
    SEQUENTIAL_CMAP,
    SERIES_COLORS,
    SIZE_ANNOTATION,
    SIZE_LABEL,
    apply_style,
    group_colors,
    legend,
    save_png,
    set_title,
    style_axes,
    subtitle_from,
)

_AXIS = {
    "ko": {"fpr": "위양성률 (1 - 특이도)", "tpr": "민감도", "recall": "민감도 (재현율)",
           "precision": "양성예측도 (정밀도)", "true": "실제", "pred": "예측",
           "chance": "무작위 (AUC 0.50)"},
    "en": {"fpr": "False positive rate (1 - specificity)", "tpr": "Sensitivity",
           "recall": "Recall", "precision": "Precision (PPV)", "true": "True label",
           "pred": "Predicted label", "chance": "Chance (AUC 0.50)"},
}


def _fmt_ci(value: float, ci: tuple[float, float] | None, name: str) -> str:
    if ci is None:
        return f"{name} {value:.2f}"
    return f"{name} {value:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def _counts(y_true: np.ndarray) -> tuple[int, int, int]:
    y = np.asarray(y_true).astype(int)
    return int(len(y)), int((y == 1).sum()), int((y == 0).sum())


# ---------------------------------------------------------------------------
# ROC
# ---------------------------------------------------------------------------

def plot_roc(
    curves: Sequence[tuple[str, np.ndarray, np.ndarray]] | tuple[np.ndarray, np.ndarray],
    *,
    title: str,
    model: str | None = None,
    evaluation: str | None = None,
    auc_ci: Mapping[str, tuple[float, float]] | tuple[float, float] | None = None,
    lang: str = "ko",
    output: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """ROC 곡선. 한 곡선이면 ``(y_true, y_score)``, 여럿이면 ``[(이름, y_true, y_score), …]``.

    AUC는 범례에 넣는다. ``auc_ci``는 단일 곡선이면 ``(lo, hi)``, 여럿이면
    ``{이름: (lo, hi)}``. 부제목은 첫 곡선의 n으로 만든다 (모든 곡선이 같은
    환자 집합이어야 나란히 놓을 수 있다).
    """
    apply_style()
    if not isinstance(curves[0], tuple) or (len(curves) == 2 and not isinstance(curves[0][0], str)):
        curves = [("", np.asarray(curves[0]), np.asarray(curves[1]))]   # type: ignore[list-item]
    words = _AXIS[lang]
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    else:
        fig = ax.figure

    ax.plot([0, 1], [0, 1], linestyle="--", color=COLOR_REFERENCE, linewidth=LINEWIDTH_REFERENCE,
            label=words["chance"], zorder=1)
    metrics: list[str] = []
    for idx, (name, y_true, y_score) in enumerate(curves):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = float(roc_auc_score(y_true, y_score))
        ci = None
        if isinstance(auc_ci, dict):
            ci = auc_ci.get(name)
        elif auc_ci is not None and len(curves) == 1:
            ci = tuple(auc_ci)  # type: ignore[assignment]
        text = _fmt_ci(auc, ci, "AUC")
        color = CLASS_COLORS["cancer"] if len(curves) == 1 else SERIES_COLORS[idx % len(SERIES_COLORS)]
        label = text if not name else f"{name} — {text}"
        ax.plot(fpr, tpr, color=color, linewidth=LINEWIDTH, label=label, zorder=3)
        metrics.append(text if not name else f"{name} {text}")

    n, n_pos, n_neg = _counts(curves[0][1])
    style_axes(ax, xlabel=words["fpr"], ylabel=words["tpr"])
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    set_title(ax, title, subtitle_from(n=n, n_pos=n_pos, n_neg=n_neg, model=model,
                                       evaluation=evaluation,
                                       metric=metrics[0] if len(curves) == 1 else None))
    legend(ax, loc="lower right")
    if output is not None:
        save_png(fig, output, close=False)
    return fig, ax


# ---------------------------------------------------------------------------
# Precision–Recall
# ---------------------------------------------------------------------------

def plot_pr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    title: str,
    model: str | None = None,
    evaluation: str | None = None,
    ap_ci: tuple[float, float] | None = None,
    lang: str = "ko",
    output: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Precision–Recall 곡선. 유병률(양성 비율)을 기준선으로 그린다."""
    apply_style()
    words = _AXIS[lang]
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    else:
        fig = ax.figure
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = float(average_precision_score(y_true, y_score))
    n, n_pos, n_neg = _counts(y_true)
    prevalence = n_pos / n if n else 0.0
    ax.axhline(prevalence, linestyle="--", color=COLOR_REFERENCE, linewidth=LINEWIDTH_REFERENCE,
               label=f"{'유병률' if lang == 'ko' else 'Prevalence'} {prevalence:.2f}", zorder=1)
    text = _fmt_ci(ap, ap_ci, "AP")
    ax.step(recall, precision, where="post", color=CLASS_COLORS["cancer"], linewidth=LINEWIDTH,
            label=text, zorder=3)
    style_axes(ax, xlabel=words["recall"], ylabel=words["precision"])
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    set_title(ax, title, subtitle_from(n=n, n_pos=n_pos, n_neg=n_neg, model=model,
                                       evaluation=evaluation, metric=text))
    legend(ax, loc="upper right")
    if output is not None:
        save_png(fig, output, close=False)
    return fig, ax


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str,
    classes: Sequence[str] | None = None,
    class_keys: Sequence[str] | None = None,
    model: str | None = None,
    evaluation: str | None = None,
    threshold: float | None = None,
    normalize: str | None = "true",
    lang: str = "ko",
    output: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """혼동행렬. 셀에 개수와 행 기준 비율을 함께 쓴다.

    Parameters
    ----------
    classes
        표시 순서의 클래스 라벨. 생략하면 ``class_keys``에서 언어별 표시명을 만든다
        (예: ``class_keys=("non_cancer", "cancer")`` → "비암", "암").
    class_keys
        ``CLASS_COLORS`` 키 또는 config 그룹 코드. 축 라벨 옆 색 표시에 쓴다.
    normalize
        "true"(행 = 실제 클래스 기준 비율, 기본) 또는 None(개수만).
    threshold
        이진 결정 임계값. 부제목에 기록한다.
    """
    apply_style()
    words = _AXIS[lang]
    names = CLASS_LABELS_KO if lang == "ko" else CLASS_LABELS_EN
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(class_keys))) if class_keys else sorted(np.unique(np.concatenate([y_true, y_pred])))
    if classes is None:
        classes = [names.get(k, k) for k in class_keys] if class_keys else [str(x) for x in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sum = cm.sum(axis=1, keepdims=True)
    frac = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum > 0)
    shown = frac if normalize == "true" else cm

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    else:
        fig = ax.figure
    k = len(labels)
    vmax_scale = 1.0 if normalize == "true" else float(shown.max() or 1)
    palette = {**group_colors(), **CLASS_COLORS}
    if class_keys:
        # 행(실제 클래스)마다 그 클래스 색으로 흰색→클래스색 램프. 색이 정체성을 따른다.
        ax.imshow(np.zeros((k, k)), cmap="Greys", vmin=0, vmax=1, alpha=0)   # 축 범위 고정
        rgba = np.ones((k, k, 4))
        for i, key in enumerate(class_keys):
            ramp = LinearSegmentedColormap.from_list(f"cm_{key}", ["#FFFFFF", palette.get(key, COLOR_TEXT)])
            rgba[i] = ramp(np.clip(shown[i] / vmax_scale, 0, 1))
        im = ax.imshow(rgba)
    else:
        im = ax.imshow(shown, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=(1.0 if normalize == "true" else None))
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    style_axes(ax, grid=None, xlabel=words["pred"], ylabel=words["true"])
    ax.tick_params(which="major", length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    # 셀 경계: 2px 흰 간격 (인접 셀 구분)
    ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    vmax = float(np.nanmax(shown)) if np.isfinite(shown).any() else 1.0
    for i in range(k):
        for j in range(k):
            dark = (shown[i, j] / vmax if vmax else 0) > 0.55
            color = "white" if dark else COLOR_TEXT
            main = f"{cm[i, j]:d}"
            sub = f"{frac[i, j] * 100:.0f}%"
            ax.text(j, i - 0.12, main, ha="center", va="center", fontsize=SIZE_LABEL,
                    fontweight="bold", color=color)
            ax.text(j, i + 0.18, sub, ha="center", va="center", fontsize=SIZE_ANNOTATION,
                    color=color if dark else COLOR_MUTED)
    # 축 라벨 옆 클래스 색 점 — 다른 그림과 같은 색으로 정체성 연결
    if class_keys:
        for idx, key in enumerate(class_keys):
            ax.plot(-0.62, idx, marker="s", markersize=8, color=palette.get(key, COLOR_TEXT),
                    clip_on=False, zorder=5)

    n = int(cm.sum())
    groups = {classes[i]: int(row_sum[i, 0]) for i in range(k)}
    acc = float(np.trace(cm) / n) if n else float("nan")
    extra = [f"threshold {threshold:.2f}"] if threshold is not None else []
    set_title(ax, title, subtitle_from(n=n, groups=groups, model=model, evaluation=evaluation,
                                       metric=f"accuracy {acc:.2f}", extra=extra))
    if not class_keys:   # 클래스 색 램프는 행마다 색이 달라 단일 colorbar가 무의미
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=SIZE_ANNOTATION)
    if output is not None:
        save_png(fig, output, close=False)
    return fig, ax
