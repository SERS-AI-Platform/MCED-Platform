"""SERS-AI 그림 스타일 단일 기준 (SSOT).

모든 결과 그림(ROC, confusion matrix, 스펙트럼, 분포 …)은 이 모듈의 상수와
헬퍼만 쓴다. 스크립트마다 글꼴·크기·색을 다시 정하지 않는다.

확정된 규칙 (2026-09-09, 사용자 결정)
-------------------------------------
- 저장 형식: **PNG만**. pdf/svg/eps는 ``save_png(..., extra_formats=...)``로
  명시할 때만 만든다. 확인용 렌더링은 프로젝트 폴더에 남기지 않는다.
- 글꼴: 영문 Arial, 한글 NanumGothic 자동 fallback. 보고용 큰 글씨
  (제목 16 / 부제목 11 / 축 13 / 눈금 11 / 범례 11).
- 제목 = 무엇을 그렸는가(왼쪽 정렬, 굵게). 부제목 = 조건(n, 군 구성, 모델,
  평가 방식, 핵심 지표)을 ``subtitle_from()``으로 자동 생성.
- 색:
  * 클래스(암 / 비암 / 정상 / 질환대조)는 ``CLASS_COLORS``. config.yaml의
    category_colors(빨강 #E53935 / 초록 #43A047)는 적록색약에서 구분되지 않아
    (ΔE 2.9, 기준 8) 검증기를 통과한 값으로 대체했다.
  * 암종별 색은 config.yaml ``display.group_colors`` (사용자 결정). 스펙트럼
    그림과 type-ID 혼동행렬이 같은 색을 쓴다.
- 축: 위·오른쪽 spine 제거, 옅은 회색 격자, 눈금은 바깥쪽.

사용 예
-------
>>> from sers.visualization.style import apply_style, save_png, set_title
>>> apply_style()
>>> fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
>>> set_title(ax, "ROC — 전립선암 vs 비암", subtitle_from(n=112, n_pos=43, n_neg=69,
...           model="LR", evaluation="5-fold OOF", metric="AUC 0.82 [0.73, 0.89]"))
>>> save_png(fig, "results/x/roc.png")
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 글꼴 · 크기
# ---------------------------------------------------------------------------

FONT_STACK: tuple[str, ...] = ("Arial", "NanumGothic", "Malgun Gothic", "DejaVu Sans")

SIZE_TITLE = 16
SIZE_SUBTITLE = 11
SIZE_LABEL = 13
SIZE_TICK = 11
SIZE_LEGEND = 11
SIZE_ANNOTATION = 10
LINEWIDTH = 2.0
LINEWIDTH_REFERENCE = 1.2   # chance line 등 보조선
DPI = 200

FIGSIZE_SINGLE = (6.4, 5.2)      # 한 패널 (ROC, CM)
FIGSIZE_WIDE = (10.0, 5.0)       # 두 패널 나란히

# ---------------------------------------------------------------------------
# 색
# ---------------------------------------------------------------------------

COLOR_TEXT = "#1A1A1A"
COLOR_MUTED = "#5F6368"
COLOR_GRID = "#D9DEE2"
COLOR_REFERENCE = "#9AA0A6"     # chance line, 보조 기준선
COLOR_SURFACE = "#FFFFFF"

#: 클래스 색 — dataviz 검증기 통과 (CVD ΔE ≥ 14.8, 대비 ≥ 3:1)
CLASS_COLORS: dict[str, str] = {
    "cancer": "#C62828",
    "non_cancer": "#1565C0",
    "control": "#1565C0",
    "disease_control": "#EF6C00",
}

#: 클래스 한글/영문 표시명 (라벨 문자열은 여기서만 정한다)
CLASS_LABELS_KO: dict[str, str] = {
    "cancer": "암", "non_cancer": "비암", "control": "정상", "disease_control": "질환대조",
}
CLASS_LABELS_EN: dict[str, str] = {
    "cancer": "Cancer", "non_cancer": "Non-cancer", "control": "Control",
    "disease_control": "Disease control",
}

#: 여러 모델/조건을 한 축에 겹칠 때의 고정 순서 색 (순환 금지 — 8개 초과는 분할)
SERIES_COLORS: tuple[str, ...] = (
    "#C62828", "#1565C0", "#EF6C00", "#2E7D32", "#6A1B9A", "#00838F", "#5D4037", "#455A64",
)

#: 혼동행렬 등 단일 색상 크기 표현
SEQUENTIAL_CMAP = "Blues"

_FALLBACK_GROUP_COLORS: dict[str, str] = {
    "PRO": "#E91E63", "BRE": "#FF69B4", "OVA": "#AB47BC", "LUN": "#42A5F5",
    "CRC": "#EF5350", "PAN": "#FFA726", "CPAN": "#FF8F00", "YPAN": "#FFB300",
    "SPAN": "#E65100", "BLC": "#7E57C2", "DIA": "#66BB6A", "HBP": "#26A69A",
    "H.D.": "#78909C", "NOR": "#8D6E63", "YNOR": "#A1887F",
}


def group_colors() -> dict[str, str]:
    """암종/그룹 색. config.yaml ``display.group_colors``가 기준, 못 읽으면 내장 사본."""
    try:
        from sers.config import load_config

        colors = dict(load_config().display.group_colors)
        if colors:
            return colors
    except Exception as exc:  # noqa: BLE001 — 그림 때문에 파이프라인이 죽으면 안 됨
        logger.debug("config group_colors unavailable (%s); using fallback", exc)
    return dict(_FALLBACK_GROUP_COLORS)


def color_for(key: str) -> str:
    """클래스 키 또는 그룹 코드에 대한 고정 색. 모르는 키는 텍스트 색."""
    if key in CLASS_COLORS:
        return CLASS_COLORS[key]
    return group_colors().get(key, COLOR_TEXT)


# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------

def rcparams() -> dict[str, object]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": list(FONT_STACK),
        "axes.unicode_minus": False,
        "font.size": SIZE_TICK,
        "axes.titlesize": SIZE_TITLE,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": SIZE_LABEL,
        "axes.labelcolor": COLOR_TEXT,
        "axes.edgecolor": COLOR_TEXT,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COLOR_GRID,
        "grid.linewidth": 0.8,
        "grid.alpha": 0.9,
        "xtick.labelsize": SIZE_TICK,
        "ytick.labelsize": SIZE_TICK,
        "xtick.color": COLOR_TEXT,
        "ytick.color": COLOR_TEXT,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": SIZE_LEGEND,
        "legend.frameon": False,
        "lines.linewidth": LINEWIDTH,
        "figure.facecolor": COLOR_SURFACE,
        "figure.dpi": 100,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "savefig.facecolor": COLOR_SURFACE,
    }


def apply_style() -> None:
    """프로세스 전역에 SERS-AI 스타일 적용. 그림 스크립트 맨 앞에서 한 번 호출."""
    mpl.rcParams.update(rcparams())


# ---------------------------------------------------------------------------
# 제목 · 부제목 · 축
# ---------------------------------------------------------------------------

def subtitle_from(
    *,
    n: int | None = None,
    n_pos: int | None = None,
    n_neg: int | None = None,
    groups: Mapping[str, int] | None = None,
    model: str | None = None,
    evaluation: str | None = None,
    metric: str | None = None,
    extra: Iterable[str] = (),
) -> str:
    """조건 부제목을 고정 순서로 조립한다: n · 군 구성 · 모델 · 평가 · 지표 · 기타.

    >>> subtitle_from(n=112, n_pos=43, n_neg=69, model="LR", evaluation="5-fold OOF",
    ...               metric="AUC 0.82 [0.73, 0.89]")
    'n=112 (암 43 / 비암 69) · LR · 5-fold OOF · AUC 0.82 [0.73, 0.89]'
    """
    parts: list[str] = []
    if n is not None:
        head = f"n={n}"
        if groups:
            head += " (" + " / ".join(f"{k} {v}" for k, v in groups.items()) + ")"
        elif n_pos is not None and n_neg is not None:
            head += f" (암 {n_pos} / 비암 {n_neg})"
        parts.append(head)
    for item in (model, evaluation, metric, *extra):
        if item:
            parts.append(str(item))
    return " · ".join(parts)


def set_title(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    """제목(굵게, 왼쪽)과 부제목(작게, 회색)을 고정 위치에 놓는다."""
    ax.set_title(title, fontsize=SIZE_TITLE, fontweight="bold", loc="left",
                 color=COLOR_TEXT, pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=SIZE_SUBTITLE, color=COLOR_MUTED)


def style_axes(ax: plt.Axes, *, grid: str | None = "both", xlabel: str | None = None,
               ylabel: str | None = None) -> None:
    """spine·격자·축 라벨을 고정 규칙으로."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.0)
        ax.spines[side].set_color(COLOR_TEXT)
    ax.tick_params(axis="both", labelsize=SIZE_TICK, colors=COLOR_TEXT, length=4, width=0.8)
    if grid:
        ax.grid(True, axis=grid, color=COLOR_GRID, linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=SIZE_LABEL, labelpad=6)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=SIZE_LABEL, labelpad=6)


def legend(ax: plt.Axes, *, loc: str = "lower right", **kwargs) -> None:
    ax.legend(loc=loc, fontsize=SIZE_LEGEND, frameon=False, **kwargs)


# ---------------------------------------------------------------------------
# 저장 — PNG만
# ---------------------------------------------------------------------------

def save_png(
    fig: plt.Figure,
    path: str | Path,
    *,
    dpi: int = DPI,
    extra_formats: Sequence[str] = (),
    close: bool = True,
) -> Path:
    """그림을 PNG로 저장한다. 다른 확장자를 주면 .png로 바꾸고 경고한다.

    pdf/svg 등이 정말 필요하면 ``extra_formats=("pdf",)``처럼 **명시**한다.
    """
    path = Path(path)
    if path.suffix.lower() != ".png":
        warnings.warn(f"그림은 PNG만 저장합니다: {path.name} → {path.stem}.png "
                      "(다른 형식은 extra_formats로 명시)", stacklevel=2)
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=COLOR_SURFACE)
    for fmt in extra_formats:
        fig.savefig(path.with_suffix(f".{fmt.lstrip('.')}"), bbox_inches="tight",
                    facecolor=COLOR_SURFACE)
    if close:
        plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path
