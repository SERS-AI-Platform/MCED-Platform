"""SOLUM SaMD layered-architecture diagram (option-agnostic base layers +
the two layers that actually change across Option A/B/C: Persistence/Storage
and Authentication & Audit). Mirrors the "Layered Architecture of the
Clinical Analysis Web Application" reference diagram style.

Usage: python scripts/visualization/generate_samd_layered_architecture.py
Output: docs/assets/images/samd_layered_architecture.png
"""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "assets" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for f in fm.fontManager.ttflist:
    if "NanumSquare" in f.name and "Bold" not in f.name:
        plt.rcParams["font.family"] = f.name
        break
else:
    plt.rcParams["font.family"] = "Noto Sans KR"
plt.rcParams["axes.unicode_minus"] = False

EDGE = "#263238"

LAYERS = [
    (
        "화면/입력 계층\n(Presentation Layer)",
        "#1e88e5",
        "#e3f2fd",
        ["로그인 화면", "환자정보 입력", "스펙트럼 업로드", "결과 조회 화면"],
    ),
    (
        "업무 흐름 계층\n(Application/Workflow Layer)",
        "#00897b",
        "#e0f2f1",
        ["검사 세션 관리", "단계 순서 강제(QC→분석→보고서)", "파일 처리 조율", "작업 상태 제어"],
    ),
    (
        "전처리·QC 계층\n(Data Processing & QC Layer)",
        "#43a047",
        "#e8f5e9",
        ["스펙트럼 전처리", "형식·범위 검증", "신호강도/노이즈/포화/상관도 QC", "결과 집계"],
    ),
    (
        "AI 추론 계층\n(ML Inference Layer)",
        "#7b1fa2",
        "#f3e5f5",
        ["모델 로딩(버전 고정)", "uSERS-Net 추론", "SSI·Risk 산출", "판정 threshold 적용"],
    ),
    (
        "저장 계층\n(Persistence/Storage Layer)",
        "#1565c0",
        "#e8eaf6",
        ["환자·검사·QC·분석결과 DB", "업로드 원본 파일", "모델 아티팩트", "시스템 로그"],
    ),
    (
        "보고서 계층\n(Reporting Layer)",
        "#f9a825",
        "#fff8e1",
        ["보고서 생성", "PDF/CSV export", "요약 출력"],
    ),
]

OPTION_SENSITIVE = {"저장 계층\n(Persistence/Storage Layer)"}


def draw_box(ax, x, y, w, h, color_edge, color_fill, lw=1.8, ls="solid"):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.05",
        linewidth=lw,
        edgecolor=color_edge,
        facecolor=color_fill,
        linestyle=ls,
    )
    ax.add_patch(box)
    return box


def main():
    fig, ax = plt.subplots(figsize=(12.5, 15))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 15)
    ax.axis("off")

    ax.text(6.0, 14.6, "SOLUM SaMD 레이어 구조", ha="center", fontsize=19, fontweight="bold")
    ax.text(
        6.0,
        14.15,
        "(1차 목표: 이 구조로 dGMP 인증 획득 — 병원별 BM/최종 벤더 선정은 후순위 결정 사항)",
        ha="center",
        fontsize=10.5,
        color="#c62828",
        fontweight="bold",
    )

    # human user
    ax.text(0.9, 13.2, "●", ha="center", fontsize=30, color="#455a64")
    ax.text(0.9, 12.65, "사용자\n(검사실 기사 등)", ha="center", fontsize=9)
    arrow = FancyArrowPatch(
        (1.5, 13.05), (2.35, 13.05), arrowstyle="-|>", mutation_scale=16, linewidth=1.6, color=EDGE
    )
    ax.add_patch(arrow)

    layer_x, layer_w = 2.35, 6.9
    layer_h = 1.55
    gap = 0.28
    top_y = 13.3

    sidebar_x = layer_x + layer_w + 0.5
    sidebar_w = 2.6

    layer_ys = []
    for i, (title, edge_c, fill_c, items) in enumerate(LAYERS):
        y = top_y - i * (layer_h + gap) - layer_h
        layer_ys.append(y)
        sensitive = title in OPTION_SENSITIVE
        draw_box(
            ax,
            layer_x,
            y,
            layer_w,
            layer_h,
            edge_c,
            fill_c,
            lw=2.6 if sensitive else 1.8,
            ls="solid",
        )
        ax.text(
            layer_x + 0.25,
            y + layer_h - 0.32,
            title,
            fontsize=12.5,
            fontweight="bold",
            color=edge_c,
            va="top",
        )
        item_text = "   ·   ".join(items)
        ax.text(
            layer_x + 0.25,
            y + 0.28,
            item_text,
            fontsize=8.7,
            color="#37474f",
            va="bottom",
            wrap=True,
        )
        if sensitive:
            ax.text(
                layer_x + layer_w - 0.15,
                y + layer_h - 0.15,
                "◀ Option A/B/C별로 구현 방식이 다름",
                fontsize=8.3,
                color="#c62828",
                ha="right",
                va="top",
                fontweight="bold",
            )
        if i > 0:
            prev_y = layer_ys[i - 1]
            arr = FancyArrowPatch(
                (layer_x + layer_w / 2, prev_y),
                (layer_x + layer_w / 2, y + layer_h),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.4,
                color=EDGE,
            )
            ax.add_patch(arr)

    # sidebar: Authentication & Audit Layer
    sidebar_top = top_y
    sidebar_bottom = layer_ys[-1]
    sidebar_h = sidebar_top - sidebar_bottom
    sb_edge, sb_fill = "#c62828", "#ffebee"
    draw_box(ax, sidebar_x, sidebar_bottom, sidebar_w, sidebar_h, sb_edge, sb_fill, lw=2.6)
    ax.text(
        sidebar_x + sidebar_w / 2,
        sidebar_bottom + sidebar_h - 0.4,
        "인증·감사 계층\n(Authentication\n& Audit Layer)",
        ha="center",
        fontsize=11.5,
        fontweight="bold",
        color=sb_edge,
    )
    ax.text(
        sidebar_x + sidebar_w / 2,
        sidebar_bottom + sidebar_h - 1.7,
        "· 로그인/세션 보안\n· Audit Trail(누가/언제/무엇을)\n· 이벤트 자동 기록\n· 권한별 접근 제어",
        ha="center",
        fontsize=8.8,
        color="#37474f",
        linespacing=1.8,
    )
    ax.text(
        sidebar_x + sidebar_w / 2,
        sidebar_bottom + 0.35,
        "◀ Option A/B/C별로\n구현 방식이 다름",
        ha="center",
        fontsize=8.5,
        color="#c62828",
        fontweight="bold",
    )

    for y in layer_ys:
        arr = FancyArrowPatch(
            (layer_x + layer_w, y + layer_h / 2),
            (sidebar_x, y + layer_h / 2),
            arrowstyle="<|-|>",
            mutation_scale=12,
            linewidth=1.1,
            color="#90a4ae",
            linestyle="dashed",
        )
        ax.add_patch(arr)

    # bottom outputs
    out_y = layer_ys[-1] - 2.0
    outs = [
        (layer_x, "데이터베이스\n(Database)"),
        (layer_x + 1.85, "업로드 원본파일\n(Raw Spectra)"),
        (layer_x + 3.7, "모델 아티팩트\n(Model Artifacts)"),
        (layer_x + 5.55, "보고서/출력\n(Reports/Output)"),
    ]
    out_w, out_h = 1.65, 1.0
    persistence_y = layer_ys[4]
    reporting_y = layer_ys[5]
    for x, label in outs:
        draw_box(ax, x, out_y, out_w, out_h, "#546e7a", "#eceff1", lw=1.4)
        ax.text(x + out_w / 2, out_y + out_h / 2, label, ha="center", va="center", fontsize=8.3)
        src_y = reporting_y if "보고서" in label else persistence_y
        arr = FancyArrowPatch(
            (x + out_w / 2, src_y),
            (x + out_w / 2, out_y + out_h),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.1,
            color="#90a4ae",
            linestyle="dashed",
        )
        ax.add_patch(arr)

    ax.text(
        6.0,
        out_y - 0.45,
        "화면/입력 · 업무흐름 · 전처리·QC · AI추론 · 보고서 계층은 Option A/B/C와 무관하게 동일 — "
        "옵션 차이는 저장 계층과 인증·감사 계층에서만 발생",
        ha="center",
        fontsize=9.3,
        color="#455a64",
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "samd_layered_architecture.png", dpi=170)
    plt.close(fig)
    print("Saved:", OUT_DIR / "samd_layered_architecture.png")


if __name__ == "__main__":
    main()
