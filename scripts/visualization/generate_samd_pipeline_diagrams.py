"""Option-specific app-structure diagrams for dGMP review.

The output intentionally keeps the existing filenames used by
docs/architecture/samd_infra_security_options.md:

    docs/assets/images/pipeline_option_{a,b,c}.png

Each figure maps one infrastructure option onto the same SaMD web-app layers.
The point is not to prescribe a hospital deployment or business model. The
figures show how each option would be explained during dGMP review: which
layers remain identical, which layers change, and what evidence package would
be prepared for data integrity and auditability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import fill

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "assets" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for font in fm.fontManager.ttflist:
    if "NanumSquare" in font.name and "Bold" not in font.name:
        plt.rcParams["font.family"] = font.name
        break
else:
    plt.rcParams["font.family"] = "Noto Sans KR"
plt.rcParams["axes.unicode_minus"] = False

EDGE = "#263238"
TEXT = "#263238"
MUTED = "#546e7a"
WARNING = "#c62828"


@dataclass(frozen=True)
class Layer:
    title: str
    subtitle: str
    edge: str
    fill: str
    items: tuple[str, ...]
    option_sensitive: bool = False


BASE_LAYERS = (
    Layer(
        "화면/입력 계층",
        "Presentation",
        "#1e88e5",
        "#e3f2fd",
        ("로그인", "환자/검체 입력", "스펙트럼 업로드", "결과 조회"),
    ),
    Layer(
        "업무 흐름 계층",
        "Application / Workflow",
        "#00897b",
        "#e0f2f1",
        ("검사 세션", "QC→분석→보고서 순서", "파일 처리", "작업 상태"),
    ),
    Layer(
        "전처리·QC 계층",
        "Data Processing & QC",
        "#43a047",
        "#e8f5e9",
        ("전처리", "형식/범위 검증", "품질관리", "집계"),
    ),
    Layer(
        "AI 추론 계층",
        "ML Inference",
        "#7b1fa2",
        "#f3e5f5",
        ("모델 버전 고정", "uSERS-Net 추론", "SSI/Risk 산출", "판정 기준 적용"),
    ),
    Layer(
        "저장 계층",
        "Persistence / Storage",
        "#1565c0",
        "#e8eaf6",
        ("DB", "원본 파일", "모델 아티팩트", "로그"),
        True,
    ),
    Layer(
        "보고서 계층",
        "Reporting",
        "#f9a825",
        "#fff8e1",
        ("보고서 생성", "PDF/CSV export", "요약 출력"),
    ),
)


OPTION_COPY = {
    "a": {
        "title": "Option A — dGMP 인증용 로컬 NAS 저장·감사증적 구조",
        "subtitle": "병원 구축/BM 결정이 아니라, 인증 심사에서 raw·DB·audit trail을 로컬 통제환경으로 증명하는 안",
        "storage_title": "저장 계층 구현",
        "storage_body": (
            "NAS + 로컬 DB\n"
            "raw spectra, DB, 모델 파일, 보고서 저장\n"
            "RAID/스냅샷/오프사이트 백업으로 보존성 보강\n"
            "업로드 시 SHA-256 해시로 원본성 확인"
        ),
        "audit_title": "인증·감사 계층 구현",
        "audit_body": (
            "자체 앱 로그인/RBAC + 로컬 audit_logs\n"
            "NAS 접근 로그와 앱 로그를 이중 보관\n"
            "누가/언제/무엇을 했는지 회사가 직접 검증\n"
            "백업·복구 시험 기록까지 회사 책임"
        ),
        "evidence_title": "dGMP 심사에서 제시할 증거",
        "evidence": (
            "NAS 구성·접근권한 SOP",
            "raw 파일 해시/무결성 기록",
            "앱 DB audit trail 샘플",
            "백업·복구 시험 결과",
            "계정 권한분리 및 변경관리 기록",
        ),
        "risk": "검증 부담은 크지만 구조가 단순하고 운영환경 변경 리스크가 낮음",
        "risk_color": "#2e7d32",
        "output": "pipeline_option_a.png",
    },
    "b": {
        "title": "Option B — dGMP 인증용 클라우드 저장·감사증적 구조",
        "subtitle": "국내 리전·암호화·WORM·감사로그 구성을 인증 문서에 고정해 데이터 무결성을 설명하는 안",
        "storage_title": "저장 계층 구현",
        "storage_body": (
            "국내 클라우드 VPC + Object Storage + 관리형 DB\n"
            "raw 파일은 Object Lock/WORM으로 수정·삭제 방지\n"
            "DB/파일/보고서는 KMS 기반 암호화\n"
            "모델 아티팩트와 배포 이력은 버전 고정"
        ),
        "audit_title": "인증·감사 계층 구현",
        "audit_body": (
            "IdP/MFA/RBAC + 클라우드 네이티브 감사로그\n"
            "CloudTrail급 로그와 앱 audit_logs를 병행\n"
            "VPN/고정 IP/방화벽으로 접근 경로 제한\n"
            "IaC 또는 설정 baseline으로 환경 재현성 확보"
        ),
        "evidence_title": "dGMP 심사에서 제시할 증거",
        "evidence": (
            "국내 리전·VPC·KMS 구성 명세",
            "Object Lock/WORM 보존 정책",
            "클라우드 감사로그 보존 정책",
            "벤더 보안 인증/책임분담 문서",
            "서비스 구성 변경관리 및 재검토 기준",
        ),
        "risk": "확장성은 좋지만 클라우드 서비스/리전/구성 변경 시 인허가 재검토 리스크가 있음",
        "risk_color": "#c62828",
        "output": "pipeline_option_b.png",
    },
    "c": {
        "title": "Option C — dGMP 인증용 검증된 기록 플랫폼 활용 구조",
        "subtitle": "raw 저장은 A/B 중 하나로 두고, 환자·QC·분석결과·audit trail 기록 계층을 검증된 EDC/CDMS에 위임하는 안",
        "storage_title": "저장 계층 구현",
        "storage_body": (
            "원본 스펙트럼/모델/보고서는 Option A 또는 B 저장소 사용\n"
            "환자정보, QC 결과, 분석결과, eCRF는 REDCap/EDC에 기록\n"
            "앱 DB는 운영용 최소 메타데이터만 보유 가능\n"
            "raw 파일 ID와 EDC record ID를 연결해 추적"
        ),
        "audit_title": "인증·감사 계층 구현",
        "audit_body": (
            "EDC/CDMS 내장 RBAC·전자서명·audit trail 활용\n"
            "기록 조작 방지와 변경 이력 검증을 벤더 검증 패키지로 보강\n"
            "회사는 raw 저장소(A/B)와 EDC 사이의 책임분담을 문서화\n"
            "검증된 플랫폼을 써서 자체 audit 기능 검증 부담을 줄임"
        ),
        "evidence_title": "dGMP 심사에서 제시할 증거",
        "evidence": (
            "EDC/CDMS 검증 패키지 또는 Part 11-ready 자료",
            "eCRF/데이터 입력·수정 이력 샘플",
            "raw 저장소(A/B) 무결성 증거",
            "raw 파일 ID ↔ EDC record ID 추적표",
            "벤더/제조사 책임분담 및 export SOP",
        ),
        "risk": "감사증적 검증 부담은 가장 낮지만, raw 저장소는 결국 A/B 중 하나의 증거를 함께 제출해야 함",
        "risk_color": "#6a1b9a",
        "output": "pipeline_option_c.png",
    },
}


def wrap_text(text: str, width: int = 36) -> str:
    return "\n".join(fill(line, width=width) for line in text.splitlines())


def draw_rounded_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    edge: str,
    fill_color: str,
    lw: float = 1.8,
    linestyle: str = "solid",
    radius: float = 0.055,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.018,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=fill_color,
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    return patch


def draw_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = EDGE,
    dashed: bool = False,
):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.35,
        color=color,
        linestyle="dashed" if dashed else "solid",
    )
    ax.add_patch(arrow)


def draw_layer(ax, layer: Layer, x: float, y: float, w: float, h: float):
    lw = 2.6 if layer.option_sensitive else 1.55
    draw_rounded_box(ax, x, y, w, h, layer.edge, layer.fill, lw=lw)

    ax.text(
        x + 0.24,
        y + h - 0.26,
        layer.title,
        ha="left",
        va="top",
        fontsize=11.3,
        fontweight="bold",
        color=layer.edge,
    )
    ax.text(
        x + w - 0.24,
        y + h - 0.28,
        layer.subtitle,
        ha="right",
        va="top",
        fontsize=8.0,
        color=layer.edge,
    )
    item_text = "  ·  ".join(layer.items)
    ax.text(x + 0.24, y + 0.29, item_text, ha="left", va="bottom", fontsize=8.0, color=TEXT)

    if layer.option_sensitive:
        ax.text(
            x + w - 0.22,
            y + 0.29,
            "Option별 구현 차이",
            ha="right",
            va="bottom",
            fontsize=8.3,
            color=WARNING,
            fontweight="bold",
        )


def draw_option_panel(ax, copy: dict[str, object], x: float, y: float, w: float, h: float):
    draw_rounded_box(ax, x, y, w, h, "#455a64", "#fafafa", lw=1.7)
    ax.text(
        x + w / 2,
        y + h - 0.38,
        "Option별로 달라지는 2개 계층",
        ha="center",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        color=TEXT,
    )

    section_gap = 0.28
    section_h = (h - 0.9 - 0.35 - section_gap) / 2
    top_y = y + h - 0.9 - section_h
    bottom_y = top_y - section_gap - section_h

    sections = [
        (copy["storage_title"], copy["storage_body"], "#1565c0", "#eef3ff", top_y),
        (copy["audit_title"], copy["audit_body"], "#c62828", "#ffebee", bottom_y),
    ]
    for title, body, edge, fill_color, sy in sections:
        draw_rounded_box(ax, x + 0.22, sy, w - 0.44, section_h, edge, fill_color, lw=1.6)
        ax.text(
            x + 0.45,
            sy + section_h - 0.26,
            title,
            ha="left",
            va="top",
            fontsize=9.4,
            fontweight="bold",
            color=edge,
        )
        ax.text(
            x + 0.45,
            sy + section_h - 0.65,
            wrap_text(str(body), width=31),
            ha="left",
            va="top",
            fontsize=7.3,
            color=TEXT,
            linespacing=1.35,
        )


def draw_evidence_panel(ax, copy: dict[str, object], x: float, y: float, w: float, h: float):
    draw_rounded_box(ax, x, y, w, h, "#546e7a", "#eceff1", lw=1.55)
    ax.text(
        x + 0.25,
        y + h - 0.28,
        copy["evidence_title"],
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=TEXT,
    )

    evidence = copy["evidence"]
    assert isinstance(evidence, tuple)
    for index, item in enumerate(evidence, start=1):
        col = 0 if index <= 3 else 1
        row = index - 1 if col == 0 else index - 4
        item_x = x + 0.32 + col * (w * 0.47)
        item_y = y + h - 0.78 - row * 0.38
        ax.text(item_x, item_y, f"{index}. {item}", ha="left", va="top", fontsize=7.5, color=TEXT)


def draw_base_app(ax, x: float, top: float, w: float, layer_h: float, gap: float):
    y_positions: list[float] = []
    for idx, layer in enumerate(BASE_LAYERS):
        y = top - (idx + 1) * layer_h - idx * gap
        y_positions.append(y)
        draw_layer(ax, layer, x, y, w, layer_h)
        if idx:
            prev_y = y_positions[idx - 1]
            draw_arrow(ax, (x + w / 2, prev_y), (x + w / 2, y + layer_h))
    return y_positions


def draw_audit_sidebar(ax, x: float, y: float, w: float, h: float):
    draw_rounded_box(ax, x, y, w, h, "#c62828", "#fff5f5", lw=2.2)
    ax.text(
        x + w / 2,
        y + h - 0.36,
        "인증·감사 계층",
        ha="center",
        va="top",
        fontsize=11.0,
        fontweight="bold",
        color="#c62828",
    )
    ax.text(
        x + w / 2,
        y + h - 0.72,
        "Authentication & Audit",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#c62828",
    )
    ax.text(
        x + 0.3,
        y + h - 1.35,
        "· 로그인/세션 보안\n· 권한별 접근 제어\n· Audit trail\n· 이벤트 자동 기록\n· 전자기록 보존",
        ha="left",
        va="top",
        fontsize=8.2,
        color=TEXT,
        linespacing=1.55,
    )
    ax.text(
        x + w / 2,
        y + 0.36,
        "Option별 구현 차이",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color=WARNING,
        fontweight="bold",
    )


def draw_bottom_outputs(ax, x: float, y: float, w: float):
    labels = ("Database", "Raw Spectra", "Model Artifacts", "Reports / Output")
    box_w = (w - 0.45) / 4
    for idx, label in enumerate(labels):
        bx = x + idx * (box_w + 0.15)
        draw_rounded_box(ax, bx, y, box_w, 0.55, "#78909c", "#ffffff", lw=1.0)
        ax.text(
            bx + box_w / 2, y + 0.275, label, ha="center", va="center", fontsize=7.0, color=TEXT
        )


def draw_diagram(option_key: str):
    copy = OPTION_COPY[option_key]
    fig, ax = plt.subplots(figsize=(15.0, 10.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10.5)
    ax.axis("off")

    ax.text(
        7.5,
        10.15,
        copy["title"],
        ha="center",
        va="top",
        fontsize=17.0,
        fontweight="bold",
        color=TEXT,
    )
    ax.text(
        7.5,
        9.72,
        copy["subtitle"],
        ha="center",
        va="top",
        fontsize=9.6,
        color=WARNING,
        fontweight="bold",
    )
    ax.text(
        7.5,
        9.36,
        "동일한 임상 분석 웹앱 구조 위에서 저장 계층과 인증·감사 계층만 옵션별로 바뀜",
        ha="center",
        va="top",
        fontsize=9.3,
        color=MUTED,
    )

    # User and base application layers
    app_x, app_w = 2.45, 6.65
    top = 8.95
    layer_h, gap = 0.86, 0.16

    first_layer_center_y = top - layer_h / 2
    ax.text(
        0.95,
        first_layer_center_y + 0.2,
        "사용자",
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color=TEXT,
    )
    ax.text(
        0.95,
        first_layer_center_y - 0.15,
        "검사실/QA\n담당자",
        ha="center",
        va="top",
        fontsize=8.0,
        color=MUTED,
        linespacing=1.25,
    )
    draw_arrow(ax, (1.55, first_layer_center_y), (app_x, first_layer_center_y))

    y_positions = draw_base_app(ax, app_x, top, app_w, layer_h, gap)

    sidebar_x = 9.55
    sidebar_y = y_positions[-1]
    sidebar_h = top - sidebar_y
    draw_audit_sidebar(ax, sidebar_x, sidebar_y, 1.78, sidebar_h)

    for y in y_positions:
        arrow = FancyArrowPatch(
            (app_x + app_w, y + layer_h / 2),
            (sidebar_x, y + layer_h / 2),
            arrowstyle="<|-|>",
            mutation_scale=10,
            linewidth=1.0,
            color="#90a4ae",
            linestyle="dashed",
        )
        ax.add_patch(arrow)

    draw_option_panel(ax, copy, 11.75, 2.78, 2.95, 5.95)

    output_y = 2.18
    draw_bottom_outputs(ax, app_x, output_y, app_w)
    storage_center = (app_x + app_w / 2, y_positions[4])
    report_center = (app_x + app_w - 0.65, y_positions[5])
    draw_arrow(
        ax, storage_center, (app_x + app_w / 2, output_y + 0.55), color="#78909c", dashed=True
    )
    draw_arrow(
        ax, report_center, (app_x + app_w - 0.65, output_y + 0.55), color="#78909c", dashed=True
    )

    bottom_y, bottom_h = 0.35, 1.72
    draw_evidence_panel(ax, copy, 0.45, bottom_y, 7.55, bottom_h)
    draw_rounded_box(ax, 8.25, bottom_y, 6.3, bottom_h, str(copy["risk_color"]), "#fffdf7", lw=1.8)
    ax.text(
        8.55,
        bottom_y + bottom_h - 0.28,
        "인증 관점 핵심 메시지",
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=str(copy["risk_color"]),
    )
    ax.text(
        8.55,
        bottom_y + bottom_h - 0.74,
        wrap_text(str(copy["risk"]), width=58),
        ha="left",
        va="top",
        fontsize=8.7,
        color=TEXT,
        linespacing=1.35,
    )
    ax.text(
        8.55,
        bottom_y + 0.42,
        "공통: 화면/업무흐름/QC/AI/보고서 로직은 동일하게 검증하고, 저장·감사증적의 구현 책임만 옵션별로 구분한다.",
        ha="left",
        va="top",
        fontsize=8.0,
        color=MUTED,
    )

    fig.tight_layout(pad=0.5)
    fig.savefig(OUT_DIR / str(copy["output"]), dpi=170)
    plt.close(fig)


def main():
    for option_key in ("a", "b", "c"):
        draw_diagram(option_key)
    outputs = [OPTION_COPY[key]["output"] for key in ("a", "b", "c")]
    print("Saved:", ", ".join(str(OUT_DIR / str(output)) for output in outputs))


if __name__ == "__main__":
    main()
