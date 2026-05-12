#!/usr/bin/env python3
"""Generate the Monthly Meeting PPT (April 2026 update).

Usage:
    python scripts/generate_monthly_ppt.py
Output:
    monthly_meeting_2026_04.pptx in the repo root.
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.chart import XL_CHART_TYPE
import os

# ── Colour palette ──────────────────────────────────────────────
SOLUM_BLUE = RGBColor(0x00, 0x4E, 0x8C)
SOLUM_LIGHT = RGBColor(0x00, 0x7A, 0xCC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK = RGBColor(0x33, 0x33, 0x33)
GREY = RGBColor(0x66, 0x66, 0x66)
GREEN = RGBColor(0x00, 0x80, 0x40)
RED = RGBColor(0xCC, 0x00, 0x00)
AMBER = RGBColor(0xFF, 0x99, 0x00)
BG_LIGHT = RGBColor(0xF5, 0xF7, 0xFA)
TABLE_HEADER_BG = RGBColor(0x00, 0x4E, 0x8C)
TABLE_ALT_BG = RGBColor(0xE8, 0xF0, 0xF8)

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "monthly_meeting_2026_04.pptx")


# ── helpers ─────────────────────────────────────────────────────
def _set_slide_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def _add_title_bar(slide, text):
    """Blue banner at top with white title text."""
    from pptx.util import Inches, Pt
    shp = slide.shapes.add_shape(
        1,  # MSO_SHAPE.RECTANGLE
        Inches(0), Inches(0), Inches(13.333), Inches(0.9),
    )
    shp.fill.solid()
    shp.fill.fore_color.rgb = SOLUM_BLUE
    shp.line.fill.background()
    tf = shp.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.LEFT
    tf.margin_left = Inches(0.5)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE


def _add_textbox(slide, left, top, width, height, text, font_size=12, bold=False, color=DARK, alignment=PP_ALIGN.LEFT):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.alignment = alignment
    return tf


def _add_bullet_list(slide, left, top, width, height, items, font_size=11):
    """Add a text box with bullet points."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, (icon, text, clr) in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = f"{icon} {text}"
        p.font.size = Pt(font_size)
        p.font.color.rgb = clr
        p.space_after = Pt(3)
    return tf


def _add_table(slide, left, top, width, rows_data, col_widths=None):
    """rows_data: list of lists (first row = header)."""
    n_rows = len(rows_data)
    n_cols = len(rows_data[0])
    table_shape = slide.shapes.add_table(n_rows, n_cols, left, top, width, Inches(0.35 * n_rows))
    table = table_shape.table

    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = w

    for r, row in enumerate(rows_data):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = str(val)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(10)
                if r == 0:
                    paragraph.font.bold = True
                    paragraph.font.color.rgb = WHITE
                else:
                    paragraph.font.color.rgb = DARK
            if r == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = TABLE_HEADER_BG
            elif r % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = TABLE_ALT_BG
    return table


def _add_progress_bar(slide, left, top, width, height, pct, label, color=SOLUM_LIGHT):
    """Horizontal progress bar with label."""
    # background
    bg = slide.shapes.add_shape(1, left, top, width, height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = RGBColor(0xDD, 0xDD, 0xDD)
    bg.line.fill.background()
    # fill
    fill_w = int(width * pct / 100)
    if fill_w > 0:
        fg = slide.shapes.add_shape(1, left, top, fill_w, height)
        fg.fill.solid()
        fg.fill.fore_color.rgb = color
        fg.line.fill.background()
    # label
    _add_textbox(slide, left, top - Inches(0.22), width, Inches(0.22),
                 f"{label}  {pct}%", font_size=9, bold=True, color=DARK)


# ── Slide builders ──────────────────────────────────────────────

def slide1_kpi(prs):
    """Slide 1: KPI Dashboard."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    _set_slide_bg(slide, BG_LIGHT)
    _add_title_bar(slide, "SERS Cancer Screening — KPI Dashboard (April 2026)")

    # KPI table
    kpi_data = [
        ["KPI", "Previous", "Current", "Change"],
        ["AI-1 Screening", "AUC 0.958\nSens 86.3% / Spec 92.3%",
         "AUC 0.9936\nSens 96.9~99.2% / Spec 95~100%", "Stacking V2 ensemble"],
        ["AI-2 Cancer Typing", "Macro AUC 0.904", "F1 0.9252 ± 0.022", "10-base model stacking"],
        ["QC", "3/4 equip PASS", "PASS + QC threshold derived\n(corr 0.925, min_reps 4)", "Phase 7 data-driven"],
        ["DATA", "N = 1,342", "N = 1,342 + BLC ongoing", "Bladder cancer cohort"],
        ["MB", "GC-MS/MS 600 design", "1,594 peak-metabolite matches", "Thermo DB 4,554 entries"],
        ["SW", "KRW 28M proposal", "Win .exe + FastAPI webapp\n+ Audit trail", "Deployment infra done"],
        ["IP", "25%", "11 patent drafts + FTO done", "Medium-to-low risk"],
        ["PUB", "0%", "AACR poster 27 figures done", "Publication-ready"],
    ]
    _add_table(slide, Inches(0.4), Inches(1.1), Inches(12.5), kpi_data,
               col_widths=[Inches(1.8), Inches(3.2), Inches(4.0), Inches(3.5)])

    # Progress bars
    tracks = [
        ("AI", 60), ("QC", 70), ("MB", 40), ("SW", 45), ("IP", 60), ("PUB", 30),
    ]
    bar_top = Inches(5.3)
    bar_w = Inches(1.6)
    bar_h = Inches(0.2)
    for i, (label, pct) in enumerate(tracks):
        _add_progress_bar(slide, Inches(0.5 + i * 2.1), bar_top, bar_w, bar_h, pct, label)

    _add_textbox(slide, Inches(0.4), Inches(5.0), Inches(5), Inches(0.3),
                 "Track Progress", font_size=14, bold=True, color=SOLUM_BLUE)


def slide2_tracks(prs):
    """Slide 2: 6-Track Detailed View."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, "6-Track Detailed Progress")

    tracks = [
        ("AI Track (35% → 60%)", [
            ("✅", "Stacking Ensemble V2 (10-base + ElasticNet meta)", GREEN),
            ("✅", "3 Operating Modes: Screening/Balanced/Confirmatory", GREEN),
            ("✅", "Feature Selection + Grad-CAM pipeline", GREEN),
            ("✅", "Multiview (raw+deriv+peak) + attention fusion", GREEN),
            ("⚠️", "Hospital confound: cross-hosp AUC −23.7pp", AMBER),
            ("🔜", "ComBat / DANN domain adaptation needed", GREY),
        ]),
        ("QC Track (30% → 70%)", [
            ("✅", "Phase 7: QC threshold data-driven derivation", GREEN),
            ("✅", "Phase 8: Hospital confound quantification", GREEN),
            ("✅", "Phase 9-10: Hospital-stratified training", GREEN),
            ("✅", "Two-stage QC system established", GREEN),
        ]),
        ("MB Track (20% → 40%)", [
            ("✅", "1,594 peak-metabolite cross-reference", GREEN),
            ("✅", "Thermo DB 4,554 entries", GREEN),
            ("✅", "Cancer-type metabolite signatures (563)", GREEN),
            ("🔜", "GC-MS/MS Phase 1 validation", GREY),
        ]),
        ("SW Track (15% → 45%)", [
            ("✅", "Windows .exe build (PyInstaller)", GREEN),
            ("✅", "FastAPI webapp + Audit trail + i18n", GREEN),
            ("✅", "CI/CD (GitHub Actions, ruff, pytest)", GREEN),
            ("🔜", "V&V documentation + external review", GREY),
        ]),
        ("IP Track (25% → 60%)", [
            ("✅", "11 patent drafts (PATENT_01~11)", GREEN),
            ("✅", "FTO analysis complete (medium-low risk)", GREEN),
            ("🔜", "External attorney review + filing", GREY),
        ]),
        ("PUB Track (0% → 30%)", [
            ("✅", "AACR poster 27 figures complete", GREEN),
            ("✅", "R scripts + demographics tables v5-v8", GREEN),
            ("🔜", "May clinical paper submission prep", GREY),
        ]),
    ]

    # 3 columns × 2 rows layout
    col_w = Inches(4.1)
    for i, (title, items) in enumerate(tracks):
        col = i % 3
        row = i // 3
        left = Inches(0.3) + col * Inches(4.3)
        top_title = Inches(1.1) + row * Inches(2.7)
        _add_textbox(slide, left, top_title, col_w, Inches(0.3), title,
                     font_size=13, bold=True, color=SOLUM_BLUE)
        _add_bullet_list(slide, left, top_title + Inches(0.32), col_w, Inches(2.3),
                         items, font_size=9)


def slide3_cohort_qc(prs):
    """Slide 3: Cohort & Equipment QC."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, BG_LIGHT)
    _add_title_bar(slide, "Cohort & Equipment QC")

    # Cohort table
    _add_textbox(slide, Inches(0.4), Inches(1.1), Inches(5), Inches(0.3),
                 "Cohort Overview", font_size=14, bold=True, color=SOLUM_BLUE)
    cohort_data = [
        ["Cohort", "Hospital", "Type", "Status"],
        ["NOR / CRC / LUN / etc.", "National Cancer Center", "Retrospective", "Active (N=1,342)"],
        ["YPAN / YNOR", "Severance Hospital", "Retrospective", "Active"],
        ["BPRO / BNOR", "Boramae Hospital", "Prospective", "Configured"],
        ["BLC", "TBD", "Bladder Cancer", "NEW — collection ongoing"],
    ]
    _add_table(slide, Inches(0.4), Inches(1.5), Inches(8), cohort_data,
               col_widths=[Inches(2.5), Inches(2.2), Inches(1.8), Inches(2.5)])

    # QC Thresholds
    _add_textbox(slide, Inches(0.4), Inches(3.7), Inches(6), Inches(0.3),
                 "QC Threshold Derivation (Phase 7 — Data-Driven)", font_size=14, bold=True, color=SOLUM_BLUE)
    qc_items = [
        ("✅", "per_spectrum_corr threshold: 0.925 (NOR 5th percentile)", GREEN),
        ("✅", "min_reps_after_qc: 4", GREEN),
        ("✅", "cosmic_isolation: 2.0, cosmic_height: 0.3", GREEN),
        ("✅", "saturation_plateau: 5", GREEN),
        ("✅", "Derived from NOR+YNOR healthy controls (no data leakage)", GREEN),
    ]
    _add_bullet_list(slide, Inches(0.4), Inches(4.1), Inches(8), Inches(1.8), qc_items, font_size=11)

    # Equipment QC (right side)
    _add_textbox(slide, Inches(8.8), Inches(1.1), Inches(4), Inches(0.3),
                 "Equipment QC", font_size=14, bold=True, color=SOLUM_BLUE)
    eq_data = [
        ["Equipment", "Status"],
        ["Thermo DXR3 #1", "PASS"],
        ["Thermo DXR3 #2", "PASS"],
        ["Thermo DXR3 #3", "PASS"],
        ["Medical (portable)", "Under review"],
    ]
    _add_table(slide, Inches(8.8), Inches(1.5), Inches(4), eq_data,
               col_widths=[Inches(2.5), Inches(1.5)])


def slide4_ai_performance(prs):
    """Slide 4: AI Performance (major update)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, "AI Performance — Stacking V2 Results")

    # Stage 1
    _add_textbox(slide, Inches(0.4), Inches(1.1), Inches(6), Inches(0.3),
                 "Stage 1: Cancer Detection (Screening)", font_size=14, bold=True, color=SOLUM_BLUE)
    s1_data = [
        ["Metric", "Previous", "Current (Stacking V2)"],
        ["AUC", "0.9582", "0.9936 ± 0.0017"],
        ["Screening (τ=0.838)", "—", "Sens 96.9% / Spec 100%"],
        ["Balanced (τ=0.444)", "Sens 86.3% / Spec 92.3%", "Sens 98.5% / Spec 99.3%"],
        ["Confirmatory (τ=0.061)", "—", "Sens 99.2% / Spec 95%"],
    ]
    _add_table(slide, Inches(0.4), Inches(1.5), Inches(6.2), s1_data,
               col_widths=[Inches(2.0), Inches(2.0), Inches(2.2)])

    # Stage 2
    _add_textbox(slide, Inches(7.0), Inches(1.1), Inches(6), Inches(0.3),
                 "Stage 2: Cancer Typing", font_size=14, bold=True, color=SOLUM_BLUE)
    s2_data = [
        ["Metric", "Previous", "Current"],
        ["Macro AUC", "0.904", "—"],
        ["F1 Score", "—", "0.9252 ± 0.0219"],
        ["Strong (CRC,LUN,SPAN,PRO)", "Good", ">0.97 AUC (within-hosp)"],
        ["Weak (BRE, CPAN)", "Low", "Hospital confound identified"],
    ]
    _add_table(slide, Inches(7.0), Inches(1.5), Inches(5.9), s2_data,
               col_widths=[Inches(2.2), Inches(1.6), Inches(2.1)])

    # Hospital Confound warning box
    _add_textbox(slide, Inches(0.4), Inches(3.9), Inches(12), Inches(0.3),
                 "⚠️  Critical Finding: Hospital Confound Effect", font_size=14, bold=True, color=RED)

    confound_items = [
        ("⚠️", "Naive CV AUC: 0.9628  →  Leave-Severance-out: 0.7356 (−23.7pp drop)", RED),
        ("⚠️", "Within-Severance CV: 0.6922 (−27pp)", RED),
        ("✅", "Within-hospital cancer discrimination: 0.9975 (biology is real)", GREEN),
        ("→", "Conclusion: Part of performance inflated by hospital/protocol batch effect", AMBER),
        ("→", "Action: ComBat / DANN domain adaptation required", AMBER),
    ]
    _add_bullet_list(slide, Inches(0.4), Inches(4.3), Inches(12), Inches(2.5), confound_items, font_size=11)


def slide5_roadmap(prs):
    """Slide 5: ROC & Improvement Roadmap."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, BG_LIGHT)
    _add_title_bar(slide, "Improvement Roadmap — Progress & Next Steps")

    # Completed roadmap items
    _add_textbox(slide, Inches(0.4), Inches(1.1), Inches(6), Inches(0.3),
                 "Completed (Original 3-Step Roadmap)", font_size=14, bold=True, color=SOLUM_BLUE)
    done_items = [
        ("✅", "QC / Batch Correction → Phase 7-10 complete", GREEN),
        ("⚠️", "Hospital confound identified → ComBat/DANN needed", AMBER),
        ("✅", "Peak filtering / redesign → Multiview experiment done", GREEN),
        ("✅", "SHAP-based feature importance analysis complete", GREEN),
        ("✅", "Clinical multimodal fusion → Early fusion experiments archived", GREEN),
    ]
    _add_bullet_list(slide, Inches(0.4), Inches(1.5), Inches(6), Inches(2.0), done_items, font_size=11)

    # New roadmap
    _add_textbox(slide, Inches(7.0), Inches(1.1), Inches(6), Inches(0.3),
                 "New Roadmap Items", font_size=14, bold=True, color=SOLUM_BLUE)

    new_data = [
        ["Priority", "Item", "Description"],
        ["P0", "Domain Adaptation", "ComBat (batch) or DANN (adversarial)\nfor hospital confound correction"],
        ["P1", "Multi-site Validation", "Cross-instrument calibration (PDS)\nThermo ↔ Medical verification"],
        ["P2", "Prospective Integration", "BPRO/BNOR (Boramae)\nYPAN/YNOR (Severance) data"],
    ]
    _add_table(slide, Inches(7.0), Inches(1.5), Inches(5.9), new_data,
               col_widths=[Inches(0.7), Inches(1.8), Inches(3.4)])

    # Timeline visual
    _add_textbox(slide, Inches(0.4), Inches(3.9), Inches(12), Inches(0.3),
                 "Timeline", font_size=14, bold=True, color=SOLUM_BLUE)

    timeline_items = [
        ("Apr", "Hospital confound analysis complete. ComBat/DANN design.", SOLUM_LIGHT),
        ("May", "Domain adaptation experiments. AACR poster submit. Clinical paper prep.", SOLUM_LIGHT),
        ("Jun", "Multi-site validation Phase 1. GC-MS/MS validation kickoff.", SOLUM_LIGHT),
        ("Jul-Aug", "Prospective cohort integration. Patent external review.", SOLUM_LIGHT),
    ]
    for i, (month, desc, clr) in enumerate(timeline_items):
        left = Inches(0.4)
        top = Inches(4.3) + i * Inches(0.4)
        _add_textbox(slide, left, top, Inches(1.0), Inches(0.35),
                     month, font_size=11, bold=True, color=SOLUM_BLUE)
        _add_textbox(slide, left + Inches(1.1), top, Inches(11), Inches(0.35),
                     desc, font_size=10, color=DARK)


def slide6_mb_sw(prs):
    """Slide 6: MB & SW GMP + This Month Focus."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, WHITE)
    _add_title_bar(slide, "MB Validation & SW GMP — Status & Next Steps")

    # MB section
    _add_textbox(slide, Inches(0.4), Inches(1.1), Inches(6), Inches(0.3),
                 "MB — GC-MS/MS Validation", font_size=14, bold=True, color=SOLUM_BLUE)
    mb_data = [
        ["Phase", "Previous", "Current"],
        ["Design", "GC-MS/MS 600 design", "Complete"],
        ["Peak-Metabolite DB", "—", "1,594 matches (DB: 4,554)"],
        ["Cancer Signatures", "—", "563 summary entries identified"],
        ["Validation Phase 1", "—", "Ready to start (n=200)"],
    ]
    _add_table(slide, Inches(0.4), Inches(1.5), Inches(6), mb_data,
               col_widths=[Inches(1.8), Inches(2.0), Inches(2.2)])

    # SW section
    _add_textbox(slide, Inches(7.0), Inches(1.1), Inches(6), Inches(0.3),
                 "SW GMP — Deployment Infrastructure", font_size=14, bold=True, color=SOLUM_BLUE)
    sw_items = [
        ("✅", "Production engine: sers_predict.py (49KB)", GREEN),
        ("✅", "Clinical webapp: FastAPI + Starlette templates", GREEN),
        ("✅", "Audit trail + i18n (KR/EN) + SQLite DB", GREEN),
        ("✅", "Windows .exe builder (PyInstaller)", GREEN),
        ("✅", "CI/CD: GitHub Actions (Py 3.10/3.11/3.12)", GREEN),
        ("🔜", "V&V documentation + external review", GREY),
    ]
    _add_bullet_list(slide, Inches(7.0), Inches(1.5), Inches(5.5), Inches(2.5), sw_items, font_size=11)

    # This Month Focus
    _add_textbox(slide, Inches(0.4), Inches(4.2), Inches(12), Inches(0.3),
                 "This Month Focus (April 2026)", font_size=14, bold=True, color=SOLUM_BLUE)

    focus_items = [
        ("1.", "Hospital Confound ComBat/DANN correction — start experiments", DARK),
        ("2.", "Multi-site validation design", DARK),
        ("3.", "GC-MS/MS validation Phase 1 kickoff", DARK),
        ("4.", "AACR poster submission + May paper preparation", DARK),
        ("5.", "Patent external attorney review — begin", DARK),
    ]
    _add_bullet_list(slide, Inches(0.4), Inches(4.6), Inches(12), Inches(2.0), focus_items, font_size=12)


# ── Main ────────────────────────────────────────────────────────
def main():
    prs = Presentation()
    # Widescreen 16:9
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slide1_kpi(prs)
    slide2_tracks(prs)
    slide3_cohort_qc(prs)
    slide4_ai_performance(prs)
    slide5_roadmap(prs)
    slide6_mb_sw(prs)

    out = os.path.abspath(OUT_PATH)
    prs.save(out)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
