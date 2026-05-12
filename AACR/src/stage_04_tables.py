"""Stage analysis - Step 4: LaTeX tables rendered as PNG images.

Three tables:
  T1. Cancer x Stage group crosstab (patient counts)
  T2. Early vs Advanced demographics per cancer (age, sex, BMI)
  T3. Stage grouping definition / rule sheet

Compiled via pdflatex, then converted to PNG via pdf2image.
Output: AACR/figures/table_T{1,2,3}_*.{pdf,png}
"""
from __future__ import annotations
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import fitz  # PyMuPDF

ROOT = Path("/home/user/SERS-AI")
CLI_UNIQ = ROOT / "AACR/data/stage_clinical_unique.csv"
OUT_DIR = ROOT / "AACR/figures"
BUILD_DIR = ROOT / "AACR/data/_latex_build"
BUILD_DIR.mkdir(parents=True, exist_ok=True)

CANCERS = ["CRC", "CPAN", "PRO", "OVA", "LUN"]
CANCER_DISPLAY = {
    "CRC": "Colorectal (CRC)",
    "CPAN": "Pancreatic (PAC)",
    "PRO": "Prostate (PRC)",
    "OVA": "Ovarian (OVC)",
    "LUN": "Lung (LC)",
}

# ---------- LaTeX preamble (shared) ----------
PREAMBLE = r"""
\documentclass[12pt]{article}
\usepackage[a4paper,landscape,margin=12mm]{geometry}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage[T1]{fontenc}
\definecolor{headblue}{HTML}{1E3A8A}
\definecolor{rowa}{HTML}{EFF6FF}
\definecolor{rowb}{HTML}{FFFFFF}
\definecolor{textmain}{HTML}{111827}
\definecolor{textmute}{HTML}{4B5563}
\pagestyle{empty}
\renewcommand{\arraystretch}{1.55}
\setlength{\arrayrulewidth}{0.6pt}
"""

def compile_latex(tex_source: str, name: str) -> Path:
    """Compile a standalone LaTeX file. Return generated PDF path."""
    tex_path = BUILD_DIR / f"{name}.tex"
    tex_path.write_text(tex_source, encoding="utf-8")
    for _ in range(2):
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode",
             "-output-directory", str(BUILD_DIR), str(tex_path)],
            capture_output=True, text=True,
        )
    pdf_path = BUILD_DIR / f"{name}.pdf"
    if not pdf_path.exists():
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise RuntimeError(f"pdflatex failed for {name}")
    return pdf_path


def pdf_to_png(pdf_path: Path, out_png: Path, dpi: int = 240) -> None:
    doc = fitz.open(str(pdf_path))
    page = doc.load_page(0)
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    pix.save(str(out_png))
    doc.close()
    print(f"saved: {out_png}")


# ============================================================
# T1. Cancer x Stage crosstab
# ============================================================
def build_t1(cli: pd.DataFrame) -> str:
    ct = cli.groupby(["group", "cancer_stage_group"]).size().unstack(fill_value=0)
    ct = ct.reindex(CANCERS).fillna(0).astype(int)
    for s in ["early", "advanced", "unknown"]:
        if s not in ct.columns:
            ct[s] = 0
    ct = ct[["early", "advanced", "unknown"]]
    ct["Total"] = ct.sum(axis=1)

    rows = []
    for c in CANCERS:
        r = ct.loc[c]
        rows.append(f"{CANCER_DISPLAY[c]} & {r['early']} & {r['advanced']} & "
                    f"{r['unknown']} & \\textbf{{{r['Total']}}} \\\\")
    tot = ct.sum(axis=0)
    rows.append(r"\midrule")
    rows.append(f"\\textbf{{Total}} & \\textbf{{{tot['early']}}} & "
                f"\\textbf{{{tot['advanced']}}} & \\textbf{{{tot['unknown']}}} & "
                f"\\textbf{{{tot['Total']}}} \\\\")
    body = "\n".join(rows)

    return PREAMBLE + r"""
\begin{document}
\vspace*{4mm}
{\LARGE\bfseries\color{headblue} Table 1. Patient Counts by Cancer Type \& Stage Group}\\[2mm]
{\large\color{textmute} Source: \texttt{cancer\_stage\_group} column (clinical\_unified\_202604172056.csv), n unique patients}\\[4mm]
{\Large
\begin{tabular}{>{\raggedright\arraybackslash}p{6.5cm} r r r r}
\toprule
\rowcolor{headblue}
\textcolor{white}{\textbf{Cancer Type}} &
\textcolor{white}{\textbf{Early (I--II)}} &
\textcolor{white}{\textbf{Advanced (III--IV)}} &
\textcolor{white}{\textbf{Unknown / X}} &
\textcolor{white}{\textbf{Total}} \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
}
\\[6mm]
{\small\color{textmute}
Notes: ``Unknown'' includes samples with TX/NX/MX or missing AJCC stage.}
\end{document}
"""


# ============================================================
# T2. Demographics per cancer x stage
# ============================================================
def fmt_mean_sd(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return "--"
    return f"{s.mean():.1f} $\\pm$ {s.std():.1f}"


def fmt_sex(series: pd.Series) -> str:
    s = series.dropna().astype(str).str.upper().str.strip()
    if len(s) == 0:
        return "--"
    m = (s == "M").sum()
    f = (s == "F").sum()
    return f"M {m} / F {f}"


def build_t2(cli: pd.DataFrame) -> str:
    rows = []
    for c in CANCERS:
        sub_c = cli[cli["group"] == c]
        for stage, label in [("early", "Early"), ("advanced", "Advanced")]:
            s = sub_c[sub_c["cancer_stage_group"] == stage]
            n = len(s)
            age = fmt_mean_sd(s["age"])
            sex = fmt_sex(s["sex"])
            bmi = fmt_mean_sd(s["bmi"])
            cancer_label = CANCER_DISPLAY[c] if stage == "early" else ""
            rows.append(f"{cancer_label} & {label} & {n} & {age} & {sex} & {bmi} \\\\")
        rows.append(r"\midrule")
    # drop trailing midrule
    if rows[-1].strip() == r"\midrule":
        rows.pop()
    body = "\n".join(rows)

    return PREAMBLE + r"""
\begin{document}
\vspace*{4mm}
{\LARGE\bfseries\color{headblue} Table 2. Demographics: Early vs Advanced}\\[2mm]
{\large\color{textmute} Values: mean $\pm$ SD for continuous, counts for sex. Unknown stage excluded.}\\[4mm]
{\Large
\begin{tabular}{>{\raggedright\arraybackslash}p{5.5cm} l r l l l}
\toprule
\rowcolor{headblue}
\textcolor{white}{\textbf{Cancer Type}} &
\textcolor{white}{\textbf{Stage}} &
\textcolor{white}{\textbf{n}} &
\textcolor{white}{\textbf{Age (yr)}} &
\textcolor{white}{\textbf{Sex}} &
\textcolor{white}{\textbf{BMI (kg/m$^2$)}} \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
}
\\[5mm]
{\small\color{textmute}
``--'' indicates missing / not computable.}
\end{document}
"""


# ============================================================
# T3. Stage grouping rule sheet
# ============================================================
def build_t3() -> str:
    return PREAMBLE + r"""
\begin{document}
\vspace*{4mm}
{\LARGE\bfseries\color{headblue} Table 3. Stage Grouping Definition}\\[2mm]
{\large\color{textmute} Applied to clinical\_unified\_202604172056.csv column \texttt{cancer\_stage\_group}.}\\[4mm]
{\Large
\begin{tabular}{>{\raggedright\arraybackslash}p{4.5cm} >{\raggedright\arraybackslash}p{11cm} >{\raggedright\arraybackslash}p{8.5cm}}
\toprule
\rowcolor{headblue}
\textcolor{white}{\textbf{Group}} &
\textcolor{white}{\textbf{AJCC / FIGO Stage Included}} &
\textcolor{white}{\textbf{Interpretation}} \\
\midrule
\rowcolor{rowa}
\textbf{Early} & Stage I, IA, IA1, IA2, IB, IIA, IIB  & Localized / regional disease,
potentially curative resection. \\
\textbf{Advanced} & Stage IIIA, IIIB, IIIC, IVA, IVB  & Locally advanced or distant
metastatic disease; systemic therapy indicated. \\
\rowcolor{rowa}
\textbf{Unknown} & TX / NX / MX present, or stage field blank  & Insufficient staging data;
excluded from Early vs Advanced comparison. \\
\midrule
\multicolumn{3}{l}{\textbf{\color{headblue} Pre-assignment rules (from staging\_mapping\_log.txt, AJCC 8th ed.)}}\\
\midrule
Pancreatic (PAC) & T1\,N0\,M0 $\rightarrow$ IA; T2\,N0\,M0 $\rightarrow$ IB; T3\,N0\,M0 $\rightarrow$ IIA; T$\leq$3\,N1\,M0 $\rightarrow$ IIB; T4 \textit{or} N2 $\rightarrow$ III; M1 $\rightarrow$ IV. & T4 or N2 = Stage III; any M1 = Stage IV. \\
\rowcolor{rowa}
Colorectal (CRC) & TNM mapped via AJCC 8th. CRC 271--300 labeled Advanced (user-assigned). & 1--270 Early, 271--300 Advanced. \\
Prostate (PRC) & pT2\,N0\,M0 / cT$\leq$2 as Early; pT3a+/N1+ as Advanced; M1 $\rightarrow$ IV. & PSA not used here. \\
\rowcolor{rowa}
Lung (LC) & cancer\_stage normalized (``IA1'', ``stage 1A'' $\rightarrow$ IA, etc.); IA/IB/IIA/IIB $\rightarrow$ Early; III+/IV $\rightarrow$ Advanced. & Original field contained formatting variations. \\
Ovarian (OVC) & FIGO/TNM from tnm\_stage; T1--T2 localized $\rightarrow$ Early; T3 \textit{or} M1 $\rightarrow$ Advanced; TX/NX frequent $\rightarrow$ Unknown. & 57/70 Unknown in this extract. \\
\bottomrule
\end{tabular}
}
\\[5mm]
{\small\color{textmute}
Reference: AJCC Cancer Staging Manual, 8th Edition; FIGO 2014 for ovarian.}
\end{document}
"""


def main() -> None:
    cli = pd.read_csv(CLI_UNIQ)

    t1_tex = build_t1(cli)
    t2_tex = build_t2(cli)
    t3_tex = build_t3()

    t1_pdf = compile_latex(t1_tex, "table_T1_stage_crosstab")
    t2_pdf = compile_latex(t2_tex, "table_T2_demographics")
    t3_pdf = compile_latex(t3_tex, "table_T3_definition")

    pdf_to_png(t1_pdf, OUT_DIR / "table_T1_stage_crosstab.png")
    pdf_to_png(t2_pdf, OUT_DIR / "table_T2_demographics.png")
    pdf_to_png(t3_pdf, OUT_DIR / "table_T3_definition.png")
    # also copy PDFs
    for src in [t1_pdf, t2_pdf, t3_pdf]:
        dst = OUT_DIR / src.name
        dst.write_bytes(src.read_bytes())


if __name__ == "__main__":
    main()
