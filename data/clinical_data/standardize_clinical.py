"""Standardize all clinical data into a unified format.

Reads raw Excel files for each disease group, maps columns to a standard
schema, normalizes encodings, and outputs a single merged CSV + per-disease CSVs.

Usage:
    python standardize_clinical.py
    python standardize_clinical.py --output-dir ./standardized
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Standard output columns ───────────────────────────────────────────────
STD_COLUMNS = [
    "patient_id", "disease_group", "source_file",
    # Demographics
    "age", "sex", "height_cm", "weight_kg", "bmi",
    "bp_systolic", "bp_diastolic",
    # Lifestyle
    "smoking_status", "drinking_status",
    # History
    "past_history",
    # Diagnosis (cancer)
    "diagnosis", "diagnosis_date", "sample_date", "surgery_date",
    "pathology", "stage", "tnm", "t_stage", "n_stage", "m_stage",
    "metastasis", "treatment", "fasting",
    "surgery_name", "chemo_date", "treatment_detail", "sample_timing",
    # CBC
    "wbc", "rbc", "hb", "hct", "platelet", "neutrophil_pct", "lymphocyte_pct",
    # Chemistry
    "ast", "alt", "alp", "ggt", "bun", "creatinine", "uric_acid",
    "glucose", "total_protein", "albumin", "total_bilirubin", "ldh",
    "calcium", "hs_crp",
    # Electrolytes
    "sodium", "potassium", "chloride",
    # Lipid
    "total_cholesterol", "triglyceride", "hdl_c", "ldl_c",
    # Diabetes
    "hba1c",
    # Tumor markers
    "afp", "cea", "ca19_9", "psa",
    # Urinalysis
    "ua_sg", "ua_ph", "ua_protein", "ua_glucose", "ua_blood",
]


# ── Normalization helpers ─────────────────────────────────────────────────
def normalize_sex(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s in ("M", "m", "남", "남자", "male", "Male"):
        return "M"
    if s in ("F", "f", "여", "여자", "female", "Female"):
        return "F"
    return s


def normalize_smoking(val):
    """Convert to 0=never, 1=former, 2=current."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s or s in (".", "모름"):
        return None
    if s in ("0", "X", "x", "비흡연", "1", "안함"):
        if s == "1":  # 방광암 encoding: 1=비흡연
            return 0
        return 0
    if s in ("2",):
        return 1  # 방광암: 2=과거흡연
    if s in ("3",):
        return 2  # 방광암: 3=현재흡연
    if "전혀 피운 적" in s or "없다" in s:
        return 0
    if "지금은 피우지 않" in s or "끊" in s or "과거" in s:
        return 1
    if "피운다" in s or "현재" in s:
        return 2
    s_lower = s.lower()
    # Ex-smoker patterns (English)
    if "ex" in s_lower or "quit" in s_lower or "stop" in s_lower or "중단" in s:
        return 1
    # Pack-year descriptions (currently smoking): "0.5갑/일 20Y", "1갑/일 30Y"
    if "갑" in s or "개피" in s or re.match(r".*\d+PY", s, re.IGNORECASE):
        return 2
    # "O" = 있음 (has smoked) → assume current
    if s == "O":
        return 2
    # "함" = does it
    if s.startswith("함"):
        return 2
    # Numeric pack-years style
    try:
        v = float(s)
        return 2 if v > 0 else 0
    except ValueError:
        pass
    return None  # unrecognized → NULL instead of preserving raw


def normalize_drinking(val):
    """Convert to 0=never, 1=former, 2=current."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s or s in (".", "모름"):
        return None
    if s in ("0", "X", "x", "비음주", "안함"):
        return 0
    if s in ("1",):  # 방광암: 1=비음주
        return 0
    if s in ("2",):
        return 1
    if s in ("3",):
        return 2
    if "안 마" in s or "못 마시" in s or "처음부터" in s or "없" in s:
        return 0
    if "마신다" in s or "현재" in s:
        return 2
    # "O" = 있음 (drinks) → current
    if s == "O":
        return 2
    # "함" = does it → current
    if s.startswith("함"):
        return 2
    # Detailed descriptions: "소주 2병/월 40Y", "맥주 8병/월 40Y" → current drinker
    if "소주" in s or "맥주" in s or "병/월" in s or "alcohol" in s.lower():
        return 2
    # Date strings → NULL (data entry error)
    if re.match(r"\d{4}-\d{2}-\d{2}", s):
        return None
    try:
        v = float(s)
        return 2 if v > 0 else 0
    except ValueError:
        pass
    return None  # unrecognized → NULL


def normalize_date(val):
    if pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp):
        return val.strftime("%Y-%m-%d")
    s = str(val).strip()
    # 202408 -> 2024-08-01
    if re.match(r"^\d{6}$", s):
        return f"{s[:4]}-{s[4:6]}-01"
    # 20210311 -> 2021-03-11
    if re.match(r"^\d{8}$", s):
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    # Already date-like
    if re.match(r"\d{4}-\d{2}-\d{2}", s):
        return s[:10]
    # datetime string
    try:
        dt = pd.to_datetime(val)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return s


def safe_float(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s in ("", "-", " "):
        return None
    # Handle ">60", "<1.6" etc
    s = re.sub(r"^[<>]", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def parse_blood_test_text(text):
    """Parse free-text blood test results into structured values.

    Handles formats like:
      "CEA 2.25", "CEA: 77.7", "CEA 0.926\\nCA19-9 8.26", "PSA 4.5"
    Returns dict with keys: cea, ca19_9, psa, afp (float or None).
    """
    result = {"cea": None, "ca19_9": None, "psa": None, "afp": None}
    if pd.isna(text):
        return result
    s = str(text).strip()
    if not s or s in ("안함", ".", "-"):
        return result

    patterns = [
        (r"CEA\s*[:\s]\s*([\d.]+)", "cea"),
        (r"CA\s*19[-.]?9\s*[:\s]\s*([\d.]+)", "ca19_9"),
        (r"PSA\s*[:\s]\s*([\d.]+)", "psa"),
        (r"AFP\s*[:\s]\s*([\d.]+)", "afp"),
    ]
    for pat, key in patterns:
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            try:
                result[key] = float(m.group(1))
            except ValueError:
                pass
    return result


def get_col(df, candidates):
    """Find first matching column name."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ── Per-disease loaders ───────────────────────────────────────────────────

def load_smcxd03(filepath, sheet, disease_group, id_prefix):
    """SMCXD03/05 546-column health screening format (정상/당뇨/고혈압/당뇨+고혈압)."""
    df = pd.read_excel(filepath, sheet_name=sheet, header=0, engine="openpyxl")
    rows = []
    for _, r in df.iterrows():
        pid = r.iloc[0]  # Unnamed:0 or Sample number
        if pd.isna(pid):
            continue
        pid = str(pid).strip()

        # Age from birth year
        birth_year = r.get("생년")
        exam_date = r.get("검진일")
        age = None
        if pd.notna(birth_year) and pd.notna(exam_date):
            try:
                exam_year = int(str(int(exam_date))[:4])
                age = exam_year - int(birth_year)
            except Exception:
                pass

        row = {
            "patient_id": pid,
            "disease_group": disease_group,
            "source_file": Path(filepath).name,
            "age": age,
            "sex": normalize_sex(r.get("성별")),
            "height_cm": safe_float(r.get("신장")) if get_col(df, ["신장"]) else None,
            "weight_kg": safe_float(r.get("체중")) if get_col(df, ["체중"]) else None,
            "bmi": safe_float(r.get("체질량지수")),
            "bp_systolic": safe_float(r.get("혈압(수축)")),
            "bp_diastolic": safe_float(r.get("혈압(이완)")),
            "smoking_status": normalize_smoking(r.get("흡연-현재담배를피웁니까")),
            "drinking_status": normalize_drinking(r.get("술을 마십니까?")),
            "sample_date": normalize_date(exam_date),
            # CBC
            "wbc": safe_float(r.get("WBC")),
            "rbc": safe_float(r.get("RBC")),
            "hb": safe_float(r.get("Hb")),
            "hct": safe_float(r.get("Hct")),
            "platelet": safe_float(r.get("Platelet")),
            "neutrophil_pct": safe_float(r.get("seg neutrophil")),
            "lymphocyte_pct": safe_float(r.get("lymphocyte")),
            # Chemistry
            "ast": safe_float(r.get("AST")),
            "alt": safe_float(r.get("ALT")),
            "alp": safe_float(r.get("ALP")),
            "ggt": safe_float(r.get("GGT")),
            "bun": safe_float(r.get("BUN")),
            "creatinine": safe_float(r.get("Creatinine")),
            "uric_acid": safe_float(r.get("Uric Acid")),
            "glucose": safe_float(r.get("Glucose")),
            "total_protein": safe_float(r.get("T. Protein")),
            "albumin": safe_float(r.get("Albumin")),
            "total_bilirubin": safe_float(r.get("T. Bilirubin")),
            "ldh": safe_float(r.get("LDH")),
            "calcium": safe_float(r.get("Ca")),
            "hs_crp": safe_float(r.get("hsCRP")),
            "sodium": safe_float(r.get("Na")),
            "potassium": safe_float(r.get("K+")),
            "chloride": safe_float(r.get("Cl")),
            # Lipid
            "total_cholesterol": safe_float(r.get("T. Cholesterol")),
            "triglyceride": safe_float(r.get("Triglyceride (TG)")),
            "hdl_c": safe_float(r.get("HDL-C")),
            "ldl_c": safe_float(r.get("LDL-C")),
            "hba1c": safe_float(r.get("Hb A1c")),
            # Tumor markers
            "afp": safe_float(r.get("AFP")),
            "cea": safe_float(r.get("CEA")),
            "ca19_9": safe_float(r.get("CA 19-9")),
            "psa": safe_float(r.get("PSA")),
            # UA
            "ua_sg": safe_float(r.get("Urine SG")),
            "ua_ph": safe_float(r.get("pH")),
            "ua_protein": str(r.get("Urine protein")) if pd.notna(r.get("Urine protein")) else None,
            "ua_glucose": str(r.get("Urine glucose")) if pd.notna(r.get("Urine glucose")) else None,
            "ua_blood": str(r.get("Urine OB")) if pd.notna(r.get("Urine OB")) else None,
        }

        # Height/weight: 546-col files have these as empty in some rows;
        # 표준체중 col has data but 신장/체중 may be in different unnamed positions
        # BMI is reliably present → compute height/weight back if needed
        if row["bmi"] and not row["weight_kg"]:
            std_wt = safe_float(r.get("표준체중"))
            fat_pct = safe_float(r.get("체지방율"))
            # Can't reliably back-compute, leave None

        # Past history: collect binary flag columns
        hist_flags = {}
        for col in df.columns:
            if "과거력(질병)" in str(col) or "과거력(암)" in str(col):
                val = r.get(col)
                if pd.notna(val) and str(val).strip() not in ("아니오", "아니요"):
                    hist_flags[str(col)] = str(val).strip()
        if hist_flags:
            row["past_history"] = "; ".join(f"{k}={v}" for k, v in hist_flags.items())

        rows.append(row)
    return pd.DataFrame(rows)


def load_smcxd01_compact(filepath, sheet, disease_group):
    """SMCXD01 compact format (전립선암/유방암/난소암1/폐암1)."""
    df = pd.read_excel(filepath, sheet_name=sheet, header=0, engine="openpyxl")
    rows = []
    for _, r in df.iterrows():
        pid_col = get_col(df, ["Unnamed: 0", "SoluM Label", "Sample number", "NO",
                                "제공자bCODE", "제공자:제공자bCODE"])
        pid = str(r[pid_col]).strip() if pid_col and pd.notna(r.get(pid_col)) else None
        if not pid or pid == "nan":
            continue

        age_col = get_col(df, ["나이"])
        bdate_col = get_col(df, ["생년월일"])

        age = safe_float(r.get(age_col)) if age_col else None
        if age is None and bdate_col:
            bd = r.get(bdate_col)
            if pd.notna(bd):
                try:
                    bd_dt = pd.to_datetime(bd)
                    age = (pd.Timestamp.now() - bd_dt).days // 365
                except Exception:
                    pass

        row = {
            "patient_id": pid,
            "disease_group": disease_group,
            "source_file": Path(filepath).name,
            "age": age,
            "sex": normalize_sex(r.get(get_col(df, ["성별"]))),
            "height_cm": safe_float(r.get("신장")),
            "weight_kg": safe_float(r.get("체중")),
            "smoking_status": normalize_smoking(r.get(get_col(df, ["흡연력"]))),
            "drinking_status": normalize_drinking(r.get(get_col(df, ["음주력"]))),
            "past_history": str(r.get("과거력")).strip() if pd.notna(r.get("과거력")) else None,
            "diagnosis": str(r.get(get_col(df, ["주상병", "진단코드 및 진단명", "진단명"]))).strip()
                if get_col(df, ["주상병", "진단코드 및 진단명", "진단명"]) and
                pd.notna(r.get(get_col(df, ["주상병", "진단코드 및 진단명", "진단명"]))) else None,
            "diagnosis_date": normalize_date(r.get(get_col(df, ["진단일"]))),
            "sample_date": normalize_date(r.get(get_col(df, ["수집일", "자원 수집일", "인체자원:자원접수일"]))),
            "treatment": str(r.get(get_col(df, ["항암치료 및 방사선 치료이력"]))).strip()
                if get_col(df, ["항암치료 및 방사선 치료이력"]) and
                pd.notna(r.get(get_col(df, ["항암치료 및 방사선 치료이력"]))) else None,
        }
        # Parse structured values from blood test text
        bt_col = get_col(df, ["혈액검사결과 : CEA, CA19-9", "혈액검사결과"])
        if bt_col and pd.notna(r.get(bt_col)):
            parsed = parse_blood_test_text(r[bt_col])
            for k, v in parsed.items():
                if v is not None and row.get(k) is None:
                    row[k] = v

        # Pathology: try multiple candidate columns
        patho_col = get_col(df, [
            "병리결과",
            "병리결과( location, histologic type, grade, TNM stage, metastasis 여부, differentiation(분화도), lympho vascular invasion (cancer 환자의 경우))",
        ])
        if patho_col and pd.notna(r.get(patho_col)):
            row["pathology"] = str(r[patho_col]).strip()

        rows.append(row)
    return pd.DataFrame(rows)


def load_smcxd06_cancer(filepath, sheet, disease_group, header_row=0):
    """SMCXD06 mid-form (대장암/췌장암/방광암)."""
    df = pd.read_excel(filepath, sheet_name=sheet, header=header_row, engine="openpyxl")
    rows = []
    for _, r in df.iterrows():
        pid_col = get_col(df, ["Unnamed: 0", "NO"])
        pid = str(r[pid_col]).strip() if pid_col and pd.notna(r.get(pid_col)) else None
        if not pid or pid == "nan":
            continue

        age = safe_float(r.get(get_col(df, ["제공자상세:최초참여시나이", "나이"])))
        birth_year = safe_float(r.get(get_col(df, ["제공자:생년"])))

        row = {
            "patient_id": pid,
            "disease_group": disease_group,
            "source_file": Path(filepath).name,
            "age": age,
            "sex": normalize_sex(r.get(get_col(df, ["제공자:성별", "성별"]))),
            "height_cm": safe_float(r.get("신장")),
            "weight_kg": safe_float(r.get("체중")),
            "smoking_status": normalize_smoking(r.get(get_col(df, [
                "흡연력 (1=비흡연 2=과거흡연 현재비흡연 3= 현재흡연)", "흡연력"]))),
            "drinking_status": normalize_drinking(r.get(get_col(df, [
                "음주력 (1=비음주 2=과거음주 현재비음주 3= 현재음주)", "음주력"]))),
            "past_history": str(r.get(get_col(df, ["과거 질병력\xa0", "질병력", "과거력"]))).strip()
                if get_col(df, ["과거 질병력\xa0", "질병력", "과거력"]) and
                pd.notna(r.get(get_col(df, ["과거 질병력\xa0", "질병력", "과거력"]))) else None,
            "diagnosis_date": normalize_date(r.get(get_col(df, ["암 진단일", "인체자원그룹:진단일1"]))),
            "surgery_date": normalize_date(r.get(get_col(df, ["암 수술일", "수술일"]))),
            "sample_date": normalize_date(r.get(get_col(df, ["검체수집일", "검사 날짜"]))),
            "fasting": str(r.get("공복여부")).strip() if pd.notna(r.get("공복여부")) else None,
            "t_stage": str(r.get(get_col(df, ["인체자원그룹:T STAGE"]))).strip()
                if get_col(df, ["인체자원그룹:T STAGE"]) and pd.notna(r.get(get_col(df, ["인체자원그룹:T STAGE"]))) else None,
            "n_stage": str(r.get(get_col(df, ["인체자원그룹:N_STAGE2"]))).strip()
                if get_col(df, ["인체자원그룹:N_STAGE2"]) and pd.notna(r.get(get_col(df, ["인체자원그룹:N_STAGE2"]))) else None,
            "m_stage": str(r.get(get_col(df, ["인체자원그룹:M_STAGE"]))).strip()
                if get_col(df, ["인체자원그룹:M_STAGE"]) and pd.notna(r.get(get_col(df, ["인체자원그룹:M_STAGE"]))) else None,
            "stage": str(r.get(get_col(df, ["Overall stage"]))).strip()
                if get_col(df, ["Overall stage"]) and pd.notna(r.get(get_col(df, ["Overall stage"]))) else None,
            "pathology": str(r.get(get_col(df, ["병리결과", "병리결과 "]))).strip()
                if get_col(df, ["병리결과", "병리결과 "]) and pd.notna(r.get(get_col(df, ["병리결과", "병리결과 "]))) else None,
            "treatment": str(r.get(get_col(df, [
                "시행된 암 치료 정보(방광암수술,항암화학요법,방사선)",
                "항암치료 및 방사선 치료이력"]))).strip()
                if get_col(df, [
                    "시행된 암 치료 정보(방광암수술,항암화학요법,방사선)",
                    "항암치료 및 방사선 치료이력"]) else None,
            # New: surgery/treatment detail columns
            "surgery_name": str(r.get(get_col(df, ["수술명"]))).strip()
                if get_col(df, ["수술명"]) and pd.notna(r.get(get_col(df, ["수술명"]))) else None,
            "chemo_date": normalize_date(r.get(get_col(df, [" 항암일자", " 항암일자(시작일)", "항암일자"]))),
            "treatment_detail": str(r.get(get_col(df, [
                "암 치료정보(수술, 항암요법, 방사선 등)"]))).strip()
                if get_col(df, ["암 치료정보(수술, 항암요법, 방사선 등)"]) and
                pd.notna(r.get(get_col(df, ["암 치료정보(수술, 항암요법, 방사선 등)"]))) else None,
            # Lab
            "wbc": safe_float(r.get("WBC")),
            "rbc": safe_float(r.get("RBC")),
            "hb": safe_float(r.get(get_col(df, ["Hb", "HB"]))),
            "hct": safe_float(r.get(get_col(df, ["Hct", "HCT"]))),
            "platelet": safe_float(r.get(get_col(df, ["Platelet", "PLT"]))),
            "neutrophil_pct": safe_float(r.get(get_col(df, ["Neutrophil", "Seg.neut."]))),
            "lymphocyte_pct": safe_float(r.get(get_col(df, ["Lymphocytes", "Lymphocyte"]))),
            "ast": safe_float(r.get("AST")),
            "alt": safe_float(r.get("ALT")),
            "alp": safe_float(r.get("ALP")),
            "ggt": safe_float(r.get(get_col(df, ["γ-GTP", "GGT"]))),
            "bun": safe_float(r.get(get_col(df, ["BUN(S)", "BUN"]))),
            "creatinine": safe_float(r.get(get_col(df, ["Creatinine(S)", "Creatinine"]))),
            "uric_acid": safe_float(r.get(get_col(df, ["Uric acid(S)", "Uric Acid"]))),
            "glucose": safe_float(r.get(get_col(df, ["Glucose(S)", "Glucose"]))),
            "total_bilirubin": safe_float(r.get(get_col(df, ["Bilirubin, Total", "T. Bilirubin"]))),
            "total_cholesterol": safe_float(r.get(get_col(df, ["Cholesterol", "T. Cholesterol"]))),
            "triglyceride": safe_float(r.get(get_col(df, ["Triglyceride"]))),
            "calcium": safe_float(r.get(get_col(df, ["Calcium(S)", "Ca"]))),
            "potassium": safe_float(r.get("K")),
            "chloride": safe_float(r.get("Cl")),
            "hba1c": safe_float(r.get("HbA1c")),
            "bp_systolic": safe_float(r.get(get_col(df, ["수축기혈압"]))),
            "bp_diastolic": safe_float(r.get(get_col(df, ["이완기혈압"]))),
            # UA
            "ua_sg": safe_float(r.get(get_col(df, ["R.UA-S.G.", "Urine SG"]))),
            "ua_ph": safe_float(r.get(get_col(df, ["R.UA-pH", "pH"]))),
            "ua_protein": str(r.get(get_col(df, ["R.UA-Protein"]))).strip()
                if get_col(df, ["R.UA-Protein"]) and pd.notna(r.get(get_col(df, ["R.UA-Protein"]))) else None,
            "ua_glucose": str(r.get(get_col(df, ["R.UA-Glucose"]))).strip()
                if get_col(df, ["R.UA-Glucose"]) and pd.notna(r.get(get_col(df, ["R.UA-Glucose"]))) else None,
            "ua_blood": str(r.get(get_col(df, ["R.UA-Blood"]))).strip()
                if get_col(df, ["R.UA-Blood"]) and pd.notna(r.get(get_col(df, ["R.UA-Blood"]))) else None,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def load_lung2(filepath):
    """폐암2 (SMCXD06_폐암 2)."""
    df = pd.read_excel(filepath, sheet_name="Sheet1", header=0, engine="openpyxl")
    rows = []
    for _, r in df.iterrows():
        pid = str(r.get("Unnamed: 0", "")).strip()
        if not pid or pid == "nan":
            continue
        rows.append({
            "patient_id": pid,
            "disease_group": "LUN",
            "source_file": Path(filepath).name,
            "age": safe_float(r.get("나이")),
            "sex": normalize_sex(r.get("성별")),
            "height_cm": safe_float(r.get("신장")),
            "weight_kg": safe_float(r.get("체중")),
            "surgery_date": normalize_date(r.get("수술일")),
            "diagnosis_date": normalize_date(r.get("조직검사 결과 진단일")),
            "tnm": str(r.get("pTNM")).strip() if pd.notna(r.get("pTNM")) else None,
            "diagnosis": str(r.get("병리검사 진단명")).strip() if pd.notna(r.get("병리검사 진단명")) else None,
            "stage": str(r.get("stage")).strip() if pd.notna(r.get("stage")) else None,
            "metastasis": str(r.get("meta")).strip() if pd.notna(r.get("meta")) else None,
        })
    return pd.DataFrame(rows)


def load_lung3(filepath):
    """폐암3 (SMCXD06_폐암 3) - 3 sheets merged by 제공자:제공자bCODE."""
    demo = pd.read_excel(filepath, sheet_name="인구학적정보 및 암 관련 정보", header=0, engine="openpyxl")
    patho = pd.read_excel(filepath, sheet_name="병리검사", header=0, engine="openpyxl")
    lab = pd.read_excel(filepath, sheet_name="진단검사", header=0, engine="openpyxl")

    # Normalize join key to string across all sheets
    join_col = "제공자:제공자bCODE"
    for sheet_df in [demo, patho, lab]:
        if join_col in sheet_df.columns:
            sheet_df[join_col] = sheet_df[join_col].astype(str).str.strip()

    rows = []
    for i, r in demo.iterrows():
        pid = str(r.get("제공자:제공자bCODE", "")).strip()
        if not pid or pid == "nan":
            continue

        # Parse 성별/나이 from combined column
        sex_age = str(r.get("성별/나이", ""))
        sex, age = None, None
        m = re.match(r"([MF남여])\s*/?\s*(\d+)", sex_age)
        if m:
            sex = normalize_sex(m.group(1))
            age = int(m.group(2))

        row = {
            "patient_id": pid,
            "disease_group": "LUN",
            "source_file": "SMCXD06_폐암 3.xlsx",
            "age": age,
            "sex": sex,
            "height_cm": safe_float(r.get("신장")),
            "weight_kg": safe_float(r.get("체중")),
            "bmi": safe_float(r.get("BMI")),
            "smoking_status": normalize_smoking(r.get("흡연 여부")),
            "drinking_status": normalize_drinking(r.get("음주 여부")),
            "past_history": str(r.get("병력")).strip() if pd.notna(r.get("병력")) else None,
            "sample_date": normalize_date(r.get("인체자원:자원접수일")),
            "diagnosis_date": normalize_date(r.get("암 진단일")),
            "treatment": str(r.get("수술력(타질환)")).strip() if pd.notna(r.get("수술력(타질환)")) else None,
            "surgery_name": str(r.get("수술력")).strip() if pd.notna(r.get("수술력")) else None,
            "treatment_detail": str(r.get("암 치료 정보")).strip()
                if pd.notna(r.get("암 치료 정보")) and str(r.get("암 치료 정보")).strip() != "." else None,
        }

        # Merge pathology
        p_row = patho[patho["제공자:제공자bCODE"] == pid] if "제공자:제공자bCODE" in patho.columns else pd.DataFrame()
        if len(p_row) > 0:
            p = p_row.iloc[0]
            row["diagnosis"] = str(p.get("Histologic")).strip() if pd.notna(p.get("Histologic")) else None
            row["stage"] = str(p.get("Stage")).strip() if pd.notna(p.get("Stage")) else None
            row["tnm"] = str(p.get("TNM")).strip() if pd.notna(p.get("TNM")) else None
            row["metastasis"] = str(p.get("Metastasis")).strip() if pd.notna(p.get("Metastasis")) else None

        # Merge lab
        l_row = lab[lab["제공자:제공자bCODE"] == pid] if "제공자:제공자bCODE" in lab.columns else pd.DataFrame()
        if len(l_row) > 0:
            l = l_row.iloc[0]
            row["wbc"] = safe_float(l.get("WBC"))
            row["rbc"] = safe_float(l.get("RBC"))
            row["hb"] = safe_float(l.get("Hb"))
            row["hct"] = safe_float(l.get("HCT"))
            row["platelet"] = safe_float(l.get("PLT"))
            row["neutrophil_pct"] = safe_float(l.get("Seg.neut."))
            row["lymphocyte_pct"] = safe_float(l.get("Lymphocyte"))
            row["ast"] = safe_float(l.get("AST"))
            row["alt"] = safe_float(l.get("ALT"))
            row["alp"] = safe_float(l.get("ALP"))
            row["bun"] = safe_float(l.get("BUN"))
            row["creatinine"] = safe_float(l.get("Creatinine"))
            row["glucose"] = safe_float(l.get("Glucose"))
            row["total_protein"] = safe_float(l.get("Protein"))
            row["albumin"] = safe_float(l.get("Albumin"))
            row["total_bilirubin"] = safe_float(l.get("T.bilirubin"))
            row["uric_acid"] = safe_float(l.get("Uric acid"))
            row["sodium"] = safe_float(l.get("Na"))
            row["potassium"] = safe_float(l.get("K"))
            row["chloride"] = safe_float(l.get("Cl"))
            row["total_cholesterol"] = safe_float(l.get("T.Cholesterol"))
            row["calcium"] = safe_float(l.get("Ca"))
            row["hba1c"] = safe_float(l.get("HbA1c"))
            row["ggt"] = safe_float(l.get("GGT"))
            row["ua_ph"] = safe_float(l.get("pH"))
            row["ua_sg"] = safe_float(l.get("specific gravity"))
            row["ua_glucose"] = safe_float(l.get("Glucose.1"))

        rows.append(row)
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Standardize clinical data")
    parser.add_argument("--output-dir", default="data/clinical_data/standardized",
                        help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = Path("data/clinical_data")
    all_dfs = []

    # ── SMCXD03/05 (546-col) ──
    smcxd03_files = [
        (base / "5. 정상/SMCXD03_정상인 1.xlsx", "정상인", "NOR", "NOR"),
        (base / "5. 정상/SMCXD03_정상인 2.xlsx", "정상인", "NOR", "NOR"),
        (base / "6. 당뇨/SMCXD03_당뇨 1.xlsx", "당뇨", "DIA", "DIA"),
        (base / "6. 당뇨/SMCXD03_당뇨 2.xlsx", "당뇨", "DIA", "DIA"),
        (base / "7. 고혈압/SMCXD03_고혈압.xlsx", "고혈압", "HBP", "HBP"),
        (base / "8. 당뇨 + 고혈압/SMCXD05_당뇨+고혈압.xlsx", "당뇨+고혈압", "H.D.", "HD"),
    ]
    for fp, sheet, group, prefix in smcxd03_files:
        # 정상인1 has header in row 0 but col[0]=nan — read with header=0 is fine
        # because pd will name it Unnamed:0
        print(f"  Loading {fp.name} [{sheet}] ...")
        df = load_smcxd03(str(fp), sheet, group, prefix)
        all_dfs.append(df)
        print(f"    -> {len(df)} rows")

    # ── SMCXD01 compact ──
    smcxd01_files = [
        (base / "1. 전립선암/SMCXD01_전립선암 임상정보.xlsx", "Sheet1", "PRO"),
        (base / "2. 유방암/SMCXD01_유방암.xlsx", "C50 임상정보", "BRE"),
        (base / "3. 난소암/SMCXD01_난소암 1.xlsx", "C56 임상정보", "OVA"),
        (base / "3. 난소암/SMCXD01_난소암 2.xlsx", "난소", "OVA"),
        (base / "4. 폐암/SMCXD01_폐암 1.xlsx", "폐", "LUN"),
    ]
    for fp, sheet, group in smcxd01_files:
        print(f"  Loading {fp.name} [{sheet}] ...")
        df = load_smcxd01_compact(str(fp), sheet, group)
        all_dfs.append(df)
        print(f"    -> {len(df)} rows")

    # ── SMCXD06 mid-form ──
    print(f"  Loading SMCXD06_폐암 2.xlsx ...")
    df = load_lung2(str(base / "4. 폐암/SMCXD06_폐암 2.xlsx"))
    all_dfs.append(df)
    print(f"    -> {len(df)} rows")

    print(f"  Loading SMCXD06_폐암 3.xlsx (3 sheets) ...")
    df = load_lung3(str(base / "4. 폐암/SMCXD06_폐암 3.xlsx"))
    all_dfs.append(df)
    print(f"    -> {len(df)} rows")

    smcxd06_files = [
        (base / "9. 대장암/SMCXD06_대장암.xlsx", "대장암1~2기270명", "CRC", 0),
        (base / "9. 대장암/SMCXD06_대장암.xlsx", "대장암3~4기30명", "CRC", 0),
        (base / "10. 췌장암/CPAN/SMCMD06_췌장암.xlsx", "췌장암70명", "PAN", 0),
        (base / "11. 방광암/SMCXD06_방광암.xlsm", "분양명단", "BLA", 1),
    ]
    for fp, sheet, group, hrow in smcxd06_files:
        print(f"  Loading {fp.name} [{sheet}] ...")
        df = load_smcxd06_cancer(str(fp), sheet, group, hrow)
        all_dfs.append(df)
        print(f"    -> {len(df)} rows")

    # ── Merge all ─────────────────────────────────────────────────────────
    merged = pd.concat(all_dfs, ignore_index=True)

    # Ensure all standard columns exist
    for col in STD_COLUMNS:
        if col not in merged.columns:
            merged[col] = None

    # Reorder
    merged = merged[STD_COLUMNS]

    # ── Post-processing fixes ─────────────────────────────────────────────

    # Fix PRO (전립선암): height_cm and weight_kg are swapped in source data
    pro_mask = merged["disease_group"] == "PRO"
    pro_h = merged.loc[pro_mask, "height_cm"].copy()
    pro_w = merged.loc[pro_mask, "weight_kg"].copy()
    merged.loc[pro_mask, "height_cm"] = pro_w
    merged.loc[pro_mask, "weight_kg"] = pro_h
    merged.loc[pro_mask, "bmi"] = None  # recalculate below
    n_swapped = pro_mask.sum()
    print(f"  [FIX] PRO height/weight swapped for {n_swapped} rows")

    # Compute BMI where missing
    mask = merged["bmi"].isna() & merged["weight_kg"].notna() & merged["height_cm"].notna()
    h_m = merged.loc[mask, "height_cm"] / 100
    merged.loc[mask, "bmi"] = (merged.loc[mask, "weight_kg"] / (h_m ** 2)).round(1)

    # ── Compute sample_timing ─────────────────────────────────────────────
    # Determines whether the sample was collected pre-op, post-op, or same-day
    # relative to surgery_date (preferred) or diagnosis_date (fallback).
    # For BLA: parse surgery date from treatment text.
    print("  [TIMING] Computing sample_timing...")

    merged["sample_timing"] = None

    for idx, row in merged.iterrows():
        sample_dt = pd.to_datetime(row.get("sample_date"), errors="coerce")
        if pd.isna(sample_dt):
            continue

        # Try surgery_date first
        ref_dt = pd.to_datetime(row.get("surgery_date"), errors="coerce")
        ref_type = "surgery"

        # BLA: parse first surgery date from treatment text if surgery_date is empty
        if pd.isna(ref_dt) and row.get("disease_group") == "BLA" and pd.notna(row.get("treatment")):
            surg_matches = re.findall(r"(\d{8})\s*-?\s*수술", str(row["treatment"]))
            if surg_matches:
                try:
                    ref_dt = pd.to_datetime(surg_matches[0], format="%Y%m%d")
                    ref_type = "surgery_parsed"
                except Exception:
                    pass

        # Fallback to diagnosis_date
        if pd.isna(ref_dt):
            ref_dt = pd.to_datetime(row.get("diagnosis_date"), errors="coerce")
            ref_type = "diagnosis"

        # For non-cancer groups, no timing needed
        if row.get("disease_group") in ("NOR", "DIA", "HBP", "H.D."):
            merged.at[idx, "sample_timing"] = "control"
            continue

        if pd.isna(ref_dt):
            continue

        delta_days = (ref_dt - sample_dt).days
        if delta_days > 7:
            merged.at[idx, "sample_timing"] = "pre-op"
        elif delta_days < -7:
            merged.at[idx, "sample_timing"] = "post-op"
        else:
            merged.at[idx, "sample_timing"] = "peri-op"  # within ±7 days

    # Summary
    timing_counts = merged["sample_timing"].value_counts(dropna=False)
    print("  [TIMING] Results:")
    for t, c in timing_counts.items():
        label = t if pd.notna(t) else "unknown"
        print(f"    {label:12s}: {c}")

    # Per cancer group
    cancer_mask = ~merged["disease_group"].isin(["NOR", "DIA", "HBP", "H.D."])
    for g, gdf in merged[cancer_mask].groupby("disease_group"):
        tc = gdf["sample_timing"].value_counts(dropna=False).to_dict()
        print(f"    {g:6s}: {tc}")

    # ── Save ──────────────────────────────────────────────────────────────
    # Full merged
    merged_path = out_dir / "all_clinical_standardized.csv"
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
    print(f"\n{'='*60}")
    print(f"Merged: {merged_path}")
    print(f"  Total: {len(merged)} rows x {len(merged.columns)} cols")

    # Per disease
    for group, gdf in merged.groupby("disease_group"):
        gpath = out_dir / f"{group}_clinical_standardized.csv"
        gdf.to_csv(gpath, index=False, encoding="utf-8-sig")
        print(f"  {group}: {len(gdf)} rows -> {gpath.name}")

    # Summary stats
    print(f"\n{'='*60}")
    print("Column fill rates:")
    for col in STD_COLUMNS:
        n = merged[col].notna().sum()
        pct = n / len(merged) * 100
        if pct > 0:
            print(f"  {col:25s}: {n:5d}/{len(merged)} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
