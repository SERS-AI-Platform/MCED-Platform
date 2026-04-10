"""
SPAN (Samsung Pancreatic Cancer) CRF 데이터 → 표준화된 임상 데이터 변환
소스: /mnt/c/.../10. 췌장암/SPAN/20260203/
출력: data/clinical_data/standardized/SPAN_clinical_standardized.csv
"""
import csv
import os
import math
from collections import defaultdict

SPAN_DIR = "/mnt/c/Users/user/OneDrive - solum/바탕 화면/SERS-AI/data/clinical_data/10. 췌장암/SPAN/20260203"
OUTPUT_DIR = "/home/user/SERS-AI/data/clinical_data/standardized"

# Standard columns matching existing format
STANDARD_COLS = [
    "patient_id", "disease_group", "source_file",
    "age", "sex", "height_cm", "weight_kg", "bmi",
    "bp_systolic", "bp_diastolic", "smoking_status", "drinking_status",
    "past_history", "diagnosis", "diagnosis_date", "sample_date", "surgery_date",
    "pathology", "stage", "tnm", "t_stage", "n_stage", "m_stage",
    "metastasis", "treatment", "fasting",
    "surgery_name", "chemo_date", "treatment_detail",
    "wbc", "rbc", "hb", "hct", "platelet", "neutrophil_pct", "lymphocyte_pct",
    "ast", "alt", "alp", "ggt", "bun", "creatinine", "uric_acid", "glucose",
    "total_protein", "albumin", "total_bilirubin", "ldh", "calcium", "hs_crp",
    "sodium", "potassium", "chloride",
    "total_cholesterol", "triglyceride", "hdl_c", "ldl_c", "hba1c",
    "afp", "cea", "ca19_9", "psa",
    "ua_sg", "ua_ph", "ua_protein", "ua_glucose", "ua_blood",
    "sample_timing",
]


def read_csv(filename):
    """Read CSV with BOM handling."""
    path = os.path.join(SPAN_DIR, filename)
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def safe_float(val):
    if val is None or val == "" or val == "nan":
        return None
    try:
        v = float(val)
        return v if not math.isnan(v) else None
    except (ValueError, TypeError):
        return None


def map_sex(code):
    """CRF sex code: 1=Male, 2=Female."""
    if code == "1":
        return "M"
    elif code == "2":
        return "F"
    return None


def map_smoking(code):
    """CRF smoking: 1=current, 2=ex, 3=never."""
    mapping = {"1": 2.0, "2": 1.0, "3": 0.0}
    return mapping.get(str(code))


def map_drinking(code):
    """CRF drinking: 1=rarely, 2=sometimes, 3=often, 4=never."""
    mapping = {"4": 0.0, "1": 0.5, "2": 1.0, "3": 2.0}
    return mapping.get(str(code))


def subjid_to_patient_id(subjid):
    """SMCXD1-001 → SPAN 1."""
    num = subjid.replace("SMCXD1-", "")
    try:
        return f"SPAN {int(num)}"
    except ValueError:
        return f"SPAN {num}"


def main():
    print("Loading SPAN CRF data...")

    # 1. Load demographics (DM)
    dm_rows = read_csv("DM.csv")
    subjects = {}
    for r in dm_rows:
        sid = r["SUBJID"]
        pid = subjid_to_patient_id(sid)

        height = safe_float(r.get("HEIGHT"))
        weight = safe_float(r.get("WEIGHT"))
        bmi = None
        if height and weight and height > 0:
            bmi = round(weight / ((height / 100) ** 2), 1)

        subjects[sid] = {
            "patient_id": pid,
            "disease_group": "SPAN",
            "source_file": "SPAN_CRF_20260203",
            "age": safe_float(r.get("AGE")),
            "sex": map_sex(r.get("SEX")),
            "height_cm": height,
            "weight_kg": weight,
            "bmi": bmi,
            "bp_systolic": None,
            "bp_diastolic": None,
            "smoking_status": map_smoking(r.get("SMK")),
            "drinking_status": map_drinking(r.get("DRK")),
            "sample_date": r.get("URDTC") or None,
            "fasting": r.get("DMFASTDUR") or None,
        }

    print(f"  DM: {len(subjects)} subjects")

    # 2. Load medical history (MH) → past_history
    mh_rows = read_csv("MH.csv")
    mh_by_subj = defaultdict(list)
    for r in mh_rows:
        term = r.get("MHTERM", "").strip()
        if term:
            mh_by_subj[r["SUBJID"]].append(term)

    for sid, terms in mh_by_subj.items():
        if sid in subjects:
            subjects[sid]["past_history"] = "; ".join(terms)

    print(f"  MH: {len(mh_by_subj)} subjects with history")

    # 3. Load lab data (LB) → lab values
    lb_rows = read_csv("LB.csv")
    # Map CRF lab test names to standard column names
    lab_map = {
        "AST (SGOT)": "ast",
        "ALT (SGPT)": "alt",
        "GGT": "ggt",
        "Total bilirubin": "total_bilirubin",
        "Amylase": None,  # no standard column
        "Lipase": None,   # no standard column
        "Carcinoembryonic Antigen (CEA)": "cea",
        "Carbohydrate Antigen19-9 (CA19-9)": "ca19_9",
    }

    for r in lb_rows:
        sid = r["SUBJID"]
        if sid not in subjects:
            continue
        test = r.get("LBTEST", "").strip()
        col = lab_map.get(test)
        if col:
            val = safe_float(r.get("LBORRES"))
            if val is not None:
                subjects[sid][col] = val

    print(f"  LB: processed {len(lb_rows)} lab records")

    # 4. Load tumor data (TU) → stage info
    try:
        tu_rows = read_csv("TU.csv")
        for r in tu_rows:
            sid = r["SUBJID"]
            if sid not in subjects:
                continue
            # TUORRES1/TUORRES2 = tumor staging codes
            t1 = r.get("TUORRES1", "")
            t2 = r.get("TUORRES2", "")
            if t1 or t2:
                subjects[sid]["stage"] = f"T{t1}N{t2}" if t1 and t2 else None
        print(f"  TU: {len(tu_rows)} tumor records")
    except Exception as e:
        print(f"  TU: skipped ({e})")

    # 5. Build output
    output_rows = []
    for sid in sorted(subjects.keys()):
        s = subjects[sid]
        row = {}
        for col in STANDARD_COLS:
            row[col] = s.get(col, None)
        # Set diagnosis for pancreatic cancer
        row["diagnosis"] = "Pancreatic Cancer"
        output_rows.append(row)

    # Write SPAN standardized file
    output_path = os.path.join(OUTPUT_DIR, "SPAN_clinical_standardized.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STANDARD_COLS)
        writer.writeheader()
        for row in output_rows:
            # Convert None to empty string
            clean = {k: ("" if v is None else v) for k, v in row.items()}
            writer.writerow(clean)

    print(f"\nOutput: {output_path}")
    print(f"Total: {len(output_rows)} SPAN subjects")

    # Also update all_clinical_standardized.csv
    all_path = os.path.join(OUTPUT_DIR, "all_clinical_standardized.csv")
    existing_rows = []
    with open(all_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        existing_cols = reader.fieldnames
        for r in reader:
            # Skip any existing SPAN rows
            if r.get("patient_id", "").startswith("SPAN "):
                continue
            existing_rows.append(r)

    # Append SPAN rows
    with open(all_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=existing_cols)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)
        for row in output_rows:
            clean = {k: ("" if v is None else v) for k, v in row.items()}
            # Only write columns that exist in the all file
            out = {k: clean.get(k, "") for k in existing_cols}
            writer.writerow(out)

    print(f"Updated: {all_path} (+{len(output_rows)} SPAN rows)")


if __name__ == "__main__":
    main()
