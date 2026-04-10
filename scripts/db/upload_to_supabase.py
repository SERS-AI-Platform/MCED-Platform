"""
SERS-AI 데이터 → Supabase 업로드 스크립트
BLC 포함 전체 데이터셋 업로드
"""
import csv
import json
import requests
import sys
import os
from pathlib import Path

# Supabase config — override via environment variables for production.
# Env vars: SUPABASE_URL, SUPABASE_KEY
SUPABASE_URL = os.environ.get(
    "SUPABASE_URL",
    "https://dufapiffjjzcvesujfew.supabase.co",
)
SUPABASE_KEY = os.environ.get(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImR1ZmFwaWZmamp6Y3Zlc3VqZmV3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzQyMDQ5MTMsImV4cCI6MjA4OTc4MDkxM30.3ktmNfwnagM6N8aKRJBkEOMMaOJsu6IGxFOt9AwiM0M",
)

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal"
}

BASE_DIR = Path("/home/user/SERS-AI")
RESULTS_DIR = BASE_DIR / "results"
CLINICAL_DIR = BASE_DIR / "data" / "clinical_data" / "standardized"
DASHBOARD_DIR = Path("/home/user/workspace/solum-dashboard")

def post_batch(table, rows, batch_size=500):
    """Upload rows in batches via REST API."""
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    total = len(rows)
    uploaded = 0
    for i in range(0, total, batch_size):
        batch = rows[i:i+batch_size]
        resp = requests.post(url, headers=HEADERS, json=batch)
        if resp.status_code not in (200, 201, 204):
            print(f"  ERROR batch {i//batch_size}: {resp.status_code} {resp.text[:200]}")
            return False
        uploaded += len(batch)
        print(f"  {table}: {uploaded}/{total} rows uploaded")
    return True


def safe_float(val):
    """Convert to float, return None for empty/invalid."""
    if val is None or val == '' or val == 'nan':
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def safe_int(val):
    if val is None or val == '' or val == 'nan':
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def safe_date(val):
    if val is None or val == '' or val == 'nan':
        return None
    # Accept YYYY-MM-DD format
    val = str(val).strip()
    if len(val) >= 10 and val[4] == '-':
        return val[:10]
    return None


def upload_subjects():
    """Upload all_clinical_standardized.csv → subjects table."""
    print("\n=== Uploading subjects ===")
    filepath = CLINICAL_DIR / "all_clinical_standardized.csv"

    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {
                "patient_id": r.get("patient_id", "").strip(),
                "disease_group": r.get("disease_group", "").strip(),
                "source_file": r.get("source_file"),
                "age": safe_float(r.get("age")),
                "sex": r.get("sex", "").strip() or None,
                "height_cm": safe_float(r.get("height_cm")),
                "weight_kg": safe_float(r.get("weight_kg")),
                "bmi": safe_float(r.get("bmi")),
                "bp_systolic": safe_float(r.get("bp_systolic")),
                "bp_diastolic": safe_float(r.get("bp_diastolic")),
                "smoking_status": safe_float(r.get("smoking_status")),
                "drinking_status": safe_float(r.get("drinking_status")),
                "past_history": r.get("past_history") or None,
                "diagnosis": r.get("diagnosis") or None,
                "diagnosis_date": safe_date(r.get("diagnosis_date")),
                "sample_date": safe_date(r.get("sample_date")),
                "surgery_date": safe_date(r.get("surgery_date")),
                "pathology": r.get("pathology") or None,
                "stage": r.get("stage") or None,
                "tnm": r.get("tnm") or None,
                "t_stage": r.get("t_stage") or None,
                "n_stage": r.get("n_stage") or None,
                "m_stage": r.get("m_stage") or None,
                "metastasis": r.get("metastasis") or None,
                "treatment": r.get("treatment") or None,
                "fasting": r.get("fasting") or None,
                "wbc": safe_float(r.get("wbc")),
                "rbc": safe_float(r.get("rbc")),
                "hb": safe_float(r.get("hb")),
                "hct": safe_float(r.get("hct")),
                "platelet": safe_float(r.get("platelet")),
                "neutrophil_pct": safe_float(r.get("neutrophil_pct")),
                "lymphocyte_pct": safe_float(r.get("lymphocyte_pct")),
                "ast": safe_float(r.get("ast")),
                "alt": safe_float(r.get("alt")),
                "alp": safe_float(r.get("alp")),
                "ggt": safe_float(r.get("ggt")),
                "bun": safe_float(r.get("bun")),
                "creatinine": safe_float(r.get("creatinine")),
                "uric_acid": safe_float(r.get("uric_acid")),
                "glucose": safe_float(r.get("glucose")),
                "total_protein": safe_float(r.get("total_protein")),
                "albumin": safe_float(r.get("albumin")),
                "total_bilirubin": safe_float(r.get("total_bilirubin")),
                "ldh": safe_float(r.get("ldh")),
                "calcium": safe_float(r.get("calcium")),
                "hs_crp": safe_float(r.get("hs_crp")),
                "sodium": safe_float(r.get("sodium")),
                "potassium": safe_float(r.get("potassium")),
                "chloride": safe_float(r.get("chloride")),
                "total_cholesterol": safe_float(r.get("total_cholesterol")),
                "triglyceride": safe_float(r.get("triglyceride")),
                "hdl_c": safe_float(r.get("hdl_c")),
                "ldl_c": safe_float(r.get("ldl_c")),
                "hba1c": safe_float(r.get("hba1c")),
                "afp": safe_float(r.get("afp")),
                "cea": safe_float(r.get("cea")),
                "ca19_9": safe_float(r.get("ca19_9")),
                "psa": safe_float(r.get("psa")),
                "ua_sg": safe_float(r.get("ua_sg")),
                "ua_ph": safe_float(r.get("ua_ph")),
                "ua_protein": r.get("ua_protein") or None,
                "ua_glucose": r.get("ua_glucose") or None,
                "ua_blood": r.get("ua_blood") or None,
                "sample_timing": r.get("sample_timing") or None,
            }
            if row["patient_id"]:
                rows.append(row)

    print(f"  Loaded {len(rows)} subjects")
    return post_batch("subjects", rows)


def upload_spectra():
    """Upload processed_spectra.csv → spectra table (intensities as JSONB)."""
    print("\n=== Uploading spectra ===")
    filepath = RESULTS_DIR / "processed_spectra.csv"

    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        wavenumber_cols = [c for c in reader.fieldnames if c.startswith('x_')]
        # Extract wavenumber values from column names (x_402.00 → 402.00)
        wavenumbers = [c[2:] for c in wavenumber_cols]

        for r in reader:
            intensities = {}
            for wn, col in zip(wavenumbers, wavenumber_cols):
                val = safe_float(r[col])
                if val is not None:
                    intensities[wn] = round(val, 8)

            row = {
                "disease_group": r["group"],
                "sample_id": safe_int(r["sample_id"]),
                "replicate": safe_int(r["replicate"]),
                "intensities": intensities,
            }
            rows.append(row)

    print(f"  Loaded {len(rows)} spectra ({len(wavenumber_cols)} wavenumbers each)")
    # Smaller batches for JSONB data
    return post_batch("spectra", rows, batch_size=100)


def upload_sers_clinical():
    """Upload sers_clinical_merged.csv → sers_clinical_merged table."""
    print("\n=== Uploading sers_clinical_merged ===")
    filepath = CLINICAL_DIR / "sers_clinical_merged.csv"

    rows = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {
                "disease_group": r.get("group", "").strip(),
                "sample_id": safe_int(r.get("sample_id")),
                "n_replicates": safe_int(r.get("n_replicates")),
                "peak_wavenumber": safe_float(r.get("peak_wavenumber")),
                "peak_intensity": safe_float(r.get("peak_intensity")),
                "total_intensity": safe_float(r.get("total_intensity")),
                "mean_intensity": safe_float(r.get("mean_intensity")),
                "std_intensity": safe_float(r.get("std_intensity")),
                "band_600_650": safe_float(r.get("band_600_650")),
                "band_720_760": safe_float(r.get("band_720_760")),
                "band_1000_1050": safe_float(r.get("band_1000_1050")),
                "band_1200_1300": safe_float(r.get("band_1200_1300")),
                "band_1400_1500": safe_float(r.get("band_1400_1500")),
                "band_1600_1700": safe_float(r.get("band_1600_1700")),
                "patient_id": r.get("patient_id", "").strip() or None,
            }
            if row["disease_group"]:
                rows.append(row)

    print(f"  Loaded {len(rows)} merged records")
    return post_batch("sers_clinical_merged", rows)


def upload_group_statistics():
    """Upload group_statistics.csv + BLC stats → group_statistics table."""
    print("\n=== Uploading group_statistics ===")
    filepath = RESULTS_DIR / "group_statistics.csv"

    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["group"] == "UNK":
                continue  # Skip unknown
            rows.append({
                "disease_group": r["group"],
                "n_samples": safe_int(r["n_samples"]),
                "n_spectra": safe_int(r["n_spectra"]),
                "mean_replicates": safe_float(r["mean_replicates"]),
                "median_replicates": safe_float(r["median_replicates"]),
                "min_replicates": safe_int(r["min_replicates"]),
                "max_replicates": safe_int(r["max_replicates"]),
            })

    # BLC가 group_statistics.csv에 없으면 추가
    blc_exists = any(r["disease_group"] == "BLC" for r in rows)
    if not blc_exists:
        rows.append({
            "disease_group": "BLC",
            "n_samples": 299,
            "n_spectra": 1495,
            "mean_replicates": 5.0,
            "median_replicates": 5.0,
            "min_replicates": 5,
            "max_replicates": 5,
        })

    print(f"  Loaded {len(rows)} groups")
    return post_batch("group_statistics", rows)


def upload_experiment_results():
    """Upload experiment phase results."""
    print("\n=== Uploading experiment_results ===")

    rows = [
        {"phase": "A", "description": "Initial medoid baseline (merged PAN)",
         "aggregation_mode": "medoid", "cancer_set": ["PRO","BRE","OVA","LUN","CRC","PAN"],
         "non_cancer_set": ["NOR","DIA","HBP","H.D."], "n_subjects": 1342,
         "model": "Various", "screening_auc": 0.793, "type_id_f1": 0.356},
        {"phase": "A-improved", "description": "Separated CPAN/SPAN",
         "aggregation_mode": "medoid", "cancer_set": ["PRO","BRE","OVA","LUN","CRC","CPAN","SPAN"],
         "non_cancer_set": ["NOR","DIA","HBP","H.D."], "n_subjects": 1342,
         "model": "Various", "screening_auc": 0.948, "type_id_f1": 0.548},
        {"phase": "B", "description": "All-spectra mode",
         "aggregation_mode": "all_spectra", "cancer_set": ["PRO","BRE","OVA","LUN","CRC","CPAN","SPAN"],
         "non_cancer_set": ["NOR","DIA","HBP","H.D."], "n_spectra": 6710,
         "model": "Various", "screening_auc": 0.958, "type_id_f1": 0.613},
        {"phase": "D", "description": "First benchmark, XGBoost best",
         "aggregation_mode": "all_spectra", "cancer_set": ["PRO","BRE","OVA","LUN","CRC","CPAN","SPAN"],
         "non_cancer_set": ["NOR","DIA","HBP","H.D."],
         "model": "XGBoost", "screening_auc": 0.970, "type_id_f1": 0.776},
        {"phase": "F", "description": "Full benchmark - LogReg wins",
         "aggregation_mode": "all_spectra", "cancer_set": ["PRO","BRE","OVA","LUN","CRC","CPAN","SPAN"],
         "non_cancer_set": ["NOR","DIA","HBP","H.D."], "n_spectra": 6200,
         "model": "LogisticRegression", "screening_auc": 0.981, "type_id_f1": 0.870},
        {"phase": "Q", "description": "Sex-based biological constraint, +4.0pp F1",
         "aggregation_mode": "all_spectra", "cancer_set": ["PRO","BRE","OVA","LUN","CRC"],
         "non_cancer_set": ["NOR","DIA","HBP","H.D."],
         "model": "LR+sex_constraint", "screening_auc": 0.977, "type_id_f1": 0.892,
         "notes": "Eliminated PRO↔OVA confusion via sex constraint"},
        {"phase": "Q-fusion", "description": "LR + age/sex/BMI fusion + sex constraint",
         "aggregation_mode": "all_spectra", "cancer_set": ["PRO","BRE","OVA","LUN","CRC"],
         "non_cancer_set": ["NOR","DIA","HBP","H.D."],
         "model": "LR+fusion+sex_constraint", "screening_auc": 0.986, "type_id_f1": 0.884},
        {"phase": "T", "description": "BLC added, 8-class model", "date": "2026-03-23",
         "aggregation_mode": "all_spectra",
         "cancer_set": ["PRO","BRE","OVA","LUN","CRC","CPAN","SPAN","BLC"],
         "non_cancer_set": ["NOR","DIA","HBP","H.D."],
         "model": "LogisticRegression", "screening_auc": 0.969, "type_id_f1": 0.738,
         "notes": "BLC per-class: Sens 96.7%, Prec 99.0%, AUC 0.998"},
    ]

    print(f"  Loaded {len(rows)} experiment phases")
    return post_batch("experiment_results", rows)


def upload_qc_reports():
    """Upload QC reports from JSON files."""
    print("\n=== Uploading qc_reports ===")

    rows = []
    qc_files = list(RESULTS_DIR.glob("qc_report_*.json"))
    for qf in qc_files:
        instrument = qf.stem.replace("qc_report_", "")
        with open(qf, 'r') as f:
            data = json.load(f)

        total = data.get("total_files", 0)
        passed = data.get("passed", 0)
        failed = data.get("failed", 0)
        pass_rate = (passed / total * 100) if total > 0 else 0

        rows.append({
            "instrument": instrument,
            "total_files": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(pass_rate, 2),
            "details": data,
        })

    print(f"  Loaded {len(rows)} QC reports")
    return post_batch("qc_reports", rows)


def upload_dashboard_config():
    """Upload dashboard_data.json → dashboard_config table."""
    print("\n=== Uploading dashboard_config ===")

    filepath = DASHBOARD_DIR / "dashboard_data.json"
    if not filepath.exists():
        filepath = BASE_DIR / "dashboard" / "dashboard_data.json"

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    version = data.get("meta", {}).get("version", "unknown")
    rows = [{"version": version, "data": data}]

    print(f"  Dashboard version: {version}")
    return post_batch("dashboard_config", rows)


def main():
    print("=" * 60)
    print("SERS-AI → Supabase 데이터 업로드")
    print("=" * 60)

    results = {}

    # 1. Subjects (must be first due to FK references)
    results["subjects"] = upload_subjects()

    # 2. Group statistics
    results["group_statistics"] = upload_group_statistics()

    # 3. SERS-clinical merged
    results["sers_clinical_merged"] = upload_sers_clinical()

    # 4. Experiment results
    results["experiment_results"] = upload_experiment_results()

    # 5. QC reports
    results["qc_reports"] = upload_qc_reports()

    # 6. Dashboard config
    results["dashboard_config"] = upload_dashboard_config()

    # 7. Spectra (largest, last)
    results["spectra"] = upload_spectra()

    print("\n" + "=" * 60)
    print("업로드 결과:")
    for table, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {table}")
    print("=" * 60)


if __name__ == "__main__":
    main()
