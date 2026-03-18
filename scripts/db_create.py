"""
db_create.py - Create and populate the SERS PostgreSQL database.

Usage:
    python db_create.py

Requires:
    - PostgreSQL running at localhost:5432 with database 'sers_db'
    - Python packages: psycopg2, pandas, numpy, openpyxl
"""

import sys
import re
import json
import logging
import time
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sers.config import (
    RAW_DATA_DIR, SERS_EQUIPMENT_TEST_DATA_DIR, CLINICAL_DATA_DIR, load_config,
)
from sers.io import read_spectrum, parse_filename, find_spectra

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_CONFIG = dict(
    host="localhost",
    port=5432,
    database="postgres",
    user="postgres",
    password="solumhc1",
)

FLUSH_INTERVAL = 200  # flush spectral_data to DB every N files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Map filename-parsed group codes to canonical disease codes
GROUP_NORMALIZE = {"SPAN": "S-PAN"}

# Seed data
DISEASE_CATEGORIES = [
    ("NOR", "Normal", 100),
    ("DIA", "Diabetes", 100),
    ("HBP", "High Blood Pressure", 100),
    ("H.D.", "High Blood Pressure + Diabetes", 100),
    ("LUN", "Lung Cancer", 30),
    ("OVA", "Ovarian Cancer", 70),
    ("PRO", "Prostate Cancer", 100),
    ("S-PAN", "S-Pancreatic Cancer", 41),
    ("BRE", "Breast Cancer", 30),
]

EQUIPMENT_TYPES = [
    ("thermo", "CSV", "50-3300"),
    ("handheld", "CSV", "400-2300"),
    ("medical", "TXT", "100-3300"),
    ("nanoscope", "TXT", "100-3300"),
]

# Clinical file -> disease code mapping
CLINICAL_FILE_DISEASE = {
    "2. Lung cancer 임상정보.xlsx": "LUN",
    "3. Colorectal cancer 임상정보.xlsx": None,
    "4. High blood pressure 임상정보.xlsx": "HBP",
    "5. Diabetes 임상정보.xlsx": "DIA",
    "CRC_patients.xlsx": None,
    "DIA_patients.xlsx": "DIA",
    "H_D_patients.xlsx": "H.D.",
    "HBP_patients.xlsx": "HBP",
    "Lung_patients.xlsx": "LUN",
    "Norm_patients.xlsx": "NOR",
}

AVERAGED_FOLDER_NAMES = ["Averaged data", "Average data"]

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------
SCHEMA_SQL = """
-- Drop existing tables (reverse dependency order)
DROP TABLE IF EXISTS clinical_info CASCADE;
DROP TABLE IF EXISTS spectral_data CASCADE;
DROP TABLE IF EXISTS sers_spectra CASCADE;
DROP TABLE IF EXISTS patients CASCADE;
DROP TABLE IF EXISTS equipment_types CASCADE;
DROP TABLE IF EXISTS disease_categories CASCADE;

-- 1. disease_categories
CREATE TABLE disease_categories (
    id SERIAL PRIMARY KEY,
    code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    expected_samples INTEGER
);

-- 2. equipment_types
CREATE TABLE equipment_types (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    file_format VARCHAR(10),
    wavenumber_range VARCHAR(50)
);

-- 3. patients
CREATE TABLE patients (
    patient_id VARCHAR(50) PRIMARY KEY,
    disease_category_id INTEGER REFERENCES disease_categories(id),
    group_name VARCHAR(20),
    sample_number INTEGER
);

-- 4. sers_spectra
CREATE TABLE sers_spectra (
    id SERIAL PRIMARY KEY,
    patient_id VARCHAR(50) REFERENCES patients(patient_id),
    equipment_type_id INTEGER REFERENCES equipment_types(id),
    replicate_num INTEGER,
    is_averaged BOOLEAN DEFAULT FALSE,
    data_source VARCHAR(20),
    source_file VARCHAR(500),
    n_points INTEGER
);

-- 5. spectral_data
CREATE TABLE spectral_data (
    spectrum_id INTEGER REFERENCES sers_spectra(id),
    point_index SMALLINT,
    raman_shift DOUBLE PRECISION,
    intensity DOUBLE PRECISION,
    PRIMARY KEY (spectrum_id, point_index)
);

-- 6. clinical_info
CREATE TABLE clinical_info (
    id SERIAL PRIMARY KEY,
    patient_id VARCHAR(50) REFERENCES patients(patient_id),
    disease_category_id INTEGER REFERENCES disease_categories(id),
    source_file VARCHAR(255),
    data JSONB
);

-- 7. Indexes
CREATE INDEX idx_spectral_data_spectrum_id ON spectral_data(spectrum_id);
CREATE INDEX idx_spectral_data_raman_shift ON spectral_data(raman_shift);
CREATE INDEX idx_sers_spectra_patient_id ON sers_spectra(patient_id);
CREATE INDEX idx_sers_spectra_data_source ON sers_spectra(data_source);
CREATE INDEX idx_clinical_info_patient_id ON clinical_info(patient_id);
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except UnicodeDecodeError:
        # Korean Windows: PostgreSQL error messages in cp949 can't be decoded as UTF-8
        log.error(
            "DB connection failed (error message encoding issue on Korean Windows). "
            "Check: 1) password is correct  2) database 'sers_db' exists  "
            "3) PostgreSQL is running on localhost:5432"
        )
        raise SystemExit(1)
    conn.set_client_encoding("UTF8")
    return conn


def normalize_group(group: str) -> str:
    return GROUP_NORMALIZE.get(group, group)


def parse_averaged_filename(path: Path):
    """Parse 'NOR 1_ave.CSV' -> (group, sample_id_str) or (None, None)."""
    stem = path.stem
    m = re.match(
        r"^([A-Za-z]+(?:\.[A-Za-z]+)*\.?)\s*(\d+)_ave$", stem, re.IGNORECASE
    )
    if m:
        return m.group(1).strip().upper(), m.group(2)
    return None, None


def read_handheld_spectrum(path: Path):
    """Parse handheld CSV with metadata header. Returns (raman_shift, intensity, name)."""
    metadata = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                key = parts[0].strip('"')
                val = parts[1].strip('"')
                metadata[key] = val

    intensities_str = metadata.get("Intensities", "")
    if not intensities_str:
        raise ValueError(f"No Intensities found in {path}")

    first_wn = float(metadata.get("Firstwavenumber", 0))
    last_wn = float(metadata.get("LastWavenumber", 0))
    name = metadata.get("Name", path.stem)

    intensities = [float(x) for x in intensities_str.split(",") if x.strip()]
    raman_shift = np.linspace(first_wn, last_wn, len(intensities))
    intensity = np.array(intensities)
    return raman_shift, intensity, name


def parse_handheld_name(name: str):
    """Parse 'NOR 1_1' -> (group, sample_id_str, replicate) or (None, None, None)."""
    m = re.match(r"^([A-Za-z]+(?:\.[A-Za-z]+)*\.?)\s*(\d+)_(\d+)$", name)
    if m:
        return m.group(1).strip().upper(), m.group(2), int(m.group(3))
    return None, None, None


def bulk_copy_spectral(cur, rows):
    """COPY FROM for fast bulk insert into spectral_data."""
    if not rows:
        return
    buf = StringIO()
    for sid, pidx, rs, inten in rows:
        buf.write(f"{sid}\t{pidx}\t{rs}\t{inten}\n")
    buf.seek(0)
    cur.copy_from(
        buf, "spectral_data",
        columns=("spectrum_id", "point_index", "raman_shift", "intensity"),
    )


def collect_csv_files(directory: Path) -> list[Path]:
    """Find CSV/csv spectrum files in a directory (non-recursive)."""
    files = set(find_spectra(directory, pattern="*.CSV", recursive=False))
    files |= set(find_spectra(directory, pattern="*.csv", recursive=False))
    return sorted(files)


def collect_txt_files(directory: Path) -> list[Path]:
    """Find TXT/txt spectrum files in a directory (non-recursive)."""
    files = set(find_spectra(directory, pattern="*.TXT", recursive=False))
    files |= set(find_spectra(directory, pattern="*.txt", recursive=False))
    return sorted(files)


# ---------------------------------------------------------------------------
# Step 1 & 2: Schema + seed
# ---------------------------------------------------------------------------

def create_schema(conn):
    log.info("Creating schema (DROP + CREATE)...")
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()
    log.info("Schema created.")


def seed_lookup_tables(conn):
    log.info("Seeding lookup tables...")
    with conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO disease_categories (code, name, expected_samples) VALUES %s",
            DISEASE_CATEGORIES,
        )
        execute_values(
            cur,
            "INSERT INTO equipment_types (name, file_format, wavenumber_range) VALUES %s",
            EQUIPMENT_TYPES,
        )
    conn.commit()

    disease_ids = {}
    equipment_ids = {}
    with conn.cursor() as cur:
        cur.execute("SELECT code, id FROM disease_categories")
        disease_ids = dict(cur.fetchall())
        cur.execute("SELECT name, id FROM equipment_types")
        equipment_ids = dict(cur.fetchall())

    log.info(
        f"Seeded {len(disease_ids)} disease categories, "
        f"{len(equipment_ids)} equipment types."
    )
    return disease_ids, equipment_ids


# ---------------------------------------------------------------------------
# Step 3: Load raw_data
# ---------------------------------------------------------------------------

def load_raw_data(conn, disease_ids):
    log.info("=" * 60)
    log.info("STEP 3: Loading raw_data...")
    t0 = time.time()

    config = load_config()
    folder_to_group = config.folder_to_group
    total_spectra = 0
    total_points = 0

    for folder_name, group_code in folder_to_group.items():
        folder_path = RAW_DATA_DIR / folder_name
        if not folder_path.is_dir():
            log.warning(f"Folder not found: {folder_path}")
            continue

        disease_cat_id = disease_ids.get(group_code)
        if disease_cat_id is None:
            log.warning(f"No disease category for '{group_code}', skipping")
            continue

        log.info(f"  [{group_code}] {folder_name}")

        # --- Individual replicate files ---
        csv_files = collect_csv_files(folder_path)
        patients_seen: set[str] = set()
        spectral_buf: list[tuple] = []
        folder_spectra = 0

        with conn.cursor() as cur:
            for i, fpath in enumerate(csv_files):
                try:
                    spec_id = parse_filename(fpath, fallback_group=group_code)
                except ValueError:
                    log.warning(f"    Cannot parse: {fpath.name}")
                    continue

                sample_num = int(spec_id.sample_id)
                patient_id = f"{group_code} {sample_num}"

                if patient_id not in patients_seen:
                    cur.execute(
                        "INSERT INTO patients (patient_id, disease_category_id, "
                        "group_name, sample_number) "
                        "VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                        (patient_id, disease_cat_id, group_code, sample_num),
                    )
                    patients_seen.add(patient_id)

                try:
                    raman_shift, intensity = read_spectrum(fpath)
                except Exception as e:
                    log.warning(f"    Read failed: {fpath.name}: {e}")
                    continue

                n_pts = len(raman_shift)
                cur.execute(
                    "INSERT INTO sers_spectra "
                    "(patient_id, equipment_type_id, replicate_num, is_averaged, "
                    "data_source, source_file, n_points) "
                    "VALUES (%s, NULL, %s, FALSE, 'raw_data', %s, %s) RETURNING id",
                    (patient_id, spec_id.replicate, fpath.name, n_pts),
                )
                spectrum_id = cur.fetchone()[0]

                for j in range(n_pts):
                    spectral_buf.append(
                        (spectrum_id, j, float(raman_shift[j]), float(intensity[j]))
                    )
                folder_spectra += 1

                # periodic flush
                if (i + 1) % FLUSH_INTERVAL == 0 and spectral_buf:
                    bulk_copy_spectral(cur, spectral_buf)
                    total_points += len(spectral_buf)
                    spectral_buf = []

            # final flush for this folder
            if spectral_buf:
                bulk_copy_spectral(cur, spectral_buf)
                total_points += len(spectral_buf)
                spectral_buf = []

        conn.commit()
        total_spectra += folder_spectra
        log.info(f"    {folder_spectra} replicate files loaded")

        # --- Averaged data ---
        avg_folder = None
        for avg_name in AVERAGED_FOLDER_NAMES:
            candidate = folder_path / avg_name
            if candidate.is_dir():
                avg_folder = candidate
                break

        if avg_folder is None:
            continue

        avg_files = collect_csv_files(avg_folder)
        avg_count = 0
        spectral_buf = []

        with conn.cursor() as cur:
            for i, fpath in enumerate(avg_files):
                group_parsed, sid = parse_averaged_filename(fpath)
                if group_parsed is None:
                    log.warning(f"    Cannot parse averaged: {fpath.name}")
                    continue

                sample_num = int(sid)
                patient_id = f"{group_code} {sample_num}"

                if patient_id not in patients_seen:
                    cur.execute(
                        "INSERT INTO patients (patient_id, disease_category_id, "
                        "group_name, sample_number) "
                        "VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                        (patient_id, disease_cat_id, group_code, sample_num),
                    )
                    patients_seen.add(patient_id)

                try:
                    raman_shift, intensity = read_spectrum(fpath)
                except Exception as e:
                    log.warning(f"    Averaged read failed: {fpath.name}: {e}")
                    continue

                n_pts = len(raman_shift)
                cur.execute(
                    "INSERT INTO sers_spectra "
                    "(patient_id, equipment_type_id, replicate_num, is_averaged, "
                    "data_source, source_file, n_points) "
                    "VALUES (%s, NULL, NULL, TRUE, 'raw_data', %s, %s) RETURNING id",
                    (patient_id, fpath.name, n_pts),
                )
                spectrum_id = cur.fetchone()[0]

                for j in range(n_pts):
                    spectral_buf.append(
                        (spectrum_id, j, float(raman_shift[j]), float(intensity[j]))
                    )
                avg_count += 1

                if (i + 1) % FLUSH_INTERVAL == 0 and spectral_buf:
                    bulk_copy_spectral(cur, spectral_buf)
                    total_points += len(spectral_buf)
                    spectral_buf = []

            if spectral_buf:
                bulk_copy_spectral(cur, spectral_buf)
                total_points += len(spectral_buf)
                spectral_buf = []

        conn.commit()
        total_spectra += avg_count
        log.info(f"    {avg_count} averaged files loaded")

    elapsed = time.time() - t0
    log.info(
        f"Raw data done: {total_spectra:,} spectra, "
        f"{total_points:,} points ({elapsed:.1f}s)"
    )


# ---------------------------------------------------------------------------
# Step 4: Load equipment_test_data
# ---------------------------------------------------------------------------

def load_equipment_data(conn, disease_ids, equipment_ids):
    log.info("=" * 60)
    log.info("STEP 4: Loading equipment_test_data...")
    t0 = time.time()

    config = load_config()
    total_spectra = 0
    total_points = 0

    for eq_folder_name, entry in config.equipment_folder_to_group.items():
        equipment_name = entry.equipment
        eq_type_id = equipment_ids.get(equipment_name)
        if eq_type_id is None:
            log.warning(f"Unknown equipment '{equipment_name}', skipping")
            continue

        eq_dir = SERS_EQUIPMENT_TEST_DATA_DIR / eq_folder_name
        if not eq_dir.is_dir():
            log.warning(f"Equipment folder not found: {eq_dir}")
            continue

        is_handheld = equipment_name == "handheld"

        for sample_folder_name, sample_group in entry.sample_folder_to_group.items():
            sample_dir = eq_dir / sample_folder_name
            if not sample_dir.is_dir():
                log.warning(f"Sample folder not found: {sample_dir}")
                continue

            disease_cat_id = disease_ids.get(sample_group)
            log.info(f"  [{equipment_name}] {sample_folder_name} ({sample_group})")

            # Collect files based on equipment type
            if equipment_name in ("medical", "nanoscope"):
                all_files = collect_txt_files(sample_dir)
            else:
                all_files = collect_csv_files(sample_dir)

            # Filter out non-spectrum files
            all_files = [f for f in all_files if not f.name.startswith("MultiData")]

            patients_seen: set[str] = set()
            spectral_buf: list[tuple] = []
            eq_count = 0

            with conn.cursor() as cur:
                for i, fpath in enumerate(all_files):
                    try:
                        if is_handheld:
                            raman_shift, intensity, name = read_handheld_spectrum(fpath)
                            group, sid, rep = parse_handheld_name(name)
                            if group is None:
                                log.warning(
                                    f"    Cannot parse handheld name from {fpath.name}"
                                )
                                continue
                        else:
                            spec_id = parse_filename(
                                fpath, fallback_group=sample_group
                            )
                            group = spec_id.group
                            sid = spec_id.sample_id
                            rep = spec_id.replicate
                            raman_shift, intensity = read_spectrum(fpath)
                    except Exception as e:
                        log.warning(f"    Failed: {fpath.name}: {e}")
                        continue

                    sample_num = int(sid)
                    patient_id = f"{sample_group} {sample_num}"

                    if patient_id not in patients_seen:
                        cur.execute(
                            "INSERT INTO patients (patient_id, disease_category_id, "
                            "group_name, sample_number) "
                            "VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                            (patient_id, disease_cat_id, sample_group, sample_num),
                        )
                        patients_seen.add(patient_id)

                    n_pts = len(raman_shift)
                    cur.execute(
                        "INSERT INTO sers_spectra "
                        "(patient_id, equipment_type_id, replicate_num, is_averaged, "
                        "data_source, source_file, n_points) "
                        "VALUES (%s, %s, %s, FALSE, 'equipment_test', %s, %s) "
                        "RETURNING id",
                        (patient_id, eq_type_id, rep, fpath.name, n_pts),
                    )
                    spectrum_id = cur.fetchone()[0]

                    for j in range(n_pts):
                        spectral_buf.append(
                            (spectrum_id, j, float(raman_shift[j]), float(intensity[j]))
                        )
                    eq_count += 1

                    if (i + 1) % FLUSH_INTERVAL == 0 and spectral_buf:
                        bulk_copy_spectral(cur, spectral_buf)
                        total_points += len(spectral_buf)
                        spectral_buf = []

                if spectral_buf:
                    bulk_copy_spectral(cur, spectral_buf)
                    total_points += len(spectral_buf)
                    spectral_buf = []

            conn.commit()
            total_spectra += eq_count
            log.info(f"    {eq_count} files loaded")

            # --- Background files ---
            bg_dir = sample_dir / "Background"
            if not bg_dir.is_dir():
                continue

            if equipment_name in ("medical", "nanoscope"):
                bg_files = collect_txt_files(bg_dir)
            else:
                bg_files = collect_csv_files(bg_dir)

            spectral_buf = []
            bg_count = 0

            with conn.cursor() as cur:
                for i, fpath in enumerate(bg_files):
                    try:
                        spec_id = parse_filename(fpath, fallback_group=sample_group)
                        raman_shift, intensity = read_spectrum(fpath)
                    except Exception as e:
                        log.warning(f"    BG read failed: {fpath.name}: {e}")
                        continue

                    sample_num = int(spec_id.sample_id)
                    patient_id = f"{sample_group} {sample_num}"

                    # Ensure patient exists
                    cur.execute(
                        "INSERT INTO patients (patient_id, disease_category_id, "
                        "group_name, sample_number) "
                        "VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                        (patient_id, disease_cat_id, sample_group, sample_num),
                    )

                    n_pts = len(raman_shift)
                    cur.execute(
                        "INSERT INTO sers_spectra "
                        "(patient_id, equipment_type_id, replicate_num, is_averaged, "
                        "data_source, source_file, n_points) "
                        "VALUES (%s, %s, %s, FALSE, 'equipment_test_bg', %s, %s) "
                        "RETURNING id",
                        (patient_id, eq_type_id, spec_id.replicate, fpath.name, n_pts),
                    )
                    spectrum_id = cur.fetchone()[0]

                    for j in range(n_pts):
                        spectral_buf.append(
                            (spectrum_id, j, float(raman_shift[j]), float(intensity[j]))
                        )
                    bg_count += 1

                    if (i + 1) % FLUSH_INTERVAL == 0 and spectral_buf:
                        bulk_copy_spectral(cur, spectral_buf)
                        total_points += len(spectral_buf)
                        spectral_buf = []

                if spectral_buf:
                    bulk_copy_spectral(cur, spectral_buf)
                    total_points += len(spectral_buf)
                    spectral_buf = []

            conn.commit()
            total_spectra += bg_count
            log.info(f"    {bg_count} background files loaded")

    elapsed = time.time() - t0
    log.info(
        f"Equipment data done: {total_spectra:,} spectra, "
        f"{total_points:,} points ({elapsed:.1f}s)"
    )


# ---------------------------------------------------------------------------
# Step 5: Load clinical_data
# ---------------------------------------------------------------------------

# Candidate columns that may contain a patient identifier
_PID_CANDIDATES = [
    "SoluM Label", "SoluM label", "solum label",
    "Patient ID", "patient_id", "ID", "id",
    "Label", "No", "No.", "번호",
]


def _row_to_json(row, columns) -> dict:
    """Convert a DataFrame row to a JSON-serialisable dict."""
    out = {}
    for col in columns:
        val = row[col]
        if pd.isna(val):
            out[str(col)] = None
        elif isinstance(val, (np.integer,)):
            out[str(col)] = int(val)
        elif isinstance(val, (np.floating,)):
            out[str(col)] = float(val)
        elif isinstance(val, pd.Timestamp):
            out[str(col)] = val.isoformat()
        else:
            out[str(col)] = str(val)
    return out


def load_clinical_data(conn, disease_ids):
    log.info("=" * 60)
    log.info("STEP 5: Loading clinical_data...")
    t0 = time.time()

    if not CLINICAL_DATA_DIR.is_dir():
        log.warning(f"Clinical data dir not found: {CLINICAL_DATA_DIR}")
        return

    total_records = 0
    xlsx_files = sorted(CLINICAL_DATA_DIR.glob("*.xlsx"))

    for xlsx_path in xlsx_files:
        filename = xlsx_path.name
        disease_code = CLINICAL_FILE_DISEASE.get(filename)
        log.info(f"  {filename} -> disease={disease_code or '(unmapped)'}")

        try:
            df = pd.read_excel(xlsx_path, engine="openpyxl")
        except Exception as e:
            log.warning(f"    Failed to read: {e}")
            continue

        if df.empty:
            log.info(f"    Empty file, skipping")
            continue

        disease_cat_id = disease_ids.get(disease_code) if disease_code else None

        # Find patient identifier column
        pid_col = None
        for cand in _PID_CANDIDATES:
            if cand in df.columns:
                pid_col = cand
                break

        with conn.cursor() as cur:
            file_count = 0
            for _, row in df.iterrows():
                patient_id = None
                if pid_col is not None and not pd.isna(row[pid_col]):
                    raw_pid = str(row[pid_col]).strip()
                    if disease_code and raw_pid.isdigit():
                        patient_id = f"{disease_code} {int(raw_pid)}"
                    else:
                        patient_id = raw_pid

                    # Ensure patient exists
                    cur.execute(
                        "SELECT 1 FROM patients WHERE patient_id = %s",
                        (patient_id,),
                    )
                    if cur.fetchone() is None:
                        m = re.search(r"(\d+)$", patient_id)
                        sample_num = int(m.group(1)) if m else None
                        cur.execute(
                            "INSERT INTO patients (patient_id, disease_category_id, "
                            "group_name, sample_number) "
                            "VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                            (patient_id, disease_cat_id, disease_code, sample_num),
                        )

                row_json = _row_to_json(row, df.columns)
                cur.execute(
                    "INSERT INTO clinical_info "
                    "(patient_id, disease_category_id, source_file, data) "
                    "VALUES (%s, %s, %s, %s)",
                    (patient_id, disease_cat_id, filename,
                     json.dumps(row_json, ensure_ascii=False)),
                )
                file_count += 1

            total_records += file_count

        conn.commit()
        log.info(f"    {file_count} records loaded")

    elapsed = time.time() - t0
    log.info(f"Clinical data done: {total_records:,} records ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(conn):
    log.info("=" * 60)
    log.info("VERIFICATION:")
    queries = [
        ("disease_categories", "SELECT COUNT(*) FROM disease_categories"),
        ("equipment_types", "SELECT COUNT(*) FROM equipment_types"),
        ("patients", "SELECT COUNT(*) FROM patients"),
        ("sers_spectra", "SELECT COUNT(*) FROM sers_spectra"),
        ("spectral_data", "SELECT COUNT(*) FROM spectral_data"),
        ("clinical_info", "SELECT COUNT(*) FROM clinical_info"),
    ]
    with conn.cursor() as cur:
        for name, query in queries:
            cur.execute(query)
            count = cur.fetchone()[0]
            log.info(f"  {name:25s} {count:>12,} rows")

        cur.execute(
            "SELECT data_source, COUNT(*) FROM sers_spectra "
            "GROUP BY data_source ORDER BY data_source"
        )
        log.info("  sers_spectra by data_source:")
        for source, count in cur.fetchall():
            log.info(f"    {source:25s} {count:>8,}")

        cur.execute(
            "SELECT group_name, COUNT(*) FROM patients "
            "GROUP BY group_name ORDER BY group_name"
        )
        log.info("  patients by group:")
        for group, count in cur.fetchall():
            log.info(f"    {group:25s} {count:>5}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    log.info("Starting SERS database creation...")

    conn = get_connection()
    try:
        create_schema(conn)
        disease_ids, equipment_ids = seed_lookup_tables(conn)
        load_raw_data(conn, disease_ids)
        load_equipment_data(conn, disease_ids, equipment_ids)
        load_clinical_data(conn, disease_ids)
        verify(conn)

        elapsed = time.time() - t_start
        log.info("=" * 60)
        log.info(f"Database creation complete! ({elapsed:.1f}s total)")
    except Exception:
        conn.rollback()
        log.exception("Database creation failed!")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
