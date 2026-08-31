"""
upload_to_postgres.py — Upload SERS spectral + clinical data to PostgreSQL.

Schema:
    disease_groups  — reference table (11 groups)
    samples         — metadata + clinical info (1342 rows)
    spectra         — spectral intensities as FLOAT[] array (1342 rows)
    wavenumber_grid — shared 901-point wavenumber axis

Usage:
    PGPASSWORD='...' python upload_to_postgres.py
    python upload_to_postgres.py --password mypass    # specify password
    python upload_to_postgres.py --drop               # drop & recreate tables

Prerequisites:
    pip install psycopg2-binary sqlalchemy pandas pyreadr
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import URL, create_engine, text

# ── Setup ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from sers.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Default DB config ─────────────────────────────────────────────────────
# Override via environment variables or CLI args (--host, --password, etc.)
# Env vars: PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
DB_HOST = os.environ.get("PGHOST", "localhost")
DB_PORT = int(os.environ.get("PGPORT", "5432"))
DB_NAME = os.environ.get("PGDATABASE", "sers_clinical")
DB_USER = os.environ.get("PGUSER", "postgres")


def resolve_password(cli_password: str | None) -> str:
    password = cli_password or os.environ.get("PGPASSWORD")
    if not password:
        raise SystemExit(
            "PostgreSQL password required: pass --password or set PGPASSWORD."
        )
    return password


def connect_pg(**kwargs):
    """
    Connect to PostgreSQL with clearer diagnostics on Korean Windows.
    A failed auth can return non-UTF8 localized messages that trigger UnicodeDecodeError.
    """
    try:
        return psycopg2.connect(**kwargs)
    except UnicodeDecodeError:
        host = kwargs.get("host")
        port = kwargs.get("port")
        user = kwargs.get("user")
        dbname = kwargs.get("dbname")
        log.error(
            "PostgreSQL connection failed (Unicode decode error from server message). "
            "This usually means auth/connection failed and Windows locale encoding broke "
            "error decoding."
        )
        log.error("Connection params: host=%s port=%s user=%s dbname=%s", host, port, user, dbname)
        log.error("Check: 1) correct --password  2) PostgreSQL service is running  3) host/port is correct")
        raise SystemExit(1)


# ==========================================================================
# Step 0: Ensure database exists
# ==========================================================================
def ensure_database(host, port, user, password, dbname):
    """Create the database if it doesn't exist."""
    conn = connect_pg(
        host=host, port=port, user=user, password=password, dbname="postgres"
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
    if cur.fetchone() is None:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
        log.info(f"Database '{dbname}' created")
    else:
        log.info(f"Database '{dbname}' already exists")

    cur.close()
    conn.close()


# ==========================================================================
# Step 1: Create schema
# ==========================================================================
DDL = """
-- Disease groups reference table
CREATE TABLE IF NOT EXISTS disease_groups (
    group_code  VARCHAR(10) PRIMARY KEY,
    label       VARCHAR(100) NOT NULL,
    category    VARCHAR(20)  NOT NULL,   -- cancer | non_cancer | control
    hospital    VARCHAR(300),
    protocol    VARCHAR(50),
    n_samples   INTEGER
);

-- Samples: metadata + clinical info
CREATE TABLE IF NOT EXISTS samples (
    sample_id       VARCHAR(20) PRIMARY KEY,
    group_code      VARCHAR(10) NOT NULL REFERENCES disease_groups(group_code),
    sample_num      INTEGER     NOT NULL,
    -- clinical columns
    sex             CHAR(1),
    age             NUMERIC,
    weight          NUMERIC,
    height          NUMERIC,
    bmi             NUMERIC,
    smoking         TEXT,
    drinking        TEXT,
    diagnosis       TEXT,
    diagnosis_date  TEXT,
    stage           VARCHAR(30),
    tnm             VARCHAR(50),
    metastasis      TEXT,
    past_history    TEXT,
    blood_test      TEXT,
    birth_year      INTEGER,
    t_stage         VARCHAR(30),
    n_stage         VARCHAR(30),
    m_stage         VARCHAR(30),
    comorbidity     TEXT,
    surgery_date    TEXT,
    fasting         TEXT,
    collection_date TEXT
);

CREATE INDEX IF NOT EXISTS idx_samples_group ON samples(group_code);
CREATE INDEX IF NOT EXISTS idx_samples_category ON samples(group_code);

-- Spectral data: one row per sample, intensity as array
CREATE TABLE IF NOT EXISTS spectra (
    sample_id   VARCHAR(20) PRIMARY KEY REFERENCES samples(sample_id),
    intensities DOUBLE PRECISION[]   -- 901 elements (400-2200 cm-1, 2 cm-1 step)
);

-- Shared wavenumber grid (901 points)
CREATE TABLE IF NOT EXISTS wavenumber_grid (
    idx        INTEGER PRIMARY KEY,
    wavenumber DOUBLE PRECISION NOT NULL
);
"""

DROP_DDL = """
DROP TABLE IF EXISTS spectra CASCADE;
DROP TABLE IF EXISTS samples CASCADE;
DROP TABLE IF EXISTS disease_groups CASCADE;
DROP TABLE IF EXISTS wavenumber_grid CASCADE;
"""


def create_schema(engine, drop_first=False):
    """Create all tables."""
    with engine.begin() as conn:
        if drop_first:
            log.info("Dropping existing tables...")
            conn.execute(text(DROP_DDL))
        conn.execute(text(DDL))
    log.info("Schema created")


# ==========================================================================
# Step 2: Load data from RDS
# ==========================================================================
def load_rds_data():
    """Load the merged RDS file created by create_rds.py."""
    rds_path = PROJECT_ROOT / "data" / "sers_clinical.rds"
    if not rds_path.exists():
        raise FileNotFoundError(
            f"{rds_path} not found. Run 'python create_rds.py' first."
        )

    import pyreadr
    result = pyreadr.read_r(str(rds_path))
    df = list(result.values())[0]
    log.info(f"Loaded RDS: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


# ==========================================================================
# Step 3: Insert data
# ==========================================================================
def insert_disease_groups(engine, config):
    """Insert disease_groups reference data from config."""
    rows = []
    for group_code, meta in config.display.group_metadata.items():
        if not isinstance(meta, dict):
            continue
        hospital = meta.get("hospital", "")
        if "hospitals" in meta:
            hospital = "; ".join(
                h.get("hospital", "") if isinstance(h, dict) else str(h)
                for h in meta["hospitals"]
            )
        rows.append({
            "group_code": group_code,
            "label": meta.get("label", group_code),
            "category": config.display.category_map.get(group_code, "unknown"),
            "hospital": hospital or None,
            "protocol": meta.get("protocol", meta.get("protocols", [None])[0]
                                 if isinstance(meta.get("protocols"), list) else None),
            "n_samples": meta.get("n_samples"),
        })

    groups_df = pd.DataFrame(rows)

    with engine.begin() as conn:
        # Upsert: delete existing, re-insert
        conn.execute(text("DELETE FROM spectra"))
        conn.execute(text("DELETE FROM samples"))
        conn.execute(text("DELETE FROM disease_groups"))

    groups_df.to_sql("disease_groups", engine, if_exists="append", index=False)
    log.info(f"Inserted {len(groups_df)} disease groups")


def insert_wavenumber_grid(engine):
    """Insert the shared wavenumber grid."""
    grid = np.linspace(400.0, 2200.0, 901)
    grid_df = pd.DataFrame({"idx": range(len(grid)), "wavenumber": grid})

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM wavenumber_grid"))
    grid_df.to_sql("wavenumber_grid", engine, if_exists="append", index=False)
    log.info(f"Inserted {len(grid_df)} wavenumber points")


def insert_samples_and_spectra(engine, df):
    """Insert sample metadata and spectral data."""
    wn_cols = [c for c in df.columns if c.startswith("wn_")]

    # ── Prepare samples table ──
    clinical_columns = [
        "sex", "age", "weight", "height", "bmi",
        "smoking", "drinking", "diagnosis", "diagnosis_date",
        "stage", "tnm", "metastasis", "past_history", "blood_test",
        "birth_year", "t_stage", "n_stage", "m_stage",
        "comorbidity", "surgery_date", "fasting", "collection_date",
    ]

    samples_df = df[["sample_id", "group", "sample_num"]].copy()
    samples_df = samples_df.rename(columns={"group": "group_code"})

    for col in clinical_columns:
        if col in df.columns:
            samples_df[col] = df[col]
        else:
            samples_df[col] = None

    # Convert birth_year to int where possible
    if "birth_year" in samples_df.columns:
        samples_df["birth_year"] = pd.to_numeric(
            samples_df["birth_year"], errors="coerce"
        ).astype("Int64")

    samples_df.to_sql("samples", engine, if_exists="append", index=False)
    log.info(f"Inserted {len(samples_df)} samples")

    # ── Prepare spectra table ──
    # Convert 901 wn_ columns → single FLOAT[] per row
    intensity_matrix = df[wn_cols].values  # (n_samples, 901)

    # Use raw psycopg2 for efficient array insertion
    url = engine.url
    conn = connect_pg(
        host=url.host, port=url.port,
        dbname=url.database, user=url.username, password=url.password,
    )
    cur = conn.cursor()

    insert_sql = "INSERT INTO spectra (sample_id, intensities) VALUES (%s, %s)"
    batch = []
    for i, row in enumerate(df.itertuples()):
        intensities = intensity_matrix[i].tolist()
        # Replace NaN with None for PostgreSQL NULL
        intensities = [None if np.isnan(v) else float(v) for v in intensities]
        batch.append((row.sample_id, intensities))

        if len(batch) >= 100:
            cur.executemany(insert_sql, batch)
            batch.clear()

    if batch:
        cur.executemany(insert_sql, batch)

    conn.commit()
    cur.close()
    conn.close()
    log.info(f"Inserted {len(df)} spectra (901 points each)")


# ==========================================================================
# Step 4: Create useful views
# ==========================================================================
VIEWS_SQL = """
-- Convenient join: sample info + group info (no spectra)
CREATE OR REPLACE VIEW v_samples AS
SELECT
    s.sample_id,
    s.group_code,
    g.label       AS group_label,
    g.category,
    g.hospital,
    s.sample_num,
    s.sex,
    s.age,
    s.weight,
    s.height,
    s.bmi,
    s.smoking,
    s.drinking,
    s.diagnosis,
    s.stage,
    s.tnm,
    s.metastasis
FROM samples s
JOIN disease_groups g ON s.group_code = g.group_code
ORDER BY g.group_code, s.sample_num;

-- Summary: sample counts per group
CREATE OR REPLACE VIEW v_group_summary AS
SELECT
    g.group_code,
    g.label,
    g.category,
    g.hospital,
    COUNT(s.sample_id)                         AS n_samples,
    COUNT(s.sex) FILTER (WHERE s.sex IS NOT NULL)  AS n_with_clinical,
    ROUND(AVG(s.age)::NUMERIC, 1)              AS mean_age,
    COUNT(s.sex) FILTER (WHERE s.sex = 'M')    AS n_male,
    COUNT(s.sex) FILTER (WHERE s.sex = 'F')    AS n_female
FROM disease_groups g
LEFT JOIN samples s ON g.group_code = s.group_code
GROUP BY g.group_code, g.label, g.category, g.hospital
ORDER BY g.category, g.group_code;
"""


def create_views(engine):
    """Create convenience views."""
    with engine.begin() as conn:
        conn.execute(text(VIEWS_SQL))
    log.info("Views created: v_samples, v_group_summary")


# ==========================================================================
# Main
# ==========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Upload SERS data to PostgreSQL")
    parser.add_argument("--host", default=DB_HOST)
    parser.add_argument("--port", type=int, default=DB_PORT)
    parser.add_argument("--dbname", default=DB_NAME)
    parser.add_argument("--user", default=DB_USER)
    parser.add_argument("--password", default=None)
    parser.add_argument("--drop", action="store_true", help="Drop and recreate tables")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info(f"Uploading SERS data to PostgreSQL: {args.host}:{args.port}/{args.dbname}")
    log.info("=" * 60)

    password = resolve_password(args.password)

    # Step 0: Ensure DB exists
    ensure_database(args.host, args.port, args.user, password, args.dbname)

    # Create SQLAlchemy engine
    db_url = URL.create(
        "postgresql",
        username=args.user,
        password=password,
        host=args.host,
        port=args.port,
        database=args.dbname,
    )
    engine = create_engine(db_url)

    # Step 1: Schema
    log.info("\n=== Step 1: Creating schema ===")
    create_schema(engine, drop_first=args.drop)

    # Step 2: Load data
    log.info("\n=== Step 2: Loading RDS data ===")
    config = load_config()
    df = load_rds_data()

    # Step 3: Insert
    log.info("\n=== Step 3: Inserting data ===")
    insert_disease_groups(engine, config)
    insert_wavenumber_grid(engine)
    insert_samples_and_spectra(engine, df)

    # Step 4: Views
    log.info("\n=== Step 4: Creating views ===")
    create_views(engine)

    # Verify
    log.info("\n=== Verification ===")
    with engine.connect() as conn:
        table_queries = [
            ("disease_groups", "SELECT COUNT(*) FROM disease_groups"),
            ("samples", "SELECT COUNT(*) FROM samples"),
            ("spectra", "SELECT COUNT(*) FROM spectra"),
            ("wavenumber_grid", "SELECT COUNT(*) FROM wavenumber_grid"),
        ]
        for table, query in table_queries:
            count = conn.execute(text(query)).scalar()
            log.info(f"  {table}: {count} rows")

        log.info("\n  Group summary:")
        rows = conn.execute(text("SELECT * FROM v_group_summary")).fetchall()
        log.info(f"  {'Group':<6} {'Label':<25} {'Cat':<12} {'N':>5} {'Clin':>5} {'Age':>5} {'M':>4} {'F':>4}")
        log.info(f"  {'-'*75}")
        for r in rows:
            log.info(
                f"  {r.group_code:<6} {r.label:<25} {r.category:<12} "
                f"{r.n_samples:>5} {r.n_with_clinical:>5} "
                f"{r.mean_age or 0:>5.1f} {r.n_male:>4} {r.n_female:>4}"
            )

    log.info(f"\nDone! Connect via pgAdmin4: {args.host}:{args.port}/{args.dbname}")
    log.info("Useful queries:")
    log.info("  SELECT * FROM v_group_summary;")
    log.info("  SELECT * FROM v_samples WHERE category = 'cancer';")
    log.info("  SELECT sample_id, intensities[1:10] FROM spectra LIMIT 5;")


if __name__ == "__main__":
    main()
