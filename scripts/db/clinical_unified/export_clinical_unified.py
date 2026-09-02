"""LEGACY NON-GOVERNED clinical_unified export for historical reproduction.

Exported rows are forbidden as governed identity, matching, label-evidence, or
dataset-manifest input.

실행:
    python /home/user/SERS-AI/scripts/db/clinical_unified/export_clinical_unified.py

출력 (기본값, 환경변수 CLINICAL_UNIFIED_OUT_DIR로 변경 가능):
    /mnt/c/Users/user/Downloads/clinical_unified.csv   (utf-8-sig, 엑셀 한글 안 깨짐)
    /mnt/c/Users/user/Downloads/clinical_unified.xlsx
"""
import os
from importlib import import_module
from pathlib import Path

from sqlalchemy import URL, create_engine
from sqlalchemy.engine import make_url

pd = import_module("pandas")

# ---------------------------------------------------------------------
# DB 연결 (환경변수 또는 기본값)
#   환경변수 PGHOST / PGPORT / PGDATABASE / PGUSER / PGPASSWORD 사용 가능
# ---------------------------------------------------------------------
DEFAULT_OUT_DIR = Path("/mnt/c/Users/user/Downloads")

QUERY = """
    SELECT *
    FROM std.clinical_unified
    ORDER BY disease_group, source_file, source_no
"""


def resolve_db_url() -> URL:
    configured_url = os.environ.get("CLINICAL_DB_URL")
    if configured_url:
        return make_url(configured_url)

    password = os.environ.get("PGPASSWORD")
    if not password:
        raise SystemExit(
            "PostgreSQL password required: set PGPASSWORD or CLINICAL_DB_URL."
        )
    return URL.create(
        "postgresql+psycopg2",
        username=os.environ.get("PGUSER", "postgres"),
        password=password,
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        database=os.environ.get("PGDATABASE", "postgres"),
    )


def resolve_out_dir() -> Path:
    configured_dir = os.environ.get("CLINICAL_UNIFIED_OUT_DIR")
    return Path(configured_dir) if configured_dir else DEFAULT_OUT_DIR


def main() -> None:
    engine = create_engine(resolve_db_url())
    df = pd.read_sql(QUERY, engine)

    out_dir = resolve_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = out_dir / "clinical_unified.csv"
    xlsx_path = out_dir / "clinical_unified.xlsx"

    # CSV: utf-8-sig (엑셀에서 한글 안 깨짐)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # xlsx (pandas → openpyxl)
    df.to_excel(xlsx_path, index=False)

    print(f"rows: {len(df):,}  cols: {len(df.columns)}")
    print(f"CSV : {csv_path}")
    print(f"XLSX: {xlsx_path}")

    # 질환그룹별 요약
    print("\nBy disease_group:")
    print(df.groupby("disease_group").size().to_string())


if __name__ == "__main__":
    main()
