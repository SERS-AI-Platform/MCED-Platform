"""
std.clinical_unified 테이블을 CSV + xlsx로 export.

실행:
    python /home/user/SERS-AI/scripts/db/clinical_unified/export_clinical_unified.py

출력:
    /mnt/c/Users/user/Downloads/clinical_unified.csv   (utf-8-sig, 엑셀 한글 안 깨짐)
    /mnt/c/Users/user/Downloads/clinical_unified.xlsx
"""
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

# ---------------------------------------------------------------------
# DB 연결 (환경변수 또는 기본값)
#   환경변수 PGHOST / PGPORT / PGDATABASE / PGUSER / PGPASSWORD 사용 가능
# ---------------------------------------------------------------------
DB_URL = os.environ.get(
    "CLINICAL_DB_URL",
    "postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}".format(
        user=os.environ.get("PGUSER",     "postgres"),
        pw  =os.environ.get("PGPASSWORD", "postgres"),
        host=os.environ.get("PGHOST",     "localhost"),
        port=os.environ.get("PGPORT",     "5432"),
        db  =os.environ.get("PGDATABASE", "postgres"),
    ),
)

OUT_DIR = Path("/mnt/c/Users/user/Downloads")
OUT_DIR.mkdir(parents=True, exist_ok=True)

QUERY = """
    SELECT *
    FROM std.clinical_unified
    ORDER BY disease_group, source_file, source_no
"""


def main() -> None:
    engine = create_engine(DB_URL)
    df = pd.read_sql(QUERY, engine)

    csv_path  = OUT_DIR / "clinical_unified.csv"
    xlsx_path = OUT_DIR / "clinical_unified.xlsx"

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
