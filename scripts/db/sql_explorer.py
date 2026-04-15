#!/usr/bin/env python3
"""SERS-AI Clinical Data SQL Explorer

DuckDB 기반 인터랙티브 SQL 쿼리 도구.
CSV 파일을 import 없이 바로 SQL로 조회할 수 있습니다.

사용법:
    python scripts/sql_explorer.py              # 인터랙티브 모드
    python scripts/sql_explorer.py -q "SQL"     # 단일 쿼리 실행
    python scripts/sql_explorer.py -o result.csv -q "SQL"  # 결과를 CSV로 저장

등록된 테이블:
    master        - master_clinical.csv (1782행, 전체 임상+검사 데이터)
    standardized  - all_clinical_standardized.csv (표준화 버전)
    summary       - sers_clinical_summary.csv (sample_id, group, hospital)

예시 쿼리:
    SELECT disease_group, count(*) FROM master GROUP BY disease_group;
    SELECT * FROM master WHERE disease_group = 'CRC' AND age > 60 LIMIT 10;
    SELECT disease_group, avg(age), avg(cea) FROM master GROUP BY disease_group;
"""

import argparse
import sys
import readline  # noqa: F401 — enables arrow-key history in input()

import duckdb


def create_connection():
    """CSV 파일들을 VIEW로 등록한 DuckDB 연결을 반환"""
    con = duckdb.connect()

    tables = {
        "master": "data/clinical_data/master_clinical.csv",
        "standardized": "data/clinical_data/standardized/all_clinical_standardized.csv",
        "summary": "data/sers_clinical_summary.csv",
    }

    # 암종별 standardized 파일도 등록
    import glob
    for f in sorted(glob.glob("data/clinical_data/standardized/*_clinical_standardized.csv")):
        name = f.split("/")[-1].replace("_clinical_standardized.csv", "").lower()
        if name == "all":
            continue
        # SQL identifier에 쓸 수 없는 문자(점 등) 치환
        name = name.replace(".", "_")
        tables[name] = f

    for name, path in tables.items():
        try:
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_csv_auto('{path}')")
        except Exception as e:
            print(f"[WARN] {name} ({path}): {e}")

    return con, tables


def run_query(con, sql):
    """쿼리 실행 후 결과를 pandas DataFrame으로 반환"""
    return con.execute(sql).fetchdf()


def print_help(tables):
    print("\n=== SERS-AI SQL Explorer ===")
    print("사용 가능한 테이블:")
    for name in sorted(tables):
        print(f"  {name}")
    print("\n명령어:")
    print("  .tables         - 테이블 목록")
    print("  .schema <table> - 테이블 컬럼 구조")
    print("  .count  <table> - 행 수")
    print("  .head   <table> - 상위 5행")
    print("  .export <file>  - 마지막 쿼리 결과를 CSV로 저장")
    print("  .quit / exit    - 종료")
    print("  help            - 이 도움말")
    print()


def interactive_mode(con, tables):
    """인터랙티브 SQL REPL"""
    print_help(tables)

    last_result = None
    buffer = ""

    while True:
        try:
            prompt = "sql> " if not buffer else " ... "
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not line:
            continue

        # 명령어 처리
        if line.lower() in (".quit", "exit", "quit"):
            print("종료합니다.")
            break

        if line.lower() in ("help", ".help"):
            print_help(tables)
            continue

        if line.lower() == ".tables":
            for name in sorted(tables):
                print(f"  {name}")
            continue

        if line.lower().startswith(".schema"):
            parts = line.split()
            tbl = parts[1] if len(parts) > 1 else "master"
            try:
                df = con.execute(f"DESCRIBE {tbl}").fetchdf()
                print(df.to_string(index=False))
            except Exception as e:
                print(f"Error: {e}")
            continue

        if line.lower().startswith(".count"):
            parts = line.split()
            tbl = parts[1] if len(parts) > 1 else "master"
            try:
                n = con.execute(f"SELECT count(*) FROM {tbl}").fetchone()[0]
                print(f"  {tbl}: {n} rows")
            except Exception as e:
                print(f"Error: {e}")
            continue

        if line.lower().startswith(".head"):
            parts = line.split()
            tbl = parts[1] if len(parts) > 1 else "master"
            try:
                df = con.execute(f"SELECT * FROM {tbl} LIMIT 5").fetchdf()
                print(df.to_string(index=False))
            except Exception as e:
                print(f"Error: {e}")
            continue

        if line.lower().startswith(".export"):
            parts = line.split()
            if last_result is None:
                print("  저장할 결과가 없습니다. 먼저 쿼리를 실행하세요.")
                continue
            fname = parts[1] if len(parts) > 1 else "query_result.csv"
            last_result.to_csv(fname, index=False, encoding="utf-8-sig")
            print(f"  → {fname} 저장 완료 ({len(last_result)} rows)")
            continue

        # SQL 실행 (여러 줄 지원: ;으로 끝나야 실행)
        buffer += " " + line if buffer else line
        if not buffer.rstrip().endswith(";"):
            continue

        sql = buffer.rstrip().rstrip(";")
        buffer = ""

        try:
            last_result = run_query(con, sql)
            if last_result.empty:
                print("  (결과 없음)")
            elif len(last_result) > 50:
                print(last_result.head(50).to_string(index=False))
                print(f"\n  ... 총 {len(last_result)}행 중 상위 50행 표시")
                print("  전체 결과를 CSV로 저장하려면: .export result.csv")
            else:
                print(last_result.to_string(index=False))
            print(f"  ({len(last_result)} rows)")
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="SERS-AI Clinical Data SQL Explorer")
    parser.add_argument("-q", "--query", help="실행할 SQL 쿼리 (비대화형)")
    parser.add_argument("-o", "--output", help="결과를 저장할 CSV 파일")
    args = parser.parse_args()

    con, tables = create_connection()

    if args.query:
        df = run_query(con, args.query)
        if args.output:
            df.to_csv(args.output, index=False, encoding="utf-8-sig")
            print(f"→ {args.output} 저장 ({len(df)} rows)")
        else:
            print(df.to_string(index=False))
        return

    interactive_mode(con, tables)


if __name__ == "__main__":
    main()
