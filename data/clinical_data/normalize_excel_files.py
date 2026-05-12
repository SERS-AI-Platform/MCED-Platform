"""Normalize / clean Excel clinical files into consistent CSV outputs.

This script is intended to help you prepare the Excel-based datasets for ingestion into a database.
It will:
  - Discover all Excel files (.xlsx/.xlsm/.xls) under a base folder
  - Attempt to detect the best header row (to avoid merged header rows)
  - Normalize column names (whitespace + special characters -> underscore)
  - Drop empty columns/rows and trim string values
  - Save cleaned output as CSV in an output folder (keeping source folder structure)

Usage:
    python normalize_excel_files.py --base <base_folder> --out <output_folder>

Example:
    python normalize_excel_files.py --base "./clinical_data" --out "./clinical_data_normalized"

Notes:
- If a file contains multiple sheets, only the first sheet is used.
- The algorithm to find the best header row is heuristic; verify results for complex files.
"""

import argparse
import os
from pathlib import Path
import re

import pandas as pd


def load_column_mapping(mapping_path: Path) -> dict[str, str]:
    """Load column name mapping from the provided Excel file.

    The mapping file should contain a source column name and a target column name.
    This function attempts to auto-detect the correct columns by looking for keywords
    such as '원본', '항목', '표준', and 'Target'.
    """
    df = pd.read_excel(mapping_path)
    cols = list(df.columns)

    source_col = None
    target_col = None

    # Prefer an explicit "source" column name, but allow flexible matching.
    for c in cols:
        if any(k in str(c) for k in ["원본", "항목", "컬럼", "source"]):
            source_col = c
            break
    for c in cols:
        if any(k in str(c) for k in ["표준", "Target", "target"]):
            target_col = c
            break

    if source_col is None or target_col is None:
        raise ValueError(f"Could not automatically detect source/target columns in mapping file. Columns found: {cols}")

    mapping = {}
    for _, row in df.iterrows():
        src = row.get(source_col)
        tgt = row.get(target_col)
        if pd.isna(src) or pd.isna(tgt):
            continue
        mapping[str(src).strip()] = str(tgt).strip()
    return mapping


def normalize_col(col: str) -> str:
    if pd.isna(col):
        return ""
    out = str(col).strip()
    out = re.sub(r"[\r\n\t]+", " ", out)
    out = re.sub(r"\s+", " ", out)

    # Keep unicode word characters (including Korean) and digits.
    # Replace other characters with underscore.
    out = re.sub(r"[^\w]+", "_", out, flags=re.UNICODE)

    # Trim leading/trailing underscores
    out = out.strip("_")
    if not out:
        return ""

    if re.match(r"^[0-9]", out):
        out = "c_" + out
    return out


def _is_number_like(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def choose_header_row(path: Path, max_header_row: int = 5) -> int:
    """Pick the best row to use as header by preferring string-like values."""
    df_try = pd.read_excel(path, header=None, nrows=max_header_row + 5, engine="openpyxl")

    best_score = float('-inf')
    best_row = 0

    for row_idx in range(min(max_header_row, len(df_try))):
        row = df_try.iloc[row_idx]
        # Convert to strings and normalize whitespace
        values = [str(x).strip() if not pd.isna(x) else '' for x in row]

        # Compute heuristics
        non_empty = [v for v in values if v]
        num_nonempty = len(non_empty)
        num_numeric = sum(1 for v in non_empty if _is_number_like(v))
        num_dates = sum(1 for v in non_empty if re.match(r"\d{4}[-/.]", v))
        num_korean = sum(1 for v in non_empty if re.search(r"[\uac00-\ud7a3]", v))

        # Score: prefer rows with many strings (esp Korean), penalize numeric-heavy rows
        score = num_nonempty + (num_korean * 2) - (num_numeric * 2) - (num_dates)

        if score > best_score:
            best_score = score
            best_row = row_idx

    return best_row


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize column names
    cols = []
    seen = {}
    for col in df.columns:
        norm = normalize_col(col)
        if not norm:
            norm = "col"
        count = seen.get(norm, 0)
        if count:
            norm = f"{norm}_{count+1}"
        seen[norm] = count + 1
        cols.append(norm)
    df.columns = cols

    # Trim whitespace from string values and normalize NaNs
    def _strip_series(s: pd.Series) -> pd.Series:
        if s.dtype == object:
            def _strip_val(x):
                if isinstance(x, str):
                    v = x.strip()
                    return None if v.lower() == "nan" else v
                return x
            return s.map(_strip_val)
        return s

    df = df.apply(_strip_series)

    # Drop fully-empty columns/rows
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")

    return df


def _normalize_header_part(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    # pandas will use "Unnamed: X" for empty header cells; treat those as empty
    if s.startswith("Unnamed"):
        return ""
    return s


def read_excel_with_multiheader(path: Path, header_row: int) -> pd.DataFrame:
    """Try reading Excel with a multi-row header (row N and N+1)."""
    try:
        df = pd.read_excel(path, header=[header_row, header_row + 1], engine="openpyxl")
        # If second header row is all NaN/Unnamed, fall back to single header
        if df.columns.nlevels == 2:
            second_level = df.columns.get_level_values(1)
            normalized = [_normalize_header_part(x) for x in second_level]
            if all(not v for v in normalized):
                raise ValueError("No meaningful second header")

        # Flatten multiindex
        new_cols = []
        for a, b in df.columns:
            a = _normalize_header_part(a)
            b = _normalize_header_part(b)
            new = " - ".join(p for p in [a, b] if p)
            new_cols.append(new)
        df.columns = new_cols
        return df
    except Exception:
        # Fallback to single header row
        return pd.read_excel(path, header=header_row, engine="openpyxl")


def apply_column_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename DataFrame columns according to a mapping.

    This will attempt:
      1) exact match
      2) normalization-based match (e.g., strip punctuation/whitespace)
      3) fuzzy match by substring

    If multiple matches conflict, the first match wins.
    """

    def _normalize_for_match(x: str) -> str:
        if pd.isna(x):
            return ""
        return normalize_col(x).lower()

    normalized_map = {_normalize_for_match(k): k for k in mapping.keys()}

    rename_map = {}
    for col in df.columns:
        if col in mapping:
            rename_map[col] = mapping[col]
            continue

        col_norm = _normalize_for_match(col)
        if not col_norm:
            continue

        # exact normalized match
        if col_norm in normalized_map:
            rename_map[col] = mapping[normalized_map[col_norm]]
            continue

        # fuzzy: try substring matches
        for src_norm, src in normalized_map.items():
            if src_norm and (col_norm in src_norm or src_norm in col_norm):
                rename_map[col] = mapping[src]
                break

    # Heuristic: if a column has no name or generic name and contains "PRO <num>" values,
    # map it to a standard patient id column if the mapping includes an ID target.
    id_target = None
    for k, v in mapping.items():
        if "id" in str(k).lower():
            id_target = v
            break

    if id_target:
        for col in df.columns:
            if col in rename_map:
                continue
            col_norm = _normalize_for_match(col)
            if not col_norm or col_norm.startswith("col") or col_norm.startswith("unnamed"):
                # Check if the first non-null value looks like a patient ID
                first_vals = df[col].dropna().astype(str)
                if not first_vals.empty and first_vals.iloc[0].startswith("PRO"):
                    rename_map[col] = id_target

    if rename_map:
        df = df.rename(columns=rename_map)

    return df

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def normalize_all_excels(
    base_dir: Path,
    out_dir: Path,
    out_format: str = "xlsx",
    dry_run: bool = False,
    mapping: dict[str, str] | None = None,
):
    base_dir = base_dir.resolve()
    out_dir = out_dir.resolve()

    for root, _, files in os.walk(base_dir):
        for f in files:
            if not f.lower().endswith((".xlsx", ".xlsm", ".xls")):
                continue

            src_path = Path(root) / f
            rel = src_path.relative_to(base_dir)
            dst_path = out_dir / rel.with_suffix(f".{out_format}")

            header_row = choose_header_row(src_path)
            try:
                df = read_excel_with_multiheader(src_path, header_row)
            except Exception as e:
                print(f"[SKIP] {rel} - read error: {e}")
                continue

            if mapping:
                df = apply_column_mapping(df, mapping)

            df = clean_dataframe(df)

            if dry_run:
                print(f"[DRY] {rel} -> {dst_path} (rows={len(df)}, cols={len(df.columns)})")
                continue

            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if out_format.lower() == "csv":
                df.to_csv(dst_path, index=False, encoding="utf-8-sig")
            else:
                df.to_excel(dst_path, index=False, engine="openpyxl")
            print(f"[WRITE] {rel} -> {dst_path} (rows={len(df)}, cols={len(df.columns)})")


def main():
    parser = argparse.ArgumentParser(description="Normalize Excel files into clean CSV/XLSX outputs")
    parser.add_argument("--base", default=".", help="Base folder containing Excel clinical files")
    parser.add_argument("--out", default="./normalized", help="Output folder for cleaned files")
    parser.add_argument("--format", choices=["csv", "xlsx"], default="xlsx", help="Output file format")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing files")
    parser.add_argument(
        "--mapping",
        default=None,
        help="Optional path to an Excel file containing source->target column mappings",
    )
    args = parser.parse_args()

    mapping = None
    if args.mapping:
        mapping = load_column_mapping(Path(args.mapping))

    normalize_all_excels(
        Path(args.base),
        Path(args.out),
        out_format=args.format,
        dry_run=args.dry_run,
        mapping=mapping,
    )


if __name__ == "__main__":
    main()
