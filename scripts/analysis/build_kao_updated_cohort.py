#!/usr/bin/env python3
"""Build the updated KAO-style SERS cohort and clinical summaries.

The cohort starts from the existing preprocessed spectra, keeps the target
clinical groups, and adds spectrum-only fallback samples when the canonical
processed source is missing a subject.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.stacking_utils import preprocess_channel  # noqa: E402
from src.sers.io import read_spectrum  # noqa: E402

DEFAULT_INPUT = PROJECT_ROOT / "results" / "processed_spectra.csv"
DEFAULT_OUT = PROJECT_ROOT / "results" / "kao_20260610_updated_cohort"
DEFAULT_CLINICAL = PROJECT_ROOT / "data" / "clinical_data" / "전체환자_임상정보_정규화.xlsx"
FALLBACK_CLINICAL = (
    PROJECT_ROOT / "data" / "clinical_data" / "standardized" / "all_clinical_standardized.csv"
)

TARGET_SOURCE_GROUPS = (
    "PRO",
    "BRE",
    "OVA",
    "LUN",
    "CRC",
    "BLC",
    "CPAN",
    "YPAN",
    "NOR",
    "DIA",
    "HBP",
    "H.D.",
    "YNOR",
)
SOURCE_GROUP_ORDER = {group: i for i, group in enumerate(TARGET_SOURCE_GROUPS)}
MODEL_GROUP_ALIASES = {"CPAN": "PAN", "YPAN": "PAN", "YNOR": "NOR"}
CANCER_MODEL_GROUPS = {"PRO", "BRE", "OVA", "LUN", "CRC", "BLC", "PAN"}

FALLBACK_SAMPLES = (
    {
        "source_group": "BLC",
        "sample_id": "241",
        "reason": "missing_from_processed_source",
        "files": [
            PROJECT_ROOT
            / "data"
            / "임상데이터"
            / "20260512_Urine test"
            / "11. BLC"
            / f"BLC 241_{rep}.CSV"
            for rep in range(1, 6)
        ],
    },
    {
        "source_group": "YNOR",
        "sample_id": "21",
        "reason": "control_top_up_spectrum_only",
        "files": [
            PROJECT_ROOT
            / "data"
            / "임상데이터"
            / "20260518_Urine test"
            / "6. PRO"
            / f"YNOR 21_{rep}.CSV"
            for rep in range(1, 6)
        ],
    },
)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if str(c).startswith("x_")]
    if not cols:
        raise ValueError("No x_* spectrum columns found")

    def key(col: str) -> tuple[int, float | str]:
        suffix = str(col)[2:]
        try:
            return (0, float(suffix))
        except ValueError:
            return (1, suffix)

    return sorted(cols, key=key)


def _feature_grid(feature_cols: list[str]) -> np.ndarray:
    grid = []
    for col in feature_cols:
        try:
            grid.append(float(str(col)[2:]))
        except ValueError:
            grid = []
            break
    if len(grid) == len(feature_cols):
        return np.asarray(grid, dtype=float)
    return np.linspace(402.0, 2198.0, len(feature_cols))


def _sample_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["group", "sample_id"]].copy()
    out["source_group"] = out["group"].astype(str).str.upper()
    out["sample_id_str"] = out["sample_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    return out


def _has_subject(df: pd.DataFrame, source_group: str, sample_id: str) -> bool:
    keys = _sample_key_frame(df)
    mask = (keys["source_group"] == source_group.upper()) & (
        keys["sample_id_str"] == str(sample_id)
    )
    return bool(mask.any())


def _add_fallback_rows(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = _feature_grid(feature_cols)
    rows: list[dict[str, object]] = []
    additions: list[dict[str, object]] = []
    metadata_cols = [c for c in df.columns if c not in feature_cols]

    for spec in FALLBACK_SAMPLES:
        source_group = spec["source_group"]
        sample_id = spec["sample_id"]
        if _has_subject(df, source_group, sample_id):
            additions.append(
                {
                    "source_group": source_group,
                    "sample_id": sample_id,
                    "status": "already_present",
                    "n_replicates_added": 0,
                    "reason": spec["reason"],
                    "files": "",
                }
            )
            continue

        replicate_count = 0
        used_files = []
        for rep, path in enumerate(spec["files"], start=1):
            if not path.exists():
                raise FileNotFoundError(path)
            x, y = read_spectrum(path)
            y_proc = preprocess_channel(x, y, grid, deriv_order=0)
            row = {col: None for col in metadata_cols}
            row.update({"group": source_group, "sample_id": sample_id, "replicate": rep})
            row.update({col: float(value) for col, value in zip(feature_cols, y_proc)})
            rows.append(row)
            replicate_count += 1
            used_files.append(str(path.relative_to(PROJECT_ROOT)))

        additions.append(
            {
                "source_group": source_group,
                "sample_id": sample_id,
                "status": "added",
                "n_replicates_added": replicate_count,
                "reason": spec["reason"],
                "files": "; ".join(used_files),
            }
        )

    if rows:
        df = pd.concat([df, pd.DataFrame(rows, columns=df.columns)], ignore_index=True)
    return df, pd.DataFrame(additions)


def _sort_cohort(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    metadata_cols = [c for c in df.columns if c not in feature_cols]
    out = df.copy()
    out["_group_sort"] = out["group"].map(lambda g: SOURCE_GROUP_ORDER.get(str(g).upper(), 999))
    out["_sid_sort"] = pd.to_numeric(out["sample_id"], errors="coerce")
    out["_rep_sort"] = pd.to_numeric(out.get("replicate", 0), errors="coerce")
    out = out.sort_values(["_group_sort", "_sid_sort", "sample_id", "_rep_sort"]).drop(
        columns=["_group_sort", "_sid_sort", "_rep_sort"]
    )
    return out[metadata_cols + feature_cols].reset_index(drop=True)


def _model_group(source_group: str) -> str:
    source_group = str(source_group).upper()
    return MODEL_GROUP_ALIASES.get(source_group, source_group)


def _analysis_group(model_group: str) -> str:
    return "Cancer" if model_group in CANCER_MODEL_GROUPS else "Control"


def _normalize_sex(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if text in {"M", "MALE", "남", "남성"}:
        return "M"
    if text in {"F", "FEMALE", "여", "여성"}:
        return "F"
    return None


def _nonempty(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and text.upper() not in {"NAN", "NONE", "UNKNOWN", "NA"}


def _stage_2cat(value: object) -> str:
    if pd.isna(value):
        return "missing"
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE", "UNKNOWN"}:
        return "missing"
    text = re.sub(r"STAGE|병기|기|\s+", "", text)
    text = text.replace("Ⅳ", "IV").replace("Ⅲ", "III").replace("Ⅱ", "II").replace("Ⅰ", "I")
    if text.startswith("IV") or text.startswith("III"):
        return "3-4"
    if text.startswith("II") or text.startswith("I"):
        return "1-2"
    if re.search(r"\b(IV|4)\b", text) or "IV" in text or re.search(r"(^|[^0-9])4([^0-9]|$)", text):
        return "3-4"
    if (
        re.search(r"\b(III|3)\b", text)
        or "III" in text
        or re.search(r"(^|[^0-9])3([^0-9]|$)", text)
    ):
        return "3-4"
    if re.search(r"\b(II|2)\b", text) or "II" in text or re.search(r"(^|[^0-9])2([^0-9]|$)", text):
        return "1-2"
    if re.search(r"\b(I|1)\b", text) or text == "I" or re.search(r"(^|[^0-9])1([^0-9]|$)", text):
        return "1-2"
    return "missing"


def _parse_tnm_num(value: object, prefix: str) -> int | None:
    if not _nonempty(value):
        return None
    text = str(value).strip().upper()
    if text in {f"{prefix}X", f"{prefix}UNKNOWN"}:
        return None
    match = re.match(rf"{prefix}(\d)", text)
    if match:
        return int(match.group(1))
    if prefix == "T" and text == "TIS":
        return 0
    return None


def _infer_stage_from_tnm(row: pd.Series) -> str | None:
    t_num = _parse_tnm_num(row.get("t_stage"), "T")
    n_num = _parse_tnm_num(row.get("n_stage"), "N")
    m_text = str(row.get("m_stage", "")).strip().upper() if _nonempty(row.get("m_stage")) else ""

    if m_text.startswith("M1"):
        return "IV"
    if t_num is None:
        return None
    if t_num == 0:
        return "I"
    if t_num >= 4:
        return "III"
    if n_num is not None and n_num >= 2:
        return "III"
    if t_num == 3 and n_num is not None and n_num >= 1:
        return "III"
    if t_num == 3:
        return "II"
    if t_num <= 2:
        return "II" if n_num is not None and n_num >= 1 else "I"
    return None


def _stage_value_and_source(row: pd.Series) -> tuple[object | None, str]:
    if _nonempty(row.get("stage")):
        return row.get("stage"), "stage"
    inferred = _infer_stage_from_tnm(row)
    if inferred:
        return inferred, "tnm_inferred"
    return None, "missing"


def _clean_bmi(value: object) -> float | None:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return None
    num = float(num)
    if num <= 0 or num > 80:
        return None
    return num


def _row_bmi(row: pd.Series) -> float | None:
    bmi = _clean_bmi(row.get("bmi"))
    if bmi is not None:
        return bmi
    height = pd.to_numeric(row.get("height_cm"), errors="coerce")
    weight = pd.to_numeric(row.get("weight_kg"), errors="coerce")
    if pd.notna(height) and pd.notna(weight) and height > 0 and weight > 0:
        computed = float(weight) / ((float(height) / 100.0) ** 2)
        return _clean_bmi(round(computed, 1))
    return None


def _load_clinical_table(path: Path) -> pd.DataFrame:
    if path.exists() and path.suffix.lower() in {".xlsx", ".xlsm"}:
        return pd.read_excel(path, sheet_name="Sheet1")
    if path.exists():
        return pd.read_csv(path)
    if FALLBACK_CLINICAL.exists():
        print(f"WARNING: clinical file not found: {path}; using {FALLBACK_CLINICAL}")
        return pd.read_csv(FALLBACK_CLINICAL)
    raise FileNotFoundError(path)


def _source_group_from_solum_label(value: object) -> tuple[str | None, str | None]:
    if not _nonempty(value):
        return None, None
    text = str(value).strip().upper()
    text = re.sub(r"\s+", " ", text)
    match = re.match(r"^([A-Z]+(?:\.\s*D\.?)?|OVARIAN)\s*_?\s*(\d+)", text)
    if not match:
        return None, None
    group, sample_id = match.groups()
    group = group.replace(" ", "")
    if group in {"H.D", "H.D."}:
        group = "H.D."
    if group == "OVARIAN":
        group = "OVA"
    return group, str(int(sample_id))


def _clinical_lookup(clinical: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    lookup: dict[tuple[str, str], pd.Series] = {}

    def add(group: str, sample_id: object, row: pd.Series) -> None:
        if pd.isna(sample_id):
            return
        sid = str(sample_id).strip().replace(".0", "")
        if sid:
            lookup[(group, sid)] = row

    if "solum_label" in clinical.columns:
        for _, row in clinical.iterrows():
            group, sample_id = _source_group_from_solum_label(row.get("solum_label"))
            if not group or not sample_id:
                continue
            # The normalized Excel stores pancreatic-control SMCXD2 labels as
            # YPAN, while spectrum files use YNOR for the same control source.
            if group == "YPAN" and str(row.get("group", "")).strip().lower() == "control":
                add("YNOR", sample_id, row)
                continue
            add(group, sample_id, row)
        return lookup

    for group in ["PRO", "LUN", "CRC", "NOR", "DIA", "HBP"]:
        sub = clinical[clinical["disease_group"].eq(group)]
        for _, row in sub.iterrows():
            match = re.search(r"(\d+)$", str(row.get("patient_id", "")))
            if match:
                add(group, match.group(1), row)

    hd = clinical[clinical["disease_group"].eq("H.D.")]
    for _, row in hd.iterrows():
        match = re.search(r"(\d+)$", str(row.get("patient_id", "")))
        if match:
            add("H.D.", match.group(1), row)

    for sers_group, clinical_group in [("BRE", "BRE"), ("OVA", "OVA")]:
        sub = clinical[clinical["disease_group"].eq(clinical_group)].reset_index(drop=True)
        for idx, (_, row) in enumerate(sub.iterrows(), start=1):
            add(sers_group, idx, row)

    pan = clinical[clinical["disease_group"].eq("PAN")]
    for _, row in pan.iterrows():
        pid = str(row.get("patient_id", ""))
        match = re.match(r"^(CPAN|YPAN)\s+(\d+)$", pid.strip(), flags=re.I)
        if match:
            add(match.group(1).upper(), match.group(2), row)

    ynor = clinical[clinical["disease_group"].eq("YNOR")]
    for _, row in ynor.iterrows():
        match = re.search(r"(\d+)$", str(row.get("patient_id", "")))
        if match:
            add("YNOR", match.group(1), row)

    bla = clinical[clinical["disease_group"].eq("BLA")]
    for _, row in bla.iterrows():
        pid = str(row.get("patient_id", ""))
        match = re.match(r"^(1기~2기|3기~4기)-(\d+)$", pid.strip())
        if not match:
            continue
        block, number = match.groups()
        sid = int(number) if block == "1기~2기" else 290 + int(number)
        add("BLC", sid, row)

    return lookup


def _subject_manifest(cohort: pd.DataFrame, clinical_path: Path) -> pd.DataFrame:
    subjects = (
        _sample_key_frame(cohort)
        .drop_duplicates(["source_group", "sample_id_str"])[["source_group", "sample_id_str"]]
        .rename(columns={"sample_id_str": "sample_id"})
        .copy()
    )
    subjects["model_group"] = subjects["source_group"].map(_model_group)
    subjects["analysis_group"] = subjects["model_group"].map(_analysis_group)

    clinical = _load_clinical_table(clinical_path)
    lookup = _clinical_lookup(clinical)

    rows = []
    for _, subject in subjects.iterrows():
        key = (subject["source_group"], subject["sample_id"])
        clin = lookup.get(key)
        base = subject.to_dict()
        base["subject_id"] = f"{subject['source_group']}_{subject['sample_id']}"
        if clin is None:
            base.update(
                {
                    "clinical_match_status": "missing",
                    "clinical_patient_id": None,
                    "clinical_disease_group": None,
                    "age": None,
                    "sex": None,
                    "bmi": None,
                    "stage": None,
                    "stage_source": "missing",
                    "stage_1_2_or_3_4": "missing",
                    "source_file": None,
                }
            )
        else:
            stage_value, stage_source = _stage_value_and_source(clin)
            base.update(
                {
                    "clinical_match_status": "matched",
                    "clinical_patient_id": clin.get("patient_id", clin.get("patient_code")),
                    "clinical_disease_group": clin.get("disease_group", clin.get("group")),
                    "age": pd.to_numeric(clin.get("age"), errors="coerce"),
                    "sex": _normalize_sex(clin.get("sex")),
                    "bmi": _row_bmi(clin),
                    "stage": stage_value,
                    "stage_source": stage_source,
                    "stage_1_2_or_3_4": _stage_2cat(stage_value),
                    "source_file": clin.get("source_file", clin.get("source_cancer_file")),
                }
            )
        rows.append(base)

    out = pd.DataFrame(rows)
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    out["bmi"] = pd.to_numeric(out["bmi"], errors="coerce")
    out["_sample_sort"] = pd.to_numeric(out["sample_id"], errors="coerce")
    out = out.sort_values(
        ["analysis_group", "model_group", "source_group", "_sample_sort", "sample_id"]
    )
    return out.drop(columns="_sample_sort").reset_index(drop=True)


def _counts(cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tmp = _sample_key_frame(cohort)
    tmp["model_group"] = tmp["source_group"].map(_model_group)
    tmp["analysis_group"] = tmp["model_group"].map(_analysis_group)
    tmp["subject_key"] = tmp["source_group"] + "_" + tmp["sample_id_str"]

    source = (
        tmp.groupby("source_group", dropna=False)
        .agg(n_spectra=("sample_id_str", "size"), n_subjects=("sample_id_str", "nunique"))
        .reset_index()
    )
    source["sort"] = source["source_group"].map(
        lambda g: SOURCE_GROUP_ORDER.get(str(g).upper(), 999)
    )
    source = source.sort_values("sort").drop(columns="sort")

    model = (
        tmp.groupby(["analysis_group", "model_group"], dropna=False)
        .agg(n_spectra=("sample_id_str", "size"), n_subjects=("subject_key", "nunique"))
        .reset_index()
        .sort_values(["analysis_group", "model_group"])
    )
    analysis = (
        tmp.groupby("analysis_group", dropna=False)
        .agg(n_spectra=("sample_id_str", "size"), n_subjects=("subject_key", "nunique"))
        .reset_index()
        .sort_values("analysis_group")
    )
    return source, model, analysis


def _summary_table(manifest: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, sub in manifest.groupby(by, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(by, key))
        row["n_subjects"] = int(len(sub))
        row["n_clinical_matched"] = int((sub["clinical_match_status"] == "matched").sum())
        row["n_age"] = int(sub["age"].notna().sum())
        row["age_mean"] = float(sub["age"].mean()) if row["n_age"] else np.nan
        row["age_sd"] = float(sub["age"].std(ddof=1)) if row["n_age"] > 1 else np.nan
        row["age_median"] = float(sub["age"].median()) if row["n_age"] else np.nan
        row["n_bmi"] = int(sub["bmi"].notna().sum())
        row["bmi_mean"] = float(sub["bmi"].mean()) if row["n_bmi"] else np.nan
        row["bmi_sd"] = float(sub["bmi"].std(ddof=1)) if row["n_bmi"] > 1 else np.nan
        row["bmi_median"] = float(sub["bmi"].median()) if row["n_bmi"] else np.nan
        row["sex_m"] = int((sub["sex"] == "M").sum())
        row["sex_f"] = int((sub["sex"] == "F").sum())
        row["sex_missing"] = int(sub["sex"].isna().sum())
        row["stage_1_2"] = int((sub["stage_1_2_or_3_4"] == "1-2").sum())
        row["stage_3_4"] = int((sub["stage_1_2_or_3_4"] == "3-4").sum())
        row["stage_missing_or_na"] = int((sub["stage_1_2_or_3_4"] == "missing").sum())
        row["stage_source_stage"] = int((sub["stage_source"] == "stage").sum())
        row["stage_source_tnm_inferred"] = int((sub["stage_source"] == "tnm_inferred").sum())
        row["stage_source_missing"] = int((sub["stage_source"] == "missing").sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(by).reset_index(drop=True)


def build(input_csv: Path, clinical_csv: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_csv)
    feature_cols = _feature_columns(raw)
    cohort = raw[raw["group"].astype(str).str.upper().isin(TARGET_SOURCE_GROUPS)].copy()
    cohort, additions = _add_fallback_rows(cohort, feature_cols)
    cohort = _sort_cohort(cohort, feature_cols)

    source_counts, model_counts, analysis_counts = _counts(cohort)
    manifest = _subject_manifest(cohort, clinical_csv)
    summary_by_model = _summary_table(manifest, ["analysis_group", "model_group"])
    summary_by_source = _summary_table(manifest, ["analysis_group", "source_group"])
    summary_overall = _summary_table(manifest.assign(cohort="all"), ["cohort"])
    summary_by_analysis = _summary_table(manifest, ["analysis_group"])

    cohort.to_csv(out_dir / "processed_spectra.csv", index=False, encoding="utf-8-sig")
    additions.to_csv(out_dir / "fallback_spectrum_additions.csv", index=False, encoding="utf-8-sig")
    source_counts.to_csv(
        out_dir / "cohort_counts_by_source_group.csv", index=False, encoding="utf-8-sig"
    )
    model_counts.to_csv(
        out_dir / "cohort_counts_by_model_group.csv", index=False, encoding="utf-8-sig"
    )
    analysis_counts.to_csv(
        out_dir / "cohort_counts_by_analysis_group.csv", index=False, encoding="utf-8-sig"
    )
    manifest.to_csv(out_dir / "cohort_subject_manifest.csv", index=False, encoding="utf-8-sig")
    summary_by_model.to_csv(
        out_dir / "clinical_summary_by_model_group.csv", index=False, encoding="utf-8-sig"
    )
    summary_by_source.to_csv(
        out_dir / "clinical_summary_by_source_group.csv", index=False, encoding="utf-8-sig"
    )
    summary_by_analysis.to_csv(
        out_dir / "clinical_summary_by_analysis_group.csv", index=False, encoding="utf-8-sig"
    )
    summary_overall.to_csv(
        out_dir / "clinical_summary_overall.csv", index=False, encoding="utf-8-sig"
    )

    try:
        with pd.ExcelWriter(out_dir / "cohort_powerbi_tables.xlsx", engine="openpyxl") as writer:
            source_counts.to_excel(writer, sheet_name="source_counts", index=False)
            model_counts.to_excel(writer, sheet_name="model_counts", index=False)
            analysis_counts.to_excel(writer, sheet_name="analysis_counts", index=False)
            manifest.to_excel(writer, sheet_name="subject_manifest", index=False)
            summary_by_model.to_excel(writer, sheet_name="clinical_model", index=False)
            summary_by_source.to_excel(writer, sheet_name="clinical_source", index=False)
            additions.to_excel(writer, sheet_name="fallback_additions", index=False)
    except Exception as exc:
        print(f"WARNING: Excel workbook was not written: {exc}")

    print(f"Output: {out_dir}")
    print("\nSource group counts:")
    print(source_counts.to_string(index=False))
    print("\nModel group counts:")
    print(model_counts.to_string(index=False))
    print("\nClinical summary by model group:")
    print(summary_by_model.to_string(index=False))
    print("\nFallback additions:")
    print(additions.to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--clinical-csv", type=Path, default=DEFAULT_CLINICAL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(args.input_csv, args.clinical_csv, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
