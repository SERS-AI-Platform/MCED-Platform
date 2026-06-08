#!/usr/bin/env python3
"""Evaluate the current STK-V2 model on pooled and lot-balanced acquisitions.

The script uses the production StackingPredictor preprocessing path, including
Medical-to-Thermo PDS transfer for Medical spectra. Replicates are preprocessed
first, averaged to subject/sample-level feature tensors, and then scored by the
current usersnet artifact.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.deployment.sers_predict import StackingPredictor
from scripts.training.train_usersnet import extract_peak_features
from src.sers.io import parse_filename, read_spectrum
from sers.models.usersnet.stacking import apply_sex_constraint


DATA_ROOT = Path("/Users/ian/Downloads/data/organized_data")
DEFAULT_OUT = PROJECT_ROOT / "results" / "evaluation" / "current_model_acquisition_sets"

CANCER_GROUPS = {"PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC", "BPRO"}
NON_CANCER_GROUPS = {"NOR", "DIA", "HBP", "H.D.", "BNOR"}
VALID_BINARY_GROUPS = CANCER_GROUPS | NON_CANCER_GROUPS

TYPE_GROUP_MAP = {
    "PRO": "PRO",
    "BPRO": "PRO",
    "LUN": "LUN",
    "CRC": "CRC",
    "PAN": "PAN",
    "OVA": "OVA",
    "BRE": "BRE",
    "BLC": "BLC",
}

GROUP_ALIASES = {
    "CPAN": "PAN",
    "YPAN": "PAN",
    "YNOR": "NOR",
    "H.D": "H.D.",
    "HD": "H.D.",
}

FOLDER_TO_GROUP = {
    "1. Prostate cancer (100개)": "PRO",
    "1. NOR": "NOR",
    "2. Breast cancer (30개)": "BRE",
    "2. HBP": "HBP",
    "2. PRO": "PRO",
    "3. DIA": "DIA",
    "3. HBP": "HBP",
    "3. Ovarian cancer (70개)": "OVA",
    "4. DIA": "DIA",
    "4. H.D": "H.D.",
    "4. Lung cancer (300개)": "LUN",
    "5. H.D": "H.D.",
    "5. Normal (100개)": "NOR",
    "5. YNOR": "YNOR",
    "6. CRC": "CRC",
    "6. Diabetes (100개)": "DIA",
    "6. PRO": "PRO",
    "7. High blood pressure (100개)": "HBP",
    "7. OVA": "OVA",
    "8. BLC": "BLC",
    "8. H.D": "H.D.",
    "8. High blood pressure + Diabetes (100개)": "H.D.",
    "8. LUN": "LUN",
    "9. Colorectal cancer (300개)": "CRC",
    "9. PAN": "PAN",
    "10. CRC": "CRC",
    "10-1. C-Pancreatic cancer (70개)": "CPAN",
    "10-2. S-Pancreatic cancer (72개)": "SPAN",
    "10-3. Y-Pancreatic cancer (27개)": "YPAN",
    "10-3. Y-Pancreatic cancer (30개)": "YPAN",
    "10-3. Y-Pancreatic cancer (YPAN)": "YPAN",
    "11 BLC (299개)": "BLC",
    "11. Bladdder Cancer (299개)": "BLC",
    "11. BLC": "BLC",
    "12. Breast cancer (30개)": "BRE",
    "12. BRE": "BRE",
    "12. Y-Normal (29개)": "YNOR",
    "12. Y-Normal (YNOR)": "YNOR",
    "BNOR": "BNOR",
    "BPRO": "BPRO",
}


@dataclass(frozen=True)
class SpectrumRecord:
    source: str
    instrument: str
    source_group: str
    group: str
    sample_id: str
    sample_key: str
    replicate: int
    path: Path

    @property
    def uid(self) -> str:
        return f"{self.source}|{self.instrument}|{self.group}|{self.sample_key}"


def normalize_group(group: str) -> str:
    g = re.sub(r"\s+", "", str(group).upper())
    g = GROUP_ALIASES.get(g, g)
    if g == "H.D":
        g = "H.D."
    return g


def normalize_source_group(group: str) -> str:
    g = re.sub(r"\s+", "", str(group).upper())
    if g == "H.D":
        g = "H.D."
    return g


def sample_key_for_alias(source_group: str, sample_id: str) -> str:
    source_group = normalize_source_group(source_group)
    if source_group in {"YPAN", "YNOR"}:
        return f"{source_group}_{sample_id}"
    return str(sample_id)


def infer_group_from_path(path: Path) -> str | None:
    for part in reversed(path.parent.parts):
        if part in FOLDER_TO_GROUP:
            return FOLDER_TO_GROUP[part]
        upper = normalize_group(part)
        if upper in VALID_BINARY_GROUPS or upper in {"CPAN", "YPAN", "YNOR", "SPAN"}:
            return upper
    return None


def should_skip_file(path: Path, include_background_only: bool) -> bool:
    name = path.name.lower()
    stem = path.stem.lower()
    parts = [p.lower() for p in path.parts]

    if path.suffix.lower() not in {".csv", ".txt"}:
        return True
    if "_ave" in stem or "zone.identifier" in name or "multidata" in stem:
        return True
    if re.search(r"(?i)(^|[\s_/.-])nf($|[\s_.-])", path.name):
        return True
    if re.search(r"(?i)(^|[\s_/.-])po\.?\s+", path.name):
        return True
    if any("_prediction_results" in p for p in parts):
        return True
    if any("before_centri" in p or "before_centrifuge" in p for p in parts):
        return True
    if any(p in {"0. reference", "0. blank", "0. ps", "0. si", "0. mb"} for p in parts):
        return True

    in_background = "background" in parts
    if include_background_only and not in_background:
        return True
    if not include_background_only and in_background:
        return True
    return False


PRIMARY_MEDICAL_CANONICAL_FOLDERS = {
    "1. Prostate cancer (100개)",
    "2. Breast cancer (30개)",
    "3. Ovarian cancer (70개)",
    "4. Lung cancer (300개)",
    "5. Normal (100개)",
    "6. Diabetes (100개)",
    "7. High blood pressure (100개)",
    "8. High blood pressure + Diabetes (100개)",
    "9. Colorectal cancer (300개)",
    "10-1. C-Pancreatic cancer (70개)",
    "10-3. Y-Pancreatic cancer (30개)",
    "11. Bladdder Cancer (299개)",
    "12. Y-Normal (29개)",
}


def matched_group_folder(path: Path) -> str | None:
    for part in reversed(path.parent.parts):
        if part in FOLDER_TO_GROUP:
            return part
    return None


def collect_records(
    root: Path,
    source: str,
    instrument: str,
    include_background_only: bool = False,
    allowed_group_folders: set[str] | None = None,
) -> list[SpectrumRecord]:
    records: list[SpectrumRecord] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or should_skip_file(path, include_background_only):
            continue
        group_folder = matched_group_folder(path)
        if allowed_group_folders is not None and group_folder not in allowed_group_folders:
            continue
        fallback = infer_group_from_path(path)
        if fallback is None:
            continue
        try:
            sid = parse_filename(path, fallback_group=fallback)
        except ValueError:
            continue
        source_group = normalize_source_group(sid.group or fallback)
        if source_group == "SPAN":
            continue
        group = normalize_group(source_group)
        group = normalize_group(group)
        if group == "NOR" and int(sid.replicate) > 6:
            continue
        if group not in VALID_BINARY_GROUPS and source_group not in {"CPAN", "YPAN", "YNOR"}:
            continue
        group = normalize_group(group)
        if group not in VALID_BINARY_GROUPS:
            continue
        sample_id = str(sid.sample_id)
        records.append(
            SpectrumRecord(
                source=source,
                instrument=instrument,
                source_group=source_group,
                group=group,
                sample_id=sample_id,
                sample_key=sample_key_for_alias(source_group, sample_id),
                replicate=int(sid.replicate),
                path=path,
            )
        )
    return records


def build_eval_sources() -> list[tuple[str, str, Path, bool, set[str] | None]]:
    return [
        (
            "primary_pooled_thermo",
            "thermo",
            DATA_ROOT / "02_sers_primary_pooled_acquisition" / "thermo" / "raw_data",
            False,
            None,
        ),
        (
            "primary_pooled_medical",
            "medical",
            DATA_ROOT / "02_sers_primary_pooled_acquisition" / "medical" / "raw_data_medical",
            True,
            PRIMARY_MEDICAL_CANONICAL_FOLDERS,
        ),
        (
            "balanced_lot_thermo_standard",
            "thermo",
            DATA_ROOT / "03_sers_date_lot_balanced_acquisition" / "thermo" / "임상데이터",
            False,
            None,
        ),
        (
            "balanced_lot_medical_all",
            "medical",
            DATA_ROOT / "03_sers_date_lot_balanced_acquisition" / "medical" / "Medical",
            False,
            None,
        ),
        (
            "balanced_lot_thermo_boramae_20260602",
            "thermo",
            DATA_ROOT
            / "03_sers_date_lot_balanced_acquisition"
            / "thermo"
            / "Thermo"
            / "20260602_Urine test",
            False,
            None,
        ),
    ]


def preprocess_to_samples(
    predictor: StackingPredictor,
    records: list[SpectrumRecord],
) -> tuple[np.ndarray, pd.DataFrame, list[dict[str, str]]]:
    by_uid: dict[str, list[np.ndarray]] = defaultdict(list)
    meta: dict[str, dict[str, object]] = {}
    errors: list[dict[str, str]] = []

    for rec in tqdm(records, desc="preprocess", unit="spectrum"):
        try:
            x, y = read_spectrum(rec.path)
            multichannel = predictor._preprocess_multichannel(x, y, instrument=rec.instrument)
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": str(rec.path), "error": str(exc)})
            continue
        by_uid[rec.uid].append(multichannel.astype(np.float32))
        meta.setdefault(
            rec.uid,
            {
                "uid": rec.uid,
                "source": rec.source,
                "instrument": rec.instrument,
                "source_group": rec.source_group,
                "group": rec.group,
                "sample_id": rec.sample_id,
                "sample_key": rec.sample_key,
            },
        )

    rows = []
    tensors = []
    for uid in sorted(by_uid):
        tensors.append(np.stack(by_uid[uid], axis=0).mean(axis=0))
        row = dict(meta[uid])
        row["n_replicates"] = len(by_uid[uid])
        rows.append(row)

    if not tensors:
        return np.empty((0, 3, len(predictor.grid))), pd.DataFrame(), errors
    return np.stack(tensors, axis=0), pd.DataFrame(rows), errors


def predict_samples(
    predictor: StackingPredictor,
    X: np.ndarray,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    n = len(X)
    n_base = len(predictor.base_models)
    n_classes = len(predictor.cancer_types)

    X_peak = extract_peak_features(X[:, 0, :], predictor.grid)
    base_s1 = np.zeros((n, n_base))
    base_s2 = np.zeros((n, n_base, n_classes))

    for bi, (name, bm) in enumerate(predictor.base_models.items()):
        channels = bm["channels"]
        if channels == "peak":
            Xin = X_peak
        else:
            Xin = X[:, channels, :].reshape(n, -1)

        base_s1[:, bi] = bm["s1"].predict_proba(Xin)[:, 1]
        raw_s2 = bm["s2"].predict_proba(Xin)
        classes = getattr(bm["s2"], "classes_", None)
        if classes is None and hasattr(bm["s2"], "__getitem__"):
            classes = getattr(bm["s2"][-1], "classes_", np.arange(raw_s2.shape[1]))
        for i, cls in enumerate(classes):
            cls = int(cls)
            if cls < n_classes:
                base_s2[:, bi, cls] = raw_s2[:, i]

    meta_s1 = base_s1
    meta_all = np.hstack([base_s1, base_s2.reshape(n, -1)])
    cancer_prob = predictor.meta_s1.predict_proba(meta_s1)[:, 1]
    type_prob = predictor.meta_s2.predict_proba(meta_all)

    s2_classes = getattr(predictor.meta_s2, "classes_", None)
    if s2_classes is None and hasattr(predictor.meta_s2, "__getitem__"):
        s2_classes = getattr(predictor.meta_s2[-1], "classes_", np.arange(type_prob.shape[1]))
    s2_full = np.zeros((n, n_classes))
    for i, cls in enumerate(s2_classes):
        cls = int(cls)
        if cls < n_classes:
            s2_full[:, cls] = type_prob[:, i]

    sex = np.full(n, np.nan)
    groups = meta["group"].astype(str).to_numpy()
    sex[np.isin(groups, ["PRO", "BPRO"])] = 1.0
    sex[groups == "OVA"] = 0.0
    s2_full = apply_sex_constraint(s2_full, predictor.cancer_types, sex)

    pred_type_idx = s2_full.argmax(axis=1)
    pred_type = [predictor.cancer_types[i] for i in pred_type_idx]

    out = meta.copy()
    out["true_binary"] = out["group"].isin(CANCER_GROUPS).astype(int)
    out["cancer_probability"] = cancer_prob
    threshold = predictor.operating_modes.get("balanced", {}).get("threshold", 0.60)
    out["balanced_threshold"] = float(threshold)
    out["pred_binary_balanced"] = (cancer_prob > threshold).astype(int)
    out["pred_type"] = pred_type
    out["true_type"] = out["group"].map(TYPE_GROUP_MAP)
    for i, ct in enumerate(predictor.cancer_types):
        out[f"type_prob_{ct}"] = s2_full[:, i]
    return out


def score_frame(df: pd.DataFrame, subset: str) -> dict[str, object]:
    valid = df[df["group"].isin(VALID_BINARY_GROUPS)].copy()
    if valid.empty:
        return {"subset": subset, "n": 0}

    y = valid["true_binary"].to_numpy()
    p = valid["cancer_probability"].to_numpy()
    pred = valid["pred_binary_balanced"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")

    cancer = valid[valid["true_binary"].eq(1) & valid["true_type"].notna()].copy()
    if cancer.empty:
        type_acc = float("nan")
        type_f1 = float("nan")
        n_type = 0
    else:
        type_acc = float(accuracy_score(cancer["true_type"], cancer["pred_type"]))
        type_f1 = float(f1_score(cancer["true_type"], cancer["pred_type"], average="macro", zero_division=0))
        n_type = int(len(cancer))

    return {
        "subset": subset,
        "n": int(len(valid)),
        "n_cancer": int(y.sum()),
        "n_non_cancer": int((1 - y).sum()),
        "threshold": float(valid["balanced_threshold"].iloc[0]),
        "auc": auc,
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(recall_score(1 - y, 1 - pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_cancer": float(f1_score(y, pred, zero_division=0)),
        "precision_cancer": float(precision_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "stage2_n": n_type,
        "stage2_type_accuracy": type_acc,
        "stage2_type_f1_macro": type_f1,
    }


def build_scores(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, g in pred_df.groupby("source"):
        rows.append(score_frame(g, source))

    pooled_defs = {
        "primary_pooled_2_instruments": pred_df["source"].str.startswith("primary_pooled_"),
        "balanced_lot_medical_standard_only": pred_df["source"].eq("balanced_lot_medical_all")
        & ~pred_df["group"].isin(["BPRO", "BNOR"]),
        "balanced_lot_standard_2_instruments": pred_df["source"].isin(
            ["balanced_lot_thermo_standard", "balanced_lot_medical_all"]
        )
        & ~pred_df["group"].isin(["BPRO", "BNOR"]),
        "balanced_lot_medical_boramae_only": pred_df["source"].eq("balanced_lot_medical_all")
        & pred_df["group"].isin(["BPRO", "BNOR"]),
        "balanced_lot_thermo_boramae_only": pred_df["source"].eq("balanced_lot_thermo_boramae_20260602"),
        "balanced_lot_boramae_2_instruments": pred_df["group"].isin(["BPRO", "BNOR"]),
        "balanced_lot_all": pred_df["source"].str.startswith("balanced_lot_"),
    }
    for name, mask in pooled_defs.items():
        rows.append(score_frame(pred_df[mask], name))
    return pd.DataFrame(rows)


def group_summary(pred_df: pd.DataFrame) -> pd.DataFrame:
    return (
        pred_df.groupby(["source", "instrument", "group"], dropna=False)
        .agg(
            n=("uid", "size"),
            n_replicates_median=("n_replicates", "median"),
            mean_prob=("cancer_probability", "mean"),
            sd_prob=("cancer_probability", "std"),
            detection_rate=("pred_binary_balanced", "mean"),
        )
        .reset_index()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "usersnet" / "current")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictor = StackingPredictor(args.model_dir)

    all_records: list[SpectrumRecord] = []
    collection_rows = []
    for source, instrument, root, include_background_only, allowed_group_folders in build_eval_sources():
        records = collect_records(
            root,
            source,
            instrument,
            include_background_only=include_background_only,
            allowed_group_folders=allowed_group_folders,
        )
        all_records.extend(records)
        collection_rows.append(
            {
                "source": source,
                "instrument": instrument,
                "root": str(root),
                "n_spectra": len(records),
                "n_samples": len({r.uid for r in records}),
            }
        )

    pd.DataFrame(collection_rows).to_csv(args.output_dir / "collection_summary.csv", index=False)
    if not all_records:
        raise RuntimeError("No spectra collected")

    X, meta, errors = preprocess_to_samples(predictor, all_records)
    pred_df = predict_samples(predictor, X, meta)
    scores = build_scores(pred_df)
    groups = group_summary(pred_df)

    pred_df.to_csv(args.output_dir / "per_sample_predictions.csv", index=False)
    scores.to_csv(args.output_dir / "performance_summary.csv", index=False)
    groups.to_csv(args.output_dir / "per_group_summary.csv", index=False)
    pd.DataFrame(errors).to_csv(args.output_dir / "preprocess_errors.csv", index=False)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(args.model_dir),
        "model": predictor.manifest.get("model_type"),
        "mode": "balanced",
        "threshold": predictor.operating_modes.get("balanced", {}).get("threshold", 0.60),
        "collection": collection_rows,
        "n_preprocess_errors": len(errors),
        "scores": scores.to_dict(orient="records"),
        "notes": [
            "Replicates are averaged in preprocessed 3-channel feature space before prediction.",
            "Medical spectra use the production Medical-to-Thermo PDS path.",
            "Reference, blank, PS/SI, averaged, NF, post-operative, and before-centrifuge files are excluded.",
            "BPRO/BNOR are kept as external Boramae binary groups; BPRO is mapped to PRO only for type-ID scoring.",
        ],
    }
    with open(args.output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(scores.to_string(index=False))
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
