"""Check whether SERS labels are dominated by blank/date batch effects."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sers import parse_filename
from src.sers.config import load_config

LOGGER = logging.getLogger("blank_batch_bio_signal")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def processed_spectra_path() -> Path:
    candidates = [
        PROJECT_ROOT / "results" / "data" / "processed_spectra_cal_newqc.csv",
        PROJECT_ROOT / "results" / "preprocessing_dacr" / "processed_spectra.csv",
        PROJECT_ROOT / "results" / "processed_spectra.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No processed spectra CSV found")


def build_raw_measurement_metadata() -> pd.DataFrame:
    config = load_config("config/config.yaml")
    rows = []
    raw_root = PROJECT_ROOT / "data" / "임상데이터"
    for path in sorted(raw_root.glob("*/*/*.CSV")):
        if "_ave" in path.stem.lower():
            continue
        folder = path.parent.name
        if folder.startswith("0."):
            continue
        try:
            spec_id = parse_filename(
                path,
                fallback_group=config.folder_to_group.get(folder, "UNK"),
            )
        except Exception:
            continue
        rows.append(
            {
                "group": spec_id.group,
                "sample_id": str(spec_id.sample_id),
                "replicate": str(spec_id.replicate),
                "measurement_date": path.parts[-3].split("_")[0],
                "measurement_batch": path.parts[-3],
                "raw_file": str(path.relative_to(PROJECT_ROOT)),
            }
        )
    return pd.DataFrame(rows)


def spectral_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("x_")]
    return sorted(cols, key=lambda c: float(c.replace("x_", "")))


def load_processed_with_date() -> tuple[pd.DataFrame, list[str]]:
    path = processed_spectra_path()
    LOGGER.info("Using processed spectra: %s", path)
    processed = pd.read_csv(path)
    cols = spectral_columns(processed)
    processed["sample_id"] = processed["sample_id"].astype(str)
    processed["replicate"] = processed["replicate"].astype(str)

    raw_meta = build_raw_measurement_metadata()
    dup = raw_meta.duplicated(["group", "sample_id", "replicate"], keep=False).sum()
    LOGGER.info(
        "Raw metadata rows: %d; duplicate processed keys across dates: %d", len(raw_meta), int(dup)
    )

    merged = processed.merge(
        raw_meta[["group", "sample_id", "replicate", "measurement_date"]],
        on=["group", "sample_id", "replicate"],
        how="left",
    )
    missing = int(merged["measurement_date"].isna().sum())
    if missing:
        LOGGER.warning(
            "%d processed replicate rows could not be matched to raw measurement dates", missing
        )
    return merged.dropna(subset=["measurement_date"]).copy(), cols


def subject_mean_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for (group, sample_id), sub in df.groupby(["group", "sample_id"], sort=False):
        rows.append(
            {
                "group": group,
                "sample_id": sample_id,
                "measurement_date": sub["measurement_date"].mode().iat[0],
                "n_replicates": len(sub),
                "spectrum": sub[cols].to_numpy(dtype=float).mean(axis=0),
            }
        )
    return pd.DataFrame(rows)


def rowwise_corr_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = np.nan
    z = x / denom
    return z @ z.T


def blank_date_stats(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    blank_path = (
        PROJECT_ROOT
        / "results"
        / "blank_lung_pattern_validation"
        / "blank_pattern_validation_long.csv"
    )
    if not blank_path.exists():
        raise FileNotFoundError(f"Blank validation data missing: {blank_path}")

    blank = pd.read_csv(blank_path)
    blank = blank[blank["component"] == "blank_baseline_corrected_snv"].copy()
    mat_df = blank.pivot_table(
        index=["date", "blank_id"],
        columns="wavenumber",
        values="intensity",
    )
    labels = mat_df.index.to_frame(index=False)
    x = mat_df.to_numpy(dtype=float)
    corr = rowwise_corr_matrix(x)

    within = []
    between = []
    dates = labels["date"].to_numpy()
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if dates[i] == dates[j]:
                within.append(corr[i, j])
            else:
                between.append(corr[i, j])

    date_means = (
        blank.groupby(["date", "wavenumber"], as_index=False)["intensity"]
        .mean()
        .pivot(index="date", columns="wavenumber", values="intensity")
    )
    date_corr = pd.DataFrame(
        rowwise_corr_matrix(date_means.to_numpy(dtype=float)),
        index=date_means.index,
        columns=date_means.index,
    )

    summary = pd.DataFrame(
        [
            {
                "n_blank_replicates": len(labels),
                "n_blank_dates": labels["date"].nunique(),
                "median_within_date_blank_corr": float(np.nanmedian(within)),
                "median_between_date_blank_corr": float(np.nanmedian(between)),
                "min_date_mean_corr": float(
                    np.nanmin(date_corr.to_numpy()[np.triu_indices(len(date_corr), k=1)])
                ),
                "median_date_mean_corr": float(
                    np.nanmedian(date_corr.to_numpy()[np.triu_indices(len(date_corr), k=1)])
                ),
            }
        ]
    )

    labels.to_csv(out_dir / "blank_replicate_labels.csv", index=False, encoding="utf-8-sig")
    date_corr.to_csv(out_dir / "blank_date_mean_correlation.csv", encoding="utf-8-sig")
    summary.to_csv(out_dir / "blank_batch_summary.csv", index=False, encoding="utf-8-sig")
    return summary, date_means


def evaluate_binary_lr(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    mode: str = "stratified5",
) -> dict[str, float | int | str]:
    if mode == "leave_date_out":
        splitter = LeaveOneGroupOut().split(x, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=7).split(x, y)

    aucs = []
    accs = []
    n_folds = 0
    for train_idx, test_idx in splitter:
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=7),
        )
        model.fit(x[train_idx], y[train_idx])
        score = model.predict_proba(x[test_idx])[:, 1]
        pred = (score >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[test_idx], score))
        accs.append(accuracy_score(y[test_idx], pred))
        n_folds += 1

    return {
        "mode": mode,
        "n_evaluable_folds": n_folds,
        "mean_auc": float(np.mean(aucs)) if aucs else np.nan,
        "mean_accuracy": float(np.mean(accs)) if accs else np.nan,
    }


def evaluate_lun_nor_confound(
    subj: pd.DataFrame,
    cols: list[str],
    blank_date_means: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    task = subj[subj["group"].isin(["LUN", "NOR"])].copy()
    task["label"] = (task["group"] == "LUN").astype(int)
    x_spec = np.vstack(task["spectrum"].to_numpy())
    y = task["label"].to_numpy()
    dates = task["measurement_date"].to_numpy()

    date_values = sorted(task["measurement_date"].unique())
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    x_date = enc.fit_transform(task[["measurement_date"]])

    blank_lookup = blank_date_means.copy()
    blank_lookup.index = blank_lookup.index.astype(str)
    x_blank = np.vstack(
        [
            blank_lookup.loc[date].to_numpy(dtype=float)
            for date in task["measurement_date"].astype(str)
        ]
    )

    rows = []
    for feature_name, x in [
        ("processed_subject_spectrum", x_spec),
        ("date_only_one_hot", x_date),
        ("date_matched_blank_mean_spectrum", x_blank),
    ]:
        rows.append(
            {
                "task": "LUN_vs_NOR",
                "feature_set": feature_name,
                "n_subjects": len(task),
                "n_lun": int(task["label"].sum()),
                "n_nor": int((1 - task["label"]).sum()),
                "n_dates": len(date_values),
                **evaluate_binary_lr(x, y, mode="stratified5"),
            }
        )
        rows.append(
            {
                "task": "LUN_vs_NOR",
                "feature_set": feature_name,
                "n_subjects": len(task),
                "n_lun": int(task["label"].sum()),
                "n_nor": int((1 - task["label"]).sum()),
                "n_dates": len(date_values),
                **evaluate_binary_lr(x, y, groups=dates, mode="leave_date_out"),
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "lun_nor_date_confound_lr.csv", index=False, encoding="utf-8-sig")
    return result


def main() -> int:
    setup_logging()
    out_dir = PROJECT_ROOT / "results" / "batch_confound_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    processed, cols = load_processed_with_date()
    subj = subject_mean_table(processed, cols)
    subj[["group", "sample_id", "measurement_date", "n_replicates"]].to_csv(
        out_dir / "processed_subject_measurement_dates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    date_group_counts = (
        subj.groupby(["measurement_date", "group"])["sample_id"]
        .nunique()
        .reset_index(name="n_subjects")
    )
    date_group_counts.to_csv(
        out_dir / "subject_group_by_measurement_date.csv", index=False, encoding="utf-8-sig"
    )

    blank_summary, blank_date_means = blank_date_stats(out_dir)
    lr_results = evaluate_lun_nor_confound(subj, cols, blank_date_means, out_dir)

    print("\nBlank batch summary")
    print(blank_summary.to_string(index=False))
    print("\nLUN/NOR confound stress test")
    print(lr_results.to_string(index=False))
    print(
        "\nSubject group/date counts written to:", out_dir / "subject_group_by_measurement_date.csv"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
