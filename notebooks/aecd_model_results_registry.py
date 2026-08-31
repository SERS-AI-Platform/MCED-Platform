from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
RESULTS_DIR = REPO_ROOT / "results"
PYTHON = sys.executable


@dataclass(frozen=True, slots=True)
class Experiment:
    key: str
    family: str
    title: str
    command: tuple[str, ...]
    output_dir: Path
    result_kind: str
    notes: str


EXPERIMENTS: dict[str, Experiment] = {
    "aecd_direct_mean": Experiment(
        key="aecd_direct_mean",
        family="AECD API",
        title="Current direct mean",
        command=(PYTHON, str(REPO_ROOT / "scripts/analysis/aecd_api_model_mean_spectrum_clinical_performance.py")),
        output_dir=NOTEBOOK_DIR / "aecd_api_model_mean_spectrum_outputs",
        result_kind="aecd_direct",
        notes="PS alignment + repeat QC + subject mean; direct four-model stacking",
    ),
    "aecd_dwt_and_raw": Experiment(
        key="aecd_dwt_and_raw",
        family="AECD API",
        title="DWT raw and DWT preprocessing",
        command=(PYTHON, str(NOTEBOOK_DIR / "aecd_api_model_patent_dwt_clinical_performance.py")),
        output_dir=NOTEBOOK_DIR / "aecd_api_model_patent_dwt_outputs",
        result_kind="aecd_dwt_binary",
        notes="Raw subject mean and residual DWT db4/AsLS evaluated with STK-V2",
    ),
    "aecd_all_qc": Experiment(
        key="aecd_all_qc",
        family="AECD API",
        title="Current all-QC spectra",
        command=(PYTHON, str(NOTEBOOK_DIR / "aecd_api_model_all_qc_spectra_clinical_performance.py")),
        output_dir=NOTEBOOK_DIR / "aecd_api_model_all_qc_spectra_outputs",
        result_kind="aecd_all_qc",
        notes="All QC-passed repeats; subject-level probability aggregation",
    ),
    "aecd_three_class": Experiment(
        key="aecd_three_class",
        family="AECD API",
        title="AECD three-class conditions",
        command=(PYTHON, str(NOTEBOOK_DIR / "aecd_api_model_3class_clinical_performance.py")),
        output_dir=NOTEBOOK_DIR / "aecd_api_model_3class_outputs",
        result_kind="aecd_three_class",
        notes="Direct mean, raw STK-V2, DWT STK-V2, and all-QC three-class runs",
    ),
    "mapping_resnet_3": Experiment(
        key="mapping_resnet_3",
        family="Mapping Raman",
        title="Spectrum LR-reference and ResNet 3-block",
        command=(PYTHON, str(REPO_ROOT / "scripts/analysis/run_mapping_multiscale_resnet.py"), "--n-blocks", "3", "--output", str(RESULTS_DIR / "notebook_resnet_3"), "--skip-stk"),
        output_dir=RESULTS_DIR / "notebook_resnet_3",
        result_kind="mapping_metrics",
        notes="Standard Raman preprocessing; LR reference and 3-block ResNet",
    ),
    "mapping_resnet_6": Experiment(
        key="mapping_resnet_6",
        family="Mapping Raman",
        title="ResNet 6-block",
        command=(PYTHON, str(REPO_ROOT / "scripts/analysis/run_mapping_multiscale_resnet.py"), "--n-blocks", "6", "--output", str(RESULTS_DIR / "notebook_resnet_6"), "--skip-stk"),
        output_dir=RESULTS_DIR / "notebook_resnet_6",
        result_kind="mapping_metrics",
        notes="Standard Raman preprocessing; 6-block ResNet",
    ),
    "mapping_resnet_9": Experiment(
        key="mapping_resnet_9",
        family="Mapping Raman",
        title="ResNet 9-block",
        command=(PYTHON, str(REPO_ROOT / "scripts/analysis/run_mapping_multiscale_resnet.py"), "--n-blocks", "9", "--output", str(RESULTS_DIR / "notebook_resnet_9"), "--skip-stk"),
        output_dir=RESULTS_DIR / "notebook_resnet_9",
        result_kind="mapping_metrics",
        notes="Standard Raman preprocessing; 9-block ResNet",
    ),
    "mapping_position_aware": Experiment(
        key="mapping_position_aware",
        family="Mapping Raman",
        title="Position-aware ResNet pilot",
        command=(PYTHON, str(REPO_ROOT / "scripts/analysis/run_mapping_multiscale_resnet.py"), "--n-blocks", "3", "--position-aware", "--output", str(RESULTS_DIR / "notebook_position_aware"), "--skip-stk"),
        output_dir=RESULTS_DIR / "notebook_position_aware",
        result_kind="mapping_metrics",
        notes="Absolute Raman position channel added to the 3-block ResNet",
    ),
    "mapping_clinical": Experiment(
        key="mapping_clinical",
        family="Mapping multimodal",
        title="Clinical-only LR and Clinical-input CNN",
        command=(PYTHON, str(REPO_ROOT / "scripts/analysis/run_mapping_multimodal_resnet.py"), "--base-output", str(RESULTS_DIR / "notebook_resnet_3"), "--output", str(RESULTS_DIR / "notebook_clinical")),
        output_dir=RESULTS_DIR / "notebook_clinical",
        result_kind="mapping_metrics",
        notes="Clinical-only LR, spectrum controls, and clinical-input CNN on identical folds",
    ),
    "mapping_clinical_spectrum_lr": Experiment(
        key="mapping_clinical_spectrum_lr",
        family="Mapping multimodal",
        title="Clinical + Spectrum LR",
        command=(PYTHON, str(REPO_ROOT / "scripts/analysis/run_mapping_combined_logistic.py"), "--base-output", str(RESULTS_DIR / "notebook_resnet_3"), "--source-output", str(RESULTS_DIR / "notebook_clinical"), "--output", str(RESULTS_DIR / "notebook_clinical_spectrum_lr")),
        output_dir=RESULTS_DIR / "notebook_clinical_spectrum_lr",
        result_kind="mapping_combined_lr",
        notes="Patient-mean spectrum concatenated with five clinical variables",
    ),
}


def experiment_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "key": item.key,
                "family": item.family,
                "title": item.title,
                "output_dir": str(item.output_dir),
                "notes": item.notes,
            }
            for item in EXPERIMENTS.values()
        ]
    )


def run_experiment(key: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    experiment = EXPERIMENTS[key]
    runtime_env = os.environ.copy()
    if env:
        runtime_env.update(env)
    return subprocess.run(
        experiment.command,
        cwd=REPO_ROOT,
        env=runtime_env,
        text=True,
        check=True,
    )


def run_selected(keys: Sequence[str], *, env: dict[str, str] | None = None) -> None:
    for key in keys:
        print(f"[run] {key}: {EXPERIMENTS[key].title}", flush=True)
        run_experiment(key, env=env)


def _number(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _result_row(key: str, title: str, family: str, model: str, notes: str) -> dict:
    return {
        "experiment_key": key,
        "family": family,
        "condition": title,
        "model": model,
        "binary_auc": np.nan,
        "binary_bacc": np.nan,
        "macro_auc_3class": np.nan,
        "bacc_3class": np.nan,
        "macro_f1_3class": np.nan,
        "control_auc": np.nan,
        "pdc_auc": np.nan,
        "cancer_auc": np.nan,
        "evaluation": "nested subject-level OOF",
        "notes": notes,
    }


def _collect_mapping(experiment: Experiment, path: Path) -> list[dict]:
    if not path.exists():
        return []
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame = frame.loc[frame["aggregation"].eq("mean")].copy()
    rows = []
    for model_name in frame["model"].drop_duplicates():
        subset = frame.loc[frame["model"].eq(model_name)]
        row = _result_row(
            experiment.key,
            str(model_name),
            experiment.family,
            str(model_name),
            experiment.notes,
        )
        binary = subset.loc[subset["task"].eq("cancer_vs_non_cancer")]
        three = subset.loc[subset["task"].eq("three_class")]
        if not binary.empty:
            row["binary_auc"] = _number(binary.iloc[0].get("roc_auc"))
            row["binary_bacc"] = _number(binary.iloc[0].get("balanced_accuracy"))
        if not three.empty:
            row["macro_auc_3class"] = _number(three.iloc[0].get("macro_roc_auc"))
            row["bacc_3class"] = _number(three.iloc[0].get("balanced_accuracy"))
            row["macro_f1_3class"] = _number(three.iloc[0].get("macro_f1"))
        rows.append(row)
    return rows


def _collect_aecd_three_class(experiment: Experiment) -> list[dict]:
    path = experiment.output_dir / "clinical_3class_performance_summary.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path, encoding="utf-8-sig")
    aliases = {
        "current direct QC-passed subject mean": "Current direct mean",
        "DWT comparison raw subject mean": "DWT raw comparison",
        "Patent DWT preprocessing": "DWT preprocessing",
        "current all QC-passed spectra": "Current all-QC spectra",
    }
    rows = []
    for _, source in frame.iterrows():
        condition = aliases.get(str(source["condition"]), str(source["condition"]))
        row = _result_row(
            experiment.key,
            condition,
            experiment.family,
            "STK-V2 or direct stacking (condition-specific)",
            experiment.notes,
        )
        row.update(
            {
                "macro_auc_3class": _number(source.get("macro_ovr_auc")),
                "bacc_3class": _number(source.get("balanced_accuracy")),
                "macro_f1_3class": _number(source.get("macro_f1")),
                "control_auc": _number(source.get("control_ovr_auc")),
                "pdc_auc": _number(source.get("prostate_disease_control_ovr_auc")),
                "cancer_auc": _number(source.get("prostate_ovr_auc")),
            }
        )
        rows.append(row)
    all_qc_path = experiment.output_dir / "all_qc_3class_performance_summary.csv"
    if all_qc_path.exists():
        all_qc_frame = pd.read_csv(all_qc_path, encoding="utf-8-sig")
        for _, source in all_qc_frame.iterrows():
            row = _result_row(
                experiment.key,
                "Current all-QC spectra",
                experiment.family,
                "Direct stacking on all QC-passed spectra with subject aggregation",
                experiment.notes,
            )
            row.update(
                {
                    "macro_auc_3class": _number(source.get("macro_ovr_auc")),
                    "bacc_3class": _number(source.get("balanced_accuracy")),
                    "macro_f1_3class": _number(source.get("macro_f1")),
                    "control_auc": _number(source.get("control_ovr_auc")),
                    "pdc_auc": _number(source.get("prostate_disease_control_ovr_auc")),
                    "cancer_auc": _number(source.get("prostate_ovr_auc")),
                }
            )
            rows.append(row)
    return rows


def _collect_simple_aecd(experiment: Experiment) -> list[dict]:
    path = experiment.output_dir / "clinical_performance_summary.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path, encoding="utf-8-sig")
    rows = []
    for _, source in frame.iterrows():
        title = experiment.title
        if "method" in source and pd.notna(source.get("method")):
            method = str(source["method"])
            if "dwt" in method.lower():
                title = "DWT preprocessing"
            elif "raw" in method.lower():
                title = "DWT raw comparison"
        row = _result_row(experiment.key, title, experiment.family, str(source.get("method", title)), experiment.notes)
        row["binary_auc"] = _number(
            source.get("overall_oof_auc", source.get("subject_oof_auc_primary"))
        )
        row["binary_bacc"] = _number(
            source.get("overall_balanced_accuracy", source.get("subject_balanced_accuracy"))
        )
        rows.append(row)
    return rows


def collect_results(keys: Iterable[str] | None = None) -> pd.DataFrame:
    selected = list(keys) if keys is not None else list(EXPERIMENTS)
    rows: list[dict] = []
    for key in selected:
        experiment = EXPERIMENTS[key]
        if experiment.result_kind == "aecd_three_class":
            rows.extend(_collect_aecd_three_class(experiment))
        elif experiment.result_kind in {"aecd_direct", "aecd_dwt_binary", "aecd_all_qc"}:
            rows.extend(_collect_simple_aecd(experiment))
        elif experiment.result_kind == "mapping_metrics":
            rows.extend(_collect_mapping(experiment, experiment.output_dir / "oof_metrics.csv"))
        elif experiment.result_kind == "mapping_combined_lr":
            rows.extend(_collect_mapping(experiment, experiment.output_dir / "combined_lr_oof_metrics.csv"))
    return pd.DataFrame(rows)


def merge_binary_and_three_class(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results.copy()
    metric_columns = [
        "binary_auc", "binary_bacc", "macro_auc_3class", "bacc_3class",
        "macro_f1_3class", "control_auc", "pdc_auc", "cancer_auc",
    ]
    identity_columns = ["family", "condition"]
    merged_rows = []
    for identity, frame in results.groupby(identity_columns, dropna=False, sort=False):
        row = {name: value for name, value in zip(identity_columns, identity, strict=True)}
        row["experiment_key"] = ", ".join(frame["experiment_key"].drop_duplicates())
        row["model"] = " | ".join(frame["model"].dropna().drop_duplicates())
        row["evaluation"] = " | ".join(frame["evaluation"].dropna().drop_duplicates())
        row["notes"] = " | ".join(frame["notes"].dropna().drop_duplicates())
        for column in metric_columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            row[column] = float(values.iloc[-1]) if len(values) else np.nan
        merged_rows.append(row)
    columns = ["experiment_key", "family", "condition", "model", *metric_columns, "evaluation", "notes"]
    return pd.DataFrame(merged_rows)[columns]


def interpretation_table(results: pd.DataFrame) -> pd.DataFrame:
    frame = results.copy()
    if frame.empty:
        return frame
    best_binary = frame["binary_auc"].idxmax() if frame["binary_auc"].notna().any() else None
    best_three = frame["macro_auc_3class"].idxmax() if frame["macro_auc_3class"].notna().any() else None
    frame["interpretation"] = ""
    if best_binary is not None:
        frame.loc[best_binary, "interpretation"] += "Highest binary AUC. "
    if best_three is not None:
        frame.loc[best_three, "interpretation"] += "Highest three-class macro AUC. "
    dwt_mask = frame["condition"].str.contains("DWT preprocessing", case=False, na=False)
    frame.loc[dwt_mask, "interpretation"] += "Compare directly with the DWT raw control. "
    return frame
