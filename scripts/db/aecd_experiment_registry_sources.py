from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from scripts.db.aecd_experiment_registry_core import (
    POST_CHANGE_VERIFIED,
    RegistryRun,
    artifact_paths,
    lineage_key,
    metric_pairs,
    read_csv,
    row_value,
    select_recent_rows,
    slug,
)


def _recent_run(row: Mapping[str, str], repo_root: Path) -> RegistryRun:
    directory = row_value(row, "directory")
    params = [("experiment_key", directory), ("source_family", "mapping")]
    for field in (
        "category",
        "comparison_group",
        "purpose",
        "status",
        "model",
        "n_blocks",
        "epochs",
        "mc_iterations",
        "seed",
        "subjects",
        "finite_repeats",
        "trim",
        "smooth",
        "baseline",
        "normalization",
        "grid",
    ):
        value = row_value(row, field)
        if value:
            params.append((field, value[:250]))
    params.extend(
        [
            ("source_phase", POST_CHANGE_VERIFIED),
            ("aggregation", "mean"),
            ("artifact_policy", "aggregate_only"),
        ]
    )
    fields = (
        ("cancer_screening_auc", "cancer_screening_auc"),
        ("cancer_screening_balanced_accuracy", "cancer_screening_balanced_accuracy"),
        ("cancer_screening_sensitivity", "cancer_screening_sensitivity"),
        ("cancer_screening_specificity", "cancer_screening_specificity"),
        ("cancer_type_id_macro_auc", "cancer_type_id_macro_auc"),
        ("cancer_type_id_balanced_accuracy", "cancer_type_id_balanced_accuracy"),
        ("cancer_type_id_macro_f1", "cancer_type_id_macro_f1"),
    )
    return RegistryRun(
        run_name=slug(directory),
        lineage_key=lineage_key("recent_mapping", directory),
        source_phase=POST_CHANGE_VERIFIED,
        evidence_status=row_value(row, "status"),
        params=tuple(params),
        metrics=metric_pairs(row, fields),
        tags=(("source_catalog", "recent_mapping_inventory"),),
        artifact_paths=artifact_paths(repo_root / "results" / directory),
    )


def build_recent_runs(repo_root: Path) -> list[RegistryRun]:
    """Build post-change runs from the screened recent mapping inventory."""
    path = repo_root / "results" / "recent_experiment_inventory_20260826_v1" / "experiment_inventory.csv"
    return [_recent_run(row, repo_root) for row in select_recent_rows(read_csv(path))]


def _find_metric_row(
    rows: list[Mapping[str, str]], condition: str, task: str
) -> Mapping[str, str] | None:
    return next(
        (
            row
            for row in rows
            if row_value(row, "condition") == condition
            and row_value(row, "model") == "Multi-scale ResNet"
            and row_value(row, "task") == task
        ),
        None,
    )


def _aligned_run(
    repo_root: Path,
    metric_rows: list[Mapping[str, str]],
    condition: str,
    label: str,
    child: Path,
) -> RegistryRun:
    params = (
        ("experiment_key", f"mapping_aligned_baseline_area_dwt:{condition}"),
        ("source_family", "mapping"),
        ("source_phase", POST_CHANGE_VERIFIED),
        ("model", "Multi-scale 1D ResNet"),
        ("preprocessing", label),
        ("aggregation", "mean"),
        ("subjects", "113"),
        ("finite_repeats", "13673"),
        ("n_blocks", "3"),
        ("epochs", "30"),
        ("mc_iterations", "100"),
        ("seed", "20260826"),
        ("artifact_policy", "aggregate_only"),
    )
    binary = _find_metric_row(metric_rows, condition, "cancer_vs_non_cancer") or {}
    three = _find_metric_row(metric_rows, condition, "three_class") or {}
    metrics = list(
        metric_pairs(
            binary,
            (
                ("roc_auc", "cancer_screening_auc"),
                ("balanced_accuracy", "cancer_screening_balanced_accuracy"),
                ("sensitivity", "cancer_screening_sensitivity"),
                ("specificity", "cancer_screening_specificity"),
            ),
        )
    )
    metrics.extend(
        metric_pairs(
            three,
            (
                ("macro_roc_auc", "cancer_type_id_macro_auc"),
                ("balanced_accuracy", "cancer_type_id_balanced_accuracy"),
                ("macro_f1", "cancer_type_id_macro_f1"),
            ),
        )
    )
    parent = repo_root / "results" / "mapping_aligned_baseline_area_dwt_20260826_v1"
    artifacts = artifact_paths(child) + tuple(
        path for path in (parent / "REPORT.md", parent / "comparison_oof_metrics.csv") if path.is_file()
    )
    return RegistryRun(
        run_name=f"mapping_aligned_{slug(condition)}",
        lineage_key=lineage_key("aligned_mapping", condition),
        source_phase=POST_CHANGE_VERIFIED,
        evidence_status="주 분석 결과",
        params=params,
        metrics=tuple(metrics),
        tags=(("source_catalog", "mapping_preprocessing_ablation"),),
        artifact_paths=artifacts,
    )


def build_aligned_runs(repo_root: Path) -> list[RegistryRun]:
    """Build one post-change run for each aligned preprocessing condition."""
    parent = repo_root / "results" / "mapping_aligned_baseline_area_dwt_20260826_v1"
    metric_rows = read_csv(parent / "comparison_oof_metrics.csv")
    conditions: list[str] = []
    labels: dict[str, str] = {}
    for row in metric_rows:
        condition = row_value(row, "condition")
        if condition and condition not in conditions:
            conditions.append(condition)
            labels[condition] = row_value(row, "label") or condition
    child_dirs = {
        "raw_aligned": "raw_aligned",
        "baseline": "baseline",
        "baseline_area": "baseline_area",
        "baseline_dwt": "baseline_dwt",
        "baseline_area_dwt": "baseline_area_dwt",
    }
    return [
        _aligned_run(repo_root, metric_rows, condition, labels[condition], parent / child_dirs[condition])
        for condition in conditions
        if condition in child_dirs
    ]
