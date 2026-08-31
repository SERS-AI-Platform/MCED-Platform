from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias

from scripts.db.aecd_experiment_registry_core import (
    PRE_CHANGE_UNVERIFIED,
    RegistryRun,
    allowed_artifact,
    finite_metric,
    lineage_key,
    read_csv,
    row_value,
    slug,
)

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _historical_params(row: Mapping[str, str], experiment: str, model: str) -> tuple[tuple[str, str], ...]:
    blocks = re.search(r"(\d+)\s*blocks", experiment, flags=re.IGNORECASE)
    params = [
        ("experiment_key", f"historical_summary:{experiment}"[:250]),
        ("experiment_label", experiment[:250]),
        ("model", model[:250]),
        ("source_phase", PRE_CHANGE_UNVERIFIED),
        ("evidence_status", "historical_summary_only"),
        ("artifact_policy", "aggregate_only"),
    ]
    if blocks:
        params.append(("n_blocks", blocks.group(1)))
    for field in ("preprocessing", "comparison_status"):
        value = row_value(row, field)
        if value:
            params.append((field, value[:250]))
    return tuple(params)


def build_historical_summary_runs(path: Path) -> list[RegistryRun]:
    """Group historical summary rows into task-complete aggregate runs."""
    groups: dict[tuple[str, str], list[Mapping[str, str]]] = {}
    for row in read_csv(path):
        key = (row_value(row, "experiment"), row_value(row, "model"))
        if all(key):
            groups.setdefault(key, []).append(row)
    runs: list[RegistryRun] = []
    for (experiment, model), rows in groups.items():
        metrics: dict[str, float] = {}
        for row in rows:
            task = row_value(row, "task")
            prefix = (
                "cancer_screening"
                if task == "cancer_vs_non_cancer"
                else "cancer_type_id"
                if task == "three_class"
                else ""
            )
            if not prefix:
                continue
            for field, suffix in (
                ("roc_auc", "auc"),
                ("macro_roc_auc", "macro_auc"),
                ("balanced_accuracy", "balanced_accuracy"),
                ("macro_f1", "macro_f1"),
            ):
                value = finite_metric(row_value(row, field))
                if value is not None:
                    metrics[f"{prefix}_{suffix}"] = value
        runs.append(
            RegistryRun(
                run_name=f"historical_summary_{slug(experiment)}_{slug(model)}",
                lineage_key=lineage_key("historical_summary", f"{experiment}|{model}"),
                source_phase=PRE_CHANGE_UNVERIFIED,
                evidence_status="historical_summary_only",
                params=_historical_params(rows[0], experiment, model),
                metrics=tuple(sorted(metrics.items())),
                tags=(
                    ("source_catalog", "historical_experiment_summary"),
                    ("direct_comparison", "not_allowed"),
                ),
                artifact_paths=(path,) if allowed_artifact(path) else (),
            )
        )
    return runs


def _string_value(value: JsonValue | None) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _flatten_numeric(value: JsonValue, prefix: str) -> list[tuple[str, float]]:
    if isinstance(value, dict):
        pairs: list[tuple[str, float]] = []
        for key, child in value.items():
            child_prefix = f"{prefix}_{slug(key)}" if prefix else slug(key)
            pairs.extend(_flatten_numeric(child, child_prefix))
        return pairs
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        number = finite_metric(value)
        if number is not None:
            return [(slug(prefix)[:200], number)]
    return []


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_legacy_log_runs(path: Path) -> list[RegistryRun]:
    """Convert the old run log into explicitly metadata-only MLflow runs."""
    if not path.is_file():
        return []
    source_hash = _sha256_file(path)
    runs: list[RegistryRun] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            try:
                payload: JsonValue = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            experiment = (
                _string_value(payload.get("experiment"))
                or _string_value(payload.get("model_name"))
                or "unnamed"
            )
            model = _string_value(payload.get("model_name"))
            params = [
                ("experiment_key", f"legacy_log:{index:04d}"),
                ("experiment_label", experiment[:250]),
                ("model", model[:250]),
                ("source_phase", PRE_CHANGE_UNVERIFIED),
                ("evidence_status", "historical_log_metadata_only"),
                ("direct_comparison", "not_allowed"),
                ("artifact_policy", "no_artifact_available"),
            ]
            for field in ("version", "aggregate", "n_splits", "n_samples", "n_features", "phase"):
                value = _string_value(payload.get(field))
                if value:
                    params.append((field, value[:250]))
            timestamp = _string_value(payload.get("timestamp"))
            if timestamp:
                params.append(("record_date", timestamp[:10]))
            raw_metrics = payload.get("metrics")
            metrics = tuple(_flatten_numeric(raw_metrics, "historical") if raw_metrics is not None else ())
            runs.append(
                RegistryRun(
                    run_name=f"legacy_log_{index:04d}_{slug(experiment)[:50]}",
                    lineage_key=lineage_key("legacy_log", f"{source_hash}:{index}"),
                    source_phase=PRE_CHANGE_UNVERIFIED,
                    evidence_status="historical_log_metadata_only",
                    params=tuple(params),
                    metrics=metrics,
                    tags=(
                        ("source_catalog", "experiment_runs_jsonl"),
                        ("direct_comparison", "not_allowed"),
                    ),
                    artifact_paths=(),
                )
            )
    return runs
