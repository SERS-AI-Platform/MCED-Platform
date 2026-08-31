from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from scripts.db.aecd_model_artifact_backfill_core import parameter_changes


@dataclass(frozen=True, slots=True)
class TimelineRun:
    run_id: str
    registry_model_name: str
    registry_model_version: str
    source_dir: str
    source_model_version: str
    model_display_name: str
    params: Mapping[str, str]
    metrics: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class TimelineRow:
    run_id: str
    registry_model_name: str
    registry_model_version: str
    source_dir: str
    source_model_version: str
    model_display_name: str
    source_timestamp: str
    source_timestamp_epoch: float | None
    source_time_status: str
    history_order: int
    hyperparameter_change_count: int
    changed_hyperparameters: tuple[str, ...]
    cancer_screening_auc: float | None
    cancer_type_id_auc: float | None


@dataclass(frozen=True, slots=True)
class ParameterChangeRow:
    registry_model_name: str
    parameter_name: str
    change_count: int


_SOURCE_TIMESTAMP_KEYS = ("timestamp", "training_date", "started")
_SCREENING_METRIC_KEYS = ("historical_val_s1_auc", "historical_s1_auc")
_TYPE_ID_METRIC_KEYS = ("historical_val_s2_auc", "historical_s2_auc")


def parse_source_timestamp(value: str) -> float | None:
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    aware = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc).timestamp()


def _source_timestamp(params: Mapping[str, str]) -> str:
    for key in _SOURCE_TIMESTAMP_KEYS:
        value = params.get(key, "").strip()
        if value:
            return value
    return ""


def _metric(metrics: Mapping[str, float], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if value is not None and math.isfinite(value):
            return value
    return None


def _hyperparameters(params: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in params.items()
        if key.startswith("config_") or key.startswith("model_params_")
    }


def _sort_key(run: TimelineRun) -> tuple[int, float, str, str]:
    timestamp = parse_source_timestamp(_source_timestamp(run.params))
    return (timestamp is None, timestamp if timestamp is not None else math.inf, run.source_dir, run.run_id)


def build_timeline(runs: Sequence[TimelineRun]) -> tuple[TimelineRow, ...]:
    rows: list[TimelineRow] = []
    families: dict[str, list[TimelineRun]] = {}
    for run in runs:
        families.setdefault(run.registry_model_name, []).append(run)
    for family in sorted(families):
        previous: Mapping[str, str] = {}
        for order, run in enumerate(sorted(families[family], key=_sort_key), start=1):
            source_timestamp = _source_timestamp(run.params)
            source_epoch = parse_source_timestamp(source_timestamp)
            current = _hyperparameters(run.params)
            changes = parameter_changes(previous, current) if previous else {}
            rows.append(
                TimelineRow(
                    run_id=run.run_id,
                    registry_model_name=run.registry_model_name,
                    registry_model_version=run.registry_model_version,
                    source_dir=run.source_dir,
                    source_model_version=run.source_model_version,
                    model_display_name=run.model_display_name,
                    source_timestamp=source_timestamp,
                    source_timestamp_epoch=source_epoch,
                    source_time_status=("observed" if source_epoch is not None else "invalid" if source_timestamp else "missing"),
                    history_order=order,
                    hyperparameter_change_count=len(changes),
                    changed_hyperparameters=tuple(changes),
                    cancer_screening_auc=_metric(run.metrics, _SCREENING_METRIC_KEYS),
                    cancer_type_id_auc=_metric(run.metrics, _TYPE_ID_METRIC_KEYS),
                )
            )
            previous = current
    return tuple(rows)


def parameter_change_rows(rows: Sequence[TimelineRow]) -> tuple[ParameterChangeRow, ...]:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        for parameter_name in row.changed_hyperparameters:
            key = (row.registry_model_name, parameter_name)
            counts[key] = counts.get(key, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    return tuple(
        ParameterChangeRow(
            registry_model_name=key[0],
            parameter_name=key[1],
            change_count=count,
        )
        for key, count in ordered
    )
