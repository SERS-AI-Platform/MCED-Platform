from __future__ import annotations

import csv
import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

PRE_CHANGE_UNVERIFIED: Final = "pre_change_unverified"
POST_CHANGE_VERIFIED: Final = "post_change_verified"
EVIDENCE_STATUSES: Final = frozenset({"주 분석 결과", "참고 결과"})
ARTIFACT_SUFFIXES: Final = frozenset({".csv", ".json", ".md", ".pdf", ".png", ".svg"})


@dataclass(frozen=True, slots=True)
class RegistryRun:
    run_name: str
    lineage_key: str
    source_phase: str
    evidence_status: str
    params: tuple[tuple[str, str], ...]
    metrics: tuple[tuple[str, float], ...]
    tags: tuple[tuple[str, str], ...]
    artifact_paths: tuple[Path, ...]


def finite_metric(value: str | int | float | None) -> float | None:
    """Parse a finite MLflow metric at the CSV/JSON boundary."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def allowed_artifact(path: Path) -> bool:
    """Permit aggregate evidence while excluding patient-linked outputs."""
    name = path.name.casefold()
    if any(token in name for token in ("patient", "array", "repeat_metrics_mc")):
        return False
    return path.suffix.casefold() in ARTIFACT_SUFFIXES


def select_recent_rows(rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    return [row for row in rows if row.get("status", "") in EVIDENCE_STATUSES]


def read_csv(path: Path) -> list[Mapping[str, str]]:
    """Read a project CSV using the required UTF-8-SIG contract."""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{key: value or "" for key, value in row.items()} for row in csv.DictReader(handle)]


def row_value(row: Mapping[str, str], key: str) -> str:
    return row.get(key, "").strip()


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower() or "run"


def lineage_key(source: str, key: str) -> str:
    """Create a stable non-PII identifier for one registry entry."""
    payload = f"{source}\0{key}".encode("utf-8")
    return f"{source}:{hashlib.sha256(payload).hexdigest()}"


def artifact_paths(root: Path) -> tuple[Path, ...]:
    if not root.is_dir():
        return ()
    return tuple(sorted(path for path in root.rglob("*") if path.is_file() and allowed_artifact(path)))


def metric_pairs(row: Mapping[str, str], fields: Sequence[tuple[str, str]]) -> tuple[tuple[str, float], ...]:
    pairs: list[tuple[str, float]] = []
    for field, name in fields:
        value = finite_metric(row_value(row, field))
        if value is not None:
            pairs.append((name, value))
    return tuple(pairs)
