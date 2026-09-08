from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

MODEL_SUFFIXES = frozenset({".pt", ".pth", ".joblib", ".pkl", ".onnx", ".ckpt", ".safetensors", ".bin", ".h5", ".keras"})
SKIPPED_PARAM_KEYS = frozenset({"metrics", "output_dir", "input", "output", "source_uri", "raw_uri", "artifacts"})


@dataclass(frozen=True, slots=True)
class HistoricalModelBundle:
    source_dir: Path
    model_files: tuple[Path, ...]
    metadata_path: Path | None
    metadata: JsonObject
    source_model_name: str
    model_display_name: str
    source_version: str
    experiment_label: str
    timestamp: str


def is_model_file(path: Path) -> bool:
    """Return whether a path is a serialized model or checkpoint file."""
    return path.is_file() and path.suffix.casefold() in MODEL_SUFFIXES


def _bundle_dir(path: Path) -> Path:
    return path.parent.parent if path.parent.name.casefold() == "checkpoints" else path.parent


def _read_json_object(path: Path) -> JsonObject:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _metadata_path(bundle: Path, roots: Sequence[Path]) -> Path | None:
    root_set = tuple(root.resolve() for root in roots)
    cursor = bundle.resolve()
    while True:
        for name in (
            "training_summary.json",
            "manifest.json",
            "experiment_summary.json",
            "config.json",
            "best_config.json",
        ):
            candidate = cursor / name
            if candidate.is_file():
                return candidate
        if cursor in root_set or cursor.parent == cursor:
            return None
        cursor = cursor.parent


def _text(metadata: Mapping[str, JsonValue], key: str) -> str:
    value = metadata.get(key)
    return value.strip() if isinstance(value, str) else ""


def _bundle_from_files(source_dir: Path, files: Sequence[Path], roots: Sequence[Path]) -> HistoricalModelBundle:
    metadata_path = _metadata_path(source_dir, roots)
    metadata = _read_json_object(metadata_path) if metadata_path else {}
    fallback_name = {
        "production": "AECD Production SERS",
        "production_stacking": "AECD Production Stacking",
    }.get(source_dir.name, source_dir.name)
    model_name = _text(metadata, "model_name") or _text(metadata, "model_type") or fallback_name
    display_name = _text(metadata, "model") or _text(metadata, "model_type") or model_name
    version = _text(metadata, "version") or "unversioned"
    experiment = _text(metadata, "experiment") or source_dir.parent.name
    timestamp = _text(metadata, "timestamp") or _text(metadata, "training_date") or _text(metadata, "started")
    return HistoricalModelBundle(
        source_dir=source_dir,
        model_files=tuple(sorted(files)),
        metadata_path=metadata_path,
        metadata=metadata,
        source_model_name=model_name,
        model_display_name=display_name,
        source_version=version,
        experiment_label=experiment,
        timestamp=timestamp,
    )


def discover_model_bundles(roots: Sequence[Path]) -> tuple[HistoricalModelBundle, ...]:
    """Discover serialized model bundles and their nearest metadata file."""
    groups: dict[Path, list[Path]] = {}
    resolved_roots = tuple(root.resolve() for root in roots if root.exists())
    for root in resolved_roots:
        for path in root.rglob("*"):
            if is_model_file(path):
                groups.setdefault(_bundle_dir(path), []).append(path)
    bundles = [_bundle_from_files(path, files, resolved_roots) for path, files in groups.items()]
    return tuple(sorted(bundles, key=lambda bundle: str(bundle.source_dir)))


def _key(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower() or "value"


def flatten_mlflow_params(metadata: Mapping[str, JsonValue]) -> tuple[tuple[str, str], ...]:
    """Flatten configuration metadata into MLflow-safe string parameters."""
    pairs: list[tuple[str, str]] = []

    def visit(value: JsonValue, prefix: str) -> None:
        if isinstance(value, dict):
            for child_key, child_value in sorted(value.items()):
                visit(child_value, f"{prefix}_{_key(child_key)}" if prefix else _key(child_key))
            return
        if isinstance(value, list):
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        elif value is None:
            rendered = "null"
        else:
            rendered = str(value)
        pairs.append((_key(prefix)[:250], rendered[:250]))

    for key, value in sorted(metadata.items()):
        if key not in SKIPPED_PARAM_KEYS:
            visit(value, _key(key))
    return tuple(pairs)


def parameter_changes(previous: Mapping[str, str], current: Mapping[str, str]) -> dict[str, dict[str, str]]:
    """Return the changed parameter values between two historical runs."""
    missing = "<missing>"
    return {
        key: {"from": previous.get(key, missing), "to": current.get(key, missing)}
        for key in sorted(set(previous) | set(current))
        if previous.get(key, missing) != current.get(key, missing)
    }


def model_bundle_hash(bundle: HistoricalModelBundle) -> str:
    """Hash model file names and bytes for stable backfill identity."""
    digest = hashlib.sha256()
    for path in bundle.model_files:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
