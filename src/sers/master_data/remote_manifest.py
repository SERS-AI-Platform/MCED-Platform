from __future__ import annotations

import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator
from typing_extensions import Annotated

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class RemoteDatasetAsset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    measurement_id: str
    artifact_sha256: Sha256
    storage_path: str
    source_group: str
    site_code: str
    instrument_key: str
    replicate_index: int
    label_value: str
    label_definition_version: str
    split_name: str
    fold_index: int

    @field_validator("storage_path")
    @classmethod
    def storage_path_is_relative(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("storage path must be relative")
        return value


class RemoteDatasetManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["sers-remote-manifest-v1"]
    manifest_id: str
    content_sha256: Sha256
    assets: tuple[RemoteDatasetAsset, ...]


@dataclass(frozen=True, slots=True)
class RemoteManifestExportResult:
    output_path: Path
    manifest_id: str
    content_sha256: str
    asset_count: int


@dataclass(frozen=True, slots=True)
class RemoteManifestNotFoundError(LookupError):
    manifest_id: str

    def __str__(self) -> str:
        return "frozen dataset manifest was not found"


@dataclass(frozen=True, slots=True)
class RemoteManifestExistsError(FileExistsError):
    output_path: Path

    def __str__(self) -> str:
        return "remote manifest already exists; pass overwrite to replace it"


def _manifest(
    connection: sqlite3.Connection,
    manifest_id: str,
) -> RemoteDatasetManifest:
    row = connection.execute(
        """SELECT content_sha256 FROM dataset_manifests
           WHERE id = ? AND status = 'frozen'""",
        (manifest_id,),
    ).fetchone()
    if row is None:
        raise RemoteManifestNotFoundError(manifest_id)
    records = connection.execute(
        """SELECT dmi.measurement_id, dmi.artifact_sha256,
                  sm.source_group, site.code, mr.instrument_key,
                  measurement.replicate_index, label.label_value,
                  dmi.label_definition_version, dmi.split_name, dmi.fold_index
           FROM dataset_manifest_items AS dmi
           JOIN measurements AS measurement ON measurement.id = dmi.measurement_id
           JOIN measurement_runs AS mr ON mr.id = measurement.measurement_run_id
           JOIN analytical_materials AS material
             ON material.id = measurement.analytical_material_id
           JOIN spectrum_material_metadata AS sm
             ON sm.analytical_material_id = material.id
           JOIN sites AS site ON site.id = sm.site_id
           JOIN sample_labels AS label ON label.id = dmi.sample_label_id
           WHERE dmi.dataset_manifest_id = ?
           ORDER BY dmi.ordinal""",
        (manifest_id,),
    )
    assets = tuple(
        RemoteDatasetAsset(
            measurement_id=str(record[0]),
            artifact_sha256=str(record[1]),
            storage_path=f"raw/sha256/{record[1]}/{record[1]}",
            source_group=str(record[2]),
            site_code=str(record[3]),
            instrument_key=str(record[4]),
            replicate_index=int(record[5]),
            label_value=str(record[6]),
            label_definition_version=str(record[7]),
            split_name=str(record[8]),
            fold_index=int(record[9]),
        )
        for record in records
    )
    return RemoteDatasetManifest(
        schema_version="sers-remote-manifest-v1",
        manifest_id=manifest_id,
        content_sha256=str(row[0]),
        assets=assets,
    )


def export_remote_dataset_manifest(
    connection: sqlite3.Connection,
    manifest_id: str,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> RemoteManifestExportResult:
    manifest = _manifest(connection, manifest_id)
    if output_path.exists() and not overwrite:
        raise RemoteManifestExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(manifest.model_dump_json(indent=2))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return RemoteManifestExportResult(
        output_path,
        manifest.manifest_id,
        manifest.content_sha256,
        len(manifest.assets),
    )


def load_remote_dataset_manifest(path: Path) -> RemoteDatasetManifest:
    return RemoteDatasetManifest.model_validate_json(path.read_bytes())
