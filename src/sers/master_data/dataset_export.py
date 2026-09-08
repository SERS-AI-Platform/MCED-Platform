from __future__ import annotations

import csv
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DatasetExportResult:
    output_path: Path
    manifest_id: str
    content_sha256: str
    row_count: int
    excluded_count: int
    preprocessing_version: str
    feature_schema_version: str
    decision_policy_version: str
    qc_policy_version: str


@dataclass(frozen=True, slots=True)
class _ManifestMetadata:
    manifest_id: str
    content_sha256: str
    preprocessing_version: str
    feature_schema_version: str
    decision_policy_version: str
    qc_policy_version: str


@dataclass(frozen=True, slots=True)
class DatasetManifestNotFoundError(LookupError):
    manifest_id: str

    def __str__(self) -> str:
        return "frozen dataset manifest was not found"


@dataclass(frozen=True, slots=True)
class DatasetExportExistsError(FileExistsError):
    output_path: Path

    def __str__(self) -> str:
        return "dataset export already exists; pass overwrite to replace it"


_COLUMNS = (
    "manifest_id",
    "manifest_content_sha256",
    "measurement_id",
    "artifact_sha256",
    "raw_uri",
    "sample_label_id",
    "label_value",
    "label_definition_version",
    "label_evidence_ids",
    "qc_evaluation_id",
    "qc_rule_version",
    "qc_evaluation_version",
    "split_name",
    "fold_index",
)


def _metadata(
    connection: sqlite3.Connection,
    manifest_id: str,
) -> _ManifestMetadata:
    row = connection.execute(
        """SELECT dm.id, dm.content_sha256, dms.preprocessing_version,
                  dms.feature_schema_version, dms.decision_policy_version,
                  dms.qc_policy_version
           FROM dataset_manifests AS dm
           JOIN dataset_manifest_specs AS dms
             ON dms.dataset_manifest_id = dm.id
           WHERE dm.id = ? AND dm.status = 'frozen'""",
        (manifest_id,),
    ).fetchone()
    if row is None:
        raise DatasetManifestNotFoundError(manifest_id)
    return _ManifestMetadata(
        str(row[0]),
        str(row[1]),
        str(row[2]),
        str(row[3]),
        str(row[4]),
        str(row[5]),
    )


def _rows(
    connection: sqlite3.Connection,
    manifest_id: str,
    content_sha256: str,
) -> tuple[tuple[str | int, ...], ...]:
    records = connection.execute(
        """SELECT dmi.measurement_id, dmi.artifact_sha256, asset.raw_uri,
                  dmi.sample_label_id, sl.label_value,
                  dmi.label_definition_version,
                  (SELECT group_concat(evidence.id, '|')
                     FROM (
                         SELECT sle.id
                         FROM dataset_manifest_label_evidence AS dmle
                         JOIN sample_label_evidence AS sle
                           ON sle.id = dmle.sample_label_evidence_id
                         WHERE dmle.dataset_manifest_item_id = dmi.id
                         ORDER BY sle.id
                     ) AS evidence),
                  dmi.qc_evaluation_id, dmi.qc_rule_version,
                  dmi.qc_evaluation_version, dmi.split_name, dmi.fold_index
           FROM dataset_manifest_items AS dmi
           JOIN sample_labels AS sl ON sl.id = dmi.sample_label_id
           JOIN measurement_artifacts AS ma
             ON ma.measurement_id = dmi.measurement_id
            AND ma.artifact_role = 'raw'
           JOIN source_assets AS asset
             ON asset.id = ma.source_asset_id
           WHERE dmi.dataset_manifest_id = ?
           ORDER BY dmi.ordinal""",
        (manifest_id,),
    )
    return tuple(
        (manifest_id, content_sha256, *tuple(record))
        for record in records
    )


def export_dataset_manifest(
    connection: sqlite3.Connection,
    manifest_id: str,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> DatasetExportResult:
    metadata = _metadata(connection, manifest_id)
    records = _rows(connection, manifest_id, metadata.content_sha256)
    if output_path.exists() and not overwrite:
        raise DatasetExportExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            writer = csv.writer(stream)
            writer.writerow(_COLUMNS)
            writer.writerows(records)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    excluded_count = connection.execute(
        """SELECT COUNT(*) FROM dataset_manifest_exclusions
           WHERE dataset_manifest_id = ?""",
        (manifest_id,),
    ).fetchone()[0]
    return DatasetExportResult(
        output_path,
        metadata.manifest_id,
        metadata.content_sha256,
        len(records),
        excluded_count,
        metadata.preprocessing_version,
        metadata.feature_schema_version,
        metadata.decision_policy_version,
        metadata.qc_policy_version,
    )
