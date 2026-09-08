from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Final
from urllib.parse import unquote, urlparse
from uuid import UUID, uuid5

from ._compat import StrEnum
from .raw_store import hash_source_file
from .spectrum_persistence import SourceAssetChangedError

_ID_NAMESPACE = UUID("184399db-531a-4f03-8362-3f0ecb16f09c")


class CrosswalkIdentityField(StrEnum):
    SOURCE_NO = "source_no"
    PROVIDER_CODE = "provider_code"
    SOLUM_LABEL = "solum_label"


_IDENTITY_FIELDS: Final = {
    "SMCMD06_췌장암.xlsx": CrosswalkIdentityField.PROVIDER_CODE,
    "SMCXD01_난소암 1.xlsx": CrosswalkIdentityField.PROVIDER_CODE,
    "SMCXD01_난소암 2.xlsx": CrosswalkIdentityField.PROVIDER_CODE,
    "SMCXD01_전립선암 임상정보.xlsx": CrosswalkIdentityField.SOURCE_NO,
    "SMCXD01_폐암 1.xlsx": CrosswalkIdentityField.PROVIDER_CODE,
    "SMCXD06_대장암.xlsx": CrosswalkIdentityField.PROVIDER_CODE,
    "SMCXD06_폐암 2.xlsx": CrosswalkIdentityField.SOURCE_NO,
    "SMCXD06_폐암 3.xlsx": CrosswalkIdentityField.SOLUM_LABEL,
}
_REQUIRED_FIELDS: Final = frozenset({"source_file", "source_no", "provider_code", "solum_label"})


@dataclass(frozen=True, slots=True)
class CrosswalkImportCounts:
    rows: int
    mapped: int
    created: int
    existing: int


@dataclass(frozen=True, slots=True)
class CrosswalkRow:
    row_number: int
    source_file: str
    solum_label: str
    identity_field: CrosswalkIdentityField
    patient_id: str


@dataclass(frozen=True, slots=True)
class InvalidCrosswalkError(ValueError):
    reason: str

    def __str__(self) -> str:
        return f"invalid legacy crosswalk: {self.reason}"


@dataclass(frozen=True, slots=True)
class CrosswalkResolutionError(ValueError):
    source_file: str
    match_count: int

    def __str__(self) -> str:
        return "legacy crosswalk row did not resolve to exactly one source subject"


@dataclass(frozen=True, slots=True)
class ConflictingCrosswalkLabelError(ValueError):
    source_file: str

    def __str__(self) -> str:
        return "legacy crosswalk label conflicts with an existing sample"


def _stable_id(kind: str, key: str) -> str:
    return str(uuid5(_ID_NAMESPACE, f"{kind}:{key}"))


def _normalize_patient_id(value: str) -> str:
    stripped = value.strip()
    try:
        numeric = Decimal(stripped)
    except InvalidOperation:
        return "".join(stripped.casefold().split())
    if numeric == numeric.to_integral():
        return str(numeric.quantize(Decimal(1)))
    return "".join(stripped.casefold().split())


def _parse_rows(path: Path) -> tuple[CrosswalkRow, ...]:
    with path.open(encoding="cp949", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = frozenset(reader.fieldnames or ())
        missing = _REQUIRED_FIELDS - fields
        if missing:
            raise InvalidCrosswalkError("missing required columns")
        rows: list[CrosswalkRow] = []
        labels: set[str] = set()
        for row_number, raw in enumerate(reader, start=1):
            source_file = (raw.get("source_file") or "").strip()
            solum_label = (raw.get("solum_label") or "").strip()
            identity_field = _IDENTITY_FIELDS.get(source_file)
            if identity_field is None:
                raise InvalidCrosswalkError("unsupported source_file")
            patient_id = (raw.get(identity_field.value) or "").strip()
            if not solum_label or not patient_id:
                raise InvalidCrosswalkError("empty identity value")
            if solum_label in labels:
                raise InvalidCrosswalkError("duplicate solum_label")
            labels.add(solum_label)
            rows.append(
                CrosswalkRow(
                    row_number,
                    source_file,
                    solum_label,
                    identity_field,
                    patient_id,
                )
            )
    if not rows:
        raise InvalidCrosswalkError("no data rows")
    return tuple(rows)


def _clinical_assets(
    connection: sqlite3.Connection,
) -> dict[str, tuple[str, ...]]:
    assets: dict[str, list[str]] = {}
    for asset_id, uri in connection.execute(
        "SELECT id, uri FROM source_assets WHERE asset_kind = 'clinical'"
    ):
        name = Path(unquote(urlparse(str(uri)).path)).name
        assets.setdefault(name, []).append(str(asset_id))
    return {name: tuple(ids) for name, ids in assets.items()}


def _resolve_subject(
    connection: sqlite3.Connection,
    asset_id: str,
    row: CrosswalkRow,
) -> tuple[str, str]:
    expected = _normalize_patient_id(row.patient_id)
    matches = tuple(
        (str(subject_id), str(site_id))
        for subject_id, site_id, patient_id in connection.execute(
            """SELECT DISTINCT subject.id, subject.site_id, subject.patient_id
               FROM clinical_events AS event
               JOIN subjects AS subject ON subject.id = event.subject_id
               WHERE event.source_asset_id = ?
                 AND subject.patient_id IS NOT NULL""",
            (asset_id,),
        )
        if _normalize_patient_id(str(patient_id)) == expected
    )
    if len(matches) != 1:
        raise CrosswalkResolutionError(row.source_file, len(matches))
    return matches[0]


def _register_crosswalk_asset(
    connection: sqlite3.Connection,
    source: Path,
) -> str:
    uri = source.resolve().as_uri()
    sha256 = hash_source_file(source)
    existing = connection.execute(
        "SELECT id, sha256 FROM source_assets WHERE uri = ?",
        (uri,),
    ).fetchone()
    if existing is not None:
        if str(existing[1]) != sha256:
            raise SourceAssetChangedError(uri)
        return str(existing[0])
    asset_id = _stable_id("crosswalk-source", uri)
    connection.execute(
        """INSERT INTO source_assets (
               id, uri, sha256, asset_kind, state, size_bytes, raw_uri
           ) VALUES (?, ?, ?, 'manifest', 'ingested', ?, ?)""",
        (asset_id, uri, sha256, source.stat().st_size, uri),
    )
    return asset_id


def ingest_legacy_crosswalk(
    connection: sqlite3.Connection,
    source: Path,
) -> CrosswalkImportCounts:
    rows = _parse_rows(source)
    assets = _clinical_assets(connection)
    resolved: list[tuple[CrosswalkRow, str, str]] = []
    for row in rows:
        asset_ids = assets.get(row.source_file, ())
        if len(asset_ids) != 1:
            raise CrosswalkResolutionError(row.source_file, len(asset_ids))
        subject_id, site_id = _resolve_subject(connection, asset_ids[0], row)
        resolved.append((row, subject_id, site_id))
    crosswalk_asset_id = _register_crosswalk_asset(connection, source)
    created = 0
    existing_count = 0
    for row, subject_id, site_id in resolved:
        existing = connection.execute(
            """SELECT id, subject_id FROM samples
               WHERE site_id = ? AND solum_label = ?""",
            (site_id, row.solum_label),
        ).fetchone()
        if existing is None:
            sample_id = _stable_id("crosswalk-sample", f"{site_id}:{row.solum_label}")
            connection.execute(
                """INSERT INTO samples (
                       id, subject_id, site_id, solum_label, sample_type
                   ) VALUES (?, ?, ?, ?, 'urine')""",
                (sample_id, subject_id, site_id, row.solum_label),
            )
            created += 1
        else:
            if str(existing[1]) != subject_id:
                raise ConflictingCrosswalkLabelError(row.source_file)
            sample_id = str(existing[0])
            existing_count += 1
        connection.execute(
            """INSERT OR IGNORE INTO sample_crosswalk_sources (
                   sample_id, source_asset_id, source_row_locator, identity_field
               ) VALUES (?, ?, ?, ?)""",
            (
                sample_id,
                crosswalk_asset_id,
                f"row:{row.row_number}",
                row.identity_field.value,
            ),
        )
    return CrosswalkImportCounts(
        rows=len(rows),
        mapped=len(resolved),
        created=created,
        existing=existing_count,
    )
