from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Literal
from uuid import UUID, uuid5

from ._compat import assert_never
from .spectrum_source_version import (
    SourceAssetChangedError,
    assert_source_version,
)
from .spectrum_types import InventoryItem, InventoryStatus, StoredRaw
from .types import SiteId

__all__ = (
    "PersistItem",
    "SourceAssetChangedError",
    "assert_source_version",
    "persist_item",
)

_ID_NAMESPACE = UUID("45954924-56d7-47c5-a761-275287ae61aa")


@dataclass(frozen=True, slots=True)
class PersistItem:
    item: InventoryItem
    stored: StoredRaw
    site_id: SiteId
    status: InventoryStatus
    reason_code: str | None


def _stable_id(kind: str, key: str) -> str:
    return str(uuid5(_ID_NAMESPACE, f"{kind}:{key}"))


def _source_asset(connection: sqlite3.Connection, draft: PersistItem) -> str:
    source_uri = draft.item.source_path.resolve().as_uri()
    existing = connection.execute(
        "SELECT id, sha256 FROM source_assets WHERE uri = ?",
        (source_uri,),
    ).fetchone()
    if existing is not None:
        if existing[1] != draft.stored.sha256:
            raise SourceAssetChangedError(source_uri)
        asset_id = str(existing[0])
    else:
        asset_id = _stable_id("source-asset", source_uri)
        connection.execute(
            """INSERT INTO source_assets (
                   id, site_id, uri, sha256, asset_kind, state, size_bytes, raw_uri
               ) VALUES (?, ?, ?, ?, 'spectrum', 'discovered', ?, ?)""",
            (
                asset_id,
                draft.site_id,
                source_uri,
                draft.stored.sha256,
                draft.stored.size_bytes,
                draft.stored.raw_uri,
            ),
        )
    return asset_id


def _set_batch_status(
    connection: sqlite3.Connection,
    asset_id: str,
    status: Literal["started", "completed", "failed"],
) -> None:
    connection.execute(
        """INSERT INTO ingest_batches (
               id, source_asset_id, status, parser_name, completed_at
           ) VALUES (
               ?, ?, ?, 'spectrum-v1',
               CASE WHEN ? = 'started' THEN NULL ELSE CURRENT_TIMESTAMP END
           )
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                completed_at = excluded.completed_at""",
        (_stable_id("ingest-batch", asset_id), asset_id, status, status),
    )


def _material_key(item: InventoryItem) -> str:
    parsed = item.parsed
    assert parsed is not None
    return "|".join(
        (
            item.root_key,
            item.acquisition_date,
            item.instrument_key,
            parsed.group_code,
            parsed.source_sample_code,
            parsed.preparation,
            parsed.fasting_state.value,
            parsed.specimen_timing.value,
            parsed.lot_code or "",
            parsed.material_kind.value,
        )
    )


def _material(
    connection: sqlite3.Connection,
    item: InventoryItem,
    site_id: SiteId,
) -> str:
    parsed = item.parsed
    assert parsed is not None
    material_key = _material_key(item)
    material_id = _stable_id("material", f"{site_id}:{material_key}")
    connection.execute(
        """INSERT OR IGNORE INTO analytical_materials (id, sample_id, material_type)
           VALUES (?, NULL, 'primary')""",
        (material_id,),
    )
    connection.execute(
        """INSERT OR IGNORE INTO spectrum_material_metadata (
               analytical_material_id, site_id, material_key, canonical_source_code,
               source_group, material_kind, preparation, fasting_state, lot_code,
               specimen_timing, identity_status,
               alias_candidates
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            material_id,
            site_id,
            material_key,
            parsed.solum_label,
            parsed.group_code,
            parsed.material_kind.value,
            parsed.preparation,
            parsed.fasting_state.value,
            parsed.lot_code,
            parsed.specimen_timing.value,
            parsed.identity_status.value,
            ",".join(parsed.alias_candidates),
        ),
    )
    return material_id


def _measurement(
    connection: sqlite3.Connection,
    draft: PersistItem,
    asset_id: str,
) -> None:
    parsed = draft.item.parsed
    assert parsed is not None
    assert parsed.replicate_index is not None
    material_id = _material(connection, draft.item, draft.site_id)
    run_key = f"{draft.site_id}:{_material_key(draft.item)}"
    run_id = _stable_id("measurement-run", run_key)
    connection.execute(
        """INSERT OR IGNORE INTO measurement_runs (
               id, site_id, ingest_batch_id, instrument_key, status, acquired_at
           ) VALUES (?, ?, ?, ?, 'complete', ?)""",
        (
            run_id,
            draft.site_id,
            _stable_id("ingest-batch", asset_id),
            draft.item.instrument_key,
            draft.item.acquisition_date,
        ),
    )
    measurement_id = _stable_id(
        "measurement",
        f"{run_id}:{material_id}:{parsed.replicate_index}",
    )
    connection.execute(
        """INSERT OR IGNORE INTO measurements (
               id, measurement_run_id, analytical_material_id, replicate_index
           ) VALUES (?, ?, ?, ?)""",
        (measurement_id, run_id, material_id, parsed.replicate_index),
    )
    connection.execute(
        """INSERT OR IGNORE INTO measurement_artifacts (
               id, measurement_id, source_asset_id, artifact_role
           ) VALUES (?, ?, ?, 'raw')""",
        (_stable_id("artifact", f"{measurement_id}:raw"), measurement_id, asset_id),
    )


def _average(
    connection: sqlite3.Connection,
    draft: PersistItem,
    asset_id: str,
) -> bool:
    material_id = _material(connection, draft.item, draft.site_id)
    row = connection.execute(
        """SELECT id FROM measurements
           WHERE analytical_material_id = ?
           ORDER BY replicate_index LIMIT 1""",
        (material_id,),
    ).fetchone()
    if row is None:
        return False
    measurement_id = row[0]
    connection.execute(
        """INSERT OR IGNORE INTO measurement_artifacts (
               id, measurement_id, source_asset_id, artifact_role
           ) VALUES (?, ?, ?, 'processed')""",
        (_stable_id("artifact", f"{measurement_id}:processed"), measurement_id, asset_id),
    )
    return True


def _inventory_record(
    connection: sqlite3.Connection,
    draft: PersistItem,
    asset_id: str,
    status: InventoryStatus,
    reason_code: str | None,
) -> None:
    parsed = draft.item.parsed
    connection.execute(
        """INSERT INTO spectrum_inventory_records (
               source_asset_id, source_kind, status, reason_code,
               acquisition_date, instrument_key, root_key, canonical_source_code,
               preparation, fasting_state, lot_code, specimen_timing,
               identity_status, alias_candidates
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(source_asset_id) DO NOTHING""",
        (
            asset_id,
            draft.item.source_kind.value,
            status.value,
            reason_code,
            draft.item.acquisition_date,
            draft.item.instrument_key,
            draft.item.root_key,
            None if parsed is None else parsed.solum_label,
            None if parsed is None else parsed.preparation,
            ("unspecified" if parsed is None else parsed.fasting_state.value),
            None if parsed is None else parsed.lot_code,
            ("unspecified" if parsed is None else parsed.specimen_timing.value),
            None if parsed is None else parsed.identity_status.value,
            "" if parsed is None else ",".join(parsed.alias_candidates),
        ),
    )


def persist_item(connection: sqlite3.Connection, draft: PersistItem) -> None:
    asset_id = _source_asset(connection, draft)
    _set_batch_status(connection, asset_id, "started")
    status = draft.status
    reason_code = draft.reason_code
    match status:
        case InventoryStatus.READY:
            _measurement(connection, draft, asset_id)
        case InventoryStatus.DERIVED:
            if not _average(connection, draft, asset_id):
                status = InventoryStatus.QUARANTINED
                reason_code = "average_without_measurement"
        case InventoryStatus.QUARANTINED:
            pass
        case InventoryStatus.EXCLUDED:
            raise AssertionError("excluded items must not cross the persistence boundary")
        case unreachable:
            assert_never(unreachable)
    _inventory_record(connection, draft, asset_id, status, reason_code)
    _set_batch_status(
        connection,
        asset_id,
        "failed" if status is InventoryStatus.QUARANTINED else "completed",
    )
    connection.execute(
        "UPDATE source_assets SET state = ? WHERE id = ?",
        ("failed" if status is InventoryStatus.QUARANTINED else "ingested", asset_id),
    )
