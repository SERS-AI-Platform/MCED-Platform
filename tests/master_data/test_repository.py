from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sers.master_data.repository import (
    create_analytical_material,
    create_ingest_batch,
    create_measurement,
    create_measurement_run,
    create_sample,
    create_source_asset,
    get_site,
    resolve_or_create_subject,
    upsert_site,
)
from sers.master_data.schema import initialize_schema
from sers.master_data.types import (
    AnalyticalMaterialDraft,
    IngestBatchDraft,
    MeasurementDraft,
    MeasurementRunDraft,
    SampleDraft,
    SiteDraft,
    SiteId,
    SourceAssetDraft,
)

SYNTHETIC_SHA256 = "Aa" * 32


@pytest.fixture
def database(tmp_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(tmp_path / "master.db")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
    initialize_schema(connection)
    yield connection
    connection.close()


def test_site_upsert_updates_one_stable_record(database: sqlite3.Connection) -> None:
    # Given: one registered site.
    first = upsert_site(database, SiteDraft(code="SITE-A", name="Site A"))

    # When: the same code is registered with updated metadata.
    second = upsert_site(database, SiteDraft(code="SITE-A", name="Site A Updated"))

    # Then: the identity is stable and only one row exists.
    assert second.id == first.id
    assert get_site(database, first.id) == second
    assert database.execute("SELECT COUNT(*) FROM sites").fetchone()[0] == 1


def test_subject_key_is_site_scoped_and_stored_verbatim(
    database: sqlite3.Connection,
) -> None:
    # Given: two distinct sites and one synthetic hospital-issued key.
    site_a = upsert_site(database, SiteDraft(code="SITE-A", name="Site A"))
    site_b = upsert_site(database, SiteDraft(code="SITE-B", name="Site B"))
    source_key = "SYNTHETIC-0001"

    # When: the key is resolved repeatedly at site A and once at site B.
    first = resolve_or_create_subject(database, site_a.id, source_key)
    repeated = resolve_or_create_subject(database, site_a.id, source_key)
    cross_site = resolve_or_create_subject(database, site_b.id, source_key)

    # Then: same-site resolution is idempotent and cross-site identity is separate.
    assert repeated == first
    assert cross_site.subject_id != first.subject_id
    assert first.patient_id == source_key
    assert (
        database.execute(
            "SELECT patient_id FROM subjects WHERE id = ?",
            (first.subject_id,),
        ).fetchone()[0]
        == source_key
    )


def test_create_measurement_primitives_persists_foreign_key_chain(
    database: sqlite3.Connection,
) -> None:
    # Given: a site, subject, asset, and ingest batch.
    site = upsert_site(database, SiteDraft(code="SITE-A", name="Site A"))
    subject = resolve_or_create_subject(database, site.id, "SYNTHETIC-0002")
    asset_id = create_source_asset(
        database,
        SourceAssetDraft(
            site_id=site.id,
            uri="synthetic://spectrum-1",
            sha256=SYNTHETIC_SHA256,
            asset_kind="spectrum",
            size_bytes=128,
        ),
    )
    batch_id = create_ingest_batch(
        database,
        IngestBatchDraft(source_asset_id=asset_id, parser_name="synthetic-parser"),
    )
    sample_id = create_sample(
        database,
        SampleDraft(
            subject_id=subject.subject_id,
            site_id=site.id,
            sample_type="urine",
            solum_label="SAMPLE-1",
        ),
    )
    material_id = create_analytical_material(
        database,
        AnalyticalMaterialDraft(sample_id=sample_id, material_type="primary"),
    )
    run_id = create_measurement_run(
        database,
        MeasurementRunDraft(
            site_id=site.id,
            instrument_key="SYNTH-INSTRUMENT",
            ingest_batch_id=batch_id,
        ),
    )

    # When: a measurement is created.
    measurement_id = create_measurement(
        database,
        MeasurementDraft(
            measurement_run_id=run_id,
            analytical_material_id=material_id,
            replicate_index=1,
        ),
    )

    # Then: the stored row references the requested run and material.
    row = database.execute(
        """SELECT measurement_run_id, analytical_material_id, replicate_index
           FROM measurements WHERE id = ?""",
        (measurement_id,),
    ).fetchone()
    assert tuple(row) == (run_id, material_id, 1)


def test_append_only_create_rejects_duplicate_source_asset(
    database: sqlite3.Connection,
) -> None:
    # Given: one persisted source asset.
    site = upsert_site(database, SiteDraft(code="SITE-A", name="Site A"))
    draft = SourceAssetDraft(
        site_id=site.id,
        uri="synthetic://immutable",
        sha256=SYNTHETIC_SHA256,
        asset_kind="clinical",
    )
    create_source_asset(database, draft)

    # When: the same immutable asset is created again.
    with pytest.raises(sqlite3.IntegrityError):
        create_source_asset(database, draft)

    # Then: the original row is not overwritten.
    assert database.execute("SELECT COUNT(*) FROM source_assets").fetchone()[0] == 1


@pytest.mark.parametrize("invalid_hash", [" " * 64, "z" * 64])
def test_non_hex_source_asset_hash_is_rejected_by_repository(
    database: sqlite3.Connection,
    invalid_hash: str,
) -> None:
    # Given: a valid site and malformed source hash.
    site = upsert_site(database, SiteDraft(code="SITE-A", name="Site A"))
    draft = SourceAssetDraft(
        site_id=site.id,
        uri="synthetic://malformed",
        sha256=invalid_hash,
        asset_kind="clinical",
    )

    # When: the malformed asset crosses the persistence boundary.
    with pytest.raises(ValueError):
        create_source_asset(database, draft)

    # Then: no source asset is persisted.
    assert database.execute("SELECT COUNT(*) FROM source_assets").fetchone()[0] == 0


def test_subject_creation_rolls_back_when_source_key_insert_fails(
    database: sqlite3.Connection,
) -> None:
    # Given: a site identifier that does not exist.
    missing_site_id = SiteId("00000000-0000-0000-0000-000000000000")

    # When: source-key insertion fails and the caller catches and commits the error.
    with pytest.raises(sqlite3.IntegrityError):
        resolve_or_create_subject(database, missing_site_id, "SYNTHETIC-ORPHAN-CHECK")
    database.commit()

    # Then: the failed identity creation leaves no orphan subject.
    assert database.execute("SELECT COUNT(*) FROM subjects").fetchone()[0] == 0
