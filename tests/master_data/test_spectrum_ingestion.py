from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sers.master_data.inventory import inventory_spectra
from sers.master_data.raw_store import store_raw_file
from sers.master_data.repository import create_analytical_material, upsert_site
from sers.master_data.schema import initialize_schema
from sers.master_data.spectrum_ingestion import (
    SpectrumIngestionRequest,
    ingest_spectra,
)
from sers.master_data.spectrum_persistence import SourceAssetChangedError
from sers.master_data.spectrum_schema import initialize_spectrum_schema
from sers.master_data.spectrum_types import InventoryRoot, SourceKind
from sers.master_data.types import AnalyticalMaterialDraft, SiteDraft


def _write_spectrum(path: Path, intensity: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"100,{intensity}\n101,{intensity + 1}\n", encoding="utf-8")


def _database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
    initialize_schema(connection)
    initialize_spectrum_schema(connection)
    return connection


def test_material_only_lineage_requires_no_subject_or_sample(tmp_path: Path) -> None:
    # Given: a v4 database without any subject/sample rows.
    with _database(tmp_path / "clinical.db") as connection:
        # When: a standalone analytical material is created.
        material_id = create_analytical_material(
            connection,
            AnalyticalMaterialDraft(sample_id=None, material_type="primary"),
        )

        # Then: the material exists without manufacturing an identity.
        assert material_id
        assert connection.execute("SELECT COUNT(*) FROM subjects").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 0


def test_ingestion_is_idempotent_and_separates_materials(tmp_path: Path) -> None:
    # Given: replicate, average, blank, reference, and malformed source files.
    root = tmp_path / "20260429_Urine test"
    _write_spectrum(root / "6. PRO" / "PRO 7_1.CSV")
    _write_spectrum(root / "6. PRO" / "PRO 7_ave.CSV", intensity=2)
    _write_spectrum(root / "0. Blank" / "Blank 8-5_1.CSV")
    _write_spectrum(root / "0. Reference" / "PS 1_1.CSV")
    _write_spectrum(root / "malformed.CSV")
    report = inventory_spectra(
        (
            InventoryRoot(
                path=root,
                source_kind=SourceKind.REMEASUREMENT,
                instrument_key="Thermo",
                preparation="liquid",
            ),
        )
    )

    # When: the same inventory is ingested twice into a real SQLite/raw store.
    with _database(tmp_path / "clinical.db") as connection:
        site = upsert_site(connection, SiteDraft(code="SYNTH", name="Synthetic"))
        connection.commit()
        request = SpectrumIngestionRequest(
            report=report,
            site_id=site.id,
            raw_store=tmp_path / "raw",
        )
        first = ingest_spectra(connection, request)
        second = ingest_spectra(connection, request)
        counts = tuple(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "source_assets",
                "ingest_batches",
                "measurement_runs",
                "measurements",
                "measurement_artifacts",
                "analytical_materials",
            )
        )
        material_kinds = {
            row[0]
            for row in connection.execute("SELECT material_kind FROM spectrum_material_metadata")
        }
        batch_states = tuple(
            tuple(row)
            for row in connection.execute(
                """SELECT sir.status, ib.status
                   FROM spectrum_inventory_records AS sir
                   JOIN ingest_batches AS ib ON ib.source_asset_id = sir.source_asset_id
                   JOIN source_assets AS asset ON asset.id = sir.source_asset_id
                   ORDER BY sir.status, asset.uri"""
            )
        )

        # Then: row counts and hash paths are stable; average is not a measurement.
        assert first == second
        assert counts == (5, 5, 3, 3, 4, 3)
        assert material_kinds == {"biological", "blank", "reference"}
        assert batch_states == (
            ("derived", "completed"),
            ("quarantined", "failed"),
            ("ready", "completed"),
            ("ready", "completed"),
            ("ready", "completed"),
        )
        assert connection.execute("SELECT COUNT(*) FROM subjects").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 0
        assert (
            connection.execute(
                """SELECT COUNT(*) FROM spectrum_inventory_records
               WHERE status = 'quarantined'"""
            ).fetchone()[0]
            == 1
        )
        raw_uris = [
            Path(row[0])
            for row in connection.execute("SELECT raw_uri FROM source_assets ORDER BY raw_uri")
        ]
        assert all(path.name == path.parent.name for path in raw_uris)


def test_content_addressed_store_never_overwrites_existing_blob(tmp_path: Path) -> None:
    # Given: two paths with identical immutable bytes.
    first_source = tmp_path / "one.CSV"
    second_source = tmp_path / "two.CSV"
    _write_spectrum(first_source)
    _write_spectrum(second_source)

    # When: both are stored in the same content-addressed raw store.
    first = store_raw_file(first_source, tmp_path / "raw")
    second = store_raw_file(second_source, tmp_path / "raw")

    # Then: they resolve to the same complete blob without duplicate writes.
    assert second == first
    assert Path(first.raw_uri).read_bytes() == first_source.read_bytes()


def test_ingestion_persists_postoperative_specimen_timing(tmp_path: Path) -> None:
    # Given: one post-operative spectrum identified by the confirmed Po. prefix.
    root = tmp_path / "prospective"
    _write_spectrum(root / "Po. YPAN 7_1.CSV")
    report = inventory_spectra(
        (
            InventoryRoot(
                path=root,
                source_kind=SourceKind.BORAMAE_POWDER,
                instrument_key="Thermo",
                preparation="powder",
            ),
        )
    )

    # When: the inventory is persisted through the real SQLite adapter.
    with _database(tmp_path / "clinical.db") as connection:
        site = upsert_site(connection, SiteDraft(code="POSTOP", name="Post-op"))
        connection.commit()
        ingest_spectra(
            connection,
            SpectrumIngestionRequest(report, site.id, tmp_path / "raw"),
        )
        timing = tuple(
            connection.execute(
                """SELECT inventory.specimen_timing, material.specimen_timing
                   FROM spectrum_inventory_records AS inventory
                   JOIN measurement_artifacts AS artifact
                     ON artifact.source_asset_id = inventory.source_asset_id
                   JOIN measurements AS measurement
                     ON measurement.id = artifact.measurement_id
                   JOIN spectrum_material_metadata AS material
                     ON material.analytical_material_id =
                        measurement.analytical_material_id"""
            ).fetchone()
        )

    # Then: both inventory and analytical-material metadata preserve the timing.
    assert timing == ("post_operative", "post_operative")


def test_stale_source_rejection_preserves_raw_blob_set(tmp_path: Path) -> None:
    # Given: one successfully ingested source whose bytes then change in place.
    root = tmp_path / "20260429_Urine test"
    source = root / "PRO 9_1.CSV"
    _write_spectrum(source)
    report = inventory_spectra(
        (
            InventoryRoot(
                path=root,
                source_kind=SourceKind.REMEASUREMENT,
                instrument_key="Thermo",
                preparation="liquid",
            ),
        )
    )
    with _database(tmp_path / "clinical.db") as connection:
        site = upsert_site(connection, SiteDraft(code="STALE", name="Stale"))
        connection.commit()
        request = SpectrumIngestionRequest(report, site.id, tmp_path / "raw")
        ingest_spectra(connection, request)
        original_blobs = {
            path.relative_to(tmp_path / "raw")
            for path in (tmp_path / "raw").rglob("*")
            if path.is_file()
        }
        _write_spectrum(source, intensity=99)

        # When: the changed source URI is ingested again.
        with pytest.raises(SourceAssetChangedError):
            ingest_spectra(connection, request)

        # Then: rejection publishes no unreferenced content-addressed blob.
        current_blobs = {
            path.relative_to(tmp_path / "raw")
            for path in (tmp_path / "raw").rglob("*")
            if path.is_file()
        }
        assert current_blobs == original_blobs


def test_malformed_spectrum_content_marks_ingest_batch_failed(tmp_path: Path) -> None:
    # Given: a correctly named spectrum file whose content is not a spectrum.
    root = tmp_path / "20260429_Urine test"
    source = root / "PRO 10_1.CSV"
    source.parent.mkdir(parents=True)
    source.write_text("not,spectral,data\n", encoding="utf-8")
    report = inventory_spectra(
        (
            InventoryRoot(
                path=root,
                source_kind=SourceKind.REMEASUREMENT,
                instrument_key="Thermo",
                preparation="liquid",
            ),
        )
    )

    # When: malformed content crosses the ingestion boundary.
    with _database(tmp_path / "clinical.db") as connection:
        site = upsert_site(connection, SiteDraft(code="BAD", name="Malformed"))
        connection.commit()
        ingest_spectra(
            connection,
            SpectrumIngestionRequest(report, site.id, tmp_path / "raw"),
        )
        row = connection.execute(
            """SELECT sir.status, sir.reason_code, ib.status
               FROM spectrum_inventory_records AS sir
               JOIN ingest_batches AS ib ON ib.source_asset_id = sir.source_asset_id"""
        ).fetchone()

        # Then: inventory and ingest-batch truthfully report the failure.
        assert tuple(row) == ("quarantined", "malformed_spectrum", "failed")
