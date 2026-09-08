from __future__ import annotations

from pathlib import Path

import pytest

from sers.master_data.clinical_ingestion import (
    ClinicalIngestionRequest,
    EmptyClinicalSourceError,
    ingest_clinical_source,
)
from sers.master_data.clinical_parser import (
    MalformedClinicalSourceError,
    MissingSubjectKeyError,
)
from sers.master_data.clinical_types import ClinicalFormat, ClinicalSource
from sers.master_data.spectrum_persistence import SourceAssetChangedError
from tests.master_data.clinical_test_support import (
    database,
    source,
    write_csv,
    write_excel,
)


def test_ingestion_preserves_site_scoped_identity_and_longitudinal_rows(
    tmp_path: Path,
) -> None:
    # Given: repeated CSV rows and an Excel row at another site using the same key.
    csv_path = tmp_path / "site-a.csv"
    excel_path = tmp_path / "site-b.xlsx"
    source_key = " SYNTHETIC-001 "
    write_csv(csv_path, (f"{source_key},V1,12.5,alpha", f"{source_key},V2,13.0,beta"))
    write_excel(excel_path, source_key)

    # When: both raw sources are ingested into a real v4 database.
    with database(tmp_path / "clinical.db") as connection:
        first = ingest_clinical_source(
            connection,
            ClinicalIngestionRequest(
                source(csv_path, "SITE-A", ClinicalFormat.CSV),
                tmp_path / "raw",
            ),
        )
        second = ingest_clinical_source(
            connection,
            ClinicalIngestionRequest(
                source(excel_path, "SITE-B", ClinicalFormat.EXCEL),
                tmp_path / "raw",
            ),
        )
        stored_keys = tuple(
            row[0]
            for row in connection.execute(
                "SELECT patient_id FROM subjects ORDER BY site_id"
            )
        )

        # Then: same-site rows deduplicate, cross-site identity separates, and rows remain events.
        assert first.subjects == 1
        assert second.subjects == 2
        assert stored_keys == (source_key, source_key)
        assert connection.execute("SELECT COUNT(*) FROM subjects").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM clinical_events").fetchone()[0] == 3
        assert connection.execute(
            """SELECT COUNT(*) FROM clinical_observations
               WHERE source_field_name IN ('mystery_field', 'unknown_excel_field')
                 AND normalization_status = 'unmapped'"""
        ).fetchone()[0] == 3


def test_solum_label_creates_sample_when_hospital_patient_id_is_missing(
    tmp_path: Path,
) -> None:
    # Given: a clinical row with a spectrum matching ID but no hospital patient ID.
    source_path = tmp_path / "nullable-patient.csv"
    source_path.write_text(
        "patient_id,solum_label,value\n,SOLUM-001,7\n",
        encoding="utf-8-sig",
    )
    clinical_source = ClinicalSource(
        path=source_path,
        site_code="SITE-A",
        site_name="Synthetic SITE-A",
        protocol_code="SYNTHETIC",
        source_group="SYNTHETIC",
        source_format=ClinicalFormat.CSV,
        patient_id_fields=("patient_id",),
        event_type="lab",
    )

    # When: the row is ingested.
    with database(tmp_path / "clinical.db") as connection:
        ingest_clinical_source(
            connection,
            ClinicalIngestionRequest(clinical_source, tmp_path / "raw"),
        )

        # Then: patient_id stays null and solum_label is the sample matching ID.
        assert connection.execute(
            "SELECT patient_id FROM subjects"
        ).fetchone()[0] is None
        assert connection.execute(
            "SELECT solum_label FROM samples"
        ).fetchone()[0] == "SOLUM-001"


def test_double_ingestion_is_stable_and_records_source_aliases(tmp_path: Path) -> None:
    # Given: two immutable files for one site and one pseudonymous source key.
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    write_csv(first_path, ("SYNTHETIC-002,V1,1.0,x",))
    write_csv(second_path, ("SYNTHETIC-002,V2,2.0,y",))

    # When: both sources and then the first source again are ingested.
    with database(tmp_path / "clinical.db") as connection:
        raw_store = tmp_path / "raw"
        first_request = ClinicalIngestionRequest(
            source(first_path, "SITE-A", ClinicalFormat.CSV),
            raw_store,
        )
        ingest_clinical_source(connection, first_request)
        ingest_clinical_source(
            connection,
            ClinicalIngestionRequest(
                source(second_path, "SITE-A", ClinicalFormat.CSV),
                raw_store,
            ),
        )
        tables = (
            "source_assets",
            "ingest_batches",
            "subjects",
            "subject_source_aliases",
            "clinical_events",
            "clinical_observations",
        )
        before = tuple(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in tables
        )
        ingest_clinical_source(connection, first_request)
        after = tuple(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in tables
        )

        # Then: identities/events are stable while both source-file aliases remain.
        assert before == after
        assert after[:5] == (2, 2, 1, 2, 2)


def test_missing_subject_key_fails_before_any_database_write(tmp_path: Path) -> None:
    # Given: a source containing a valid row followed by a row with an empty key.
    source_path = tmp_path / "missing.csv"
    write_csv(source_path, ("SYNTHETIC-003,V1,1.0,x", ",V2,2.0,y"))

    # When: the full source is parsed before persistence.
    with database(tmp_path / "clinical.db") as connection:
        with pytest.raises(MissingSubjectKeyError):
            ingest_clinical_source(
                connection,
                ClinicalIngestionRequest(
                    source(source_path, "SITE-A", ClinicalFormat.CSV),
                    tmp_path / "raw",
                ),
            )

        # Then: no partial source, subject, event, or observation state is visible.
        assert tuple(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "source_assets",
                "subjects",
                "clinical_events",
                "clinical_observations",
            )
        ) == (0, 0, 0, 0)


def test_changed_source_is_rejected_without_mutating_ingested_state(
    tmp_path: Path,
) -> None:
    # Given: an ingested source whose bytes change at the same URI.
    source_path = tmp_path / "stale.csv"
    write_csv(source_path, ("SYNTHETIC-004,V1,1.0,x",))
    request = ClinicalIngestionRequest(
        source(source_path, "SITE-A", ClinicalFormat.CSV),
        tmp_path / "raw",
    )
    with database(tmp_path / "clinical.db") as connection:
        ingest_clinical_source(connection, request)
        before = connection.total_changes
        write_csv(source_path, ("SYNTHETIC-004,V1,99.0,changed",))

        # When: the stale source URI is submitted again.
        with pytest.raises(SourceAssetChangedError):
            ingest_clinical_source(connection, request)

        # Then: the database is unchanged after stale-source rejection.
        assert connection.total_changes == before


def test_malformed_excel_fails_before_any_database_write(tmp_path: Path) -> None:
    # Given: a file named as an Excel source whose bytes are not an OOXML workbook.
    source_path = tmp_path / "malformed.xlsx"
    source_path.write_bytes(b"not-an-excel-workbook")

    # When: the malformed source crosses the parser boundary.
    with database(tmp_path / "clinical.db") as connection:
        with pytest.raises(MalformedClinicalSourceError):
            ingest_clinical_source(
                connection,
                ClinicalIngestionRequest(
                    source(source_path, "SITE-A", ClinicalFormat.EXCEL),
                    tmp_path / "raw",
                ),
            )

        # Then: malformed input cannot publish source or clinical rows.
        assert connection.execute("SELECT COUNT(*) FROM source_assets").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM clinical_events").fetchone()[0] == 0


def test_empty_source_cannot_report_success(tmp_path: Path) -> None:
    # Given: a syntactically valid CSV source with headers and no clinical rows.
    source_path = tmp_path / "empty.csv"
    write_csv(source_path, ())

    # When: the empty source is submitted for ingestion.
    with database(tmp_path / "clinical.db") as connection:
        with pytest.raises(EmptyClinicalSourceError):
            ingest_clinical_source(
                connection,
                ClinicalIngestionRequest(
                    source(source_path, "SITE-A", ClinicalFormat.CSV),
                    tmp_path / "raw",
                ),
            )

        # Then: the source is not misleadingly marked ingested.
        assert connection.execute("SELECT COUNT(*) FROM source_assets").fetchone()[0] == 0
