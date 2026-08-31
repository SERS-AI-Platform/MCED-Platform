from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from click.testing import CliRunner

from sers.cli import cli
from sers.master_data.schema import initialize_schema


@dataclass(frozen=True, slots=True)
class SeedSubject:
    source_file: str
    site_id: str
    subject_id: str
    patient_id: str


def _seed_subject(
    connection: sqlite3.Connection,
    draft: SeedSubject,
) -> None:
    asset_id = f"asset-{draft.subject_id}"
    connection.execute(
        "INSERT OR IGNORE INTO sites (id, code, name) VALUES (?, ?, ?)",
        (draft.site_id, draft.site_id, draft.site_id),
    )
    connection.execute(
        """INSERT INTO source_assets (
               id, site_id, uri, sha256, asset_kind, state
           ) VALUES (?, ?, ?, ?, 'clinical', 'ingested')""",
        (
            asset_id,
            draft.site_id,
            (Path("/clinical") / draft.source_file).as_uri(),
            "a" * 64,
        ),
    )
    connection.execute(
        "INSERT INTO subjects (id, site_id, patient_id) VALUES (?, ?, ?)",
        (draft.subject_id, draft.site_id, draft.patient_id),
    )
    connection.execute(
        """INSERT INTO clinical_events (
               id, subject_id, site_id, source_asset_id, event_type
           ) VALUES (?, ?, ?, ?, 'other')""",
        (
            f"event-{draft.subject_id}",
            draft.subject_id,
            draft.site_id,
            asset_id,
        ),
    )


def test_cli_ingests_authoritative_legacy_crosswalk_with_provenance(
    tmp_path: Path,
) -> None:
    # Given: three source-specific patient identities and their official SoluM labels.
    database = tmp_path / "master.db"
    with sqlite3.connect(database) as connection:
        initialize_schema(connection)
        _seed_subject(
            connection,
            SeedSubject(
                "SMCXD01_전립선암 임상정보.xlsx",
                "site-pro",
                "subject-pro",
                "1",
            ),
        )
        _seed_subject(
            connection,
            SeedSubject(
                "SMCXD01_난소암 1.xlsx",
                "site-ova",
                "subject-ova",
                "20636026",
            ),
        )
        _seed_subject(
            connection,
            SeedSubject(
                "SMCXD06_폐암 3.xlsx",
                "site-lun",
                "subject-lun",
                "LUN 246",
            ),
        )
    source = tmp_path / "clinical_unified_202604172056.csv"
    source.write_text(
        "source_file,source_no,provider_code,solum_label\n"
        "SMCXD01_전립선암 임상정보.xlsx,1.0,,PRO 1\n"
        "SMCXD01_난소암 1.xlsx,,20636026.0,OVA 23\n"
        "SMCXD06_폐암 3.xlsx,,,LUN 246\n",
        encoding="cp949",
    )

    # When: the governed crosswalk command is invoked.
    result = CliRunner().invoke(
        cli,
        [
            "data",
            "ingest-legacy-crosswalk",
            "--db",
            str(database),
            "--source",
            str(source),
        ],
    )

    # Then: every row becomes a site-scoped sample with source-row provenance.
    assert result.exit_code == 0
    assert "rows=3,mapped=3,created=3" in result.output
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 3
        assert (
            connection.execute("SELECT COUNT(*) FROM sample_crosswalk_sources").fetchone()[0] == 3
        )
