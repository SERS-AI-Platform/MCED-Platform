from __future__ import annotations

import sqlite3
from pathlib import Path

from click.testing import CliRunner

from sers.cli import cli
from sers.master_data.schema import initialize_schema
from tests.master_data.clinical_registry_test_support import (
    write_complete_clinical_registry,
)


def test_cli_validates_and_idempotently_ingests_builtin_clinical_registry(
    tmp_path: Path,
) -> None:
    # Given: every fixed and prospective registry contract has a valid source.
    clinical_root = tmp_path / "clinical"
    write_complete_clinical_registry(clinical_root)
    runner = CliRunner()
    db = tmp_path / "master.db"
    raw_store = tmp_path / "raw"
    command = [
        "data",
        "ingest-clinical-registry",
        "--clinical-root",
        str(clinical_root),
        "--db",
        str(db),
        "--raw-store",
        str(raw_store),
    ]

    # When: validation and two operator ingestion runs use the built-in registry.
    validation = runner.invoke(cli, [*command, "--validate-only"])
    first = runner.invoke(cli, command)
    second = runner.invoke(cli, command)

    # Then: output is aggregate-only and persisted state is stable.
    assert validation.exit_code == first.exit_code == second.exit_code == 0
    visible = validation.output + first.output + second.output
    assert "FIXED" not in visible
    assert "sources=21" in validation.output
    assert "issues=0" in validation.output
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM source_assets").fetchone()[0] == 21
        assert connection.execute("SELECT COUNT(*) FROM clinical_events").fetchone()[0] == 25


def test_cli_registry_sqlite_failure_rolls_back_all_sources_and_blobs(
    tmp_path: Path,
) -> None:
    # Given: a complete valid registry and a trigger rejecting the second source.
    clinical_root = tmp_path / "clinical"
    write_complete_clinical_registry(clinical_root)

    db = tmp_path / "master.db"
    raw_store = tmp_path / "raw"
    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "data",
            "ingest-clinical-registry",
            "--clinical-root",
            str(clinical_root),
            "--validate-only",
        ],
    )
    with sqlite3.connect(db) as connection:
        initialize_schema(connection)
        connection.execute(
            """CREATE TRIGGER reject_second_registry_source
               BEFORE INSERT ON clinical_observations
               WHEN NEW.raw_value = 'BRE-FIXED-2'
               BEGIN
                   SELECT RAISE(ABORT, 'synthetic second-source failure');
               END"""
        )

    # When: the CLI reaches the SQLite failure in the second source.
    result = runner.invoke(
        cli,
        [
            "data",
            "ingest-clinical-registry",
            "--clinical-root",
            str(clinical_root),
            "--db",
            str(db),
            "--raw-store",
            str(raw_store),
        ],
    )

    # Then: it is nonzero and neither source nor raw blob is published.
    assert result.exit_code != 0
    assert "clinical registry ingestion failed" in result.output
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM source_assets").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM clinical_events").fetchone()[0] == 0
    assert not tuple(path for path in raw_store.rglob("*") if path.is_file())


def test_cli_registry_missing_contracts_fail_before_database_mutation(
    tmp_path: Path,
) -> None:
    # Given: configured contracts are missing except one unreadable directory.
    clinical_root = tmp_path / "clinical"
    clinical_root.mkdir()
    (clinical_root / "2. 유방암/SMCXD01_유방암.xlsx").mkdir(parents=True)
    db = tmp_path / "master.db"
    raw_store = tmp_path / "raw"

    # When: registry ingestion is requested through the public CLI.
    result = CliRunner().invoke(
        cli,
        [
            "data",
            "ingest-clinical-registry",
            "--clinical-root",
            str(clinical_root),
            "--db",
            str(db),
            "--raw-store",
            str(raw_store),
        ],
    )

    # Then: aggregate issues are nonzero and no database or blob is created.
    assert result.exit_code != 0
    assert "missing_source=" in result.output
    assert "unreadable_source=1" in result.output
    assert "expected_group_empty=2" in result.output
    assert not db.exists()
    assert not raw_store.exists()
