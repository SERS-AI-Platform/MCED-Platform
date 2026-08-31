from __future__ import annotations

import csv
import sqlite3
import subprocess
from pathlib import Path
from typing import NoReturn

import pytest
from click.testing import CliRunner

from sers.cli import cli
from tests.master_data.data_cli_test_support import (
    build_manifest,
    invoke_ingestion,
    invoke_labels,
    write_inputs,
)


class InjectedUnexpectedError(RuntimeError):
    pass


def test_data_help_loads_without_mlflow() -> None:
    # Given: the core package without requesting optional experiment tracking.
    runner = CliRunner()

    # When: the canonical data CLI help is loaded.
    result = runner.invoke(cli, ["data", "--help"])

    # Then: every master-data workflow is discoverable without importing MLflow.
    assert result.exit_code == 0
    assert "ingest-clinical" in result.output
    assert "export-dataset" in result.output


def test_installed_cli_help_works_outside_repository(tmp_path: Path) -> None:
    # Given: the installed console script runs without the repository on sys.path.
    command = ("sers", "data", "--help")

    # When: help is requested from an unrelated working directory.
    result = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then: package-local imports are complete and MLflow is still optional.
    assert result.returncode == 0
    assert "ingest-clinical" in result.stdout
    assert "mlflow" not in result.stderr.casefold()


def test_validate_reports_malformed_spectrum_without_source_code(
    tmp_path: Path,
) -> None:
    # Given: one source-code-bearing filename with malformed spectrum bytes.
    source_code = "SECRET999"
    (tmp_path / f"PRO {source_code}_1.csv").write_text(
        "not,a,spectrum\n",
        encoding="utf-8",
    )

    # When: validation runs through the canonical CLI.
    result = CliRunner().invoke(
        cli,
        ["data", "validate", "--input", str(tmp_path)],
    )

    # Then: failure is aggregate-only and does not disclose the source code.
    assert result.exit_code != 0
    assert source_code not in result.output
    assert "Traceback" not in result.output
    assert "issues=1" in result.output
    assert "spectrum validation failed" in result.output


def test_validate_propagates_unexpected_programmer_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a valid candidate whose reader raises an unexpected implementation fault.
    (tmp_path / "PRO 001_1.csv").write_text("100,1\n101,2\n", encoding="utf-8")

    def raise_unexpected(_path: Path) -> NoReturn:
        raise InjectedUnexpectedError

    monkeypatch.setattr("sers.io.read_spectrum", raise_unexpected)

    # When/Then: validation does not misclassify or suppress the programmer error.
    with pytest.raises(InjectedUnexpectedError):
        CliRunner().invoke(
            cli,
            ["data", "validate", "--input", str(tmp_path)],
            catch_exceptions=False,
        )


def test_cli_runs_private_idempotent_master_data_workflow(tmp_path: Path) -> None:
    # Given: one clinical fact and one spectrum sharing a pseudonymous source key.
    source_key = "PSEUDONYM001"
    clinical, spectra = write_inputs(tmp_path, source_key)
    db = tmp_path / "master.db"
    raw_store = tmp_path / "raw"
    runner = CliRunner()

    # When: inventory, ingestion, matching, labels, reconciliation, and export run.
    inventory = runner.invoke(
        cli,
        [
            "data",
            "inventory",
            "--db",
            str(db),
            "--clinical-root",
            str(tmp_path),
            "--spectrum-root",
            str(spectra),
        ],
    )
    first_outputs = invoke_ingestion(runner, db, raw_store, clinical, spectra)
    match = runner.invoke(
        cli,
        ["data", "match", "--db", str(db), "--rule-version", "exact-v1"],
    )
    invoke_labels(runner, db)
    manifest_id = build_manifest(db)
    reconciliation = runner.invoke(
        cli,
        ["data", "reconcile", "--db", str(db), "--rule-version", "exact-v1"],
    )
    output = tmp_path / "dataset.csv"
    exported = runner.invoke(
        cli,
        [
            "data",
            "export-dataset",
            "--db",
            str(db),
            "--manifest-id",
            manifest_id,
            "--output",
            str(output),
        ],
    )
    second_outputs = invoke_ingestion(runner, db, raw_store, clinical, spectra)

    # Then: the workflow is stable and no patient/sample source key leaves the CLI.
    results = (inventory, match, reconciliation, exported)
    assert all(result.exit_code == 0 for result in results)
    visible = "".join(
        (
            *(result.output for result in results),
            *first_outputs,
            *second_outputs,
        )
    )
    assert source_key not in visible
    assert output.read_bytes().startswith(b"\xef\xbb\xbf")
    assert source_key not in output.read_text(encoding="utf-8-sig")
    with output.open(encoding="utf-8-sig", newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    assert len(rows) == 1
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM source_assets").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM match_candidates").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM sample_labels").fetchone()[0] == 1


def test_bad_clinical_source_is_atomic_and_nonzero(tmp_path: Path) -> None:
    # Given: a clinical source without the required subject key.
    source = tmp_path / "bad.csv"
    source.write_text("SUBJID,pathology_result\n,Cancer\n", encoding="utf-8-sig")
    db = tmp_path / "master.db"
    raw_store = tmp_path / "raw"

    # When: ingestion crosses the CLI boundary.
    result = CliRunner().invoke(
        cli,
        [
            "data",
            "ingest-clinical",
            "--db",
            str(db),
            "--source",
            str(source),
            "--site-code",
            "SITE-A",
            "--protocol-code",
            "SYNTHETIC",
            "--source-group",
            "PRO",
            "--patient-id-field",
            "SUBJID",
            "--raw-store",
            str(raw_store),
        ],
    )

    # Then: failure is nonzero and publishes no partial source or raw blob.
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "clinical ingestion failed" in result.output
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM source_assets").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM subjects").fetchone()[0] == 0
    assert not tuple(raw_store.rglob("*"))


def test_export_failure_cleans_atomic_temporary_file(tmp_path: Path) -> None:
    # Given: a frozen manifest and an invalid final target that is a directory.
    source_key = "PSEUDONYM002"
    clinical, spectra = write_inputs(tmp_path, source_key)
    db = tmp_path / "master.db"
    runner = CliRunner()
    invoke_ingestion(runner, db, tmp_path / "raw", clinical, spectra)
    runner.invoke(cli, ["data", "match", "--db", str(db)])
    invoke_labels(runner, db)
    manifest_id = build_manifest(db)
    output = tmp_path / "occupied"
    output.mkdir()

    # When: atomic publication cannot replace the final target.
    result = runner.invoke(
        cli,
        [
            "data",
            "export-dataset",
            "--db",
            str(db),
            "--manifest-id",
            manifest_id,
            "--output",
            str(output),
        ],
    )

    # Then: the CLI fails and no temporary export remains.
    assert result.exit_code != 0
    assert not tuple(tmp_path.glob(".occupied.*.tmp"))


def test_missing_mlflow_fails_before_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a frozen manifest but no optional MLflow installation.
    source_key = "PSEUDONYM003"
    clinical, spectra = write_inputs(tmp_path, source_key)
    db = tmp_path / "master.db"
    runner = CliRunner()
    invoke_ingestion(runner, db, tmp_path / "raw", clinical, spectra)
    runner.invoke(cli, ["data", "match", "--db", str(db)])
    invoke_labels(runner, db)
    manifest_id = build_manifest(db)
    monkeypatch.setattr("sers.mlflow_tracking.mlflow_available", lambda: False)
    output = tmp_path / "dataset.csv"

    # When: export explicitly requests optional experiment tracking.
    result = runner.invoke(
        cli,
        [
            "data",
            "export-dataset",
            "--db",
            str(db),
            "--manifest-id",
            manifest_id,
            "--output",
            str(output),
            "--mlflow-db",
            str(tmp_path / "mlflow.db"),
        ],
    )

    # Then: failure is typed and no untracked dataset is published.
    assert result.exit_code != 0
    assert "MLflow is unavailable" in result.output
    assert not output.exists()
