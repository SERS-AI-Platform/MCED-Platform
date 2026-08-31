"""CLI smoke tests."""

from importlib import import_module
from pathlib import Path

from click.testing import CliRunner

from sers.cli import cli
from sers.cli.data import data


def test_cli_help_lists_primary_workflows():
    runner = CliRunner()

    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "preprocess" in result.output
    assert "train" in result.output
    assert "data" in result.output
    assert "production" in result.output


def test_cli_exposes_data_validate_help():
    runner = CliRunner()

    result = runner.invoke(cli, ["data", "validate", "--help"])

    assert result.exit_code == 0
    assert "Validate spectrum data files" in result.output


def test_cli_exposes_active_stacking_train_help():
    runner = CliRunner()

    result = runner.invoke(cli, ["train", "stacking", "--help"])

    assert result.exit_code == 0
    assert "Train stacking ensemble" in result.output


def test_legacy_standardize_warns_before_running(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # Given: the compatibility command delegates to its legacy script.
    calls: list[tuple[str, list[str]]] = []
    data_module = import_module("sers.cli.data")
    monkeypatch.setattr(data_module, "run_script", lambda script, args: calls.append((script, args)))

    # When: an operator invokes the predecessor command.
    result = CliRunner().invoke(
        cli,
        ["data", "standardize", "--output-dir", str(tmp_path)],
    )

    # Then: execution remains compatible but is visibly classified as legacy.
    assert result.exit_code == 0
    assert "DEPRECATED" in result.output
    assert calls == [
        (
            "scripts/pipeline/standardize_clinical_data.py",
            ["--output-dir", str(tmp_path)],
        )
    ]


def test_canonical_data_cli_excludes_row_order_and_clinical_unified_predecessors() -> None:
    # Given: the canonical governed data command registry.
    forbidden_commands = {"reingest-staging", "clinical-unified", "ingest-bla"}

    # When: registered subcommand names are inspected.
    registered_commands = set(data.commands)

    # Then: no predecessor utility is reachable from the governed CLI.
    assert registered_commands.isdisjoint(forbidden_commands)
