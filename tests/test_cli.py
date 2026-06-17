"""CLI smoke tests."""

from click.testing import CliRunner

from sers.cli import cli


def test_cli_help_lists_primary_workflows():
    runner = CliRunner()

    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "preprocess" in result.output
    assert "train" in result.output
    assert "data" in result.output
    assert "predict" in result.output


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
