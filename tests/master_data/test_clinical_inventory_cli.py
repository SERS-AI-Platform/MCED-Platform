from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from sers.cli import cli


def test_cli_inventory_aggregates_missing_and_unreadable_contracts(
    tmp_path: Path,
) -> None:
    # Given: an otherwise empty root has one unreadable configured source path.
    clinical_root = tmp_path / "clinical"
    spectrum_root = tmp_path / "spectra"
    clinical_root.mkdir()
    spectrum_root.mkdir()
    (clinical_root / "2. 유방암/SMCXD01_유방암.xlsx").mkdir(parents=True)

    # When: aggregate inventory runs through the public CLI.
    result = CliRunner().invoke(
        cli,
        [
            "data",
            "inventory",
            "--db",
            str(tmp_path / "master.db"),
            "--clinical-root",
            str(clinical_root),
            "--spectrum-root",
            str(spectrum_root),
        ],
    )

    # Then: only contract-level issue counts are disclosed.
    assert result.exit_code == 0
    assert "issues=21" in result.output
    assert "missing_source=18" in result.output
    assert "unreadable_source=1" in result.output
    assert "expected_group_empty=2" in result.output
