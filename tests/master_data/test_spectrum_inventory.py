from __future__ import annotations

from pathlib import Path

import pytest

from sers.config import Config
from sers.master_data.inventory import configured_inventory_roots, inventory_spectra
from sers.master_data.spectrum_parser import parse_spectrum_name
from sers.master_data.spectrum_types import (
    IdentityStatus,
    InventoryRoot,
    InventoryStatus,
    MaterialKind,
    SourceKind,
)


def _write_spectrum(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("100,1\n101,2\n", encoding="utf-8")


def test_parser_preserves_alias_ambiguity_and_derived_role() -> None:
    # Given: a date-based PAN average whose identity is historically ambiguous.
    path = Path("20260429_Urine test/9. PAN/PAN 12_ave.CSV")

    # When: the spectrum filename is parsed.
    parsed = parse_spectrum_name(path, preparation="liquid")

    # Then: the source code is preserved and the alias is review-only.
    assert parsed.group_code == "PAN"
    assert parsed.status is InventoryStatus.DERIVED
    assert parsed.identity_status is IdentityStatus.ALIAS_REVIEW
    assert parsed.alias_candidates == ("CPAN",)


def test_parser_marks_ypan_alias_for_review_but_excludes_span() -> None:
    # Given: institution-specific pancreatic source codes.
    ypan = Path("10. Pancreatic/YPAN/YPAN 7_1.CSV")
    span = Path("10. Pancreatic/SPAN/SPAN 7_1.CSV")

    # When: the spectrum identities are parsed.
    parsed_ypan = parse_spectrum_name(ypan, preparation="liquid")
    parsed_span = parse_spectrum_name(span, preparation="liquid")

    # Then: YPAN→PAN is metadata only and SPAN is not an alias candidate.
    assert parsed_ypan.group_code == "YPAN"
    assert parsed_ypan.identity_status is IdentityStatus.ALIAS_REVIEW
    assert parsed_ypan.alias_candidates == ("PAN",)
    assert parsed_span.group_code == "SPAN"
    assert parsed_span.identity_status is IdentityStatus.CANONICAL
    assert parsed_span.alias_candidates == ()


@pytest.mark.parametrize(
    ("path", "group_code", "source_sample_code"),
    [
        (Path("2. Breast cancer/Box 03/017/BRE 017_2.CSV"), "BRE", "017"),
        (Path("3. Ovarian cancer/Order 09/Box 4/OVA A070_5.CSV"), "OVA", "A070"),
    ],
)
def test_parser_uses_filename_identity_not_numeric_order_or_box_positions(
    path: Path,
    group_code: str,
    source_sample_code: str,
) -> None:
    # Given: real-shaped BRE/OVA paths containing numeric order and box positions.
    # When: the filename is parsed.
    parsed = parse_spectrum_name(path, preparation="liquid")

    # Then: only the explicit filename key is identity; path positions stay irrelevant.
    assert parsed.group_code == group_code
    assert parsed.source_sample_code == source_sample_code
    assert parsed.identity_status is IdentityStatus.CANONICAL
    assert parsed.alias_candidates == ()


def test_parser_classifies_blank_as_analytical_material() -> None:
    # Given: a technical replicate below a Blank folder.
    path = Path("20260429_Urine test/0. Blank/Blank 8-5_4.CSV")

    # When: the filename is parsed.
    parsed = parse_spectrum_name(path, preparation="liquid")

    # Then: it is a blank material and retains its replicate.
    assert parsed.material_kind is MaterialKind.BLANK
    assert parsed.replicate_index == 4


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("YPAN 1 F_1.CSV", "fasting"),
        ("YPAN 1 NF_1.CSV", "non_fasting"),
    ],
)
def test_parser_preserves_fasting_state_from_prospective_filename(
    filename: str,
    expected: str,
) -> None:
    # Given: the confirmed F/NF naming contract for the prospective powder set.
    path = Path("OneDrive_2026-07-21 (2)") / filename

    # When: the spectrum filename is parsed.
    parsed = parse_spectrum_name(path, preparation="powder")

    # Then: fasting state is typed metadata and identity stays the SoluM label.
    assert parsed.fasting_state.value == expected
    assert parsed.solum_label == "YPAN 1"


def test_parser_preserves_sigma_as_lot_instead_of_preparation() -> None:
    # Given: a powder reproducibility filename whose Sigma value is a lot.
    path = Path("20260715_Powder_Reproducibility test/Sigma 3_BNOR 2_1.CSV")

    # When: the spectrum filename is parsed.
    parsed = parse_spectrum_name(path, preparation="powder")

    # Then: preparation and lot remain separate experimental dimensions.
    assert parsed.preparation == "powder"
    assert parsed.lot_code == "Sigma 3"


def test_parser_preserves_postoperative_specimen_prefix() -> None:
    # Given: the confirmed Po. prefix for a post-operative specimen.
    path = Path("OneDrive_2026-07-21 (2)/Po. YPAN 7_1.CSV")

    # When: the spectrum filename is parsed.
    parsed = parse_spectrum_name(path, preparation="powder")

    # Then: surgical timing is metadata and does not alter the SoluM label.
    assert parsed.specimen_timing.value == "post_operative"
    assert parsed.solum_label == "YPAN 7"


def test_parser_accepts_explicit_post_surgery_specimen_marker() -> None:
    # Given: the same post-operative state written out in the filename.
    path = Path("OneDrive_2026-07-21 (2)/YPAN_Post surgery_23_1.CSV")

    # When: the spectrum filename is parsed.
    parsed = parse_spectrum_name(path, preparation="powder")

    # Then: it has the same typed timing and canonical SoluM label as Po.
    assert parsed.specimen_timing.value == "post_operative"
    assert parsed.solum_label == "YPAN 23"


def test_parser_accepts_explicit_fasting_specimen_marker() -> None:
    # Given: the confirmed fasting state written out in the filename.
    path = Path("OneDrive_2026-07-21 (2)/YPAN_Fasting_28_1.CSV")

    # When: the spectrum filename is parsed.
    parsed = parse_spectrum_name(path, preparation="powder")

    # Then: it maps to the same typed state as the F marker.
    assert parsed.fasting_state.value == "fasting"
    assert parsed.solum_label == "YPAN 28"


def test_inventory_uses_medical_background_and_excludes_non_measurements(
    tmp_path: Path,
) -> None:
    # Given: Medical raw and Background spectra plus excluded artifacts.
    root = tmp_path / "raw_data_medical"
    group = root / "1. Prostate cancer (100개)"
    _write_spectrum(group / "PRO 1_1.txt")
    _write_spectrum(group / "Background" / "PRO 1_1.txt")
    _write_spectrum(group / "Background" / "PRO 1_ave.txt")
    _write_spectrum(group / "Background" / "MultiData.txt")
    _write_spectrum(group / "Background" / "PRO 1_2.txt:Zone.Identifier")

    # When: the typed source root is inventoried.
    report = inventory_spectra(
        (
            InventoryRoot(
                path=root,
                source_kind=SourceKind.MEDICAL,
                instrument_key="Medical Raman",
                preparation="liquid",
            ),
        )
    )

    # Then: only the Background replicate is ready and the summary exposes no key.
    assert report.counts.ready == 1
    assert report.counts.excluded == 2
    assert len(report.items) == 3
    assert "PRO" not in report.redacted_summary()


def test_inventory_quarantines_malformed_spectrum_filename(tmp_path: Path) -> None:
    # Given: a spectrum-shaped file without a valid identity/replicate name.
    root = tmp_path / "20260429_Urine test"
    _write_spectrum(root / "mystery.CSV")

    # When: the root is inventoried.
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

    # Then: the file remains visible with an explicit quarantine status.
    assert report.counts.quarantined == 1
    assert report.items[0].reason_code == "malformed_filename"


def test_configured_inventory_excludes_unmapped_experiment_folders(
    tmp_path: Path,
) -> None:
    # Given: one configured clinical folder and one equipment experiment.
    _write_spectrum(tmp_path / "data" / "raw_data" / "mapped" / "PRO 1_1.CSV")
    _write_spectrum(tmp_path / "data" / "raw_data" / "equipment" / "PRO 2_1.CSV")
    config = Config(folder_to_group={"mapped": "PRO"})

    # When: inventory roots are derived from the application folder mapping.
    roots = configured_inventory_roots(tmp_path, config)
    report = inventory_spectra(roots)

    # Then: only the explicitly mapped clinical folder is in scope.
    assert report.counts.ready == 1
