from __future__ import annotations

from collections import Counter
from pathlib import Path

from sers.master_data.clinical_inventory import inventory_clinical_sources
from sers.master_data.clinical_types import (
    ClinicalIdentityStatus,
    ClinicalSourceVersion,
)


def _touch(root: Path, relative_path: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_inventory_accepts_only_explicit_raw_clinical_sources(tmp_path: Path) -> None:
    # Given: raw CRF sources alongside generated master and standardized outputs.
    for relative_path in (
        "10. 췌장암/SPAN/20260203/LB.csv",
        "10. 췌장암/YPAN/20260203/UR.csv",
        "10. 췌장암/YPAN/SMCXD04_CRF_data.xlsx",
        "보라매 병원 임상정보.xlsx",
        "보라매 병원 임상정보 (pre-2026-07-14).xlsx",
        "standardized/all_clinical_standardized.csv",
        "master_clinical.csv",
        "powerbi_patient_groups/csv/generated.csv",
    ):
        _touch(tmp_path, relative_path)

    # When: the configured clinical inventory is built.
    report = inventory_clinical_sources(tmp_path)

    # Then: raw sources are present and generated outputs are absent.
    relative_paths = {source.path.relative_to(tmp_path).as_posix() for source in report.sources}
    assert relative_paths == {
        "10. 췌장암/SPAN/20260203/LB.csv",
        "10. 췌장암/YPAN/20260203/UR.csv",
        "10. 췌장암/YPAN/SMCXD04_CRF_data.xlsx",
        "보라매 병원 임상정보.xlsx",
        "보라매 병원 임상정보 (pre-2026-07-14).xlsx",
    }


def test_inventory_marks_old_boramae_as_history_not_a_second_cohort(
    tmp_path: Path,
) -> None:
    # Given: current and pre-cutover Boramae workbooks.
    _touch(tmp_path, "보라매 병원 임상정보.xlsx")
    _touch(tmp_path, "보라매 병원 임상정보 (pre-2026-07-14).xlsx")

    # When: both immutable source versions are inventoried.
    report = inventory_clinical_sources(tmp_path)
    current, historical = sorted(
        report.sources,
        key=lambda source: source.source_version.value,
    )

    # Then: they share one site/cohort identity and differ only by source version.
    assert current.site_code == historical.site_code == "BORAMAE"
    assert current.source_group == historical.source_group == "BPRO/BNOR"
    assert {current.source_version, historical.source_version} == {
        ClinicalSourceVersion.CURRENT,
        ClinicalSourceVersion.HISTORICAL,
    }
    assert report.redacted_summary() == (
        "current=1,historical=1,issues=19,"
        "missing_source=17,expected_group_empty=2"
    )


def test_inventory_retains_aliases_as_review_metadata(tmp_path: Path) -> None:
    # Given: raw CPAN, YPAN, and BLA sources.
    for relative_path in (
        "10. 췌장암/CPAN/SMCMD06_췌장암.xlsx",
        "10. 췌장암/YPAN/20260203/DM.csv",
        "11. 방광암/SMCXD06_방광암.xlsm",
    ):
        _touch(tmp_path, relative_path)

    # When: their identity contracts are inventoried.
    report = inventory_clinical_sources(tmp_path)
    identities = {
        source.source_group: (source.canonical_alias, source.identity_status)
        for source in report.sources
    }

    # Then: source groups are not silently rewritten to aliases.
    assert identities == {
        "CPAN": ("PAN", ClinicalIdentityStatus.ALIAS_REVIEW),
        "YPAN": ("PAN", ClinicalIdentityStatus.ALIAS_REVIEW),
        "BLA": ("BLC", ClinicalIdentityStatus.ALIAS_REVIEW),
    }


def test_inventory_preserves_source_file_hospital_identity(tmp_path: Path) -> None:
    # Given: one raw workbook for every institution-bearing retrospective source.
    expected_sites = {
        "1. 전립선암/SMCXD01_전립선암 임상정보.xlsx": "CBNUH",
        "2. 유방암/SMCXD01_유방암.xlsx": "IJBPH",
        "3. 난소암/SMCXD01_난소암 1.xlsx": "IJBPH",
        "3. 난소암/SMCXD01_난소암 2.xlsx": "SNUH",
        "4. 폐암/SMCXD01_폐암 1.xlsx": "SNUH",
        "4. 폐암/SMCXD06_폐암 2.xlsx": "SSMH",
        "4. 폐암/SMCXD06_폐암 3.xlsx": "SNUH",
        "9. 대장암/SMCXD06_대장암.xlsx": "CBNUH",
        "10. 췌장암/CPAN/SMCMD06_췌장암.xlsx": "CBNUH",
        "11. 방광암/SMCXD06_방광암.xlsm": "CBNUH",
    }
    for relative_path in expected_sites:
        _touch(tmp_path, relative_path)

    # When: the canonical built-in source registry is inventoried.
    report = inventory_clinical_sources(tmp_path)

    # Then: each physical source retains its institution instead of generic SMC.
    actual_sites = {
        source.path.relative_to(tmp_path).as_posix(): source.site_code
        for source in report.sources
    }
    assert actual_sites == expected_sites


def test_inventory_encodes_multisheet_and_physical_header_contracts(
    tmp_path: Path,
) -> None:
    # Given: the CRC, bladder, and two source-specific lung workbooks.
    for relative_path in (
        "9. 대장암/SMCXD06_대장암.xlsx",
        "11. 방광암/SMCXD06_방광암.xlsm",
        "4. 폐암/SMCXD06_폐암 2.xlsx",
        "4. 폐암/SMCXD06_폐암 3.xlsx",
    ):
        _touch(tmp_path, relative_path)

    # When: their workbook contracts are inventoried.
    report = inventory_clinical_sources(tmp_path)
    contracts = {source.path.name: source for source in report.sources}

    # Then: every physical sheet/header/key is explicit and deterministic.
    crc = contracts["SMCXD06_대장암.xlsx"].sheet_contracts
    assert tuple(contract.sheet_name for contract in crc) == (
        "대장암1~2기270명",
        "대장암3~4기30명",
    )
    assert {contract.patient_id_fields for contract in crc} == {
        ("제공자:제공자bCODE",),
    }
    bladder = contracts["SMCXD06_방광암.xlsm"].sheet_contracts
    assert tuple(
        (contract.sheet_name, contract.header_row) for contract in bladder
    ) == (("분양명단", 2),)
    lung2 = contracts["SMCXD06_폐암 2.xlsx"].sheet_contracts
    assert lung2[0].patient_id_fields == ("no",)
    assert tuple(
        contract.sheet_name
        for contract in contracts["SMCXD06_폐암 3.xlsx"].sheet_contracts
    ) == (
        "자원리스트",
        "인구학적정보 및 암 관련 정보",
        "병리검사",
        "진단검사",
    )


def test_inventory_full_protocol_matrix_uses_canonical_study_codes(
    tmp_path: Path,
) -> None:
    # Given: all 58 source paths supported by the built-in clinical registry.
    expected = {
        ("1. 전립선암/SMCXD01_전립선암 임상정보.xlsx", "PRO"): "SMCXD01",
        ("2. 유방암/SMCXD01_유방암.xlsx", "BRE"): "SMCXD01",
        ("3. 난소암/SMCXD01_난소암 1.xlsx", "OVA"): "SMCXD01",
        ("3. 난소암/SMCXD01_난소암 2.xlsx", "OVA"): "SMCXD01",
        ("4. 폐암/SMCXD01_폐암 1.xlsx", "LUN"): "SMCXD01",
        ("4. 폐암/SMCXD06_폐암 2.xlsx", "LUN"): "SMCXD06",
        ("4. 폐암/SMCXD06_폐암 3.xlsx", "LUN"): "SMCXD06",
        ("5. 정상/SMCXD03_정상인 1.xlsx", "NOR"): "SMCXD03",
        ("5. 정상/SMCXD03_정상인 2.xlsx", "NOR"): "SMCXD03",
        ("6. 당뇨/SMCXD03_당뇨 1.xlsx", "DIA"): "SMCXD03",
        ("6. 당뇨/SMCXD03_당뇨 2.xlsx", "DIA"): "SMCXD03",
        ("7. 고혈압/SMCXD03_고혈압.xlsx", "HBP"): "SMCXD03",
        ("8. 당뇨 + 고혈압/SMCXD05_당뇨+고혈압.xlsx", "H.D."): "SMCXD05",
        ("9. 대장암/SMCXD06_대장암.xlsx", "CRC"): "SMCXD06",
        ("10. 췌장암/CPAN/SMCMD06_췌장암.xlsx", "CPAN"): "SMCMD06",
        ("10. 췌장암/YPAN/SMCXD04_CRF_data.xlsx", "YPAN"): "SMCXD04",
        ("11. 방광암/SMCXD06_방광암.xlsm", "BLA"): "SMCXD06",
        ("보라매 병원 임상정보.xlsx", "BPRO/BNOR"): "BORAMAE_CURRENT",
        (
            "보라매 병원 임상정보 (pre-2026-07-14).xlsx",
            "BPRO/BNOR",
        ): "BORAMAE_CURRENT",
    }
    prospective_files = {
        "SPAN": (
            "CA.csv", "CM.csv", "CT.csv", "CY.csv", "DM.csv", "EN.csv",
            "IE.csv", "LB.csv", "LY.csv", "MH.csv", "MI.csv", "MY.csv",
            "PC.csv", "PR.csv", "PY.csv", "RT.csv", "RY.csv",
            "SUBJECT_INFO.csv", "TU.csv",
        ),
        "YPAN": (
            "CM.csv", "CT.csv", "CY.csv", "DM.csv", "EN.csv", "IE.csv",
            "LB.csv", "LY.csv", "MH.csv", "MI.csv", "MY.csv", "PC.csv",
            "SM.csv", "SP.csv", "ST.csv", "SUBJECT_INFO.csv", "SY.csv",
            "TY.csv", "UR.csv", "UY.csv",
        ),
    }
    for (relative_path, _group), _protocol in expected.items():
        _touch(tmp_path, relative_path)
    for group, filenames in prospective_files.items():
        protocol = "SMCXD02" if group == "SPAN" else "SMCXD04"
        for filename in filenames:
            relative_path = f"10. 췌장암/{group}/20260203/{filename}"
            expected[(relative_path, group)] = protocol
            _touch(tmp_path, relative_path)

    # When: every supported physical source is inventoried.
    report = inventory_clinical_sources(tmp_path)
    actual = {
        (source.path.relative_to(tmp_path).as_posix(), source.source_group):
            source.protocol_code
        for source in report.sources
    }

    # Then: the complete matrix uses canonical protocols with no CRF aliases.
    assert len(actual) == len(expected) == 58
    assert actual == expected
    assert Counter(actual.values()) == Counter(expected.values())


def test_inventory_retains_missing_and_unreadable_fixed_contracts(
    tmp_path: Path,
) -> None:
    # Given: one configured source path is absent and another is a directory.
    unreadable_path = tmp_path / "2. 유방암/SMCXD01_유방암.xlsx"
    unreadable_path.mkdir(parents=True)

    # When: the built-in registry is inventoried.
    report = inventory_clinical_sources(tmp_path)
    issues = {
        issue.path.relative_to(tmp_path).as_posix(): issue
        for issue in report.issues
    }

    # Then: both contracts remain visible without reading patient-level values.
    missing = issues["1. 전립선암/SMCXD01_전립선암 임상정보.xlsx"]
    assert (
        missing.reason_code,
        missing.site_code,
        missing.protocol_code,
        missing.source_group,
    ) == ("missing_source", "CBNUH", "SMCXD01", "PRO")
    unreadable = issues["2. 유방암/SMCXD01_유방암.xlsx"]
    assert (
        unreadable.reason_code,
        unreadable.site_code,
        unreadable.protocol_code,
        unreadable.source_group,
    ) == ("unreadable_source", "IJBPH", "SMCXD01", "BRE")


def test_inventory_reports_missing_and_empty_prospective_groups(
    tmp_path: Path,
) -> None:
    # Given: SPAN is absent and YPAN contains only configured exclusions.
    ypan = tmp_path / "10. 췌장암/YPAN"
    ypan.mkdir(parents=True)
    (ypan / "SMCXD04_SERS_dataset.csv").touch()
    (ypan / "SN.csv").touch()

    # When: dynamic prospective source groups are inventoried.
    report = inventory_clinical_sources(tmp_path)
    group_issues = {
        issue.source_group: issue
        for issue in report.issues
        if issue.reason_code == "expected_group_empty"
    }

    # Then: both expected groups retain site/protocol/path review metadata.
    assert set(group_issues) == {"SPAN", "YPAN"}
    assert (
        group_issues["SPAN"].site_code,
        group_issues["SPAN"].protocol_code,
        group_issues["SPAN"].path.relative_to(tmp_path).as_posix(),
    ) == ("SAMSUNG", "SMCXD02", "10. 췌장암/SPAN")
    assert (
        group_issues["YPAN"].site_code,
        group_issues["YPAN"].protocol_code,
        group_issues["YPAN"].path.relative_to(tmp_path).as_posix(),
    ) == ("YONSEI", "SMCXD04", "10. 췌장암/YPAN")
