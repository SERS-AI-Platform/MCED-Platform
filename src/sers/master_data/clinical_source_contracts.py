from __future__ import annotations

from pathlib import Path
from typing import Final

from .clinical_types import (
    ClinicalEventType,
    ClinicalFormat,
    ClinicalIdentityStatus,
    ClinicalSheetContract,
    ClinicalSource,
    ClinicalSourceVersion,
)

_SITE_NAMES: Final = {
    "BORAMAE": "Seoul National University Boramae Hospital",
    "CBNUH": "Chungbuk National University Hospital",
    "IJBPH": "Inje University Busan Paik Hospital",
    "SAMSUNG": "Samsung Seoul Hospital",
    "SNUH": "Seoul National University Hospital",
    "SSMH": "Seoul St. Mary's Hospital",
    "YONSEI": "Yonsei Severance Hospital",
    "YPNUH": "Yangsan Pusan National University Hospital",
}


def clinical_source(
    root: Path,
    relative_path: str,
    site_code: str,
    protocol_code: str,
    source_group: str,
    *,
    sheet_contracts: tuple[ClinicalSheetContract, ...] = (),
    patient_id_fields: tuple[str, ...] = ("SUBJID",),
    event_type: ClinicalEventType = "other",
    canonical_alias: str | None = None,
    historical: bool = False,
) -> ClinicalSource:
    path = root / relative_path
    return ClinicalSource(
        path=path,
        site_code=site_code,
        site_name=_SITE_NAMES[site_code],
        protocol_code=protocol_code,
        source_group=source_group,
        source_format=(
            ClinicalFormat("csv")
            if path.suffix.casefold() == ".csv"
            else ClinicalFormat("excel")
        ),
        patient_id_fields=patient_id_fields,
        event_type=event_type,
        canonical_alias=canonical_alias,
        identity_status=(
            ClinicalIdentityStatus("alias_review")
            if canonical_alias is not None
            else ClinicalIdentityStatus("canonical")
        ),
        source_version=(
            ClinicalSourceVersion("historical")
            if historical
            else ClinicalSourceVersion("current")
        ),
        sheet_contracts=sheet_contracts,
    )


def fixed_clinical_sources(root: Path) -> tuple[ClinicalSource, ...]:
    entries = (
        ("1. 전립선암/SMCXD01_전립선암 임상정보.xlsx", "CBNUH", "SMCXD01", "PRO", "Sheet1", 1, ("NO",)),
        ("2. 유방암/SMCXD01_유방암.xlsx", "IJBPH", "SMCXD01", "BRE", "C50 임상정보", 1, ("제공자bCODE",)),
        ("3. 난소암/SMCXD01_난소암 1.xlsx", "IJBPH", "SMCXD01", "OVA", "C56 임상정보", 1, ("제공자bCODE",)),
        ("3. 난소암/SMCXD01_난소암 2.xlsx", "SNUH", "SMCXD01", "OVA", "난소", 1, ("제공자:제공자bCODE",)),
        ("4. 폐암/SMCXD01_폐암 1.xlsx", "SNUH", "SMCXD01", "LUN", "폐", 1, ("제공자:제공자bCODE",)),
        ("4. 폐암/SMCXD06_폐암 2.xlsx", "SSMH", "SMCXD06", "LUN", "Sheet1", 1, ("no",)),
        ("5. 정상/SMCXD03_정상인 1.xlsx", "YPNUH", "SMCXD03", "NOR", "정상인", 1, ("column_1", "No.")),
        ("5. 정상/SMCXD03_정상인 2.xlsx", "YPNUH", "SMCXD03", "NOR", "정상인", 1, ("column_1", "No.")),
        ("6. 당뇨/SMCXD03_당뇨 1.xlsx", "YPNUH", "SMCXD03", "DIA", "당뇨", 1, ("column_1", "No.")),
        ("6. 당뇨/SMCXD03_당뇨 2.xlsx", "YPNUH", "SMCXD03", "DIA", "당뇨", 1, ("Sample number", "No.")),
        ("7. 고혈압/SMCXD03_고혈압.xlsx", "YPNUH", "SMCXD03", "HBP", "고혈압", 1, ("column_1", "No.")),
        ("8. 당뇨 + 고혈압/SMCXD05_당뇨+고혈압.xlsx", "YPNUH", "SMCXD05", "H.D.", "당뇨+고혈압", 1, ("column_1", "No.")),
        (
            "10. 췌장암/CPAN/SMCMD06_췌장암.xlsx",
            "CBNUH",
            "SMCMD06",
            "CPAN",
            "췌장암70명",
            1,
            ("제공자:제공자bCODE",),
        ),
        ("11. 방광암/SMCXD06_방광암.xlsm", "CBNUH", "SMCXD06", "BLA", "분양명단", 2, ("NO",)),
    )
    sources = [
        clinical_source(
            root,
            relative_path,
            site_code,
            protocol,
            group,
            sheet_contracts=(
                ClinicalSheetContract(
                    sheet,
                    header,
                    keys,
                    "other",
                    data_start_row=3 if group == "PRO" else None,
                ),
            ),
            patient_id_fields=keys,
            canonical_alias={"CPAN": "PAN", "BLA": "BLC"}.get(group),
        )
        for relative_path, site_code, protocol, group, sheet, header, keys in entries
    ]
    sources.extend(
        (
            clinical_source(
                root,
                "4. 폐암/SMCXD06_폐암 3.xlsx",
                "SNUH",
                "SMCXD06",
                "LUN",
                sheet_contracts=(
                    ClinicalSheetContract(
                        "자원리스트",
                        1,
                        ("column_1",),
                        "collection",
                        "제공자:제공자bCODE",
                    ),
                    ClinicalSheetContract("인구학적정보 및 암 관련 정보", 1, ("제공자:제공자bCODE",), "other"),
                    ClinicalSheetContract("병리검사", 1, ("제공자:제공자bCODE",), "diagnosis"),
                    ClinicalSheetContract("진단검사", 1, ("제공자:제공자bCODE",), "lab"),
                ),
                patient_id_fields=("column_1", "제공자:제공자bCODE"),
            ),
            clinical_source(
                root,
                "9. 대장암/SMCXD06_대장암.xlsx",
                "CBNUH",
                "SMCXD06",
                "CRC",
                sheet_contracts=(
                    ClinicalSheetContract(
                        "대장암1~2기270명",
                        1,
                        ("제공자:제공자bCODE",),
                        "other",
                    ),
                    ClinicalSheetContract(
                        "대장암3~4기30명",
                        1,
                        ("제공자:제공자bCODE",),
                        "other",
                    ),
                ),
                patient_id_fields=("제공자:제공자bCODE",),
            ),
            clinical_source(
                root,
                "10. 췌장암/YPAN/SMCXD04_CRF_data.xlsx",
                "YONSEI",
                "SMCXD04",
                "YPAN",
                sheet_contracts=(
                    ClinicalSheetContract("CRF Data", 1, ("스크리닝번호",), "other"),
                ),
                patient_id_fields=("스크리닝번호",),
                canonical_alias="PAN",
            ),
            clinical_source(
                root,
                "보라매 병원 임상정보.xlsx",
                "BORAMAE",
                "SMCXD07",
                "BPRO/BNOR",
                sheet_contracts=(
                    ClinicalSheetContract("Sheet1", 1, ("patient_code",), "other"),
                ),
                patient_id_fields=("patient_code",),
            ),
            clinical_source(
                root,
                "보라매 병원 임상정보 (pre-2026-07-14).xlsx",
                "BORAMAE",
                "SMCXD07",
                "BPRO/BNOR",
                sheet_contracts=(
                    ClinicalSheetContract("Sheet1", 1, ("patient_code",), "other"),
                ),
                patient_id_fields=("patient_code",),
                historical=True,
            ),
        )
    )
    return tuple(sources)
