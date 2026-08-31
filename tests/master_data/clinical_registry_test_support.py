from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook

from sers.master_data.clinical_source_contracts import fixed_clinical_sources
from sers.master_data.clinical_types import (
    ClinicalSheetContract,
    ClinicalSource,
)


def _sheet_values(
    source: ClinicalSource,
    contract: ClinicalSheetContract,
    source_index: int,
) -> tuple[tuple[str | None, ...], tuple[str, ...]]:
    key_field = contract.patient_id_fields[0]
    header_key = None if key_field == "column_1" else key_field
    headers = [header_key, "synthetic_field"]
    values = [f"{source.source_group}-FIXED-{source_index}", "synthetic"]
    if contract.subject_key_alias_field is not None:
        headers.append(contract.subject_key_alias_field)
        values.append(f"ALIAS-FIXED-{source_index}")
    elif key_field == "제공자:제공자bCODE" and len(source.sheet_contracts) > 1:
        values[0] = f"ALIAS-FIXED-{source_index}"
    return tuple(headers), tuple(values)


def _write_workbook(source: ClinicalSource, source_index: int) -> None:
    source.path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    contracts = source.sheet_contracts
    for contract_index, contract in enumerate(contracts):
        sheet = (
            workbook.active
            if contract_index == 0
            else workbook.create_sheet()
        )
        sheet.title = contract.sheet_name or "Sheet"
        for _ in range(1, contract.header_row):
            sheet.append(("title",))
        headers, values = _sheet_values(source, contract, source_index)
        sheet.append(headers)
        data_start = contract.data_start_row or contract.header_row + 1
        for _ in range(contract.header_row + 1, data_start):
            sheet.append(())
        sheet.append(values)
    workbook.save(source.path)


def write_complete_clinical_registry(root: Path) -> None:
    for source_index, source in enumerate(
        fixed_clinical_sources(root),
        start=1,
    ):
        _write_workbook(source, source_index)
    for group in ("SPAN", "YPAN"):
        path = root / f"10. 췌장암/{group}/20260203/DM.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"SUBJID,synthetic_field\n{group}-DYNAMIC,synthetic\n",
            encoding="utf-8-sig",
        )
