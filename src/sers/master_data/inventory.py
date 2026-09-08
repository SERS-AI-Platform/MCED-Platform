from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from sers.config import Config

from .spectrum_parser import MalformedSpectrumNameError, parse_spectrum_name
from .spectrum_types import (
    InventoryCounts,
    InventoryItem,
    InventoryReport,
    InventoryRoot,
    InventoryStatus,
    SourceKind,
)

_DATE_PATTERN = re.compile(r"(?P<date>20\d{6})")


def _candidate_paths(root: InventoryRoot) -> tuple[Path, ...]:
    scan_root = root.path
    if root.source_kind is SourceKind.MEDICAL:
        if (scan_root / "Background").is_dir():
            scan_root = scan_root / "Background"
            return tuple(
                sorted(
                    path
                    for path in scan_root.iterdir()
                    if path.is_file() and path.suffix.casefold() == ".txt"
                )
            )
        return tuple(
            sorted(
                path
                for path in scan_root.glob("*/Background/*")
                if path.is_file() and path.suffix.casefold() == ".txt"
            )
        )
    if root.source_kind is SourceKind.THERMO:
        direct_files = [
            path
            for path in scan_root.iterdir()
            if path.is_file() and path.suffix.casefold() in {".csv", ".txt"}
        ]
        average_files = [
            path
            for directory in scan_root.iterdir()
            if directory.is_dir() and "average" in directory.name.casefold()
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.casefold() in {".csv", ".txt"}
        ]
        return tuple(sorted((*direct_files, *average_files)))
    return tuple(
        sorted(
            path
            for path in scan_root.rglob("*")
            if path.is_file() and path.suffix.casefold() in {".csv", ".txt"}
        )
    )


def configured_inventory_roots(repo_root: Path, config: Config) -> tuple[InventoryRoot, ...]:
    data_root = repo_root / "data"
    roots = [
        InventoryRoot(
            path=data_root / "raw_data" / folder,
            source_kind=SourceKind.THERMO,
            instrument_key="Thermo",
            preparation="liquid",
        )
        for folder in config.folder_to_group
        if (data_root / "raw_data" / folder).is_dir()
    ]
    roots.extend(
        InventoryRoot(
            path=data_root / "raw_data_medical" / folder,
            source_kind=SourceKind.MEDICAL,
            instrument_key="Medical Raman",
            preparation="liquid",
        )
        for folder in config.folder_to_group_medical
        if (data_root / "raw_data_medical" / folder).is_dir()
    )
    remeasurement_root = data_root / "임상데이터"
    if remeasurement_root.is_dir():
        roots.extend(
            InventoryRoot(
                path=path,
                source_kind=SourceKind.REMEASUREMENT,
                instrument_key="Thermo",
                preparation="liquid",
            )
            for path in sorted(remeasurement_root.iterdir())
            if path.is_dir() and _DATE_PATTERN.search(path.name) is not None
        )
    special_roots = (
        (
            data_root / "raw_data" / "20260709_BPRO,BNOR_1mW_0.05s_Ave100",
            SourceKind.BORAMAE_LIQUID,
            "liquid",
        ),
        (
            data_root / "20260716_Urine test (Powder_BNOR, BPRO)",
            SourceKind.BORAMAE_POWDER,
            "powder",
        ),
        (
            data_root / "20260715_Powder_Reproducibility test",
            SourceKind.POWDER_REPRODUCIBILITY,
            "powder",
        ),
    )
    roots.extend(
        InventoryRoot(
            path=path,
            source_kind=source_kind,
            instrument_key="Thermo",
            preparation=preparation,
        )
        for path, source_kind, preparation in special_roots
        if path.is_dir()
    )
    return tuple(roots)


def _acquisition_date(root: InventoryRoot) -> str:
    if root.acquisition_date is not None:
        return root.acquisition_date
    match = _DATE_PATTERN.search(root.path.name)
    return match.group("date") if match is not None else "undated"


def _inventory_item(root: InventoryRoot, path: Path) -> InventoryItem:
    source_name = path.name.casefold()
    if source_name == "multidata.txt" or "zone.identifier" in source_name:
        return InventoryItem(
            source_path=path,
            source_kind=root.source_kind,
            instrument_key=root.instrument_key,
            acquisition_date=_acquisition_date(root),
            root_key=root.path.resolve().as_uri(),
            parsed=None,
            status=InventoryStatus.EXCLUDED,
            reason_code="configured_exclusion",
        )
    if root.source_kind is SourceKind.MEDICAL and path.stem.casefold().endswith("_ave"):
        return InventoryItem(
            source_path=path,
            source_kind=root.source_kind,
            instrument_key=root.instrument_key,
            acquisition_date=_acquisition_date(root),
            root_key=root.path.resolve().as_uri(),
            parsed=None,
            status=InventoryStatus.EXCLUDED,
            reason_code="configured_average_exclusion",
        )
    try:
        parsed = parse_spectrum_name(path, root.preparation)
    except MalformedSpectrumNameError:
        return InventoryItem(
            source_path=path,
            source_kind=root.source_kind,
            instrument_key=root.instrument_key,
            acquisition_date=_acquisition_date(root),
            root_key=root.path.resolve().as_uri(),
            parsed=None,
            status=InventoryStatus.QUARANTINED,
            reason_code="malformed_filename",
        )
    return InventoryItem(
        source_path=path,
        source_kind=root.source_kind,
        instrument_key=root.instrument_key,
        acquisition_date=_acquisition_date(root),
        root_key=root.path.resolve().as_uri(),
        parsed=parsed,
        status=parsed.status,
    )


def inventory_spectra(roots: Sequence[InventoryRoot]) -> InventoryReport:
    items = tuple(
        _inventory_item(root, path)
        for root in roots
        for path in _candidate_paths(root)
    )
    return InventoryReport(
        items=items,
        counts=InventoryCounts(
            ready=sum(item.status is InventoryStatus.READY for item in items),
            derived=sum(item.status is InventoryStatus.DERIVED for item in items),
            excluded=sum(item.status is InventoryStatus.EXCLUDED for item in items),
            quarantined=sum(
                item.status is InventoryStatus.QUARANTINED for item in items
            ),
        ),
    )
