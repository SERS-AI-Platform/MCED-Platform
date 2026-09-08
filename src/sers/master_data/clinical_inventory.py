from __future__ import annotations

import stat
from pathlib import Path
from typing import Final

from .clinical_source_contracts import (
    clinical_source,
    fixed_clinical_sources,
)
from .clinical_types import (
    ClinicalInventory,
    ClinicalInventoryCounts,
    ClinicalInventoryIssue,
    ClinicalInventoryIssueReason,
    ClinicalSource,
    ClinicalSourceVersion,
)

_PROSPECTIVE_GROUPS: Final = (
    ("SPAN", "SAMSUNG", "SMCXD02"),
    ("YPAN", "YONSEI", "SMCXD04"),
)
_PROSPECTIVE_EXCLUSIONS: Final = {
    "SMCXD04_SERS_dataset.csv",
    "SN.csv",
}


def _availability(path: Path) -> ClinicalInventoryIssueReason | None:
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        return "missing_source"
    except OSError:
        return "unreadable_source"
    if not stat.S_ISREG(mode) or mode & 0o444 == 0:
        return "unreadable_source"
    try:
        with path.open("rb") as stream:
            stream.read(0)
    except OSError:
        return "unreadable_source"
    return None


def _issue(
    source: ClinicalSource,
    reason_code: ClinicalInventoryIssueReason,
) -> ClinicalInventoryIssue:
    return ClinicalInventoryIssue(
        path=source.path,
        site_code=source.site_code,
        protocol_code=source.protocol_code,
        source_group=source.source_group,
        reason_code=reason_code,
    )


def _fixed_inventory(
    root: Path,
) -> tuple[tuple[ClinicalSource, ...], tuple[ClinicalInventoryIssue, ...]]:
    sources: list[ClinicalSource] = []
    issues: list[ClinicalInventoryIssue] = []
    for source in fixed_clinical_sources(root):
        reason = _availability(source.path)
        if reason is None:
            sources.append(source)
        else:
            issues.append(_issue(source, reason))
    return tuple(sources), tuple(issues)


def _prospective_inventory(
    root: Path,
) -> tuple[tuple[ClinicalSource, ...], tuple[ClinicalInventoryIssue, ...]]:
    sources: list[ClinicalSource] = []
    issues: list[ClinicalInventoryIssue] = []
    for group, site_code, protocol_code in _PROSPECTIVE_GROUPS:
        group_root = root / f"10. 췌장암/{group}"
        paths = tuple(
            path
            for path in sorted(group_root.glob("*/*.csv"))
            if path.name not in _PROSPECTIVE_EXCLUSIONS
        )
        if not paths:
            issues.append(
                ClinicalInventoryIssue(
                    path=group_root,
                    site_code=site_code,
                    protocol_code=protocol_code,
                    source_group=group,
                    reason_code="expected_group_empty",
                )
            )
            continue
        for path in paths:
            source = clinical_source(
                root,
                str(path.relative_to(root)),
                site_code,
                protocol_code,
                group,
                canonical_alias="PAN" if group == "YPAN" else None,
            )
            reason = _availability(path)
            if reason is None:
                sources.append(source)
            else:
                issues.append(_issue(source, reason))
    return tuple(sources), tuple(issues)


def inventory_clinical_sources(root: Path) -> ClinicalInventory:
    fixed_sources, fixed_issues = _fixed_inventory(root)
    prospective_sources, prospective_issues = _prospective_inventory(root)
    sources = (*fixed_sources, *prospective_sources)
    issues = (*fixed_issues, *prospective_issues)
    return ClinicalInventory(
        sources=sources,
        counts=ClinicalInventoryCounts(
            current=sum(
                source.source_version is ClinicalSourceVersion.CURRENT
                for source in sources
            ),
            historical=sum(
                source.source_version is ClinicalSourceVersion.HISTORICAL
                for source in sources
            ),
        ),
        issues=issues,
    )
