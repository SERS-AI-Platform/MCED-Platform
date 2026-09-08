from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ._compat import assert_never
from .spectrum_types import (
    FastingState,
    IdentityStatus,
    InventoryStatus,
    MaterialKind,
    ParsedSpectrumName,
    SpecimenTiming,
)

_SPECTRUM_PATTERN = re.compile(
    r"^(?:(?P<specimen>Po\.)\s+)?"
    r"(?:(?P<sigma>Sigma\s+\d+)_)?"
    r"(?P<group>[A-Za-z]+(?:\.[A-Za-z]+)*\.?)[\s_]+"
    r"(?:(?P<descriptor>Fasting|Post\s+surgery)[\s_]+"
    r"(?P<descriptor_sample>[A-Za-z0-9-]+)|"
    r"(?P<sample>[A-Za-z0-9-]+)(?:[\s_]+(?P<fasting>NF|F))?)"
    r"_(?P<replicate>\d+|ave)$",
    re.IGNORECASE,
)
_AMBIGUOUS_ALIASES = {
    "BLA": ("BLC",),
    "BLC": ("BLA",),
    "PAN": ("CPAN",),
    "CPAN": ("PAN",),
    "YPAN": ("PAN",),
}


@dataclass(frozen=True, slots=True)
class MalformedSpectrumNameError(ValueError):
    filename: str

    def __str__(self) -> str:
        return f"unsupported spectrum filename: {self.filename}"


def _material_kind(path: Path, group_code: str) -> MaterialKind:
    normalized_parts = {part.casefold() for part in path.parts}
    if any("reference" in part for part in normalized_parts):
        return MaterialKind.REFERENCE
    if any("blank" in part for part in normalized_parts):
        return MaterialKind.BLANK
    if group_code in {"MB", "MATRIXBLANK"}:
        return MaterialKind.MATRIX_BLANK
    return MaterialKind.BIOLOGICAL


def parse_spectrum_name(path: Path, preparation: str) -> ParsedSpectrumName:
    match = _SPECTRUM_PATTERN.fullmatch(path.stem)
    if match is None:
        raise MalformedSpectrumNameError(path.name)
    group_code = match.group("group").strip(".").upper()
    replicate_text = match.group("replicate").lower()
    sigma = match.group("sigma")
    descriptor = (match.group("descriptor") or "").casefold()
    fasting = match.group("fasting") or {"fasting": "F"}.get(descriptor)
    specimen = match.group("specimen") or {"post surgery": "Post surgery"}.get(descriptor)
    source_sample_code = match.group("sample") or match.group("descriptor_sample")
    assert source_sample_code is not None
    match fasting:
        case None:
            fasting_state = FastingState.UNSPECIFIED
        case str() as marker:
            fasting_state = {
                "F": FastingState.FASTING,
                "NF": FastingState.NON_FASTING,
            }[marker.upper()]
        case unreachable:
            assert_never(unreachable)
    match specimen:
        case None:
            specimen_timing = SpecimenTiming.UNSPECIFIED
        case str():
            specimen_timing = SpecimenTiming.POST_OPERATIVE
        case unreachable:
            assert_never(unreachable)
    aliases = _AMBIGUOUS_ALIASES.get(group_code, ())
    identity_status = IdentityStatus.ALIAS_REVIEW if aliases else IdentityStatus.CANONICAL
    return ParsedSpectrumName(
        group_code=group_code,
        source_sample_code=source_sample_code,
        replicate_index=None if replicate_text == "ave" else int(replicate_text),
        preparation=preparation,
        status=(InventoryStatus.DERIVED if replicate_text == "ave" else InventoryStatus.READY),
        material_kind=_material_kind(path, group_code),
        identity_status=identity_status,
        fasting_state=fasting_state,
        specimen_timing=specimen_timing,
        lot_code=sigma,
        alias_candidates=aliases,
    )
