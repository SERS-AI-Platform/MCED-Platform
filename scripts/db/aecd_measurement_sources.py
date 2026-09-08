from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

FOLDER_GROUPS: dict[str, str] = {
    "1. Prostate cancer (100개)": "PRO",
    "2. Breast cancer (30개)": "BRE",
    "3. Ovarian cancer (70개)": "OVA",
    "4. Lung cancer (300개)": "LUN",
    "5. Normal (100개)": "NOR",
    "6. Diabetes (100개)": "DIA",
    "7. High blood pressure (100개)": "HBP",
    "8. High blood pressure + Diabetes (100개)": "H.D.",
    "9. Colorectal cancer (300개)": "CRC",
    "10-1. C-Pancreatic cancer (70개)": "CPAN",
    "10-2. S-Pancreatic cancer (72개)": "SPAN",
    "10-3. Y-Pancreatic cancer (YPAN)": "YPAN",
    "11 BLC (299개)": "BLC",
    "12. Y-Normal (YNOR)": "YNOR",
}

REAGENT_PHASE_PRE_CHANGE_UNVERIFIED = "pre_change_unverified"
REAGENT_PHASE_POST_CHANGE_VERIFIED = "post_change_verified"
REAGENT_PHASE_NOT_APPLICABLE = "not_applicable"
_REPLICATE_TOKEN = r"ave\d*|\d+"


class ClassificationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SourceSpec:
    source_domain: str
    instrument_key: str
    preparation: str
    source_kind: str
    input_root: Path | None = None
    reagent_phase: str = REAGENT_PHASE_PRE_CHANGE_UNVERIFIED
    reagent_name: str | None = None
    reagent_evidence: str = "reducing-agent metadata not recorded in source files"


@dataclass(frozen=True, slots=True)
class MeasurementRecord:
    path: Path
    source_root: Path
    relative_path: str
    source_domain: str
    source_kind: str
    instrument_key: str
    preparation: str
    source_batch: str
    artifact_role: str
    variant: str
    is_averaged: bool
    measurement_key: str
    group_code: str | None
    sample_id: str
    replicate: int | None
    control_type: str | None
    acquisition_date: str | None
    source_sha256: str
    file_format: str
    reagent_phase: str
    reagent_name: str | None
    reagent_evidence: str
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class RejectedFile:
    path: Path
    source_domain: str
    reason_code: str
    reason: str
    source_sha256: str | None


@dataclass(frozen=True, slots=True)
class MeasurementScan:
    records: tuple[MeasurementRecord, ...]
    rejected: tuple[RejectedFile, ...]
    total_files: int
    points: int
    role_counts: tuple[tuple[str, int], ...]
    source_counts: tuple[tuple[str, int], ...]
    format_counts: tuple[tuple[str, int], ...]
    manifest_sha256: str
    averages_excluded: int = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _folder_group(source_batch: str) -> str | None:
    if source_batch in FOLDER_GROUPS:
        return FOLDER_GROUPS[source_batch]
    upper = source_batch.upper()
    codes = ("BPRO", "BNOR", "CPAN", "SPAN", "YPAN", "YNOR", "H.D.", "BLC", "PRO", "BRE", "OVA", "LUN", "DIA", "HBP", "CRC", "NOR", "PAN")
    for code in codes:
        if re.search(rf"(?<![A-Z]){re.escape(code)}(?![A-Z])", upper):
            return code
    return None


def _replicate_number(token: str) -> int | None:
    normalized = token.casefold()
    if normalized == "ave":
        return None
    if normalized.startswith("ave"):
        normalized = normalized[3:]
    return int(normalized)


def _sample_match(stem: str) -> tuple[str | None, str, int | None]:
    matches = list(re.finditer(
        r"(?P<group>[A-Za-z]+(?:\.[A-Za-z]+)*\.?)\s+"
        rf"(?P<sample>\d+(?:-\d+)*)_(?P<replicate>{_REPLICATE_TOKEN})",
        stem,
        re.IGNORECASE,
    ))
    if matches:
        match = matches[-1]
        group = match.group("group").strip().upper()
        replicate_token = match.group("replicate")
        replicate = _replicate_number(replicate_token)
        return group, match.group("sample"), replicate
    numeric = re.fullmatch(
        rf"(?P<sample>\d+(?:-\d+)*)_(?P<replicate>{_REPLICATE_TOKEN})",
        stem,
        re.IGNORECASE,
    )
    if numeric is None:
        raise ClassificationError(f"Cannot parse group/sample/replicate from {stem}")
    token = numeric.group("replicate")
    return None, numeric.group("sample"), _replicate_number(token)


def _control_match(stem: str) -> tuple[str, str, int | None] | None:
    match = re.fullmatch(
        r"(?P<control>PS|Sensor|Si\s+wafer|Si)\s+"
        rf"(?P<sample>\d+)_(?P<replicate>{_REPLICATE_TOKEN})",
        stem,
        re.IGNORECASE,
    )
    if match is None:
        return None
    control = re.sub(r"\s+", "_", match.group("control").upper())
    token = match.group("replicate")
    replicate = _replicate_number(token)
    return control, match.group("sample"), replicate


def _reference_match(stem: str) -> tuple[str, str, int | None] | None:
    match = re.fullmatch(
        r"(?P<sample>.+?)_(?P<control>PS|Si(?:\s+wafer)?)_"
        rf"(?P<replicate>{_REPLICATE_TOKEN})",
        stem,
        re.IGNORECASE,
    )
    if match is None:
        return None
    control = re.sub(r"\s+", "_", match.group("control").upper())
    token = match.group("replicate")
    replicate = _replicate_number(token)
    return control, match.group("sample"), replicate


def _source_kind(spec: SourceSpec, source_batch: str) -> tuple[str, str]:
    batch = source_batch.casefold()
    if spec.source_domain == "thermo":
        if spec.source_kind != "thermo":
            return spec.source_kind, spec.preparation
        if "repro" in batch:
            return "powder_reproducibility", "powder"
        if "powder" in batch:
            return "boramae_powder", "powder"
        if "boramae" in batch or "bpro" in batch or "bnor" in batch or "20260709" in batch:
            return "boramae_liquid", "liquid"
        return "thermo_liquid", spec.preparation
    if spec.source_domain == "equipment_test":
        return f"equipment_{spec.source_kind}", spec.preparation
    return spec.source_kind, spec.preparation


def _date_from_path(path: Path) -> str | None:
    match = re.search(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)", str(path))
    if match is None:
        return None
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"


def _source_batch(path: Path, relative: Path, root: Path, spec: SourceSpec, stem: str) -> str:
    first = relative.parts[0] if relative.parts else root.name
    if path.parent != root:
        return root.name if first.casefold() in {"background", "reference", "0. blank"} else first
    if spec.source_kind == "powder_reproducibility":
        condition = re.match(r"(?P<condition>.+?)_(?:BPRO|BNOR)(?:\s|_)", stem, re.IGNORECASE)
        if condition is not None:
            return f"{root.name}:{condition.group('condition')}"
    return root.name


def classify_measurement_file(path: Path, root: Path, spec: SourceSpec) -> MeasurementRecord:
    relative = path.relative_to(root)
    if path.name.casefold() == "multidata.txt":
        raise ClassificationError("MultiData.txt contains multiple intensity columns")
    stem = path.stem.strip()
    source_batch = _source_batch(path, relative, root, spec, stem)
    source_kind, preparation = _source_kind(spec, source_batch)
    is_averaged = bool(re.search(r"_ave$", stem, re.IGNORECASE))
    condition_match = re.fullmatch(
        r"(?P<group>[A-Za-z]+)_(?P<condition>Fasting|Post surgery|Post chemo)_"
        r"(?P<sample>\d+)_(?P<replicate>ave|\d+)",
        stem,
        re.IGNORECASE,
    )
    short_condition_match = re.fullmatch(
        r"(?P<group>[A-Za-z]+)(?:\s+|_)(?P<sample>\d+)(?:\s+|_)"
        rf"(?P<condition>F|NF)_(?P<replicate>{_REPLICATE_TOKEN})",
        stem,
        re.IGNORECASE,
    )
    metabolite_match = re.fullmatch(
        rf"(?P<name>.+?)(?:_(?P<replicate>{_REPLICATE_TOKEN}))?",
        stem,
        re.IGNORECASE,
    )
    control = _control_match(stem)
    reference = _reference_match(stem)
    condition: str | None = None
    group_code: str | None
    control_type: str | None = None
    if spec.source_domain == "metabolite":
        if metabolite_match is None:
            raise ClassificationError(f"Cannot parse metabolite file: {relative}")
        group_code = "METABOLITE"
        sample_id = metabolite_match.group("name")
        token = metabolite_match.group("replicate")
        replicate = None if token is None else _replicate_number(token)
        control_type = "METABOLITE"
    elif condition_match is not None or short_condition_match is not None:
        match = condition_match or short_condition_match
        group_code = match.group("group").upper()
        sample_id = match.group("sample")
        token = match.group("replicate")
        replicate = _replicate_number(token)
        condition_token = match.group("condition").casefold()
        condition = {"f": "fasting", "nf": "non_fasting"}.get(condition_token, condition_token.replace(" ", "_"))
    elif reference is not None and spec.source_domain == "mapping_reference":
        control_type, sample_id, replicate = reference
        group_code = None
    elif control is not None:
        control_type, sample_id, replicate = control
        group_code = None
    else:
        group_code, sample_id, replicate = _sample_match(stem)
        if group_code is None:
            group_code = _folder_group(source_batch)
        if group_code is None:
            raise ClassificationError(f"No group code for {relative}")
        control_type = group_code if group_code in {"BLANK", "PS", "SENSOR", "SI", "SI_WAFER"} else None
    background_path = any(part.casefold() == "background" for part in relative.parts)
    reference_path = any(part.casefold() in {"reference", "calibration_control"} for part in relative.parts)
    if control_type is None and reference_path and group_code is not None:
        control_type = group_code
    if is_averaged:
        artifact_role = "average"
        variant = "average"
    elif spec.source_domain == "metabolite":
        artifact_role = "metabolite_reference"
        variant = "reference"
    elif condition is not None:
        artifact_role = "replicate"
        variant = condition
    elif control_type is not None or reference_path and group_code is None:
        artifact_role = "calibration_control"
        variant = "control"
    elif background_path:
        artifact_role = "background"
        variant = "background"
    elif spec.source_domain == "medical":
        artifact_role = "raw"
        variant = "raw"
    elif spec.source_domain == "equipment_test":
        artifact_role = "equipment_test"
        variant = "equipment"
    else:
        artifact_role = "replicate"
        variant = "replicate"
    suffix = "average" if is_averaged else str(replicate or 0)
    identity = ":".join(part for part in (control_type or group_code or "UNSET", condition) if part)
    measurement_key = f"{spec.source_domain}:{source_batch}:{identity}:{sample_id}:{suffix}"
    return MeasurementRecord(
        path=path,
        source_root=root,
        relative_path=relative.as_posix(),
        source_domain=spec.source_domain,
        source_kind=source_kind,
        instrument_key=spec.instrument_key,
        preparation=preparation,
        source_batch=source_batch,
        artifact_role=artifact_role,
        variant=variant,
        is_averaged=is_averaged,
        measurement_key=measurement_key,
        group_code=group_code,
        sample_id=sample_id,
        replicate=replicate,
        control_type=control_type,
        acquisition_date=_date_from_path(path),
        source_sha256="",
        file_format=path.suffix.casefold().lstrip("."),
        reagent_phase=spec.reagent_phase,
        reagent_name=spec.reagent_name,
        reagent_evidence=spec.reagent_evidence,
        metadata={},
    )
