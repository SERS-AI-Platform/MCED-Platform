from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import replace
from pathlib import Path

try:
    from .aecd_measurement_sources import (
        REAGENT_PHASE_NOT_APPLICABLE,
        REAGENT_PHASE_POST_CHANGE_VERIFIED,
        REAGENT_PHASE_PRE_CHANGE_UNVERIFIED,
        ClassificationError,
        MeasurementRecord,
        MeasurementScan,
        RejectedFile,
        SourceSpec,
        classify_measurement_file,
        sha256_file,
    )
    from .aecd_spectrum_formats import SpectrumFormatError, read_spectrum
except ImportError:
    from aecd_measurement_sources import (
        REAGENT_PHASE_NOT_APPLICABLE,
        REAGENT_PHASE_POST_CHANGE_VERIFIED,
        REAGENT_PHASE_PRE_CHANGE_UNVERIFIED,
        ClassificationError,
        MeasurementRecord,
        MeasurementScan,
        RejectedFile,
        SourceSpec,
        classify_measurement_file,
        sha256_file,
    )
    from aecd_spectrum_formats import SpectrumFormatError, read_spectrum


def discover_source_specs(repo_root: Path) -> tuple[SourceSpec, ...]:
    pre_change_evidence = (
        "source is temporally earlier than the current mapping cohort; "
        "the reducing-agent identity is not recorded in the source metadata"
    )
    post_change_evidence = (
        "measurement.runs reagent_lot observation: Methoxyamine hydrochloride 98%, "
        "acquired mapping runs dated 2026-08-10 through 2026-08-14"
    )
    specs: list[SourceSpec] = [
        SourceSpec(
            "thermo",
            "Thermo",
            "liquid",
            "thermo",
            reagent_phase=REAGENT_PHASE_PRE_CHANGE_UNVERIFIED,
            reagent_evidence=pre_change_evidence,
        ),
        SourceSpec(
            "medical",
            "Medical Raman",
            "liquid",
            "medical",
            reagent_phase=REAGENT_PHASE_PRE_CHANGE_UNVERIFIED,
            reagent_evidence=pre_change_evidence,
        ),
        SourceSpec(
            "remeasurement",
            "Thermo",
            "liquid",
            "remeasurement_liquid",
            reagent_phase=REAGENT_PHASE_PRE_CHANGE_UNVERIFIED,
            reagent_evidence=pre_change_evidence,
        ),
    ]
    equipment_root = repo_root / "data" / "equipment_test_data"
    instruments = {"handheld": "Handheld", "medical_raw": "Medical Raman", "medical_removed": "Medical Raman", "nanoscope": "NanoScope", "thermo": "Thermo"}
    for child in sorted(equipment_root.iterdir()) if equipment_root.is_dir() else ():
        if child.is_dir():
            specs.append(
                SourceSpec(
                    "equipment_test",
                    instruments.get(child.name, child.name),
                    "liquid",
                    child.name,
                    reagent_phase=REAGENT_PHASE_NOT_APPLICABLE,
                    reagent_evidence="instrument reproducibility/control data; not a clinical reducing-agent cohort",
                )
            )
    data_root = repo_root / "data"
    mapping_root = data_root / "mapping"
    if mapping_root.is_dir():
        specs.append(
            SourceSpec(
                "mapping",
                "Thermo",
                "liquid",
                "mapping_clinical",
                mapping_root,
                REAGENT_PHASE_POST_CHANGE_VERIFIED,
                "Methoxyamine hydrochloride 98%",
                post_change_evidence,
            )
        )
        reference_root = mapping_root / "Thermo Reference"
        if reference_root.is_dir():
            specs.append(
                SourceSpec(
                    "mapping_reference",
                    "Thermo",
                    "liquid",
                    "thermo_reference",
                    reference_root,
                    REAGENT_PHASE_NOT_APPLICABLE,
                    None,
                    "polystyrene/Si calibration reference; not a clinical reducing-agent cohort",
                )
            )
    extra_specs = (
        ("20260715_Powder_Reproducibility test", "thermo", "Thermo", "powder", "powder_reproducibility", REAGENT_PHASE_PRE_CHANGE_UNVERIFIED, None, pre_change_evidence),
        ("20260716_Urine test (Powder_BNOR, BPRO)", "thermo", "Thermo", "powder", "boramae_powder", REAGENT_PHASE_PRE_CHANGE_UNVERIFIED, None, pre_change_evidence),
        ("Metabolite analysis_Thermo", "metabolite", "Thermo", "liquid", "metabolite_reference", REAGENT_PHASE_NOT_APPLICABLE, None, "metabolite reference material; not a clinical reducing-agent cohort"),
        ("OneDrive_2026-07-21 (2)", "thermo", "Thermo", "liquid", "thermo_onedrive_20260721_2", REAGENT_PHASE_PRE_CHANGE_UNVERIFIED, None, pre_change_evidence),
        ("OneDrive_2026-07-21 (3)", "thermo", "Thermo", "liquid", "thermo_onedrive_20260721_3", REAGENT_PHASE_PRE_CHANGE_UNVERIFIED, None, pre_change_evidence),
    )
    for directory, domain, instrument, preparation, kind, phase, name, evidence in extra_specs:
        input_root = data_root / directory
        if input_root.is_dir():
            specs.append(SourceSpec(domain, instrument, preparation, kind, input_root, phase, name, evidence))
    return tuple(specs)


def _spec_root(repo_root: Path, spec: SourceSpec) -> Path:
    if spec.input_root is not None:
        return spec.input_root
    roots = {"thermo": repo_root / "data" / "raw_data", "medical": repo_root / "data" / "raw_data_medical", "remeasurement": repo_root / "data" / "임상데이터"}
    return roots.get(spec.source_domain, repo_root / "data" / "equipment_test_data" / spec.source_kind)


def _rejection(path: Path, spec: SourceSpec, exc: Exception) -> RejectedFile:
    return RejectedFile(path, spec.source_domain, type(exc).__name__, str(exc), sha256_file(path))


def _source_paths(root: Path, spec: SourceSpec) -> tuple[Path, ...]:
    roots = tuple(sorted(root.glob("*_mapping"))) if spec.source_domain == "mapping" else (root,)
    return tuple(
        sorted(
            path
            for source_root in roots
            for path in source_root.rglob("*")
            if path.is_file() and path.suffix.casefold() in {".csv", ".txt"}
        )
    )


def scan_sources(repo_root: Path, specs: tuple[SourceSpec, ...], include_averages: bool) -> MeasurementScan:
    records: list[MeasurementRecord] = []
    rejected: list[RejectedFile] = []
    role_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()
    manifest = hashlib.sha256()
    points = 0
    total_files = 0
    averages_excluded = 0
    for spec in specs:
        root = _spec_root(repo_root, spec)
        if not root.is_dir():
            continue
        for path in _source_paths(root, spec):
            total_files += 1
            try:
                record = classify_measurement_file(path, root, spec)
                if not include_averages and record.is_averaged:
                    averages_excluded += 1
                    continue
                data = read_spectrum(path)
                record = replace(record, source_sha256=sha256_file(path), file_format=data.file_format, metadata=data.metadata)
                records.append(record)
                points += len(data.wavenumber)
                role_counts[record.artifact_role] += 1
                source_counts[f"{record.source_domain}:{record.source_kind}"] += 1
                format_counts[record.file_format] += 1
                manifest.update(record.source_sha256.encode())
                manifest.update(f"{record.source_domain}/{record.relative_path}".encode())
                manifest.update(record.reagent_phase.encode())
            except (ClassificationError, SpectrumFormatError) as exc:
                rejected.append(_rejection(path, spec, exc))
    if not records:
        raise ValueError("No valid measurement files found")
    return MeasurementScan(
        tuple(records),
        tuple(rejected),
        total_files,
        points,
        tuple(sorted(role_counts.items())),
        tuple(sorted(source_counts.items())),
        tuple(sorted(format_counts.items())),
        manifest.hexdigest(),
        averages_excluded,
    )
