from __future__ import annotations

import csv
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar, NewType

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import override

DiagnosisId = NewType("DiagnosisId", int)
DatabaseDiagnosis = tuple[int, str]


class CaseStatus(str, Enum):
    CANCER_CASE = "cancer_case"
    HEALTHY_CONTROL = "healthy_control"
    DISEASE_CONTROL = "disease_control"
    EXCLUDED = "excluded"


class BoundaryModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, str_strip_whitespace=True)


class LabelRule(BoundaryModel):
    mapping_version: str = Field(min_length=1)
    source_cancer_file: str = Field(min_length=1)
    cohort_group: str = Field(min_length=1)
    source_cancer_type: str = Field(min_length=1)
    study_cancer_type: str = Field(min_length=1)
    case_status: CaseStatus


class ClinicalSource(BoundaryModel):
    solum_label: str = Field(min_length=1)
    source_cancer_file: str = Field(min_length=1)
    cohort_group: str = Field(min_length=1)
    source_cancer_type: str = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class PlannedDiagnosis:
    diagnosis_id: DiagnosisId
    source_cancer_file_raw: str
    study_cancer_type: str
    case_status: CaseStatus


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    mapping_version: str
    rows: tuple[PlannedDiagnosis, ...]
    input_rows: int
    database_rows: int


@dataclass(frozen=True, slots=True)
class MigrationError(Exception):
    message: str

    @override
    def __str__(self) -> str:
        return self.message


def normalized_label(value: str) -> str:
    return re.sub(r"[_\s]+", "", value).casefold()


def csv_rows(path: Path) -> tuple[Mapping[str, str | None], ...]:
    if not path.is_file():
        raise MigrationError(f"CSV not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise MigrationError(f"CSV has no header: {path}")
        return tuple(reader)


def load_rules(path: Path) -> tuple[LabelRule, ...]:
    rules = tuple(LabelRule.model_validate(row) for row in csv_rows(path))
    versions = {rule.mapping_version for rule in rules}
    if len(versions) != 1:
        raise MigrationError("Mapping file must contain exactly one version")
    keys = {(rule.source_cancer_file, rule.cohort_group, rule.source_cancer_type) for rule in rules}
    if len(keys) != len(rules):
        raise MigrationError("Mapping file contains duplicate source keys")
    return rules


def load_sources(path: Path) -> tuple[ClinicalSource, ...]:
    sources = tuple(
        ClinicalSource.model_validate(
            {
                "solum_label": row.get("solum_label"),
                "source_cancer_file": row.get("source_cancer_file"),
                "cohort_group": row.get("group"),
                "source_cancer_type": row.get("cancer_type"),
            }
        )
        for row in csv_rows(path)
    )
    labels = {normalized_label(source.solum_label) for source in sources}
    if len(labels) != len(sources):
        raise MigrationError("Input contains duplicate normalized solum_label values")
    return sources


def resolve_labels(
    database_rows: tuple[DatabaseDiagnosis, ...],
    sources: tuple[ClinicalSource, ...],
    rules: tuple[LabelRule, ...],
) -> MigrationPlan:
    rule_by_key = {
        (rule.source_cancer_file, rule.cohort_group, rule.source_cancer_type): rule
        for rule in rules
    }
    source_by_label = {normalized_label(source.solum_label): source for source in sources}
    planned: list[PlannedDiagnosis] = []
    for raw_id, raw_label in database_rows:
        source = source_by_label.get(normalized_label(raw_label))
        if source is None:
            raise MigrationError("Database contains a sample absent from the input CSV")
        key = (source.source_cancer_file, source.cohort_group, source.source_cancer_type)
        rule = rule_by_key.get(key)
        if rule is None:
            raise MigrationError(f"No mapping rule for source combination: {key}")
        planned.append(
            PlannedDiagnosis(
                diagnosis_id=DiagnosisId(raw_id),
                source_cancer_file_raw=source.source_cancer_file,
                study_cancer_type=rule.study_cancer_type,
                case_status=rule.case_status,
            )
        )
    if len(planned) != len(sources):
        raise MigrationError("Input CSV and database diagnosis counts do not match")
    return MigrationPlan(
        mapping_version=rules[0].mapping_version,
        rows=tuple(planned),
        input_rows=len(sources),
        database_rows=len(database_rows),
    )
