"""Typed validation for patient and retest form boundaries."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Final

PATIENT_ID_PATTERN: Final = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True, slots=True)
class ClinicalInputError(ValueError):
    field: str
    reason: str

    def __str__(self) -> str:
        return f"{self.field}: {self.reason}"


def parse_patient_id(raw_value: str) -> str:
    patient_id = raw_value.strip()
    if not patient_id or PATIENT_ID_PATTERN.fullmatch(patient_id) is None:
        raise ClinicalInputError("patient_id", "unsafe_format")
    return patient_id


def parse_bmi(raw_value: str) -> float:
    try:
        bmi = float(raw_value.strip())
    except ValueError as error:
        raise ClinicalInputError("bmi", "required_number") from error
    if not math.isfinite(bmi) or not 10.0 <= bmi <= 60.0:
        raise ClinicalInputError("bmi", "out_of_range")
    return bmi
