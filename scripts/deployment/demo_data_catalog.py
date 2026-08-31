from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

CancerGroup = Literal["PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC", "NOR"]


@dataclass(frozen=True, slots=True)
class UsabilityPatient:
    patient_id: str
    source_group: CancerGroup
    source_folder: str
    source_prefix: str
    source_sample_id: int
    age: int
    sex: Literal["F", "M"]
    bmi: float
    expected_ssi: float
    expected_cancer_type: str

    @property
    def expected_ssi_band(self) -> str:
        if self.expected_ssi < 1.0:
            return "low"
        if self.expected_ssi < 4.0:
            return "medium"
        return "high"


USABILITY_PATIENTS: Final = (
    UsabilityPatient(
        "UT-DUMMY-001", "PRO", "1. Prostate cancer (100개)", "PRO", 31, 71, "M", 30.9, 6.96, "PRO"
    ),
    UsabilityPatient(
        "UT-DUMMY-002", "LUN", "4. Lung cancer (300개)", "LUN", 212, 72, "F", 25.45, 7.35, "LUN"
    ),
    UsabilityPatient(
        "UT-DUMMY-003",
        "CRC",
        "9. Colorectal cancer (300개)",
        "CRC",
        277,
        79,
        "F",
        24.8,
        9.22,
        "CRC",
    ),
    UsabilityPatient(
        "UT-DUMMY-004",
        "PAN",
        "10-1. C-Pancreatic cancer (70개)",
        "CPAN",
        43,
        78,
        "F",
        18.7,
        9.98,
        "PAN",
    ),
    UsabilityPatient(
        "UT-DUMMY-005", "OVA", "3. Ovarian cancer (70개)", "OVA", 48, 65, "F", 28.2, 4.81, "OVA"
    ),
    UsabilityPatient(
        "UT-DUMMY-006", "BRE", "2. Breast cancer (30개)", "BRE", 29, 54, "F", 22.0, 6.81, "BRE"
    ),
    UsabilityPatient(
        "UT-DUMMY-007", "BLC", "11 BLC (299개)", "BLC", 298, 82, "M", 23.8, 5.90, "BLC"
    ),
    UsabilityPatient(
        "UT-DUMMY-008", "NOR", "5. Normal (100개)", "NOR", 13, 43, "F", 20.1, 0.17, ""
    ),
    UsabilityPatient("UT-DUMMY-009", "NOR", "5. Normal (100개)", "NOR", 1, 35, "F", 22.5, 1.32, ""),
)
QC_PATIENT: Final = UsabilityPatient(
    "UT-QC-001",
    "LUN",
    "4. Lung cancer (300개)",
    "LUN",
    261,
    75,
    "M",
    21.66,
    9.98,
    "LUN",
)
