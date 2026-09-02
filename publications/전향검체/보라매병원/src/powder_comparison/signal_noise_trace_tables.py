from __future__ import annotations

import csv
from pathlib import Path

from .signal_noise_types import SignalNoiseDiagnostics


def write_signal_noise_trace_tables(
    diagnostics: SignalNoiseDiagnostics,
    directory: Path,
) -> None:
    trace_path = directory / "powder_patient_noise_error_trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "patient_id",
                "sample_id",
                "clinical_group",
                "powder_replicate_correlation",
                "powder_consensus_fraction",
                "powder_native_cancer_probability",
                "powder_native_screening_correct",
            ]
        )
        for row in diagnostics.patient_links:
            writer.writerow(
                [
                    row.patient_id,
                    row.sample_id,
                    row.clinical_group,
                    row.powder_replicate_correlation,
                    row.powder_consensus_fraction,
                    row.native_cancer_probability,
                    row.native_screening_correct,
                ]
            )

    conclusion_path = directory / "powder_signal_noise_conclusions.csv"
    with conclusion_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["evidence_order", "statement"])
        for index, conclusion in enumerate(diagnostics.conclusions, start=1):
            writer.writerow([index, conclusion])
