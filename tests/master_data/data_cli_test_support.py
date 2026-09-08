from __future__ import annotations

import sqlite3
from pathlib import Path

from click.testing import CliRunner

from sers.cli import cli
from sers.master_data.dataset_manifest import build_dataset_manifest
from sers.master_data.lineage_types import (
    CreationProvenance,
    DatasetBuildRequest,
    DatasetMemberDraft,
    DatasetPolicy,
    QCEvaluationDraft,
)
from sers.master_data.qc_lineage import append_qc_evaluation


def write_inputs(root: Path, source_key: str) -> tuple[Path, Path]:
    clinical = root / "clinical.csv"
    clinical.write_text(
        f"SUBJID,solum_label,pathology_result\n"
        f"{source_key},PRO {source_key},Cancer\n",
        encoding="utf-8-sig",
    )
    spectra = root / "spectra"
    spectra.mkdir()
    (spectra / f"PRO {source_key}_1.CSV").write_text(
        "100,1\n101,2\n",
        encoding="utf-8",
    )
    return clinical, spectra


def invoke_ingestion(
    runner: CliRunner,
    db: Path,
    raw_store: Path,
    clinical: Path,
    spectra: Path,
) -> tuple[str, str]:
    clinical_result = runner.invoke(
        cli,
        [
            "data",
            "ingest-clinical",
            "--db",
            str(db),
            "--source",
            str(clinical),
            "--site-code",
            "SITE-A",
            "--protocol-code",
            "SYNTHETIC",
            "--source-group",
            "PRO",
            "--patient-id-field",
            "SUBJID",
            "--raw-store",
            str(raw_store),
        ],
    )
    spectrum_result = runner.invoke(
        cli,
        [
            "data",
            "ingest-spectra",
            "--db",
            str(db),
            "--root",
            str(spectra),
            "--site-code",
            "SITE-A",
            "--source-kind",
            "remeasurement",
            "--raw-store",
            str(raw_store),
        ],
    )
    assert clinical_result.exit_code == 0
    assert spectrum_result.exit_code == 0
    return clinical_result.output, spectrum_result.output


def invoke_labels(runner: CliRunner, db: Path) -> None:
    result = runner.invoke(
        cli,
        [
            "data",
            "build-labels",
            "--db",
            str(db),
            "--task-name",
            "prostate-screening",
            "--definition-version",
            "pathology-v1",
            "--observation-code",
            "pathology_result",
            "--label-source",
            "pathology",
            "--mapping",
            "Cancer=Cancer",
        ],
    )
    assert result.exit_code == 0


def build_manifest(db: Path) -> str:
    with sqlite3.connect(db) as connection:
        measurement_id = connection.execute(
            "SELECT id FROM measurements"
        ).fetchone()[0]
        label_id = connection.execute("SELECT id FROM sample_labels").fetchone()[0]
        append_qc_evaluation(
            connection,
            QCEvaluationDraft(
                measurement_id,
                "spectrum-qc-v1",
                "engine-v1",
                "pass",
                "{}",
            ),
        )
        manifest = build_dataset_manifest(
            connection,
            DatasetBuildRequest(
                name="synthetic-training",
                members=(
                    DatasetMemberDraft(measurement_id, label_id, "train", 0),
                ),
                policy=DatasetPolicy(
                    "qc-policy-v1",
                    "spectrum-qc-v1",
                    ("pass",),
                ),
                preprocessing_version="baseline-v1",
                feature_schema_version="raman-grid-v1",
                decision_policy_version="screening-v1",
                provenance=CreationProvenance(
                    "cli-test",
                    "revision-1",
                    "model-training",
                ),
            ),
        )
        return str(manifest.id)
