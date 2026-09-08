from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sers.master_data.dataset_manifest import build_dataset_manifest
from sers.master_data.lineage_types import (
    CreationProvenance,
    DatasetBuildRequest,
    DatasetMemberDraft,
    DatasetPolicy,
    QCEvaluationDraft,
)
from sers.master_data.qc_lineage import append_qc_evaluation
from sers.master_data.remote_cache import (
    LocalAssetSource,
    RemoteDatasetIntegrityError,
    pull_remote_dataset,
)
from sers.master_data.remote_manifest import (
    export_remote_dataset_manifest,
    load_remote_dataset_manifest,
)
from tests.master_data.lineage_test_support import seed_lineage
from tests.master_data.matching_test_support import database


def _frozen_manifest(
    connection: sqlite3.Connection,
    tmp_path: Path,
) -> tuple[str, Path]:
    seeded = seed_lineage(connection, "SITE-A", "PSEUDONYM001")
    raw_path = tmp_path / "source.csv"
    raw_path.write_bytes(seeded.measurement_id.encode())
    connection.execute(
        """UPDATE source_assets SET raw_uri = ?
           WHERE sha256 = ?""",
        (str(raw_path), seeded.artifact_sha256),
    )
    append_qc_evaluation(
        connection,
        QCEvaluationDraft(
            seeded.measurement_id,
            "qc-v1",
            "engine-v1",
            "pass",
            "{}",
        ),
    )
    manifest = build_dataset_manifest(
        connection,
        DatasetBuildRequest(
            name="remote",
            members=(
                DatasetMemberDraft(
                    seeded.measurement_id,
                    seeded.sample_label_id,
                    "train",
                    0,
                ),
            ),
            policy=DatasetPolicy("policy-v1", "qc-v1", ("pass",)),
            preprocessing_version="raw-v1",
            feature_schema_version="thermo-grid-v1",
            decision_policy_version="screening-v1",
            provenance=CreationProvenance(
                "pipeline",
                "revision-1",
                "remote-training",
            ),
        ),
    )
    return str(manifest.id), raw_path


def test_remote_manifest_is_portable_and_excludes_source_identity(
    tmp_path: Path,
) -> None:
    # Given: a frozen manifest backed by one immutable raw spectrum.
    with database(tmp_path / "master.db") as connection:
        manifest_id, raw_path = _frozen_manifest(connection, tmp_path)
        output = tmp_path / "remote-manifest.json"

        # When: the frozen lineage is exported for remote training.
        result = export_remote_dataset_manifest(connection, manifest_id, output)
        manifest = load_remote_dataset_manifest(output)

    # Then: the asset is content-addressed without exposing local or patient identity.
    asset = manifest.assets[0]
    assert result.asset_count == 1
    assert asset.storage_path == (
        f"raw/sha256/{asset.artifact_sha256}/{asset.artifact_sha256}"
    )
    serialized = output.read_text(encoding="utf-8")
    assert str(raw_path) not in serialized
    assert "PSEUDONYM001" not in serialized


def test_remote_pull_caches_hash_verified_assets(tmp_path: Path) -> None:
    # Given: a portable manifest and matching content-addressed source.
    with database(tmp_path / "master.db") as connection:
        manifest_id, raw_path = _frozen_manifest(connection, tmp_path)
        manifest_path = tmp_path / "remote-manifest.json"
        export_remote_dataset_manifest(connection, manifest_id, manifest_path)
    manifest = load_remote_dataset_manifest(manifest_path)
    asset = manifest.assets[0]
    source_root = tmp_path / "remote"
    remote_path = source_root / asset.storage_path
    remote_path.parent.mkdir(parents=True)
    remote_path.write_bytes(raw_path.read_bytes())

    # When: a remote training cache is populated twice.
    first = pull_remote_dataset(
        manifest,
        LocalAssetSource(source_root),
        tmp_path / "cache",
    )
    second = pull_remote_dataset(
        manifest,
        LocalAssetSource(source_root),
        tmp_path / "cache",
    )

    # Then: the first pull downloads and the second reuses verified bytes.
    assert (first.downloaded_count, first.cached_count) == (1, 0)
    assert (second.downloaded_count, second.cached_count) == (0, 1)
    assert first.assets[0].read_bytes() == raw_path.read_bytes()


def test_remote_pull_rejects_asset_with_wrong_hash(tmp_path: Path) -> None:
    # Given: a manifest whose remote asset bytes do not match its SHA-256.
    with database(tmp_path / "master.db") as connection:
        manifest_id, _ = _frozen_manifest(connection, tmp_path)
        manifest_path = tmp_path / "remote-manifest.json"
        export_remote_dataset_manifest(connection, manifest_id, manifest_path)
    manifest = load_remote_dataset_manifest(manifest_path)
    asset = manifest.assets[0]
    source_root = tmp_path / "remote"
    remote_path = source_root / asset.storage_path
    remote_path.parent.mkdir(parents=True)
    remote_path.write_bytes(b"corrupted")

    # When/Then: cache publication fails rather than trusting corrupted bytes.
    with pytest.raises(RemoteDatasetIntegrityError):
        pull_remote_dataset(
            manifest,
            LocalAssetSource(source_root),
            tmp_path / "cache",
        )
