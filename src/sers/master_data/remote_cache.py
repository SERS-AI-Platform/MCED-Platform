from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .remote_manifest import RemoteDatasetManifest


class AssetSource(Protocol):
    def copy_to(self, storage_path: str, destination: Path) -> None: ...


@dataclass(frozen=True, slots=True)
class LocalAssetSource:
    root: Path

    def copy_to(self, storage_path: str, destination: Path) -> None:
        with (
            (self.root / storage_path).open("rb") as source,
            destination.open("wb") as target,
        ):
            shutil.copyfileobj(source, target, length=1024 * 1024)


@dataclass(frozen=True, slots=True)
class RemoteDatasetIntegrityError(OSError):
    expected_sha256: str
    actual_sha256: str

    def __str__(self) -> str:
        return "remote dataset asset does not match its manifest SHA-256"


@dataclass(frozen=True, slots=True)
class RemotePullResult:
    assets: tuple[Path, ...]
    downloaded_count: int
    cached_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected_sha256: str) -> None:
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise RemoteDatasetIntegrityError(expected_sha256, actual_sha256)


def pull_remote_dataset(
    manifest: RemoteDatasetManifest,
    source: AssetSource,
    cache_root: Path,
) -> RemotePullResult:
    cached = 0
    downloaded = 0
    paths: list[Path] = []
    for asset in manifest.assets:
        destination = cache_root / asset.storage_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            _verify(destination, asset.artifact_sha256)
            cached += 1
            paths.append(destination)
            continue
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
            source.copy_to(asset.storage_path, temporary_path)
            _verify(temporary_path, asset.artifact_sha256)
            os.replace(temporary_path, destination)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        downloaded += 1
        paths.append(destination)
    return RemotePullResult(tuple(paths), downloaded, cached)
