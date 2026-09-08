from __future__ import annotations

import hashlib
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .spectrum_types import RawUri, Sha256, StoredRaw


@dataclass(frozen=True, slots=True)
class RawStoreIntegrityError(OSError):
    path: Path

    def __str__(self) -> str:
        return f"raw-store blob does not match its content address: {self.path}"


@dataclass(frozen=True, slots=True)
class RawStoreWrite:
    stored: StoredRaw
    created: bool


def hash_source_file(path: Path) -> Sha256:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return Sha256(digest.hexdigest())


def _hash_descriptor(descriptor: int) -> Sha256:
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return Sha256(digest.hexdigest())


def _open_directory(parent: int, name: str, path: Path) -> int:
    try:
        os.mkdir(name, dir_fd=parent)
    except FileExistsError as error:
        metadata = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode):
            raise RawStoreIntegrityError(path) from error
    try:
        return os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent,
        )
    except OSError as error:
        raise RawStoreIntegrityError(path) from error


def _inspect_blob(
    parent: int,
    name: str,
    expected_sha256: Sha256,
    path: Path,
) -> int | None:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=parent,
        )
    except FileNotFoundError:
        return None
    except OSError as error:
        raise RawStoreIntegrityError(path) from error
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise RawStoreIntegrityError(path)
    if _hash_descriptor(descriptor) != expected_sha256:
        raise RawStoreIntegrityError(path)
    return metadata.st_size


def _open_store_root(raw_store: Path) -> tuple[Path, int]:
    absolute = raw_store.absolute()
    try:
        absolute.mkdir(parents=True, exist_ok=True)
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise RawStoreIntegrityError(absolute) from error
    if resolved != absolute:
        raise RawStoreIntegrityError(absolute)
    try:
        descriptor = os.open(
            resolved,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
    except OSError as error:
        raise RawStoreIntegrityError(resolved) from error
    return resolved, descriptor


def _root_is_unchanged(root: Path, descriptor: int) -> bool:
    try:
        current = os.stat(root, follow_symlinks=False)
    except OSError:
        return False
    opened = os.fstat(descriptor)
    return (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino)


def _unlink_if_present(parent: int, name: str) -> None:
    try:
        os.unlink(name, dir_fd=parent)
    except FileNotFoundError:
        return


def store_raw_file(
    source: Path,
    raw_store: Path,
    expected_sha256: Sha256 | None = None,
) -> StoredRaw:
    return store_raw_file_with_status(source, raw_store, expected_sha256).stored


def store_raw_file_with_status(
    source: Path,
    raw_store: Path,
    expected_sha256: Sha256 | None = None,
) -> RawStoreWrite:
    source_hash = (
        hash_source_file(source) if expected_sha256 is None else expected_sha256
    )
    root, root_descriptor = _open_store_root(raw_store)
    destination = root / "sha256" / source_hash / source_hash
    sha_descriptor = -1
    hash_descriptor = -1
    temporary_name = f".{source_hash}.{uuid4().hex}.tmp"
    created = False
    try:
        sha_descriptor = _open_directory(
            root_descriptor,
            "sha256",
            root / "sha256",
        )
        hash_descriptor = _open_directory(
            sha_descriptor,
            source_hash,
            destination.parent,
        )
        size_bytes = _inspect_blob(
            hash_descriptor,
            source_hash,
            source_hash,
            destination,
        )
        if size_bytes is None:
            try:
                temporary_descriptor = os.open(
                    temporary_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=hash_descriptor,
                )
            except OSError as error:
                raise RawStoreIntegrityError(destination.parent / temporary_name) from error
            with os.fdopen(
                temporary_descriptor,
                "wb",
            ) as output_stream, source.open("rb") as input_stream:
                shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
                output_stream.flush()
                os.fsync(output_stream.fileno())
            temporary_size = _inspect_blob(
                hash_descriptor,
                temporary_name,
                source_hash,
                destination.parent / temporary_name,
            )
            assert temporary_size is not None
            try:
                os.link(
                    temporary_name,
                    source_hash,
                    src_dir_fd=hash_descriptor,
                    dst_dir_fd=hash_descriptor,
                    follow_symlinks=False,
                )
                created = True
            except FileExistsError:
                created = False
            size_bytes = _inspect_blob(
                hash_descriptor,
                source_hash,
                source_hash,
                destination,
            )
            assert size_bytes is not None
        if not _root_is_unchanged(root, root_descriptor):
            if created:
                os.unlink(source_hash, dir_fd=hash_descriptor)
            raise RawStoreIntegrityError(root)
        return RawStoreWrite(
            stored=StoredRaw(
                sha256=source_hash,
                raw_uri=RawUri(str(destination)),
                size_bytes=size_bytes,
            ),
            created=created,
        )
    finally:
        if hash_descriptor >= 0:
            _unlink_if_present(hash_descriptor, temporary_name)
            os.close(hash_descriptor)
        if sha_descriptor >= 0:
            os.close(sha_descriptor)
        os.close(root_descriptor)
