from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sers.master_data.raw_store import RawStoreIntegrityError, store_raw_file


@pytest.mark.parametrize("target_bytes", (b"matching", b"different"))
def test_store_rejects_symlink_blob_for_matching_or_different_target(
    tmp_path: Path,
    target_bytes: bytes,
) -> None:
    # Given: the content-addressed destination is a symlink outside the store.
    source = tmp_path / "source.csv"
    source.write_bytes(b"matching")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    outside = tmp_path / "outside.csv"
    outside.write_bytes(target_bytes)
    destination = tmp_path / "raw" / "sha256" / digest / digest
    destination.parent.mkdir(parents=True)
    destination.symlink_to(outside)

    # When/Then: matching content cannot make a symlink an accepted raw blob.
    with pytest.raises(RawStoreIntegrityError):
        store_raw_file(source, tmp_path / "raw")
    assert outside.read_bytes() == target_bytes


@pytest.mark.parametrize("symlink_component", ("root", "hash-parent"))
def test_store_rejects_symlinked_root_or_parent_escape(
    tmp_path: Path,
    symlink_component: str,
) -> None:
    # Given: a store component redirects content-addressed writes outside.
    source = tmp_path / "source.csv"
    source.write_bytes(b"spectrum")
    outside = tmp_path / "outside"
    outside.mkdir()
    raw_store = tmp_path / "raw"
    if symlink_component == "root":
        raw_store.symlink_to(outside, target_is_directory=True)
    else:
        raw_store.mkdir()
        (raw_store / "sha256").symlink_to(outside, target_is_directory=True)

    # When/Then: the escape is rejected and no outside blob is published.
    with pytest.raises(RawStoreIntegrityError):
        store_raw_file(source, raw_store)
    assert tuple(outside.rglob("*")) == ()
