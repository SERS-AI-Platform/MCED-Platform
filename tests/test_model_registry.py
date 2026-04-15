"""Enforce parity between `src/sers/models/_registry.py` and `models/MODEL_STATUS.md`.

If this test fails, either:
    - You added a model to the registry but forgot to document it, or
    - You documented a new model but forgot to register it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from sers.models._registry import MODEL_REGISTRY, ModelSpec, get_spec, list_by_category

REPO_ROOT = Path(__file__).resolve().parents[1]
STATUS_MD = REPO_ROOT / "models" / "MODEL_STATUS.md"


def _extract_status_keys() -> set[str]:
    """Extract model keys from MODEL_STATUS.md's ModelSpec code block."""
    text = STATUS_MD.read_text(encoding="utf-8")
    # Match lines like:   "usersnet": ModelSpec(
    return set(re.findall(r'"([a-z][a-z0-9\-]*)":\s*ModelSpec\(', text))


def test_registry_has_at_least_one_production_model():
    prods = list_by_category("production")
    assert len(prods) >= 1, "At least one production model must exist"


def test_all_keys_are_kebab_case():
    for key in MODEL_REGISTRY:
        assert re.fullmatch(r"[a-z][a-z0-9\-]*", key), f"Bad key: {key!r}"


def test_key_matches_spec_key():
    for key, spec in MODEL_REGISTRY.items():
        assert key == spec.key, f"Dict key {key!r} != spec.key {spec.key!r}"


def test_categories_are_valid():
    valid = {"production", "baseline-legacy", "experimental", "archived"}
    for spec in MODEL_REGISTRY.values():
        assert spec.category in valid, f"{spec.key}: invalid category {spec.category!r}"


def test_get_spec_raises_on_unknown():
    with pytest.raises(KeyError, match="Unknown model key"):
        get_spec("nonexistent-model")


def test_registry_parity_with_status_doc():
    """MODEL_STATUS.md and MODEL_REGISTRY must declare the same set of keys."""
    doc_keys = _extract_status_keys()
    reg_keys = set(MODEL_REGISTRY.keys())
    # Allow doc-only placeholders like "contrastive": ModelSpec(...)
    missing_in_reg = doc_keys - reg_keys
    missing_in_doc = reg_keys - doc_keys
    assert not missing_in_reg, f"In MODEL_STATUS.md but not in registry: {missing_in_reg}"
    assert not missing_in_doc, f"In registry but not in MODEL_STATUS.md: {missing_in_doc}"


def test_production_models_have_artifact_dir():
    for spec in list_by_category("production"):
        assert spec.artifact_dir, f"{spec.key}: production model must define artifact_dir"
