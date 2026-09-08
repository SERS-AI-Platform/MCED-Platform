from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "publications" / "aacr" / "src"
sys.path.insert(0, str(SRC))

import stk_v2_fixed_non_ypan as aacr  # noqa: E402


def test_env_tuple_when_value_is_blank_uses_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setenv("SERS_TEST_CANCERS", " , ")
    default = ("PAN",)

    # When
    configured = aacr._env_tuple("SERS_TEST_CANCERS", default, frozenset(default))

    # Then
    assert configured == default


def test_env_tuple_when_value_is_unknown_rejects_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setenv("SERS_TEST_CANCERS", "PAN,UNKNOWN")

    # When
    def configure() -> tuple[str, ...]:
        return aacr._env_tuple("SERS_TEST_CANCERS", ("PAN",), frozenset({"PAN"}))

    # Then
    with pytest.raises(RuntimeError, match="SERS_TEST_CANCERS.*UNKNOWN"):
        configure()
