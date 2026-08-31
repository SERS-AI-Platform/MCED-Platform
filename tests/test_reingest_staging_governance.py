from __future__ import annotations

import importlib
import runpy
import sys
from types import ModuleType

import pytest


class UnsafeExecutionReached(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def _load_sers_before_pandas_is_replaced() -> None:
    importlib.import_module("sers")


def test_reingest_staging_refuses_row_order_linkage_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: pandas is replaced so reaching legacy data reads is observable and harmless.
    pandas_stub = ModuleType("pandas")

    def refuse_read(*_args: str, **_kwargs: str) -> None:
        raise UnsafeExecutionReached

    setattr(pandas_stub, "read_csv", refuse_read)
    setattr(pandas_stub, "read_excel", refuse_read)
    monkeypatch.setitem(sys.modules, "pandas", pandas_stub)

    # When: the predecessor module is invoked without its explicit legacy override.
    with pytest.raises(RuntimeError) as caught:
        runpy.run_module("sers.reingest_staging", run_name="__main__")

    # Then: governance rejects execution before any row-order data read.
    assert type(caught.value).__name__ == "UngovernedRowOrderLinkageError"
    assert not isinstance(caught.value, UnsafeExecutionReached)


def test_reingest_staging_requires_explicit_historical_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the unmistakable legacy override and a harmless observable read boundary.
    pandas_stub = ModuleType("pandas")

    def refuse_read(*_args: str, **_kwargs: str) -> None:
        raise UnsafeExecutionReached

    setattr(pandas_stub, "read_csv", refuse_read)
    setattr(pandas_stub, "read_excel", refuse_read)
    monkeypatch.setitem(sys.modules, "pandas", pandas_stub)
    monkeypatch.setattr(
        sys,
        "argv",
        ["reingest_staging", "--allow-legacy-row-order-linkage"],
    )

    # When/Then: the explicit historical override alone reaches legacy reads.
    with pytest.raises(UnsafeExecutionReached):
        runpy.run_module("sers.reingest_staging", run_name="__main__")
