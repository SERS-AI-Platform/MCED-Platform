from __future__ import annotations

from sers.master_data._compat import StrEnum


class ExampleState(StrEnum):
    READY = "ready"


def test_str_enum_preserves_string_runtime_contract_on_python_310() -> None:
    # Given: a string enum implemented without Python 3.11-only enum.StrEnum.
    state = ExampleState.READY

    # When: callers use it through the established string interfaces.
    rendered = str(state)
    looked_up = ExampleState("ready")

    # Then: serialization and value lookup match enum.StrEnum behavior.
    assert rendered == "ready"
    assert looked_up is state
    assert isinstance(state, str)
