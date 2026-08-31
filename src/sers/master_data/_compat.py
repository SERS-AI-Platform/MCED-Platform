from __future__ import annotations

from enum import Enum
from typing import NoReturn


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return str.__str__(self)

    def __format__(self, format_spec: str) -> str:
        return str.__format__(self, format_spec)


def assert_never(value: NoReturn) -> NoReturn:
    raise AssertionError(f"Expected code to be unreachable, but got: {value!r}")
