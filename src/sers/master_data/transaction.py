from __future__ import annotations

import sqlite3


class ActiveCallerTransactionError(RuntimeError):
    def __init__(self, operation: str) -> None:
        self.operation = operation
        super().__init__(operation)

    def __str__(self) -> str:
        return (
            f"{self.operation} requires a connection without an active "
            "caller-owned transaction"
        )


def require_clean_connection(
    connection: sqlite3.Connection,
    operation: str,
) -> None:
    if connection.in_transaction:
        raise ActiveCallerTransactionError(operation)
