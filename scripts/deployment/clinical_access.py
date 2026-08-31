"""Shared account ownership checks for clinical session routes."""

from __future__ import annotations

from collections.abc import Mapping

from . import clinical_db as db


def get_authorized_session(user: Mapping[str, str | int], session_id: str) -> dict | None:
    return db.get_session_for_user(
        session_id,
        user_id=int(user["user_id"]),
        is_admin=user["role"] == "admin",
    )
