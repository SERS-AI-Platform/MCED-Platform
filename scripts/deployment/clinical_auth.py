"""
SERS Clinical Webapp - Authentication & Session Management
SHA-256 기반 간이 인증 (MFDS 초기 제출용)
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional

from starlette.requests import Request
from starlette.responses import RedirectResponse

from . import clinical_db as db

# In-memory session store (서버 재시작 시 초기화됨)
_sessions: dict[str, dict] = {}

SESSION_COOKIE = "sers_session"
SESSION_EXPIRY_HOURS = 8  # 근무 시간 기준


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    return hash_password(password) == password_hash


def login(username: str, password: str) -> Optional[str]:
    """Authenticate user and return session token, or None if failed."""
    user = db.get_user_by_username(username)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None

    token = secrets.token_urlsafe(32)
    _sessions[token] = {
        "user_id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "display_name": user["display_name"],
        "operating_mode": None,  # 로그인 후 선택
        "created_at": datetime.now(),
    }

    db.log_audit("login", user_id=user["id"], detail={"username": username})
    return token


def logout(token: str):
    """Invalidate session."""
    session = _sessions.pop(token, None)
    if session:
        db.log_audit("logout", user_id=session["user_id"])


def get_current_user(request: Request) -> Optional[dict]:
    """Get authenticated user from request cookie."""
    token = request.cookies.get(SESSION_COOKIE)
    if not token or token not in _sessions:
        return None

    session = _sessions[token]

    # Check expiry
    if datetime.now() - session["created_at"] > timedelta(hours=SESSION_EXPIRY_HOURS):
        _sessions.pop(token, None)
        return None

    return session


def get_session_token(request: Request) -> Optional[str]:
    return request.cookies.get(SESSION_COOKIE)


def set_operating_mode(token: str, mode: str):
    """Set operating mode for the current session."""
    if token in _sessions:
        old_mode = _sessions[token].get("operating_mode")
        _sessions[token]["operating_mode"] = mode
        db.log_audit(
            "mode_change",
            user_id=_sessions[token]["user_id"],
            detail={"from": old_mode, "to": mode},
        )


def get_operating_mode(request: Request) -> Optional[str]:
    token = request.cookies.get(SESSION_COOKIE)
    if token and token in _sessions:
        return _sessions[token].get("operating_mode")
    return None


def require_auth(request: Request) -> Optional[RedirectResponse]:
    """Returns RedirectResponse to login if not authenticated, else None."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    return None


def require_role(request: Request, roles: list[str]) -> Optional[RedirectResponse]:
    """Check if user has one of the required roles."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if user["role"] not in roles:
        return RedirectResponse("/mode", status_code=303)
    return None
