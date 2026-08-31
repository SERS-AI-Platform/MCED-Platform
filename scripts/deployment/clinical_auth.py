"""
SERS Clinical Webapp - Authentication & Session Management
SHA-256 기반 간이 인증 (MFDS 초기 제출용)
"""

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from typing import Optional

from starlette.requests import Request
from starlette.responses import RedirectResponse

from . import clinical_db as db

# In-memory session store (서버 재시작 시 초기화됨)
_sessions: dict[str, dict] = {}

SESSION_COOKIE = "sers_session"
SESSION_EXPIRY_MINUTES = 30
PBKDF2_ITERATIONS = 240_000
PBKDF2_PREFIX = "pbkdf2_sha256"


def hash_password(password: str) -> str:
    """Return a salted password hash while retaining legacy-read compatibility."""
    salt_hex = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt_hex.encode("ascii"),
        PBKDF2_ITERATIONS,
    ).hex()
    return f"{PBKDF2_PREFIX}${PBKDF2_ITERATIONS}${salt_hex}${digest}"


def verify_password(password: str, password_hash: str) -> bool:
    """Verify current PBKDF2 hashes and historical unsalted SHA-256 hashes."""
    if password_hash.startswith(f"{PBKDF2_PREFIX}$"):
        try:
            prefix, iterations_text, salt_hex, expected = password_hash.split("$", 3)
            if prefix != PBKDF2_PREFIX:
                return False
            iterations = int(iterations_text)
            if iterations <= 0:
                return False
            actual = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode(),
                salt_hex.encode("ascii"),
                iterations,
            ).hex()
        except (TypeError, ValueError):
            return False
        return hmac.compare_digest(actual, expected)

    if len(password_hash) != 64:
        return False
    legacy = hashlib.sha256(password.encode()).hexdigest()
    return hmac.compare_digest(legacy, password_hash)


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
        "created_at": datetime.now(),
        "last_activity": datetime.now(),
    }

    db.log_audit("login", user_id=user["id"], detail={"username": username})
    return token


def logout(token: str):
    """Invalidate session."""
    session = _sessions.pop(token, None)
    if session:
        db.log_audit("logout", user_id=session["user_id"])


def invalidate_user_sessions(user_id: int) -> None:
    """Revoke every active session owned by one user."""
    tokens = [
        token for token, session in _sessions.items() if session["user_id"] == user_id
    ]
    for token in tokens:
        _sessions.pop(token, None)


def get_current_user(request: Request) -> Optional[dict]:
    """Get authenticated user from request cookie."""
    token = request.cookies.get(SESSION_COOKIE)
    if not token or token not in _sessions:
        return None

    session = _sessions[token]

    # Check expiry
    if datetime.now() - session.get("last_activity", session["created_at"]) > timedelta(minutes=SESSION_EXPIRY_MINUTES):
        _sessions.pop(token, None)
        return None
    session["last_activity"] = datetime.now()

    return session


def get_session_token(request: Request) -> Optional[str]:
    return request.cookies.get(SESSION_COOKIE)


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
        return RedirectResponse("/patient/new", status_code=303)
    return None
