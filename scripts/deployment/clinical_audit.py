"""
SERS Clinical Webapp - Audit Trail Middleware
모든 주요 행위를 자동으로 기록
"""

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from . import clinical_auth as auth
from . import clinical_db as db

# Routes that trigger audit logging (POST actions)
AUDITED_ACTIONS = {
    ("POST", "/login"): "login_attempt",
    ("POST", "/patient/new"): "patient_create",
    ("POST", "/logout"): "logout",
}


class AuditMiddleware(BaseHTTPMiddleware):
    """Logs POST actions and slow requests to audit trail."""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = time.time() - start

        # Log slow requests (> 10s, likely analysis)
        if elapsed > 10.0:
            user = auth.get_current_user(request)
            db.log_audit(
                "slow_request",
                user_id=user["user_id"] if user else None,
                detail={"path": request.url.path, "elapsed_seconds": round(elapsed, 2)},
            )

        return response
