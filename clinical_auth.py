from __future__ import annotations

import hashlib
import os
import secrets

from fastapi import APIRouter, HTTPException, Response, Request
from clinical_staff_auth import (
    STAFF_COOKIE, STAFF_LIFETIME, issue_staff_cookie,
    match_staff_key, require_staff_actor, require_staff_clinician, staff_enabled,
)
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


COOKIE_NAME = "clinical_session"


def _expected_key() -> str:
    return os.environ.get("CLINICAL_DATA_KEY", "")


def _session_token(key: str) -> str:
    return hashlib.sha256(("osteoporosis-clinical-session-v1:" + key).encode("utf-8")).hexdigest()


class ClinicalLoginRequest(BaseModel):
    key: str = Field(min_length=1, max_length=512)


class ClinicalCookieMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path.startswith("/clinical/") and path not in {"/clinical/login", "/clinical/logout"}:
            expected = _expected_key()
            cookie = request.cookies.get(COOKIE_NAME, "")
            staff_cookie_present = bool(request.cookies.get(STAFF_COOKIE))
            staff_clinician = False
            if staff_cookie_present and staff_enabled():
                try:
                    require_staff_clinician(request)
                    staff_clinician = True
                except HTTPException:
                    pass
            # Staff sessions are the only authority when their cookie is present.
            # Rotation/expiry cannot fall back to a previously issued broad cookie.
            legacy_clinician = (not staff_cookie_present and bool(cookie)
                                and secrets.compare_digest(cookie, _session_token(expected)))
            if expected and (staff_clinician or legacy_clinician):
                headers = list(request.scope.get("headers", []))
                headers = [(k, v) for (k, v) in headers if k.lower() != b"x-clinical-key"]
                headers.append((b"x-clinical-key", expected.encode("utf-8")))
                request.scope["headers"] = headers
        return await call_next(request)


def build_auth_router() -> APIRouter:
    router = APIRouter(prefix="/clinical", tags=["clinical-auth"])

    @router.post("/login")
    def login(req: ClinicalLoginRequest, response: Response):
        expected = _expected_key()
        if not expected:
            raise HTTPException(status_code=503, detail="CLINICAL_DATA_KEY is not configured")
        # Staff sign-in is independent of the shared Clinical Key. Secretary
        # never receives the legacy clinician cookie or its broad privileges.
        actor = match_staff_key(req.key) if staff_enabled() else None
        if actor:
            response.set_cookie(
                key=STAFF_COOKIE, value=issue_staff_cookie(actor),
                httponly=True, secure=True, samesite="strict",
                path="/clinical", max_age=STAFF_LIFETIME,
            )
            # Staff-clinician uses this versioned session for *all* protected
            # clinical endpoints, instead of receiving an unrevocable legacy cookie.
            # The original Clinical Key remains supported for legacy-only access.
            response.delete_cookie(COOKIE_NAME, path="/")
            return {"authenticated": True, "session": "browser", "staff_role": actor.role}
        elif not secrets.compare_digest(req.key, expected):
            raise HTTPException(status_code=401, detail="Invalid clinical data key")
        else:
            # A shared Clinical Key is not an attributable staff principal.
            response.delete_cookie(STAFF_COOKIE, path="/clinical")
        response.set_cookie(
            key=COOKIE_NAME,
            value=_session_token(expected),
            httponly=True,
            secure=True,
            samesite="strict",
            path="/",
        )
        result = {"authenticated": True, "session": "browser"}
        if actor:
            result["staff_role"] = actor.role
        return result

    @router.get("/staff-session")
    def staff_session(request: Request):
        actor = require_staff_actor(request)
        return {"staff_role": actor.role, "principal_id": actor.principal_id}

    @router.post("/logout")
    def logout(response: Response):
        response.delete_cookie(COOKIE_NAME, path="/")
        response.delete_cookie(STAFF_COOKIE, path="/clinical")
        return {"authenticated": False}

    return router
