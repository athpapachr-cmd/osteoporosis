from __future__ import annotations

import hashlib
import os
import secrets
from typing import Optional

from fastapi import APIRouter, HTTPException, Response
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
            if expected and cookie and secrets.compare_digest(cookie, _session_token(expected)):
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
        if not secrets.compare_digest(req.key, expected):
            raise HTTPException(status_code=401, detail="Invalid clinical data key")
        response.set_cookie(
            key=COOKIE_NAME,
            value=_session_token(expected),
            httponly=True,
            secure=True,
            samesite="strict",
            max_age=60 * 60 * 12,
            path="/",
        )
        return {"authenticated": True}

    @router.post("/logout")
    def logout(response: Response):
        response.delete_cookie(COOKIE_NAME, path="/")
        return {"authenticated": False}

    return router
