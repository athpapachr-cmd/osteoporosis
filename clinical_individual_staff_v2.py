"""Individually attributable staff sessions for the *future* clinical follow-through.

V2 is feature-OFF unless COCKPIT_INDIVIDUAL_STAFF_V2_ENABLED=1. No provider,
patient, appointment or medical-record reads or writes. No current Pending or
Visit Brief endpoint trusts this new token without a separately reviewed adapter.

This is intentionally a separate credential namespace from the legacy shared
Clinical Key and the role-wide clinician/secretary v1 staff credentials.

Configuration is private server environment, never source-controlled PHI:
COCKPIT_INDIVIDUAL_STAFF_V2_ACTORS_JSON=[{
  "principal_id":"example_clinician_01", "role":"clinician",
  "credential_sha256":"<sha256 of 256-bit random URL-safe API token>",
  "revision":"v1", "active":true
}]
COCKPIT_INDIVIDUAL_STAFF_V2_SESSION_SECRET=<independent 32+ char random secret>

The sha256 above is suitable ONLY for high-entropy generated tokens, never for
human-memorable passwords. Provisioning tokens is a separate approved action.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import time
from dataclasses import dataclass

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from clinical_staff_auth import STAFF_COOKIE, StaffActor

COOKIE_V2 = "cockpit_individual_staff_v2"
LIFETIME_SECONDS = 3600
_ID = re.compile(r"^[a-z][a-z0-9_-]{3,63}$")
_REV = re.compile(r"^[A-Za-z0-9_-]{1,24}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"^[A-Za-z0-9_-]{43,128}$")
_ROLES = frozenset(("clinician", "secretary"))
_MAX_ACTORS = 32
_MAX_JSON_CHARS = 16384


@dataclass(frozen=True)
class IndividualPrincipal:
    principal_id: str
    role: str
    credential_sha256: str
    revision: str
    active: bool

    @property
    def session_revision(self) -> str:
        # Rotating the individual credential invalidates all prior sessions
        # even when an administrator does not bump the explicit revision.
        return hashlib.sha256(
            (self.revision + ":" + self.credential_sha256).encode("ascii")
        ).hexdigest()[:24]


def _configured() -> tuple[dict[str, IndividualPrincipal], str]:
    if os.getenv("COCKPIT_INDIVIDUAL_STAFF_V2_ENABLED", "") != "1":
        raise HTTPException(503, "Individual staff identity is disabled")
    secret = os.getenv("COCKPIT_INDIVIDUAL_STAFF_V2_SESSION_SECRET", "")
    raw = os.getenv("COCKPIT_INDIVIDUAL_STAFF_V2_ACTORS_JSON", "")
    if len(secret) < 32 or len(raw) < 2 or len(raw) > _MAX_JSON_CHARS:
        raise HTTPException(503, "Individual staff identity not configured")
    # Credentials/session secret must not be identical to any legacy shared key.
    for legacy in ("CLINICAL_DATA_KEY", "COCKPIT_STAFF_CLINICIAN_KEY",
                   "COCKPIT_STAFF_SECRETARY_KEY", "COCKPIT_STAFF_SESSION_SECRET"):
        shared = os.getenv(legacy, "")
        if shared and hmac.compare_digest(secret, shared):
            raise HTTPException(503, "Individual staff signing secret not independent")
    try:
        records = json.loads(raw)
    except (ValueError, TypeError):
        raise HTTPException(503, "Invalid individual staff registry") from None
    if not isinstance(records, list) or not 1 <= len(records) <= _MAX_ACTORS:
        raise HTTPException(503, "Invalid individual staff registry")
    allowed = {"principal_id", "role", "credential_sha256", "revision", "active"}
    actors: dict[str, IndividualPrincipal] = {}
    credential_digests: set[str] = set()
    legacy_digests = {
        hashlib.sha256(v.encode("utf-8")).hexdigest()
        for v in (
            os.getenv("CLINICAL_DATA_KEY", ""),
            os.getenv("COCKPIT_STAFF_CLINICIAN_KEY", ""),
            os.getenv("COCKPIT_STAFF_SECRETARY_KEY", ""),
        ) if v
    }
    for record in records:
        if not isinstance(record, dict) or set(record) != allowed:
            raise HTTPException(503, "Invalid individual staff principal")
        principal_id, role = record["principal_id"], record["role"]
        digest, revision, active = (
            record["credential_sha256"], record["revision"], record["active"]
        )
        if (not isinstance(principal_id, str) or not _ID.fullmatch(principal_id)
                or role not in _ROLES or not isinstance(digest, str)
                or not _HEX64.fullmatch(digest)
                or not isinstance(revision, str) or not _REV.fullmatch(revision)
                or type(active) is not bool
                or principal_id in actors or digest in credential_digests
                or digest in legacy_digests):
            raise HTTPException(503, "Invalid or non-independent principal configuration")
        actors[principal_id] = IndividualPrincipal(
            principal_id, role, digest, revision, active
        )
        credential_digests.add(digest)
    return actors, secret


def _https(request: Request) -> None:
    # Direct HTTPS is mandatory. Reverse proxies must be configured by the
    # infrastructure to expose the trusted external HTTPS scheme to ASGI.
    if request.url.scheme != "https":
        raise HTTPException(403, "Secure staff connection required")


def _find_credential(token: str) -> IndividualPrincipal | None:
    actors, _ = _configured()
    if not isinstance(token, str) or not _TOKEN.fullmatch(token):
        return None
    digest = hashlib.sha256(token.encode("ascii")).hexdigest()
    winner = None
    # Do not branch on a role before comparing every provisioned token hash.
    for principal in actors.values():
        matched = hmac.compare_digest(digest, principal.credential_sha256)
        if matched and principal.active:
            winner = principal
    return winner


def _issue(principal: IndividualPrincipal) -> str:
    actors, secret = _configured()
    actual = actors.get(principal.principal_id)
    if actual != principal or not principal.active:
        raise HTTPException(401, "Individual staff session unavailable")
    issued = str(int(time.time()))
    nonce = secrets.token_hex(16)
    body = ".".join(("v2", principal.principal_id, principal.role,
                     principal.session_revision, issued, nonce))
    mac = hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256)
    return body + "." + mac.hexdigest()


def require_individual_staff(request: Request) -> StaffActor:
    """No legacy-cookie, role-wide key, header or client-chosen role fallback."""
    _https(request)
    actors, secret = _configured()
    value = request.cookies.get(COOKIE_V2, "")
    parts = value.split(".")
    if len(parts) != 7 or parts[0] != "v2":
        raise HTTPException(401, "Individual staff sign-in required")
    _, principal_id, role, revision, issued, nonce, signature = parts
    principal = actors.get(principal_id)
    if (principal is None or not principal.active or role != principal.role
            or revision != principal.session_revision or len(nonce) != 32
            or not re.fullmatch(r"[0-9a-f]{32}", nonce)
            or not _HEX64.fullmatch(signature)):
        raise HTTPException(401, "Individual staff session expired")
    try:
        issued_at = int(issued)
    except (ValueError, TypeError):
        raise HTTPException(401, "Invalid individual staff session") from None
    now = int(time.time())
    if (issued != str(issued_at) or issued_at > now + 15
            or now - issued_at >= LIFETIME_SECONDS):
        raise HTTPException(401, "Individual staff session expired")
    body = ".".join(parts[:-1])
    correct = hmac.new(
        secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(signature, correct):
        raise HTTPException(401, "Invalid individual staff session")
    return StaffActor(principal_id=principal.principal_id, role=principal.role)


def require_individual_clinician(request: Request) -> StaffActor:
    actor = require_individual_staff(request)
    if actor.role != "clinician":
        raise HTTPException(403, "Clinician identity required")
    return actor


def require_individual_secretary(request: Request) -> StaffActor:
    actor = require_individual_staff(request)
    if actor.role != "secretary":
        raise HTTPException(403, "Secretary identity required")
    return actor


class IndividualLogin(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str = Field(min_length=43, max_length=128)


def build_individual_staff_v2_router() -> APIRouter:
    """Mounted under /clinical but ALL actions fail-closed while feature OFF."""
    router = APIRouter(prefix="/individual-staff-v2", tags=["individual-staff-v2"])

    @router.post("/login")
    def login(body: IndividualLogin, request: Request, response: Response):
        _https(request)
        found = _find_credential(body.key)
        if found is None:
            raise HTTPException(401, "Invalid individual staff credential")
        response.delete_cookie("clinical_session", path="/")
        response.delete_cookie(STAFF_COOKIE, path="/clinical")
        response.set_cookie(
            COOKIE_V2, _issue(found),
            path="/clinical", httponly=True, secure=True,
            samesite="strict", max_age=LIFETIME_SECONDS,
        )
        return {"authenticated": True, "principal_id": found.principal_id,
                "staff_role": found.role}

    @router.get("/me")
    def me(actor: StaffActor = Depends(require_individual_staff)):
        return {"principal_id": actor.principal_id, "staff_role": actor.role}

    @router.post("/logout")
    def logout(request: Request, response: Response):
        _https(request)
        response.delete_cookie(COOKIE_V2, path="/clinical")
        return {"authenticated": False}

    return router
