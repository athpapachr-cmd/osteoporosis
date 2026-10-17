"""Separated attributable staff sessions for booking-scoped Visit Brief only.

OFF by default. No shared Clinical Key, browser-supplied role, or old clinical cookie
can create a staff identity. Credential rotation/version change invalidates sessions.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass

from fastapi import HTTPException
from starlette.requests import Request

STAFF_COOKIE = "cockpit_staff_session"
STAFF_LIFETIME = 3600
ROLES = ("clinician", "secretary")


@dataclass(frozen=True)
class StaffActor:
    principal_id: str
    role: str


def staff_enabled() -> bool:
    return os.environ.get("COCKPIT_STAFF_AUTH_ENABLED", "") == "1"


def _staff_settings() -> tuple[str, str, str]:
    if not staff_enabled():
        raise HTTPException(503, "Staff access is not enabled")
    clinician = os.environ.get("COCKPIT_STAFF_CLINICIAN_KEY", "")
    secretary = os.environ.get("COCKPIT_STAFF_SECRETARY_KEY", "")
    secret = os.environ.get("COCKPIT_STAFF_SESSION_SECRET", "")
    legacy = os.environ.get("CLINICAL_DATA_KEY", "")
    if (min(len(clinician), len(secretary), len(secret)) < 32
            or clinician == secretary or clinician == legacy or secretary == legacy
            or not legacy):
        raise HTTPException(503, "Independent staff credentials are not ready")
    return clinician, secretary, secret


def match_staff_key(candidate: str) -> StaffActor | None:
    clinician, secretary, _ = _staff_settings()
    # Both comparisons are evaluated to reduce obvious role-dependent timing.
    matches = (hmac.compare_digest(candidate, clinician),
               hmac.compare_digest(candidate, secretary))
    if matches[0]:
        return StaffActor("clinic-clinician", "clinician")
    if matches[1]:
        return StaffActor("clinic-secretary", "secretary")
    return None


def _version(role: str) -> str:
    value = os.environ.get("COCKPIT_STAFF_" + role.upper() + "_VERSION", "1")
    if not value.isascii() or not value.isalnum() or len(value) > 32:
        raise HTTPException(503, "Invalid staff credential version")
    # Credential rotation invalidates an existing cookie without relying
    # on a human remembering to bump the optional version variable.
    clinician, secretary, _ = _staff_settings()
    credential = clinician if role == "clinician" else secretary
    digest = hashlib.sha256(credential.encode("utf-8")).hexdigest()[:16]
    return value + "-" + digest


def issue_staff_cookie(actor: StaffActor) -> str:
    _, _, secret = _staff_settings()
    created = str(int(time.time()))
    nonce = secrets.token_hex(16)
    body = ".".join(("v1", actor.role, _version(actor.role), created, nonce))
    mac = hmac.new(secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).hexdigest()
    return body + "." + mac


def require_staff_actor(request: Request) -> StaffActor:
    _, _, secret = _staff_settings()
    token = request.cookies.get(STAFF_COOKIE, "")
    parts = token.split(".")
    if len(parts) != 6 or parts[0] != "v1" or parts[1] not in ROLES:
        raise HTTPException(401, "Staff sign-in required")
    version, created, nonce, supplied = parts[2:]
    role = parts[1]
    if (version != _version(role) or len(nonce) != 32
            or not nonce.isascii() or len(supplied) != 64
            or not supplied.isascii()):
        raise HTTPException(401, "Staff session expired")
    try:
        ts = int(created)
    except ValueError:
        raise HTTPException(401, "Invalid staff session") from None
    now = int(time.time())
    if ts > now + 15 or now - ts >= STAFF_LIFETIME:
        raise HTTPException(401, "Staff session expired")
    body = ".".join(parts[:-1])
    expected = hmac.new(secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, supplied):
        raise HTTPException(401, "Invalid staff session")
    return StaffActor("clinic-" + role, role)


def require_staff_clinician(request: Request) -> StaffActor:
    actor = require_staff_actor(request)
    if actor.role != "clinician":
        raise HTTPException(403, "Clinician role required")
    return actor
