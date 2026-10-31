"""Offline, entirely synthetic attributable clinician/secretary auth V2 tests.

No real user names, credentials, patient data, third-party provider or clinical
database. Existing legacy login behavior is explicitly regression-tested.
"""
from __future__ import annotations

import hashlib
import json
import secrets

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from clinical_auth import ClinicalCookieMiddleware, build_auth_router
from clinical_individual_staff_v2 import (
    COOKIE_V2, require_individual_staff, require_individual_clinician,
    require_individual_secretary,
)
from clinical_staff_auth import STAFF_COOKIE, StaffActor, issue_staff_cookie

DOCTOR_ID = "synthetic_doctor_01"
SECRETARY_ID = "synthetic_secretary_01"
CLINICAL_KEY = "synthetic-legacy-clinical-key-" + ("L" * 40)
SESSION_SECRET = "synthetic-personal-session-secret-" + ("S" * 44)


def actor_record(principal_id: str, role: str, token: str,
                 revision: str = "v1", active: bool = True):
    return {
        "principal_id": principal_id,
        "role": role,
        "credential_sha256": hashlib.sha256(token.encode("ascii")).hexdigest(),
        "revision": revision, "active": active,
    }


def app_factory():
    app = FastAPI()
    app.add_middleware(ClinicalCookieMiddleware)
    app.include_router(build_auth_router())

    @app.get("/clinical/v2-test-clinician")
    def clinician(actor: StaffActor = Depends(require_individual_clinician)):
        return {"role": actor.role, "principal": actor.principal_id}

    @app.get("/clinical/v2-test-secretary")
    def secretary(actor: StaffActor = Depends(require_individual_secretary)):
        return {"role": actor.role, "principal": actor.principal_id}

    @app.get("/clinical/v2-test-any")
    def either(actor: StaffActor = Depends(require_individual_staff)):
        return {"role": actor.role, "principal": actor.principal_id}

    @app.get("/clinical/v2-test-legacy-header")
    def legacy_injection(request: Request):
        return {"has_broad_clinical_key": bool(
            request.headers.get("x-clinical-key")
        )}

    return app


@pytest.fixture
def setup(monkeypatch):
    clinician_token = secrets.token_urlsafe(32)
    secretary_token = secrets.token_urlsafe(32)
    actors = [
        actor_record(DOCTOR_ID, "clinician", clinician_token),
        actor_record(SECRETARY_ID, "secretary", secretary_token),
    ]
    monkeypatch.setenv("COCKPIT_INDIVIDUAL_STAFF_V2_ENABLED", "1")
    monkeypatch.setenv("COCKPIT_INDIVIDUAL_STAFF_V2_SESSION_SECRET", SESSION_SECRET)
    monkeypatch.setenv("COCKPIT_INDIVIDUAL_STAFF_V2_ACTORS_JSON", json.dumps(actors))
    monkeypatch.setenv("CLINICAL_DATA_KEY", CLINICAL_KEY)
    monkeypatch.delenv("COCKPIT_STAFF_AUTH_ENABLED", raising=False)
    app = app_factory()
    return app, clinician_token, secretary_token, actors, monkeypatch


def login(c, credential):
    return c.post("/clinical/individual-staff-v2/login", json={"key": credential})


def test_individual_actor_login_and_role_boundaries(setup):
    app, doctor, secretary, _, _ = setup
    clinician = TestClient(app, base_url="https://testserver")
    staff = TestClient(app, base_url="https://testserver")

    assert clinician.get("/clinical/individual-staff-v2/me").status_code == 401
    result = login(clinician, doctor)
    assert result.status_code == 200, result.text
    assert result.json() == {
        "authenticated": True, "principal_id": DOCTOR_ID, "staff_role": "clinician"
    }
    cookie = result.headers.get("set-cookie", "")
    assert COOKIE_V2 in cookie and "httponly" in cookie.lower()
    assert "secure" in cookie.lower() and "samesite=strict" in cookie.lower()
    assert "path=/clinical" in cookie.lower()
    assert clinician.get("/clinical/individual-staff-v2/me").json() == {
        "principal_id": DOCTOR_ID, "staff_role": "clinician"
    }
    assert clinician.get("/clinical/v2-test-clinician").status_code == 200
    assert clinician.get("/clinical/v2-test-secretary").status_code == 403
    assert clinician.get("/clinical/v2-test-any").json()["principal"] == DOCTOR_ID

    assert login(staff, secretary).status_code == 200
    assert staff.get("/clinical/v2-test-secretary").status_code == 200
    assert staff.get("/clinical/v2-test-clinician").status_code == 403
    assert staff.get("/clinical/v2-test-any").json()["principal"] == SECRETARY_ID
    assert clinician.get("/clinical/v2-test-any").json()["principal"] == DOCTOR_ID


def test_individual_login_never_issues_broad_clinical_access(setup):
    app, doctor, secretary, _, _ = setup
    c = TestClient(app, base_url="https://testserver")
    assert login(c, doctor).status_code == 200
    assert not c.get("/clinical/v2-test-legacy-header").json()["has_broad_clinical_key"]
    # Legacy role-wide staff session is not silently created.
    assert c.get("/clinical/staff-session").status_code == 503
    assert not c.cookies.get(STAFF_COOKIE)
    assert c.get("/clinical/v2-test-clinician").status_code == 200
    d = TestClient(app, base_url="https://testserver")
    assert login(d, secretary).status_code == 200
    assert not d.get("/clinical/v2-test-legacy-header").json()["has_broad_clinical_key"]


def test_existing_shared_clinical_login_does_not_count_as_attributable(setup):
    app, doctor, secretary, _, _ = setup
    c = TestClient(app, base_url="https://testserver")
    legacy = c.post("/clinical/login", json={"key": CLINICAL_KEY})
    assert legacy.status_code == 200
    assert c.get("/clinical/v2-test-legacy-header").json()["has_broad_clinical_key"]
    assert c.get("/clinical/v2-test-clinician").status_code == 401
    # If a V2 cookie is present but invalid, middleware cannot fall back to
    # the previously issued broad clinical_session cookie.
    c.cookies.set(COOKIE_V2, "v2.invalid", path="/clinical")
    assert not c.get("/clinical/v2-test-legacy-header").json()["has_broad_clinical_key"]
    assert c.get("/clinical/v2-test-clinician").status_code == 401
    assert login(c, doctor).status_code == 200
    assert not c.get("/clinical/v2-test-legacy-header").json()["has_broad_clinical_key"]
    assert c.get("/clinical/v2-test-clinician").status_code == 200


def test_old_staff_cookie_cannot_claim_individual_identity(setup):
    app, doctor, secretary, _, monkeypatch = setup
    monkeypatch.setenv("COCKPIT_STAFF_AUTH_ENABLED", "1")
    monkeypatch.setenv("COCKPIT_STAFF_CLINICIAN_KEY", "synthetic-v1-doctor-" + "A" * 42)
    monkeypatch.setenv("COCKPIT_STAFF_SECRETARY_KEY", "synthetic-v1-secretary-" + "B" * 42)
    monkeypatch.setenv("COCKPIT_STAFF_SESSION_SECRET", "synthetic-v1-signing-" + "X" * 42)
    c = TestClient(app, base_url="https://testserver")
    c.cookies.set(
        STAFF_COOKIE, issue_staff_cookie(StaffActor("clinic-clinician", "clinician")),
        path="/clinical",
    )
    assert c.get("/clinical/v2-test-clinician").status_code == 401
    assert c.get("/clinical/staff-session").json()["principal_id"] == "clinic-clinician"
    assert login(c, doctor).status_code == 200
    assert c.get("/clinical/staff-session").status_code == 401
    assert c.get("/clinical/v2-test-clinician").json()["principal"] == DOCTOR_ID


def test_wrong_key_and_legacy_keys_cannot_become_v2_tokens(setup):
    app, doctor, secretary, _, _ = setup
    c = TestClient(app, base_url="https://testserver")
    assert login(c, secrets.token_urlsafe(32)).status_code == 401
    assert login(c, CLINICAL_KEY).status_code == 401
    assert c.get("/clinical/individual-staff-v2/me").status_code == 401
    assert not c.cookies.get(COOKIE_V2)


def test_session_revoked_on_account_disable_or_registry_removal(setup):
    app, doctor, secretary, actors, monkeypatch = setup
    c = TestClient(app, base_url="https://testserver")
    assert login(c, doctor).status_code == 200
    assert c.get("/clinical/v2-test-clinician").status_code == 200
    disabled = [dict(a) for a in actors]
    disabled[0]["active"] = False
    monkeypatch.setenv("COCKPIT_INDIVIDUAL_STAFF_V2_ACTORS_JSON", json.dumps(disabled))
    assert c.get("/clinical/v2-test-clinician").status_code == 401
    # A disabled account cannot log in again.
    assert login(TestClient(app, base_url="https://testserver"), doctor).status_code == 401
    monkeypatch.setenv(
        "COCKPIT_INDIVIDUAL_STAFF_V2_ACTORS_JSON", json.dumps([actors[1]])
    )
    assert c.get("/clinical/v2-test-clinician").status_code == 401


def test_credential_and_version_rotation_invalidate_old_cookie(setup):
    app, doctor, secretary, actors, monkeypatch = setup
    c = TestClient(app, base_url="https://testserver")
    assert login(c, doctor).status_code == 200
    new_actor = [dict(a) for a in actors]
    new_actor[0]["revision"] = "v2"
    monkeypatch.setenv("COCKPIT_INDIVIDUAL_STAFF_V2_ACTORS_JSON", json.dumps(new_actor))
    assert c.get("/clinical/v2-test-clinician").status_code == 401
    assert login(c, doctor).status_code == 200
    assert c.get("/clinical/v2-test-clinician").status_code == 200
    rotated_token = secrets.token_urlsafe(32)
    new_actor[0]["credential_sha256"] = hashlib.sha256(
        rotated_token.encode("ascii")
    ).hexdigest()
    monkeypatch.setenv("COCKPIT_INDIVIDUAL_STAFF_V2_ACTORS_JSON", json.dumps(new_actor))
    assert c.get("/clinical/v2-test-clinician").status_code == 401
    assert login(c, doctor).status_code == 401
    assert login(c, rotated_token).status_code == 200
    assert c.get("/clinical/v2-test-clinician").status_code == 200


def test_cookie_tampering_and_role_swapping_are_denied(setup):
    app, doctor, secretary, _, _ = setup
    c = TestClient(app, base_url="https://testserver")
    assert login(c, doctor).status_code == 200
    authentic = c.cookies.get(COOKIE_V2)
    assert authentic and authentic.startswith("v2.")
    tokens = (
        authentic.replace(".clinician.", ".secretary."),
        authentic[:-1] + ("0" if authentic[-1] != "0" else "1"),
        "v2." + "synthetic_fake_doctor" + "." + ".".join(authentic.split(".")[2:]),
        "v2.malformed",
        "v3." + authentic.split(".", 1)[1],
    )
    for forged in tokens:
        c.cookies.set(COOKIE_V2, forged, path="/clinical")
        assert c.get("/clinical/v2-test-any").status_code == 401
    c.cookies.set(COOKIE_V2, authentic, path="/clinical")
    assert c.get("/clinical/v2-test-clinician").status_code == 200


def test_session_expiry_and_clock_skew_fail_closed(setup, monkeypatch):
    import clinical_individual_staff_v2 as v2
    app, doctor, secretary, _, _ = setup
    c = TestClient(app, base_url="https://testserver")
    clock = [1782950000]
    monkeypatch.setattr(v2.time, "time", lambda: clock[0])
    assert login(c, doctor).status_code == 200
    assert c.get("/clinical/v2-test-clinician").status_code == 200
    clock[0] += 3600
    assert c.get("/clinical/v2-test-clinician").status_code == 401
    clock[0] -= 3600 + 17
    assert c.get("/clinical/v2-test-clinician").status_code == 401


def test_disabled_flag_and_http_never_allow_session(setup):
    app, doctor, secretary, _, monkeypatch = setup
    c = TestClient(app, base_url="http://testserver")
    assert login(c, doctor).status_code == 403
    assert c.get("/clinical/v2-test-clinician").status_code == 403
    good = TestClient(app, base_url="https://testserver")
    assert login(good, doctor).status_code == 200
    monkeypatch.setenv("COCKPIT_INDIVIDUAL_STAFF_V2_ENABLED", "0")
    assert good.get("/clinical/v2-test-clinician").status_code == 503
    assert login(good, doctor).status_code == 503


@pytest.mark.parametrize("broken", [
    {},
    {"not": "a list"},
    [{"principal_id": "x"}],
    [{"principal_id": ["unhashable"], "role": "clinician",
      "credential_sha256": "0" * 64, "revision": "v1", "active": True}],
    [{"principal_id": "synthetic_test01", "role": ["malformed"],
      "credential_sha256": "0" * 64, "revision": "v1", "active": True}],
])
def test_malformed_registry_fails_closed_without_runtime_500(setup, broken):
    app, doctor, secretary, _, monkeypatch = setup
    monkeypatch.setenv("COCKPIT_INDIVIDUAL_STAFF_V2_ACTORS_JSON", json.dumps(broken))
    c = TestClient(app, base_url="https://testserver")
    assert login(c, doctor).status_code == 503


def test_duplicate_actor_and_shared_legacy_credential_are_forbidden(setup):
    app, doctor, secretary, actors, monkeypatch = setup
    duplicated = [dict(actors[0]), dict(actors[0])]
    monkeypatch.setenv("COCKPIT_INDIVIDUAL_STAFF_V2_ACTORS_JSON", json.dumps(duplicated))
    c = TestClient(app, base_url="https://testserver")
    assert login(c, doctor).status_code == 503
    insecure = [dict(a) for a in actors]
    insecure[0]["credential_sha256"] = hashlib.sha256(
        CLINICAL_KEY.encode("utf-8")
    ).hexdigest()
    monkeypatch.setenv("COCKPIT_INDIVIDUAL_STAFF_V2_ACTORS_JSON", json.dumps(insecure))
    assert login(c, doctor).status_code == 503


def test_logout_clears_only_individual_session_and_existing_logout_also_clears(setup):
    app, doctor, secretary, _, _ = setup
    c = TestClient(app, base_url="https://testserver")
    assert login(c, doctor).status_code == 200
    assert c.post("/clinical/individual-staff-v2/logout").status_code == 200
    assert c.get("/clinical/v2-test-clinician").status_code == 401
    assert login(c, doctor).status_code == 200
    assert c.post("/clinical/logout").status_code == 200
    assert c.get("/clinical/v2-test-clinician").status_code == 401


def test_v2_is_disabled_by_default_without_environment_opt_in(setup):
    app, doctor, secretary, _, monkeypatch = setup
    monkeypatch.delenv("COCKPIT_INDIVIDUAL_STAFF_V2_ENABLED", raising=False)
    client = TestClient(app, base_url="https://testserver")
    assert login(client, doctor).status_code == 503
    assert client.get("/clinical/v2-test-any").status_code == 503
    # Existing shared Clinical Key login remains unchanged when V2 is OFF.
    assert client.post("/clinical/login", json={"key": CLINICAL_KEY}).status_code == 200
    assert client.get("/clinical/v2-test-legacy-header").json()["has_broad_clinical_key"]
