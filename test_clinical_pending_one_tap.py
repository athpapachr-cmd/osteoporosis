"""Bounded #162 synthetic MRI-CD one-tap, no Aiven/Render/real patient access."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from clinical_auth import ClinicalCookieMiddleware, build_auth_router
from clinical_data import (
    ClinicalPendingORM, EncounterORM, build_clinical_router,
)
from clinical_pending_one_tap import PendingCheckpointEventORM, PendingEventBase
from test_visit_capture import _candidate


CLINICIAN_KEY = "synthetic-one-tap-clinician-key-00000001"
SECRETARY_KEY = "synthetic-one-tap-secretary-key-00000001"
CLINICAL_KEY = "synthetic-one-tap-clinical-key"


def setup(monkeypatch):
    monkeypatch.setenv("CLINICAL_DATA_KEY", CLINICAL_KEY)
    monkeypatch.setenv("COCKPIT_STAFF_AUTH_ENABLED", "1")
    monkeypatch.setenv("COCKPIT_STAFF_CLINICIAN_KEY", CLINICIAN_KEY)
    monkeypatch.setenv("COCKPIT_STAFF_SECRETARY_KEY", SECRETARY_KEY)
    monkeypatch.setenv("COCKPIT_STAFF_SESSION_SECRET", "synthetic-unique-signing-secret-000000001")
    monkeypatch.setenv("CLINICAL_PENDING_ONE_TAP_SYNTHETIC_ENABLED", "1")
    engine = create_engine("sqlite+pysqlite:///:memory:",
                           connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    app = FastAPI()
    app.add_middleware(ClinicalCookieMiddleware)
    app.include_router(build_auth_router())
    app.include_router(build_clinical_router(engine))
    # Explicitly isolated DDL: production startup never creates checkpoint events.
    PendingEventBase.metadata.create_all(engine)
    client = TestClient(app, base_url="https://testserver")
    assert client.post("/clinical/login", json={"key": CLINICIAN_KEY}).status_code == 200
    return client, engine


def create_pending(client, patient="SYN-CD"):
    assert client.post("/clinical/patients", json={
        "patient_id": patient, "demographics": {"full_name": "Synthetic case"},
    }).status_code == 200
    ctx = client.post("/clinical/visit-capture/context", json={
        "patient_id": patient, "client_session_id": "synthetic-one-tap-session-00001",
    })
    assert ctx.status_code == 200, ctx.text
    candidate = deepcopy(_candidate())
    candidate["pending"] = [{
        "item_type": "material_followthrough",
        "description": "Ασθενής θα προσκομίσει CD μαγνητικής",
        "status": "awaiting_patient",
        "responsible_role": "clinician",
        "provenance": ["heidi_today"],
        "checkpoint_plan": {
            "version": 1,
            "steps": [
                {"key": "cd_received", "question": "Προσκομίστηκε το CD;",
                 "meaning": "Επιβεβαίωση πραγματικής παραλαβής"},
                {"key": "images_reviewed", "question": "Εξετάστηκαν οι εικόνες;",
                 "meaning": "Προσωπική εξέταση εικόνων από ιατρό"},
            ],
        },
    }]
    saved = client.post("/clinical/visit-capture/save", json={
        "context_id": ctx.json()["context_id"], "candidate": candidate,
    })
    assert saved.status_code == 200, saved.text
    body = saved.json()
    return patient, body["pending"][0]["pending_id"], body["encounter"]["encounter_id"]


def view(client, patient, pending):
    r = client.get("/clinical/patient/" + patient + "/pending")
    assert r.status_code == 200, r.text
    row = next(x for x in r.json() if x["pending_id"] == pending)
    return row["checkpoint"]


def tap(client, patient, pending, key, observation, revision, idem, retracted=None):
    return client.post("/clinical/patient/" + patient + "/pending/" + pending + "/checkpoint",
                       json={"checkpoint_key": key, "observation": observation,
                             "expected_revision": revision, "idempotency_key": idem,
                             "retracts_event_id": retracted})


def test_cd_lifecycle_common_projection_and_midnight(monkeypatch):
    client, engine = setup(monkeypatch)
    patient, pending, signed_id = create_pending(client)
    before = view(client, patient, pending)
    assert before["revision"] == 0
    assert before["next_checkpoint"]["state"] == "unknown"
    assert before["next_checkpoint"]["key"] == "cd_received"
    assert client.get("/clinical/daily-brief/pending").json()["items"][0] == before
    with Session(engine) as session:
        assert session.scalar(select(PendingCheckpointEventORM)) is None

    import clinical_pending_one_tap as steps
    monkeypatch.setattr(steps, "utcnow", lambda: datetime(2026, 10, 17, 23, 59))
    not_yet = tap(client, patient, pending, "cd_received", "not_yet_observed", 0,
                  "synthetic-not-yet-00001")
    assert not_yet.status_code == 200, not_yet.text
    assert not_yet.json()["current"]["next_checkpoint"]["state"] == "not_yet"
    assert not_yet.json()["current"]["completed"] is False

    monkeypatch.setattr(steps, "utcnow", lambda: datetime(2026, 10, 18, 0, 1))
    received = tap(client, patient, pending, "cd_received", "yes_confirmed", 1,
                   "synthetic-cd-received-00001")
    assert received.status_code == 200, received.text
    assert received.json()["current"]["next_checkpoint"]["key"] == "images_reviewed"
    assert received.json()["current"]["completed"] is False
    assert received.json()["current"]["status"] != "resolved"
    assert view(client, patient, pending) == client.get(
        "/clinical/daily-brief/pending").json()["items"][0]
    assert view(client, patient, pending)["source_encounter_id"] == signed_id
    with Session(engine) as session:
        encounter = session.get(EncounterORM, signed_id)
        assert encounter.payload_json["_visit_capture_v1"]["signed"] is True
        assert encounter.status == "completed"
        assert len(session.scalars(select(PendingCheckpointEventORM)).all()) == 2

    # A report read is not image inspection; only the approved image stage qualifies.
    wrong_stage = tap(client, patient, pending, "written_report_read",
                      "yes_confirmed", 2, "synthetic-wrong-report-0001")
    assert wrong_stage.status_code == 409
    reviewed = tap(client, patient, pending, "images_reviewed", "yes_confirmed", 2,
                   "synthetic-image-review-0001")
    assert reviewed.status_code == 200, reviewed.text
    assert reviewed.json()["current"]["completed"] is True
    assert reviewed.json()["current"]["status"] == "resolved"
    with Session(engine) as session:
        row = session.get(ClinicalPendingORM, pending)
        assert row.resolution_event_id == reviewed.json()["event_id"]
    assert client.get("/clinical/daily-brief/pending").json()["items"] == []

    # Compensating event, not silent overwrite; unresolved again.
    retract = tap(client, patient, pending, "images_reviewed", "retracted", 3,
                  "synthetic-retract-review-001", reviewed.json()["event_id"])
    assert retract.status_code == 200, retract.text
    assert retract.json()["current"]["status"] == "open"
    assert retract.json()["current"]["next_checkpoint"]["key"] == "images_reviewed"
    assert "images_reviewed" not in retract.json()["current"]["confirmed_keys"]
    assert view(client, patient, pending) == client.get(
        "/clinical/daily-brief/pending").json()["items"][0]
    with Session(engine) as session:
        assert len(session.scalars(select(PendingCheckpointEventORM)).all()) == 4


def test_replay_revision_patient_role_and_legacy_safety(monkeypatch):
    client, engine = setup(monkeypatch)
    patient, pending, signed_id = create_pending(client)
    a = tap(client, patient, pending, "cd_received", "yes_confirmed", 0,
            "synthetic-repeat-0000001")
    assert a.status_code == 200
    replay = tap(client, patient, pending, "cd_received", "yes_confirmed", 0,
                 "synthetic-repeat-0000001")
    assert replay.status_code == 200
    assert replay.json()["replayed"] is True
    assert replay.json()["event_id"] == a.json()["event_id"]
    assert replay.json()["event_revision"] == 1
    assert tap(client, patient, pending, "cd_received", "yes_confirmed", 0,
               "synthetic-race-00000001").status_code == 409
    assert tap(client, patient, pending, "images_reviewed", "yes_confirmed", 1,
               "synthetic-repeat-0000001").status_code == 409
    assert tap(client, "WRONG-PATIENT", pending, "images_reviewed", "yes_confirmed", 1,
               "synthetic-wrong-patient-01").status_code == 404
    assert tap(client, patient, pending, "cd_received", "retracted", 1,
               "synthetic-wrong-retract-01", "missing-event-id").status_code == 409
    with Session(engine) as session:
        assert len(session.scalars(select(PendingCheckpointEventORM)).all()) == 1

    # A shared clinical key is NOT proof of a named doctor.
    client.post("/clinical/logout")
    shared = tap(client, patient, pending, "images_reviewed", "yes_confirmed", 1,
                 "synthetic-shared-key-0001")
    assert shared.status_code == 401
    assert client.post("/clinical/login", json={"key": SECRETARY_KEY}).status_code == 200
    forbidden = tap(client, patient, pending, "images_reviewed", "yes_confirmed", 1,
                    "synthetic-secretary-00001")
    # The existing Clinical Key dependency rejects secretary sessions with 401
    # before the independently checked clinician-role dependency can return 403.
    assert forbidden.status_code in (401, 403)
    assert client.get("/clinical/daily-brief/pending").status_code in (401, 403)
    client.post("/clinical/logout")
    assert client.post("/clinical/login", json={"key": CLINICIAN_KEY}).status_code == 200

    monkeypatch.delenv("CLINICAL_PENDING_ONE_TAP_SYNTHETIC_ENABLED")
    assert tap(client, patient, pending, "images_reviewed", "yes_confirmed", 1,
               "synthetic-disabled-00001").status_code == 503
    monkeypatch.setenv("CLINICAL_PENDING_ONE_TAP_SYNTHETIC_ENABLED", "1")

    # Old Visit Capture pending with no plan survives unchanged and cannot be advanced.
    ctx = client.post("/clinical/visit-capture/context", json={
        "patient_id": patient, "client_session_id": "synthetic-legacy-visit-00001",
    }).json()
    legacy = client.post("/clinical/visit-capture/save", json={
        "context_id": ctx["context_id"], "candidate": _candidate(),
    })
    assert legacy.status_code == 200, legacy.text
    old = legacy.json()["pending"][0]
    assert old["checkpoint"] is None and old["confirmation_revision"] == 0
    assert tap(client, patient, old["pending_id"], "cd_received", "yes_confirmed", 0,
               "synthetic-no-plan-000001").status_code == 409
    assert view(client, patient, pending)["revision"] == 1


def test_untrusted_shared_key_cannot_save_checkpoint_plan(monkeypatch):
    client, engine = setup(monkeypatch)
    patient, _, _ = create_pending(client)
    client.post("/clinical/logout")
    assert client.post("/clinical/login", json={"key": CLINICAL_KEY}).status_code == 200
    ctx = client.post("/clinical/visit-capture/context", json={
        "patient_id": patient, "client_session_id": "synthetic-shared-session-0001",
    })
    assert ctx.status_code == 200
    candidate = _candidate()
    candidate["pending"][0]["status"] = "open"
    candidate["pending"][0]["checkpoint_plan"] = {
        "steps": [{"key": "receipt", "question": "Παραλαβή;", "meaning": "Παραλαβή"}]
    }
    response = client.post("/clinical/visit-capture/save", json={
        "context_id": ctx.json()["context_id"], "candidate": candidate,
    })
    assert response.status_code == 401
