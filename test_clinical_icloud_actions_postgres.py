"""Disposable PostgreSQL 17 test for the real one-doctor iCloud action schema.

All messages, patients, credentials and mailbox UIDs are synthetic. The only
permitted database is a fixed localhost CI service. No real iCloud login.
"""
from contextlib import contextmanager
from datetime import datetime
import os
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import Session

from clinical_data import ClinicalBase, PatientORM, EncounterORM, ClinicalPendingORM
from clinical_calendar import CalendarBase, ClinicalAppointmentORM
from clinical_icloud_actions import (
    MailDecisionBase, MailDecisionORM, MailDecisionEventORM, build_icloud_actions_router,
)
from clinical_icloud_mail import build_icloud_mail_router
import clinical_icloud_mail as ic
from test_clinical_icloud_mail import FakeMail, mail

pytestmark = pytest.mark.skipif(
    os.getenv("ICLOUD_PILOT_PG_TEST") != "local-postgres17",
    reason="Fixed local disposable PostgreSQL CI only",
)
PG_URL = ("postgresql+psycopg://postgres:synthetic-ci-only"
          "@127.0.0.1:5432/synthetic_icloud_actions_ci")
PATIENT = "synthetic-pg-clinician-patient"
VISIT = "synthetic-pg-signed-visit"
PENDING = "synthetic-pg-signed-pending"
ALIAS = "practice.alias@icloud.com"
DOCTOR_KEY = "synthetic-doctor-clinical-key"
NOW = datetime(2026, 10, 31)


@pytest.fixture
def pg(monkeypatch):
    engine = create_engine(
        PG_URL, connect_args={"connect_timeout": 5}, pool_size=8, max_overflow=0,
    )
    try:
        with engine.connect() as conn:
            assert conn.scalar(text("SELECT current_database()")) == "synthetic_icloud_actions_ci"
            assert conn.scalar(text("SELECT current_user")) == "postgres"
            version = int(conn.scalar(text("SHOW server_version_num")))
            assert 170000 <= version < 180000
        ClinicalBase.metadata.create_all(engine)
        CalendarBase.metadata.create_all(engine)
        MailDecisionBase.metadata.create_all(engine)
        with Session(engine) as db, db.begin():
            db.add(PatientORM(
                patient_id=PATIENT, demographics_json={
                    "full_name": "Synthetic Gamma Patient",
                    "phone": "+35799000001",
                }, created_at=NOW, updated_at=NOW,
            ))
            db.add(EncounterORM(
                id=VISIT, patient_id=PATIENT,
                encounter_date="2026-10-31", status="completed",
                payload_json={"_visit_capture_v1": {
                    "signed": True,
                    "signed_at": "2026-10-31T10:00:00+00:00",
                    "pending_ids": [PENDING],
                    "candidate": {"pending": [{"item_type": "imaging"}]},
                }}, created_at=NOW, updated_at=NOW,
            ))
            db.add(ClinicalPendingORM(
                id=PENDING, patient_id=PATIENT, source_encounter_id=VISIT,
                item_type="imaging", description="Synthetic image study",
                status="awaiting_result", responsible_role="patient",
                provenance_json=["synthetic-pg-fixture"],
                created_at=NOW, updated_at=NOW,
            ))
        monkeypatch.setenv("CLINICAL_DATA_KEY", DOCTOR_KEY)
        monkeypatch.setenv("ICLOUD_CLINICAL_READ_ENABLED", "1")
        monkeypatch.setenv("ICLOUD_CLINICAL_IMAP_USERNAME", "synthetic-mailbox")
        monkeypatch.setenv("ICLOUD_CLINICAL_APP_PASSWORD", "synthetic-mail-token")
        monkeypatch.setenv("ICLOUD_CLINICAL_ALIAS", ALIAS)
        fake = FakeMail()
        fake.rows["50"] = mail(
            subject="MRI Synthetic Gamma Patient",
            sender="Radiology <radiology@example.invalid>",
        )

        @contextmanager
        def source(_user, _password):
            yield fake, "765"

        monkeypatch.setattr(ic, "_session", source)
        app = FastAPI()
        app.include_router(build_icloud_mail_router(engine))
        app.include_router(build_icloud_actions_router(engine))
        with TestClient(app, base_url="https://testserver") as client:
            yield engine, client, fake
    finally:
        MailDecisionBase.metadata.drop_all(engine)
        CalendarBase.metadata.drop_all(engine)
        ClinicalBase.metadata.drop_all(engine)
        engine.dispose()


def post(client, kind, version, request_id=None, **extra):
    return client.post(
        "/clinical/icloud-actions/items/765-50",
        headers={"X-Clinical-Key": DOCTOR_KEY, "Origin": "https://testserver"},
        json={"action": kind, "expected_version": version,
              "request_id": request_id or str(uuid4()), **extra},
    )


def get(client, suffix=""):
    return client.get(
        "/clinical/icloud-actions/items/765-50" + suffix,
        headers={"X-Clinical-Key": DOCTOR_KEY},
    )


def test_pg_signed_visit_pending_and_event_history_survive_new_engine(pg):
    engine, client, mail_source = pg
    first = post(client, "link", 0, patient_id=PATIENT, pending_id=PENDING)
    assert first.status_code == 200, first.text
    assert post(client, "review_notify", 1).status_code == 200
    notified = post(client, "patient_notified", 2)
    assert notified.status_code == 200
    assert notified.json()["callback_complete"] is False
    completed = post(client, "callback_complete", 3)
    assert completed.status_code == 200 and completed.json()["callback_complete"]
    original = get(client).json()
    assert original["version"] == 4
    other_engine = create_engine(PG_URL, connect_args={"connect_timeout": 5})
    try:
        with Session(other_engine) as db:
            assert db.scalar(select(func.count()).select_from(MailDecisionORM)) == 1
            assert db.scalar(select(func.count()).select_from(MailDecisionEventORM)) == 4
            pending = db.get(ClinicalPendingORM, PENDING)
            assert pending.status == "awaiting_result"
            assert db.get(EncounterORM, VISIT).payload_json["_visit_capture_v1"]["signed"] is True
    finally:
        other_engine.dispose()
    assert "STORE" not in [op for op, _ in mail_source.operations]


def test_pg_revision_idempotency_and_correction_are_persistent(pg):
    engine, client, _source = pg
    request_id = str(uuid4())
    first = post(client, "classify", 0, request_id, category="administrative")
    assert first.status_code == 200 and first.json()["version"] == 1
    replay = post(client, "classify", 0, request_id, category="administrative")
    assert replay.status_code == 200 and replay.json()["version"] == 1
    assert post(client, "classify", 0, category="clinical").status_code == 409
    recovered = post(client, "classify", 1, category="clinical")
    assert recovered.status_code == 200 and recovered.json()["version"] == 2
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(MailDecisionEventORM)) == 2


def test_pg_reviewed_mail_can_be_corrected_without_false_closure(pg):
    engine, client, _source = pg
    assert post(client, "link", 0, patient_id=PATIENT, pending_id=PENDING).status_code == 200
    assert post(client, "review_notify", 1).status_code == 200
    assert post(client, "patient_notified", 2).status_code == 200
    undone = post(client, "reopen_review", 3)
    assert undone.status_code == 200
    assert undone.json()["review_status"] == "not_reviewed"
    assert undone.json()["contact_status"] == "not_requested"
    assert post(client, "callback_complete", 4).status_code == 409
    assert get(client).json()["callback_complete"] is False
