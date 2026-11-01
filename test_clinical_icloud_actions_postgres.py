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


def test_pg_missing_optional_mail_tables_preserve_existing_pending_chart(pg):
    """A missing mail feature must not abort PostgreSQL chart transactions.

    PostgreSQL fails the whole transaction after SELECT on an absent table.
    Its failure is isolated to the optional SAVEPOINT, so signed Pending,
    checkpoint history and the global obligations reader remain available.
    """
    from sqlalchemy import inspect
    from clinical_data import build_clinical_router
    from clinical_pending_one_tap import PendingEventBase

    engine, _client, _mail = pg
    second_id = "synthetic-pg-second-pending"
    PendingEventBase.metadata.create_all(engine)
    try:
        with Session(engine) as db, db.begin():
            encounter = db.get(EncounterORM, VISIT)
            payload = dict(encounter.payload_json)
            signed = dict(payload["_visit_capture_v1"])
            signed["pending_ids"] = [PENDING, second_id]
            signed["candidate"] = {
                **signed["candidate"],
                "pending": [{"item_type": "imaging"}, {"item_type": "laboratory"}],
            }
            encounter.payload_json = {**payload, "_visit_capture_v1": signed}
            db.add(ClinicalPendingORM(
                id=second_id, patient_id=PATIENT, source_encounter_id=VISIT,
                item_type="laboratory", description="Synthetic laboratory follow-up",
                status="awaiting_result", responsible_role="patient",
                provenance_json=["synthetic-pg-fixture"],
                created_at=NOW, updated_at=NOW,
            ))

        # Simulate the current Aiven deployment: iCloud decision tables are NOT
        # present while signed visits/checkpoint tables remain available.
        MailDecisionBase.metadata.drop_all(engine)
        assert not inspect(engine).has_table(MailDecisionORM.__tablename__)

        app = FastAPI()
        app.include_router(build_clinical_router(engine))
        with TestClient(app, base_url="https://testserver") as clinic:
            auth = {"X-Clinical-Key": DOCTOR_KEY}
            chart = clinic.get(f"/clinical/patient/{PATIENT}/chart", headers=auth)
            assert chart.status_code == 200, chart.text
            pending = chart.json()["pending"]
            assert {p["pending_id"] for p in pending} == {PENDING, second_id}
            assert all(p["mail_followthrough"] == [] for p in pending)
            assert all(p["checkpoint_storage_ready"] is True for p in pending)
            assert all(p["lifecycle"]["revision"] == 0 for p in pending)

            updates = clinic.get(
                "/clinical/clinical-updates?scope=real&active_only=true",
                headers=auth,
            )
            assert updates.status_code == 200, updates.text
            items = [p for p in updates.json()["items"]
                     if p["pending_id"] in {PENDING, second_id}]
            assert len(items) == 2
            assert all(p["mail_followthrough"] == [] for p in items)
            assert all(p["checkpoint_storage_ready"] is True for p in items)
    finally:
        PendingEventBase.metadata.drop_all(engine)


@pytest.mark.parametrize("only_decisions_table", [False, True])
def test_pg_readiness_never_creates_partial_mail_schema(pg, only_decisions_table):
    """The running Cockpit role checks metadata, never creates missing tables."""
    from sqlalchemy import event, inspect

    engine, _client, _mail = pg
    MailDecisionBase.metadata.drop_all(engine)
    if only_decisions_table:
        # Migration-1-only rehearsal, still in a disposable synthetic database.
        MailDecisionORM.__table__.create(engine)

    unexpected_ddl = []

    def forbid_startup_ddl(conn, cursor, statement, parameters, context, executemany):
        first = statement.lstrip().split(None, 1)[0].upper() if statement.strip() else ""
        if first in {"CREATE", "ALTER", "DROP", "TRUNCATE"}:
            unexpected_ddl.append(first)
            raise AssertionError("Application attempted DDL during iCloud readiness")

    event.listen(engine, "before_cursor_execute", forbid_startup_ddl)
    try:
        app = FastAPI()
        app.include_router(build_icloud_actions_router(engine))
        with TestClient(app, base_url="https://testserver") as client:
            denied = get(client)
            assert denied.status_code == 503
            assert denied.json() == {"detail": "Mail action storage unavailable"}
        inspector = inspect(engine)
        assert inspector.has_table(MailDecisionORM.__tablename__) is only_decisions_table
        assert not inspector.has_table(MailDecisionEventORM.__tablename__)
        assert unexpected_ddl == []
    finally:
        event.remove(engine, "before_cursor_execute", forbid_startup_ddl)


def test_pg_complete_schema_is_readable_at_startup_without_ddl(pg):
    """An already installed schema remains usable without any boot mutation."""
    from sqlalchemy import event

    engine, _client, _mail = pg
    unexpected_ddl = []

    def forbid_startup_ddl(conn, cursor, statement, parameters, context, executemany):
        first = statement.lstrip().split(None, 1)[0].upper() if statement.strip() else ""
        if first in {"CREATE", "ALTER", "DROP", "TRUNCATE"}:
            unexpected_ddl.append(first)
            raise AssertionError("Application attempted DDL during iCloud readiness")

    event.listen(engine, "before_cursor_execute", forbid_startup_ddl)
    try:
        app = FastAPI()
        app.include_router(build_icloud_actions_router(engine))
        with TestClient(app, base_url="https://testserver") as client:
            loaded = get(client)
            assert loaded.status_code == 200, loaded.text
            assert loaded.json()["review_status"] == "not_reviewed"
        assert unexpected_ddl == []
    finally:
        event.remove(engine, "before_cursor_execute", forbid_startup_ddl)
