from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from clinical_office_cases import build_office_cases_router
from clinical_office_cases_models import (
    AccidentCaseAuditORM, AccidentPaymentORM, OfficeCaseBase, SurgeryDetailORM,
)
from clinical_surgery_queue import SurgeryQueueBase, SurgeryQueueORM, build_surgery_queue_router
from clinic_utilities.rf.persistence import RFApplicationORM, RFApplicationProgressORM, RFBase
from clinic_utilities.rf.api import build_rf_router


KEY = "synthetic-office-key"
HEADERS = {"X-Clinical-Key": KEY}


def _client(monkeypatch):
    monkeypatch.setenv("CLINICAL_DATA_KEY", KEY)
    engine = create_engine("sqlite+pysqlite:///:memory:",
                           connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SurgeryQueueBase.metadata.create_all(engine)
    RFBase.metadata.create_all(engine)
    OfficeCaseBase.metadata.create_all(engine)
    app = FastAPI()
    app.include_router(build_surgery_queue_router(engine))
    app.include_router(build_office_cases_router(engine))
    return TestClient(app), engine


def _case_payload(due=10000):
    return {"patient_name":"Synthetic Accident Patient", "phone":"+35799000001",
            "accident_date":"2026-10-01", "visit_date":"2026-10-03",
            "amount_due_cents":due}


def _rf_source(engine, *, stage="approved", revision=2, application_id="synthetic-rf-office-01"):
    with Session(engine) as session:
        session.add(RFApplicationORM(
            id=application_id, patient_identity_key="SYN001",
            patient_name="Synthetic RF Patient", gesy_number="SYN-GESY",
            patient_age=60, pathway="A1", indication_code="synthetic-indication",
            site_key="knee", laterality="left", exact_location="synthetic site",
            product_key="synthetic-product", payload_json={"identity_number":"SYN-001"},
            created_at=datetime(2026, 10, 1),
        ))
        session.add(RFApplicationProgressORM(
            application_id=application_id, revision=revision, stage=stage,
            submitted_date="2026-10-02", decision="approved", decision_date="2026-10-03",
            scheduled_date="2026-11-01" if stage == "scheduled" else None,
            updated_at=datetime(2026, 10, 3),
        ))
        session.commit()
    return application_id


def test_auth_and_accident_case_payment_history(monkeypatch):
    client, engine = _client(monkeypatch)
    assert client.get("/clinical/office-cases/accidents").status_code == 401
    assert client.post("/clinical/office-cases/accidents", json=_case_payload()).status_code == 401
    created = client.post("/clinical/office-cases/accidents", headers=HEADERS,
                          json=_case_payload())
    assert created.status_code == 200
    assert created.headers["cache-control"] == "no-store"
    case_id = created.json()["case_id"]
    assert created.json()["balance_cents"] == 10000
    first_id, second_id = str(uuid4()), str(uuid4())
    first = {"payment_request_id":first_id, "amount_cents":3000,
             "paid_on":"2026-10-04", "payer":"Synthetic Insurer"}
    second = {"payment_request_id":second_id, "amount_cents":2000,
              "paid_on":"2026-10-05", "payer":"Synthetic Patient"}
    path = f"/clinical/office-cases/accidents/{case_id}/payments"
    response = client.post(path, headers=HEADERS, json=first)
    assert response.status_code == 200
    assert response.json()["balance_cents"] == 7000
    replay = client.post(path, headers=HEADERS, json=first)
    assert replay.status_code == 200
    assert replay.json()["revision"] == response.json()["revision"]
    conflict = client.post(path, headers=HEADERS, json={**first, "amount_cents":4000})
    assert conflict.status_code == 409
    other = client.post(path, headers=HEADERS, json=second)
    assert other.status_code == 200
    assert other.json()["received_cents"] == 5000
    assert [p["payer"] for p in other.json()["payments"]] == ["Synthetic Insurer", "Synthetic Patient"]
    assert client.post(path, headers=HEADERS, json={**second,
        "payment_request_id":str(uuid4()), "amount_cents":6000}).status_code == 409
    with Session(engine) as session:
        assert len(session.scalars(select(AccidentPaymentORM)).all()) == 2
        assert len(session.scalars(select(AccidentCaseAuditORM)).all()) == 3


def test_accident_revision_void_and_unknown_balance(monkeypatch):
    client, _ = _client(monkeypatch)
    created = client.post("/clinical/office-cases/accidents", headers=HEADERS,
                          json=_case_payload(None)).json()
    case_id = created["case_id"]
    path = f"/clinical/office-cases/accidents/{case_id}"
    payment = client.post(path + "/payments", headers=HEADERS,
        json={"payment_request_id":str(uuid4()), "amount_cents":2500,
              "paid_on":"2026-10-05", "payer":"Synthetic Payer"}).json()
    assert payment["balance_cents"] is None
    assert client.put(path, headers=HEADERS,
        json={"expected_revision":1, "amount_due_cents":3000}).status_code == 409
    assert client.put(path, headers=HEADERS,
        json={"expected_revision":2, "amount_due_cents":2000}).status_code == 409
    edited = client.put(path, headers=HEADERS,
        json={"expected_revision":2, "amount_due_cents":5000,
              "writing_started_date":"2026-10-06"})
    assert edited.status_code == 200
    assert edited.json()["balance_cents"] == 2500
    payment_id = payment["payments"][0]["payment_id"]
    voided = client.post(path + f"/payments/{payment_id}/void", headers=HEADERS,
        json={"expected_revision":3, "reason":"Synthetic correction"})
    assert voided.status_code == 200
    assert voided.json()["received_cents"] == 0
    assert voided.json()["payments"][0]["void_reason"] == "Synthetic correction"
    assert client.post(path + f"/payments/{payment_id}/void", headers=HEADERS,
        json={"expected_revision":4, "reason":"Again"}).status_code == 409


def test_exact_rf_bridge_and_surgery_time(monkeypatch):
    client, engine = _client(monkeypatch)
    rf_id = _rf_source(engine, stage="scheduled")
    rf_list = client.get("/clinical/office-cases/rf?stage=scheduled", headers=HEADERS)
    assert rf_list.status_code == 200
    card = rf_list.json()["applications"][0]
    assert card["surgery_id"] is None
    path = f"/clinical/office-cases/surgeries/from-rf/{rf_id}"
    base = {"expected_rf_revision":2, "date_of_birth":"1966-01-01",
            "phone":"+35799000002", "procedure_type":"Synthetic RF surgery",
            "surgery_date":"2026-11-01", "room_entry_time":"09:30"}
    assert client.post(path, headers=HEADERS, json={**base,
        "expected_rf_revision":1}).status_code == 409
    assert client.post(path, headers=HEADERS, json={**base,
        "date_of_birth":None}).status_code == 422
    result = client.post(path, headers=HEADERS, json=base)
    assert result.status_code == 200
    assert client.post(path, headers=HEADERS, json=base).status_code == 409
    rows = client.get("/clinical/surgeries", headers=HEADERS).json()
    assert len(rows) == 1
    assert rows[0]["identity_number"] == "SYN-001"
    assert rows[0]["room_entry_time"] == "09:30"
    assert rows[0]["rf_application_id"] == rf_id
    assert client.put("/clinical/surgeries/" + rows[0]["surgery_id"], headers=HEADERS,
        json={"surgery_date":None}).status_code == 422
    changed = client.put("/clinical/surgeries/" + rows[0]["surgery_id"], headers=HEADERS,
        json={"surgery_date":None, "room_entry_time":None})
    assert changed.status_code == 200
    assert changed.json()["room_entry_time"] is None
    linked = client.get("/clinical/office-cases/rf?stage=scheduled", headers=HEADERS).json()
    assert linked["applications"][0]["surgery_id"] == rows[0]["surgery_id"]
    with Session(engine) as session:
        assert len(session.scalars(select(SurgeryQueueORM)).all()) == 1
        assert len(session.scalars(select(SurgeryDetailORM)).all()) == 1


def test_rf_ineligible_stage_does_not_create_surgery(monkeypatch):
    client, engine = _client(monkeypatch)
    rf_id = _rf_source(engine, stage="performed")
    result = client.post(f"/clinical/office-cases/surgeries/from-rf/{rf_id}", headers=HEADERS,
        json={"expected_rf_revision":2, "date_of_birth":"1966-01-01",
              "phone":"+35799000002", "procedure_type":"Synthetic surgery"})
    assert result.status_code == 409
    assert client.get("/clinical/surgeries", headers=HEADERS).json() == []


def test_linked_rf_delete_returns_conflict_and_preserves_source(monkeypatch):
    monkeypatch.setenv("CLINICAL_DATA_KEY", KEY)
    engine = create_engine("sqlite+pysqlite:///:memory:",
                           connect_args={"check_same_thread": False}, poolclass=StaticPool)
    @event.listens_for(engine, "connect")
    def _foreign_keys(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")
    SurgeryQueueBase.metadata.create_all(engine)
    RFBase.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(text("""CREATE TABLE clinical_surgery_details (
          surgery_id varchar PRIMARY KEY REFERENCES clinical_surgery_queue(id) ON DELETE RESTRICT,
          rf_application_id varchar UNIQUE REFERENCES clinic_rf_applications(id) ON DELETE RESTRICT,
          room_entry_time varchar(5), created_at datetime NOT NULL, updated_at datetime NOT NULL
        )"""))
    OfficeCaseBase.metadata.create_all(engine)
    app = FastAPI()
    app.include_router(build_surgery_queue_router(engine))
    app.include_router(build_office_cases_router(engine))
    app.include_router(build_rf_router(engine))
    client = TestClient(app)
    rf_id = _rf_source(engine, application_id="synthetic-rf-linked-01")
    added = client.post(f"/clinical/office-cases/surgeries/from-rf/{rf_id}", headers=HEADERS,
        json={"expected_rf_revision":2, "date_of_birth":"1966-01-01",
              "phone":"+35799000002", "procedure_type":"Synthetic surgery"})
    assert added.status_code == 200
    response = client.post(f"/clinical/clinic-utilities/rf/api/outcomes/{rf_id}/delete",
        headers=HEADERS, json={"expected_revision":2, "confirmation_code":rf_id[-8:]})
    assert response.status_code == 409
    with Session(engine) as session:
        assert session.get(RFApplicationORM, rf_id) is not None
        assert session.scalar(select(SurgeryDetailORM).where(
            SurgeryDetailORM.rf_application_id == rf_id)) is not None
