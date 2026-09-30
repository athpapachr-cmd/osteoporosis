from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from clinical_surgery_queue import SurgeryQueueORM, build_surgery_queue_router


CLINICAL_KEY = "synthetic-clinical-key"
HEADERS = {"X-Clinical-Key": CLINICAL_KEY}


def _client(monkeypatch):
    monkeypatch.setenv("CLINICAL_DATA_KEY", CLINICAL_KEY)
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    app = FastAPI()
    app.include_router(build_surgery_queue_router(engine))
    return TestClient(app), engine


def _payload(
    *,
    identity_number: str,
    full_name: str,
    procedure_type: str,
    laterality: str,
    surgery_date: str | None = None,
    phone: str = "+35799000000",
    date_of_birth: str = "1970-01-01",
):
    return {
        "identity_number": identity_number,
        "full_name": full_name,
        "date_of_birth": date_of_birth,
        "phone": phone,
        "procedure_type": procedure_type,
        "laterality": laterality,
        "surgery_date": surgery_date,
    }


def test_surgery_queue_requires_clinical_auth(monkeypatch):
    client, _ = _client(monkeypatch)

    assert client.get("/clinical/surgeries").status_code == 401
    assert client.get(
        "/clinical/surgeries",
        headers={"X-Clinical-Key": "wrong"},
    ).status_code == 401


def test_create_list_preserves_identity_fields_and_pending_order(monkeypatch):
    client, engine = _client(monkeypatch)

    first = client.post(
        "/clinical/surgeries",
        headers=HEADERS,
        json=_payload(
            identity_number="ID-SYN-001",
            full_name="Synthetic Patient One",
            procedure_type="Total knee arthroplasty",
            laterality="left",
        ),
    )
    second = client.post(
        "/clinical/surgeries",
        headers=HEADERS,
        json=_payload(
            identity_number="ID-SYN-002",
            full_name="Synthetic Patient Two",
            procedure_type="Rotator cuff repair",
            laterality="right",
            surgery_date="2026-10-15",
        ),
    )

    assert first.status_code == 200
    assert second.status_code == 200

    rows = client.get("/clinical/surgeries", headers=HEADERS).json()
    assert [row["identity_number"] for row in rows] == ["ID-SYN-001", "ID-SYN-002"]
    assert [row["queue_position"] for row in rows] == [1, 2]
    assert rows[1]["surgery_date"] == "2026-10-15"

    with Session(engine) as session:
        stored = session.execute(
            select(SurgeryQueueORM).order_by(SurgeryQueueORM.queue_position)
        ).scalars().all()
        assert [row.identity_number for row in stored] == ["ID-SYN-001", "ID-SYN-002"]
        assert stored[0].full_name == "Synthetic Patient One"


def test_update_patient_details_and_surgery_date(monkeypatch):
    client, engine = _client(monkeypatch)

    created = client.post(
        "/clinical/surgeries",
        headers=HEADERS,
        json=_payload(
            identity_number="ID-SYN-010",
            full_name="Synthetic Patient",
            procedure_type="Hip arthroscopy",
            laterality="right",
        ),
    ).json()

    response = client.put(
        f"/clinical/surgeries/{created['surgery_id']}",
        headers=HEADERS,
        json={
            "full_name": "Synthetic Patient Updated",
            "phone": "+35799111111",
            "procedure_type": "Total hip arthroplasty",
            "laterality": "left",
            "surgery_date": "2026-11-04",
        },
    )

    assert response.status_code == 200
    row = response.json()
    assert row["full_name"] == "Synthetic Patient Updated"
    assert row["phone"] == "+35799111111"
    assert row["procedure_type"] == "Total hip arthroplasty"
    assert row["laterality"] == "left"
    assert row["surgery_date"] == "2026-11-04"

    with Session(engine) as session:
        stored = session.get(SurgeryQueueORM, created["surgery_id"])
        assert stored is not None
        assert stored.identity_number == "ID-SYN-010"
        assert stored.full_name == "Synthetic Patient Updated"
        assert stored.phone == "+35799111111"


def test_move_up_down_persists_manual_queue_order(monkeypatch):
    client, _ = _client(monkeypatch)

    ids = []
    for index in range(3):
        row = client.post(
            "/clinical/surgeries",
            headers=HEADERS,
            json=_payload(
                identity_number=f"ID-SYN-MOVE-{index}",
                full_name=f"Synthetic Move {index}",
                procedure_type=f"Procedure {index}",
                laterality="not_applicable",
            ),
        ).json()
        ids.append(row["surgery_id"])

    moved = client.post(
        f"/clinical/surgeries/{ids[2]}/move",
        headers=HEADERS,
        json={"direction": "up"},
    )
    assert moved.status_code == 200
    assert [row["surgery_id"] for row in moved.json()] == [ids[0], ids[2], ids[1]]

    moved_again = client.post(
        f"/clinical/surgeries/{ids[2]}/move",
        headers=HEADERS,
        json={"direction": "up"},
    )
    assert [row["surgery_id"] for row in moved_again.json()] == [ids[2], ids[0], ids[1]]

    stable_at_top = client.post(
        f"/clinical/surgeries/{ids[2]}/move",
        headers=HEADERS,
        json={"direction": "up"},
    )
    assert [row["surgery_id"] for row in stable_at_top.json()] == [ids[2], ids[0], ids[1]]
    assert [row["queue_position"] for row in stable_at_top.json()] == [1, 2, 3]


def test_complete_removes_case_from_default_pending_but_preserves_record(monkeypatch):
    client, engine = _client(monkeypatch)

    first = client.post(
        "/clinical/surgeries",
        headers=HEADERS,
        json=_payload(
            identity_number="ID-SYN-COMP-1",
            full_name="Synthetic Complete One",
            procedure_type="Procedure A",
            laterality="left",
            surgery_date="2026-10-01",
        ),
    ).json()
    second = client.post(
        "/clinical/surgeries",
        headers=HEADERS,
        json=_payload(
            identity_number="ID-SYN-COMP-2",
            full_name="Synthetic Complete Two",
            procedure_type="Procedure B",
            laterality="right",
        ),
    ).json()

    completed = client.post(
        f"/clinical/surgeries/{first['surgery_id']}/complete",
        headers=HEADERS,
    )
    assert completed.status_code == 200
    assert completed.json()["status"] == "completed"

    pending = client.get("/clinical/surgeries", headers=HEADERS).json()
    assert [row["surgery_id"] for row in pending] == [second["surgery_id"]]
    assert pending[0]["queue_position"] == 1

    completed_rows = client.get(
        "/clinical/surgeries?status=completed",
        headers=HEADERS,
    ).json()
    assert [row["surgery_id"] for row in completed_rows] == [first["surgery_id"]]

    with Session(engine) as session:
        stored = session.get(SurgeryQueueORM, first["surgery_id"])
        assert stored is not None
        assert stored.status == "completed"
        assert stored.completed_at is not None


def test_invalid_status_and_nonpending_move_fail_closed(monkeypatch):
    client, _ = _client(monkeypatch)

    assert client.get(
        "/clinical/surgeries?status=unknown",
        headers=HEADERS,
    ).status_code == 422

    created = client.post(
        "/clinical/surgeries",
        headers=HEADERS,
        json=_payload(
            identity_number="ID-SYN-GUARD",
            full_name="Synthetic Guard",
            procedure_type="Procedure",
            laterality="bilateral",
        ),
    ).json()
    client.post(f"/clinical/surgeries/{created['surgery_id']}/complete", headers=HEADERS)

    assert client.post(
        f"/clinical/surgeries/{created['surgery_id']}/move",
        headers=HEADERS,
        json={"direction": "down"},
    ).status_code == 409
