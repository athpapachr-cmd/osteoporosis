from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from clinical_calendar import ClinicalAppointmentORM, build_clinical_calendar_router


INGEST_KEY = "snapshot-test-key"
HEADERS = {"X-Clinical-Ingest-Key": INGEST_KEY}


def _client(monkeypatch):
    monkeypatch.setenv("CLINICAL_INGEST_KEY", INGEST_KEY)
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    app = FastAPI()
    app.include_router(build_clinical_calendar_router(engine))
    return TestClient(app), engine


def _appointment(
    source_id: str,
    *,
    start: datetime,
    minutes: int,
    label: str,
    comment: str = "",
    source: str = "cal.com",
):
    return {
        "source": source,
        "source_appointment_id": source_id,
        "start_at": start.isoformat(),
        "end_at": (start + timedelta(minutes=minutes)).isoformat(),
        "clinic": "limassol",
        "patient_display_name": "Synthetic Patient",
        "phone_e164": "+35799000000",
        "label": label,
        "comment": comment,
        "status": "scheduled",
    }


def _snapshot(window_start, window_end, appointments, *, source="cal.com"):
    return {
        "source": source,
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "appointments": appointments,
    }


def test_snapshot_upserts_relevant_rows_filters_unrelated_and_removes_missing(monkeypatch):
    client, engine = _client(monkeypatch)
    start = datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    first = client.post(
        "/clinical/calendar/appointments/snapshot",
        headers=HEADERS,
        json=_snapshot(
            start,
            end,
            [
                _appointment(
                    "cal-1",
                    start=start + timedelta(hours=1),
                    minutes=40,
                    label="Οστεοπόρωση",
                    comment="Οστεοπόρωση - επανέλεγχος",
                ),
                _appointment(
                    "cal-unrelated",
                    start=start + timedelta(hours=2),
                    minutes=40,
                    label="Πόνος γόνατος",
                ),
            ],
        ),
    )
    assert first.status_code == 200
    assert first.json() == {
        "received": 2,
        "imported": 1,
        "inserted": 1,
        "updated": 0,
        "skipped_unrelated": 1,
        "removed_unrelated": 0,
        "skipped_invalid": 0,
        "removed_missing": 0,
    }

    with Session(engine) as session:
        rows = session.execute(select(ClinicalAppointmentORM)).scalars().all()
        assert [(row.source, row.source_appointment_id, row.category) for row in rows] == [
            ("cal.com", "cal-1", "osteoporosis_review")
        ]

    second = client.post(
        "/clinical/calendar/appointments/snapshot",
        headers=HEADERS,
        json=_snapshot(
            start,
            end,
            [
                _appointment(
                    "cal-2",
                    start=start + timedelta(hours=3),
                    minutes=10,
                    label="Prolia",
                    comment="Prolia injection",
                )
            ],
        ),
    )
    assert second.status_code == 200
    assert second.json()["inserted"] == 1
    assert second.json()["removed_missing"] == 1

    with Session(engine) as session:
        rows = session.execute(select(ClinicalAppointmentORM)).scalars().all()
        assert [(row.source_appointment_id, row.category) for row in rows] == [
            ("cal-2", "prolia")
        ]


def test_snapshot_reclassification_removes_previously_relevant_row(monkeypatch):
    client, engine = _client(monkeypatch)
    start = datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    seed = client.post(
        "/clinical/calendar/appointments/snapshot",
        headers=HEADERS,
        json=_snapshot(
            start,
            end,
            [
                _appointment(
                    "cal-1",
                    start=start + timedelta(hours=1),
                    minutes=60,
                    label="Πρώτη επίσκεψη οστεοπόρωσης",
                )
            ],
        ),
    )
    assert seed.status_code == 200

    reclassified = client.post(
        "/clinical/calendar/appointments/snapshot",
        headers=HEADERS,
        json=_snapshot(
            start,
            end,
            [
                _appointment(
                    "cal-1",
                    start=start + timedelta(hours=1),
                    minutes=60,
                    label="Ορθοπεδικός έλεγχος",
                )
            ],
        ),
    )
    assert reclassified.status_code == 200
    assert reclassified.json()["removed_unrelated"] == 1
    assert reclassified.json()["removed_missing"] == 0

    with Session(engine) as session:
        assert session.execute(select(ClinicalAppointmentORM)).scalars().all() == []


def test_snapshot_rejects_incomplete_or_ambiguous_scope_before_deleting(monkeypatch):
    client, engine = _client(monkeypatch)
    start = datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    seed = client.post(
        "/clinical/calendar/appointments/snapshot",
        headers=HEADERS,
        json=_snapshot(
            start,
            end,
            [
                _appointment(
                    "cal-keep",
                    start=start + timedelta(hours=1),
                    minutes=10,
                    label="Prolia",
                )
            ],
        ),
    )
    assert seed.status_code == 200

    bad_payloads = [
        _snapshot(
            start,
            end,
            [
                _appointment(
                    "other-source",
                    start=start + timedelta(hours=2),
                    minutes=10,
                    label="Prolia",
                    source="setmore",
                )
            ],
        ),
        _snapshot(
            start,
            end,
            [
                _appointment(
                    "duplicate",
                    start=start + timedelta(hours=2),
                    minutes=10,
                    label="Prolia",
                ),
                _appointment(
                    "duplicate",
                    start=start + timedelta(hours=3),
                    minutes=10,
                    label="Prolia",
                ),
            ],
        ),
        _snapshot(
            start,
            end,
            [
                _appointment(
                    "outside",
                    start=end + timedelta(hours=1),
                    minutes=10,
                    label="Prolia",
                )
            ],
        ),
    ]
    for payload in bad_payloads:
        response = client.post(
            "/clinical/calendar/appointments/snapshot",
            headers=HEADERS,
            json=payload,
        )
        assert response.status_code == 422

    with Session(engine) as session:
        rows = session.execute(select(ClinicalAppointmentORM)).scalars().all()
        assert [row.source_appointment_id for row in rows] == ["cal-keep"]


def test_snapshot_accepts_cyprus_dst_fallback_31_local_day_window(monkeypatch):
    client, _ = _client(monkeypatch)

    # 2026-09-28 00:00 Asia/Nicosia -> 2026-10-29 00:00 Asia/Nicosia.
    # Cyprus leaves DST during this interval, so UTC elapsed time is 31d + 1h.
    start = datetime(2026, 9, 27, 21, 0, tzinfo=timezone.utc)
    dst_compatible_end = datetime(2026, 10, 28, 22, 0, tzinfo=timezone.utc)

    accepted = client.post(
        "/clinical/calendar/appointments/snapshot",
        headers=HEADERS,
        json=_snapshot(start, dst_compatible_end, []),
    )
    assert accepted.status_code == 200
    assert accepted.json()["received"] == 0

    too_long = client.post(
        "/clinical/calendar/appointments/snapshot",
        headers=HEADERS,
        json=_snapshot(start, dst_compatible_end + timedelta(seconds=1), []),
    )
    assert too_long.status_code == 422


def test_snapshot_accepts_producer_complete_cardinality_ceiling(monkeypatch):
    client, engine = _client(monkeypatch)
    start = datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    # Producer bound is 20 pages × 100 rows. Use unrelated synthetic rows so
    # this test exercises snapshot acceptance without populating clinical data.
    rows = [
        _appointment(
            f"bounded-{index}",
            start=start + timedelta(hours=1),
            minutes=20,
            label="Synthetic unrelated visit",
        )
        for index in range(2000)
    ]

    accepted = client.post(
        "/clinical/calendar/appointments/snapshot",
        headers=HEADERS,
        json=_snapshot(start, end, rows),
    )
    assert accepted.status_code == 200
    assert accepted.json()["received"] == 2000
    assert accepted.json()["skipped_unrelated"] == 2000

    with Session(engine) as session:
        assert session.execute(select(ClinicalAppointmentORM)).scalars().all() == []

    rejected = client.post(
        "/clinical/calendar/appointments/snapshot",
        headers=HEADERS,
        json=_snapshot(
            start,
            end,
            rows
            + [
                _appointment(
                    "bounded-overflow",
                    start=start + timedelta(hours=1),
                    minutes=20,
                    label="Synthetic unrelated visit",
                )
            ],
        ),
    )
    assert rejected.status_code == 422

    with Session(engine) as session:
        assert session.execute(select(ClinicalAppointmentORM)).scalars().all() == []


def test_snapshot_requires_ingest_key_and_canonical_source(monkeypatch):
    client, _ = _client(monkeypatch)
    start = datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    payload = _snapshot(start, end, [])

    assert client.post("/clinical/calendar/appointments/snapshot", json=payload).status_code == 401
    assert client.post(
        "/clinical/calendar/appointments/snapshot",
        headers={"X-Clinical-Ingest-Key": "wrong"},
        json=payload,
    ).status_code == 401

    payload["source"] = " cal.com "
    assert client.post(
        "/clinical/calendar/appointments/snapshot",
        headers=HEADERS,
        json=payload,
    ).status_code == 422


def test_legacy_single_import_remains_available_after_snapshot_refactor(monkeypatch):
    client, engine = _client(monkeypatch)
    start = datetime(2026, 9, 28, 9, 0, tzinfo=timezone.utc)

    response = client.post(
        "/clinical/calendar/appointments/import",
        headers=HEADERS,
        json=[
            _appointment(
                "legacy-1",
                start=start,
                minutes=60,
                label="Aclasta infusion",
            )
        ],
    )
    assert response.status_code == 200
    assert response.json()["inserted"] == 1

    with Session(engine) as session:
        row = session.get(ClinicalAppointmentORM, "cal.com:legacy-1")
        assert row is not None
        assert row.category == "aclasta"
