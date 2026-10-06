from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import clinical_calendar
from clinical_calendar import (
    ClinicalAppointmentORM, ClinicalAppointmentClassificationORM, build_clinical_calendar_router,
)


INGEST_KEY = "snapshot-test-key"
CLINICAL_KEY = "clinical-test-key"
HEADERS = {"X-Clinical-Ingest-Key": INGEST_KEY}
CLINICAL_HEADERS = {"X-Clinical-Key": CLINICAL_KEY}


def _client(monkeypatch):
    monkeypatch.setenv("CLINICAL_INGEST_KEY", INGEST_KEY)
    monkeypatch.setenv("CLINICAL_DATA_KEY", CLINICAL_KEY)
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
        "linked_patient_id": "synthetic-link",
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


def test_snapshot_filters_unrelated_rows_and_removes_missing(monkeypatch):
    client, engine = _client(monkeypatch)
    start = datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    first = client.post("/clinical/calendar/appointments/snapshot", headers=HEADERS,
        json=_snapshot(start, end, [
            _appointment("cal-1", start=start + timedelta(hours=1), minutes=40,
                         label="Οστεοπόρωση", comment="Οστεοπόρωση - επανέλεγχος"),
            _appointment("cal-unrelated", start=start + timedelta(hours=2), minutes=40,
                         label="Πόνος γόνατος")]))
    assert first.status_code == 200
    assert first.json()["inserted"] == 1
    assert first.json()["skipped_unrelated"] == 1
    with Session(engine) as session:
        assert [(row.source_appointment_id, row.category) for row in session.execute(
            select(ClinicalAppointmentORM)).scalars()] == [("cal-1", "osteoporosis_review")]
    now = start.replace(tzinfo=None)
    monkeypatch.setattr(clinical_calendar, "utcnow", lambda: now)
    _set_actual_source(
        monkeypatch,
        [_appointment("cal-1", start=start + timedelta(hours=1), minutes=40,
                      label="Οστεοπόρωση", comment="Οστεοπόρωση - επανέλεγχος")],
        now,
    )
    weekly = client.get("/clinical/calendar/appointments", headers=CLINICAL_HEADERS,
                        params={"start": start.isoformat(), "end": end.isoformat()})
    assert weekly.status_code == 200
    assert [row["appointment_id"] for row in weekly.json()] == ["cal.com:cal-1"]
    second = client.post("/clinical/calendar/appointments/snapshot", headers=HEADERS,
        json=_snapshot(start, end, [_appointment("cal-2", start=start + timedelta(hours=3),
                                                minutes=10, label="Prolia")]))
    assert second.json()["inserted"] == 1
    assert second.json()["removed_missing"] == 1
    with Session(engine) as session:
        assert [row.source_appointment_id for row in session.execute(
            select(ClinicalAppointmentORM)).scalars()] == ["cal-2"]


def test_snapshot_reclassification_to_other_removes_clinical_copy(monkeypatch):
    client, engine = _client(monkeypatch)
    start = datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    relevant = _appointment("cal-1", start=start + timedelta(hours=1), minutes=60,
                            label="Πρώτη επίσκεψη οστεοπόρωσης")
    assert client.post("/clinical/calendar/appointments/snapshot", headers=HEADERS,
                       json=_snapshot(start, end, [relevant])).json()["inserted"] == 1
    unrelated = dict(relevant, label="Ορθοπεδικός έλεγχος")
    changed = client.post("/clinical/calendar/appointments/snapshot", headers=HEADERS,
                          json=_snapshot(start, end, [unrelated]))
    assert changed.json()["removed_unrelated"] == 1
    with Session(engine) as session:
        assert session.get(ClinicalAppointmentORM, "cal.com:cal-1") is None
    repeated = client.post("/clinical/calendar/appointments/snapshot", headers=HEADERS,
                           json=_snapshot(start, end, [unrelated]))
    assert repeated.json()["skipped_unrelated"] == 1
    assert repeated.json()["removed_unrelated"] == 0


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
        _snapshot(start, end, [_appointment("invalid-interval", start=start, minutes=0, label="Other")]),
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
    rows = [_appointment(f"bounded-{index}", start=start + timedelta(hours=1),
                         minutes=20, label="Synthetic unrelated visit") for index in range(2000)]
    accepted = client.post("/clinical/calendar/appointments/snapshot", headers=HEADERS,
                           json=_snapshot(start, end, rows))
    assert accepted.status_code == 200
    assert accepted.json()["received"] == 2000
    assert accepted.json()["skipped_unrelated"] == 2000
    with Session(engine) as session:
        assert session.execute(select(ClinicalAppointmentORM)).scalars().all() == []


def test_manual_classification_survives_future_snapshot_and_can_return_to_auto(monkeypatch):
    client, _ = _client(monkeypatch)
    start = datetime(2026, 10, 1, 15, 0, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    payload = _snapshot(
        start,
        end,
        [
            _appointment(
                "manual-classification",
                start=start + timedelta(minutes=20),
                minutes=20,
                label="",
                comment="Οστεοπόρωση",
            )
        ],
    )

    first = client.post("/clinical/calendar/appointments/snapshot", headers=HEADERS, json=payload)
    assert first.status_code == 200
    now = start.replace(tzinfo=None)
    monkeypatch.setattr(clinical_calendar, "utcnow", lambda: now)
    _set_actual_source(monkeypatch, payload["appointments"], now)

    listed = client.get(
        "/clinical/calendar/appointments",
        headers=CLINICAL_HEADERS,
        params={"start": start.isoformat(), "end": end.isoformat()},
    )
    assert listed.status_code == 200
    assert listed.json()[0]["category"] == "osteoporosis_unspecified"
    assert listed.json()[0]["manual_category"] is None
    appointment_id = listed.json()[0]["appointment_id"]

    classified = client.put(
        f"/clinical/calendar/appointments/{appointment_id}/classification",
        headers=CLINICAL_HEADERS,
        json={"category": "osteoporosis_review"},
    )
    assert classified.status_code == 200
    assert classified.json() == {
        "appointment_id": appointment_id,
        "manual_category": "osteoporosis_review",
    }

    repeated = client.post("/clinical/calendar/appointments/snapshot", headers=HEADERS, json=payload)
    assert repeated.status_code == 200

    after_sync = client.get(
        "/clinical/calendar/appointments",
        headers=CLINICAL_HEADERS,
        params={"start": start.isoformat(), "end": end.isoformat()},
    ).json()[0]
    assert after_sync["category"] == "osteoporosis_review"
    assert after_sync["manual_category"] == "osteoporosis_review"

    cleared = client.put(
        f"/clinical/calendar/appointments/{appointment_id}/classification",
        headers=CLINICAL_HEADERS,
        json={"category": None},
    )
    assert cleared.status_code == 200
    assert cleared.json() == {"appointment_id": appointment_id, "manual_category": None}
    returned_to_auto = client.get(
        "/clinical/calendar/appointments",
        headers=CLINICAL_HEADERS,
        params={"start": start.isoformat(), "end": end.isoformat()},
    ).json()[0]
    assert returned_to_auto["category"] == "osteoporosis_unspecified"
    assert returned_to_auto["manual_category"] is None

    invalid = client.put(
        f"/clinical/calendar/appointments/{appointment_id}/classification",
        headers=CLINICAL_HEADERS,
        json={"category": "other"},
    )
    assert invalid.status_code == 422

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


def _set_actual_source(monkeypatch, rows, now, extra_fields=None):
    import httpx
    monkeypatch.setenv("RECEPTION_SCHEDULE_CONTEXT_URL", "https://reception.example/private/schedule-context")
    source_rows = []
    for row in rows:
        category = "aclasta" if row["label"].lower() == "aclasta" else "other"
        source_rows.append({
            "uid": f"cal.com:{row['source_appointment_id']}",
            "start_at": datetime.fromisoformat(row["start_at"]).replace(tzinfo=timezone.utc).isoformat(),
            "end_at": datetime.fromisoformat(row["end_at"]).replace(tzinfo=timezone.utc).isoformat(),
            "clinic": row["clinic"], "patient_display_name": row["patient_display_name"],
            "reason": row["comment"], "event_type_id": 393065,
            "category": category,
        })
        source_rows[-1].update(extra_fields or {})
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, url, headers, params=None):
            assert url == "https://reception.example/private/schedule-context"
            assert headers == {"X-Clinical-Ingest-Key": INGEST_KEY}
            return httpx.Response(200, json={
                "fetched_at": now.replace(tzinfo=timezone.utc).isoformat(),
                "coverage_start": (now.date() - timedelta(days=31)).isoformat(),
                "coverage_end": (now.date() + timedelta(days=62)).isoformat(),
                "appointments": source_rows,
            })
    monkeypatch.setattr(clinical_calendar.httpx, "AsyncClient", Client)


def _context(client):
    response = client.get("/clinical/calendar/cockpit-context", headers=CLINICAL_HEADERS)
    assert response.status_code == 200
    context = response.json()
    assert set(context) == {"generated_at", "source_updated_at", "today_total", "previous",
                            "current", "current_conflict_count", "next"}
    for slot in ("previous", "current", "next"):
        if context[slot]:
            assert set(context[slot]) == {"appointment_id", "start_at", "end_at", "clinic",
                                         "category", "patient_display_name", "reason"}
    assert "phone_e164" not in response.text
    assert "linked_patient_id" not in response.text
    assert "cal_uid" not in response.text
    assert "synthetic-link" not in response.text
    return context


def test_global_context_other_slots_cross_day_next_and_minimal_projection(monkeypatch):
    client, _ = _client(monkeypatch)
    now = datetime(2026, 10, 3, 8, 30)
    monkeypatch.setattr(clinical_calendar, "utcnow", lambda: now)
    rows = [
        _appointment("previous", start=now - timedelta(hours=1), minutes=20, label="Other"),
        _appointment("current", start=now - timedelta(minutes=10), minutes=30, label="Other",
                     comment="clinic=limassol | source=sync | cal_uid=hidden | Πόνος γόνατος"),
        _appointment("tomorrow", start=now + timedelta(days=1), minutes=20, label="Other"),
    ]
    _set_actual_source(monkeypatch, rows, now)
    context = _context(client)
    assert context["today_total"] == 2
    assert context["previous"]["appointment_id"] == "cal.com:previous"
    assert context["current"]["category"] == "other"
    assert context["current"]["reason"] == "Πόνος γόνατος"
    assert context["next"]["appointment_id"] == "cal.com:tomorrow"
    assert context["generated_at"] == context["source_updated_at"]


def test_global_context_is_protected_including_existing_browser_session(monkeypatch):
    from clinical_auth import ClinicalCookieMiddleware, build_auth_router
    client, engine = _client(monkeypatch)
    now = datetime(2026, 10, 3, 8, 30)
    monkeypatch.setattr(clinical_calendar, "utcnow", lambda: now)
    _set_actual_source(monkeypatch, [], now)
    assert client.get("/clinical/calendar/cockpit-context").status_code == 401
    assert client.get("/clinical/calendar/cockpit-context", headers=HEADERS).status_code == 401
    assert client.get("/clinical/calendar/cockpit-context",
                      headers={"X-Clinical-Key": "wrong"}).status_code == 401
    app = FastAPI()
    app.add_middleware(ClinicalCookieMiddleware)
    app.include_router(build_auth_router())
    app.include_router(build_clinical_calendar_router(engine))
    with TestClient(app, base_url="https://testserver") as browser:
        assert browser.post("/clinical/login", json={"key": CLINICAL_KEY}).status_code == 200
        assert browser.get("/clinical/calendar/cockpit-context").status_code == 200
    monkeypatch.delenv("CLINICAL_DATA_KEY")
    assert client.get("/clinical/calendar/cockpit-context", headers=CLINICAL_HEADERS).status_code == 503


def test_legacy_unrelated_discard_and_reclassification_removal_remain_compatible(monkeypatch):
    client, engine = _client(monkeypatch)
    now = datetime(2026, 10, 3, 8, 0)
    item = _appointment("legacy", start=now, minutes=40, label="Other")
    discarded = client.post("/clinical/calendar/appointments/import", headers=HEADERS, json=[item])
    assert discarded.json()["skipped_unrelated"] == 1
    item["label"] = "Prolia"
    assert client.post("/clinical/calendar/appointments/import", headers=HEADERS,
                       json=[item]).json()["inserted"] == 1
    item["label"] = "Other"
    removed = client.post("/clinical/calendar/appointments/import", headers=HEADERS, json=[item])
    assert removed.json()["removed_unrelated"] == 1
    with Session(engine) as session:
        assert session.get(ClinicalAppointmentORM, "cal.com:legacy") is None


def test_clearing_manual_relevant_override_minimizes_other_before_commit(monkeypatch):
    client, engine = _client(monkeypatch)
    now = datetime(2026, 10, 3, 8, 0)
    item = _appointment("manual-other", start=now, minutes=40, label="Prolia")
    payload = _snapshot(now, now + timedelta(days=1), [item])
    assert client.post("/clinical/calendar/appointments/snapshot", headers=HEADERS,
                       json=payload).json()["inserted"] == 1
    url = "/clinical/calendar/appointments/cal.com:manual-other/classification"
    assert client.put(url, headers=CLINICAL_HEADERS,
                      json={"category": "osteoporosis_review"}).status_code == 200
    item["label"] = "Other"
    client.post("/clinical/calendar/appointments/snapshot", headers=HEADERS,
                json=_snapshot(now, now + timedelta(days=1), [item]))
    cleared = client.put(url, headers=CLINICAL_HEADERS, json={"category": None})
    assert cleared.json() == {"appointment_id": "cal.com:manual-other", "manual_category": None}
    with Session(engine) as session:
        row = session.get(ClinicalAppointmentORM, "cal.com:manual-other")
        assert row.category == "other" and row.phone_e164 == "" and row.linked_patient_id is None


def test_context_preserves_start_order_aclasta_exception_and_fail_closed_overlaps(monkeypatch):
    base = datetime(2026, 10, 3, 9, 0)
    cases = [
        ("aclasta", "other", 40, 40, 50, "a", "b", 0),
        ("aclasta", "other", 40, 15, 65, "b", None, 0),
        ("other", "other", 40, 40, 50, None, None, 2),
        ("aclasta", "other", 0, 40, 10, None, None, 2),
        ("other", "aclasta", 40, 40, 50, None, None, 2),
    ]
    for cat_a, cat_b, offset_b, duration_b, now_offset, previous, current, conflicts in cases:
        client, _ = _client(monkeypatch)
        now = base + timedelta(minutes=now_offset)
        monkeypatch.setattr(clinical_calendar, "utcnow", lambda: now)
        rows = [_appointment("a", start=base, minutes=60, label=cat_a),
                _appointment("b", start=base + timedelta(minutes=offset_b), minutes=duration_b, label=cat_b)]
        _set_actual_source(monkeypatch, rows, now)
        context = _context(client)
        for slot, expected in (("previous", previous), ("current", current)):
            assert (context[slot]["appointment_id"] if context[slot] else None) == (
                f"cal.com:{expected}" if expected else None)
        assert context["current_conflict_count"] == conflicts


def test_context_uses_nicosia_day_including_dst_and_active_midnight_slot(monkeypatch):
    client, _ = _client(monkeypatch)
    now = datetime(2026, 10, 25, 22, 20)
    monkeypatch.setattr(clinical_calendar, "utcnow", lambda: now)
    rows = [_appointment("yesterday-active", start=now - timedelta(minutes=40), minutes=60, label="Other"),
            _appointment("today-next", start=now + timedelta(minutes=30), minutes=30, label="Other")]
    _set_actual_source(monkeypatch, rows, now)
    context = _context(client)
    assert context["today_total"] == 1
    assert context["previous"] is None
    assert context["current"]["appointment_id"] == "cal.com:yesterday-active"
    assert context["next"]["appointment_id"] == "cal.com:today-next"


def test_context_rejects_stale_and_unexpected_private_fields_without_snapshot_fallback(monkeypatch):
    client, engine = _client(monkeypatch)
    now = datetime(2026, 10, 3, 8, 30)
    monkeypatch.setattr(clinical_calendar, "utcnow", lambda: now)
    row = _appointment("urgent", start=now + timedelta(minutes=30), minutes=20, label="Other")
    client.post("/clinical/calendar/appointments/snapshot", headers=HEADERS,
                json=_snapshot(now, now + timedelta(days=1), [
                    _appointment("weekly", start=now, minutes=20, label="Prolia")]))
    _set_actual_source(monkeypatch, [row], now - timedelta(minutes=6))
    assert client.get("/clinical/calendar/cockpit-context", headers=CLINICAL_HEADERS).status_code == 503
    _set_actual_source(monkeypatch, [row], now, {"phone_e164": "+35799000000"})
    response = client.get("/clinical/calendar/cockpit-context", headers=CLINICAL_HEADERS)
    assert response.status_code == 503
    assert "+35799000000" not in response.text
    with Session(engine) as session:
        assert session.get(ClinicalAppointmentORM, "cal.com:weekly") is not None




def test_weekly_calendar_requests_displayed_week_and_requires_full_coverage(monkeypatch):
    import httpx
    client, _ = _client(monkeypatch)
    monkeypatch.setenv("RECEPTION_SCHEDULE_CONTEXT_URL", "https://reception.example/private/schedule-context")
    now = datetime(2026, 10, 6, 8, 0)
    monkeypatch.setattr(clinical_calendar, "utcnow", lambda: now)
    calls = []

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, url, headers, params=None):
            calls.append(params)
            return httpx.Response(200, json={
                "fetched_at": now.replace(tzinfo=timezone.utc).isoformat(),
                "coverage_start": "2026-10-11",
                "coverage_end": "2026-11-12",
                "appointments": [],
            })

    monkeypatch.setattr(clinical_calendar.httpx, "AsyncClient", Client)
    start = datetime(2026, 10, 12, 0, 0, tzinfo=timezone(timedelta(hours=3)))
    end = start + timedelta(days=7)
    response = client.get(
        "/clinical/calendar/appointments",
        headers=CLINICAL_HEADERS,
        params={"start": start.isoformat(), "end": end.isoformat()},
    )
    assert response.status_code == 200
    assert response.json() == []
    assert calls == [{"target_date": "2026-10-12"}]

    class Incomplete(Client):
        async def get(self, url, headers, params=None):
            return httpx.Response(200, json={
                "fetched_at": now.replace(tzinfo=timezone.utc).isoformat(),
                "coverage_start": "2026-10-12",
                "coverage_end": "2026-10-18",
                "appointments": [],
            })

    monkeypatch.setattr(clinical_calendar.httpx, "AsyncClient", Incomplete)
    unavailable = client.get(
        "/clinical/calendar/appointments",
        headers=CLINICAL_HEADERS,
        params={"start": start.isoformat(), "end": end.isoformat()},
    )
    assert unavailable.status_code == 503


def test_weekly_live_rows_preserve_classification_and_override_without_snapshot_row(monkeypatch):
    client, engine = _client(monkeypatch)
    now = datetime(2026, 10, 6, 8, 0)
    monkeypatch.setattr(clinical_calendar, "utcnow", lambda: now)
    start = datetime(2026, 10, 6, 8, 0)
    rows = [
        _appointment("aclasta-live", start=start, minutes=20, label="Other", comment="Aclasta"),
        _appointment("osteo-live", start=start + timedelta(hours=1), minutes=40,
                     label="Other", comment="Οστεοπόρωση"),
        _appointment("other-live", start=start + timedelta(hours=2), minutes=20,
                     label="Other", comment="Πόνος γόνατος"),
    ]
    _set_actual_source(monkeypatch, rows, now)

    listed = client.get(
        "/clinical/calendar/appointments",
        headers=CLINICAL_HEADERS,
        params={
            "start": datetime(2026, 10, 6, 0, 0, tzinfo=timezone.utc).isoformat(),
            "end": datetime(2026, 10, 7, 0, 0, tzinfo=timezone.utc).isoformat(),
        },
    )
    assert listed.status_code == 200
    by_id = {row["appointment_id"]: row for row in listed.json()}
    assert by_id["cal.com:aclasta-live"]["category"] == "aclasta"
    assert by_id["cal.com:osteo-live"]["category"] == "osteoporosis_review"
    assert "cal.com:other-live" not in by_id
    with Session(engine) as session:
        assert session.get(ClinicalAppointmentORM, "cal.com:osteo-live") is None

    classified = client.put(
        "/clinical/calendar/appointments/cal.com:other-live/classification",
        headers=CLINICAL_HEADERS,
        json={"category": "osteoporosis_review"},
    )
    assert classified.status_code == 200
    assert classified.json()["manual_category"] == "osteoporosis_review"

    relisted = client.get(
        "/clinical/calendar/appointments",
        headers=CLINICAL_HEADERS,
        params={
            "start": datetime(2026, 10, 6, 0, 0, tzinfo=timezone.utc).isoformat(),
            "end": datetime(2026, 10, 7, 0, 0, tzinfo=timezone.utc).isoformat(),
        },
    ).json()
    assert any(
        row["appointment_id"] == "cal.com:other-live"
        and row["manual_category"] == "osteoporosis_review"
        for row in relisted
    )


def test_snapshot_removes_only_missing_same_source_exact_window(monkeypatch):
    client, engine = _client(monkeypatch)
    start = datetime(2026, 10, 3, 8, 0)
    for source, source_id, when in (("cal.com", "missing", start),
                                    ("cal.com", "outside", start + timedelta(days=1)),
                                    ("setmore", "other-source", start)):
        client.post("/clinical/calendar/appointments/snapshot", headers=HEADERS,
                    json=_snapshot(when, when + timedelta(days=1),
                        [_appointment(source_id, start=when, minutes=20,
                                      label="Prolia", source=source)], source=source))
    response = client.post("/clinical/calendar/appointments/snapshot", headers=HEADERS,
                           json=_snapshot(start, start + timedelta(days=1), []))
    assert response.json()["removed_missing"] == 1
    with Session(engine) as session:
        assert {row.id for row in session.execute(select(ClinicalAppointmentORM)).scalars()} == {
            "cal.com:outside", "setmore:other-source"}
