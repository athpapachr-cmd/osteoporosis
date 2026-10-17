"""Synthetic only: exact Reception booking -> protected patient Visit Brief boundaries."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import clinical_visit_brief as brief
from clinical_calendar import build_clinical_calendar_router
from clinical_data import (
    ClinicalBase, PatientORM, EncounterORM, ClinicalPendingORM,
)
from clinical_visit_brief import (
    BookingPatientLinkORM, BookingPatientLinkAuditORM,
    VisitBriefBase, VisitBriefStaffAuditORM,
)
from clinical_staff_auth import StaffActor, STAFF_COOKIE, issue_staff_cookie


KEY = "synthetic-clinic-key"
HEAD = {"X-Clinical-Key": KEY}
NOW = datetime(2026, 10, 16, 9, 0, tzinfo=timezone.utc)
BOOKING = {
    "uid": "cal.com:opaque-123",
    "start_at": "2026-10-16T14:00:00+03:00",
    "end_at": "2026-10-16T14:30:00+03:00",
    "clinic": "limassol",
    "patient_display_name": "Synthetic Person",
    "reason": "Επανέλεγχος",
    "event_type_id": 123,
    "category": "other",
}


def client(monkeypatch, *, enabled=True):
    monkeypatch.setenv("CLINICAL_DATA_KEY", KEY)
    monkeypatch.setenv("CLINICAL_INGEST_KEY", "synthetic-ingest")
    monkeypatch.setenv("RECEPTION_SCHEDULE_CONTEXT_URL", "https://example.invalid/reception")
    monkeypatch.setenv("CLINICAL_VISIT_BRIEF_LINKS_ENABLED", "1" if enabled else "0")
    monkeypatch.setenv("COCKPIT_STAFF_AUTH_ENABLED", "1")
    monkeypatch.setenv("COCKPIT_VISIT_BRIEF_PRIVATE_STORAGE_QUALIFIED", "1")
    monkeypatch.setenv("COCKPIT_VISIT_BRIEF_LINK_RETENTION_APPROVED", "1")
    monkeypatch.setenv("COCKPIT_VISIT_BRIEF_AUDIT_RETENTION_APPROVED", "1")
    monkeypatch.setenv("COCKPIT_STAFF_CLINICIAN_KEY", "clinician-test-" + "c" * 36)
    monkeypatch.setenv("COCKPIT_STAFF_SECRETARY_KEY", "secretary-test-" + "s" * 36)
    monkeypatch.setenv("COCKPIT_STAFF_SESSION_SECRET", "signing-test-" + "x" * 40)
    monkeypatch.setattr(brief, "_utcnow", lambda: NOW)
    engine = create_engine("sqlite+pysqlite:///:memory:",
                           connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    ClinicalBase.metadata.create_all(bind=engine)
    # Synthetic isolated schema fixture; runtime never does Visit Brief DDL on boot.
    VisitBriefBase.metadata.create_all(bind=engine)
    app = FastAPI()
    app.include_router(build_clinical_calendar_router(engine))
    session = TestClient(app)
    session.cookies.set(STAFF_COOKIE, issue_staff_cookie(
        StaffActor("clinic-clinician", "clinician")), path="/clinical")
    with Session(engine) as db:
        for patient_id in ("P1", "P2"):
            db.add(PatientORM(patient_id=patient_id, demographics_json={},
                              created_at=NOW.replace(tzinfo=None),
                              updated_at=NOW.replace(tzinfo=None)))
        db.commit()
    set_source(monkeypatch)
    return session, engine


def set_source(monkeypatch, *, bookings=None, fetched_at=None,
               coverage=True, error=False):
    rows = [dict(BOOKING)] if bookings is None else bookings

    class FakeClient:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, url, params=None, headers=None):
            assert url == "https://example.invalid/reception"
            assert headers == {"X-Clinical-Ingest-Key": "synthetic-ingest"}
            assert params == {"target_date": "2026-10-16"}
            if error:
                return httpx.Response(503)
            return httpx.Response(200, json={
                "fetched_at": (fetched_at or NOW).isoformat(),
                "coverage_start": "2026-10-16" if coverage else "2026-10-17",
                "coverage_end": "2026-11-16",
                "appointments": rows,
            })
    monkeypatch.setattr(brief.httpx, "AsyncClient", FakeClient)


def read(client, uid=BOOKING["uid"]):
    return client.post("/clinical/calendar/visit-brief/read",
                       json={"booking_uid": uid}, headers=HEAD)


def confirm(client, patient="P1", revision=0, uid=BOOKING["uid"], **extra):
    # Observe the *actual* fresh booking snapshot before claiming verification.
    response = read(client, uid)
    seen = response.json().get("booking", {}) if response.status_code == 200 else {}
    observation = extra.pop("observed_booking", seen.get("observation", "0" * 64))
    return client.post("/clinical/calendar/visit-brief/confirm", headers=HEAD,
                       json={"booking_uid": uid, "patient_id": patient,
                             "expected_revision": revision,
                             "independently_verified": True,
                             "observed_booking": observation,
                             "evidence_type": "protected_chart_identifiers", **extra})


def signed_candidate(day, when=None):
    return {
        "schema_version": "visit_capture_candidate_v1",
        "source_identity_state": "consistent",
        "encounter_date": day,
        "reason_for_visit": {"text": "Αξιολόγηση", "provenance": ["synthetic"]},
        "source_bindings": [{"source_ref": "synthetic", "role": "clinician_edit"}],
        "comparison_basis": {"type": "unavailable"},
        "decisions": [{"text": "Επανέλεγχος συμπτωμάτων", "type": "advice",
                       "provenance": ["synthetic"]}],
        "medications": [{"name": "SyntheticMed", "action": "continued",
                         "provenance": ["synthetic"]}],
        "next_contact": ({"when": when, "purpose": "Έλεγχος συμπτωμάτων",
                          "assess": ["Επανεξέταση ROM"], "provenance": ["synthetic"]}
                         if when is not None else None),
    }


def add_encounter(engine, ident, day, signed_at, *, created=None, updated=None,
                  status="completed", when=None, signed=True):
    created = created or datetime(2026, 10, 15, 8)
    updated = updated or created
    with Session(engine) as db:
        db.add(EncounterORM(
            id=ident, patient_id="P1", encounter_date=day, status=status,
            payload_json={"_visit_capture_v1": {
                "signed": signed, "signed_at": signed_at,
                "candidate": signed_candidate(day, when),
            }},
            created_at=created, updated_at=updated))
        db.commit()


def add_pending(engine, ident, status, when):
    with Session(engine) as db:
        db.add(ClinicalPendingORM(
            id=ident, patient_id="P1", source_encounter_id="previous",
            item_type="test", description="Εργαστηριακές εξετάσεις",
            trigger_text=None, status=status, responsible_role="patient",
            external_dependency_json=None, review_date=when,
            provenance_json=["synthetic"], resolution_event_id=None,
            created_at=datetime(2026, 10, 15, 8),
            updated_at=datetime(2026, 10, 15, 8),
        ))
        db.commit()


def test_unlinked_and_feature_disabled_never_reveal_patient(monkeypatch):
    c, _ = client(monkeypatch, enabled=False)
    r = read(c)
    assert r.status_code == 200
    assert r.json()["reason"] == "disabled"
    assert "patient_id" not in r.text
    assert confirm(c).status_code == 503
    monkeypatch.setenv("CLINICAL_VISIT_BRIEF_LINKS_ENABLED", "1")
    r = read(c)
    assert r.json()["reason"] == "unlinked"
    assert "brief" not in r.json()
    assert confirm(c, revision=2).status_code == 409
    assert confirm(c, patient="NONEXISTENT").status_code == 404
    assert c.post("/clinical/calendar/visit-brief/confirm", json={
        "booking_uid": BOOKING["uid"], "patient_id": "P1",
        "expected_revision": 0, "independently_verified": False,
        "evidence_type": "protected_chart_identifiers",
    }, headers=HEAD).status_code == 422


def test_per_open_reconfirmation_no_reusable_read_authority(monkeypatch):
    c, engine = client(monkeypatch)
    assert read(c).json()["mode"] == "booking_only"
    first = confirm(c)
    assert first.status_code == 200
    assert first.json()["mode"] == "confirmed_once"
    assert first.json()["brief"]["patient_id"] == "P1"
    assert read(c).json() == {
        "mode": "booking_only", "reason": "needs_reconfirmation",
        "booking": first.json()["booking"], "revision": 1, "linked": True,
    }
    assert confirm(c, revision=0).status_code == 409
    second = confirm(c, revision=1)
    assert second.status_code == 200 and second.json()["revision"] == 2
    assert read(c).json()["mode"] == "booking_only"
    with Session(engine) as db:
        assert db.scalar(select(BookingPatientLinkORM)).state == "needs_reconfirmation"
        assert [event.event for event in db.scalars(select(BookingPatientLinkAuditORM)
                        .order_by(BookingPatientLinkAuditORM.revision)).all()] == [
                            "confirmed", "reconfirmed"]


def test_same_name_reassignment_collision_second_uid_and_revision(monkeypatch):
    c, engine = client(monkeypatch)
    assert confirm(c).status_code == 200
    # Replacement of the actual person with IDENTICAL source display fields
    # cannot grant an automatic read based on a stable UID or fingerprint.
    set_source(monkeypatch, bookings=[dict(BOOKING)])
    assert read(c).json()["reason"] == "needs_reconfirmation"
    assert confirm(c, patient="P2", revision=1).status_code == 409
    changed = confirm(c, patient="P2", revision=1, replace_existing=True)
    assert changed.status_code == 200
    assert changed.json()["brief"]["patient_id"] == "P2"
    assert read(c).json()["mode"] == "booking_only"
    with Session(engine) as db:
        assert db.scalar(select(BookingPatientLinkORM)).patient_id == "P2"
    other = dict(BOOKING, uid="cal.com:opaque-456")
    set_source(monkeypatch, bookings=[dict(BOOKING), other])
    assert read(c, other["uid"]).json()["reason"] == "unlinked"
    assert confirm(c, uid=other["uid"], revision=2).status_code == 409
    set_source(monkeypatch, bookings=[dict(BOOKING), dict(BOOKING)])
    assert read(c).status_code == 409
    assert confirm(c, revision=2).status_code == 409


def test_source_freshness_coverage_missing_cancel_and_auth(monkeypatch):
    c, _ = client(monkeypatch)
    # Shared Clinical Key alone cannot create an attributable staff read.
    anonymous = TestClient(c.app)
    assert anonymous.post("/clinical/calendar/visit-brief/read",
                          headers=HEAD,
                          json={"booking_uid": BOOKING["uid"]}).status_code == 401
    for opts in (
        {"fetched_at": NOW - timedelta(minutes=6)},
        {"coverage": False},
        {"error": True},
    ):
        set_source(monkeypatch, **opts)
        assert read(c).status_code == 503
        assert confirm(c).status_code == 503
    set_source(monkeypatch, bookings=[])
    assert read(c).status_code == 409
    assert confirm(c).status_code == 409


def test_selected_booking_start_same_day_future_backfill_amendment_dst(monkeypatch):
    c, engine = client(monkeypatch)
    # 2026-10-16 14:00 Cyprus = 11:00 UTC. 10:00 UTC is a legitimate
    # same-local-day signed predecessor; 11:01 is future to this booking.
    add_encounter(engine, "previous", "2026-10-15",
                  "2026-10-15T08:00:00", when="2026-10-16")
    add_encounter(engine, "same-day", "2026-10-16",
                  "2026-10-16T10:00:00", created=datetime(2026, 10, 16, 9),
                  updated=datetime(2026, 10, 16, 10), when="2026-10-16")
    add_encounter(engine, "signed-later", "2026-10-16",
                  "2026-10-16T11:01:00", created=datetime(2026, 10, 16, 9),
                  updated=datetime(2026, 10, 16, 11, 1))
    add_encounter(engine, "amended-later", "2026-10-15",
                  "2026-10-15T07:00:00", created=datetime(2026, 10, 15, 7),
                  updated=datetime(2026, 10, 16, 12), status="amended")
    add_encounter(engine, "unsigned", "2026-10-16",
                  "2026-10-16T09:00:00", signed=False)
    add_pending(engine, "due", "awaiting_result", "2026-10-16")
    add_pending(engine, "other", "open", "2026-10-20")
    add_pending(engine, "no-date", "open", None)
    add_pending(engine, "closed", "resolved", "2026-10-16")
    response = confirm(c).json()["brief"]
    assert response["previous_visit"]["source_encounter_id"] == "same-day"
    assert response["previous_visit"]["signed_at"] == "2026-10-16T10:00:00+00:00"
    decision = response["prior_plan"]["decisions"][0]
    assert decision["text"] == "Επανέλεγχος συμπτωμάτων"
    assert decision["verification"] == "source_stated"
    assert decision["certainty"] == "certain"
    assert decision["source_roles"] == ["clinician_edit"]
    assert response["check_heading"] == "Σήμερα"
    assert len(response["pending"]) == 3
    assert sum(x["kind"] == "pending_status_check" for x in response["appointment_checks"]) == 1
    assert any(x["kind"] == "previous_next_contact" for x in response["appointment_checks"])
    assert all(x["truth"] == "last_recorded_at_read" for x in response["pending"])
    assert "candidate" not in str(response)
    # Cyprus DST boundary: autumn 02:30Z is 04:30 +02, not 05:30 +03.
    assert datetime.fromisoformat("2026-10-25T02:30:00+00:00").astimezone(
        ZoneInfo("Asia/Nicosia")).hour == 4


def test_nonmatching_and_undated_plans_stay_historical(monkeypatch):
    c, engine = client(monkeypatch)
    add_encounter(engine, "prior", "2026-10-15", "2026-10-15T08:00:00",
                  when="2026-10-18")
    add_pending(engine, "not-date", "open", None)
    add_pending(engine, "another-day", "awaiting_result", "2026-10-17")
    info = confirm(c).json()["brief"]
    assert info["previous_visit"]["source_encounter_id"] == "prior"
    assert info["appointment_checks"] == []
    assert info["historical_plan"]["when"] == "2026-10-18"
    assert len(info["pending"]) == 2


def test_revoke_requires_revision_and_stops_pending_history(monkeypatch):
    c, engine = client(monkeypatch)
    assert confirm(c).status_code == 200
    url = "/clinical/calendar/visit-brief/unlink"
    assert c.post(url, headers=HEAD, json={
        "booking_uid": BOOKING["uid"], "expected_revision": 0,
        "explicit_unlink": True}).status_code == 422  # schema rejects a non-positive expected revision
    assert c.post(url, headers=HEAD, json={
        "booking_uid": BOOKING["uid"], "expected_revision": 1,
        "explicit_unlink": True}).status_code == 200
    assert read(c).json()["reason"] == "unlinked"
    assert confirm(c, revision=1).status_code == 409
    assert confirm(c, revision=2).status_code == 200
    with Session(engine) as db:
        events = [e.event for e in db.scalars(
            select(BookingPatientLinkAuditORM).order_by(
                BookingPatientLinkAuditORM.revision)).all()]
        assert events == ["confirmed", "revoked", "reconfirmed"]



def test_revoked_link_explicit_patient_correction_uses_cas_and_audit(monkeypatch):
    c, engine = client(monkeypatch)
    assert confirm(c, patient="P1").status_code == 200
    url = "/clinical/calendar/visit-brief/unlink"
    revoked = c.post(url, headers=HEAD, json={
        "booking_uid": BOOKING["uid"], "expected_revision": 1,
        "explicit_unlink": True})
    assert revoked.status_code == 200
    assert revoked.json()["revision"] == 2

    # A revoked row must remain booking-only, with a usable correction revision.
    unlinked = read(c).json()
    assert unlinked["mode"] == "booking_only"
    assert unlinked["reason"] == "unlinked"
    assert unlinked["revision"] == 2 and unlinked["linked"] is False
    assert "patient_id" not in str(unlinked)

    # No silent reassignment, and no stale CAS when changing P1 to P2.
    assert confirm(c, patient="P2", revision=2).status_code == 409
    assert confirm(c, patient="P2", revision=1, replace_existing=True).status_code == 409
    changed = confirm(c, patient="P2", revision=2, replace_existing=True)
    assert changed.status_code == 200
    assert changed.json()["revision"] == 3
    assert changed.json()["mode"] == "confirmed_once"
    assert changed.json()["brief"]["patient_id"] == "P2"
    assert read(c).json()["reason"] == "needs_reconfirmation"
    with Session(engine) as db:
        link = db.scalar(select(BookingPatientLinkORM))
        assert link.patient_id == "P2"
        assert link.revision == 3
        assert link.state == "needs_reconfirmation"
        events = db.scalars(
            select(BookingPatientLinkAuditORM).order_by(
                BookingPatientLinkAuditORM.revision)).all()
        assert [(e.event, e.revision, e.patient_id) for e in events] == [
            ("confirmed", 1, "P1"), ("revoked", 2, "P1"), ("corrected", 3, "P2")]


def test_changed_booking_since_preview_requires_fresh_reconfirmation(monkeypatch):
    c, engine = client(monkeypatch)
    old = read(c).json()["booking"]["observation"]
    changed = dict(BOOKING, patient_display_name="Different Synthetic Person")
    set_source(monkeypatch, bookings=[changed])
    response = confirm(c, observed_booking=old)
    assert response.status_code == 409
    assert "Booking changed since clinician preview" in response.text
    with Session(engine) as db:
        assert db.scalar(select(BookingPatientLinkORM)) is None
    fresh = confirm(c)
    assert fresh.status_code == 200
    assert read(c).json()["reason"] == "needs_reconfirmation"
