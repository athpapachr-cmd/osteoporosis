"""Persistent #161 boundary and real shared-backend tests. No external provider."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import Session
from fastapi.staticfiles import StaticFiles

from clinical_auth import ClinicalCookieMiddleware, build_auth_router
from clinical_calendar import CalendarBase, build_clinical_calendar_router
from clinical_data import ClinicalBase, PatientORM, EncounterORM, build_clinical_router
from clinical_visit_brief import VisitBriefBase, BookingPatientLinkORM, VisitBriefStaffAuditORM
from clinical_synthetic_161 import build_synthetic_161_router, PREFIX, BOOKINGS, SYNTHETIC_161_PATIENT_IDS

KEY = "synthetic-test-existing-key"


def application(engine):
    app = FastAPI()
    app.add_middleware(ClinicalCookieMiddleware)
    app.include_router(build_auth_router())
    app.include_router(build_clinical_router(engine))
    app.include_router(build_clinical_calendar_router(engine))
    app.include_router(build_synthetic_161_router(engine))
    app.mount("/static", StaticFiles(directory="static"), name="static")
    return app


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("CLINICAL_DATA_KEY", KEY)
    monkeypatch.delenv("COCKPIT_STAFF_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("CLINICAL_VISIT_BRIEF_LINKS_ENABLED", raising=False)
    engine = create_engine("sqlite:///" + str(tmp_path / "persistent.db"), connect_args={"check_same_thread": False})
    ClinicalBase.metadata.create_all(engine)
    CalendarBase.metadata.create_all(engine)
    VisitBriefBase.metadata.create_all(engine)  # test fixture, never runtime DDL
    with Session(engine) as db:
        from datetime import datetime
        stamp = datetime(2026, 10, 1)
        db.add(PatientORM(patient_id="outside-synthetic-scope", demographics_json={"full_name":"Protected sentinel"},
                          created_at=stamp, updated_at=stamp))
        db.commit()
    return application(engine), engine


def login(app):
    client = TestClient(app, base_url="https://testserver")
    assert client.post("/clinical/login", json={"key":KEY}).status_code == 200
    return client


def confirm(c, uid=BOOKINGS[1], patient=SYNTHETIC_161_PATIENT_IDS[1], revision=None):
    read = c.post(PREFIX + "/calendar/visit-brief/read", json={"booking_uid":uid})
    assert read.status_code == 200, read.text
    body = read.json()
    return c.post(PREFIX + "/calendar/visit-brief/confirm", json={
        "booking_uid":uid, "patient_id":patient,
        "expected_revision":body["revision"] if revision is None else revision,
        "independently_verified":True, "evidence_type":"protected_chart_identifiers",
        "observed_booking":body["booking"]["observation"],
    })


def test_key_required_for_every_test_data_and_write_route(setup):
    app, _ = setup
    c = TestClient(app, base_url="https://testserver")
    assert c.get(PREFIX + "/").status_code == 200  # empty public shell
    for path in ("/patients", "/recent-encounters", "/staff-session", "/calendar/cockpit-context",
                 "/patient/" + SYNTHETIC_161_PATIENT_IDS[0] + "/chart"):
        assert c.get(PREFIX + path).status_code == 401
    for path, body in (("/prepare", {}), ("/calendar/visit-brief/read", {"booking_uid":BOOKINGS[0]})):
        assert c.post(PREFIX + path, json=body).status_code == 401


def test_shared_transaction_persists_across_sessions_and_engine_restart(setup):
    app, engine = setup
    c = login(app)
    assert c.post(PREFIX + "/prepare").status_code == 200
    response = confirm(c)
    assert response.status_code == 200, response.text
    brief = response.json()["brief"]
    assert brief["previous_visit"]["date"] == "2026-10-07"
    assert "επανέλεγχος" in brief["prior_plan"]["decisions"][0]["text"]
    assert brief["pending"][0]["status"] == "awaiting_patient"
    engine.dispose()
    second = login(application(engine))
    r = second.post(PREFIX + "/calendar/visit-brief/read", json={"booking_uid":BOOKINGS[1]}).json()
    assert r["linked"] and r["revision"] == 1 and "brief" not in r
    assert second.post(PREFIX + "/prepare").status_code == 200
    stale = confirm(second, revision=0)
    assert stale.status_code == 409
    assert confirm(second).json()["revision"] == 2
    chart = second.get(PREFIX + "/patient/" + SYNTHETIC_161_PATIENT_IDS[1] + "/chart").json()
    assert len(chart["visits"]) == 2 and "SAFETY-NET" in chart["visits"][0]["detail"]
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(EncounterORM)) == 6
        link = db.get(BookingPatientLinkORM, ("cockpit-161-v1", BOOKINGS[1]))
        assert link.patient_id == SYNTHETIC_161_PATIENT_IDS[1]
        audits = db.scalars(select(VisitBriefStaffAuditORM)).all()
        assert audits and all(a.principal_id == "synthetic-key-test" and a.role == "synthetic_tester" for a in audits)


def test_real_data_and_provider_namespace_cannot_enter_test_route(setup):
    app, engine = setup
    c = login(app)
    c.post(PREFIX + "/prepare")
    assert confirm(c, patient="outside-synthetic-scope").status_code == 404
    for path in ("/patient/outside-synthetic-scope/chart", "/calendar/appointments", "/surgeries"):
        assert c.get(PREFIX + path).status_code == 404
    r = c.post(PREFIX + "/calendar/visit-brief/read", json={"booking_uid":"cal.com:real-provider"})
    assert r.status_code == 404
    for path in ("/patients?query=Protected", "/calendar/visit-brief/staff-audit"):
        response = c.get(PREFIX + path)
        assert "Protected sentinel" not in response.text
    assert c.get("/clinical/patients?query=Δοκιμαστικός").json() == []
    assert c.get("/clinical/recent-encounters").json() == []
    assert c.get("/clinical/patients?query=Protected").json()[0]["patient_id"] == "outside-synthetic-scope"
    # The existing key has NOT become a staff or live Visit Brief identity.
    assert c.get("/clinical/staff-session").status_code == 503
    assert c.post("/clinical/calendar/visit-brief/read", json={"booking_uid":BOOKINGS[1]}).status_code == 503
    context = c.get(PREFIX + "/calendar/cockpit-context").json()
    assert all(context[k]["appointment_id"] in BOOKINGS for k in ("previous","current","next"))


def test_seed_collision_never_overwrites_existing_patient(setup):
    app, engine = setup
    with Session(engine) as db:
        from datetime import datetime
        db.add(PatientORM(patient_id=SYNTHETIC_161_PATIENT_IDS[0], demographics_json={"full_name":"Protected collision"},
                          created_at=datetime(2026, 10, 1), updated_at=datetime(2026, 10, 1)))
        db.commit()
    c = login(app)
    assert c.post(PREFIX + "/prepare").status_code == 409
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(EncounterORM)) == 0
        assert db.get(PatientORM, SYNTHETIC_161_PATIENT_IDS[0]).demographics_json == {"full_name":"Protected collision"}


def test_real_browser_two_sessions_refresh_and_complete_chart(setup, tmp_path):
    import socket
    import shutil
    import threading
    import uvicorn
    from playwright.sync_api import sync_playwright, expect

    app, engine = setup
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets":[sock]}, daemon=True)
    thread.start()
    url = f"http://localhost:{port}" + PREFIX + "/"
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=shutil.which("google-chrome") or shutil.which("chromium"))
            for index, width in enumerate((1280, 680, 390)):
                # Independent browser contexts have no shared cookie or storage.
                context = browser.new_context(viewport={"width":width,"height":900})
                page = context.new_page()
                page.set_default_timeout(10000)
                errors, requests = [], []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.on("request", lambda r: requests.append(r.url))
                page.goto(url)
                page.locator("#ceGlobalKey").click()
                page.locator("#ceGlobalKeyInput").fill(KEY)
                page.locator("#ceGlobalLoginForm button[type=submit]").click()
                if index == 0:
                    page.locator("#synthetic161Prepare").click()
                expect(page.locator("#currentAppointmentPatient")).to_have_text("Δοκιμαστικός Ασθενής 161 2")
                page.locator("#visitSidebarCurrent").click()
                if index:
                    expect(page.locator("#visitBriefIdentity")).to_contain_text("προηγούμενη σύνδεση")
                page.locator("#visitBriefChoosePatient").click()
                page.locator("#visitPatientMatches .clinical-patient-row").click()
                page.locator("#visitBookingIndependentCheck").check()
                page.locator("#visitConfirmBooking").click()
                expect(page.locator("#visitBriefPrevious")).to_contain_text("2026-10-07")
                expect(page.locator("#visitBriefPlan")).to_contain_text("Συμφωνήθηκε επανέλεγχος")
                expect(page.locator("#visitBriefPending")).to_contain_text("awaiting_patient")
                page.locator("#visitBriefOpenChart").click()
                expect(page.locator("#synthetic161Chart")).to_contain_text("2 αποθηκευμένες επισκέψεις")
                expect(page.locator("#synthetic161Chart")).to_contain_text("SAFETY-NET")
                page.reload()
                expect(page.locator("#synthetic161Chart")).to_contain_text("2 αποθηκευμένες επισκέψεις")
                assert not errors, errors
                clinical = [u.split(f"localhost:{port}")[1] for u in requests if "/clinical/" in u]
                assert all(p.startswith(PREFIX + "/") or p in ("/clinical/status","/clinical/login") for p in clinical), clinical
                page.screenshot(path=str(tmp_path / f"synthetic-161-{width}.png"), full_page=True)
                if index == 0:
                    page.locator("#visitRecentRows .clinical-patient-row").first.click()
                    expect(page.locator("#synthetic161Chart")).to_be_empty()
                    expect(page.locator("#visitSelectedName")).to_have_text("Δοκιμαστικός Ασθενής 161 1")
                    assert "chart=" not in page.url
                context.close()
            browser.close()
        with Session(engine) as db:
            assert db.get(BookingPatientLinkORM, ("cockpit-161-v1", BOOKINGS[1])).revision == 3
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_two_browsers_prepare_concurrently_without_duplicate_or_reset(setup):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    app, engine = setup
    first, second = login(app), login(app)
    barrier = Barrier(2)
    def prepare(client):
        barrier.wait(timeout=5)
        return client.post(PREFIX + "/prepare").status_code
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(prepare, (first, second))) == [200, 200]
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(EncounterORM)) == 6


def test_live_booking_confirmation_cannot_link_to_synthetic_patient(monkeypatch):
    from test_clinical_visit_brief import client, confirm as live_confirm, NOW
    live, engine = client(monkeypatch)
    with Session(engine) as db:
        db.add(PatientORM(patient_id=SYNTHETIC_161_PATIENT_IDS[0],
                          demographics_json={"synthetic_scenario":"cockpit-161-v1"},
                          created_at=NOW, updated_at=NOW))
        db.commit()
    assert live_confirm(live, patient=SYNTHETIC_161_PATIENT_IDS[0]).status_code == 404
