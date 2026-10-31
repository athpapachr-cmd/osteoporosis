"""Persistent #161 boundary and real shared-backend tests. No external provider."""
from datetime import datetime
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


def assert_booking_uses_pending_projection(client, *, stage, source_valid):
    """The booking brief must repeat the canonical Pending projection, never raw events."""
    response = client.post(PREFIX + "/open-booking", json={"booking_uid": BOOKINGS[0]})
    assert response.status_code == 200, response.text
    brief = response.json()["brief"]
    assert brief["followthrough"] == [
        item["followthrough"] for item in brief["pending"]
        if "followthrough" in item
    ]
    assert len(brief["followthrough"]) == 1
    cycle = brief["followthrough"][0]
    assert cycle["stage"] == stage
    assert cycle["source_valid"] is source_valid
    if not source_valid:
        assert cycle["complete"] is False


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
    for path in ("/patients", "/clinical-updates", "/recent-encounters", "/staff-session", "/calendar/cockpit-context",
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
                expect(page.locator("#visitBookingConfirmation")).to_be_hidden()
                expect(page.locator("#visitBriefChoosePatient")).to_be_hidden()
                expect(page.locator("#visitBriefPrevious")).to_contain_text("2026-10-07")
                expect(page.locator("#visitBriefPlan")).to_contain_text("Συμφωνήθηκε επανέλεγχος")
                expect(page.locator("#visitBriefPending")).to_contain_text("Αναμένεται από τον ασθενή")
                page.locator("#visitBriefPending button",has_text="Παραλήφθηκε").click()
                expect(page.locator("#visitBriefPending .pending-action")).to_have_text("Αναίρεση")
                page.locator("#visitBriefOpenChart").click()
                expect(page.locator("#synthetic161Chart")).to_contain_text("2 αποθηκευμένες επισκέψεις")
                expect(page.locator("#synthetic161Chart")).to_contain_text("Πορεία ασθενούς")
                page.reload()
                expect(page.locator("#synthetic161Chart")).to_contain_text("2 αποθηκευμένες επισκέψεις")
                expect(page.locator("#synthetic161Chart .pending-action")).to_have_text("Αναίρεση")
                page.locator("#synthetic161Chart .pending-action").click()
                expect(page.locator("#synthetic161Chart .pending-action")).to_have_text("Παραλήφθηκε")
                assert not errors, errors
                clinical = [u.split(f"localhost:{port}")[1] for u in requests if "/clinical/" in u]
                assert all(p.startswith(PREFIX + "/") or p in ("/clinical/status","/clinical/login") for p in clinical), clinical
                page.locator("#synthetic161Chart .clinical-source-link").click()
                expect(page.locator(".patient-visit[open] .patient-obligation-origin")).to_contain_text("Υποχρεώσεις που δημιουργήθηκαν στην επίσκεψη")
                expect(page.locator(".patient-timeline-event")).not_to_have_count(0)
                page.locator("#doctorTasksButton").click()
                expect(page.locator("#doctorPopover")).to_be_visible()
                expect(page.locator("#doctorPopoverContent")).to_contain_text("Δοκιμαστικός Ασθενής 161")
                expect(page.locator("#doctorObligationsView")).to_be_hidden()
                page.locator("#doctorPopoverContent").get_by_role("button",name="Όλες οι υποχρεώσεις").click()
                expect(page.locator("#doctorObligationsView")).to_be_visible()
                expect(page.locator("#doctorObligationsTotal")).to_have_text("3")
                expect(page.locator("#doctorObligationsRows")).to_contain_text("Δοκιμαστικός Ασθενής 161 2")
                page.locator("#doctorInboxButton").click()
                expect(page.locator("#doctorPopover")).to_be_visible()
                expect(page.locator("#doctorPopoverContent")).to_contain_text("Συνθετική γνωμάτευση MRI")
                expect(page.locator("#doctorClinicalInboxView")).to_be_hidden()
                # Regression: a recent-item click must focus the actual synthetic
                # detail, not call the account <select> DOM element as a function.
                page.locator("#doctorPopoverContent").get_by_role(
                    "button",name="Συνθετική γνωμάτευση MRI").click()
                expect(page.locator("#doctorClinicalInboxView")).to_be_visible()
                expect(page.locator("#doctorClinicalInboxDetails")).to_contain_text("Συνθετική γνωμάτευση MRI")
                expect(page.locator('#doctorClinicalInboxRows [data-attention-id="SYN-ATT-MRI"]')).to_have_attribute(
                    "aria-pressed","true")
                assert not errors, errors
                page.locator("#doctorClinicalInboxBack").click()
                expect(page.locator("#doctorClinicalInboxDetails")).to_be_empty()
                page.locator("#doctorInboxButton").click()
                expect(page.locator("#doctorPopover")).to_be_visible()
                page.locator("#doctorPopoverContent").get_by_role("button",name="Όλα τα κλινικά εισερχόμενα").click()
                expect(page.locator("#doctorClinicalInboxView")).to_be_visible()
                expect(page.locator("#doctorClinicalInboxState")).to_contain_text("Καμία πραγματική")
                expect(page.locator("#doctorClinicalInboxRows [data-attention-id]")).to_have_count(4)
                page.locator("#doctorClinicalInboxAccount").select_option("syn-icloud")
                expect(page.locator("#doctorClinicalInboxRows [data-attention-id]")).to_have_count(1)
                expect(page.locator('[data-attention-id="SYN-ATT-LAB"]')).to_be_visible()
                page.locator("#doctorClinicalInboxAccount").select_option("")
                expect(page.locator("#doctorClinicalInboxRows [data-attention-id]")).to_have_count(4)
                expect(page.locator("#doctorInboxBadge")).to_be_hidden()
                page.locator('[data-attention-id="SYN-ATT-MIXED"]').click()
                expect(page.locator("#doctorClinicalInboxDetails")).to_contain_text("μόνο στο Inbox")
                expect(page.locator("#doctorClinicalInboxDetails form")).to_have_count(0)
                page.locator('[data-attention-id="SYN-ATT-GESY"]').click()
                expect(page.locator("#doctorClinicalInboxDetails")).to_contain_text("Αποτελέσματα διαθέσιμα")
                expect(page.locator("#doctorClinicalInboxDetails")).to_contain_text("παραπεμπτικό")
                expect(page.locator("#doctorClinicalInboxDetails form")).to_have_count(0)
                page.locator('[data-attention-id="SYN-ATT-LAB"]').click()
                expect(page.locator("#doctorClinicalInboxDetails")).to_contain_text("Δεν υπάρχει επαληθευμένη αντιστοίχιση")
                expect(page.locator("#doctorClinicalInboxDetails .clinical-inbox-link")).to_have_count(0)
                expect(page.locator("#doctorClinicalInboxDetails input")).to_have_count(0)
                if index == 0:
                    # A single tap visibly marks source review; never clinically
                    # accepts values or attaches the weak source to any patient.
                    page.locator("#doctorClinicalInboxDetails").get_by_role(
                        "button",name="Το είδα").click()
                    expect(page.locator('[data-attention-id="SYN-ATT-LAB"]')).to_contain_text("✓ Επισκοπήθηκε")
                    expect(page.locator("#doctorClinicalInboxDetails")).to_contain_text("✓ Το εισερχόμενο επισκοπήθηκε")
                    expect(page.locator("#doctorClinicalInboxDetails").get_by_role(
                        "button",name="Το είδα")).to_have_count(0)
                    assert all(x["item_id"]!="SYN-ATT-LAB" for x in
                        login(app).get(PREFIX+"/attention/brief/"+SYNTHETIC_161_PATIENT_IDS[0]).json()["items"])
                page.locator('[data-attention-id="SYN-ATT-MRI"]').click()
                # A previous item's success must not appear under a new source.
                expect(page.locator("#doctorClinicalInboxFeedback")).to_have_text("")
                if index == 0:
                    details=page.locator("#doctorClinicalInboxDetails")
                    expect(details).to_contain_text("Ασθενής: Δοκιμαστικός Ασθενής 161 2")
                    expect(details.locator(".clinical-inbox-link input")).to_have_count(0)
                    expect(details.get_by_role("button",name="Επιβεβαίωση ασθενούς")).to_be_enabled()
                    details.get_by_role("button",name="Επιβεβαίωση ασθενούς").click()
                    expect(details).to_contain_text(SYNTHETIC_161_PATIENT_IDS[1])
                    details.get_by_role("button",name="Το είδα",exact=True).click()
                    expect(details).to_contain_text("✓ Το εισερχόμενο επισκοπήθηκε")
                    expect(page.locator('[data-attention-id="SYN-ATT-MRI"]')).to_contain_text("✓ Επισκοπήθηκε")
                page.locator("#doctorClinicalInboxBack").click()
                expect(page.locator("#doctorClinicalInboxView")).to_be_hidden()
                expect(page.locator("#doctorClinicalInboxDetails")).to_be_empty()
                page.locator("#doctorTasksButton").click()
                expect(page.locator("#doctorPopover")).to_be_visible()
                page.locator("#doctorPopoverContent").get_by_role("button",name="Όλες οι υποχρεώσεις").click()
                page.locator("#doctorObligationsRows .doctor-obligation-group").filter(
                    has_text="Δοκιμαστικός Ασθενής 161 2").locator(".doctor-obligation-row").first.click()
                expect(page.locator("#visitSelectedName")).to_have_text("Δοκιμαστικός Ασθενής 161 2")
                expect(page.locator("#doctorObligationsView")).to_be_hidden()
                expect(page.locator("#doctorPopover")).to_be_hidden()
                expect(page.locator("#synthetic161Chart .patient-chart-grid")).to_be_visible()
                expect(page.locator("#synthetic161Chart")).to_contain_text("Δημιουργήθηκε υποχρέωση")
                page.screenshot(path=str(tmp_path / f"synthetic-161-{width}.png"), full_page=True)
                if index == 0:
                    page.locator("#visitRecentRows .clinical-patient-row").first.click()
                    expect(page.locator("#synthetic161Chart")).to_contain_text("2 αποθηκευμένες επισκέψεις")
                    expect(page.locator("#visitSelectedName")).to_have_text("Δοκιμαστικός Ασθενής 161 1")
                    assert "chart=cockpit-synthetic-161-1" in page.url
                    # A delayed 401 from the old detail read may not log out
                    # the newly authenticated synthetic clinician.
                    page.locator("#doctorInboxButton").click()
                    expect(page.locator("#doctorPopover")).to_be_visible()
                    page.locator("#doctorPopoverContent").get_by_role("button",name="Όλα τα κλινικά εισερχόμενα").click()
                    expect(page.locator("#doctorClinicalInboxView")).to_be_visible()
                    page.evaluate("""() => {
                        window.__savedFetch=window.fetch;
                        window.__staleAttentionLogout=0;
                        window.addEventListener("clinical:unauthenticated",
                            ()=>window.__staleAttentionLogout++);
                        window.__releaseLateAttention401=null;
                        window.fetch=(url,opts)=>{
                            if(String(url).endsWith("/attention/SYN-ATT-LAB"))
                                return new Promise(resolve=>{
                                    window.__releaseLateAttention401=()=>
                                        resolve(new Response(null,{status:401}));
                                });
                            return window.__savedFetch(url,opts);
                        };
                    }""")
                    page.locator('[data-attention-id="SYN-ATT-LAB"]').click()
                    page.wait_for_function("typeof window.__releaseLateAttention401==='function'")
                    page.evaluate('window.dispatchEvent(new Event("clinical:authenticated"))')
                    expect(page.locator("#doctorClinicalInboxRows [data-attention-id]")).to_have_count(4)
                    page.evaluate("""async () => {
                        window.__releaseLateAttention401();
                        await new Promise(resolve=>setTimeout(resolve,30));
                    }""")
                    assert page.evaluate("window.__staleAttentionLogout")==0
                    expect(page.locator("#doctorClinicalInboxView")).to_be_visible()
                    # A genuine CURRENT 401 on detail must purge patient-facing
                    # UI and request the existing clinical login.
                    page.evaluate("""() => {
                        window.fetch=(url,opts)=>
                            String(url).endsWith("/attention/SYN-ATT-MIXED")?
                                Promise.resolve(new Response(null,{status:401})):
                                window.__savedFetch(url,opts);
                    }""")
                    page.locator('[data-attention-id="SYN-ATT-MIXED"]').click()
                    expect(page.locator("#doctorClinicalInboxView")).to_be_hidden()
                    expect(page.locator("#doctorClinicalInboxDetails")).to_be_empty()
                context.close()
            browser.close()
        with Session(engine) as db:
            assert db.get(BookingPatientLinkORM, ("cockpit-161-v1", BOOKINGS[1])) is None
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


def test_one_click_fixture_and_pending_action_persist_with_conflict_protection(setup):
    from clinical_data import ClinicalPendingORM
    app, engine = setup
    c = login(app)
    c.post(PREFIX + "/prepare")
    body = {"booking_uid": BOOKINGS[1]}
    route = PREFIX + "/open-booking"
    assert TestClient(app).post(route, json=body).status_code == 401
    assert c.post(route, json={"booking_uid":"external-booking"}).status_code == 404
    data = c.post(route, json=body).json()
    assert data["mode"] == "synthetic_fixture"
    assert data["brief"]["patient_id"] == SYNTHETIC_161_PATIENT_IDS[1]
    pending = data["brief"]["pending"][0]
    path = PREFIX + "/patient/" + data["brief"]["patient_id"] + "/pending/" + pending["pending_id"]
    payload = {"status":"resolved", "expected_updated_at":pending["updated_at"]}
    assert TestClient(app).post(path, json=payload).status_code == 401
    assert c.post(path.replace(SYNTHETIC_161_PATIENT_IDS[1], "outside-synthetic-scope"),json=payload).status_code == 404
    saved = c.post(path, json=payload)
    assert saved.status_code == 200
    assert c.post(path, json=payload).status_code == 409
    engine.dispose()
    second = login(application(engine))
    assert second.post(route,json=body).json()["brief"]["pending"][0]["status"] == "resolved"
    assert second.post(PREFIX+"/prepare").status_code == 200
    assert second.post(route,json=body).json()["brief"]["pending"][0]["status"] == "resolved"
    undo = {"status":"awaiting_patient", "expected_updated_at":saved.json()["updated_at"]}
    assert second.post(path,json=undo).json()["status"] == "awaiting_patient"
    with Session(engine) as db:
        audit = db.scalars(select(VisitBriefStaffAuditORM).where(VisitBriefStaffAuditORM.event.like("synthetic_pending_%"))).all()
        assert len(audit) == 2 and all(a.role == "synthetic_tester" for a in audit)
        row = db.get(ClinicalPendingORM, pending["pending_id"])
        assert row.source_encounter_id.endswith("-visit-10")


def test_obligation_origin_and_updates_share_persisted_history(setup):
    app, engine = setup
    c = login(app)
    c.post(PREFIX + "/prepare")
    route = PREFIX + "/patient/" + SYNTHETIC_161_PATIENT_IDS[1] + "/chart"
    chart = c.get(route).json()
    pending = chart["pending"][0]
    origin = next(v for v in chart["visits"] if v["encounter_id"] == pending["source_encounter_id"])
    assert pending["source_date"] == origin["date"] == "2026-10-07"
    assert origin["clinical"]["pending"][0]["description"] == pending["description"]
    assert pending["history"][0]["kind"] == "created"
    updates = c.get(PREFIX + "/clinical-updates").json()
    assert updates["open_count"] == 3
    assert all(p["patient_id"] in SYNTHETIC_161_PATIENT_IDS for p in updates["items"])
    path = PREFIX + "/patient/" + chart["patient_id"] + "/pending/" + pending["pending_id"]
    received = c.post(path, json={"status":"resolved", "expected_updated_at":pending["updated_at"]}).json()
    assert [e["kind"] for e in received["history"]] == ["created", "received"]
    reopened = c.post(path, json={"status":"awaiting_patient", "expected_updated_at":received["updated_at"]}).json()
    second = login(application(engine))
    item = next(p for p in second.get(PREFIX + "/clinical-updates").json()["items"] if p["patient_id"] == chart["patient_id"])
    assert item["history"] == second.get(route).json()["pending"][0]["history"] == reopened["history"]
    assert [e["kind"] for e in item["history"]] == ["created", "received", "reopened"]
    assert second.get(route).json()["visits"] == chart["visits"]  # source is never rewritten


def test_browser_eligible_patient_lookup_401_is_current_session_only(setup):
    """Real #161 browser: superseded 401 ignored, current 401 purges and asks for login."""
    import shutil
    import socket
    import threading
    import uvicorn
    from playwright.sync_api import sync_playwright, expect

    app, engine = setup
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=shutil.which("google-chrome") or shutil.which("chromium"))
            context = browser.new_context(viewport={"width": 680, "height": 900})
            page = context.new_page()
            page.set_default_timeout(10000)
            page.goto(f"http://localhost:{port}" + PREFIX + "/")
            page.locator("#ceGlobalKey").click()
            page.locator("#ceGlobalKeyInput").fill(KEY)
            page.locator("#ceGlobalLoginForm button[type=submit]").click()
            page.locator("#synthetic161Prepare").click()
            expect(page.locator("#synthetic161Status")).to_contain_text("Η δοκιμή αποθηκεύτηκε")
            page.locator("#doctorInboxButton").click()
            expect(page.locator("#doctorPopover")).to_be_visible()
            page.locator("#doctorPopoverContent").get_by_role("button",name="Όλα τα κλινικά εισερχόμενα").click()
            expect(page.locator("#doctorClinicalInboxRows [data-attention-id]")).to_have_count(4)
            page.evaluate("""() => {
                window.__unauthedEligibleCount = 0;
                window.addEventListener("clinical:unauthenticated",
                    () => window.__unauthedEligibleCount++);
                window.__realEligibleFetch = window.fetch.bind(window);
                window.fetch = (url,opts) =>
                    String(url).endsWith("/attention/eligible-patients")
                    ? new Promise(resolve => {
                        window.__releaseOldEligible401 = () =>
                            resolve(new Response(null,{status:401}));
                      })
                    : window.__realEligibleFetch(url,opts);
            }""")
            page.locator('[data-attention-id="SYN-ATT-MRI"]').click()
            page.wait_for_function('typeof window.__releaseOldEligible401==="function"')
            expect(page.locator("#doctorClinicalInboxDetails .clinical-inbox-link")).to_have_count(1)
            # Submit the ACTUAL protected login, which replaces the in-flight
            # source context; dispatching a fabricated auth event is insufficient.
            if page.locator("#ceGlobalAccessPanel").is_hidden():
                page.locator("#ceGlobalKey").click()
            page.locator("#ceGlobalKeyInput").fill(KEY)
            page.locator("#ceGlobalLoginForm button[type=submit]").click()
            expect(page.locator("#ceGlobalAuthStatus")).to_contain_text("Συνδέθηκες")
            expect(page.locator("#doctorClinicalInboxRows [data-attention-id]")).to_have_count(4)
            page.evaluate("""async () => {
                window.__releaseOldEligible401();
                await new Promise(resolve => setTimeout(resolve, 50));
            }""")
            assert page.evaluate("window.__unauthedEligibleCount") == 0
            expect(page.locator("#doctorClinicalInboxView")).to_be_visible()
            expect(page.locator("#doctorClinicalInboxDetails")).to_be_empty()

            # A 401 from the CURRENT eligible-patients read must immediately
            # remove all protected rows and forms and show the existing login.
            page.evaluate("""() => {
                window.fetch = (url,opts) =>
                    String(url).endsWith("/attention/eligible-patients")
                    ? Promise.resolve(new Response(null,{status:401}))
                    : window.__realEligibleFetch(url,opts);
            }""")
            page.locator('[data-attention-id="SYN-ATT-MRI"]').click()
            expect(page.locator("#doctorClinicalInboxView")).to_be_hidden()
            expect(page.locator("#doctorClinicalInboxDetails")).to_be_empty()
            expect(page.locator("#doctorClinicalInboxRows")).to_be_empty()
            expect(page.locator("#ceGlobalAccessPanel")).to_be_visible()
            assert page.evaluate("window.__unauthedEligibleCount") == 1
            context.close()
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_followthrough_demo_rejects_closed_or_unbound_signed_pending(setup):
    app, engine = setup
    c = login(app)
    assert c.post(PREFIX + "/prepare").status_code == 200
    from clinical_data import ClinicalPendingORM
    patient_id = SYNTHETIC_161_PATIENT_IDS[0]
    pending_id = patient_id + "-pending"
    with Session(engine) as db, db.begin():
        pending = db.get(ClinicalPendingORM, pending_id)
        pending.status = "resolved"
    assert c.post(PREFIX + "/followthrough/prepare").status_code == 409
    with Session(engine) as db, db.begin():
        pending = db.get(ClinicalPendingORM, pending_id)
        pending.status = "awaiting_patient"
        encounter = db.get(EncounterORM, pending.source_encounter_id)
        payload = dict(encounter.payload_json)
        signed = dict(payload["_visit_capture_v1"])
        signed["pending_ids"] = []
        payload["_visit_capture_v1"] = signed
        encounter.payload_json = payload
    assert c.post(PREFIX + "/followthrough/prepare").status_code == 409


def test_followthrough_revocation_after_review_blocks_actions_and_clears_live_projection(setup):
    app, engine = setup
    c = login(app)
    assert c.post(PREFIX + "/prepare").status_code == 200
    ready = c.post(PREFIX + "/followthrough/prepare")
    assert ready.status_code == 200, ready.text
    signed = next(x for x in ready.json()["cases"] if x["mode"] == "signed_pending")
    other = next(x for x in ready.json()["cases"] if x["mode"] == "provisional_visit")
    case_id = signed["case_id"]
    route = PREFIX + "/followthrough"

    def action(path, no, **attrs):
        current = next(x for x in c.get(route).json()["cases"] if x["case_id"] == case_id)
        return c.post(route + path, json={
            "case_id": case_id, "expected_version": current["version"],
            "idempotency_key": f"synthetic-revocation-event-{no:04}",
            **attrs,
        })

    assert action("/simulate-arrival", 1, part="report").status_code == 200
    assert action("/doctor-review", 2, part="report", choice="notify").status_code == 200
    assert c.get(route + "/secretary/" + case_id).status_code == 200
    assert_booking_uses_pending_projection(c, stage="partially_received", source_valid=True)
    original = c.get(PREFIX + "/patient/" + SYNTHETIC_161_PATIENT_IDS[0] + "/chart").json()
    with Session(engine) as db, db.begin():
        from clinical_data import ClinicalPendingORM
        db.get(ClinicalPendingORM, SYNTHETIC_161_PATIENT_IDS[0] + "-pending").status = "cancelled"

    state = next(x for x in c.get(route).json()["cases"] if x["case_id"] == case_id)
    assert state["source_valid"] is False
    assert state["stage"] == "source_invalidated" and not state["complete"]
    assert state["version"] == 3
    assert_booking_uses_pending_projection(c, stage="source_invalidated", source_valid=False)
    assert c.get(route + "/secretary/" + case_id).status_code == 409
    for path, attrs in (
        ("/simulate-arrival", {"part": "images"}),
        ("/doctor-review", {"part": "report", "choice": "notify"}),
        ("/secretary-result", {"choice": "notified"}),
        ("/doctor-callback", {}),
    ):
        assert action(path, 20 + len(path), **attrs).status_code == 409, path
    after = c.get(PREFIX + "/patient/" + SYNTHETIC_161_PATIENT_IDS[0] + "/chart").json()
    assert after["visits"] == original["visits"], "Signed narrative must remain unchanged"
    assert after["pending"][0]["followthrough"]["source_valid"] is False
    assert after["pending"][0]["followthrough"] == after["followthrough"][0]
    assert after["followthrough"][0]["stage"] == "source_invalidated"
    assert after["pending"][0]["status"] == "cancelled"
    assert c.get(PREFIX + "/clinical-updates").json()["open_count"] == 2
    assert next(x for x in c.get(route).json()["cases"] if x["case_id"] == case_id)["version"] == 3
    # The independent no-chart obligation remains actionable.
    response = c.post(route + "/simulate-arrival", json={
        "case_id": other["case_id"], "expected_version": other["version"],
        "idempotency_key": "synthetic-provisional-after-revoke-01",
        "part": "labs",
    })
    assert response.status_code == 200 and response.json()["stage"] == "awaiting_doctor_review"


def test_followthrough_cancel_then_reopen_remains_suspended(setup):
    app, engine = setup
    c = login(app)
    assert c.post(PREFIX + "/prepare").status_code == 200
    ready = c.post(PREFIX + "/followthrough/prepare").json()
    case_id = next(x["case_id"] for x in ready["cases"] if x["mode"] == "signed_pending")
    route = PREFIX + "/followthrough"
    current = next(x for x in c.get(route).json()["cases"] if x["case_id"] == case_id)
    def action(path, serial, **extra):
        return c.post(route + path, json={
            "case_id": case_id, "expected_version": current["version"] + serial - 1,
            "idempotency_key": f"synthetic-reopen-guard-{serial:04}", **extra,
        })
    assert action("/simulate-arrival", 1, part="report").status_code == 200
    assert action("/doctor-review", 2, part="report", choice="notify").status_code == 200
    assert c.get(route + "/secretary/" + case_id).status_code == 200
    patient_id = SYNTHETIC_161_PATIENT_IDS[0]
    with Session(engine) as db, db.begin():
        from clinical_data import ClinicalPendingORM
        row = db.get(ClinicalPendingORM, patient_id + "-pending")
        row.status = "cancelled"
        row.updated_at = datetime(2026, 10, 31, 12)
    assert c.get(route + "/secretary/" + case_id).status_code == 409
    assert_booking_uses_pending_projection(c, stage="source_invalidated", source_valid=False)
    with Session(engine) as db, db.begin():
        row = db.get(ClinicalPendingORM, patient_id + "-pending")
        row.status = "awaiting_patient"
        row.updated_at = datetime(2026, 10, 31, 13)
    reopened = next(x for x in c.get(route).json()["cases"] if x["case_id"] == case_id)
    assert reopened["source_valid"] is False and reopened["stage"] == "source_invalidated"
    assert_booking_uses_pending_projection(c, stage="source_invalidated", source_valid=False)
    assert c.get(route + "/secretary/" + case_id).status_code == 409
    assert action("/secretary-result", 3, choice="notified").status_code == 409
    chart = c.get(PREFIX + "/patient/" + patient_id + "/chart").json()
    assert chart["followthrough"][0] == chart["pending"][0]["followthrough"]
    assert chart["followthrough"][0]["complete"] is False
    assert chart["followthrough"][0]["source_valid"] is False
    assert c.get(PREFIX + "/clinical-updates").json()["open_count"] == 3


def test_followthrough_joint_patient_transfer_is_denied(setup):
    app, engine = setup
    c = login(app)
    assert c.post(PREFIX + "/prepare").status_code == 200
    ready = c.post(PREFIX + "/followthrough/prepare").json()
    signed = next(x for x in ready["cases"] if x["mode"] == "signed_pending")
    original, other = SYNTHETIC_161_PATIENT_IDS[:2]
    with Session(engine) as db, db.begin():
        from clinical_data import ClinicalPendingORM
        pending = db.get(ClinicalPendingORM, original + "-pending")
        visit = db.get(EncounterORM, original + "-visit-10")
        pending.patient_id = other
        visit.patient_id = other
    route = PREFIX + "/followthrough"
    case_id = signed["case_id"]
    state = next(x for x in c.get(route).json()["cases"] if x["case_id"] == case_id)
    assert state["source_valid"] is False
    assert c.get(route + "/secretary/" + case_id).status_code == 409
    assert c.post(route + "/doctor-callback", json={
        "case_id": case_id, "expected_version": state["version"],
        "idempotency_key": "synthetic-reassigned-patient-01",
    }).status_code == 409


def test_followthrough_amended_signed_membership_requires_reconciliation(setup):
    app, engine = setup
    c = login(app)
    assert c.post(PREFIX + "/prepare").status_code == 200
    ready = c.post(PREFIX + "/followthrough/prepare").json()
    case_id = next(x["case_id"] for x in ready["cases"] if x["mode"] == "signed_pending")
    with Session(engine) as db, db.begin():
        patient_id = SYNTHETIC_161_PATIENT_IDS[0]
        visit = db.get(EncounterORM, patient_id + "-visit-10")
        payload = dict(visit.payload_json)
        signed = dict(payload["_visit_capture_v1"])
        signed["pending_ids"] = []
        signed["candidate"] = {**signed["candidate"], "pending": []}
        payload["_visit_capture_v1"] = signed
        visit.payload_json = payload
        visit.status = "amended"
    state = next(x for x in c.get(PREFIX + "/followthrough").json()["cases"] if x["case_id"] == case_id)
    assert state["source_valid"] is False and state["stage"] == "source_invalidated"
    assert_booking_uses_pending_projection(c, stage="source_invalidated", source_valid=False)
    pending = c.get(PREFIX + "/patient/" + SYNTHETIC_161_PATIENT_IDS[0] + "/chart").json()["pending"][0]
    assert pending["status"] == "awaiting_patient"
    assert pending["followthrough"]["source_valid"] is False
    assert c.get(PREFIX + "/clinical-updates").json()["open_count"] == 3
    assert c.post(PREFIX + "/followthrough/doctor-callback", json={
        "case_id": case_id, "expected_version": state["version"],
        "idempotency_key": "synthetic-amended-source-guard-01",
    }).status_code == 409


def test_followthrough_synthetic_full_result_review_secretary_and_closure(setup):
    """Original signed Pending, and no-chart visit, share one event semantics."""
    app, _engine = setup
    anonymous = TestClient(app, base_url="https://testserver")
    assert anonymous.get(PREFIX + "/followthrough").status_code == 401
    assert anonymous.post(PREFIX + "/followthrough/prepare").status_code == 401

    c = login(app)
    before = c.post(PREFIX + "/followthrough/prepare")
    assert before.status_code == 409, before.text
    assert c.post(PREFIX + "/prepare").status_code == 200
    ready = c.post(PREFIX + "/followthrough/prepare")
    assert ready.status_code == 200, ready.text
    payload = ready.json()
    assert payload["mode"] == "synthetic_only"
    assert payload["live_provider_connected"] is False
    assert payload["real_messages_used"] is False
    assert len(payload["cases"]) == 2

    signed = next(x for x in payload["cases"] if x["mode"] == "signed_pending")
    provisional = next(x for x in payload["cases"] if x["mode"] == "provisional_visit")
    assert signed["pending_ref"] == SYNTHETIC_161_PATIENT_IDS[0] + "-pending"
    assert signed["verified_contact"] is True
    assert provisional["pending_ref"] is None
    assert provisional["verified_contact"] is False
    ref = signed["case_id"]
    route = PREFIX + "/followthrough"

    def apply(path, case, serial, **attrs):
        current = c.get(route).json()["cases"]
        version = next(x["version"] for x in current if x["case_id"] == case)
        response = c.post(route + path, json={
            "case_id": case, "expected_version": version,
            "idempotency_key": f"synthetic-cycle-api-{serial:04}",
            **attrs
        })
        assert response.status_code == 200, response.text
        return response.json()

    assert apply("/simulate-arrival", ref, 1, part="report")["stage"] == "partially_received"
    reviewed = apply("/doctor-review", ref, 2, part="report", choice="notify")
    assert reviewed["reviewed"]["report"] == "revision-01"
    secretary = c.get(route + "/secretary/" + ref)
    assert secretary.status_code == 200
    assert "source_ref" not in secretary.text
    assert "report" not in secretary.text
    assert secretary.json()["status"] == "to_notify"

    signed_chart = c.get(PREFIX + "/patient/" + SYNTHETIC_161_PATIENT_IDS[0] + "/chart")
    assert signed_chart.status_code == 200, signed_chart.text
    assert signed_chart.json()["followthrough"][0]["case_id"] == ref
    assert signed_chart.json()["followthrough"][0]["reviewed"]["report"] == "revision-01"
    before_visits = signed_chart.json()["visits"]

    assert apply("/simulate-arrival", ref, 3, part="images")["stage"] == "awaiting_doctor_review"
    assert apply("/doctor-review", ref, 4, part="images", choice="no_contact")["stage"] == "awaiting_secretary"
    contacted = apply("/secretary-result", ref, 5, choice="attempted")
    assert contacted["contact_status"] == "attempted"
    informed = apply("/secretary-result", ref, 6, choice="notified")
    assert informed["stage"] == "awaiting_callback" and not informed["complete"]
    closed = apply("/doctor-callback", ref, 7)
    assert closed["stage"] == "complete"

    after_chart = c.get(PREFIX + "/patient/" + SYNTHETIC_161_PATIENT_IDS[0] + "/chart").json()
    assert after_chart["visits"] == before_visits, "Signed medical narrative never rewritten"
    assert after_chart["followthrough"][0]["stage"] == "complete"

    other = provisional["case_id"]
    assert apply("/simulate-arrival", other, 8, part="labs")["stage"] == "awaiting_doctor_review"
    completed = apply("/doctor-review", other, 9, part="labs", choice="no_contact")
    assert completed["complete"] and completed["pending_ref"] is None
    assert c.get(route + "/secretary/" + other).status_code == 403
    assert c.post(route + "/followthrough/prepare").status_code == 404


def test_followthrough_revoked_browser_shows_no_stale_task_or_completion(setup):
    import shutil
    import socket
    import threading
    import uvicorn
    from playwright.sync_api import sync_playwright, expect
    from clinical_data import ClinicalPendingORM

    app, engine = setup
    c = login(app)
    assert c.post(PREFIX + "/prepare").status_code == 200
    assert c.post(PREFIX + "/followthrough/prepare").status_code == 200
    with Session(engine) as db, db.begin():
        db.get(ClinicalPendingORM, SYNTHETIC_161_PATIENT_IDS[0] + "-pending").status = "cancelled"

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                executable_path=shutil.which("google-chrome") or shutil.which("chromium"))
            ctx = browser.new_context(viewport={"width": 1280, "height": 900})
            page = ctx.new_page()
            page.set_default_timeout(12000)
            page.goto(f"http://127.0.0.1:{port}" + PREFIX + "/")
            page.locator("#ceGlobalKey").click()
            page.locator("#ceGlobalKeyInput").fill(KEY)
            page.locator("#ceGlobalLoginForm button[type=submit]").click()
            expect(page.locator("#syntheticFollowthroughDemo")).to_be_visible()
            expect(page.locator(".synthetic-cycle-card").first).to_contain_text(
                "Η υποχρέωση άλλαξε στην επίσκεψη")
            expect(page.locator(".synthetic-cycle-card").first.locator(".synthetic-cycle-action")).to_have_count(0)
            # Provisional visit remains actionable with revoked signed source.
            expect(page.locator(".synthetic-cycle-card").nth(1).locator(".synthetic-cycle-action")).to_have_count(1)
            page.evaluate("(patientId) => window.CockpitStoredChart.open(patientId)",
                          SYNTHETIC_161_PATIENT_IDS[0])
            expect(page.locator("#synthetic161Chart .patient-obligation-origin")).to_contain_text(
                "Τρέχουσα κατάσταση: Ακυρώθηκε")
            expect(page.locator("#synthetic161Chart")).to_contain_text("0 ανοικτές")
            assert page.evaluate(
                "document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1")
            ctx.close()
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_followthrough_synthetic_browser_one_calm_card(setup):
    """Synthetic-only real Chromium: doctor, staff simulation and no-chart case."""
    import shutil
    import socket
    import threading
    import uvicorn
    from playwright.sync_api import sync_playwright, expect

    app, _engine = setup
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        with sync_playwright() as pw:
            for width in (1280, 680, 390):
                browser = pw.chromium.launch(
                    executable_path=shutil.which("google-chrome") or shutil.which("chromium")
                )
                ctx = browser.new_context(viewport={"width": width, "height": 900})
                page = ctx.new_page()
                page.set_default_timeout(12000)
                errors = []
                page.on("pageerror", lambda err: errors.append(str(err)))
                page.goto(f"http://127.0.0.1:{port}" + PREFIX + "/")
                # Optional demo's anonymous bootstrap must NEVER steal
                # login focus or open the shared Clinical Key panel.
                expect(page.locator("#ceGlobalAccessPanel")).to_be_hidden()
                page.locator("#ceGlobalKey").click()
                expect(page.locator("#ceGlobalAccessPanel")).to_be_visible()
                page.locator("#ceGlobalKeyInput").fill(KEY)
                page.locator("#ceGlobalLoginForm button[type=submit]").click()
                page.locator("#synthetic161Prepare").click()
                expect(page.locator("#syntheticFollowthroughDemo")).to_be_visible()
                expect(page.locator(".synthetic-cycle-card")).to_have_count(2)

                if width == 1280:
                    page.get_by_role("button", name="Δοκιμή παραλαβής · γνωμάτευση").click()
                    page.get_by_role("button", name="Είδα τη γνωμάτευση — ενημέρωση ασθενούς").click()
                    page.get_by_role("button", name="Δοκιμή παραλαβής · εικόνες MRI").click()
                    page.get_by_role("button", name="Είδα εικόνες MRI — χωρίς επικοινωνία").click()
                    page.get_by_role("button", name="Γραμματεία · ο ασθενής ενημερώθηκε").click()
                    page.get_by_role("button", name="Έγινε η επικοινωνία με τον ιατρό").click()
                    page.get_by_role("button", name="Δοκιμή παραλαβής · αναλύσεις").click()
                    page.get_by_role("button", name="Είδα αναλύσεις — χωρίς επικοινωνία").click()
                expect(page.locator(".synthetic-cycle-card").first).to_contain_text("Ολοκληρώθηκε")
                expect(page.locator(".synthetic-cycle-card").nth(1)).to_contain_text("Ολοκληρώθηκε")
                if width == 1280:
                    # Same event-derived status in signed visit and open count.
                    page.evaluate(
                        "(patientId) => window.CockpitStoredChart.open(patientId)",
                        SYNTHETIC_161_PATIENT_IDS[0],
                    )
                    expect(page.locator("#synthetic161Chart .patient-obligation-origin")).to_contain_text(
                        "Τρέχουσα κατάσταση: Ολοκληρώθηκε"
                    )
                    expect(page.locator("#synthetic161Chart")).to_contain_text("0 ανοικτές")
                assert page.evaluate(
                    "document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1"
                ), f"Horizontal overflow at {width}"
                assert errors == [], errors
                ctx.close()
                browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=5)
