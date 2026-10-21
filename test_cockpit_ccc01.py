"""CCC-01: protected read-only obligations and exact due attention; invented data only."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import socket
import threading
import shutil
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session
import pytest

from clinical_data import PatientORM, EncounterORM, ClinicalPendingORM
from clinical_capture_cp1 import TEST_ID
from test_cockpit_synthetic_161 import setup, login, KEY


def seed(engine):
    """130 synthetic active tasks, two namesakes, current/future/unknown, no writes through API."""
    now = datetime.now(ZoneInfo("Asia/Nicosia")).date()
    stamp = datetime(2026, 10, 20, 8)
    with Session(engine) as db:
        for n, pid in enumerate(("ccc-patient-a", "ccc-patient-b")):
            db.add(PatientORM(patient_id=pid, demographics_json={
                "full_name":"Αντωνίου Μαρία"}, created_at=stamp, updated_at=stamp))
            db.add(EncounterORM(
                id="ccc-enc-"+str(n), patient_id=pid, encounter_date="2026-10-20",
                status="completed", payload_json={},created_at=stamp,updated_at=stamp))
        for i in range(130):
            pid = "ccc-patient-a" if i % 2 == 0 else "ccc-patient-b"
            review = None
            if i == 0: review=now.isoformat()
            if i == 1: review=(now-timedelta(days=1)).isoformat()
            if i == 2: review=(now+timedelta(days=7)).isoformat()
            if i == 3: review="2026-00-99"  # must never become a due alert
            db.add(ClinicalPendingORM(
                id=f"ccc-item-{i:03d}",patient_id=pid,
                source_encounter_id="ccc-enc-"+str(i%2),
                item_type="test" if i%2==0 else "follow_up",
                description="Ανάλυση αίματος" if i%2==0 else "Επανεξέταση",
                trigger_text="Μόνο στην επόμενη επίσκεψη" if i >= 4 else None,
                status="awaiting_result" if i%2==0 else "open",
                responsible_role="clinician",external_dependency_json=None,
                review_date=review,provenance_json=["synthetic"],
                resolution_event_id=None,
                created_at=stamp,updated_at=stamp+timedelta(minutes=i),
            ))
        # Must not appear in active queue or badges.
        db.add(ClinicalPendingORM(
            id="ccc-resolved",patient_id="ccc-patient-b",source_encounter_id="ccc-enc-1",
            item_type="test",description="Ολοκληρωμένη ανάλυση",trigger_text=None,
            status="resolved",responsible_role="clinician",external_dependency_json=None,
            review_date=now.isoformat(),provenance_json=["synthetic"],
            resolution_event_id=None,created_at=stamp,updated_at=stamp))
        db.commit()
    return now


def test_backend_exact_due_date_filters_pagination_and_privacy(setup):
    app,engine=setup
    today=seed(engine)
    c=login(app)
    route="/clinical/clinical-updates"
    default=c.get(route+"?active_only=true")
    assert default.status_code==200,default.text
    body=default.json()
    assert body["total"]==130 and body["shown"]==100 and body["has_more"]
    assert body["next_offset"]==100 and body["today"]==today.isoformat()
    second=c.get(route+"?active_only=true&offset=100").json()
    assert len(second["items"])==30 and not second["has_more"]
    ids=[p["pending_id"] for p in body["items"]+second["items"]]
    assert len(set(ids))==130 and "ccc-resolved" not in ids
    assert all(x["patient_id"] in ("ccc-patient-a","ccc-patient-b") for x in body["items"])
    due=c.get(route+"?active_only=true&due=today&limit=4").json()
    assert due["total"]==2 and {p["pending_id"] for p in due["items"]}=={
        "ccc-item-000","ccc-item-001"}
    assert all(p["review_date"]<=today.isoformat() for p in due["items"])
    assert {p["pending_id"] for p in c.get(route+"?active_only=true&due=future").json()["items"]}=={"ccc-item-002"}
    undated=c.get(route+"?active_only=true&due=undated&sort=due").json()
    assert undated["total"]==127  # 126 unspecified + invalid / no today or future
    assert {p["pending_id"] for p in undated["items"]} & {"ccc-item-003"}
    assert c.get(route+"?active_only=true&status=resolved").json()["total"]==0
    assert c.get(route+"?status=resolved").json()["total"]==1
    assert c.get(route+"?active_only=true&category=test").json()["total"]==65
    search=c.get(route+"?active_only=true&query=αντωνιου").json()
    assert search["total"]==130,"Greek name accent fold must search across whole registry"
    assert c.get(route+"?active_only=true&query=ανάλυση").json()["total"]==65
    # Exact patient ID must remain separate even for identical full names.
    assert len({x["patient_id"] for x in search["items"]})==2
    assert c.get(route+"?scope=test").json()["items"]==[]
    for params in ("limit=101","offset=-1","due=unknown","sort=bad","status=wrong"):
        assert c.get(route+"?"+params).status_code==422,params
    assert TestClient(app).get(route+"?active_only=true&due=today").status_code==401
    with Session(engine) as db:
        assert sum(1 for _ in db.scalars(select(ClinicalPendingORM)))==131


def test_real_chromium_popup_full_screen_and_exact_pending(setup,tmp_path):
    import uvicorn
    from playwright.sync_api import sync_playwright,expect
    app,engine=setup;seed(engine)
    sock=socket.socket();sock.bind(("127.0.0.1",0));port=sock.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,log_level="error"))
    thread=threading.Thread(target=server.run,kwargs={"sockets":[sock]},daemon=True)
    thread.start()
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(executable_path=shutil.which("google-chrome") or shutil.which("chromium"),args=["--no-sandbox"])
            for width in (1280,680,390):
                context=browser.new_context(viewport={"width":width,"height":900})
                page=context.new_page();errors=[]
                page.on("pageerror",lambda e:errors.append(str(e)))
                page.goto(f"http://127.0.0.1:{port}/static/cockpit/index.html")
                page.locator("#ceGlobalKey").click()
                page.locator("#ceGlobalKeyInput").fill(KEY)
                page.locator("#ceGlobalLoginForm button[type=submit]").click()
                expect(page.locator("#ceGlobalAuthStatus")).to_contain_text("Συνδέθηκες")
                page.keyboard.press("Escape")
                expect(page.locator("#doctorTasksBadge")).to_have_text("2")
                page.locator("#doctorTasksButton").click()
                expect(page.locator("#doctorPopoverTitle")).to_have_text("Εργασίες σήμερα")
                expect(page.locator("#doctorPopoverContent .ccc-obligation-row")).to_have_count(2)
                assert "ccc-item-002" not in page.locator("#doctorPopoverContent").inner_text()
                page.locator(".ccc-open-full").click()
                expect(page.locator("#doctorLibraryView")).to_be_visible()
                expect(page.locator("[data-doctor-library-panel='obligations']")).to_be_visible()
                expect(page.locator("#cccObligationsList .ccc-obligation-row")).to_have_count(40)
                page.locator("#cccNext").click()
                expect(page.locator("#cccPage")).to_contain_text("Σελίδα 2")
                page.locator("#cccDue").select_option("future")
                expect(page.locator("#cccObligationsList .ccc-obligation-row")).to_have_count(1)
                expect(page.locator("#cccObligationsList")).to_contain_text("Ανάλυση αίματος")
                page.locator("#cccDue").select_option("undated")
                expect(page.locator("#cccObligationsList .ccc-obligation-row")).to_have_count(40)
                page.locator("#cccGroup").select_option("category")
                expect(page.locator("#cccObligationsList h3").first).to_be_visible()
                page.locator("#cccSearch").fill("Αντωνίου")
                expect(page.locator("#cccObligationsList .ccc-obligation-row")).to_have_count(40)
                page.locator("#cccDue").select_option("today")
                expect(page.locator("#cccObligationsList .ccc-obligation-row")).to_have_count(2)
                page.locator("#cccObligationsList .ccc-obligation-row").first.click()
                expect(page.locator("#doctorLibraryView")).to_be_hidden()
                expect(page.locator("#patientChart")).to_be_visible()
                expect(page.locator(".patient-pending-dialog")).to_be_visible()
                assert not errors,errors
                assert page.evaluate("document.documentElement.scrollWidth<=innerWidth+1")
                context.close()
            browser.close()
    finally:
        server.should_exit=True;thread.join(10);sock.close()
