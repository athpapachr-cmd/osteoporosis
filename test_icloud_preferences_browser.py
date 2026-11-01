"""Actual production assets + live local API/DB and invented IMAP only."""
import json
from pathlib import Path
import shutil
import socket
from threading import Thread
import time

import pytest
from fastapi import Depends
from fastapi.staticfiles import StaticFiles
from playwright.sync_api import sync_playwright, expect
import uvicorn

from clinical_reception_calls import _require_clinical_key
from test_icloud_preferences import store, api
from test_clinical_icloud_mail import client as source_client, mail


@pytest.fixture()
def server(api):
    client, fake = api
    app = client.app
    @app.get("/clinical/status", dependencies=[Depends(_require_clinical_key)])
    def status():
        return {"status": "ok"}
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    runner = uvicorn.Server(uvicorn.Config(app, log_level="critical"))
    thread = Thread(target=runner.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    for _ in range(200):
        if runner.started:
            break
        time.sleep(.01)
    assert runner.started
    try:
        yield "http://127.0.0.1:" + str(sock.getsockname()[1]), fake
    finally:
        runner.should_exit = True
        thread.join(timeout=5)
        sock.close()


@pytest.mark.parametrize("width", [1280, 390])
def test_visible_sort_rule_undo_and_revision_coherence(server, width, tmp_path):
    url, fake = server
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=shutil.which("google-chrome") or shutil.which("chromium") or None)
        context = browser.new_context(viewport={"width": width, "height": 1000},
            extra_http_headers={"X-Clinical-Key": "synthetic-clinical-key"})
        external, mutations, errors = [], [], []
        context.on("request", lambda r: external.append(r.url) if not r.url.startswith(url) else None)
        context.on("request", lambda r: mutations.append(r.url) if r.method not in ("GET", "HEAD", "OPTIONS") else None)
        page = context.new_page()
        page.set_default_timeout(5000)
        page.on("pageerror", lambda e: errors.append(str(e)))
        # Other Cockpit sources are outside this test. Only the triage API is real.
        def other_sources(route):
            path = route.request.url.split(url)[-1]
            if path.startswith("/clinical/icloud-mail") or path == "/clinical/status":
                route.continue_()
            elif "staff-session" in path:
                route.fulfill(status=401, json={})
            else:
                route.fulfill(json=[])
        context.route(url + "/clinical/**", other_sources)
        page.add_init_script("""(() => {
          const nativeFetch=window.fetch.bind(window);
          window.fetch=async (input,options={})=>{
            const u=new URL(typeof input==='string'?input:input.url,location.href);
            const label=u.pathname==='/clinical/icloud-mail/peek'?'Peek':u.pathname==='/clinical/icloud-mail'?'List':null;
            if(label&&window['__hold'+label]){
              window['__hold'+label]=false;
              const response=await nativeFetch(input,{...options,signal:undefined});
              const body=await response.text();
              return await new Promise(resolve=>{window['__release'+label]=()=>resolve(new Response(body,{status:response.status,headers:{'Content-Type':'application/json'}}));});
            }
            return nativeFetch(input,options);
          };
        })();""")
        page.goto(url + "/static/cockpit/")
        page.evaluate("window.CockpitOpenIcloudMail()")
        rows = page.locator("#doctorIcloudInboxRows")
        rows.get_by_role("button", name="Χρειάζεται χαρακτηρισμό · Καλημέρα").click()
        editor = page.locator(".doctor-icloud-preferences").first
        expect(editor.get_by_role("button", name="Αποθήκευση χαρακτηρισμού")).to_be_visible()
        page.screenshot(path=str(tmp_path / "triage-editor.png"),full_page=True)
        editor.get_by_label("Κατηγορία", exact=True).select_option("administrative")
        editor.get_by_role("button", name="Αποθήκευση χαρακτηρισμού").click()
        expect(page.locator("#doctorIcloudTriageNotice")).to_contain_text("αποθηκεύτηκε")
        page.reload()
        page.evaluate("window.CockpitOpenIcloudMail()")
        rows.get_by_role("button", name="Διοικητικό · Καλημέρα").click()
        expect(page.locator(".doctor-icloud-preferences").first.get_by_label("Κατηγορία", exact=True)).to_have_value("administrative")
        # Second browser session reads the same committed preference, not localStorage.
        second = browser.new_context(extra_http_headers={"X-Clinical-Key": "synthetic-clinical-key"})
        other = second.new_page()
        response = other.request.get(url + "/clinical/icloud-mail/items/765-3")
        assert response.json()["item"]["category"] == "administrative"
        # Prime a filtered empty clinical peek; then change source category in inbox.
        for uid in ("3", "4"):
            pref = other.request.get(url + "/clinical/icloud-mail/preferences").json()
            import uuid
            other.request.post(url + "/clinical/icloud-mail/preferences", headers={"Origin": url}, data={
                "kind": "category", "operation_id": str(uuid.uuid4()), "expected_revision": pref["triage"]["revision"],
                "item_id": "765-" + uid, "category": "administrative"})
        page.locator("#doctorIcloudInboxBack").click()
        page.locator("#doctorInboxButton").click()
        expect(page.locator("#doctorPopoverContent")).to_contain_text("Δεν υπάρχουν")
        page.evaluate("window.__holdPeek=true")
        page.locator("#doctorPopoverContent").get_by_text("Ανανέωση τώρα",exact=True).click()
        page.wait_for_function("typeof window.__releasePeek==='function'")
        page.evaluate("window.__holdList=true; window.CockpitOpenIcloudMail()")
        page.wait_for_function("typeof window.__releaseList==='function'")
        page.evaluate("window.CockpitOpenIcloudMail('765-3')")
        editor = page.locator(".doctor-icloud-preferences").first
        editor.get_by_label("Κατηγορία", exact=True).select_option("clinical")
        editor.get_by_role("button", name="Αποθήκευση χαρακτηρισμού").click()
        expect(page.locator("#doctorIcloudTriageNotice")).to_contain_text("αποθηκεύτηκε")
        page.locator("#doctorIcloudInboxBack").click()
        page.locator("#doctorInboxButton").click()
        expect(page.locator("#doctorPopoverContent")).to_contain_text("Καλημέρα")
        page.evaluate("window.__releasePeek(); window.__releaseList()")
        expect(page.locator("#doctorPopoverContent")).to_contain_text("Καλημέρα")
        page.evaluate("window.CockpitOpenIcloudMail()")
        page.locator("#doctorIcloudTriageNotice").get_by_role("button", name="Αναίρεση τελευταίας αλλαγής").click()
        expect(rows).to_contain_text("Διοικητικό · Καλημέρα")
        # Another tab deliberately sends no BroadcastChannel notification.
        prefs=other.request.get(url+"/clinical/icloud-mail/preferences").json()
        other.request.post(url+"/clinical/icloud-mail/preferences",headers={"Origin":url},data={
            "kind":"category","operation_id":str(uuid.uuid4()),"expected_revision":prefs["triage"]["revision"],"item_id":"765-3","category":"other"})
        page.evaluate("window.dispatchEvent(new Event('focus'))")
        expect(rows).to_contain_text("Άλλο · Καλημέρα")
        page.locator("#doctorIcloudRules").click()
        form = page.locator(".doctor-icloud-preferences").last
        form.get_by_label("Ακριβής διεύθυνση αποστολέα ή domain").fill("noreply@wetransfer.com")
        form.get_by_label("Λέξεις θέματος, χωρισμένες με κόμμα").fill("αξονικί")
        form.get_by_role("button", name="Αποθήκευση κανόνα").click()
        expect(page.locator("#doctorIcloudTriageNotice")).to_contain_text("αποθηκεύτηκε")
        fake.rows["5"] = mail(subject="Αξονικί δοκιμή.rar", sender="noreply@wetransfer.com")
        page.locator("#doctorIcloudInboxReload").click()
        expect(rows).to_contain_text("Κλινικό · Αξονικί")
        page.locator("#doctorIcloudRules").click()
        page.get_by_role("button", name="Απενεργοποίηση κανόνα", exact=True).click()
        page.get_by_role("button", name="Επιβεβαίωση απενεργοποίησης", exact=True).click()
        expect(page.locator("#doctorIcloudTriageNotice")).to_contain_text("αποθηκεύτηκε")
        def unavailable(route):
            route.fulfill(status=503,json={"detail":{"code":"TRIAGE_STORAGE_UNAVAILABLE"}})
        page.route(url+"/clinical/icloud-mail/preferences",unavailable)
        page.locator("#doctorIcloudInboxBack").click()
        page.locator("#doctorInboxButton").click()
        expect(page.locator("#doctorPopoverContent")).to_contain_text("Οι κατηγορίες δεν επιβεβαιώθηκαν")
        expect(page.locator("#doctorPopoverContent")).not_to_contain_text("Καλημέρα")
        page.unroute(url+"/clinical/icloud-mail/preferences",unavailable)
        context.set_extra_http_headers({"X-Clinical-Key":"wrong"})
        page.evaluate("window.CockpitOpenIcloudMail()")
        expect(rows).to_be_empty()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert not errors, errors
        assert not external
        assert all(path.endswith("/preferences") or path.endswith("/clinic-utilities/rf/api/outcomes/search") for path in mutations), mutations
        assert all(op in ("FETCH", "SEARCH") for op, _ in fake.operations)
        page.screenshot(path=str(tmp_path / "triage-synthetic.png"), full_page=True)
        second.close()
        browser.close()
