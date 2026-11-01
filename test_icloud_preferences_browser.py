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
            const label=u.pathname==='/clinical/icloud-mail/preferences'&&options.method==='POST'?'Save':u.pathname==='/clinical/icloud-mail/peek'?'Peek':u.pathname==='/clinical/icloud-mail'?'List':null;
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
        page.evaluate("window.__holdSave=true")
        editor.get_by_role("button", name="Αποθήκευση χαρακτηρισμού").click()
        page.wait_for_function("typeof window.__releaseSave==='function'")
        expect(editor.get_by_label("Κατηγορία",exact=True)).to_be_disabled()
        expect(editor.get_by_label("Κανόνας και για επόμενα παρόμοια",exact=True)).to_be_disabled()
        expect(editor.get_by_role("button",name="Έλεγχος νεότερων επιλογών")).to_be_disabled()
        page.evaluate("window.__releaseSave(); delete window.__releaseSave")
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
        page.evaluate("window.__holdSave=true")
        form.get_by_role("button", name="Αποθήκευση κανόνα").click()
        page.wait_for_function("typeof window.__releaseSave==='function'")
        expect(form.get_by_label("Λέξεις θέματος, χωρισμένες με κόμμα")).to_be_disabled()
        expect(form.get_by_label("Ακριβής διεύθυνση αποστολέα ή domain")).to_be_disabled()
        expect(form.get_by_role("button",name="Έλεγχος νεότερων επιλογών")).to_be_disabled()
        page.evaluate("window.__releaseSave(); delete window.__releaseSave")
        expect(page.locator("#doctorIcloudTriageNotice")).to_contain_text("αποθηκεύτηκε")
        fake.rows["5"] = mail(subject="Αξονικί δοκιμή.rar", sender="noreply@wetransfer.com")
        page.locator("#doctorIcloudInboxReload").click()
        expect(rows).to_contain_text("Κλινικό · Αξονικί")
        page.locator("#doctorIcloudRules").click()
        page.get_by_role("button",name="Τροποποίηση / ενεργοποίηση",exact=True).click()
        page.get_by_role("button",name="Τροποποίηση / ενεργοποίηση",exact=True).click()
        forms=page.locator("section.doctor-icloud-preferences").filter(has=page.get_by_role("button",name="Αποθήκευση κανόνα",exact=True))
        first_edit=forms.nth(1); second_edit=forms.nth(2)
        first_edit.get_by_label("Λέξεις θέματος, χωρισμένες με κόμμα").fill("αξονική")
        second_edit.get_by_label("Λέξεις θέματος, χωρισμένες με κόμμα").fill("MRI")
        page.evaluate("window.__holdSave=true")
        first_edit.get_by_role("button",name="Αποθήκευση κανόνα",exact=True).click()
        page.wait_for_function("typeof window.__releaseSave==='function'")
        expect(second_edit.get_by_label("Λέξεις θέματος, χωρισμένες με κόμμα")).to_be_disabled()
        expect(first_edit.get_by_label("Λέξεις θέματος, χωρισμένες με κόμμα")).to_be_disabled()
        page.evaluate("window.__releaseSave(); delete window.__releaseSave")
        expect(page.locator("#doctorIcloudTriageNotice")).to_contain_text("αποθηκεύτηκε")
        page.locator("#doctorIcloudRules").click()
        page.get_by_role("button",name="Τροποποίηση / ενεργοποίηση",exact=True).click()
        expect(page.locator(".doctor-icloud-preferences").last.get_by_label("Λέξεις θέματος, χωρισμένες με κόμμα")).to_have_value("MRI")
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


@pytest.fixture(params=[1280, 390])
def category_browser(server, request):
    """Real local API/DB/IMAP; collect the actual source requests and failures."""
    url, fake = server
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=shutil.which("google-chrome") or shutil.which("chromium") or None)
        context = browser.new_context(viewport={"width": request.param, "height": 1000},
            extra_http_headers={"X-Clinical-Key": "synthetic-clinical-key"})
        page = context.new_page()
        page.set_default_timeout(6000)
        requests, errors = [], []
        page.on("request", lambda r: requests.append((r.method, r.url)))
        page.on("pageerror", lambda e: errors.append(str(e)))
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
          const fetchSource=window.fetch.bind(window);
          window.fetch=async (input,options={})=>{
            const u=new URL(typeof input==='string'?input:input.url,location.href);
            if(u.pathname.startsWith('/clinical/icloud-mail/items/') && window.__holdItem){
              window.__holdItem=false;
              await new Promise(resolve=>{window.__releaseItem=resolve;});
              // Deliberately ignore abort, so generation checks must reject late results.
              return fetchSource(input,{...options,signal:undefined});
            }
            const label=u.pathname==='/clinical/icloud-mail/preferences'?
              (options.method==='POST'?'SaveReply':'Qualify'):null;
            if(label&&window['__hold'+label]){
              window['__hold'+label]=false;
              const response=await fetchSource(input,options),body=await response.text();
              await new Promise(resolve=>{window['__release'+label]=resolve;});
              return new Response(body,{status:response.status,headers:{'Content-Type':'application/json'}});
            }
            return fetchSource(input,options);
          };
        })();""")
        try:
            yield page, context, url, fake, requests
            assert not errors, errors
            assert all(u.startswith(url) for _, u in requests)
            assert all(method in ("GET", "HEAD", "OPTIONS") or u.endswith("/preferences")
                       or u.endswith("/clinic-utilities/rf/api/outcomes/search") for method, u in requests)
            assert all(op in ("FETCH", "SEARCH") for op, _ in fake.operations)
        finally:
            browser.close()


def source_counts(requests, fake):
    from urllib.parse import urlsplit
    paths = [urlsplit(u).path for method, u in requests if method == "GET"]
    return (paths.count("/clinical/icloud-mail"), paths.count("/clinical/icloud-mail/peek"),
            sum(op == "SEARCH" for op, _ in fake.operations))


def test_single_category_reset_undo_retains_source_page_without_rescan(category_browser, tmp_path):
    page, context, url, fake, requests = category_browser
    for uid in range(5, 36):
        fake.rows[str(uid)] = mail(subject=f"Συνθετικό-{uid}", body=f"Σώμα μόνο για δοκιμή {uid}")
    page.goto(url + "/static/cockpit/")
    page.wait_for_load_state("networkidle")
    page.evaluate("window.CockpitOpenIcloudMail()")
    rows = page.locator("#doctorIcloudInboxRows")
    details = page.locator("#doctorIcloudInboxDetails")
    rows.get_by_role("button", name="Παλαιότερα →", exact=True).click()
    expect(page.locator("#doctorIcloudInboxShown")).to_contain_text("Σελίδα 2")
    rows.get_by_role("button", name="Χρειάζεται χαρακτηρισμό · Συνθετικό-10 ").click()
    choose = details.get_by_label("Κατηγορία", exact=True)
    expect(choose).to_have_value("review")
    page.wait_for_load_state("networkidle")
    page.evaluate("window.__originalBody=document.querySelector('.doctor-icloud-body'); window.__holdItem=true")
    baseline = source_counts(requests, fake)
    request_start, operation_start = len(requests), len(fake.operations)
    choose.select_option("administrative")
    details.get_by_role("button", name="Αποθήκευση χαρακτηρισμού", exact=True).click()
    page.wait_for_function("typeof window.__releaseItem==='function'")
    expect(details.locator(".doctor-icloud-body")).to_contain_text("Σώμα μόνο για δοκιμή 10")
    assert page.evaluate("window.__originalBody===document.querySelector('.doctor-icloud-body')")
    expect(rows).to_contain_text("Συνθετικό-11")
    expect(rows).to_contain_text("Ενημέρωση χαρακτηρισμού · Συνθετικό-10")
    assert source_counts(requests, fake) == baseline
    page.evaluate("window.__releaseItem(); delete window.__releaseItem")
    expect(choose).to_have_value("administrative")
    expect(details.locator("h3")).to_have_text("Διοικητικό · Συνθετικό-10")
    expect(rows.get_by_role("button", name="Διοικητικό · Συνθετικό-10 ")).to_have_attribute("aria-pressed", "true")
    expect(page.locator("#doctorIcloudInboxShown")).to_contain_text("Σελίδα 2")
    # Reset and undo are read from the server, never guessed from the submitted value.
    choose.select_option("automatic")
    details.get_by_role("button", name="Αποθήκευση χαρακτηρισμού", exact=True).click()
    expect(choose).to_have_value("review")
    expect(details).to_contain_text("Αυτόματη πρόταση")
    page.locator("#doctorIcloudTriageNotice").get_by_role("button", name="Αναίρεση τελευταίας αλλαγής").click()
    expect(choose).to_have_value("administrative")
    page.wait_for_load_state("networkidle")
    assert source_counts(requests, fake) == baseline
    expect(page.locator("#doctorIcloudInboxShown")).to_contain_text("Σελίδα 2")
    expect(rows.get_by_role("button", name="← Νεότερα", exact=True)).to_be_visible()
    from urllib.parse import urlsplit
    assert [urlsplit(u).path for method, u in requests[request_start:]
            if method == "GET" and "/items/" in u] == ["/clinical/icloud-mail/items/765-10"] * 3
    assert {args[0] for op, args in fake.operations[operation_start:] if op == "FETCH"} == {"10"}
    # A filtered-out selected message stays open; undo restores its original row.
    page.get_by_role("button", name="Κλινικά", exact=True).click()
    rows.get_by_role("button", name="Χρειάζεται χαρακτηρισμό · Συνθετικό-35 ").click()
    expect(choose).to_have_value("review")
    page.wait_for_load_state("networkidle")
    baseline = source_counts(requests, fake)
    choose.select_option("administrative")
    details.get_by_role("button", name="Αποθήκευση χαρακτηρισμού", exact=True).click()
    expect(choose).to_have_value("administrative")
    expect(rows).not_to_contain_text("Συνθετικό-35")
    expect(details.locator(".doctor-icloud-body")).to_contain_text("Σώμα μόνο για δοκιμή 35")
    page.locator("#doctorIcloudTriageNotice").get_by_role("button", name="Αναίρεση τελευταίας αλλαγής").click()
    expect(choose).to_have_value("review")
    expect(rows).to_contain_text("Χρειάζεται χαρακτηρισμό · Συνθετικό-35")
    page.wait_for_load_state("networkidle")
    assert source_counts(requests, fake) == baseline
    assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
    page.screenshot(path=str(tmp_path / "single-item-refresh-synthetic.png"), full_page=True)


def test_single_refresh_concurrent_revision_retry_and_auth_fail_closed(category_browser):
    from uuid import uuid4
    page, context, url, fake, requests = category_browser
    page.goto(url + "/static/cockpit/")
    page.wait_for_load_state("networkidle")
    page.evaluate("window.CockpitOpenIcloudMail()")
    rows = page.locator("#doctorIcloudInboxRows")
    details = page.locator("#doctorIcloudInboxDetails")
    rows.get_by_role("button", name="Χρειάζεται χαρακτηρισμό · Καλημέρα").click()
    choose = details.get_by_label("Κατηγορία", exact=True)
    expect(choose).to_have_value("review")
    page.wait_for_load_state("networkidle")
    baseline = source_counts(requests, fake)
    page.evaluate("window.__holdItem=true")
    choose.select_option("administrative")
    details.get_by_role("button", name="Αποθήκευση χαρακτηρισμού", exact=True).click()
    page.wait_for_function("typeof window.__releaseItem==='function'")
    # A different writer advances the DB before targeted qualification, without notification.
    pref = context.request.get(url + "/clinical/icloud-mail/preferences").json()
    response = context.request.post(url + "/clinical/icloud-mail/preferences", headers={"Origin": url}, data={
        "kind": "category", "operation_id": str(uuid4()), "expected_revision": pref["triage"]["revision"],
        "item_id": "765-4", "category": "other"})
    assert response.status == 200
    page.evaluate("window.__releaseItem(); delete window.__releaseItem")
    expect(rows).to_contain_text("Άλλο · Αποτελέσματα MRI")
    expect(rows).to_contain_text("Διοικητικό · Καλημέρα")
    assert source_counts(requests, fake)[0] > baseline[0]  # Cannot advance unchanged rows across an unknown delta.
    # The conditional undo for the earlier operation cannot overwrite that writer.
    page.locator("#doctorIcloudTriageNotice").get_by_role("button", name="Αναίρεση τελευταίας αλλαγής").click()
    expect(page.locator("#doctorIcloudTriageNotice")).to_contain_text("Οι επιλογές άλλαξαν")
    expect(rows).to_contain_text("Άλλο · Αποτελέσματα MRI")
    # A lost successful response must retry the same UUID as replay, then fully requalify.
    rows.get_by_role("button", name="Διοικητικό · Καλημέρα").click()
    expect(choose).to_have_value("administrative")
    def lost_response(route):
        response = route.fetch()
        assert response.status == 200
        route.abort()
    page.route(url + "/clinical/icloud-mail/preferences", lost_response, times=1)
    choose.select_option("clinical")
    details.get_by_role("button", name="Αποθήκευση χαρακτηρισμού", exact=True).click()
    expect(details).to_contain_text("Failed to fetch")
    baseline = source_counts(requests, fake)
    details.get_by_role("button", name="Αποθήκευση χαρακτηρισμού", exact=True).click()
    expect(rows).to_contain_text("Κλινικό · Καλημέρα")
    assert source_counts(requests, fake)[0] > baseline[0]
    # A target response completing after navigation cannot replace the new selection.
    rows.get_by_role("button", name="Κλινικό · Καλημέρα").click()
    expect(choose).to_have_value("clinical")
    page.evaluate("window.__holdItem=true")
    choose.select_option("review")
    details.get_by_role("button", name="Αποθήκευση χαρακτηρισμού", exact=True).click()
    page.wait_for_function("typeof window.__releaseItem==='function'")
    page.evaluate("window.CockpitOpenIcloudMail('765-4')")
    expect(details.locator("h3")).to_have_text("Άλλο · Αποτελέσματα MRI")
    page.evaluate("window.__releaseItem(); delete window.__releaseItem")
    page.wait_for_load_state("networkidle")
    expect(details.locator("h3")).to_have_text("Άλλο · Αποτελέσματα MRI")
    # Auth loss while a successful Save's targeted read is pending purges all source DOM.
    page.evaluate("window.CockpitOpenIcloudMail()")
    rows.get_by_role("button", name="Χρειάζεται χαρακτηρισμό · Καλημέρα").click()
    expect(choose).to_have_value("review")
    page.evaluate("window.__holdItem=true")
    choose.select_option("other")
    details.get_by_role("button", name="Αποθήκευση χαρακτηρισμού", exact=True).click()
    page.wait_for_function("typeof window.__releaseItem==='function'")
    context.set_extra_http_headers({"X-Clinical-Key": "wrong"})
    page.evaluate("window.__releaseItem(); delete window.__releaseItem")
    expect(rows).to_be_empty()
    expect(details).to_be_empty()


def test_earlier_qualification_cannot_cancel_successful_save_confirmation(category_browser):
    page, context, url, fake, requests = category_browser
    page.goto(url + "/static/cockpit/")
    page.wait_for_load_state("networkidle")
    page.evaluate("window.CockpitOpenIcloudMail()")
    rows = page.locator("#doctorIcloudInboxRows")
    details = page.locator("#doctorIcloudInboxDetails")
    rows.get_by_role("button", name="Χρειάζεται χαρακτηρισμό · Καλημέρα").click()
    choose = details.get_by_label("Κατηγορία", exact=True)
    expect(choose).to_have_value("review")
    page.wait_for_load_state("networkidle")
    baseline = source_counts(requests, fake)
    page.evaluate("window.__holdQualify=true")
    details.get_by_role("button", name="Έλεγχος νεότερων επιλογών", exact=True).click()
    page.wait_for_function("typeof window.__releaseQualify==='function'")
    choose.select_option("administrative")
    page.evaluate("window.__holdSaveReply=true")
    details.get_by_role("button", name="Αποθήκευση χαρακτηρισμού", exact=True).click()
    page.wait_for_function("typeof window.__releaseSaveReply==='function'")
    # The real DB commit exists, but the UI has not received its response yet.
    response = context.request.get(url + "/clinical/icloud-mail/items/765-3")
    assert response.status == 200 and response.json()["item"]["category"] == "administrative"
    expect(choose).to_be_disabled()
    page.evaluate("window.__releaseQualify(); delete window.__releaseQualify")
    expect(details).to_contain_text("Φορτώθηκαν οι νεότερες επιλογές")
    expect(choose).to_be_disabled()
    page.evaluate("window.__releaseSaveReply(); delete window.__releaseSaveReply")
    expect(page.locator("#doctorIcloudTriageNotice")).to_contain_text("Η επιλογή αποθηκεύτηκε")
    expect(details.locator("h3")).to_have_text("Διοικητικό · Καλημέρα")
    expect(choose).to_have_value("administrative")
    expect(rows).to_contain_text("Διοικητικό · Καλημέρα")
    page.wait_for_load_state("networkidle")
    assert source_counts(requests, fake) == baseline
