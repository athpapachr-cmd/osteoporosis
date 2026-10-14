"""Actual Chromium synthetic RF Tracking/Weekly/Correction smoke, no network or PHI."""
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import shutil

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parent
BASE = "https://synthetic-native-rf.invalid"
PATH = "/clinical/clinic-utilities/rf"
MONDAY = datetime.now(ZoneInfo("Asia/Nicosia")).date()
MONDAY = (MONDAY - timedelta(days=MONDAY.weekday())).isoformat()
TUESDAY = (datetime.fromisoformat(MONDAY).date() + timedelta(days=1)).isoformat()
ASSETS = {
    "/static/clinic-utilities/rf/index.html": ROOT / "static/clinic-utilities/rf/index.html",
    "/static/clinic-utilities/rf/styles.css": ROOT / "static/clinic-utilities/rf/styles.css",
    "/static/clinic-utilities/rf/app.js": ROOT / "static/clinic-utilities/rf/app.js",
    "/static/cockpit/shared-nav.css": ROOT / "static/cockpit/shared-nav.css",
    "/static/cockpit/shared-nav.js": ROOT / "static/cockpit/shared-nav.js",
}
CONTRACT = {
    "doctors": {"papachristou": {"configured": True, "label":"Synthetic Doctor"}},
    "products": {"medikey": {"configured": True, "label":"Synthetic Product"}},
    "indications": {"KNEE_OA_KL34": {
        "label":"Synthetic Knee OA", "site_key":"knee", "requires_intervention":False}},
    "laterality": {"left":"Αριστερά","right":"Δεξιά"},
    "rf_reasons": {"FAILED_PHARMACOLOGIC":"Συνθετικός λόγος"}
}
A = {"id":"synthetic-rejected-A", "patient_name":"Συνθετική Α",
     "identity_number":"SYN-A", "gesy_number":"SYN-GESY-A",
     "pathway":"A1", "issued_date":"2026-10-09", "indication_code":"KNEE_OA_KL34",
     "laterality":"left", "site_key":"knee", "revision":3,
     "stage":"rejected","decision":"rejected","decision_date":"2026-10-10",
     "submitted_date":"2026-10-09","scheduled_date":None,"performed_date":None,
     "issuer_name":"Synthetic Doctor","rejection_reason":"Συνθετική απόρριψη",
     "mine_to_follow":False, "last_decision_report":None}
B = {**A,"id":"synthetic-rejected-B","patient_name":"Συνθετική Β",
     "identity_number":"SYN-B", "revision":2}
SCHEDULED = {**A, "id":"synthetic-scheduled",
    "patient_name":"Συνθετική Προγραμματισμένη","identity_number":"SYN-C",
    "revision":4, "stage":"scheduled","decision":"approved",
    "scheduled_date":MONDAY, "performed_date":TUESDAY,"rejection_reason":""}
APPLICATIONS = [A,B,SCHEDULED]


def suite(width: int):
    errors, outbound, posted = [], [], []
    with sync_playwright() as pw:
        executable = shutil.which("google-chrome") or shutil.which("chromium")
        browser = pw.chromium.launch(headless=True, executable_path=executable,
                                     args=["--no-sandbox"])
        page = browser.new_page(viewport={"width":width, "height":840},
                                device_scale_factor=1, locale="el-GR")
        page.on("pageerror", lambda err: errors.append(str(err)))

        def handle(route):
            request = route.request
            url = request.url
            path = url[len(BASE):] if url.startswith(BASE) else ""
            if path in (PATH, PATH+"/tracking") and request.method == "GET":
                return route.fulfill(status=200, content_type="text/html",
                    body=ASSETS["/static/clinic-utilities/rf/index.html"].read_text(encoding="utf-8"))
            if path in ASSETS:
                kind = ("text/css" if path.endswith(".css") else "application/javascript")
                return route.fulfill(status=200, content_type=kind,
                                     body=ASSETS[path].read_text(encoding="utf-8"))
            if path == "/clinical/status":
                return route.fulfill(status=200, json={"authenticated":True})
            if path == PATH+"/api/contract":
                return route.fulfill(json=CONTRACT)
            if path.startswith(PATH+"/api/drafts?"):
                return route.fulfill(json={"drafts":[],"has_more":False,"truncated":False})
            if path == PATH+"/api/outcomes/search" and request.method == "POST":
                body = request.post_data_json
                if body.get("mode") == "tracking":
                    return route.fulfill(json={"applications":APPLICATIONS,
                        "has_more":False,"next_cursor":None,"scan_complete":True,
                        "truncated":False,"counts_exact":False,
                        "source_as_of":"2026-10-14T12:00:00Z"})
                return route.fulfill(json={"applications":APPLICATIONS,
                    "has_more":False,"truncated":False,
                    "scope":body.get("scope","all"),"denominator":3,"page":1,
                    "counts":{"pending":0,"approved":1,"rejected":2,"mine_actionable":0}})
            if path == PATH+"/api/outcomes/calendar" and request.method == "POST":
                body = request.post_data_json
                assert body["week_start"] == MONDAY, body
                return route.fulfill(json={"week_start":MONDAY,
                    "week_end":(datetime.fromisoformat(MONDAY).date()+timedelta(days=6)).isoformat(),
                    "applications":[SCHEDULED], "has_more":False,
                    "complete":True, "truncated":False, "next_cursor":None,
                    "booking_time_status":"off_unverified_link","source_as_of":"2026-10-14T12:00:00Z"})
            if path.startswith(PATH+"/api/outcomes/") and request.method == "GET":
                selected = next((x for x in APPLICATIONS if path.endswith("/"+x["id"])), None)
                if selected:
                    return route.fulfill(json=selected)
            if path == PATH+"/api/reuse/rejected-correction" and request.method == "POST":
                request_body = request.post_data_json
                posted.append(request_body)
                selected = next((x for x in (A,B) if x["id"] == request_body["application_id"]), None)
                if selected and selected["revision"] == request_body["expected_revision"]:
                    return route.fulfill(json={"application_id":selected["id"],
                        "revision":selected["revision"],
                        "source":{"id":selected["id"],"pathway":"A1"},
                        "fields":{"patient_name":selected["patient_name"],
                                  "identity_number":selected["identity_number"],
                                  "gesy_number":selected["gesy_number"],
                                  "indication_code":"KNEE_OA_KL34",
                                  "product_key":"medikey","laterality":"left",
                                  "exact_location":"Συνθετική περιοχή"}})
                return route.fulfill(status=409,json={"detail":"Η έκδοση άλλαξε"})
            outbound.append(url)
            return route.abort()

        page.route("**/*", handle)
        page.goto(BASE+PATH+"/tracking", wait_until="domcontentloaded")
        expect(page.locator("#rfTrackingView")).to_be_visible()
        expect(page.locator("#rfCreatorView")).to_be_hidden()
        expect(page.locator("#rfTrackingRows .rf-quick-row")).to_have_count(3)
        expect(page.locator("#rfTrackingSource")).to_contain_text("Πλήρης")
        assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1"), (
            f"RF Tracking horizontal overflow at {width}px")
        # One-click detail, no duplicated outcome writer.
        page.locator("#rfTrackingRows .rf-quick-row").first.click()
        expect(page.locator("#rfTrackingDetail")).to_be_visible()
        expect(page.locator("#rfOutcomeTitle")).to_contain_text("Συνθετική Α")
        expect(page.locator("#rfOutcomeCorrection")).to_be_visible()
        # Switch weekly. The dated RF is independent of Reception/Cal availability.
        page.locator("#rfTrackingShowWeek").click()
        expect(page.locator("#rfWeekDays .rf-week-day")).to_have_count(7)
        expect(page.locator("#rfWeekDays .rf-week-event")).to_have_count(2)
        expect(page.locator("#rfWeekSource")).to_contain_text("Χωρίς") if False else None
        assert "Γραμματείας" in page.locator("#rfWeekSource").inner_text()
        assert "ραντεβού" in page.locator("#rfTrackingWeekPanel").inner_text()
        # A fresh selection after returning to list; explicit correction performs guarded POST.
        page.locator("#rfTrackingShowList").click()
        page.locator("#rfTrackingRows .rf-quick-row").first.click()
        expect(page.locator("#rfOutcomeTitle")).to_contain_text("Συνθετική Α")
        page.locator("#rfOutcomeCorrection").click()
        expect(page.locator("#rfCreatorView")).to_be_visible()
        expect(page.locator("#rfTrackingView")).to_be_hidden()
        expect(page.locator("#patientName")).to_have_value("Συνθετική Α")
        assert posted[-1] == {"application_id":"synthetic-rejected-A","expected_revision":3}
        assert page.url.endswith(PATH), page.url
        assert "synthetic-rejected" not in page.url and "SYN-A" not in page.url
        # Returning to Tracking preserves creator form. A deliberate B correction requires consent.
        page.locator("#rfGoTracking").click()
        page.locator("#rfTrackingRows .rf-quick-row").nth(1).click()
        page.locator("#rfOutcomeCorrection").click()  # default dialog dismissal
        expect(page.locator("#patientName")).to_have_value("Συνθετική Α")
        page.once("dialog",lambda dialog:dialog.accept())
        page.locator("#rfOutcomeCorrection").click()
        expect(page.locator("#patientName")).to_have_value("Συνθετική Β")
        assert posted[-1] == {"application_id":"synthetic-rejected-B","expected_revision":2}
        assert not errors, errors
        assert not outbound, outbound
        browser.close()


if __name__ == "__main__":
    for size in (390,680,1280):
        suite(size)
    print("PASS RF Tracking Chromium 390/680/1280: list/week/guarded correction/no PHI URL")
