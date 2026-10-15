"""Synthetic RF Tracking v1 smoke in real Chromium at three widths; zero network/data writes."""
import json
import shutil
from pathlib import Path
from urllib.parse import urlsplit
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).parent
OUT = ROOT / "rf-tracking-browser-evidence"
OUT.mkdir(exist_ok=True)
BASE = "https://rf-synthetic.invalid"
CONTRACT = {
    "doctors": {"papachristou": {"label": "Synthetic Doctor", "configured": True}},
    "products": {"medikey": {"label": "Synthetic Product", "configured": True}},
    "indications": {
        "KNEE_OA_KL34": {"label": "Synthetic Knee", "site_key": "knee",
                         "location_labels": {}, "location_mode": "free",
                         "requires_intervention": False}},
    "laterality": {"left": "Αριστερά", "right": "Δεξιά"},
    "rf_reasons": {},
}
def card(n):
    return {
        "id": "synthetic-rf-%02d" % n, "patient_name": "Συνθετικός RF %02d" % n,
        "identity_number": "SYN-ID-%02d" % n, "issued_date": "2026-10-01",
        "pathway": "A1", "site_key": "knee", "laterality": "left",
        "indication_code": "KNEE_OA_KL34", "issuer_name": "Synthetic Issuer",
        "stage": "rejected", "decision": "rejected", "revision": 2,
        "rejection_reason": "Synthetic reason", "decision_date": "2026-10-02",
        "submitted_date": None, "scheduled_date": None, "performed_date": None,
        "mine_to_follow": False,
    }

def suite(width):
    errors, outbound = [], []
    mock = {"tracking_status": 200, "write_status": 409}
    with sync_playwright() as pw:
        executable = shutil.which("google-chrome") or shutil.which("chromium")
        browser = pw.chromium.launch(headless=True, executable_path=executable,
                                      args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": width, "height": 850})
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.on("dialog", lambda dialog: dialog.accept())
        def handle(route):
            url = urlsplit(route.request.url)
            path = url.path
            if path in ("/clinical/clinic-utilities/rf",
                        "/clinical/clinic-utilities/rf/tracking"):
                return route.fulfill(status=200, content_type="text/html",
                    body=(ROOT / "static/clinic-utilities/rf/index.html").read_text(encoding="utf-8"))
            if path.startswith("/static/"):
                source = ROOT / path.lstrip("/")
                if source.is_file():
                    kind = ("text/javascript" if source.suffix == ".js"
                            else "text/css" if source.suffix == ".css" else "text/plain")
                    return route.fulfill(status=200, content_type=kind, body=source.read_bytes())
            if path == "/clinical/status":
                return route.fulfill(json={"authenticated": True})
            if path == "/clinical/login":
                return route.fulfill(json={"authenticated": True})
            if path.startswith("/clinical/clinic-utilities/rf/api/"):
                endpoint = path.split("/rf/api/", 1)[-1]
                request = route.request.post_data_json if route.request.method == "POST" else {}
                if endpoint == "contract":
                    return route.fulfill(json=CONTRACT)
                if endpoint == "drafts" or endpoint.startswith("drafts?"):
                    return route.fulfill(json={"drafts": [], "has_more": False})
                if endpoint == "outcomes/search":
                    if request.get("mode") == "tracking" and mock["tracking_status"] != 200:
                        return route.fulfill(status=mock["tracking_status"],
                            json={"detail": "Synthetic tracking source unavailable"})
                    if request.get("mode") == "tracking":
                        if request.get("stage") == "approved":
                            result = {"applications": [], "next_cursor": None,
                                      "has_more": False, "scan_complete": True,
                                      "truncated": False, "counts": None}
                        elif request.get("cursor"):
                            result = {"applications": [card(n) for n in range(20, 23)],
                                      "next_cursor": None, "has_more": False,
                                      "scan_complete": True, "truncated": False, "counts": None}
                        else:
                            result = {"applications": [card(n) for n in range(20)],
                                      "next_cursor": "signed-synthetic-next",
                                      "has_more": True, "scan_complete": False,
                                      "truncated": True, "counts": None}
                    else:
                        result = {"applications": [], "counts": {
                            "pending": 0, "approved": 0, "rejected": 0,
                            "mine_actionable": 0}, "denominator": 0,
                            "truncated": False, "has_more": False}
                    return route.fulfill(json=result)
                if endpoint == "outcomes/calendar":
                    monday = request["week_start"]
                    return route.fulfill(json={
                        "facts": [{"application_id": "synthetic-rf-00",
                                   "patient_name": "Συνθετικός RF 00", "kind": "scheduled",
                                   "date": monday, "site_key": "knee", "laterality": "left",
                                   "revision": 2}],
                        "complete": True, "truncated": False, "time_verified": False,
                        "source_as_of": "2026-10-14T12:00:00Z"})
                if endpoint.startswith("outcomes/") and endpoint.endswith("/events"):
                    return route.fulfill(status=mock["write_status"],
                        json={"detail": "Synthetic changed revision"
                              if mock["write_status"] == 409 else "Synthetic authorization lost"})
                if endpoint.startswith("outcomes/") and endpoint.endswith("/delete"):
                    assert endpoint == "outcomes/synthetic-rf-00/delete"
                    assert request["expected_revision"] == 2
                    assert request["confirmation_code"] == "IC-RF-00"
                    return route.fulfill(status=409,
                        json={"detail": "Synthetic linked RF prevents deletion"})
                if endpoint.startswith("outcomes/"):
                    item_id = endpoint.split("/")[-1]
                    try:
                        n = int(item_id.rsplit("-", 1)[1])
                    except (IndexError, ValueError):
                        return route.fulfill(status=404, json={"detail": "missing"})
                    return route.fulfill(json=card(n))
                if endpoint == "reuse/issued-copy":
                    n = int(request["application_id"].rsplit("-", 1)[1])
                    return route.fulfill(json={
                        "application_id": request["application_id"],
                        "revision": request["expected_revision"],
                        "source": {"kind": "application", "pathway": "A1"},
                        "files_saved": False,
                        "fields": {"identity_number": "SYN-ID-%02d" % n,
                                   "patient_name": "Συνθετικός RF %02d" % n,
                                   "age": 60, "gesy_number": "SYN-GESY",
                                   "indication_code": "KNEE_OA_KL34",
                                   "laterality": "left", "product_key": "medikey",
                                   "exact_location": "Synthetic knee",
                                   "pain_onset_date": "2026-01-01",
                                   "pain_onset_vas": 8,
                                   "last_assessment_date": "2026-04-01",
                                   "last_assessment_vas": 7,
                                   "nsaid_trials": [{"drug_name": "Synthetic Nsaid",
                                                     "dose": "90 mg", "duration": "3 months"}],
                                   "physio_dates_text": "2026-02-01\n2026-02-08"},
                    }, headers={"cache-control": "no-store"})
                if endpoint == "reuse/rejected-correction":
                    n = int(request["application_id"].rsplit("-", 1)[1])
                    return route.fulfill(json={
                        "application_id": request["application_id"],
                        "revision": request["expected_revision"],
                        "source": {"pathway": "A1", "kind": "application"},
                        "fields": {"identity_number": "SYN-ID-%02d" % n,
                                   "patient_name": "Συνθετικός RF %02d" % n,
                                   "age": 60, "gesy_number": "SYN-GESY",
                                   "indication_code": "KNEE_OA_KL34",
                                   "laterality": "left", "product_key": "medikey",
                                   "exact_location": "Synthetic knee"},
                    }, headers={"cache-control": "no-store"})
            outbound.append(route.request.url)
            return route.abort()
        page.route("**/*", handle)
        page.goto(BASE + "/clinical/clinic-utilities/rf/tracking")
        expect(page.locator("#rfTrackingView")).to_be_visible()
        expect(page.locator("#rfBuilderView")).to_be_hidden()
        expect(page.locator("#rfTrackResults .rf-outcome-card")).to_have_count(20)
        assert page.locator("#rfTrackScope").input_value() == "all"
        assert page.locator("#rfTrackStatus").inner_text().startswith("Μερική")
        page.locator("#rfTrackMore").click()
        expect(page.locator("#rfTrackResults .rf-outcome-card")).to_have_count(23)
        page.screenshot(path=str(OUT / ("rf-list-%s.png" % width)), full_page=True)
        # The original implementation invalidated older row handlers after "More".
        first = page.locator("#rfTrackResults .rf-outcome-card").first
        first.get_by_role("button", name="Άνοιγμα").focus()
        page.keyboard.press("Enter")
        expect(page.locator("#rfOutcomeTitle")).to_contain_text("Συνθετικός RF 00")
        # A real failed write must show in the visible Tracking view, not hidden builder.
        page.locator("#rfOutcomeApply").click()
        expect(page.locator("#rfTrackStatus")).to_contain_text("Synthetic changed revision")
        first.get_by_role("button", name="Άνοιγμα").click()
        expect(page.locator("#rfOutcomeCorrection")).to_be_visible()
        page.locator("#rfOutcomeCorrection").click()
        expect(page.locator("#rfBuilderView")).to_be_visible()
        expect(page.locator("#rfTrackingView")).to_be_hidden()
        expect(page.locator("#rfCorrectionContext")).to_be_visible()
        assert page.locator("#doctorSelect").input_value() == "", "issuer must be reselected"
        assert page.locator("#identityNumber").input_value() == "SYN-ID-00"
        assert page.locator("#imagingReport").input_value() == "", "attachments must not carry over"
        assert page.evaluate("location.pathname") == "/clinical/clinic-utilities/rf"
        assert "synthetic-rf-" not in page.url
        page.locator("#rfNavTracking").click()
        expect(page.locator("#rfTrackingView")).to_be_visible()
        expect(page.locator("#rfCorrectionContext")).to_be_hidden()
        # Existing creator belongs to RF 00. Guard may not mutate it using RF 01.
        second = page.locator("#rfTrackResults .rf-outcome-card").nth(1)
        second.get_by_role("button", name="Άνοιγμα").click()
        expect(page.locator("#rfOutcomeTitle")).to_contain_text("Συνθετικός RF 01")
        page.locator("#rfOutcomeCorrection").click()
        expect(page.locator("#rfTrackStatus")).to_contain_text("άλλη ταυτότητα")
        assert page.locator("#identityNumber").input_value() == "SYN-ID-00"
        page.locator("#rfTrackWeekTab").click()
        expect(page.locator("#rfTrackWeekView")).to_be_visible()
        expect(page.locator("#rfWeekGrid .rf-week-day")).to_have_count(7)
        expect(page.locator("#rfWeekStatus")).to_contain_text("όχι ραντεβού")
        page.screenshot(path=str(OUT / ("rf-week-%s.png" % width)), full_page=True)
        assert page.locator("#rfWeekGrid").inner_text().find("00") >= 0
        assert "12:00" not in page.locator("#rfWeekGrid").inner_text()
        page.locator("#rfWeekNoDate").click()
        expect(page.locator("#rfTrackListView")).to_be_visible()
        assert page.locator("#rfTrackStage").input_value() == "approved"
        mock["tracking_status"] = 503
        page.locator("#rfTrackStage").select_option("")
        expect(page.locator("#rfTrackStatus")).to_contain_text("δεν είναι διαθέσιμα")
        expect(page.locator("#rfTrackResults .rf-outcome-card")).to_have_count(0)
        mock["tracking_status"] = 200
        page.locator("#rfTrackRefresh").click()
        expect(page.locator("#rfTrackResults .rf-outcome-card")).to_have_count(20)
        page.locator("#rfTrackResults .rf-outcome-card").first.get_by_role(
            "button", name="Άνοιγμα").click()
        expect(page.locator("#rfOutcomeTitle")).to_contain_text("Συνθετικός RF 00")
        mock["write_status"] = 401
        page.locator("#rfOutcomeApply").click()
        expect(page.locator("#rfTrackStatus")).to_contain_text("Απαιτείται νέα εξουσιοδότηση")
        expect(page.locator("#rfTrackResults .rf-outcome-card")).to_have_count(0)
        expect(page.locator("#rfOutcomeSelected")).to_be_hidden()
        mock["write_status"] = 409
        page.evaluate("window.dispatchEvent(new Event('clinical:authenticated'))")
        expect(page.locator("#rfTrackResults .rf-outcome-card")).to_have_count(20)
        assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1"), "horizontal overflow"
        page.screenshot(path=str(OUT / ("rf-tracking-%s.png" % width)), full_page=True)
        page.locator("#rfTrackResults .rf-outcome-card").first.get_by_role(
            "button", name="Άνοιγμα").click()
        expect(page.locator("#rfOutcomeTitle")).to_contain_text("Συνθετικός RF 00")
        expect(page.locator("#rfOutcomeDelete")).to_be_visible()
        expect(page.locator("#rfOutcomeCopy")).to_be_visible()
        page.evaluate("window.prompt = () => 'IC-RF-00'")
        page.locator("#rfOutcomeDelete").click()
        expect(page.locator("#rfTrackStatus")).to_contain_text("Synthetic linked RF prevents deletion")
        expect(page.locator("#rfTrackResults .rf-outcome-card")).to_have_count(20)
        page.locator("#rfTrackResults .rf-outcome-card").first.locator(
            "button").filter(has_text="Φόρτωση").click()
        expect(page.locator("#rfBuilderView")).to_be_visible()
        expect(page.locator("#rfTrackingView")).to_be_hidden()
        expect(page.locator("#identityNumber")).to_have_value("SYN-ID-00")
        expect(page.locator("#patientName")).to_have_value("Συνθετικός RF 00")
        expect(page.locator("#doctorSelect")).to_have_value("")
        expect(page.locator("#physioDates")).to_have_value("2026-02-01\n2026-02-08")
        expect(page.locator("#imagingReport")).to_have_value("")
        expect(page.locator("#rfCorrectionContext")).to_be_hidden()
        assert page.evaluate("location.pathname") == "/clinical/clinic-utilities/rf"
        page.locator("#rfNavTracking").click()
        page.reload()
        expect(page.locator("#rfTrackingView")).to_be_visible()
        expect(page.locator("#rfCorrectionContext")).to_be_hidden()
        assert "synthetic-rf-" not in page.url
        assert not errors, errors
        assert not any("render.com" in x or "cal.com" in x for x in outbound), outbound
        browser.close()

for width in (390, 680, 1280):
    suite(width)
print("PASS RF Tracking Chromium: 390/680/1280, cursor, keyboard, correction, issuer, errors, calendar, reload")
