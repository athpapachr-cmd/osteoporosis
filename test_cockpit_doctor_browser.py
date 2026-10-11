"""Real Chromium smoke with invented records; no production, Gmail or patient access."""
from pathlib import Path
import shutil
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).parent
OUT = Path("cockpit-v4-browser-evidence")
OUT.mkdir(exist_ok=True)
HTML = (ROOT / "static/cockpit/index.html").read_text(encoding="utf-8")
CSS = "\n".join((ROOT / part).read_text(encoding="utf-8") for part in (
    "static/cockpit/styles.css",
    "static/cockpit/clinical-workspace.css",
    "static/cockpit/doctor-shell.css",
))
JS = [(ROOT / part).read_text(encoding="utf-8") for part in (
    "static/cockpit/clinical-workspace.js",
    "static/cockpit/app.js",
    "static/cockpit/doctor-shell.js",
)]
RECENT = [
    {"patient_id": "SYN-1", "patient_display_name": "Ασθενής Συνθετική",
     "encounter_date": "2026-10-09", "visit_type": "Επανέλεγχος"},
    {"patient_id": "SYN-2", "patient_display_name": "Ασθενής Δοκιμαστικός",
     "encounter_date": "2026-10-08", "visit_type": "Κλινική επίσκεψη"},
]
SURGERIES = [
    {"surgery_id": "synthetic-surgery-1", "full_name": "Συνθετικός Ασθενής Α",
     "identity_number": "SYNTHETIC", "date_of_birth": "1970-01-01",
     "procedure_type": "Δοκιμή", "laterality": "left", "phone": "000000",
     "surgery_date": None, "queue_position": 1},
    {"surgery_id": "synthetic-surgery-2", "full_name": "Συνθετικός Ασθενής Β",
     "identity_number": "SYNTHETIC", "date_of_birth": "1970-01-01",
     "procedure_type": "Δοκιμή", "laterality": "right", "phone": "000000",
     "surgery_date": "2026-11-01", "queue_position": 2},
]
DAY = [
    {"appointment_id": "cal.com:synthetic-1", "patient_display_name": "Ασθενής Δοκιμής Α",
     "start_at": "2026-10-11T07:30:00Z", "reason": "Επανεξέταση"},
    {"appointment_id": "cal.com:synthetic-2", "patient_display_name": "Ασθενής Δοκιμής Β",
     "start_at": "2026-10-11T08:00:00Z", "reason": "Prolia"},
]
CONTEXT = {
    "generated_at": "2026-10-11T06:00:00Z",
    "source_updated_at": "2026-10-11T05:58:00Z",
    "today_total": 2, "previous": None, "current": None,
    "current_conflict_count": 0,
    "next": DAY[0], "upcoming_today": DAY,
}

def browser_suite(width, screenshot_name):
    errors = []
    outbound = []
    with sync_playwright() as pw:
        executable = shutil.which("google-chrome") or shutil.which("chromium")
        browser = pw.chromium.launch(headless=True, executable_path=executable, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": width, "height": 820}, device_scale_factor=1)
        page.on("pageerror", lambda err: errors.append(str(err)))
        def handle(route):
            url = route.request.url
            if "/clinical/calendar/cockpit-context" in url:
                return route.fulfill(json=CONTEXT)
            if "/clinical/recent-encounters" in url:
                return route.fulfill(json=RECENT)
            if "/clinical/calendar/appointments?" in url:
                return route.fulfill(json=DAY)
            if url.endswith("/clinical/surgeries"):
                return route.fulfill(json=SURGERIES)
            outbound.append(url)
            return route.abort()
        page.route("**/*", handle)
        # set_content() starts at about:blank, where /clinical/... is invalid.
        # Supply a synthetic origin so protected same-origin reads can be routed
        # without any real network or session.
        synthetic_html = HTML.replace("<head>", '<head><base href="https://cockpit.invalid/">', 1)
        page.set_content(synthetic_html, wait_until="domcontentloaded")
        page.add_style_tag(content=CSS)
        for source in JS:
            page.add_script_tag(content=source)

        expect(page.locator("#doctorMainView")).to_be_visible()
        expect(page.locator("#doctorLibraryView")).to_be_hidden()
        expect(page.locator("#visitRecentRows .clinical-patient-row")).to_have_count(2)
        expect(page.locator("#doctorAttentionList .doctor-action-row")).to_have_count(3)
        expect(page.locator("#doctorTasksBadge")).to_have_text("2")
        page.screenshot(path=str(OUT / screenshot_name), full_page=True)

        page.locator("#doctorDateButton").click()
        expect(page.locator("#doctorPopover")).to_be_visible()
        expect(page.locator("#doctorPopoverContent")).to_contain_text("Ασθενής Δοκιμής Α")
        assert "Δεν περιλαμβάνει κατ’ ανάγκη όλο το πρόγραμμα" in page.locator("#doctorPopoverContent").inner_text()
        if width <= 990:
            page.screenshot(path=str(OUT / "cockpit-day-popover-halfwidth.png"), full_page=True)
        page.locator("#doctorPopoverContent button.doctor-action-row").first.click()
        expect(page.locator("#visitWorkspacePatient")).to_be_visible()
        expect(page.locator("#visitIdentityStatus")).to_have_text("Απαιτείται επιλογή φακέλου")
        page.locator("#visitBackHome").click()

        page.locator("#doctorInboxButton").click()
        expect(page.locator("#doctorPopoverContent")).to_contain_text("Δεν έχει συνδεθεί ακόμη")
        expect(page.locator("#doctorInboxBadge")).to_be_hidden()
        page.locator("#doctorInboxButton").click()

        page.locator("#doctorActionsButton").click()
        page.locator("#doctorCommandSearch").fill("Ραδιο")
        expect(page.locator('#doctorPopoverContent a[href="/clinical/clinic-utilities/rf"]')).to_be_visible()
        page.locator("#doctorActionsButton").click()

        page.locator("#doctorTasksButton").click()
        expect(page.locator("#doctorPopoverContent")).to_contain_text("2")
        page.locator("#doctorPopoverContent").get_by_role("button", name="Εκκρεμή χειρουργεία").click()
        expect(page.locator("#doctorMainView")).to_be_hidden()
        expect(page.locator("#surgerySection")).to_be_visible()
        page.locator("#doctorBackHome").click()

        if width <= 990:
            expect(page.locator("#visitSidebar")).to_be_hidden()
            page.locator("#doctorRailToggle").click()
            expect(page.locator("#visitSidebar")).to_be_visible()
        else:
            expect(page.locator("#visitSidebarPrevious")).to_be_disabled()
            expect(page.locator("#visitBriefOverlay")).to_be_hidden()
        assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1"), "horizontal overflow"
        assert not errors, errors
        assert not any("/clinical/visit-capture/save" in url for url in outbound)
        browser.close()

browser_suite(1280, "cockpit-desktop.png")
browser_suite(680, "cockpit-dia-halfwidth.png")
print("PASS Cockpit V4 Chromium: compact Home, day popover, identity boundary, honest Inbox, tasks and half-width")
