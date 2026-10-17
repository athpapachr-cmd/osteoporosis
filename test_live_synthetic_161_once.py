"""One-off live synthetic proof. Existing Actions secret only; never print errors/keys."""
import os
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

URL = "https://ortho-reception-backend.onrender.com/clinical/synthetic-161/"
stage = "start"
try:
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        out = Path("live-161-proof")
        out.mkdir(exist_ok=True)
        for i in range(2):
            stage = f"browser-{i+1}-existing-key-login"
            ctx = browser.new_context(viewport={"width":1280,"height":1000})
            page = ctx.new_page()
            page.set_default_timeout(20000)
            page.goto(URL)
            page.locator("#ceGlobalKey").click()
            page.locator("#ceGlobalKeyInput").fill(os.environ["CLINICAL_DATA_KEY"])
            page.locator("#ceGlobalLoginForm button[type=submit]").click()
            expect(page.locator("#ceGlobalAuthStatus")).to_contain_text("Συνδέθηκες")
            page.locator("#ceGlobalKey").click()
            stage = f"browser-{i+1}-persisted-scenario"
            if i == 0:
                page.locator("#synthetic161Prepare").click()
                expect(page.locator("#synthetic161Status")).to_contain_text("αποθηκεύτηκε")
            expect(page.locator("#currentAppointmentPatient")).to_have_text("Δοκιμαστικός Ασθενής 161 2")
            page.screenshot(path=str(out / f"home-browser-{i+1}.png"), full_page=True)
            stage = f"browser-{i+1}-backend-confirmation"
            page.locator("#visitSidebarCurrent").click()
            expect(page.locator("#visitBookingConfirmation")).to_be_hidden()
            expect(page.locator("#visitBriefChoosePatient")).to_be_hidden()
            expect(page.locator("#visitBriefPrevious")).to_contain_text("2026-10-07")
            expect(page.locator("#visitBriefPlan")).to_contain_text("Συμφωνήθηκε επανέλεγχος")
            action=page.locator("#visitBriefPending .pending-action")
            if i==0:
                initial_action=action.inner_text()
                assert initial_action in ("Παραλήφθηκε","Αναίρεση")
                changed_action="Αναίρεση" if initial_action=="Παραλήφθηκε" else "Παραλήφθηκε"
                action.click()
            expect(action).to_have_text(changed_action)
            page.screenshot(path=str(out / f"brief-browser-{i+1}.png"), full_page=True)
            stage = f"browser-{i+1}-full-chart-refresh"
            page.locator("#visitBriefOpenChart").click()
            expect(page.locator("#synthetic161Chart")).to_contain_text("2 αποθηκευμένες επισκέψεις")
            page.reload()
            expect(page.locator("#synthetic161Chart")).to_contain_text("2 αποθηκευμένες επισκέψεις")
            expect(page.locator("#synthetic161Chart .pending-action")).to_have_text(changed_action)
            page.locator("#synthetic161Chart .clinical-source-link").click()
            expect(page.locator(".patient-visit[open] .patient-obligation-origin")).to_contain_text("Υποχρεώσεις που δημιουργήθηκαν στην επίσκεψη")
            expect(page.locator(".patient-visit[open] .patient-obligation-origin")).to_contain_text("7 Οκτωβρίου 2026")
            expect(page.locator(".patient-timeline-event")).not_to_have_count(0)
            page.screenshot(path=str(out / f"origin-browser-{i+1}.png"), full_page=True)
            stage=f"browser-{i+1}-clinical-updates-source-navigation"
            page.locator("#doctorInboxButton").click()
            expect(page.locator("#doctorPopoverTitle")).to_have_text("Κλινικές ενημερώσεις")
            row=page.locator("#doctorPopoverContent .doctor-action-row").filter(has_text="Δοκιμαστικός Ασθενής 161 2")
            expect(row).to_contain_text("Από την επίσκεψη 7 Οκτωβρίου 2026")
            page.screenshot(path=str(out / f"updates-browser-{i+1}.png"), full_page=True)
            row.click()
            expect(page.locator("#synthetic161Chart .patient-chart-grid")).to_be_visible()
            expect(page.locator("#synthetic161Chart .pending-action")).to_have_text(changed_action)
            if i==1:
                page.locator("#synthetic161Chart .pending-action").click()
                expect(page.locator("#synthetic161Chart .pending-action")).to_have_text(initial_action)
            page.screenshot(path=str(out / f"chart-browser-{i+1}.png"), full_page=True)
            stage = f"browser-{i+1}-patient-switch-isolation"
            page.locator("#visitRecentRows .clinical-patient-row").first.click()
            expect(page.locator("#visitSelectedName")).to_have_text("Δοκιμαστικός Ασθενής 161 1")
            expect(page.locator("#synthetic161Chart")).to_contain_text("2 αποθηκευμένες επισκέψεις")
            assert "chart=cockpit-synthetic-161-1" in page.url
            ctx.close()
            print(f"PASS live browser {i+1}: one-click Brief, persisted lifecycle, source encounter, structured chart, clinical updates navigation, refresh, patient switch")
        browser.close()
    print("PASS LIVE #161: two independent browser sessions against existing PostgreSQL backend")
except Exception:
    # Playwright exception strings can contain arguments supplied to fill().
    # Emit the finite stage only, with no traceback, request bodies or secret.
    print("LIVE #161 FAILED at stage: " + stage)
    sys.exit(1)
