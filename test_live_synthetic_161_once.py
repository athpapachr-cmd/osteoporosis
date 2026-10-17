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
            if i:
                expect(page.locator("#visitBriefIdentity")).to_contain_text("προηγούμενη σύνδεση")
            page.locator("#visitBriefChoosePatient").click()
            page.locator("#visitPatientMatches .clinical-patient-row").click()
            page.locator("#visitBookingIndependentCheck").check()
            page.locator("#visitConfirmBooking").click()
            expect(page.locator("#visitBriefPrevious")).to_contain_text("2026-10-07")
            expect(page.locator("#visitBriefPlan")).to_contain_text("Συμφωνήθηκε επανέλεγχος")
            expect(page.locator("#visitBriefPending")).to_contain_text("awaiting_patient")
            page.screenshot(path=str(out / f"brief-browser-{i+1}.png"), full_page=True)
            stage = f"browser-{i+1}-full-chart-refresh"
            page.locator("#visitBriefOpenChart").click()
            expect(page.locator("#synthetic161Chart")).to_contain_text("2 αποθηκευμένες επισκέψεις")
            page.reload()
            expect(page.locator("#synthetic161Chart")).to_contain_text("2 αποθηκευμένες επισκέψεις")
            expect(page.locator("#synthetic161Chart")).to_contain_text("SAFETY-NET")
            page.screenshot(path=str(out / f"chart-browser-{i+1}.png"), full_page=True)
            stage = f"browser-{i+1}-patient-switch-isolation"
            page.locator("#visitRecentRows .clinical-patient-row").first.click()
            expect(page.locator("#synthetic161Chart")).to_be_empty()
            assert "chart=" not in page.url
            ctx.close()
            print(f"PASS live browser {i+1}: stored appointments, confirmation, Brief, complete chart, refresh, patient switch")
        browser.close()
    print("PASS LIVE #161: two independent browser sessions against existing PostgreSQL backend")
except Exception:
    # Playwright exception strings can contain arguments supplied to fill().
    # Emit the finite stage only, with no traceback, request bodies or secret.
    print("LIVE #161 FAILED at stage: " + stage)
    sys.exit(1)
