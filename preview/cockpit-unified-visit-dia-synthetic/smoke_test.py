"""Synthetic-only UI smoke. Requires playwright Python package and Chromium."""
from pathlib import Path
from playwright.sync_api import sync_playwright

source = Path(__file__).with_name("index.html").read_text(encoding="utf-8")

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    errors, outbound = [], []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.on("request", lambda req: outbound.append(req.url))
    page.set_content(source, wait_until="load")
    assert page.locator("#welcomeView").is_visible()
    assert page.locator("#todayList .patient-item").count() == 3
    page.locator("#todayList .patient-item").first.click()
    assert page.locator("#patientName").inner_text() == "Μαρία Παπαδοπούλου"
    page.locator("#fillExample").click()
    assert page.locator("#previewBadge").inner_text() == "Τρεις προβολές διαθέσιμες"
    page.get_by_role("tab", name="Encounter Detail").click()
    assert "Προηγούμενο πλαίσιο" in page.locator("#previewContent").inner_text()
    page.locator("#editPreview").click()
    page.locator("#editInput").fill("Διορθωμένη συνθετική καταγραφή")
    page.locator("#applyEdit").click()
    assert "Διορθωμένη" in page.locator("#previewContent").inner_text()
    page.locator("#reviewCheck").check()
    page.locator("#confirmTrial").click()
    assert page.locator("#completeNotice").is_visible()
    page.locator("#patientSearch").fill("μαρια")
    assert page.locator("#searchResults .patient-item").count() == 2
    page.locator("#searchResults .patient-item").last.click()
    assert not page.locator("#reviewCheck").is_checked()
    assert page.locator("#diaInput").input_value() == ""
    assert not errors and not outbound
    mobile = browser.new_page(viewport={"width": 390, "height": 844})
    mobile.set_content(source, wait_until="load")
    mobile.locator("#todayList .patient-item").first.click()
    mobile.locator("#fillExample").click()
    assert mobile.evaluate("document.documentElement.scrollWidth") <= mobile.evaluate("document.documentElement.clientWidth") + 1
    browser.close()
print("PASS: synthetic calendar -> Dia -> three views -> correction -> finalization; no network")
