"""Bounded Chromium smoke for static, zero-network synthetic Visit Brief preview."""
from pathlib import Path
import shutil
from playwright.sync_api import sync_playwright, expect

ROOT=Path(__file__).resolve().parent
HTML=(ROOT/"static/cockpit/visit-brief-demo/index.html").read_text(encoding="utf-8")
assert 'connect-src \x27none\x27' in HTML
for forbidden in ("fetch(", "XMLHttpRequest", "localStorage", "sessionStorage", "http://", "https://"):
    assert forbidden not in HTML, f"Preview must be stand-alone and synthetic-only: {forbidden}"
assert "ΣΥΝΘΕΤΙΚΗ ΔΟΚΙΜΗ" in HTML

with sync_playwright() as pw:
    executable=shutil.which("google-chrome") or shutil.which("chromium")
    browser=pw.chromium.launch(headless=True,executable_path=executable,args=["--no-sandbox"])
    for width in (1280,680,390):
        page=browser.new_page(viewport={"width":width,"height":820})
        errs=[]
        outbound=[]
        page.on("pageerror",lambda exc:errs.append(str(exc)))
        page.on("request",lambda request:outbound.append(request.url))
        page.set_content(HTML,wait_until="load")
        expect(page.locator(".slot")).to_have_count(3)
        expect(page.locator("#overlay")).to_be_hidden()
        page.locator("#open").click()
        expect(page.locator("#overlay")).to_be_visible()
        expect(page.locator("#clinicalStage")).to_be_hidden()
        page.locator("#select").click()
        expect(page.locator("#confirm")).to_be_disabled()
        page.locator("#verified").check()
        page.locator("#confirm").click()
        expect(page.locator("#clinicalStage")).to_be_visible()
        expect(page.locator("#pending")).to_contain_text("CD μαγνητικής")
        page.locator("#reopen").click()
        expect(page.locator("#overlay")).to_be_hidden()
        assert page.locator("#pending").text_content()==""
        page.locator(".slot").nth(2).click()
        page.locator("#select").click()
        page.locator("#verified").check()
        page.locator("#confirm").click()
        expect(page.locator("#previous")).to_contain_text("Δεν υπάρχει τεκμηριωμένη")
        page.keyboard.press("Escape")
        expect(page.locator("#overlay")).to_be_hidden()
        assert page.locator("#pending").text_content()==""
        assert page.evaluate("document.documentElement.scrollWidth<=document.documentElement.clientWidth+1")
        assert not outbound,outbound
        assert not errs,errs
        print("Visit Brief demo chromium PASS",width)
        page.close()
    browser.close()
