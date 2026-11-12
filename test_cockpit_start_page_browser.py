"""Chromium proof for selectable Cockpit start pages; no real service or patient data."""
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).parent
START = (ROOT / "static/cockpit/start-page.js").read_text(encoding="utf-8")
NAV = (ROOT / "static/cockpit/shared-nav.js").read_text(encoding="utf-8")
CSS = (ROOT / "static/cockpit/shared-nav.css").read_text(encoding="utf-8")


def _page_html(calendar=False):
    body = ("<main id='calendarApp'></main>" if calendar else
            "<main id='doctorMainView'><div id='visitWorkspaceHome'></div></main>")
    return f"""<!doctype html><html lang='el'><head><meta charset='utf-8'>
      <meta name='viewport' content='width=device-width,initial-scale=1'>
      <style>{CSS}</style><script>{START}</script></head><body>{body}
      <script>{NAV}</script></body></html>"""


def test_selectable_start_pages_in_real_chromium():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        for width in (1280, 390):
            context = browser.new_context(viewport={"width": width, "height": 820})
            page = context.new_page()

            def route_request(route):
                url = route.request.url
                if url.split("?", 1)[0] == "https://cockpit.invalid/static/cockpit/":
                    return route.fulfill(content_type="text/html", body=_page_html())
                if url == "https://cockpit.invalid/static/clinical-calendar/":
                    return route.fulfill(content_type="text/html", body=_page_html(calendar=True))
                if url.endswith("/clinical/staff-session") or url.endswith("/clinical/status"):
                    return route.fulfill(status=401, content_type="application/json",
                                         body='{"detail":"Unauthorized"}')
                return route.abort()

            page.route("**/*", route_request)
            page.goto("https://cockpit.invalid/static/cockpit/")
            set_home = page.locator("#ceSetHome")
            expect(set_home).to_be_visible()
            expect(set_home).to_have_text("✓ Αρχική σελίδα")
            expect(page.locator("#ceGlobalNav a.ce-global-button").first).to_have_attribute(
                "href", "/static/cockpit/?view=day")

            page.goto("https://cockpit.invalid/static/clinical-calendar/")
            expect(set_home).to_be_visible()
            expect(set_home).to_have_text("Ορισμός ως αρχικής")
            set_home.click()
            expect(set_home).to_have_text("✓ Αρχική σελίδα")
            assert page.evaluate("localStorage.getItem('clinical-excellence.start-page.v1')") == "calendar"

            # Generic Cockpit entry now resolves to the current-week Calendar.
            page.goto("https://cockpit.invalid/static/cockpit/")
            page.wait_for_url("https://cockpit.invalid/static/clinical-calendar/")
            expect(page.locator("#ceSetHome")).to_have_text("✓ Αρχική σελίδα")

            # The explicit Day route bypasses a different saved start page.
            page.goto("https://cockpit.invalid/static/cockpit/?view=day")
            expect(page.locator("#ceSetHome")).to_have_text("Ορισμός ως αρχικής")
            page.locator("#ceSetHome").click()
            assert page.evaluate("localStorage.getItem('clinical-excellence.start-page.v1')") == "day"

            for fragment, value in (("#icloud-mail", "inbox"), ("#office-cases", "cases")):
                page.goto("https://cockpit.invalid/static/cockpit/" + fragment)
                expect(page.locator("#ceSetHome")).to_have_text("Ορισμός ως αρχικής")
                page.locator("#ceSetHome").click()
                expect(page.locator("#ceSetHome")).to_have_text("✓ Αρχική σελίδα")
                assert page.evaluate(
                    "localStorage.getItem('clinical-excellence.start-page.v1')") == value

            # Generic entry follows the last fixed same-origin selection.
            page.goto("https://cockpit.invalid/static/cockpit/")
            page.wait_for_url("https://cockpit.invalid/static/cockpit/#office-cases")
            assert page.evaluate("document.querySelectorAll('#ceGlobalNav').length") == 1
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
            context.close()
        browser.close()
