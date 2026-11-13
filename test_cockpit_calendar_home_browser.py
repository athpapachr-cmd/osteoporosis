"""Synthetic calendar home proof, including category and unavailable source."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).parent


def test_calendar_home_all_filter_and_unavailable():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for width in (1280, 390):
            page = browser.new_page(viewport={"width": width, "height": 900})
            state = {"fail": False}

            def route_request(route):
                parts = urlsplit(route.request.url)
                path = parts.path
                if path == "/clinical/status":
                    return route.fulfill(json={"database_dialect": "synthetic"})
                if path.startswith("/clinical/calendar/"):
                    if route.request.method == "PUT":
                        return route.fulfill(status=401, json={"detail": "unauthorized"})
                    if state["fail"]:
                        return route.fulfill(status=503, json={"detail": {"status": "unavailable"}})
                    start = datetime.fromisoformat(parse_qs(parts.query)["start"][0].replace("Z", "+00:00"))
                    begin = (start + timedelta(hours=9)).astimezone(timezone.utc).isoformat()
                    end = (start + timedelta(hours=9, minutes=30)).astimezone(timezone.utc).isoformat()
                    base = {"start_at": begin, "end_at": end, "clinic": "Synthetic",
                            "reason": "Synthetic", "status": "scheduled", "duration_minutes": 30}
                    rows = [{**base, "appointment_id": "synthetic-ost", "patient_display_name": "Synthetic Osteo",
                             "category": "osteoporosis_review", "manual_category": "osteoporosis_review"}]
                    if path.endswith("/agenda"):
                        rows.append({**base, "appointment_id": "synthetic-other",
                                     "patient_display_name": "Synthetic Other", "category": "other"})
                    return route.fulfill(json=rows)
                relative = path.removeprefix("/static/")
                asset = ROOT / "static" / relative
                if asset.is_dir():
                    asset = asset / "index.html"
                if asset.is_file():
                    mime = "text/html" if asset.suffix == ".html" else (
                        "text/css" if asset.suffix == ".css" else "application/javascript")
                    return route.fulfill(content_type=mime, body=asset.read_text(encoding="utf8"))
                return route.fulfill(status=404, body="")

            page.route("**/*", route_request)
            page.goto("https://cockpit.invalid/static/clinical-calendar/")
            expect(page.locator("#connectionStatus")).to_contain_text("2 ραντεβού")
            expect(page.get_by_text("Synthetic Other", exact=True)).to_be_visible()
            expect(page.locator(".classification-select")).to_have_value("osteoporosis_review")
            page.locator(".classification-select").select_option("")
            expect(page.locator("#calendarApp")).to_be_hidden()
            expect(page.locator("#authCard")).to_be_visible()
            expect(page.get_by_text("Synthetic Osteo", exact=True)).to_have_count(0)
            page.reload()
            expect(page.locator("#connectionStatus")).to_contain_text("2 ραντεβού")
            page.locator("#showOsteoporosisAppointments").click()
            expect(page.locator("#connectionStatus")).to_contain_text("1 ραντεβού")
            expect(page.get_by_text("Synthetic Other", exact=True)).to_have_count(0)
            expect(page.get_by_text("Synthetic Osteo", exact=True)).to_be_visible()
            state["fail"] = True
            page.locator("#showAllAppointments").click()
            expect(page.locator("#connectionStatus")).to_contain_text("Μη διαθέσιμο")
            expect(page.get_by_text("Synthetic Osteo", exact=True)).to_have_count(0)
            expect(page.locator("#countProlia")).to_have_text("—")
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
            page.close()
        browser.close()
