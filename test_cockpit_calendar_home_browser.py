"""Synthetic calendar home proof, including category and unavailable source."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import re
from urllib.parse import parse_qs, urlsplit

from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).parent


def test_calendar_home_all_filter_and_unavailable():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for width in (1280, 390):
            page = browser.new_page(viewport={"width": width, "height": 900})
            state = {"fail": False, "lose_session": False, "manual": "osteoporosis_review"}

            def route_request(route):
                parts = urlsplit(route.request.url)
                path = parts.path
                if path == "/clinical/status":
                    return route.fulfill(json={"database_dialect": "synthetic"})
                if path.startswith("/clinical/calendar/"):
                    if route.request.method == "PUT":
                        if state["lose_session"]:
                            return route.fulfill(status=401, json={"detail": "unauthorized"})
                        state["manual"] = json.loads(route.request.post_data)["category"]
                        return route.fulfill(json={"manual_category": state["manual"]})
                    if state["fail"]:
                        return route.fulfill(status=503, json={"detail": {"status": "unavailable"}})
                    start = datetime.fromisoformat(parse_qs(parts.query)["start"][0].replace("Z", "+00:00"))
                    begin = (start + timedelta(hours=9)).astimezone(timezone.utc).isoformat()
                    end = (start + timedelta(hours=9, minutes=30)).astimezone(timezone.utc).isoformat()
                    base = {"start_at": begin, "end_at": end, "clinic": "Synthetic",
                            "reason": "Synthetic", "status": "scheduled", "duration_minutes": 30}
                    rows = [{**base, "appointment_id": "synthetic-ost", "patient_display_name": "Synthetic Osteo",
                             "category": state["manual"] or "osteoporosis_review", "manual_category": state["manual"]}]
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
            page.get_by_role("button", name="Επιλογές ραντεβού · Synthetic Osteo").click()
            expect(page.locator("#appointmentDialog")).to_be_visible()
            expect(page.locator(".classification-select")).to_have_value("osteoporosis_review")
            page.evaluate("""() => {
              window.originalFetch = window.fetch.bind(window);
              window.fetch = (url, opts) => opts?.method === "PUT"
                ? new Promise((resolve, reject) => { window.resolveWrite = resolve; window.rejectWrite = reject; })
                : window.originalFetch(url, opts);
            }""")
            page.locator(".classification-select").select_option("prolia")
            expect(page.locator(".appointment-dialog-close")).to_be_disabled()
            page.keyboard.press("Escape")
            expect(page.locator("#appointmentDialog")).to_be_visible()
            state["manual"] = "prolia"
            page.evaluate("""() => window.resolveWrite(new Response(JSON.stringify({manual_category:"prolia"}), {
              status:200, headers:{"Content-Type":"application/json"}}))""")
            expect(page.locator("#appointmentDialog")).to_be_hidden()
            page.get_by_role("button", name="Επιλογές ραντεβού · Synthetic Osteo").click()
            expect(page.locator(".classification-select")).to_have_value("prolia")
            page.locator(".classification-select").select_option("osteoporosis_first")
            page.evaluate("""() => window.rejectWrite(new Error("synthetic network loss"))""")
            expect(page.locator("#appointmentDialog [role=status]")).to_contain_text("μπορεί να έχει ολοκληρωθεί")
            expect(page.locator(".classification-select")).to_be_disabled()
            page.locator(".appointment-dialog-close").click()
            page.get_by_role("button", name="Επιλογές ραντεβού · Synthetic Osteo").click()
            expect(page.locator(".classification-select")).to_be_disabled()
            page.get_by_role("button", name="Ανανέωση προγράμματος").click()
            expect(page.locator("#appointmentDialog")).to_be_hidden()
            page.evaluate("() => { window.fetch = window.originalFetch; }")
            page.get_by_role("button", name="Επιλογές ραντεβού · Synthetic Osteo").click()
            state["lose_session"] = True
            page.locator(".classification-select").select_option("")
            expect(page.locator("#calendarApp")).to_be_hidden()
            expect(page.locator("#authCard")).to_be_visible()
            expect(page.get_by_text("Synthetic Osteo", exact=True)).to_have_count(0)
            page.reload()
            expect(page.locator("#connectionStatus")).to_contain_text("2 ραντεβού")
            page.locator("#calendarCategory").select_option("osteoporosis")
            expect(page.locator("#connectionStatus")).to_contain_text("1 ραντεβού")
            expect(page.get_by_text("Synthetic Other", exact=True)).to_have_count(0)
            expect(page.get_by_text("Synthetic Osteo", exact=True)).to_be_visible()
            state["fail"] = True
            page.locator("#calendarCategory").select_option("all")
            page.locator("#thisWeek").click()
            expect(page.locator("#connectionStatus")).to_contain_text("Μη διαθέσιμο")
            expect(page.get_by_text("Synthetic Osteo", exact=True)).to_have_count(0)
            expect(page.locator("#countProlia")).to_have_text("—")
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
            page.close()
        browser.close()


def test_busy_week_compact_rows_details_and_keyboard():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for width in (1440, 390):
            page = browser.new_page(viewport={"width": width, "height": 900})

            def route_request(route):
                parts = urlsplit(route.request.url)
                if parts.path == "/clinical/status":
                    return route.fulfill(json={"database_dialect": "synthetic"})
                if parts.path.endswith("/visit-brief/chart-read"):
                    return route.fulfill(status=503, json={"detail":"synthetic unavailable"})
                if parts.path.startswith("/clinical/calendar/"):
                    start = datetime.fromisoformat(parse_qs(parts.query)["start"][0].replace("Z", "+00:00"))
                    rows = []
                    for day in range(5):
                        for index in range(15):
                            begin = start + timedelta(days=day, hours=8, minutes=20 * index)
                            rows.append({"appointment_id": f"busy-{day}-{index}",
                                "start_at": begin.isoformat(), "end_at": (begin + timedelta(minutes=20)).isoformat(),
                                "clinic": "Ιατρείο δοκιμής", "category": "prolia" if index == 0 else "other",
                                "patient_display_name": f"Ασθενής Δοκιμής {day}-{index}",
                                "reason": "  \n\t " if index == 1 else "<img src=x> Πλήρης συνθετικός λόγος για την επίσκεψη με αρκετές επιπλέον λέξεις για δοκιμή συντόμευσης",
                                "manual_category": None})
                    for index in range(2):
                        begin = start + timedelta(hours=16, minutes=20 * index)
                        rows.append({"appointment_id":"duplicate-id", "start_at":begin.isoformat(),
                            "end_at":(begin + timedelta(minutes=20)).isoformat(), "clinic":"Synthetic",
                            "category":"prolia" if index == 0 else "other", "patient_display_name":f"Διπλό δοκιμαστικό {index}",
                            "reason":f"Διαφορετικός λόγος {index}", "manual_category":None})
                    return route.fulfill(json=rows)
                asset = ROOT / "static" / parts.path.removeprefix("/static/")
                if asset.is_dir():
                    asset = asset / "index.html"
                if asset.is_file():
                    mime = "text/html" if asset.suffix == ".html" else (
                        "text/css" if asset.suffix == ".css" else "application/javascript")
                    return route.fulfill(content_type=mime, body=asset.read_text(encoding="utf8"))
                return route.fulfill(status=404, body="")

            page.route("**/*", route_request)
            page.goto("https://cockpit.invalid/static/clinical-calendar/")
            expect(page.locator("#connectionStatus")).to_contain_text("77 ραντεβού")
            rows = page.locator(".appt-row")
            expect(rows).to_have_count(77)
            assert rows.first.bounding_box()["height"] <= 40
            expect(rows.first.locator(".appt-reason")).to_have_text("<img src=x> Πλήρης συνθετικός λόγος για την επίσκεψη με αρκετές επιπλέον λέξεις για δοκιμή συντόμευσης")
            assert rows.first.locator("img").count() == 0
            assert rows.nth(1).locator(".appt-reason").count() == 0
            assert rows.first.locator(".appt-reason").evaluate("(el) => el.scrollWidth > el.clientWidth")
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
            if width == 1440:
                assert rows.nth(14).bounding_box()["y"] + rows.nth(14).bounding_box()["height"] <= 900
            rows.first.click()
            expect(page.locator("#appointmentDialog")).to_be_visible()
            expect(page.locator("#previewReason")).to_contain_text("Πλήρης συνθετικός λόγος")
            assert "/static/clinical-calendar/" in page.url
            page.keyboard.press("Escape")
            page.get_by_role("button", name="Επιλογές ραντεβού · Ασθενής Δοκιμής 0-0").click()
            dialog = page.locator("#appointmentDialog")
            expect(dialog).to_be_visible()
            expect(dialog).to_contain_text("Ασθενής Δοκιμής 0-0")
            expect(dialog).to_contain_text("<img src=x> Πλήρης συνθετικός λόγος για την επίσκεψη με αρκετές επιπλέον λέξεις για δοκιμή συντόμευσης")
            assert dialog.locator("img").count() == 0
            expect(dialog.locator("a.appointment-chart")).to_have_attribute(
                "href", re.compile(r"booking=busy-0-0"))
            page.keyboard.press("Escape")
            expect(dialog).to_be_hidden()
            expect(page.get_by_role("button", name="Επιλογές ραντεβού · Ασθενής Δοκιμής 0-0")).to_be_focused()
            for index in range(2):
                page.get_by_role("button", name=f"Διπλό δοκιμαστικό {index}", exact=False).click()
                expect(dialog).to_contain_text(f"Διπλό δοκιμαστικό {index}")
                expect(dialog).to_contain_text(f"Διαφορετικός λόγος {index}")
                expect(dialog).to_contain_text("εμφανίζεται περισσότερες από μία φορές")
                assert dialog.locator("a.appointment-chart").count() == 0
                assert dialog.locator(".classification-select").count() == 0
                page.keyboard.press("Escape")
            page.locator("#calendarCategory").select_option("osteoporosis")
            expect(page.get_by_role("button", name="Διπλό δοκιμαστικό 0", exact=False)).to_be_visible()
            expect(page.get_by_role("button", name="Διπλό δοκιμαστικό 1", exact=False)).to_have_count(0)
            page.get_by_role("button", name="Διπλό δοκιμαστικό 0", exact=False).click()
            expect(dialog).to_contain_text("εμφανίζεται περισσότερες από μία φορές")
            assert dialog.locator("a.appointment-chart").count() == 0
            assert dialog.locator(".classification-select").count() == 0
            page.close()
        browser.close()


def test_week_views_search_now_and_weekend_duplicate_safety():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.clock.install(time=datetime(2026, 11, 9, 6, 10, tzinfo=timezone.utc))
        state = {"fail": False, "reads": 0}

        def route_request(route):
            parts = urlsplit(route.request.url)
            if parts.path == "/clinical/status":
                return route.fulfill(json={"database_dialect":"synthetic"})
            if parts.path == "/clinical/calendar/agenda":
                state["reads"] += 1
                if state["fail"]:
                    return route.fulfill(status=503, json={"detail":{"status":"unavailable"}})
                start = datetime.fromisoformat(parse_qs(parts.query)["start"][0].replace("Z","+00:00"))
                def item(uid, name, day, minute, duration, category="other", reason="", clinic=""):
                    begin = start + timedelta(days=day, hours=8, minutes=minute)
                    return {"appointment_id":uid,"patient_display_name":name,
                        "start_at":begin.isoformat(),"end_at":(begin+timedelta(minutes=duration)).isoformat(),
                        "clinic":clinic,"category":category,"reason":reason,"manual_category":None}
                return route.fulfill(json=[
                    item("duplicate","Συνθετικός Τώρα",0,0,20,"prolia",clinic="limassol"),
                    item("overlap","Συνθετική Παράλληλη",0,10,20,clinic="evrychou"),
                    item("search","Αλέξης Ιωάννου",1,0,20,reason="Πόνος ώμου",clinic="Άγνωστο"),
                    item("duplicate","Συνθετικός Σάββατο",5,0,20,clinic="enrychou"),
                    item("sunday","Συνθετική Κυριακή",6,0,20),
                ])
            asset = ROOT / "static" / parts.path.removeprefix("/static/")
            if asset.is_dir():
                asset = asset / "index.html"
            if asset.is_file():
                mime = "text/html" if asset.suffix == ".html" else (
                    "text/css" if asset.suffix == ".css" else "application/javascript")
                return route.fulfill(content_type=mime, body=asset.read_text(encoding="utf8"))
            return route.fulfill(status=404, body="")

        page.route("**/*", route_request)
        page.goto("https://cockpit.invalid/static/clinical-calendar/")
        expect(page.locator("#connectionStatus")).to_contain_text("5 ραντεβού")
        expect(page.locator(".day-column")).to_have_count(7)
        expect(page.locator(".day-column.is-today")).to_have_count(1)
        expect(page.locator(".day-column.is-today")).to_contain_text("Δευτέρα")
        expect(page.locator(".appt-row.is-current")).to_have_count(2)
        expect(page.get_by_role("button", name="Συνθετικός Τώρα", exact=False)).to_have_attribute("aria-current","true")
        reads = state["reads"]
        page.locator("#showFiveDays").click()
        expect(page.locator(".day-column")).to_have_count(5)
        expect(page.locator("#weekendNotice")).to_contain_text("2 ραντεβού")
        expect(page.get_by_role("button", name="Συνθετικός Σάββατο", exact=False)).to_have_count(0)
        assert state["reads"] == reads
        assert page.evaluate("localStorage.getItem('clinical-excellence.calendar-day-count.v1')") == "5"
        page.get_by_role("button", name="Συνθετικός Τώρα", exact=False).click()
        expect(page.locator("#appointmentDialog")).to_contain_text("εμφανίζεται περισσότερες από μία φορές")
        assert page.locator("#appointmentDialog a.appointment-chart").count() == 0
        page.keyboard.press("Escape")
        page.locator("#calendarSearch").fill("αλεξης ιωαννου")
        expect(page.locator(".appt-row")).to_have_count(1)
        expect(page.locator(".appt-row")).to_contain_text("Αλέξης Ιωάννου")
        assert state["reads"] == reads
        page.locator("#calendarSearch").fill("πονος ωμου")
        expect(page.locator(".appt-row")).to_have_count(1)
        page.locator("#calendarSearch").fill("")
        page.locator("#calendarClinic").select_option("limassol")
        expect(page.locator(".appt-row")).to_have_count(1)
        expect(page.locator("#countProlia")).to_have_text("1")
        page.locator("#calendarClinic").select_option("evrychou")
        expect(page.locator(".appt-row")).to_have_count(1)
        expect(page.locator(".appt-row")).to_contain_text("Συνθετική Παράλληλη")
        page.locator("#calendarClinic").select_option("unknown")
        expect(page.locator(".appt-row")).to_have_count(1)
        expect(page.locator(".appt-row")).to_contain_text("Αλέξης Ιωάννου")
        page.locator("#calendarClinic").select_option("all")
        page.locator("#calendarCategory").select_option("prolia")
        expect(page.locator(".appt-row")).to_have_count(1)
        page.locator("#calendarCategory").select_option("all")
        page.clock.fast_forward(20 * 60 * 1000)
        expect(page.locator(".appt-row.is-current")).to_have_count(0)
        page.get_by_role("button", name="Προβολή 7 ημερών").click()
        expect(page.locator(".day-column")).to_have_count(7)
        expect(page.get_by_role("button", name="Συνθετικός Σάββατο", exact=False)).to_be_visible()
        page.locator("#showFiveDays").click()
        page.reload()
        expect(page.locator("#showFiveDays")).to_have_attribute("aria-pressed","true")
        expect(page.locator(".day-column")).to_have_count(5)
        page.locator("#calendarSearch").fill("Αλέξης")
        page.evaluate("window.dispatchEvent(new Event('clinical:unauthenticated'))")
        expect(page.locator("#calendarApp")).to_be_hidden()
        expect(page.locator("#calendarSearch")).to_have_value("")
        expect(page.locator(".appt-row")).to_have_count(0)
        page.close()
        browser.close()


def test_clinic_labels_remain_available_without_visit_kind_classification():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        state = {"visit_kind":None, "clinic_override":None, "revision":0}
        def route_request(route):
            parts = urlsplit(route.request.url)
            if parts.path == "/clinical/status":
                return route.fulfill(json={"database_dialect":"synthetic"})
            if parts.path.endswith("/labels"):
                body = json.loads(route.request.post_data)
                assert body["expected_revision"] == state["revision"]
                assert datetime.fromisoformat(body["target_date"]).weekday() == 0
                state[body["field"]] = body["value"]
                state["revision"] += 1
                return route.fulfill(json={"revision":state["revision"]})
            if parts.path == "/clinical/calendar/agenda":
                start = datetime.fromisoformat(parse_qs(parts.query)["start"][0].replace("Z", "+00:00")) + timedelta(hours=9)
                return route.fulfill(json=[{"appointment_id":"cal.com:synthetic", "patient_display_name":"Synthetic",
                    "start_at":start.isoformat(), "end_at":(start+timedelta(minutes=20)).isoformat(),
                    "clinic":"limassol", "category":"other", "reason":"Synthetic reason",
                    "visit_kind":state["visit_kind"], "clinic_override":state["clinic_override"],
                    "classification_revision":state["revision"]}])
            asset = ROOT / "static" / parts.path.removeprefix("/static/")
            if asset.is_dir(): asset = asset / "index.html"
            if asset.is_file():
                mime = "text/html" if asset.suffix == ".html" else ("text/css" if asset.suffix == ".css" else "application/javascript")
                return route.fulfill(content_type=mime, body=asset.read_text(encoding="utf8"))
            return route.fulfill(status=404, body="")
        page.route("**/*", route_request)
        page.goto("https://cockpit.invalid/static/clinical-calendar/")
        assert page.locator("#calendarVisitKind").count() == 0
        expect(page.locator(".appt-row")).to_have_count(1)
        page.get_by_role("button", name="Επιλογές ραντεβού · Synthetic").click()
        assert page.locator('[data-field="visit_kind"]').count() == 0
        page.locator('[data-field="clinic_override"]').select_option("evrychou")
        expect(page.locator("#appointmentDialog")).to_be_hidden()
        page.locator("#calendarClinic").select_option("limassol")
        expect(page.locator(".appt-row")).to_have_count(0)
        page.locator("#calendarClinic").select_option("evrychou")
        expect(page.locator(".appt-row")).to_have_count(1)
        page.get_by_role("button", name="Επιλογές ραντεβού · Synthetic").click()
        expect(page.locator("#appointmentDialog")).to_contain_text("δεν αλλάζει την κράτηση")
        expect(page.locator("#appointmentDialog")).to_contain_text("limassol")
        browser.close()


def test_inline_preview_uses_fresh_booking_and_discards_late_results():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for width in (1280, 390):
            page = browser.new_page(viewport={"width":width,"height":900})
            def route_request(route):
                parts = urlsplit(route.request.url)
                if parts.path == "/clinical/status":
                    return route.fulfill(json={"database_dialect":"synthetic"})
                if parts.path == "/clinical/calendar/agenda":
                    start = datetime.fromisoformat(parse_qs(parts.query)["start"][0].replace("Z", "+00:00")) + timedelta(hours=9)
                    rows = [{"appointment_id":uid,"patient_display_name":name,"start_at":start.isoformat(),
                             "end_at":(start+timedelta(minutes=20)).isoformat(),"clinic":"limassol",
                             "category":"other","reason":"Α"*(994 if uid=="one" else 240)+" ΠΑΛΙΟ"}
                            for uid,name in (("one","Συνθετικός Ένας"),("two","Συνθετικός Δύο"))]
                    return route.fulfill(json=rows)
                asset = ROOT / "static" / parts.path.removeprefix("/static/")
                if asset.is_dir(): asset = asset / "index.html"
                if asset.is_file():
                    mime = "text/html" if asset.suffix == ".html" else ("text/css" if asset.suffix == ".css" else "application/javascript")
                    return route.fulfill(content_type=mime, body=asset.read_text(encoding="utf8"))
                return route.fulfill(status=404, body="")
            page.route("**/*", route_request)
            page.goto("https://cockpit.invalid/static/clinical-calendar/")
            expect(page.locator(".appt-row")).to_have_count(2)
            page.evaluate("""() => {
              const original=window.fetch.bind(window);
              window.fetch=(url,opts)=>{
                if(!String(url).endsWith('/visit-brief/chart-read'))return original(url,opts);
                const uid=JSON.parse(opts.body).booking_uid;
                if(uid==='one')return new Promise(resolve=>{window.resolveFirst=resolve;});
                return Promise.resolve(new Response(JSON.stringify({mode:'chart_linked',linked:true,patient_id:'synthetic-two',patient_name:'Συνθετικός Δύο',booking:{booking_uid:'cal.com:two',
                  patient_display_name:'Συνθετικός Δύο',start_at:'2026-11-10T08:00:00Z',reason_available:'Α'.repeat(240)+' ΝΕΟ',
                  reason_may_be_truncated:false},latest_visit:{source_encounter_id:'visit-two',date:'2026-01-01',reason:'Δεύτερη επίσκεψη',findings:[],decisions:[]},pending:[]}),
                  {status:200,headers:{'Content-Type':'application/json'}}));
              };
            }""")
            page.locator('.appt-row').filter(has_text='Συνθετικός Ένας').click()
            expect(page.locator('#previewReason')).to_contain_text('ΠΑΛΙΟ')
            expect(page.locator('#previewReasonLimit')).to_be_visible()
            page.keyboard.press('Escape')
            page.locator('.appt-row').filter(has_text='Συνθετικός Δύο').click()
            expect(page.locator('#previewReason')).to_contain_text('ΝΕΟ')
            expect(page.locator('#calendarVisitSummary')).to_contain_text('Δεύτερη επίσκεψη')
            assert '/static/clinical-calendar/' in page.url
            page.evaluate("""() => window.resolveFirst(new Response(JSON.stringify({mode:'chart_linked',booking:{
              booking_uid:'cal.com:one',patient_display_name:'Συνθετικός Ένας',start_at:'2026-11-10T08:00:00Z',
              reason_available:'ΠΑΛΑΙΟ ΑΠΟΤΕΛΕΣΜΑ',reason_may_be_truncated:false},
              latest_visit:{date:'2026-01-01',reason:'Πρώτη επίσκεψη',findings:[],decisions:[]},pending:[]}),
              {status:200,headers:{'Content-Type':'application/json'}}))""")
            page.wait_for_timeout(50)
            expect(page.locator('#appointmentDialogTitle')).to_have_text('Συνθετικός Δύο')
            expect(page.locator('#calendarVisitSummary')).not_to_contain_text('Πρώτη επίσκεψη')
            page.keyboard.press('Escape')
            page.evaluate("""() => {
              window.reasonLength=999; window.returnUnauthorized=false; window.malformedKind='';
              window.fetch=(url,opts)=> {
                if(window.returnUnauthorized)return Promise.resolve(new Response('{}',{status:401}));
                if(window.malformedKind){
                  const payload={mode:'chart_linked',linked:true,patient_id:'synthetic-two',patient_name:'Συνθετικός Δύο',
                    booking:{booking_uid:'cal.com:two',patient_display_name:'Συνθετικός Δύο',start_at:'2026-11-10T08:00:00Z',
                      reason_available:'Φρέσκος λόγος',reason_may_be_truncated:false},
                    latest_visit:{source_encounter_id:'visit-two',date:'2026-01-01',reason:'Επίσκεψη',findings:[],decisions:[]},pending:[]};
                  if(window.malformedKind==='pending')delete payload.pending;
                  if(window.malformedKind==='latest')delete payload.latest_visit;
                  if(window.malformedKind==='decisions')delete payload.latest_visit.decisions;
                  return Promise.resolve(new Response(JSON.stringify(payload),{status:200,headers:{'Content-Type':'application/json'}}));
                }
                return Promise.resolve(new Response(JSON.stringify({mode:'booking_only',booking:{booking_uid:'cal.com:two',
                  patient_display_name:'Συνθετικός Δύο',start_at:'2026-11-10T08:00:00Z',reason_available:'λ'.repeat(window.reasonLength),
                  reason_may_be_truncated:window.reasonLength>=1000}}),{status:200,headers:{'Content-Type':'application/json'}}));
              };
            }""")
            page.locator('.appt-row').filter(has_text='Συνθετικός Δύο').click()
            expect(page.locator('#previewReason')).to_have_text('λ'*999)
            expect(page.locator('#previewReasonLimit')).to_be_hidden()
            expect(page.locator('#calendarVisitSummary')).to_contain_text('Δεν υπάρχει επιβεβαιωμένη σύνδεση')
            page.keyboard.press('Escape')
            page.evaluate('window.reasonLength=1000')
            page.locator('.appt-row').filter(has_text='Συνθετικός Δύο').click()
            expect(page.locator('#previewReason')).to_have_text('λ'*1000)
            expect(page.locator('#previewReasonLimit')).to_be_visible()
            page.keyboard.press('Escape')
            page.evaluate('window.reasonLength=0')
            page.locator('.appt-row').filter(has_text='Συνθετικός Δύο').click()
            expect(page.locator('#previewReason')).to_contain_text('Δεν υπάρχει διαθέσιμος λόγος')
            page.keyboard.press('Escape')
            for malformed in ('pending','latest','decisions'):
                page.evaluate('(kind) => window.malformedKind=kind', malformed)
                page.locator('.appt-row').filter(has_text='Συνθετικός Δύο').click()
                expect(page.locator('#calendarVisitSummary')).to_contain_text('Η σύνοψη δεν είναι διαθέσιμη')
                expect(page.locator('#calendarVisitSummary')).not_to_contain_text('Δεν υπάρχουν ανοικτές υποχρεώσεις')
                expect(page.locator('#calendarVisitSummary')).not_to_contain_text('Δεν έχει καταγραφεί προηγούμενο σχέδιο')
                page.keyboard.press('Escape')
            page.evaluate('window.returnUnauthorized=true')
            page.locator('.appt-row').filter(has_text='Συνθετικός Δύο').click()
            expect(page.locator('#authCard')).to_be_visible()
            expect(page.locator('#appointmentDialog')).to_be_hidden()
            assert page.locator('#appointmentDialogContent').inner_text()==''
            page.close()
        browser.close()
