"""Real Chromium smoke with invented records; no production, Gmail or patient access."""
from pathlib import Path
import shutil
from urllib.parse import urlparse,parse_qs
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).parent
OUT = Path("cockpit-v4-browser-evidence")
OUT.mkdir(exist_ok=True)
HTML = (ROOT / "static/cockpit/index.html").read_text(encoding="utf-8")
CSS = "\n".join((ROOT / part).read_text(encoding="utf-8") for part in (
    "static/cockpit/styles.css",
    "static/cockpit/clinical-workspace.css",
    "static/cockpit/doctor-shell.css",
    "static/cockpit/shared-nav.css",
))
JS = [(ROOT / part).read_text(encoding="utf-8") for part in (
    "static/cockpit/clinical-workspace.js",
    "static/cockpit/visit-brief.js",
    "static/cockpit/app.js",
    "static/cockpit/office-cases.js",
    "static/cockpit/obligations-view.js",
    "static/cockpit/doctor-shell.js",
    "static/cockpit/shared-nav.js",
)]
RECENT = [
    {"patient_id": "SYN-1", "patient_display_name": "Ασθενής Συνθετική",
     "encounter_date": "2026-10-09", "visit_type": "Επανέλεγχος"},
    {"patient_id": "SYN-2", "patient_display_name": "Ασθενής Δοκιμαστικός",
     "encounter_date": "2026-10-08", "visit_type": "Κλινική επίσκεψη"},
]
FUTURE_SURGERY_DATE = (datetime.now(ZoneInfo("Asia/Nicosia")).date() + timedelta(days=2)).isoformat()
RF_COUNTS = {
    "scope": "all", "page": 1, "truncated": False, "denominator": 3,
    "applications": [], "counts": {"issued_unknown": 1, "pending": 1,
        "approved": 1, "rejected": 0, "scheduled": 1, "performed": 1,
        "mine_actionable": 1},
}
SURGERIES = [
    {"surgery_id": "synthetic-surgery-1", "full_name": "Συνθετικός Ασθενής Α",
     "identity_number": "SYNTHETIC", "date_of_birth": "1970-01-01",
     "procedure_type": "Δοκιμή", "laterality": "left", "phone": "000000",
     "surgery_date": None, "queue_position": 1},
    {"surgery_id": "synthetic-surgery-2", "full_name": "Συνθετικός Ασθενής Β",
     "identity_number": "SYNTHETIC", "date_of_birth": "1970-01-01",
     "procedure_type": "Δοκιμή", "laterality": "right", "phone": "000000",
     "surgery_date": FUTURE_SURGERY_DATE, "queue_position": 2},
]
DAY = [
    {"appointment_id": "cal.com:synthetic-1", "patient_display_name": "Ασθενής Δοκιμής Α",
     "start_at": "2026-10-11T07:30:00Z", "reason": "Επανεξέταση"},
    {"appointment_id": "cal.com:synthetic-2", "patient_display_name": "Ασθενής Δοκιμής Β",
     "start_at": "2026-10-11T08:00:00Z", "reason": "Prolia"},
]
OBLIGATION_TODAY = datetime.now(ZoneInfo("Asia/Nicosia")).date()
OBLIGATIONS = [
    {"pending_id":"SYN-OBL-1","patient_id":"SYN-A","patient_name":"Ασθενής Δοκιμής Α",
     "description":"Εργαστηριακές εξετάσεις","item_type":"test","status":"awaiting_result",
     "review_date":(OBLIGATION_TODAY-timedelta(days=1)).isoformat(),"source_date":"2026-10-10"},
    {"pending_id":"SYN-OBL-2","patient_id":"SYN-A","patient_name":"Ασθενής Δοκιμής Α",
     "description":"Επανεξέταση γόνατος","item_type":"follow_up","status":"open",
     "review_date":OBLIGATION_TODAY.isoformat(),"source_date":"2026-10-10"},
    {"pending_id":"SYN-OBL-3","patient_id":"SYN-B","patient_name":"Ασθενής Δοκιμής Β",
     "description":"Απεικονιστική γνωμάτευση","item_type":"document","status":"awaiting_patient",
     "review_date":(OBLIGATION_TODAY+timedelta(days=2)).isoformat(),"source_date":"2026-10-12"},
    {"pending_id":"SYN-OBL-4","patient_id":"SYN-B","patient_name":"Ασθενής Δοκιμής Β",
     "description":"Παραλαβή αποτελέσματος","item_type":"test","status":"awaiting_result",
     "review_date":None,"source_date":"2026-10-12"},
]

CONTEXT = {
    "generated_at": "2026-10-11T06:00:00Z",
    "source_updated_at": "2026-10-11T05:58:00Z",
    "today_total": 2, "previous": None, "current": None,
    "current_conflict_count": 0,
    "next": DAY[0], "upcoming_today": DAY,
}

CALL_PEEK = {
    "source": "reception_dashboard", "is_unread_count": False,
    "coverage": "available_call_log_max_200", "unhandled_count": 2,
    "calls": [
        {"call_id": "RM-SYN-1", "todo_id": "td_RM-SYN-1", "handled": False,
         "logged_at": "2026-10-25T09:00:00+02:00",
         "name": "<img src=x onerror=alert(1)>", "phone": "0000000001",
         "summary": "<svg onload=alert(1)> Ζήτησε ενημέρωση", "summary_pending": False,
         "kind": "message"},
        {"call_id": None, "todo_id": None, "handled": False,
         "logged_at": "2026-10-25T08:00:00+02:00",
         "name": "Συνθετική δεύτερη κλήση", "phone": "",
         "summary": "Η τελική σύνοψη ετοιμάζεται", "summary_pending": True,
         "kind": "information"},
    ],
}

NOTION_TASK = {
    "id": "00000000-0000-4000-8000-000000000001",
    "title": "<img src=x onerror=alert(1)> Συνθετική εργασία",
    "status": "Not started", "due": "2026-10-30", "category": "Ιατρείο",
    "revision": "2026-10-25T08:00:00.000Z",
    "url": "https://www.notion.so/00000000000040008000000000000001",
}


def exercise_notion_tasks(page, width, source):
    page.locator("#doctorNotionButton").click()
    popover = page.locator("#doctorPopover")
    expect(popover).to_be_visible()
    expect(popover).to_contain_text("Συνθετική εργασία")
    assert popover.locator("img").count() == 0, "Notion title must render as text"
    expect(popover.locator(".doctor-notion-entry")).to_have_count(1)
    expect(popover.locator(".doctor-notion-entry a")).to_have_attribute("href", NOTION_TASK["url"])
    page.locator(".doctor-notion-entry").get_by_role("button", name="Ενημέρωση").click()
    expect(popover.locator(".doctor-notion-form")).to_be_visible()
    assert source["writes"] == 0
    popover.locator(".doctor-notion-checkbox input").check()
    popover.locator(".doctor-notion-field textarea").first.fill("Συνθετική ολοκλήρωση")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    if width <= 620:
        popover.screenshot(path=str(OUT / "cockpit-notion-review-mobile.png"))
    popover.get_by_role("button", name="Επιβεβαίωση στο Notion").click()
    expect(popover).to_contain_text("Δεν υπάρχουν ενεργές εργασίες")
    assert source["writes"] == 1
    page.locator("#doctorNotionButton").click()

def assert_delayed_recent_purged_after_logout(page):
    # A successful old clinician response arriving AFTER logout cannot restore PHI.
    assert page.evaluate("""() => {
        const originalFetch = window.fetch.bind(window);
        window.fetch = (url, options) => {
            if (String(url) === "/clinical/recent-encounters?limit=3") {
                return new Promise(resolve => {
                    window.__releaseOldRecent = () => resolve({
                        ok: true,
                        json: async () => [{
                            patient_id: "SYN-OLD",
                            patient_display_name: "SYN-STALE-PHI",
                            encounter_date: "2026-10-10",
                            visit_type: "Synthetic visit"
                        }]
                    });
                });
            }
            return originalFetch(url, options);
        };
        window.CockpitStaffRole = "clinician";
        window.dispatchEvent(new Event("clinical:authenticated"));
        return typeof window.__releaseOldRecent === "function";
    }""")
    assert page.evaluate("""async () => {
        window.dispatchEvent(new Event("clinical:unauthenticated"));
        window.__releaseOldRecent();
        await new Promise(resolve => setTimeout(resolve, 0));
        return !document.getElementById("visitRecentRows").textContent.includes("SYN-STALE-PHI");
    }"""), "old clinician result repopulated protected patient names"
    assert page.locator("#visitRecentRows .clinical-patient-row").count() == 0


def exercise_day_navigation(page,width):
    page.locator('#doctorDateButton').click()
    expect(page.locator('#doctorPopoverContent')).to_contain_text('Ασθενής Δοκιμής Α')
    # Tomorrow browse uses a calendar day, and its booking keeps its own Cyprus date.
    held=[];ranges=[];mode={'value':'normal'}
    def browse_day(route):
        q=parse_qs(urlparse(route.request.url).query);ranges.append(q)
        if mode['value']=='hold':held.append(route);return
        if mode['value']=='error':route.fulfill(status=503,json={'detail':'Unavailable'});return
        if mode['value']=='empty':route.fulfill(json=[]);return
        start=datetime.fromisoformat(q['start'][0].replace('Z','+00:00'))+timedelta(hours=10)
        route.fulfill(json=[{'appointment_id':'cal.com:tomorrow-test','patient_display_name':'Συνθετικό αυριανό ραντεβού','reason':'Προετοιμασία επανελέγχου','start_at':start.isoformat()}])
    page.route('**/clinical/calendar/agenda?*',browse_day)
    page.locator('#doctorDayNext').click();expect(page.locator('#doctorDayLabel')).to_contain_text('Αύριο')
    expect(page.locator('#doctorPopoverContent')).to_contain_text('Συνθετικό αυριανό ραντεβού')
    expect(page.locator('#doctorDayNext')).to_be_focused()
    tomorrow=(datetime.now(ZoneInfo('Asia/Nicosia')).date()+timedelta(days=1)).isoformat()
    start=datetime.fromisoformat(ranges[-1]['start'][0].replace('Z','+00:00'))
    assert start.astimezone(ZoneInfo('Asia/Nicosia')).date().isoformat()==tomorrow
    assert start.astimezone(ZoneInfo('Asia/Nicosia')).hour==0
    page.locator('#doctorPopover').screenshot(path=str(OUT/f'cockpit-tomorrow-{width}.png'))
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.evaluate("window.__dayOpen=CockpitHome.openAppointment;CockpitHome.openAppointment=item=>window.__dayPicked=item")
    page.locator('#doctorPopoverContent button.doctor-action-row').first.click()
    picked=page.evaluate('window.__dayPicked');assert picked['date']==tomorrow and picked['appointmentId']=='cal.com:tomorrow-test'
    page.evaluate('CockpitHome.openAppointment=window.__dayOpen;delete window.__dayPicked;delete window.__dayOpen')
    page.locator('#doctorDateButton').click();expect(page.locator('#doctorDayLabel')).to_contain_text('Σήμερα')
    mode['value']='empty';page.locator('#doctorDayNext').click()
    expect(page.locator('#doctorPopoverContent')).to_contain_text('Δεν υπάρχουν διαθέσιμα ραντεβού')
    mode['value']='error';page.locator('#doctorDayNext').click()
    expect(page.locator('#doctorPopoverContent')).to_contain_text('δεν είναι διαθέσιμο')
    expect(page.locator('#doctorDayPrevious')).to_be_enabled()
    mode['value']='hold';page.locator('#doctorDayPrevious').click()
    page.wait_for_timeout(50);assert held
    mode['value']='normal';page.locator('#doctorDayToday').click()
    expect(page.locator('#doctorPopoverContent')).to_contain_text('Συνθετικό αυριανό ραντεβού')
    held[0].fulfill(json=[{'appointment_id':'old','patient_display_name':'STALE_DAY','start_at':'2026-10-19T09:00:00Z'}])
    page.wait_for_timeout(100);expect(page.locator('#doctorDayLabel')).to_contain_text('Σήμερα')
    expect(page.locator('#doctorPopoverContent')).not_to_contain_text('STALE_DAY')
    page.unroute('**/clinical/calendar/agenda?*',browse_day)
    page.locator('#doctorDateButton').click();page.locator('#doctorDateButton').click()
    expect(page.locator('#doctorPopoverContent')).to_contain_text('Ασθενής Δοκιμής Α')
    page.locator('#doctorDateButton').click()


def exercise_obligations(page,width):
    # Actual obligations JS/CSS with synthetic protected read, no clinical writes.
    page.locator("#doctorTasksButton").click()
    expect(page.locator("#doctorPopover")).to_be_visible()
    expect(page.locator("#doctorPopoverContent button.doctor-action-row")).to_have_count(5)
    expect(page.locator("#doctorPopoverContent")).to_contain_text("Ασθενής Δοκιμής Α")
    expect(page.locator("#doctorObligationsView")).to_be_hidden()
    # Selecting a recent item opens its OWN protected editor in the existing
    # full list; it never forces a chart navigation or trusts stale row data.
    page.locator("#doctorPopoverContent button.doctor-action-row").first.click()
    expect(page.locator("#doctorObligationsView")).to_be_visible()
    peek_modal=page.locator("#doctorObligationDialog")
    expect(peek_modal).to_be_visible()
    expect(peek_modal).to_contain_text("Εργαστηριακές εξετάσεις")
    peek_modal.locator(".doctor-obligation-dialog-close").click()
    expect(peek_modal).to_be_hidden()
    page.locator("#doctorObligationsBack").click()
    expect(page.locator("#doctorMainView")).to_be_visible()
    page.locator("#doctorTasksButton").click()
    expect(page.locator("#doctorPopover")).to_be_visible()
    # The complete view is still a separate, user-selected action.
    page.locator("#doctorPopoverContent").get_by_role("button",name="Όλες οι υποχρεώσεις").click()
    expect(page.locator("#doctorObligationsView")).to_be_visible()
    expect(page.locator("#doctorMainView")).to_be_hidden()
    expect(page.locator("#doctorLibraryView")).to_be_hidden()
    expect(page.locator("#doctorPopover")).to_be_hidden()
    expect(page.locator("#doctorObligationsTotal")).to_have_text("4")
    expect(page.locator("#doctorObligationsOverdue")).to_have_text("1")
    expect(page.locator("#doctorObligationsToday")).to_have_text("1")
    expect(page.locator(".doctor-obligation-row")).to_have_count(4)
    expect(page.locator(".doctor-obligation-group")).to_have_count(2)
    assert page.evaluate("document.documentElement.scrollWidth<=innerWidth+1"), "task-screen horizontal overflow"
    page.locator('[data-doctor-obligation-group="category"]').click()
    expect(page.locator(".doctor-obligation-group")).to_have_count(3)
    expect(page.locator('[data-doctor-obligation-group="category"]')).to_have_attribute("aria-pressed","true")
    examinations=page.locator(".doctor-obligation-group").filter(has=page.locator("h3",has_text="Εξετάσεις"))
    expect(examinations).to_have_count(1)
    expect(examinations.locator(".doctor-obligation-subgroup")).to_have_count(2)
    expect(examinations.locator(".doctor-obligation-subhead")).to_contain_text(["Με προθεσμία", "Χωρίς προθεσμία"])
    assert sorted(examinations.locator(".doctor-obligation-subgroup").evaluate_all(
        "groups => groups.map(g=>g.querySelectorAll('.doctor-obligation-row').length)"))==[1,1]
    page.locator('[data-doctor-obligation-group="date"]').click()
    expect(page.locator(".doctor-obligation-group")).to_have_count(4)
    expect(page.locator("#doctorObligationsRows")).to_contain_text("Εκπρόθεσμες")
    expect(page.locator("#doctorObligationsRows")).to_contain_text("Χωρίς προθεσμία")
    page.locator("#doctorObligationsSearch").fill("Δοκιμής Β")
    expect(page.locator(".doctor-obligation-row")).to_have_count(2)
    page.locator("#doctorObligationsSearch").fill("")
    page.locator("#doctorObligationsStatusFilter").select_option("awaiting_result")
    expect(page.locator(".doctor-obligation-row")).to_have_count(2)
    page.locator("#doctorObligationsStatusFilter").select_option("")
    page.locator("#doctorObligationsTypeFilter").select_option("document")
    expect(page.locator(".doctor-obligation-row")).to_have_count(1)
    page.locator("#doctorObligationsTypeFilter").select_option("")
    # Opening a task stays INSIDE the obligations screen; going to the chart
    # is an optional explicit action rather than forced navigation.
    page.evaluate("""() => {
        window.CockpitStoredChart = {openPending: (p,i) => window.__chosenObligation={patient:p,pending:i}};
    }""")
    page.locator("#doctorObligationsSearch").fill("Δοκιμής Α")
    expect(page.locator('[data-pending-id="SYN-OBL-1"]')).to_be_visible()
    page.locator('[data-pending-id="SYN-OBL-1"]').click()
    expect(page.locator("#doctorObligationsView")).to_be_visible()
    modal=page.locator("#doctorObligationDialog")
    expect(modal).to_be_visible()
    expect(modal).to_contain_text("Εργαστηριακές εξετάσεις")
    expect(modal).to_contain_text("Ασθενής Δοκιμής Α")
    expect(modal).to_contain_text("Για αλλαγές απαιτείται")
    assert modal.bounding_box()["width"] <= width + 1
    # ESC is equivalent to the close button, not just hidden retained PHI.
    page.keyboard.press("Escape")
    expect(modal).to_be_hidden()
    assert modal.text_content()=="" , "Escape retained patient details"
    page.locator('[data-pending-id="SYN-OBL-1"]').click()
    expect(modal).to_be_visible()
    # Changing key can establish a new principal without a preceding explicit
    # logout. Purge old patient details and any reversible-delete authority
    # synchronously before the new authenticated source has returned.
    page.evaluate("""() => {
        document.getElementById("doctorObligationsFeedback").textContent =
            "OLD_PRINCIPAL_UNDO";
        window.dispatchEvent(new Event("clinical:authenticated"));
    }""")
    expect(modal).to_be_hidden()
    assert modal.text_content()=="" , "principal replacement retained old patient's dialog"
    assert "OLD_PRINCIPAL_UNDO" not in page.locator("#doctorObligationsFeedback").text_content()
    assert page.locator("#doctorObligationsSearch").input_value()=="", (
        "successful principal switch retained the prior patient's search text")
    expect(page.locator('[data-pending-id="SYN-OBL-1"]')).to_be_visible()
    page.locator('[data-pending-id="SYN-OBL-1"]').click()
    expect(modal).to_be_visible()
    modal.get_by_role("button",name="Άνοιγμα στον φάκελο ασθενούς").click()
    expect(modal).to_be_hidden()
    assert modal.text_content()=="" , "modal retained clinical text after dismissal"
    expect(page.locator("#doctorObligationsView")).to_be_hidden()
    assert page.locator("#doctorObligationsSearch").input_value()=="" , (
        "hidden patient-name search must be purged on screen close")
    expect(page.locator("#doctorMainView")).to_be_visible()
    assert page.evaluate("window.__chosenObligation")=={"patient":"SYN-A","pending":"SYN-OBL-1"}
    assert page.locator("#doctorObligationsRows").text_content()=="", "hidden screen retained PHI"
    # Closing during a pending protected response must also purge a late success.
    page.evaluate("""() => {
        window.__priorFetch=window.fetch;
        window.fetch=(url,opts) => {
            if(String(url).startsWith('/clinical/clinical-updates?'))
                return new Promise(resolve=>{window.__releaseLateObligations=()=>resolve({
                    ok:true,json:async()=>({items:[{pending_id:'LATE',patient_id:'LATE',
                      patient_name:'STALE_PATIENT_NAME',description:'STALE_TASK',
                      status:'open',item_type:'other'}]})
                });});
            return window.__priorFetch(url,opts);
        };
    }""")
    page.locator("#doctorTasksButton").click()
    assert page.evaluate("typeof window.__releaseLateObligations === 'function'")
    page.locator("#doctorPopoverClose").click()
    page.evaluate("""async () => {
        window.__releaseLateObligations();
        await new Promise(r=>setTimeout(r,0));
        window.fetch=window.__priorFetch;
    }""")
    expect(page.locator("#doctorPopover")).to_be_hidden()
    assert "STALE_PATIENT_NAME" not in page.locator("#doctorPopoverContent").text_content()
    expect(page.locator("#doctorObligationsView")).to_be_hidden()
    assert page.evaluate("document.documentElement.scrollWidth<=innerWidth+1")
    assert not page.locator("#doctorPopoverContent").locator("*").count()
    page.evaluate("delete window.CockpitStoredChart; delete window.__chosenObligation;")


def exercise_reception_call_peek(page, width, call_source):
    """Actual Chromium: DOM, auth failures, stale network, logout and XSS."""
    phone = page.locator("#doctorCallsButton")
    popover = page.locator("#doctorPopover")
    content = page.locator("#doctorPopoverContent")
    links = content.locator("a.doctor-action-row")
    dashboard = "https://ortho-reception-backend-v2.onrender.com/dashboard"

    call_source.update(status=200, handled=False, write_status=503)
    phone.click()
    expect(popover).to_be_visible()
    expect(phone).to_have_attribute("aria-expanded", "true")
    expect(links).to_have_count(3)
    expect(links.first).to_have_attribute("href", dashboard)
    expect(links.first).to_have_attribute("rel", "noopener noreferrer")
    expect(content).to_contain_text("<img src=x onerror=alert(1)>")
    expect(content).to_contain_text("<svg onload=alert(1)>")
    expect(content).to_contain_text("Η τελική σύνοψη ετοιμάζεται")
    expect(content).to_contain_text("2 μη σημειωμένες ως διεκπεραιωμένες")
    expect(content).not_to_contain_text("Χρειάζεται προσοχή")
    assert content.locator("a[href^='tel:']").count() == 1
    assert content.locator("button.doctor-call-control").count() == 1
    content.locator("button.doctor-call-control").click()
    expect(content).to_contain_text("Η ολοκλήρωση δεν επιβεβαιώθηκε")
    assert call_source["handled"] is False
    call_source["write_status"] = 200
    content.locator("button.doctor-call-control").click()
    expect(content).to_contain_text("1 μη σημειωμένες ως διεκπεραιωμένες")
    expect(content).to_contain_text("Σημειώθηκε ως διεκπεραιωμένη")
    assert call_source["handled"] is True
    assert content.locator("img, svg").count() == 0, "source text parsed as unsafe markup"
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    box = popover.bounding_box()
    assert box and box["x"] >= -1 and box["x"] + box["width"] <= width + 1
    if width == 390:
        popover.screenshot(path=str(OUT / "cockpit-reception-calls-mobile.png"))

    page.keyboard.press("Escape")
    expect(popover).to_be_hidden()
    expect(phone).to_be_focused()
    assert content.locator("*").count() == 0, "dismissed caller content retained in DOM"
    phone.click()
    page.evaluate("document.body.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true}))")
    expect(popover).to_be_hidden()
    assert content.locator("*").count() == 0, "outside close retained caller content"

    for code in (401, 403, 503):
        call_source["status"] = code
        phone.click()
        expect(content).to_contain_text(
            "Απαιτείται κλινική σύνδεση" if code in (401, 403)
            else "δεν είναι συνδεδεμένες")
        expect(links).to_have_count(1)
        expect(links.first).to_have_attribute("href", dashboard)
        assert "Συνθετική δεύτερη κλήση" not in content.inner_text()
        phone.click()
        expect(popover).to_be_hidden()
    call_source["status"] = 200

    # An obsolete successful result must not restore caller details after panel switch.
    held = []
    def pause(route):
        held.append(route)
    page.route("**/clinical/reception-calls/peek", pause)
    phone.click()
    page.wait_for_timeout(60)
    assert len(held) == 1, "expected one pending source request"
    page.locator("#doctorInboxButton").click()
    expect(popover).to_be_visible()
    expect(popover).to_have_attribute("data-kind", "inbox")
    held.pop().fulfill(json=CALL_PEEK)
    page.wait_for_timeout(60)
    expect(content).not_to_contain_text("<img src=x")
    page.unroute("**/clinical/reception-calls/peek", pause)
    page.locator("#doctorInboxButton").click()

    # Closing/logout purges the DOM and fences an already-issued late response.
    held = []
    page.route("**/clinical/reception-calls/peek", pause)
    phone.click()
    page.wait_for_timeout(60)
    assert len(held) == 1
    page.evaluate("window.dispatchEvent(new Event('clinical:unauthenticated'))")
    expect(popover).to_be_hidden()
    assert content.locator("*").count() == 0
    held.pop().fulfill(json=CALL_PEEK)
    page.wait_for_timeout(60)
    assert content.locator("*").count() == 0
    assert "<img src=x" not in content.inner_text()
    page.unroute("**/clinical/reception-calls/peek", pause)


def browser_suite(width, screenshot_name):
    errors = []
    outbound = []
    auth = {"authenticated": False, "linked": False, "confirmations": 0}
    call_source = {"status": 200, "handled": False, "write_status": 200}
    notion_source = {"writes": 0}
    with sync_playwright() as pw:
        executable = shutil.which("google-chrome") or shutil.which("chromium")
        browser = pw.chromium.launch(headless=True, executable_path=executable or None, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": width, "height": 820}, device_scale_factor=1)
        page.on("pageerror", lambda err: errors.append(str(err)))
        def handle(route):
            url = route.request.url
            if url.endswith("/clinical/status"):
                return route.fulfill(status=200 if auth["authenticated"] else 401,
                                     json={"authenticated": auth["authenticated"]})
            if url.endswith("/clinical/reception-calls/peek"):
                assert route.request.method == "GET"
                assert not any(k.lower() == "x-clinical-key" for k in route.request.headers)
                code = call_source["status"] if auth["authenticated"] else 401
                source = {**CALL_PEEK, "unhandled_count": 1 if call_source["handled"] else 2,
                    "calls": [{**CALL_PEEK["calls"][0], "handled": call_source["handled"]},
                              CALL_PEEK["calls"][1]]}
                return route.fulfill(status=code,
                    json=source if code == 200 else {"detail": "Unavailable"})
            if "/clinical/office-cases/rf?" in url:
                return route.fulfill(status=200 if auth["authenticated"] else 401,
                    json={"applications": [{"id":"synthetic-rf-office-01", "revision":2,
                        "stage":"scheduled", "patient_name":"RF Συνθετικός",
                        "identity_number":"SYN-001", "laterality":"left",
                        "indication_code":"SYN", "scheduled_date":"2026-11-01",
                        "surgery_id":None}], "next_cursor":None, "has_more":False}
                    if auth["authenticated"] else {"detail":"Unauthorized"})
            if url.endswith("/clinical/reception-calls/handled"):
                assert route.request.method == "POST"
                assert route.request.post_data_json == {"todo_id": "td_RM-SYN-1", "handled": True}
                code = call_source["write_status"] if auth["authenticated"] else 401
                if code == 200:
                    call_source["handled"] = True
                return route.fulfill(status=code, json={"handled": True} if code == 200 else {"detail": "Unavailable"})
            if url.endswith("/clinical/notion-tasks/peek"):
                assert route.request.method == "GET"
                return route.fulfill(status=200 if auth["authenticated"] else 401,
                    json={"source": "notion", "voice_available": False,
                          "database_url": "https://app.notion.com/p/241379c6877a40fc8ab4a0efc042c887",
                          "tasks": [] if notion_source["writes"] else [NOTION_TASK]})
            if url.endswith("/clinical/notion-tasks/confirm"):
                assert route.request.method == "POST"
                body = route.request.post_data_json
                assert body["task_id"] == NOTION_TASK["id"]
                assert body["revision"] == NOTION_TASK["revision"]
                assert body["mark_done"] is True
                assert body["completion_note"] == "Συνθετική ολοκλήρωση"
                notion_source["writes"] += 1
                return route.fulfill(json={"source": "notion", "updated": True,
                    "followup_created": False, "task_url": NOTION_TASK["url"]})
            if url.endswith("/clinical/login"):
                assert route.request.method == "POST"
                assert route.request.post_data_json is not None
                if route.request.post_data_json.get("key") == "synthetic-key":
                    auth["authenticated"] = True
                    return route.fulfill(status=200, json={"authenticated": True, "staff_role": "clinician"})
                return route.fulfill(status=401, json={"detail": "Unauthorized"})
            if "/clinical/calendar/cockpit-context" in url:
                return route.fulfill(json=CONTEXT)
            if url.endswith("/clinical/calendar/visit-brief/read"):
                assert route.request.method == "POST"
                uid = route.request.post_data_json["booking_uid"]
                return route.fulfill(status=200 if auth["authenticated"] else 401, json={
                    "mode": "booking_only",
                    "reason": "needs_reconfirmation" if auth["linked"] else "unlinked",
                    "revision": 1 if auth["linked"] else 0,
                    "linked": auth["linked"], "booking": {"booking_uid": uid,
                      "observation": "a" * 64,
                      "patient_display_name": "Ασθενής Δοκιμής Α",
                      "reason": "Επανεξέταση", "clinic": "Limassol",
                      "start_at": "2026-10-11T07:30:00Z"},
                } if auth["authenticated"] else {"detail": "Unauthorized"})
            if url.endswith("/clinical/staff-session"):
                return route.fulfill(status=200 if auth["authenticated"] else 401,
                    json=({"staff_role": "clinician", "principal_id": "clinic-clinician"}
                          if auth["authenticated"] else {"detail": "Unauthorized"}))
            if url.endswith("/clinical/calendar/visit-brief/patient-search"):
                assert route.request.method == "POST"
                payload = route.request.post_data_json
                assert payload["booking_uid"] == DAY[0]["appointment_id"]
                assert payload["observed_booking"] == "a" * 64
                assert payload["query"].strip()
                return route.fulfill(status=200 if auth["authenticated"] else 401,
                    json=[{"patient_id": "SYN-BRIEF",
                           "demographics": {"full_name": "Ασθενής Δοκιμής Α",
                                            "date_of_birth": "1980-01-01"}}]
                    if auth["authenticated"] else {"detail": "Unauthorized"})
            if "/clinical/patients?query=" in url:
                return route.fulfill(status=200 if auth["authenticated"] else 401,
                    json=[{"patient_id": "SYN-BRIEF",
                      "demographics": {"full_name": "Ασθενής Δοκιμής Α",
                                       "date_of_birth": "1980-01-01"}}]
                    if auth["authenticated"] else {"detail": "Unauthorized"})
            if url.endswith("/clinical/calendar/visit-brief/confirm"):
                assert route.request.method == "POST"
                data = route.request.post_data_json
                assert data["patient_id"] == "SYN-BRIEF"
                assert data["independently_verified"] is True
                assert data["observed_booking"] == "a" * 64
                assert data["expected_revision"] == (1 if auth["linked"] else 0)
                auth["linked"] = True
                auth["confirmations"] += 1
                return route.fulfill(status=200, json={
                    "mode": "confirmed_once", "revision": auth["confirmations"],
                    "booking": {
                      "booking_uid": data["booking_uid"],
                      "patient_display_name": "Ασθενής Δοκιμής Α",
                      "reason": "Επανεξέταση", "clinic": "Limassol",
                      "start_at": "2026-10-11T07:30:00Z"},
                    "brief": {"patient_id": "SYN-BRIEF", "check_heading": "Για το ραντεβού 2026-10-11",
                      "previous_visit": {"date": "2026-10-10",
                        "reason": "Προηγούμενη υπογεγραμμένη επίσκεψη",
                        "findings": ["Προηγούμενο εύρημα"]},
                      "prior_plan": {"decisions": ["Κλινικός επανέλεγχος"],
                         "medications": [], "next_contact": {
                           "when": "2026-10-11", "purpose": "Επανεξέταση"}},
                      "historical_plan": None,
                      "pending": [{"description": "Εργαστηριακά",
                          "status": "awaiting_result", "review_date": "2026-10-11"}],
                      "appointment_checks": [{"text": "Επανεξέταση",
                          "kind": "previous_next_contact"}],
                      "partial": False}
                })
            if "/clinical/recent-encounters" in url:
                return route.fulfill(json=RECENT)
            if "/clinical/calendar/agenda?" in url:
                return route.fulfill(json=DAY)
            if "/clinical/clinical-updates?" in url:
                if not auth["authenticated"]:
                    return route.fulfill(status=401,json={"detail":"Unauthorized"})
                args=parse_qs(urlparse(url).query)
                rows=OBLIGATIONS if args.get("scope",["real"])[0]=="real" else []
                assert args.get("active_only")==["true"]
                return route.fulfill(json={"items":rows,"shown":len(rows)})
            if url.endswith("/clinical/surgeries"):
                return route.fulfill(status=200 if auth["authenticated"] else 401,
                                     json=SURGERIES if auth["authenticated"] else {"detail": "Unauthorized"})
            if url.endswith("/clinical/clinic-utilities/rf/api/outcomes/search"):
                assert route.request.method == "POST"
                assert route.request.post_data_json == {"query": "", "scope": "all", "page": 1}
                return route.fulfill(status=200 if auth["authenticated"] else 401,
                                     json=RF_COUNTS if auth["authenticated"] else {"detail": "Unauthorized"})
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

        # Exercise the ACTUAL shared JS/CSS inside Chromium, not just the
        # synthetic DOM test. The original Home smoke omitted these assets.
        nav = page.locator("#ceGlobalNav")
        expect(nav).to_be_visible()
        expect(nav.locator("a.ce-global-button").first).to_have_attribute("href", "/static/cockpit/")
        expect(page.locator("#ceGlobalShortcuts")).to_be_visible()
        expect(page.locator("#ceGlobalKey")).to_be_visible()
        assert page.evaluate("""() => {
            const bar = document.getElementById('ceGlobalNav');
            const box = bar.getBoundingClientRect();
            return getComputedStyle(bar).position === 'sticky'
                && box.left >= -1 && box.right <= innerWidth + 1
                && bar.scrollWidth <= bar.clientWidth + 1;
        }"""), "shared toolbar overflows the viewport or is not sticky"
        page.locator("#ceGlobalShortcuts").click()
        expect(page.locator("#ceGlobalShortcutsPanel")).to_be_visible()
        assert page.locator("#ceGlobalShortcutsPanel a.ce-global-shortcut").count() == 8
        panel = page.locator("#ceGlobalShortcutsPanel").bounding_box()
        assert panel and panel["x"] >= -1 and panel["x"] + panel["width"] <= width + 1, "shortcuts popover off-screen"
        page.keyboard.press("Escape")
        expect(page.locator("#ceGlobalShortcutsPanel")).to_be_hidden()
        page.locator("#ceGlobalKey").click()
        expect(page.locator("#ceGlobalAccessPanel")).to_be_visible()
        expect(page.locator("#ceGlobalAuthStatus")).to_contain_text("Απαιτείται σύνδεση")
        panel = page.locator("#ceGlobalAccessPanel").bounding_box()
        assert panel and panel["x"] >= -1 and panel["x"] + panel["width"] <= width + 1, "key popover off-screen"
        if width <= 620:
            page.screenshot(path=str(OUT / "cockpit-shared-nav-mobile.png"), full_page=True)
        if width == 1280:
            key = page.locator("#ceGlobalKeyInput")
            key.fill("wrong-synthetic-key")
            page.locator("#ceGlobalLoginForm button[type=submit]").click()
            expect(page.locator("#ceGlobalAuthStatus")).to_contain_text("Λανθασμένο Clinical Key")
            expect(key).to_have_value("")
        key = page.locator("#ceGlobalKeyInput")
        key.fill("synthetic-key")
        page.locator("#ceGlobalLoginForm button[type=submit]").click()
        expect(page.locator("#ceGlobalAuthStatus")).to_contain_text("Συνδέθηκες")
        expect(key).to_have_value("")
        page.keyboard.press("Escape")
        expect(page.locator("#ceGlobalAccessPanel")).to_be_hidden()
        # End-to-end source-only → protected registry → independently checked
        # explicit confirmation → brief → reopen booking-only, at all 3 widths.
        page.locator("#visitSidebarNext").click()
        expect(page.locator("#visitBriefOverlay")).to_be_visible()
        expect(page.locator("#visitBriefIdentity")).to_contain_text("Δεν υπάρχει επιβεβαιωμένη σύνδεση")
        expect(page.locator("#visitBriefClinical")).to_be_hidden()
        page.locator("#visitBriefChoosePatient").click()
        expect(page.locator("#visitBriefOverlay")).to_be_hidden()
        expect(page.locator("#visitPatientMatches .clinical-patient-row")).to_have_count(1)
        if width <= 990:
            expect(page.locator("#visitSidebar")).to_be_hidden()
        # The full-registry search belongs to central Home, not the collapsed rail.
        page.locator("#visitPatientMatches .clinical-patient-row").first.click()
        expect(page.locator("#visitBookingConfirmation")).to_be_visible()
        if width <= 990:
            expect(page.locator("#visitSidebar")).to_be_hidden()
        page.locator("#visitConfirmBooking").click()
        expect(page.locator("#visitBookingConfirmStatus")).to_contain_text("Χρειάζεται ανεξάρτητη")
        assert auth["confirmations"] == 0
        page.locator("#visitBookingIndependentCheck").check()
        page.locator("#visitConfirmBooking").click()
        expect(page.locator("#visitBriefOverlay")).to_be_visible()
        expect(page.locator("#visitBriefClinical")).to_be_visible()
        expect(page.locator("#visitBriefPrevious")).to_contain_text("Προηγούμενη υπογεγραμμένη")
        expect(page.locator("#visitBriefChecks")).to_contain_text("Επανεξέταση")
        expect(page.locator("#visitBriefOpenChart")).to_be_visible()
        expect(page.locator("#visitBriefUnlink")).to_be_visible()
        assert auth["confirmations"] == 1
        page.locator("#visitBriefClose").click()
        expect(page.locator("#visitBriefClinical")).to_be_hidden()
        # P1: hidden DOM is not a privacy boundary, including at 390/680/1280.
        for node in ("visitBriefName", "visitBriefWhen", "visitBriefReason",
                     "visitBriefFreshness", "visitBriefIdentity", "visitBookingContext"):
            assert page.locator("#" + node).text_content() == "", f"stale DOM {node}"
        assert page.locator("#visitBriefPrevious").locator("*").count() == 0
        assert page.locator("#visitBriefPlan").locator("*").count() == 0
        assert not page.locator("#visitBookingIndependentCheck").is_checked()
        expect(page.locator("#visitBookingConfirmation")).to_be_hidden()
        page.locator("#visitBackHome").click()
        page.locator("#visitSidebarNext").click()
        expect(page.locator("#visitBriefIdentity")).to_contain_text("νέα ανεξάρτητη επιβεβαίωση")
        expect(page.locator("#visitBriefClinical")).to_be_hidden()
        page.evaluate("window.CockpitStaffRole='secretary'")
        page.locator("#visitBriefClose").click()
        page.locator("#visitSidebarNext").click()
        expect(page.locator("#visitBriefUnlink")).to_be_hidden()
        page.evaluate("window.CockpitStaffRole='clinician'")
        assert auth["confirmations"] == 1
        page.locator("#visitBriefClose").click()
        expect(page.locator("#visitBriefOverlay")).to_be_hidden()
        exercise_day_navigation(page,width)
        exercise_obligations(page,width)
        exercise_notion_tasks(page,width,notion_source)
        if width <= 620:
            expect(page.locator("#doctorAttentionList .doctor-action-row")).to_have_count(4)
            expect(page.locator("#doctorAttentionList")).to_contain_text("Συνθετικός Ασθενής Β")
            expect(page.locator("#doctorAttentionList")).to_contain_text("Εκδοθείσες αιτήσεις")
            expect(page.locator("#doctorAttentionList")).to_contain_text("Δεν υπάρχει ακόμη αξιόπιστη καταμέτρηση")
            assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1")
            # Actual Chromium: hidden registry and Dia details are removed on auth loss.
            exercise_reception_call_peek(page, width, call_source)
            assert_delayed_recent_purged_after_logout(page)
            page.evaluate("""() => {
                document.getElementById('visitDiaPreviewText').textContent='SYN-PHI';
                document.getElementById('visitSelectedName').textContent='SYN-PHI';
                window.dispatchEvent(new Event('clinical:unauthenticated'));
            }""")
            assert page.locator("#visitPatientMatches").locator("*").count() == 0
            assert page.locator("#visitDiaPreviewText").text_content() == ""
            assert page.locator("#visitSelectedName").text_content() == ""
            assert not errors, errors
            assert not any("/clinical/visit-capture/save" in url for url in outbound)
            browser.close()
            return

        expect(page.locator("#doctorMainView")).to_be_visible()
        expect(page.locator("#doctorLibraryView")).to_be_hidden()
        expect(page.locator("#visitRecentRows .clinical-patient-row")).to_have_count(2)
        # Split width must show appointments in primary Home, not hide them in ☰.
        assert page.locator("#visitRecentRows").evaluate("(el) => el.closest('#visitSidebar') !== null")
        assert page.locator("#visitSidebarNext").evaluate("(el) => el.closest('#visitWorkspaceHome') !== null")
        for slot in ("Previous","Current","Next"):
            expect(page.locator("#visitSidebar"+slot)).to_be_visible()
        expect(page.locator("#nextAppointmentPatient")).to_contain_text("Ασθενής Δοκιμής Α")
        if width <= 990:
            expect(page.locator("#visitSidebar")).to_be_hidden()
            expect(page.locator("#visitRecentRows")).to_be_hidden()
        assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1"), "calendar swap horizontal overflow"

        expect(page.locator("#doctorAttentionList .doctor-action-row")).to_have_count(4)
        expect(page.locator("#doctorAttentionList")).to_contain_text("Συνθετικός Ασθενής Β")
        expect(page.locator("#doctorAttentionList")).to_contain_text("Αναμένουν απόφαση")
        expect(page.locator("#doctorAttentionList")).not_to_contain_text("Υποβλήθηκαν")
        expect(page.locator("#doctorTasksBadge")).to_be_hidden()  # surgery count is not the patient obligation badge
        page.screenshot(path=str(OUT / screenshot_name), full_page=True)

        page.locator("#doctorDateButton").click()
        expect(page.locator("#doctorPopover")).to_be_visible()
        expect(page.locator("#doctorPopoverContent")).to_contain_text("Ασθενής Δοκιμής Α")
        assert "προβολή της Γραμματείας" in page.locator("#doctorPopoverContent").inner_text()
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

        page.locator("#doctorActionsButton").click()
        page.locator("#doctorPopoverContent").get_by_role("button", name="Περιστατικά ιατρείου").click()
        expect(page.locator("#doctorMainView")).to_be_hidden()
        expect(page.locator("#surgerySection")).to_be_visible()
        page.locator('[data-office-tab="rf"]').click()
        expect(page.locator("#officeRfList")).to_contain_text("RF Συνθετικός")
        page.get_by_role("button", name="Προσθήκη στη λίστα χειρουργείων").click()
        expect(page.locator('[data-office-tab-panel="surgery"]')).to_be_visible()
        expect(page.locator("#surgeryFullName")).to_have_value("RF Συνθετικός")
        expect(page.locator("#surgeryIdentityNumber")).to_have_value("SYN-001")
        expect(page.locator("#surgeryDate")).to_have_value("2026-11-01")
        expect(page.locator("#surgeryDateOfBirth")).to_have_value("")
        expect(page.locator("#surgeryPhone")).to_have_value("")
        expect(page.locator("#surgeryProcedureType")).to_have_value("")
        page.locator("#doctorBackHome").click()

        if width <= 990:
            expect(page.locator("#visitSidebar")).to_be_hidden()
            page.locator("#doctorRailToggle").click()
            expect(page.locator("#visitSidebar")).to_be_visible()
            page.locator("#visitRecentRows .clinical-patient-row").first.click()
            expect(page.locator("#visitSidebar")).to_be_hidden()
            expect(page.locator("#visitWorkspacePatient")).to_be_visible()
            page.locator("#visitBackHome").click()
            expect(page.locator("#visitSidebarNext")).to_be_visible()
        else:
            expect(page.locator("#visitSidebarPrevious")).to_be_disabled()
            expect(page.locator("#visitBriefOverlay")).to_be_hidden()
        assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1"), "horizontal overflow"
        exercise_reception_call_peek(page, width, call_source)
        assert_delayed_recent_purged_after_logout(page)
        page.evaluate("""() => {
            document.getElementById('visitDiaPreviewText').textContent='SYN-PHI';
            document.getElementById('visitSelectedName').textContent='SYN-PHI';
            window.dispatchEvent(new Event('clinical:unauthenticated'));
        }""")
        assert page.locator("#visitPatientMatches").locator("*").count() == 0
        assert page.locator("#visitDiaPreviewText").text_content() == ""
        assert page.locator("#visitSelectedName").text_content() == ""
        assert not errors, errors
        assert not any("/clinical/visit-capture/save" in url for url in outbound)
        browser.close()

browser_suite(1280, "cockpit-desktop.png")
browser_suite(680, "cockpit-dia-halfwidth.png")
browser_suite(390, "cockpit-shared-nav-mobile.png")
print("PASS Cockpit V4 Chromium + actual shared toolbar: desktop, half-width, mobile, key and popovers")
