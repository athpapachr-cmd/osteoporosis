import json
import subprocess
from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_service_root_enters_global_cockpit_home():
    main = _read("main.py")
    assert 'RedirectResponse(url="/static/cockpit/"' in main
    assert 'request.url.path.startswith("/static/cockpit/")' in main


def test_cockpit_home_has_global_information_architecture():
    html = _read("static/cockpit/index.html")

    assert "Clinical Excellence Cockpit" in html
    assert "Οστεοπόρωση" in html
    assert "Module 01" in html
    assert "/static/baseline-audit/" in html
    assert "/static/clinical-calendar/" in html
    assert "/static/clinical-learning/" in html

    shell = _read("static/cockpit/doctor-shell.js")
    assert "Γραμματεία" in shell
    assert "https://ortho-reception-backend-v2.onrender.com/dashboard" in shell

    assert html.count('/clinical/clinic-utilities/physio-referral') == 1
    assert "Παραπεμπτικό Φυσιοθεραπείας" in html
    assert "/clinical/clinic-utilities/sick-leave" in html
    assert "/clinical/clinic-utilities/medical-report" in html
    assert "/clinical/clinic-utilities/rf" in html


def test_osteoporosis_sidebar_contains_no_global_tools_or_top_level_heidi():
    html = _read("static/baseline-audit/index.html")
    helper = _read("static/baseline-audit/g4-workspace-ergonomics.js")

    assert 'data-nav-action="heidi"' not in html
    assert "Clinic Utilities" not in helper
    assert "/clinical/clinic-utilities/" not in helper


def test_cockpit_today_context_strip_uses_protected_calendar_without_second_calendar():
    html = _read("static/cockpit/index.html")
    js = _read("static/cockpit/app.js")

    assert "/clinical/calendar/cockpit-context" in js
    assert "/clinical/calendar/appointments" not in js
    assert "/static/clinical-calendar/" in html
    assert "Άνοιγμα εβδομαδιαίου ημερολογίου" in html
    for label in ("Προηγούμενο", "Τώρα", "Επόμενο"):
        assert label in html
    for prefix in ("previous", "current", "next"):
        for field in ("Time", "Patient", "Type"):
            assert f'id="{prefix}Appointment{field}"' in html
    assert "patient_display_name" in js
    assert "phone_e164" not in js and "linked_patient_id" not in js
    assert 'timeZone: "Asia/Nicosia"' in js
    assert "current_conflict_count" in js
    assert "ταυτόχρονα ραντεβού" in js
    assert "window.setInterval(loadCockpitContext, 60000)" in js
    for counter in ("todayOsteoporosisCount", "todayTreatmentCount"):
        assert counter not in js and counter not in html
    assert "localStorage" not in js and "sessionStorage" not in js


def _run_home_responses(responses):
    js = _read("static/cockpit/app.js")
    # Execute the real loader/renderer, with a minimal DOM and transport harness.
    source = js[js.index('  const $'):js.index('  async function apiJson')]
    script = f"""
const nodes = {{}};
const document = {{ getElementById(id) {{
  return nodes[id] ||= {{textContent: "", dataset: {{}},
    classList: {{toggle() {{}}, remove() {{}}}}}};
}}}};
const responses = {json.dumps(responses)};
const calls = [];
async function fetch(url, options) {{
  calls.push({{url, options}});
  const response = responses.shift();
  return {{ok: response.ok !== false, status: response.status || 200,
    json: async () => response.body}};
}}
{source}
(async () => {{
  const results = [];
  while (responses.length) {{
    await loadCockpitContext();
    results.push(JSON.parse(JSON.stringify(nodes)));
  }}
  console.log(JSON.stringify({{calls, results}}));
}})();
"""
    return json.loads(subprocess.check_output(["node", "-e", script], text=True))


def _home_context(**overrides):
    context = {"generated_at": "2026-10-03T20:00:00Z", "source_updated_at": "2026-10-03T04:00:00Z",
               "today_total": 1, "previous": None, "current": None, "current_conflict_count": 0,
               "next": {"appointment_id": "synthetic-next", "patient_display_name": "Synthetic Next",
                        "start_at": "2026-10-04T07:00:00Z", "end_at": "2026-10-04T07:40:00Z",
                        "category": "other", "clinic": "limassol", "reason": "Πόνος γόνατος"}}
    return context | overrides


def test_global_home_renders_later_day_and_clears_after_auth_failure_without_fallback():
    result = _run_home_responses([{"body": _home_context()}, {"ok": False, "status": 401}])
    loaded, failed = result["results"]
    assert loaded["nextAppointmentPatient"]["textContent"] == "Synthetic Next"
    assert loaded["nextAppointmentPatient"]["dataset"]["appointmentId"] == "synthetic-next"
    assert "4/10/2026" in loaded["nextAppointmentTime"]["textContent"]
    assert "10:00" in loaded["nextAppointmentTime"]["textContent"]
    assert loaded["nextAppointmentType"]["textContent"] == "limassol · Πόνος γόνατος"
    assert loaded["calendarState"]["textContent"] == "1 σήμερα"
    assert "Τελευταία ανάγνωση κρατήσεων" in loaded["calendarNote"]["textContent"]
    for prefix in ("previous", "current", "next"):
        assert failed[f"{prefix}AppointmentPatient"]["textContent"] == "Μη διαθέσιμο"
        assert failed[f"{prefix}AppointmentPatient"]["dataset"] == {}
    assert all(call == {"url": "/clinical/calendar/cockpit-context",
                        "options": {"credentials": "same-origin"}} for call in result["calls"])


def test_global_home_conflict_stale_and_unknown_freshness_visible():
    results = _run_home_responses([
        {"body": _home_context(current_conflict_count=2, source_updated_at="2026-10-02T04:00:00Z")},
        {"body": _home_context(source_updated_at=None, next=None)},
    ])["results"]
    assert results[0]["currentAppointmentPatient"]["textContent"] == "2 ταυτόχρονα ραντεβού"
    assert "μπορεί να έχει παλιώσει" in results[0]["calendarNote"]["textContent"]
    assert "Δεν υπάρχει διαθέσιμη πρόσφατη ανάγνωση" in results[1]["calendarNote"]["textContent"]
    assert "Δεν υπάρχει επόμενο στο πρόγραμμα" == results[1]["nextAppointmentPatient"]["textContent"]


def test_v3_clinician_workspace_is_existing_home_not_separate_page():
    html = _read("static/cockpit/index.html")
    css = _read("static/cockpit/clinical-workspace.css")
    js = _read("static/cockpit/clinical-workspace.js")
    assert html.count('id="clinicalWorkspace"') == 1
    assert html.count('id="visitRecentRows"') == 1
    assert html.count('id="doctorAttentionList"') == 1
    assert html.count('id="visitPatientSearch"') == 1
    assert '<script src="/static/cockpit/clinical-workspace.js" defer></script>' in html
    assert "Πρόσφατες επισκέψεις" in html
    assert "Χρειάζονται προσοχή" in html
    assert html.count('<span>έως 3</span>') == 1
    assert "/clinical/recent-encounters?limit=3" in js
    shell = _read("static/cockpit/doctor-shell.js")
    assert "/clinical/calendar/appointments" in shell
    assert "Σημερινό πρόγραμμα" in shell
    assert "/clinical/recent-encounters?limit=3" in js
    assert '"/clinical/patients?query="' in js
    assert "clinical-two-panels" in css
    assert 'id="doctorLibraryView"' in html
    assert 'id="doctorPopover"' in html
    assert 'id="doctorDateButton"' in html
    assert 'id="doctorInboxButton"' in html
    assert 'id="doctorActionsButton"' in html
    assert 'id="doctorTasksButton"' in html
    assert 'id="visitDiaComposer"' in html
    assert 'id="visitDiaPrompt"' in html
    assert 'id="visitDiaText"' in html
    assert 'id="visitDiaPreview"' in html
    assert 'data-clinical-tab="snapshot"' in html
    assert 'data-clinical-tab="brief"' in html
    assert 'data-clinical-tab="detail"' in html
    assert 'id="surgeryTableBody"' in html


def test_v3_browser_interaction_uses_mocked_protected_gets_without_writes():
    output = subprocess.check_output(["node", "test_cockpit_unified_home.js"], text=True)
    assert "PASS Cockpit V3" in output


def test_clinician_home_v4_preserves_existing_tools_without_legacy_blocks_under_home():
    html = _read("static/cockpit/index.html")
    shell = _read("static/cockpit/doctor-shell.js")
    css = _read("static/cockpit/doctor-shell.css")
    assert html.index('id="doctorMainView"') < html.index('id="doctorLibraryView"')
    assert 'id="doctorLibraryView" class="doctor-library" hidden' in html
    assert 'id="surgeryForm"' in html and 'id="surgeryTableBody"' in html
    assert "Πρόγραμμα και πηγές" not in html
    assert "Οστεοπόρωση" in html and "Καταγραφή επίσκεψης" in html
    assert "Πλήρες πρόγραμμα Γραμματείας" in shell
    assert "Δεν έχει συνδεθεί ακόμη το Clinical Inbox" in shell
    assert 'doctorInboxBadge").hidden=true' in shell
    assert "last-minute" not in shell
    assert ".doctor-popover" in css and "doctor-rail-open" in css
    assert "max-width:990px" in css or "max-width:990px" in css.replace(" ", "")
    assert "localStorage" not in shell and "sessionStorage" not in shell


def test_clinician_home_v4_synthetic_command_strip_interactions():
    output = subprocess.check_output(["node", "test_cockpit_doctor_shell.js"], text=True)
    assert "PASS Cockpit Doctor Shell" in output
