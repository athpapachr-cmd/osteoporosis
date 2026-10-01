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

    assert "Reception" in html
    assert "https://ortho-reception-backend-v2.onrender.com/dashboard" in html

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

    assert "/clinical/calendar/appointments" in js
    assert "/static/clinical-calendar/" in html
    assert "Άνοιγμα εβδομαδιαίου ημερολογίου" in html

    for label in ("Προηγούμενο", "Τώρα", "Επόμενο"):
        assert label in html

    for required_id in (
        "previousAppointmentTime",
        "previousAppointmentPatient",
        "previousAppointmentType",
        "currentAppointmentTime",
        "currentAppointmentPatient",
        "currentAppointmentType",
        "nextAppointmentTime",
        "nextAppointmentPatient",
        "nextAppointmentType",
    ):
        assert f'id="{required_id}"' in html

    # D1 may show the appointment display name only after the authenticated
    # Clinical Calendar request succeeds. It does not use phone or protected
    # patient identifiers as a new identity/linkage mechanism.
    assert "patient_display_name" in js
    assert "phone_e164" not in js
    assert "linked_patient_id" not in js

    # Schedule-context semantics: ended -> previous, interval-containing -> now,
    # future start -> next. Overlapping current rows fail closed to an ambiguity
    # message instead of choosing one patient silently.
    assert r"[+-]\d{2}:?\d{2}" in js
    assert r"[+-]\\d{2}:?\\d{2}" not in js
    assert "item.end <= nowMs" in js
    assert "completedItems[completedItems.length - 1]" in js
    assert "item.start <= nowMs && nowMs < item.end" in js
    assert "item.start > nowMs" in js
    assert "ταυτόχρονα ραντεβού" in js
    assert "window.setInterval(loadClinicalCalendarSummary, 60000)" in js

    assert "todayOsteoporosisCount" in js
    assert "todayTreatmentCount" in js
    assert "localStorage" not in js
    assert "sessionStorage" not in js


def test_previous_slot_follows_realistic_schedule_order():
    # Normal clinic schedule is sequential. At 11:30, after 08:00–09:00 and
    # 09:00–10:00 appointments, Previous must be the 09:00–10:00 slot.
    rows = [
        {"patient": "A", "start": 8 * 60, "end": 9 * 60},
        {"patient": "B", "start": 9 * 60, "end": 10 * 60},
    ]
    now_minutes = 11 * 60 + 30
    completed = [row for row in rows if row["end"] <= now_minutes]
    previous = completed[-1]
    assert previous["patient"] == "B"

    js = _read("static/cockpit/app.js")
    assert "const completedItems = items.filter((item) => item.end <= nowMs);" in js
    assert "completedItems[completedItems.length - 1]" in js
    assert "previous: previousItem ? previousItem.row : null" in js
