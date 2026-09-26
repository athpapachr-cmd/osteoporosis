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


def test_cockpit_calendar_summary_is_privacy_minimized():
    js = _read("static/cockpit/app.js")

    assert "/clinical/calendar/appointments" in js
    assert "todayOsteoporosisCount" in js
    assert "todayTreatmentCount" in js

    # The Home uses aggregate category counts only. It must not render patient
    # identity fields returned by the detailed Calendar endpoint.
    assert "patient_display_name" not in js
    assert "phone_e164" not in js
    assert "linked_patient_id" not in js

    assert "localStorage" not in js
    assert "sessionStorage" not in js
