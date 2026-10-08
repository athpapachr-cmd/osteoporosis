from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_cockpit_home_links_visit_capture_surface():
    html = _read("static/cockpit/index.html")
    assert "Καταγραφή επίσκεψης" in html
    assert 'href="/static/cockpit/visit-capture/"' in html


def test_visit_capture_surface_has_one_final_save_and_three_projections():
    html = _read("static/cockpit/visit-capture/index.html")
    assert html.count('id="saveBtn"') == 1
    assert "Αποθήκευση επίσκεψης" in html
    assert 'data-level="snapshot"' in html
    assert 'data-level="brief"' in html
    assert 'data-level="detail"' in html
    assert 'id="candidateInput"' in html
    assert "raw source pages" in html


def test_visit_capture_client_uses_only_protected_local_endpoints_and_transient_candidate():
    js = _read("static/cockpit/visit-capture/app.js")
    assert '"/clinical/visit-capture/context"' in js
    assert '"/clinical/visit-capture/preview"' in js
    assert '"/clinical/visit-capture/save"' in js
    assert '"/clinical/login"' in js
    assert "localStorage" not in js
    assert "sessionStorage" in js
    assert "patient_id: patientId" in js
    assert "candidate: state.candidate" in js
    assert "heidi" not in js.lower()
    assert "gesy" not in js.lower()
    assert "gmail" not in js.lower()
    assert "zadarma" not in js.lower()
    assert "http://" not in js and "https://" not in js


def test_visit_capture_preview_is_automatic_and_save_remains_explicit():
    js = _read("static/cockpit/visit-capture/app.js")
    assert 'addEventListener("input", schedulePreview)' in js
    assert "setTimeout(requestPreview, 250)" in js
    assert 'addEventListener("click", save)' in js
    assert 'preview.can_save' in js
    assert 'state.contextId = ""' in js
