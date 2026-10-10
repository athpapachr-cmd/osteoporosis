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


def test_visit_capture_authenticated_access_collapses_and_can_be_reopened():
    html = _read("static/cockpit/visit-capture/index.html")
    js = _read("static/cockpit/visit-capture/app.js")
    css = _read("static/cockpit/visit-capture/styles.css")
    assert 'id="authToggle"' in html
    assert 'id="authCard"' in html
    assert 'aria-expanded="false"' in html
    assert 'state.authenticated = true' in js
    assert '$("authCard").hidden = true' in js
    assert '$("authToggle").hidden = false' in js
    assert 'addEventListener("click", () => {' in js
    assert '[hidden]{display:none!important}' in css


def test_visit_capture_demo_requires_no_patient_id_and_cannot_save():
    html = _read("static/cockpit/visit-capture/index.html")
    js = _read("static/cockpit/visit-capture/app.js")
    assert 'id="demoModeBtn"' in html
    assert 'id="recordModeBtn"' in html
    assert 'id="candidateInput"' in html
    assert 'id="saveRow" hidden' in html
    assert 'id="patientId"' not in html
    assert 'id="patientSelect"' in html
    assert 'id="patientSearch"' in html
    assert 'state.mode === "demo"' in js
    assert 'state.mode !== "record" || !state.preview?.can_save' in js
    assert 'state.preview = parseDiaSummary(text)' in js
    assert 'state.candidate = null' in js
    assert 'state.patients' in js
    assert '"/clinical/patients?limit=100"' in js


def test_dia_copy_prompt_produces_three_explicit_nonidentifying_sections():
    html = _read("static/cockpit/visit-capture/index.html")
    js = _read("static/cockpit/visit-capture/app.js")
    assert 'id="diaPromptTemplate"' in html
    assert 'id="copyDiaPromptBtn"' in html
    assert 'id="exampleBtn"' in html
    for section in ("SNAPSHOT", "VISIT BRIEF", "ENCOUNTER DETAIL"):
        assert section in html
    assert "μόνο συνθετικό" in html.lower()
    assert 'navigator.clipboard.writeText(prompt)' in js
    assert 'parseDiaSummary(input)' in js
    assert 'previewText").textContent' in js
