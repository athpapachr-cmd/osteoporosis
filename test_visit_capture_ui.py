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
    assert '"/clinical/patients?query="' in js
    assert '"&limit=20&offset="' in js
    assert 'addEventListener("input", schedulePatientSearch)' in js
    assert 'id="morePatientsBtn"' in html
    assert "Εμφανίζονται οι 100" not in js
    assert '"/clinical/patients?limit=100"' not in js


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


def test_visit_capture_does_not_autoload_recent_patients_on_entry():
    html = _read("static/cockpit/visit-capture/index.html")
    js = _read("static/cockpit/visit-capture/app.js")
    assert "Αναζήτηση σε ολόκληρο το μητρώο" in html
    assert 'id="patientSearch"' in html
    assert 'id="patientSelect"' in html
    assert 'id="morePatientsBtn"' in html
    assert '"/clinical/patients?query="' in js
    assert 'if (mode === "record") loadPatients();' in js
    assert "schedulePatientSearch();" in js
    assert 'state.patientSearchRevision' in js
    assert 'if (!term || !state.authenticated || state.mode !== "record") return;' in js
    assert 'const offset = append ? state.patients.length : 0;' in js


def test_dia_prompt_distinguishes_ordered_performed_results_and_frax_mof():
    html = _read("static/cockpit/visit-capture/index.html")
    prompt = html.split('<template id="diaPromptTemplate">', 1)[1].split("</template>", 1)[0]
    assert "SNAPSHOT" in prompt and "VISIT BRIEF" in prompt and "ENCOUNTER DETAIL" in prompt
    assert "παραγγέλθηκε" in prompt
    assert "πραγματοποιήθηκε" in prompt
    assert "αποτελέσματα" in prompt
    assert "ΜΕΙΖΟΝΟΣ οστεοπορωτικού κατάγματος" in prompt
    assert "MOF" in prompt
    assert "κίνδυνο ισχίου" in prompt
    assert "μια ενιαία και εσωτερικά συνεπή" in prompt
    assert "Σημεία προς επιβεβαίωση" in prompt


def test_dia_readability_is_demo_only_and_preserves_original_protected_save():
    html = _read("static/cockpit/visit-capture/index.html")
    js = _read("static/cockpit/visit-capture/app.js")
    css = _read("static/cockpit/visit-capture/styles.css")
    for id_ in ("diaReviewPanel", "diaReviewDetails", "diaReviewCount",
                "diaReviewList", "structuredPreview", "previewText"):
        assert f'id="{id_}"' in html
    assert "Προέρχονται από το Dia — δεν είναι επιβεβαιωμένα λάθη" in html
    assert 'readExplicitReviewNotes(input)' in js
    assert 'sourceSectionHeading(line, level)' in js
    assert 'if (state.mode !== "demo") return;' in js
    assert "line.textContent = item;" in js
    assert "body.textContent = part.body" in js
    assert 'state.mode !== "record" || !state.preview?.can_save' in js
    assert '"/clinical/visit-capture/save"' in js
    assert "localStorage" not in js
    assert ".detail-section" in css
    assert "[hidden]{display:none!important}" in css
