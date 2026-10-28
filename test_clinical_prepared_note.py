"""Synthetic no-provider acceptance for the two clinical note entry paths."""
from copy import deepcopy
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from clinical_data import EncounterORM, ClinicalPendingORM
from clinical_prepared_note import note_sections, note_tasks, prepared_candidate
from clinical_capture_cp1 import TEST_ID
from test_cockpit_synthetic_161 import setup, login, KEY
from test_cockpit_cp1 import ctx

NOTE = '''Λόγος επίσκεψης:
Θωρακικός πόνος εδώ και 3 μήνες.
Ατομικό ιστορικό:
Κάταγμα ΠΧΚ πριν δύο χρόνια. Εμμηνόπαυση στα 50.
Τρέχουσα αγωγή:
Αμλοδιπίνη. Δεν λαμβάνει alendronate ακόμη.
Συμπτώματα:
Επιδείνωση στην ορθοστασία, χωρίς νευρολογική σημειολογία.
Κλινική εξέταση:
Ύψος 1,58 m. Βάρος 61 kg. Ευαισθησία Θ8–Θ10.
Απεικονιστικός έλεγχος:
T-score −2,8 στην οσφυϊκή μοίρα και −2,3 στον αυχένα μηριαίου.
Εργαστηριακός έλεγχος:
Βιταμίνη D 18 ng/ml.
Εκτίμηση:
Καταγεγραμμένη εκτίμηση προς έλεγχο.
Νέα φαρμακευτική αγωγή:
Σύσταση για έναρξη alendronate μετά από εκτίμηση από οδοντίατρο.
Πλάνο:
Ζητήθηκαν ακτινογραφίες. Δεν υπάρχουν ακόμη αποτελέσματα.
Υποχρεώσεις:
- [ ] Προσκόμιση ακτινογραφιών από τον ασθενή σε 3 μήνες,
  μετά την πραγματοποίησή τους — δεν έχουν ακόμη γίνει.
- [x] Η ήδη ολοκληρωμένη ενέργεια δεν ανοίγει ξανά.
Παρακολούθηση:
Έλεγχος πόνου και σωστής λήψης αγωγής όταν αρχίσει.
Απρόβλεπτη ενότητα:
Ολόκληρο το άγνωστο περιεχόμενο: χωρίς περικοπή.
Απρόβλεπτη ενότητα:
Και η επανάληψη παραμένει. <script>alert("text only")</script>
  ## Προτάσεις AI για επανέλεγχο — δεν αποτελούν αποφάσεις:
- [ ] Ενδεικτική σκέψη AI: να διευκρινιστούν τυχόν νέες πτώσεις.
'''


def test_exact_note_import_preview_atomic_save_independent_read_no_provider(setup, monkeypatch):
    app, engine = setup
    def forbidden(*a, **kw):
        pytest.fail('Direct import called AI')
    monkeypatch.setattr('clinical_capture_cp1.extract_candidate', forbidden)
    monkeypatch.setenv('CLINICAL_DOCUMENTS_AI_ENABLED', 'false')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    c = login(app); c.post('/clinical/visit-capture/test-patient')
    context = ctx(c)
    imported = c.post('/clinical/visit-capture/import', json={'context_id': context, 'text': NOTE, 'encounter_date': '2026-10-28'})
    assert imported.status_code == 200, imported.text
    assert imported.headers['cache-control'] == 'no-store'
    candidate = imported.json()['candidate']
    assert candidate['prepared_note']['text'] == NOTE
    assert imported.json()['detail'] == imported.json()['brief'] == NOTE
    assert len(candidate['pending']) == 1
    assert 'μετά την πραγματοποίησή τους' in candidate['pending'][0]['description']
    assert candidate['pending'][0]['source_excerpt'] in NOTE
    assert candidate['medications'] == []  # conditional therapy stays exact, never an invented start
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(EncounterORM)) == 0
        assert db.scalar(select(func.count()).select_from(ClinicalPendingORM)) == 0
    body = {'context_id': context, 'candidate': candidate}
    preview = c.post('/clinical/visit-capture/preview', json=body).json()
    assert preview['can_save'] and preview['gesy_copy'] == NOTE
    saved = c.post('/clinical/visit-capture/save', json=body)
    assert saved.status_code == 200, saved.text
    assert saved.json()['gesy_copy'] == NOTE
    assert c.post('/clinical/visit-capture/save', json=body).status_code == 409
    chart = login(app).get('/clinical/patient/'+TEST_ID+'/chart').json()
    assert chart['visits'][0]['clinical']['prepared_note']['text'] == NOTE
    assert len(chart['pending']) == 1


def test_limits_auth_source_binding_and_stale_context_are_private(setup):
    app, _ = setup; c = login(app); c.post('/clinical/visit-capture/test-patient'); context = ctx(c)
    body = {'context_id': context, 'text': NOTE, 'encounter_date': '2026-10-28'}
    assert TestClient(app).post('/clinical/visit-capture/import', json=body).status_code == 401
    for text in [' ', 'x'*30001, 'Υποχρεώσεις:\n- [ ] '+'s'*1001, 'Υποχρεώσεις:\n'+'- [ ] task\n'*61]:
        r = c.post('/clinical/visit-capture/import', json={**body, 'text': text})
        assert r.status_code == 422 and (not text.strip() or text not in r.text)
    candidate = prepared_candidate(NOTE, '2026-10-28').model_dump(mode='json')
    forged = deepcopy(candidate); forged['source_bindings'][0]['role'] = 'web_visit_audio'
    assert c.post('/clinical/visit-capture/preview', json={'context_id': context, 'candidate': forged}).status_code == 422
    forged = deepcopy(candidate); forged['prepared_note'] = None
    assert c.post('/clinical/visit-capture/save', json={'context_id': context, 'candidate': forged}).status_code == 422
    c.post('/clinical/patients', json={'patient_id': TEST_ID, 'demographics': {'full_name': 'Synthetic changed'}})
    assert c.post('/clinical/visit-capture/import', json=body).status_code == 409


def test_unknown_markdown_inline_repeated_sections_and_multiline_tasks():
    text = 'Πρόλογος\n# Ιστορικό\nΠΧΚ δύο έτη.\n**Συμπτώματα:** Πόνος 3 μήνες\nΆγνωστο: Α\nΆγνωστο: Β\n'
    sections = note_sections(text)
    assert [title for title, _ in sections] == ['', 'Ιστορικό', 'Συμπτώματα', 'Άγνωστο', 'Άγνωστο']
    assert 'Πόνος 3 μήνες' in sections[2][1]
    assert note_tasks('Υποχρεώσεις:\n- [ ] εξέταση\n  μόνο μετά τον έλεγχο\nΠροτάσεις AI:\n- [ ] ιδέα') == [
        ('εξέταση\n  μόνο μετά τον έλεγχο', '- [ ] εξέταση\n  μόνο μετά τον έλεγχο')]
    assert note_tasks('Πλάνο:\n- [ ] Δεν ενεργοποιείται εδώ.\nΥποχρεώσεις:\nΚαμία.\n- [x] Έγινε.') == []


@pytest.mark.parametrize('title', ['  ## Προτάσεις AI για επανέλεγχο — δεν αποτελούν αποφάσεις:',
                                  '  **Προτάσεις AI:**', '  Προτάσεις AI:'])
def test_indented_ai_heading_never_opens_an_obligation(setup, title):
    text = 'Υποχρεώσεις:\n- [ ] Ακτινογραφία\n'+title+'\n- [ ] Έλεγχος νέων πτώσεων'
    assert note_tasks(text) == [('Ακτινογραφία', '- [ ] Ακτινογραφία')]
    app, _ = setup; c = login(app); c.post('/clinical/visit-capture/test-patient')
    r = c.post('/clinical/visit-capture/import', json={'context_id': ctx(c), 'text': text, 'encounter_date': '2026-10-28'})
    assert r.status_code == 200
    assert len(r.json()['candidate']['pending']) == 1
    assert r.json()['candidate']['prepared_note']['text'] == text
    with pytest.raises(ValueError):
        note_tasks('Υποχρεώσεις:\n- [ ] Ενέργεια\n  Μόνο μετά: απαραίτητη προϋπόθεση')


def test_browser_direct_paste_one_save_full_chart_prompt_edit_and_mode_race(setup, monkeypatch, tmp_path):
    import socket, threading, shutil
    import uvicorn
    from playwright.sync_api import sync_playwright, expect
    app, engine = setup
    monkeypatch.setenv('CLINICAL_DOCUMENTS_AI_ENABLED', 'false')
    from clinical_pending_one_tap import PendingEventBase
    PendingEventBase.metadata.create_all(engine)
    monkeypatch.setattr('clinical_capture_cp1.extract_candidate', lambda *a: pytest.fail('Unexpected AI call'))
    sock = socket.socket(); sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level='error'))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True); thread.start()
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=shutil.which('google-chrome') or shutil.which('chromium') or '/Applications/Dia.app/Contents/MacOS/Dia')
            for width in (1280, 390):
                context = browser.new_context(viewport={'width': width, 'height': 950}, permissions=['clipboard-read', 'clipboard-write'])
                page = context.new_page(); errors = []; calls = []
                page.on('pageerror', lambda e: errors.append(str(e)))
                page.on('request', lambda r: calls.append(r.url))
                page.goto(f'http://localhost:{port}/static/cockpit/index.html')
                page.locator('#ceGlobalKey').click(); page.locator('#ceGlobalKeyInput').fill(KEY)
                page.locator('#ceGlobalLoginForm button[type=submit]').click()
                expect(page.locator('#ceGlobalAuthStatus')).to_contain_text('Συνδέθηκες'); page.keyboard.press('Escape')
                page.locator('#captureTestPatient').click(); page.locator('#captureOpen').click()
                expect(page.locator('#captureModeDirect')).to_have_attribute('aria-pressed', 'true')
                expect(page.locator('#captureWebAudio')).not_to_be_visible()
                page.locator('#captureCopyChatGPTPrompt').click()
                copied = page.evaluate('navigator.clipboard.readText()')
                for term in ('Ατομικό ιστορικό:', 'Τρέχουσα αγωγή:', 'Απεικονιστικός έλεγχος:', 'Υποχρεώσεις:', 'Προτάσεις AI'):
                    assert term in copied
                # Overlength is retained rather than truncated by the browser.
                page.locator('#captureText').fill('z'*30001)
                expect(page.locator('#captureText')).to_have_value('z'*30001)
                expect(page.locator('#captureSave')).not_to_be_visible()
                page.locator('#captureText').dispatch_event('paste')
                page.locator('#captureText').fill(NOTE)
                expect(page.locator('#captureAudioSummary')).to_contain_text('Κάταγμα ΠΧΚ πριν δύο χρόνια')
                expect(page.locator('#captureSave')).to_be_visible()
                expect(page.locator('#captureStatus')).to_contain_text('Δεν έγινε κλήση AI')
                with Session(engine) as db:
                    # The indented AI heading remains visible but creates no task.
                    assert 'Έλεγχος νέων πτώσεων' not in [p.description for p in db.scalars(select(ClinicalPendingORM))]
                assert not any('/visit-capture/extract' in u or '/audio/draft' in u for u in calls)
                assert page.locator('#captureAudioSummary script').count() == 0
                with Session(engine) as db:
                    before = db.scalar(select(func.count()).select_from(EncounterORM))
                # One final user action; preview is internal.
                page.locator('#captureSave').click()
                expect(page.locator('#captureComposer')).not_to_be_visible()
                expect(page.locator('#patientChart')).to_contain_text('T-score −2,8')
                expect(page.locator('#patientChart')).to_contain_text('μετά από εκτίμηση από οδοντίατρο')
                with Session(engine) as db:
                    assert db.scalar(select(func.count()).select_from(EncounterORM)) == before+1
                    assert len(list(db.scalars(select(ClinicalPendingORM).where(ClinicalPendingORM.source_encounter_id == db.scalar(select(EncounterORM).order_by(EncounterORM.created_at.desc())).id)))) == 1
                page.reload(); expect(page.locator('#captureOpen')).to_be_visible()
                expect(page.locator('#patientChart')).to_contain_text('Και η επανάληψη παραμένει')
                page.locator('#patientChart').get_by_role('button', name='Αντιγραφή πλήρους σημείωσης').first.click()
                assert page.evaluate('navigator.clipboard.readText()') == NOTE
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                page.screenshot(path=str(tmp_path/f'direct-saved-note-{width}.png'), full_page=True)
                # Stored prepared notes use the complete text editor and one amendment Save.
                with Session(engine) as db:
                    eid = db.scalar(select(EncounterORM).order_by(EncounterORM.created_at.desc())).id
                    original_ids = list(db.scalars(select(ClinicalPendingORM.id).where(ClinicalPendingORM.source_encounter_id == eid)))
                page.evaluate('(eid)=>window.CockpitStoredChart.openVisit("cockpit-cp1-test",eid)', eid)
                page.locator('.patient-visit[open] .visit-edit').click()
                expect(page.locator('#captureText')).to_have_value(NOTE)
                changed = NOTE.replace('Βάρος 61 kg.', 'Βάρος 62 kg.')
                page.locator('#captureText').fill(changed)
                expect(page.locator('#captureStatus')).to_contain_text('Δεν έγινε κλήση AI')
                page.locator('#captureSave').click()
                expect(page.locator('#captureComposer')).not_to_be_visible()
                with Session(engine) as db:
                    assert db.get(EncounterORM, eid).payload_json['_visit_capture_v1']['candidate']['prepared_note']['text'] == changed
                    assert list(db.scalars(select(ClinicalPendingORM.id).where(ClinicalPendingORM.source_encounter_id == eid))) == original_ids
                # Mode switch invalidates a pending automatic import, with no provider call.
                page.locator('#patientVisitBack').click()
                held = []
                page.route('**/clinical/visit-capture/import', lambda route: held.append(route))
                page.locator('#captureOpen').click(); page.locator('#captureText').fill('Λόγος επίσκεψης:\nΝέο μη αποθηκευμένο κείμενο.')
                page.wait_for_timeout(350)
                assert len(held) == 1
                page.locator('#captureModeAI').click()
                body = held[0].request.post_data_json
                import json
                held[0].fulfill(status=200, content_type='application/json', body=json.dumps({
                    'patient_id': TEST_ID, 'context_id': body['context_id'],
                    'candidate': prepared_candidate(body['text'], body['encounter_date']).model_dump(mode='json')}))
                expect(page.locator('#captureSave')).not_to_be_visible()
                expect(page.locator('#captureWebAudio')).to_be_visible()
                page.wait_for_timeout(350)
                expect(page.locator('#captureSave')).not_to_be_visible()
                assert not errors
                page.screenshot(path=str(tmp_path/f'direct-note-{width}.png'), full_page=True)
                context.close()
            browser.close()
    finally:
        server.should_exit = True; thread.join(5)
