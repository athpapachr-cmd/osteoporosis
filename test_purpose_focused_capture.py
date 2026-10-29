"""Purpose/role and compact UI regression. Entirely fictional, no patient source."""
from copy import deepcopy
from uuid import uuid4
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient
import clinical_visit_audio as audio
from clinical_capture_cp1 import TEST_ID
from clinical_data import EncounterORM, ClinicalPendingORM, VisitCaptureCandidateV1, _validate_visit_capture_candidate
from clinical_pending_one_tap import PendingEventBase
from test_cockpit_synthetic_161 import setup, login, KEY
from test_cockpit_cp1 import ctx
from test_clinical_visit_audio import enabled, _client_with_engine, _patient, _context, draft, HEADERS, labels, TRANSCRIPT

# Adapted clinical pattern only: invented fixture, no identifier or real report.
LEGACY = '''Λόγος επίσκεψης:
Παρακολούθηση οστεοπόρωσης και ανασκόπηση συνθετικής MRI για αποκλεισμό σπονδυλικού κατάγματος.
Παρούσα νόσος και συμπτώματα:
Αισθάνεται καλά, χωρίς οσφυϊκά συμπτώματα.

Αναφορά σε ενόχληση ώμου χωρίς λεπτομέρειες.
Απεικονιστικός έλεγχος:
Δεν αναγνωρίστηκε σπονδυλικό κάταγμα. Ήπιες εκφυλιστικές αλλοιώσεις.

ΔΙΣΚΙΚΟ_ΥΛΙΚΟ_ΛΕΠΤΟΜΕΡΕΙΑ: πιθανή μικρή μετακίνηση υλικού χωρίς σημαντική νευρική συμπίεση.
Τυχαία ευρήματα:
ΝΕΦΡΙΚΗ_ΠΥΕΛΟΣ_ΛΕΠΤΟΜΕΡΕΙΑ: τυχαίο εύρημα που συζητήθηκε με παραπομπή στον προσωπικό ιατρό.
Εκτίμηση και διαγνώσεις:
1. Γνωστή οστεοπόρωση χωρίς αναγνωρισμένο σπονδυλικό κάταγμα στην περιγραφόμενη MRI.
2. Ήπια εκφυλιστική δισκοπάθεια χωρίς σχετική συμπτωματολογία.
3. Αβέβαιο επίπεδο ήπιας δισκικής προβολής.
4. Τυχαίο νεφρικό εύρημα.
Θεραπευτικό πλάνο:
Καταγράφηκαν συντηρητικές οδηγίες και άσκηση. Δεν καταγράφηκε έναρξη νέου φαρμάκου.
Παρακολούθηση και επόμενη επίσκεψη:
Προγραμματισμένη DEXA τον Ιανουάριο και ενδοκρινολογική εκτίμηση σε λίγες εβδομάδες για πιθανή θεραπευτική απόφαση.
Υποχρεώσεις:
- [ ] ΔΙΟΙΚΗΤΙΚΟ_ΠΑΡΑΠΕΜΠΤΙΚΟ από τον γιατρό.
Αβεβαιότητες και αντιφάσεις:
Το ακριβές επίπεδο του δισκικού ευρήματος δεν προσδιορίστηκε. Δεν δόθηκαν αριθμητικές τιμές DEXA.
Προτάσεις AI για επανέλεγχο — δεν αποτελούν αποφάσεις:
1. Να αποσαφηνιστεί ΔΙΣΚΙΚΟ_ΥΛΙΚΟ_ΛΕΠΤΟΜΕΡΕΙΑ.
2. Να διερευνηθεί ΝΕΦΡΙΚΗ_ΠΥΕΛΟΣ_ΛΕΠΤΟΜΕΡΕΙΑ.
'''
READY = '''Λόγος επίσκεψης:
Παρακολούθηση οστεοπόρωσης και έλεγχος για σπονδυλικό κάταγμα.
Κλινική εικόνα:
Κλινικά καλά, χωρίς οσφυϊκά συμπτώματα. Στην ανασκόπηση της συνθετικής MRI δεν αναγνωρίστηκε σπονδυλικό κάταγμα ή σημαντική νευρική συμπίεση. Ήπιες εκφυλιστικές αλλοιώσεις χωρίς σχετική συμπτωματολογία.
Πλάνο και παρακολούθηση:
DEXA τον Ιανουάριο και ενδοκρινολογική επανεκτίμηση για θεραπευτική απόφαση. Δεν καταγράφηκε έναρξη νέας αγωγής.
Απεικονιστικός έλεγχος:
ΔΙΣΚΙΚΟ_ΥΛΙΚΟ_ΛΕΠΤΟΜΕΡΕΙΑ παραμένει αβέβαιο. ΝΕΦΡΙΚΗ_ΠΥΕΛΟΣ_ΛΕΠΤΟΜΕΡΕΙΑ ως τυχαίο εύρημα με σύσταση προς τον προσωπικό ιατρό.
Ενέργειες που ολοκληρώθηκαν:
ΔΙΟΙΚΗΤΙΚΟ_ΠΑΡΑΠΕΜΠΤΙΚΟ ζητήθηκε ήδη από τον γιατρό στην επίσκεψη.
Προτάσεις για την επόμενη επίσκεψη:
1. Έγινε η DEXA και ποιο είναι το συγκριτικό αποτέλεσμα; Συνδέεται με την παρακολούθηση της οστεοπόρωσης.
2. Ποια ήταν η ενδοκρινολογική απόφαση; Μπορεί να μεταβάλει το θεραπευτικό πλάνο.
'''


def test_reasonless_audio_preserves_facts_and_one_save_but_no_suggestions(monkeypatch):
    enabled(monkeypatch); client, engine = _client_with_engine(monkeypatch)
    _patient(client, 'SYN-WEB'); context = _context(client, 'SYN-WEB')
    class Transcriber:
        async def transcribe(self, *_): return 'Η MRI δεν δείχνει κάταγμα. Αναμένεται DEXA τον Ιανουάριο.'
    class Summarizer:
        async def classify(self, batch, identity):
            return {'identity_state':'consistent','facts':[{'source_id':i,'excerpt':t,'kind':'finding' if i==0 else 'follow_up'} for i,t in batch], 'suggested_next_checks':[], 'unclassified_ids':[]}
        async def consider(self, facts): pytest.fail('No purpose must not generate contextual suggestions')
    monkeypatch.setattr(audio,'make_providers',lambda s:(Transcriber(),Summarizer()))
    r=draft(client,context['context_id']); assert r.status_code==200,r.text
    result=r.json(); assert result['processing_status']=='complete'
    c=result['candidate']; assert c['reason_is_unspecified'] is True
    assert c['reason_for_visit']['text']=='Κλινική καταγραφή'
    assert c['findings'][0]['text']=='Η MRI δεν δείχνει κάταγμα.'
    assert c['follow_up'][0]['text']=='Αναμένεται DEXA τον Ιανουάριο.'
    assert c['next_visit_focus']==[] and result['contextual_considerations_status']=='purpose_unspecified'
    body={'context_id':context['context_id'],'candidate':c,'recording_patient_confirmed':True}
    assert client.post('/clinical/visit-capture/save',headers=HEADERS,json=body).status_code==200
    assert client.post('/clinical/visit-capture/save',headers=HEADERS,json=body).status_code==409
    with Session(engine) as db: assert db.scalar(select(EncounterORM)).payload_json['_visit_capture_v1']['candidate']['reason_is_unspecified'] is True
    for kind in ['administrative','completed','uncertainty']:
        with pytest.raises(ValueError,match='clinical_content_missing'):
            audio.build_candidate(['Ενέργεια'],[{'source_id':0,'excerpt':'Ενέργεια','kind':kind}],[],'2026-10-29')


def suggestion(text, sid, **kw):
    return dict(category='management_review',text=text,rationale='Επηρεάζει τη συνέχεια του δηλωμένου σκοπού.',source_ids=[sid],confidence='moderate',purpose_source_ids=[0],role_scope='within_role',impact='continuity',**kw)


def test_purpose_role_filter_and_no_global_disc_blacklist():
    facts=[{'source_id':0,'excerpt':'Παρακολούθηση οστεοπόρωσης','kind':'reason'},
           {'source_id':1,'excerpt':'DEXA τον Ιανουάριο','kind':'follow_up'},
           {'source_id':2,'excerpt':'Ενδοκρινολογική απόφαση','kind':'follow_up'},
           {'source_id':3,'excerpt':'Πιθανό δισκικό υλικό','kind':'finding'},
           {'source_id':4,'excerpt':'Τυχαίο νεφρικό εύρημα, προσωπικός ιατρός','kind':'finding'}]
    dex=suggestion('Αποτέλεσμα DEXA;',1); endo=suggestion('Ενδοκρινολογική απόφαση;',2)
    disc={**suggestion('Δισκικό υλικό;',3),'role_scope':'incidental','impact':'none'}
    renal={**suggestion('Νεφρική πύελος;',4),'role_scope':'handoff'}
    checked=audio.checked_considerations({'contextual_clinical_considerations':[dex,endo,disc,renal]},facts)
    assert [v['text'] for v in checked]==['Αποτέλεσμα DEXA;','Ενδοκρινολογική απόφαση;']
    assert audio.checked_considerations({'contextual_clinical_considerations':[{**dex,'purpose_source_ids':[4]}]},facts)==[]
    other=deepcopy(facts);other[0]['excerpt']='Επανεκτίμηση οσφυϊκής ριζοπάθειας'
    assert audio.checked_considerations({'contextual_clinical_considerations':[suggestion('Συσχέτιση δισκικού ευρήματος;',3)]},other)


def test_planning_storage_is_grounded_and_separate_from_tasks(monkeypatch):
    import asyncio
    enabled(monkeypatch)
    facts=[{'source_id':0,'excerpt':'Παρακολούθηση οστεοπόρωσης','kind':'reason'},
           {'source_id':1,'excerpt':'DEXA τον Ιανουάριο','kind':'follow_up'}]
    candidate,_=audio.build_candidate([f['excerpt'] for f in facts],facts,[],'2026-10-29')
    from clinical_data import VisitPlanningSuggestion
    candidate.next_visit_focus=[VisitPlanningSuggestion(text='Αποτέλεσμα DEXA;',rationale='Συνδέεται με την παρακολούθηση.',purpose=candidate.reason_for_visit.text,source_excerpts=['DEXA τον Ιανουάριο'],provenance=['web_visit_audio'],role_scope='within_role',impact='continuity')]
    _validate_visit_capture_candidate(candidate)
    assert candidate.pending==[] and candidate.decisions==[]
    client,engine=_client_with_engine(monkeypatch);_patient(client,'SYN-WEB');context=_context(client,'SYN-WEB')
    saved=client.post('/clinical/visit-capture/save',headers=HEADERS,json={'context_id':context['context_id'],'candidate':candidate.model_dump(mode='json'),'recording_patient_confirmed':True})
    assert saved.status_code==200,saved.text
    reopened=client.get('/clinical/patient/SYN-WEB/chart',headers=HEADERS).json()['visits'][0]['clinical']
    assert reopened['next_visit_focus'][0]['origin']=='ai_suggested'
    assert reopened['next_visit_focus'][0]['status']=='suggestion'
    assert reopened['next_visit_focus'][0]['text']=='Αποτέλεσμα DEXA;'
    assert reopened['pending']==[] and reopened['decisions']==[]
    with Session(engine) as db:assert list(db.scalars(select(ClinicalPendingORM)))==[]
    changed=candidate.model_copy(deep=True);changed.follow_up=[]
    from fastapi import HTTPException
    with pytest.raises(HTTPException):_validate_visit_capture_candidate(changed)
    changed=candidate.model_copy(deep=True);changed.reason_for_visit.text='Άλλος σκοπός'
    with pytest.raises(HTTPException):_validate_visit_capture_candidate(changed)


def test_browser_supplied_pattern_compact_full_optional_and_completed_history(setup,monkeypatch,tmp_path):
    import socket,threading,shutil
    import uvicorn
    from playwright.sync_api import sync_playwright,expect
    app,engine=setup;PendingEventBase.metadata.create_all(engine)
    monkeypatch.setattr('clinical_capture_cp1.extract_candidate',lambda *a:pytest.fail('No direct-import AI'))
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,log_level='error'));thread=threading.Thread(target=server.run,kwargs={'sockets':[sock]},daemon=True);thread.start()
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(executable_path=shutil.which('google-chrome') or shutil.which('chromium') or '/Applications/Dia.app/Contents/MacOS/Dia')
            for width in [1280,390]:
                context=browser.new_context(viewport={'width':width,'height':1000},permissions=['clipboard-read','clipboard-write']);page=context.new_page()
                page.goto(f'http://localhost:{port}/static/cockpit/index.html');page.locator('#ceGlobalKey').click();page.locator('#ceGlobalKeyInput').fill(KEY);page.locator('#ceGlobalLoginForm button[type=submit]').click();expect(page.locator('#ceGlobalAuthStatus')).to_contain_text('Συνδέθηκες');page.keyboard.press('Escape')
                page.locator('#captureTestPatient').click()
                conditional_plan = READY.replace('DEXA τον Ιανουάριο και ενδοκρινολογική επανεκτίμηση για θεραπευτική απόφαση. Δεν καταγράφηκε έναρξη νέας αγωγής.', '- Ενδεχόμενη έναρξη συνθετικού φαρμάκου Χ.\n- Άσκηση.\n- DEXA.\n- Ενδοκρινολογική επανεκτίμηση.\n- ΜΟΝΟ_ΜΕΤΑ_ΠΡΟΕΛΕΓΧΟ: η έναρξη εξαρτάται από ολοκλήρωση του απαραίτητου προελέγχου.')
                for text in [LEGACY,READY,conditional_plan]:
                    page.locator('#captureOpen').click();page.locator('#captureText').dispatch_event('paste');page.locator('#captureText').fill(text)
                    expect(page.locator('#captureSave')).to_be_visible()
                    main=page.locator('#captureAudioSummary .clinical-compact-overview')
                    expect(main).to_contain_text('οστεοπόρωση');expect(main).to_contain_text('DEXA');expect(main).to_contain_text('ενδοκρινολογική')
                    if text==conditional_plan: expect(main).to_contain_text('ΜΟΝΟ_ΜΕΤΑ_ΠΡΟΕΛΕΓΧΟ')
                    visible=main.inner_text();assert 'ΔΙΣΚΙΚΟ_ΥΛΙΚΟ_ΛΕΠΤΟΜΕΡΕΙΑ' not in visible and 'ΝΕΦΡΙΚΗ_ΠΥΕΛΟΣ_ΛΕΠΤΟΜΕΡΕΙΑ' not in visible
                    assert 'ΔΙΟΙΚΗΤΙΚΟ_ΠΑΡΑΠΕΜΠΤΙΚΟ' not in visible and len(visible.split())<240
                    full=page.locator('#captureAudioSummary .clinical-full-record');assert full.get_attribute('open') is None
                    full.locator('summary').first.click();expect(full).to_contain_text('ΔΙΣΚΙΚΟ_ΥΛΙΚΟ_ΛΕΠΤΟΜΕΡΕΙΑ')
                    full.get_by_role('button',name='Αντιγραφή πλήρους σημείωσης').click();assert page.evaluate('navigator.clipboard.readText()')==text
                    full.locator('summary').first.click();page.locator('#captureSave').click();expect(page.locator('#captureComposer')).not_to_be_visible()
                    if text==LEGACY:
                        data=page.request.get(f'http://localhost:{port}/clinical/patient/{TEST_ID}/chart').json();task=next(t for t in data['pending'] if t['status']=='open' and 'ΔΙΟΙΚΗΤΙΚΟ' in t['description'])
                        r=page.request.post(f'http://localhost:{port}/clinical/patient/{TEST_ID}/pending/{task["pending_id"]}/checkpoint-plan',data={'template':'completion','first_observation':'yes_confirmed','idempotency_key':uuid4().hex});assert r.status==200
                    page.reload();expect(page.locator('#captureOpen')).to_be_visible()
                    assert 'ΔΙΟΙΚΗΤΙΚΟ_ΠΑΡΑΠΕΜΠΤΙΚΟ' not in page.locator('.patient-chart-overview').inner_text()
                    if text==conditional_plan: expect(page.locator('.patient-chart-overview .clinical-compact-overview')).to_contain_text('ΜΟΝΟ_ΜΕΤΑ_ΠΡΟΕΛΕΓΧΟ')
                    expect(page.locator('.completed-obligations').first).to_contain_text('Ολοκληρωμένες')
                    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                    page.screenshot(path=str(tmp_path/f'purpose-{width}-{"ready" if text==READY else "legacy"}.png'),full_page=True)
                context.close()
            browser.close()
    finally:server.should_exit=True;thread.join(5)


def test_contextual_response_schema_strict_objects_require_every_property():
    # The actual provider rejects strict object schemas with optional properties.
    def check(node):
        if isinstance(node, dict):
            if node.get('type') == 'object':
                assert node.get('additionalProperties') is False
                assert set(node['required']) == set(node['properties'])
            for value in node.values(): check(value)
        elif isinstance(node, list):
            for value in node: check(value)
    check(audio._consideration_schema())


def test_optional_hint_bad_link_omitted_individually_with_facts_intact():
    batch=[(0,'Επανέλεγχος οστεοπόρωσης.'),(1,'Επανεκτίμηση τον Ιανουάριο.'),(2,'Ακατάτακτο κείμενο.')]
    raw={'identity_state':'consistent','facts':[{'source_id':i,'excerpt':text,'kind':'reason' if i==0 else 'follow_up'} for i,text in batch[:2]],'unclassified_ids':[2],
         'suggested_next_checks':[{'intent':'planned_result_review','source_id':0},{'intent':'follow_up_status','source_id':99},{'intent':'history_update','source_id':2},{'intent':'follow_up_status','source_id':1}]}
    identity,facts,hints,unclassified=audio.checked_analysis(raw,batch)
    assert identity=='consistent' and facts==raw['facts'] and unclassified==[2]
    assert [(h['intent'],h['source_id']) for h in hints]==[('follow_up_status',1)]
    for malformed in [{'intent':'follow_up_status','source_id':'1'},{'intent':'new_diagnosis','source_id':1},{'intent':'follow_up_status','source_id':True}]:
        with pytest.raises(ValueError,match='classification_suggestions'):
            audio.checked_analysis({**raw,'suggested_next_checks':[malformed]},batch)
    # Exact clinical source/accounting checks still run before optional hints.
    for changed,code in [({**raw,'facts':[{**raw['facts'][0],'excerpt':'Κατασκευασμένο εύρημα.'},raw['facts'][1]]},'classification_excerpt'),({**raw,'unclassified_ids':[]},'classification_accounting')]:
        with pytest.raises(ValueError,match=code):audio.checked_analysis(changed,batch)


def test_actual_failure_class_incompatible_hint_does_not_block_audio_or_save(monkeypatch):
    enabled(monkeypatch);client,engine=_client_with_engine(monkeypatch)
    _patient(client,'SYN-WEB');context=_context(client,'SYN-WEB')
    class Transcriber:
        async def transcribe(self,*_):return TRANSCRIPT
    class Summarizer:
        async def classify(self,batch,identity):
            result=labels(batch)
            result['suggested_next_checks'].insert(0,{'intent':'planned_result_review','source_id':0})
            return result
        async def classify_retry(self,*_):pytest.fail('An incompatible optional hint must not repeat classification')
        async def consider(self,*_):return {'contextual_clinical_considerations':[]}
    monkeypatch.setattr(audio,'make_providers',lambda s:(Transcriber(),Summarizer()))
    response=draft(client,context['context_id']);assert response.status_code==200
    data=response.json();assert data['processing_status']=='complete' and data['transcript']==TRANSCRIPT
    candidate=data['candidate'];assert candidate['pending']==[]
    assert candidate['reason_for_visit']['text']=='Επανέλεγχος οστεοπόρωσης.'
    assert candidate['findings'][0]['text']=='DEXA T-score -2.8 στο ισχίο.'
    assert candidate['recommendations'][0]['text']=='Ξεκινά alendronate 70 mg εβδομαδιαίως.'
    assert [h['intent'] for h in data['suggested_next_checks']]==['therapy_tolerance']
    saved=client.post('/clinical/visit-capture/save',headers=HEADERS,json={'context_id':context['context_id'],'candidate':candidate,'recording_patient_confirmed':True})
    assert saved.status_code==200
    reopened=client.get('/clinical/patient/SYN-WEB/chart',headers=HEADERS).json()['visits'][0]['clinical']
    assert reopened['findings']==candidate['findings'] and reopened['pending']==[]
