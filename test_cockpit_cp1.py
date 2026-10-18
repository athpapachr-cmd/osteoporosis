"""CP1 contract + real browser/DB workflow; provider mocked only in local tests."""
from copy import deepcopy
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from clinical_data import EncounterORM, ClinicalPendingORM, VisitCaptureCandidateV1
from clinical_capture_cp1 import TEST_ID, extract_candidate
from test_cockpit_synthetic_161 import setup, login, KEY

TEXT='Επανέλεγχος γόνατος. Ήπιος πόνος στην κάμψη. Να προσκομιστεί το CD της μαγνητικής. Να σταλεί το πόρισμα.'

def draft():
    fact=lambda s:{'text':s,'provenance':['dia_summary']}
    return VisitCaptureCandidateV1.model_validate({
        'schema_version':'visit_capture_candidate_v1','source_identity_state':'consistent','encounter_date':'2026-10-18',
        'reason_for_visit':fact('Επανέλεγχος γόνατος'),
        'source_bindings':[{'source_ref':'dia_summary','role':'dia_summary'}],
        'next_contact':{'purpose':'Κλινικός επανέλεγχος','assess':['Κάμψη γόνατος'],'provenance':['dia_summary']},
        'comparison_basis':{'type':'unavailable'},'findings':[fact('Ήπιος πόνος στην κάμψη')],
        'pending':[{'item_type':'investigation','description':s,'status':'awaiting_patient','responsible_role':'patient',
                    'provenance':['dia_summary'],'source_excerpt':s} for s in ['Να προσκομιστεί το CD της μαγνητικής.','Να σταλεί το πόρισμα.']]})

def ctx(c,patient=TEST_ID):
    return c.post('/clinical/visit-capture/context',json={'patient_id':patient,'client_session_id':'cp1-browser-test-session'}).json()['context_id']

def test_proposal_approval_atomic_persistence_and_exact_origin(setup,monkeypatch):
    app,engine=setup;c=login(app)
    assert c.post('/clinical/visit-capture/test-patient').status_code==200
    assert TEST_ID not in [p['patient_id'] for p in c.get('/clinical/patients').json()]
    monkeypatch.setattr('clinical_capture_cp1.extract_candidate',lambda *a:draft())
    context=ctx(c)
    result=c.post('/clinical/visit-capture/extract',json={'context_id':context,'text':TEXT,'encounter_date':'2026-10-18'})
    assert result.status_code==200
    with Session(engine) as db: assert db.scalar(select(func.count()).select_from(ClinicalPendingORM))==0
    candidate=result.json()['candidate'];candidate['pending'].pop();candidate['findings'][0]['text']='Διορθωμένο εύρημα'
    body={'context_id':context,'candidate':candidate}
    saved=c.post('/clinical/visit-capture/save',json=body);assert saved.status_code==200,saved.text
    assert c.post('/clinical/visit-capture/save',json=body).status_code==409
    second=login(app)
    chart=second.get('/clinical/patient/'+TEST_ID+'/chart').json()
    assert chart['visits'][0]['clinical']['findings'][0]['text']=='Διορθωμένο εύρημα'
    assert len(chart['pending'])==1 and chart['pending'][0]['source_excerpt']=='Να προσκομιστεί το CD της μαγνητικής.'
    assert chart['pending'][0]['source_encounter_id']==saved.json()['encounter']['encounter_id']
    assert second.get('/clinical/clinical-updates').json()['items']==[]
    assert second.get('/clinical/clinical-updates?scope=test').json()['items'][0]['pending_id']==chart['pending'][0]['pending_id']
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(EncounterORM))==1
        payload=db.scalar(select(EncounterORM)).payload_json
        assert TEXT not in str(payload)


def test_stale_after_provider_and_protected_routes(setup,monkeypatch):
    app,_=setup;c=login(app);c.post('/clinical/visit-capture/test-patient');context=ctx(c)
    def mutate(*args):
        c.post('/clinical/patients',json={'patient_id':TEST_ID,'demographics':{'full_name':'Changed'}})
        return draft()
    monkeypatch.setattr('clinical_capture_cp1.extract_candidate',mutate)
    assert c.post('/clinical/visit-capture/extract',json={'context_id':context,'text':TEXT,'encounter_date':'2026-10-18'}).status_code==409
    invalid=c.post('/clinical/visit-capture/extract',json={'context_id':context,'text':TEXT,'encounter_date':'bad'})
    assert invalid.status_code==422 and TEXT not in invalid.text
    oversized=c.post('/clinical/visit-capture/extract',json={'context_id':context,'text':TEXT*1000,'encounter_date':'2026-10-18'})
    assert oversized.status_code in (413,422) and TEXT not in oversized.text
    anonymous=TestClient(app)
    assert anonymous.get('/clinical/patient/'+TEST_ID+'/chart').status_code==401
    assert anonymous.get('/clinical/clinical-updates').status_code==401
    assert anonymous.post('/clinical/visit-capture/test-patient').status_code==401
    assert anonymous.post('/clinical/visit-capture/extract',json={'context_id':context,'text':TEXT,'encounter_date':'2026-10-18'}).status_code==401


def test_provider_boundary_rejects_fabricated_excerpt_completion_and_failure(monkeypatch):
    from types import SimpleNamespace
    from fastapi import HTTPException
    import openai
    monkeypatch.setenv('CLINICAL_DOCUMENTS_AI_ENABLED','true');monkeypatch.setenv('OPENAI_API_KEY','fake-local')
    monkeypatch.setenv('CLINICAL_DOCUMENTS_PHI_PROVIDER_APPROVED','true')
    current=draft();calls=[]
    def parse(**kw):calls.append(kw);return SimpleNamespace(status='completed',output_parsed=current)
    monkeypatch.setattr(openai,'OpenAI',lambda **kw:SimpleNamespace(responses=SimpleNamespace(parse=parse)))
    assert extract_candidate(TEXT,'2026-10-18',{}).pending
    assert calls[0]['store'] is False and 'tools' not in calls[0]
    current.pending[0].source_excerpt='invented'
    with pytest.raises(HTTPException) as error:extract_candidate(TEXT,'2026-10-18',{})
    assert error.value.status_code==502 and 'invented' not in str(error.value.detail)
    current=draft();current.pending[0].status='resolved'
    with pytest.raises(HTTPException):extract_candidate(TEXT,'2026-10-18',{})
    class Exhausted(Exception):
        code='credit_balance_exhausted'
    def exhausted(**kw):raise Exhausted('DO_NOT_ECHO_SOURCE_OR_PROVIDER_ERROR')
    monkeypatch.setattr(openai,'OpenAI',lambda **kw:SimpleNamespace(responses=SimpleNamespace(parse=exhausted)))
    with pytest.raises(HTTPException) as error:extract_candidate(TEXT,'2026-10-18',{})
    assert error.value.status_code==503 and error.value.detail['code']=='provider_budget_exhausted'
    assert 'DO_NOT_ECHO' not in str(error.value.detail)
    monkeypatch.setenv('CLINICAL_DOCUMENTS_PHI_PROVIDER_APPROVED','false')
    with pytest.raises(HTTPException) as error:extract_candidate(TEXT,'2026-10-18',{})
    assert error.value.status_code==503


def test_browser_capture_edit_reject_save_refresh_and_second_session(setup,monkeypatch,tmp_path):
    import socket,threading,shutil
    import uvicorn
    from playwright.sync_api import sync_playwright,expect
    app,engine=setup
    started=threading.Event();release=threading.Event();first=[True]
    def provider(*args):
        if first[0]:
            first[0]=False;started.set();release.wait(15)
        return draft()
    monkeypatch.setattr('clinical_capture_cp1.extract_candidate',provider)
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,log_level='error'))
    thread=threading.Thread(target=server.run,kwargs={'sockets':[sock]},daemon=True);thread.start()
    url=f'http://localhost:{port}/static/cockpit/index.html'
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(executable_path=shutil.which('google-chrome') or shutil.which('chromium'))
            for width in [1280,390]:
                context=browser.new_context(viewport={'width':width,'height':950},permissions=['clipboard-read','clipboard-write']);page=context.new_page();errors=[]
                page.on('pageerror',lambda e:errors.append(str(e)));page.goto(url)
                page.locator('#ceGlobalKey').click();page.locator('#ceGlobalKeyInput').fill(KEY)
                page.locator('#ceGlobalLoginForm button[type=submit]').click()
                expect(page.locator('#ceGlobalAuthStatus')).to_contain_text('Συνδέθηκες')
                page.keyboard.press('Escape')
                page.locator('#captureTestPatient').click();expect(page.locator('#captureOpen')).to_be_visible()
                page.locator('#captureOpen').click()
                page.locator('#captureCopyDiaPrompt').click()
                expect(page.locator('#captureDiaPromptStatus')).to_contain_text('Αντιγράφηκε')
                copied=page.evaluate('navigator.clipboard.readText()')
                assert copied==page.locator('#captureDiaPromptText').text_content()
                assert 'ενεργής επίσκεψης' in copied and 'Δοκιμαστικός ασθενής' not in copied and '1972-01-01' not in copied
                page.locator('#captureText').fill('Το πρόχειρο παραμένει μετά την αντιγραφή')
                page.evaluate("()=>{navigator.clipboard.writeText=async()=>{throw new Error('Clipboard denied')};}")
                page.locator('#captureCopyDiaPrompt').click()
                expect(page.locator('#captureDiaPromptText')).to_be_visible()
                expect(page.locator('#captureDiaPromptStatus')).to_contain_text('Επίλεξε και αντέγραψε')
                expect(page.locator('#captureText')).to_have_value('Το πρόχειρο παραμένει μετά την αντιγραφή')
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                page.locator('.capture-dia-prompt').screenshot(path=str(tmp_path/f'dia-prompt-{width}.png'))
                page.locator('#captureComposer').get_by_role('button',name='Κλείσιμο',exact=True).click()
                if width==1280:
                    page.locator('#captureOpen').click();page.locator('#captureText').fill(TEXT)
                    page.get_by_role('button',name='Προετοιμασία καταγραφής',exact=True).click()
                    assert started.wait(5)
                    page.locator('#captureComposer').get_by_role('button',name='Κλείσιμο',exact=True).click()
                    page.locator('#captureOpen').click()
                    with page.expect_response('**/visit-capture/extract'):
                        release.set()
                    expect(page.locator('#captureSave')).to_be_hidden()
                    expect(page.locator('#captureReview')).to_be_empty()
                    page.locator('#captureOpen').click();page.locator('#captureText').fill(TEXT)
                    page.get_by_role('button',name='Προετοιμασία καταγραφής',exact=True).click()
                    expect(page.locator('#captureSave')).to_be_visible(timeout=15000)
                    finding=page.locator('#captureReview section').filter(has=page.get_by_role('heading',name='Ιστορικό και ευρήματα',exact=True))
                    finding.locator('textarea').fill('Διορθωμένο εύρημα από τον γιατρό')
                    pending=page.locator('#captureReview section').filter(has=page.get_by_role('heading',name='Προτεινόμενες υποχρεώσεις',exact=True))
                    pending.get_by_role('button',name='Αφαίρεση').last.click()
                    assess=page.locator('#captureReview section').filter(has=page.get_by_role('heading',name='Προς αξιολόγηση',exact=True))
                    assess.locator('input').fill('Αφαιρεθέν στοιχείο')
                    assess.get_by_role('button',name='Αφαίρεση').click()
                    page.locator('#captureSave').click();expect(page.locator('#captureComposer')).to_be_hidden()
                expect(page.locator('#patientChart')).to_contain_text('Δημιουργήθηκε υποχρέωση: Να προσκομιστεί το CD')
                page.reload();expect(page.locator('#patientChart')).to_contain_text('Να προσκομιστεί το CD')
                page.locator('.patient-journey-point').first.click()
                page.get_by_role('button',name='Άνοιγμα',exact=True).click()
                expect(page.locator('#patientChart')).to_contain_text('Διορθωμένο εύρημα από τον γιατρό')
                assert 'Να σταλεί το πόρισμα' not in page.locator('#patientChart').inner_text()
                assert 'Αφαιρεθέν στοιχείο' not in page.locator('#patientChart').inner_text()
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                page.screenshot(path=str(tmp_path/f'cp1-{width}.png'),full_page=True)
                assert not errors,errors
                context.close()
            browser.close()
    finally:server.should_exit=True;thread.join(timeout=10);sock.close()
    with Session(engine) as db:assert db.scalar(select(func.count()).select_from(EncounterORM))==1
