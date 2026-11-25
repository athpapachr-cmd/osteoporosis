"""Audited correction, exact task continuity and actual composer; no external provider."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
from types import SimpleNamespace
from uuid import uuid4
import pytest
from sqlalchemy import select,func
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient
from clinical_data import EncounterORM,ClinicalPendingORM,PatientORM
from clinical_pending_one_tap import PendingEventBase,PendingCheckpointEventORM
from clinical_capture_cp1 import TEST_ID
from test_cockpit_cp1 import draft,ctx
from test_cockpit_synthetic_161 import setup,login,KEY

@pytest.fixture
def seeded(setup):
    app,engine=setup;PendingEventBase.metadata.create_all(engine);c=login(app)
    c.post('/clinical/visit-capture/test-patient')
    saved=c.post('/clinical/visit-capture/save',json={'context_id':ctx(c),'candidate':draft().model_dump(mode='json')}).json()
    eid=saved['encounter']['encounter_id'];path=f'/clinical/patient/{TEST_ID}/visit/{eid}'
    return app,engine,c,eid,path

def edit(c,path):
    r=c.get(path+'/edit');assert r.status_code==200,r.text
    d=r.json();return {'context_id':ctx(c),'expected_token':d['expected_token'],'request_id':str(uuid4()),'candidate':d['candidate'],'pending_bindings':d['pending_bindings']}

def chart(c):return c.get('/clinical/patient/'+TEST_ID+'/chart').json()

def test_full_amendment_same_id_flat_history_task_preservation_and_retry(seeded):
    app,engine,c,eid,path=seeded
    before=chart(c);task=before['pending'][0]
    plan='/clinical/patient/'+TEST_ID+'/pending/'+task['pending_id']+'/checkpoint-plan'
    assert c.post(plan,json={'template':'completion','first_observation':'yes_confirmed','idempotency_key':uuid4().hex}).status_code==200
    task=next(p for p in chart(c)['pending'] if p['pending_id']==task['pending_id']);body=edit(c,path)
    body['candidate']['reason_for_visit']['text']='Διορθωμένος επανέλεγχος'
    body['candidate']['encounter_date']='2026-10-09'
    body['candidate']['findings'][0]['text']='Παρακέντηση γόνατος'
    body['candidate']['decisions']=[{'text':'Ανάπαυση','type':'advice','provenance':['dia_summary']}]
    body['candidate']['next_contact']['purpose']='Εκτίμηση ανταπόκρισης'
    with Session(engine) as db:original=deepcopy(db.get(EncounterORM,eid).payload_json)
    r=c.post(path+'/amend',json=body);assert r.status_code==200,r.text
    assert c.post(path+'/amend',json=body).json()['replayed'] is True
    conflict=deepcopy(body);conflict['candidate']['reason_for_visit']['text']='Different';assert c.post(path+'/amend',json=conflict).status_code==409
    after=chart(login(app));assert len(after['visits'])==1 and after['visits'][0]['encounter_id']==eid
    assert after['visits'][0]['date']=='2026-10-09' and after['visits'][0]['revision']==1
    assert next(p for p in after['pending'] if p['pending_id']==task['pending_id'])=={**task,'source_date':'2026-10-09'}
    with Session(engine) as db:
        row=db.get(EncounterORM,eid);signed=row.payload_json['_visit_capture_v1'];h=signed['amendment_history'][0]
        assert row.status=='amended' and h['previous']['signed']==original['_visit_capture_v1']
        assert h['actor']==['shared-clinical-key','shared_clinical_key']
    second=edit(c,path);second['candidate']['uncertainties']=[];assert c.post(path+'/amend',json=second).status_code==200
    with Session(engine) as db:
        h=db.get(EncounterORM,eid).payload_json['_visit_capture_v1']['amendment_history'];assert len(h)==2
        assert all('amendment_history' not in v['previous']['signed'] for v in h)
    assert c.post(path+'/amend',json=body).json()['replayed'] is True
    assert c.put('/clinical/encounter/'+eid,json={'encounter_date':'2026-10-10'}).status_code==409


def test_removed_replaced_new_tasks_exact_sources_and_terminal_nonrevival(seeded):
    _,engine,c,eid,path=seeded;before=chart(c);body=edit(c,path)
    old_ids=list(body['pending_bindings']);original={p['pending_id']:p['source_excerpt'] for p in before['pending']}
    body['candidate']['pending']=body['candidate']['pending'][:1];body['pending_bindings']=old_ids[:1]
    body['candidate']['pending'][0]['description']='Νέα πραγματική υποχρέωση'
    assert c.post(path+'/amend',json=body).status_code==200
    after=chart(c);old=[p for p in after['pending'] if p['pending_id'] in old_ids];new=[p for p in after['pending'] if p['pending_id'] not in old_ids]
    assert len(old)==2 and all(p['status']=='cancelled' and p['source_excerpt']==original[p['pending_id']] for p in old)
    assert len(new)==1 and new[0]['status']=='awaiting_patient' and new[0]['checkpoints'] is None
    deleted=old[0];lp=f"/clinical/patient/{TEST_ID}/pending/{deleted['pending_id']}/lifecycle"
    assert c.post(lp,json={'action':'delete_permanently','expected_revision':deleted['lifecycle']['revision'],'cancellation_event_id':deleted['lifecycle']['cancellation_event_id'],'idempotency_key':uuid4().hex}).status_code==200
    next_body=edit(c,path);assert next_body['pending_bindings']==[new[0]['pending_id']]
    bad=deepcopy(next_body);bad['pending_bindings'][0]=deleted['pending_id'];assert c.post(path+'/amend',json=bad).status_code==409
    assert c.post(path+'/amend',json=next_body).status_code==200
    assert all(p['pending_id']!=deleted['pending_id'] for p in chart(c)['pending'])
    with Session(engine) as db:assert db.get(ClinicalPendingORM,deleted['pending_id']).status=='deleted'


def test_stale_task_context_foreign_binding_and_private_validation_roll_back(seeded):
    _,engine,c,eid,path=seeded;body=edit(c,path);before=chart(c)
    duplicate=deepcopy(body);duplicate['pending_bindings']=[body['pending_bindings'][0]]*2
    assert c.post(path+'/amend',json=duplicate).status_code==409 and chart(c)==before
    foreign=deepcopy(body);foreign['pending_bindings'][0]=str(uuid4());assert c.post(path+'/amend',json=foreign).status_code==409
    wrong=deepcopy(body);wrong['context_id']=ctx(c,'outside-synthetic-scope');assert c.post(path+'/amend',json=wrong).status_code==409
    bad=deepcopy(body);bad['candidate']['reason_for_visit']['text']='';bad['private_sentinel']='PRIVATE_SENTINEL'
    r=c.post(path+'/amend',json=bad);assert r.status_code==422 and 'PRIVATE_SENTINEL' not in r.text and chart(c)==before
    body=edit(c,path);pid=body['pending_bindings'][0]
    assert c.post(f'/clinical/patient/{TEST_ID}/pending/{pid}/checkpoint-plan',json={'template':'completion','first_observation':'not_yet_observed','idempotency_key':uuid4().hex}).status_code==200
    current=chart(c);assert c.post(path+'/amend',json=body).status_code==409 and chart(c)==current
    first=edit(c,path);second=edit(c,path) # shared capture session invalidates first
    assert c.post(path+'/amend',json=first).status_code==409
    assert c.post(path+'/amend',json=second).status_code==200


def test_failed_reconciliation_rolls_back_visit_tasks_events_and_context(seeded,monkeypatch):
    from fastapi import HTTPException
    import clinical_visit_amendments as amendments
    _,engine,c,eid,path=seeded;body=edit(c,path);before=chart(c)
    body['candidate']['pending']=[];body['pending_bindings']=[]
    original=amendments.cancel_pending
    def fail_after_cancel(*args):
        original(*args)
        raise HTTPException(409,'Simulated competing state')
    monkeypatch.setattr(amendments,'cancel_pending',fail_after_cancel)
    assert c.post(path+'/amend',json=body).status_code==409 and chart(c)==before
    with Session(engine) as db:assert db.scalar(select(func.count()).select_from(PendingCheckpointEventORM))==0
    monkeypatch.setattr(amendments,'cancel_pending',original)
    assert c.post(path+'/amend',json=body).status_code==200
    assert all(p['status']=='cancelled' for p in chart(c)['pending'])


def test_staff_cookie_precedence_and_no_anonymous_edit(seeded,monkeypatch):
    app,_,c,eid,path=seeded
    from clinical_staff_auth import STAFF_COOKIE
    monkeypatch.setenv('COCKPIT_STAFF_AUTH_ENABLED','1');monkeypatch.setenv('COCKPIT_STAFF_CLINICIAN_KEY','C'*40)
    monkeypatch.setenv('COCKPIT_STAFF_SECRETARY_KEY','S'*40);monkeypatch.setenv('COCKPIT_STAFF_SESSION_SECRET','T'*40)
    body=edit(c,path)
    for cookie in ['', 'invalid']:
        denied=login(app);denied.cookies.set(STAFF_COOKIE,cookie,domain='testserver.local',path='/clinical')
        assert denied.get(path+'/edit',headers={'X-Clinical-Key':KEY}).status_code in (401,403)
        assert denied.post(path+'/amend',json=body,headers={'X-Clinical-Key':KEY}).status_code in (401,403)
    secretary=TestClient(app,base_url='https://testserver');secretary.post('/clinical/login',json={'key':'S'*40})
    assert secretary.get(path+'/edit').status_code in (401,403) and secretary.post(path+'/amend',json=body).status_code in (401,403)
    assert TestClient(app).get(path+'/edit').status_code==401
    clinician=TestClient(app,base_url='https://testserver');clinician.post('/clinical/login',json={'key':'C'*40})
    body=edit(clinician,path);assert clinician.post(path+'/amend',json=body).status_code==200
    monkeypatch.setenv('COCKPIT_STAFF_CLINICIAN_KEY','R'*40)
    assert clinician.post(path+'/amend',json=body,headers={'X-Clinical-Key':KEY}).status_code==401


def test_temporal_brief_keeps_prior_date_and_signed_content(seeded,monkeypatch):
    _,engine,c,eid,path=seeded
    with Session(engine) as db:
        row=db.get(EncounterORM,eid);payload=deepcopy(row.payload_json);payload['_visit_capture_v1']['signed_at']='2026-10-01T10:00:00'
        payload['_visit_capture_v1']['candidate']['encounter_date']='2026-10-01';row.encounter_date='2026-10-01';row.payload_json=payload
        row.created_at=row.updated_at=datetime(2026,10,1,10);db.commit()
    from clinical_visit_brief import _minimal_brief
    booking=SimpleNamespace(start_at=datetime(2026,10,5,12,tzinfo=timezone.utc))
    with Session(engine) as db:before=_minimal_brief(db,TEST_ID,booking)['previous_visit']
    body=edit(c,path);body['candidate']['reason_for_visit']['text']='Μεταγενέστερη διόρθωση';body['candidate']['encounter_date']='2026-10-09'
    assert c.post(path+'/amend',json=body).status_code==200
    with Session(engine) as db:assert _minimal_brief(db,TEST_ID,booking)['previous_visit']==before
    assert chart(c)['visits'][0]['clinical']['reason_for_visit']['text']=='Μεταγενέστερη διόρθωση'


def test_effective_revision_future_clinical_date_never_resurrects_old_version(seeded,monkeypatch):
    _,engine,c,eid,path=seeded
    with Session(engine) as db:
        row=db.get(EncounterORM,eid);payload=deepcopy(row.payload_json);payload['_visit_capture_v1']['signed_at']='2026-10-01T10:00:00'
        payload['_visit_capture_v1']['candidate']['encounter_date']='2026-10-01';row.encounter_date='2026-10-01';row.payload_json=payload
        row.created_at=row.updated_at=datetime(2026,10,1,10);db.commit()
    monkeypatch.setattr('clinical_visit_amendments.utcnow',lambda:datetime(2026,10,3,10))
    body=edit(c,path);body['candidate']['encounter_date']='2026-10-09'
    assert c.post(path+'/amend',json=body).status_code==200
    from clinical_visit_brief import _minimal_brief
    with Session(engine) as db:
        assert _minimal_brief(db,TEST_ID,SimpleNamespace(start_at=datetime(2026,10,5,12,tzinfo=timezone.utc)))['previous_visit'] is None


def test_browser_edit_existing_visit_cancel_reextract_save_and_independent_read(seeded,tmp_path,monkeypatch):
    import socket,threading,shutil
    import uvicorn
    from playwright.sync_api import sync_playwright,expect
    app,engine,c,eid,path=seeded
    mismatched=edit(c,path)
    mismatched['candidate']['next_contact'].update(when='2027-05-01',purpose='Επόμενο ραντεβού στις 17 Μαΐου 2027')
    assert c.post(path+'/amend',json=mismatched).status_code==200
    monkeypatch.setattr('clinical_capture_cp1.extract_candidate',lambda *a:draft().model_copy(update={'pending':[]}))
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,log_level='error'));thread=threading.Thread(target=server.run,kwargs={'sockets':[sock]},daemon=True);thread.start()
    url=f'http://localhost:{port}/static/cockpit/index.html?chart={TEST_ID}'
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(executable_path=shutil.which('google-chrome') or shutil.which('chromium'))
            for width in [1280,390]:
                context=browser.new_context(viewport={'width':width,'height':950});page=context.new_page();page.set_default_timeout(8000);errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
                page.goto(url.split('?')[0]);page.locator('#ceGlobalKey').click();page.locator('#ceGlobalKeyInput').fill(KEY)
                page.locator('#ceGlobalLoginForm button[type=submit]').click();expect(page.locator('#ceGlobalAuthStatus')).to_contain_text('Συνδέθηκες');page.keyboard.press('Escape');page.goto(url)
                page.locator('.patient-journey-point').click();page.get_by_role('button',name='Άνοιγμα',exact=True).click()
                page.get_by_role('button',name='Επεξεργασία επίσκεψης',exact=True).click();composer=page.locator('#captureComposer')
                expect(composer).to_be_visible();expect(page.locator('.patient-visit-reader')).not_to_be_visible()
                summary=page.locator('#captureAudioSummary')
                editor=page.locator('#captureReview')
                def edit_section(label):
                    summary.get_by_role('button',name='Διόρθωση · '+label,exact=True).click()
                    expect(editor).to_be_visible()
                def finish_section():
                    editor.get_by_role('button',name='Επιστροφή στην καταγραφή',exact=True).click()
                    expect(editor).to_be_hidden()
                expect(editor).to_be_hidden()
                edit_section('Σκοπός')
                expect(editor.locator('textarea')).to_have_value('Επανέλεγχος γόνατος' if width==1280 else 'Διόρθωση 1280')
                editor.locator('textarea').fill('Πρόχειρη αλλαγή')
                page.locator('#visitBackHome').click()
                expect(composer).to_be_visible()
                expect(editor.locator('textarea')).to_have_value('Πρόχειρη αλλαγή')
                composer.get_by_role('button',name='Κλείσιμο',exact=True).click()
                expect(page.locator('.patient-visit:visible')).to_have_count(1)
                page.get_by_role('button',name='Επεξεργασία επίσκεψης',exact=True).click()
                if width==1280:
                    held=[];writes=[]
                    page.on('request',lambda r:writes.append(r.url) if r.url.endswith('/amend') else None)
                    def hold_first_preview(route):
                        if not held:held.append(route)
                        else:route.continue_()
                    page.route('**/visit-capture/preview',hold_first_preview)
                    with page.expect_request('**/visit-capture/preview'):
                        page.locator('#captureSave').click()
                    page.wait_for_timeout(50);assert held
                    composer.get_by_role('button',name='Κλείσιμο',exact=True).click()
                    page.get_by_role('button',name='Επεξεργασία επίσκεψης',exact=True).click()
                    held[0].fulfill(json={'snapshot':'old','brief':'old','detail':'old','can_save':True})
                    page.wait_for_timeout(150);assert not writes
                    expect(summary).to_contain_text('Επανέλεγχος γόνατος')
                    page.unroute('**/visit-capture/preview')
                    checks=summary.locator('.capture-draft-checks')
                    use_date=checks.get_by_role('button',name='Χρήση',exact=False)
                    expect(use_date).to_be_visible()
                    use_date.click()
                    expect(use_date).to_have_count(0)
                    assert not writes
                    edit_section('Επόμενη επαφή')
                    expect(editor.get_by_label('Ημερομηνία',exact=True)).to_have_value('2027-05-17')
                    finish_section()
                    edit_section('Υποχρεώσεις')
                    tasks=editor.locator('section[data-field=pending] .capture-field-group')
                    tasks.last.get_by_role('button',name='Αφαίρεση',exact=True).click()
                    editor.get_by_role('button',name='Προσθήκη · Προτεινόμενες υποχρεώσεις',exact=True).click()
                    tasks.last.get_by_label('Περιγραφή',exact=True).fill('Νέα κλινική εκκρεμότητα')
                    finish_section()
                    summary.get_by_text('Περισσότερα πεδία',exact=True).click()
                    summary.get_by_role('button',name='Κλινική εκτίμηση',exact=True).click()
                    editor.get_by_role('button',name='Προσθήκη · Κλινική εκτίμηση',exact=True).click()
                    editor.locator('section[data-field=clinical_impression] textarea').fill('Διορθωμένη εκτίμηση')
                    finish_section()
                else:
                    summary.get_by_role('button',name='Κείμενο εισόδου και ημερομηνία').click()
                    page.locator('#captureText').fill('Νέα διορθωμένη σύνοψη χωρίς υποχρεώσεις.')
                    composer.get_by_role('button',name='Προετοιμασία καταγραφής',exact=True).click()
                    expect(summary).to_contain_text('Καμία νέα υποχρέωση.')
                edit_section('Σκοπός')
                editor.locator('textarea').fill('Διόρθωση '+str(width))
                finish_section()
                effects=composer.locator('.capture-amendment-effects')
                expect(effects.locator('summary')).to_be_in_viewport()
                expect(page.locator('#captureSave')).to_be_disabled()
                effects.locator('summary').click()
                expect(effects).to_contain_text('Στις διαγραμμένες')
                expect(page.locator('#captureSave')).to_be_enabled()
                composer.screenshot(path=str(tmp_path/f'amend-edit-{width}.png'))
                if width==1280:
                    edit_section('Υποχρεώσεις')
                    status=editor.locator('section[data-field=pending] .capture-field-group').last.get_by_label('Κατάσταση')
                    status.evaluate("n=>{n.add(new Option('resolved','resolved'));n.value='resolved';n.dispatchEvent(new Event('input'));}")
                    expect(page.locator('#captureSave')).to_be_disabled()
                    effects.locator('summary').click()
                    page.locator('#captureSave').click()
                    expect(page.locator('#captureStatus')).to_contain_text('κράτησε τις νέες υποχρεώσεις ανοικτές')
                    expect(editor).to_be_visible()
                    status.select_option('open')
                    finish_section()
                    if page.locator('#captureSave').is_disabled():
                        effects.locator('summary').click()
                with page.expect_request(lambda r: r.url.endswith('/amend') and r.method=='POST'):
                    page.locator('#captureSave').click()
                expect(page.locator('.patient-visit:visible')).to_contain_text('Διόρθωση '+str(width))
                if width==1280:
                    assert chart(login(app))['visits'][0]['clinical']['next_contact']['when']=='2027-05-17'
                expect(page.locator('.visit-amendment-history')).to_be_visible()
                page.reload();page.locator('.patient-journey-point').click();page.get_by_role('button',name='Άνοιγμα',exact=True).click()
                expect(page.locator('.patient-visit:visible')).to_contain_text('Διόρθωση '+str(width))
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth');assert not errors,errors
                if width==390:
                    page.get_by_role('button',name='Επεξεργασία επίσκεψης',exact=True).click()
                    summary.get_by_role('button',name='Κείμενο εισόδου και ημερομηνία').click()
                    page.locator('#captureText').fill('Επανεπεξεργασία επίσκεψης χωρίς υποχρεώσεις.')
                    composer.get_by_role('button',name='Προετοιμασία καταγραφής',exact=True).click()
                    expect(effects).to_be_hidden()
                    expect(page.locator('#captureSave')).to_be_enabled()
                    page.locator('#captureSave').click()
                    expect(page.locator('.patient-visit:visible')).to_contain_text('Επανέλεγχος γόνατος')
                    page.get_by_role('button',name='Επεξεργασία επίσκεψης',exact=True).click()
                    summary.get_by_role('button',name='Κείμενο εισόδου και ημερομηνία').click()
                    composer.locator('#captureModeDirect').click()
                    note=('Λόγος επίσκεψης:\nΣυνθετική σημείωση\n'
                          'Επόμενη επαφή:\n01/05/2027\nΕπόμενη επίσκεψη:\n17 Μαΐου 2027\n'
                          'Παρακολούθηση και επόμενη επίσκεψη:\nΕξέταση στις 10/05/2027. Ραντεβού στις 17/05/2027.\n'
                          'Νέα αγωγή:\n- Συνθετικό Α 10 mg\n- Συνθετικό Β\n'
                          'Προτάσεις AI για την επόμενη επίσκεψη:\nΕπανεξέταση του καταγεγραμμένου αποτελέσματος')
                    page.locator('#captureText').fill(note)
                    expect(summary.locator('.capture-draft-checks')).to_contain_text('αναφέρονται διαφορετικές ημερομηνίες')
                    expect(summary.locator('.capture-draft-checks')).to_contain_text('συμφωνημένη δόση')
                    expect(summary.locator('.capture-draft-checks')).to_contain_text('Συνθετικό Β')
                    expect(summary.locator('.capture-draft-checks')).not_to_contain_text('Συνθετικό Α')
                    expect(summary.locator('.clinical-compact-overview')).to_contain_text('Επανεξέταση του καταγεγραμμένου αποτελέσματος')
                    with page.expect_request(lambda r:r.url.endswith('/amend') and r.method=='POST'):
                        page.locator('#captureSave').click()
                    expect(page.locator('.patient-visit:visible')).to_contain_text('Συνθετική σημείωση')
                    page.get_by_role('button',name='Επεξεργασία επίσκεψης',exact=True).click()
                    expect(page.locator('#captureText')).to_be_hidden()
                    summary.get_by_role('button',name='Διόρθωση · Επόμενη επαφή',exact=False).click()
                    expect(page.locator('#captureText')).to_be_focused()
                    assert page.locator('#captureText').evaluate('n=>n.value.slice(n.selectionStart).startsWith("Επόμενη επαφή:")')
                    corrected=note.replace('01/05/2027','17/05/2027')
                    page.locator('#captureText').fill(corrected)
                    expect(summary.locator('.capture-draft-checks')).not_to_contain_text('αναφέρονται διαφορετικές ημερομηνίες')
                    expect(page.locator('#captureSave')).to_be_visible()
                    expect(page.locator('#captureSave')).to_be_enabled()
                    with page.expect_request(lambda r:r.url.endswith('/amend') and r.method=='POST'):
                        page.locator('#captureSave').click()
                    expect(page.locator('.patient-visit:visible')).to_contain_text('Συνθετική σημείωση')
                    visit=chart(login(app))['visits'][0]
                    assert visit['encounter_id']==eid and visit['clinical']['prepared_note']['text']==corrected
                    assert '01/05/2027' in visit['amendments'][-1]['detail']
                assert len(chart(login(app))['visits'])==1;context.close()
            browser.close()
    finally:server.should_exit=True;thread.join(10);sock.close()
