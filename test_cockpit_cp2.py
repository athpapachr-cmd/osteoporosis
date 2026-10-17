"""Continuation uses one event owner across chart, updates and independent sessions."""
from uuid import uuid4
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient
from clinical_data import ClinicalPendingORM, EncounterORM, PatientORM
from clinical_pending_one_tap import PendingEventBase, PendingCheckpointEventORM
from test_cockpit_cp1 import draft, ctx
from test_cockpit_synthetic_161 import setup, login, KEY
from clinical_capture_cp1 import TEST_ID


def seed(app,engine):
    PendingEventBase.metadata.create_all(engine) # isolated fixture only
    c=login(app);c.post('/clinical/visit-capture/test-patient')
    candidate=draft().model_dump(mode='json');candidate['pending']=candidate['pending'][:1]
    r=c.post('/clinical/visit-capture/save',json={'context_id':ctx(c),'candidate':candidate});assert r.status_code==200
    p=c.get('/clinical/patient/'+TEST_ID+'/chart').json()['pending'][0]
    return c,p,'/clinical/patient/'+TEST_ID+'/pending/'+p['pending_id']


def test_persistent_independent_stages_reverse_replay_conflict_and_scope(setup):
    app,engine=setup;c,p,path=seed(app,engine)
    body={'template':'material_images','first_observation':'not_yet_observed','idempotency_key':uuid4().hex}
    r=c.post(path+'/checkpoint-plan',json=body);assert r.status_code==200,r.text
    assert r.json()['current']['next_checkpoint']['state']=='not_yet'
    assert c.post(path+'/checkpoint-plan',json=body).json()['replayed'] is True
    assert c.post(path+'/checkpoint-plan',json={**body,'first_observation':'yes_confirmed'}).status_code==409
    def act(key,revision,observation='yes_confirmed',retract=None):
        return {'checkpoint_key':key,'expected_revision':revision,'observation':observation,'retracts_event_id':retract,'idempotency_key':uuid4().hex}
    # Prior real action may be confirmed without fabricating a receipt.
    reviewed=act('images_reviewed',1);r=c.post(path+'/checkpoint',json=reviewed);assert r.status_code==200
    state=r.json()['current'];assert state['confirmed_keys']==['images_reviewed'] and not state['completed']
    assert c.post(path+'/checkpoint',json=act('received',1)).status_code==409
    received=act('received',2);r=c.post(path+'/checkpoint',json=received);assert r.status_code==200
    receipt_event=r.json()['event_id'];assert r.json()['current']['completed'] and r.json()['current']['status']=='resolved'
    assert c.post(path+'/checkpoint',json=received).json()['replayed']
    second=login(app);chart=second.get('/clinical/patient/'+TEST_ID+'/chart').json()
    updates=second.get('/clinical/clinical-updates?scope=test').json()
    assert chart['pending'][0]['checkpoints']==updates['items'][0]['checkpoints']
    assert len(chart['pending'][0]['checkpoints']['history'])==3
    # Reverse exact receipt: image review remains true; receipt returns to not-yet.
    r=second.post(path+'/checkpoint',json=act('received',3,'retracted',receipt_event));assert r.status_code==200
    assert r.json()['current']['confirmed_keys']==['images_reviewed'] and r.json()['current']['status']=='open'
    assert second.post(path+'/checkpoint',json=act('received',4,'retracted',receipt_event)).status_code==409
    assert TestClient(app).post(path+'/checkpoint',json=act('received',4)).status_code==401
    assert c.post(path.replace(TEST_ID,'outside-synthetic-scope')+'/checkpoint',json=act('received',4)).status_code==404
    # Shared key cannot attest real patients or a marker collision.
    with Session(engine) as db:
        patient=db.get(PatientORM,TEST_ID);patient.demographics_json={'full_name':'No synthetic marker'};db.commit()
    assert c.post(path+'/checkpoint',json=act('received',4)).status_code==503
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(EncounterORM))==1
        assert db.scalar(select(func.count()).select_from(PendingCheckpointEventORM))==5
        assert db.scalar(select(EncounterORM)).payload_json['_visit_capture_v1']['candidate']['pending'][0]['status']=='awaiting_patient'


def test_no_schema_read_fallback_and_resolved_plan_refusal(setup):
    app,engine=setup;c=login(app);c.post('/clinical/visit-capture/test-patient')
    c.post('/clinical/visit-capture/save',json={'context_id':ctx(c),'candidate':draft().model_dump(mode='json')})
    p=c.get('/clinical/patient/'+TEST_ID+'/chart').json()['pending'][0]
    assert p['checkpoint_storage_ready'] is False
    path='/clinical/patient/'+TEST_ID+'/pending/'+p['pending_id']+'/checkpoint-plan'
    action={'template':'material_images','idempotency_key':uuid4().hex}
    assert c.post(path,json=action).status_code==503
    PendingEventBase.metadata.create_all(engine)
    with Session(engine) as db:
        db.get(ClinicalPendingORM,p['pending_id']).status='resolved';db.commit()
    assert c.post(path,json=action).status_code==409


def test_browser_receipt_review_timeline_refresh_second_session(setup,tmp_path):
    import socket,threading,shutil
    import uvicorn
    from playwright.sync_api import sync_playwright,expect
    app,engine=setup;seed(app,engine)
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,log_level='error'));thread=threading.Thread(target=server.run,kwargs={'sockets':[sock]},daemon=True);thread.start()
    url=f'http://localhost:{port}/static/cockpit/index.html?chart={TEST_ID}'
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(executable_path=shutil.which('google-chrome') or shutil.which('chromium'))
            for width in [1280,390]:
                context=browser.new_context(viewport={'width':width,'height':950});page=context.new_page();errors=[]
                page.on('pageerror',lambda e:errors.append(str(e)))
                page.goto(url.split('?')[0]);page.locator('#ceGlobalKey').click();page.locator('#ceGlobalKeyInput').fill(KEY)
                page.locator('#ceGlobalLoginForm button[type=submit]').click();expect(page.locator('#ceGlobalAuthStatus')).to_contain_text('Συνδέθηκες');page.keyboard.press('Escape');page.goto(url)
                row=page.locator('#patientChart .patient-chart-grid .synthetic-pending-row').first
                if width==1280:
                    expect(row).to_contain_text('Παραλήφθηκε το CD;')
                    row.locator('summary').filter(has_text='Έγινε ήδη επόμενο βήμα;').click()
                    row.locator('details').get_by_role('button',name='Ναι',exact=True).click()
                    expect(row).to_contain_text('✓ Εξετάστηκαν οι εικόνες')
                    expect(row).to_contain_text('Παραλήφθηκε το CD;')
                    state=page.request.get(f'http://localhost:{port}/clinical/patient/{TEST_ID}/chart').json()['pending'][0]['checkpoints']
                    assert state['confirmed_keys']==['images_reviewed']
                    assert not any(e['checkpoint_key']=='received' for e in state['history'])
                    row.get_by_role('button',name='Αναίρεση',exact=True).click()
                    expect(row).not_to_contain_text('✓ Εξετάστηκαν οι εικόνες')
                    row.get_by_role('button',name='Ναι',exact=True).first.click()
                    expect(row).to_contain_text('Εξετάστηκαν οι εικόνες;')
                    expect(row).not_to_contain_text('Ολοκληρώθηκε')
                    page.reload();expect(row).to_contain_text('Εξετάστηκαν οι εικόνες;')
                    row.get_by_role('button',name='Όχι ακόμη',exact=True).click()
                    expect(row).to_contain_text('Όχι ακόμη · καταγράφηκε')
                else:
                    expect(row).to_contain_text('Όχι ακόμη · καταγράφηκε')
                    row.get_by_role('button',name='Ναι',exact=True).click()
                    expect(row).to_contain_text('Ολοκληρώθηκε')
                points=page.locator('.patient-journey-point');assert points.count()>=2
                points.filter(has_text='Επανέλεγχος γόνατος').click();expect(page.locator('.patient-journey-preview')).to_contain_text('Επανέλεγχος γόνατος')
                page.get_by_role('button',name='Άνοιγμα επίσκεψης',exact=True).click()
                expect(page.locator('.patient-visit')).to_be_visible()
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), page.evaluate("Array.from(document.querySelectorAll('body *')).filter(e=>e.getBoundingClientRect().right>innerWidth && e.getBoundingClientRect().width>0).slice(0,15).map(e=>[e.tagName,e.id,e.className,e.getBoundingClientRect().width])")
                page.locator('#patientChart').screenshot(path=str(tmp_path/f'cp2-{width}.png'))
                assert not errors;context.close()
            browser.close()
    finally:server.should_exit=True;thread.join(10);sock.close()


def test_real_patient_requires_existing_clinician_session(setup,monkeypatch):
    app,engine=setup;c,p,path=seed(app,engine)
    with Session(engine) as db:
        patient=db.get(PatientORM,TEST_ID);patient.demographics_json={'full_name':'Synthetic real-role boundary fixture'};db.commit()
    monkeypatch.setenv('COCKPIT_STAFF_AUTH_ENABLED','1')
    monkeypatch.setenv('COCKPIT_STAFF_CLINICIAN_KEY','C'*40)
    monkeypatch.setenv('COCKPIT_STAFF_SECRETARY_KEY','S'*40)
    monkeypatch.setenv('COCKPIT_STAFF_SESSION_SECRET','T'*40)
    action={'template':'material_images','first_observation':'yes_confirmed','idempotency_key':uuid4().hex}
    assert c.post(path+'/checkpoint-plan',json=action).status_code==401
    clinician=TestClient(app,base_url='https://testserver')
    assert clinician.post('/clinical/login',json={'key':'C'*40}).status_code==200
    assert clinician.post(path+'/checkpoint-plan',json=action).status_code==200
    with Session(engine) as db:
        event=db.scalar(select(PendingCheckpointEventORM).where(PendingCheckpointEventORM.kind=='yes_confirmed'))
        assert event.actor_role=='clinician' and event.actor_principal_id=='clinic-clinician'
