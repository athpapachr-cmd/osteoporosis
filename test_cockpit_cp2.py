"""Continuation uses one event owner across chart, updates and independent sessions."""
from uuid import uuid4
import pytest
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
    # Normal clinical-key access also works without the synthetic marker; audit stays honest.
    with Session(engine) as db:
        patient=db.get(PatientORM,TEST_ID);patient.demographics_json={'full_name':'No synthetic marker'};db.commit()
    assert c.post(path+'/checkpoint',json=act('received',4)).status_code==200
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(EncounterORM))==1
        assert db.scalar(select(func.count()).select_from(PendingCheckpointEventORM))==6
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
    app,engine=setup;c,p,path=seed(app,engine)
    # Existing explicitly approved CD/image stages remain intact after simplifying generic tasks.
    assert c.post(path+'/checkpoint-plan',json={'template':'material_images','idempotency_key':uuid4().hex}).status_code==200
    # A later visit must extend the main rail without stealing the older visit's branch.
    later=draft().model_dump(mode='json');later['encounter_date']='2026-10-19';later['pending']=[]
    later['reason_for_visit']['text']='Νεότερη επίσκεψη χωρίς εκκρεμότητες'
    assert c.post('/clinical/visit-capture/save',json={'context_id':ctx(c),'candidate':later}).status_code==200
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
                row=page.locator('#patientChart .patient-followthrough-branch .synthetic-pending-row').first
                if width==1280:
                    expect(row).to_contain_text('Παραλήφθηκε το CD;')
                    expect(page.locator('.journey-fork')).to_have_count(1)
                    page.locator('#captureOpen').click();page.locator('#captureText').fill('Πρόχειρη επίσκεψη')
                    expect(row.get_by_role('button',name='Ναι',exact=True).first).to_be_disabled()
                    expect(row.locator('select')).to_have_count(0)
                    expect(page.locator('#captureText')).to_have_value('Πρόχειρη επίσκεψη')
                    page.locator('#captureComposer').get_by_role('button',name='Κλείσιμο',exact=True).click()
                    expect(row.get_by_role('button',name='Ναι',exact=True).first).to_be_enabled()
                    row.locator('.pending-stage').filter(has_text='Εξετάστηκαν οι εικόνες;').get_by_role('button',name='Ναι',exact=True).click()
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
                    row.locator('.pending-stage').filter(has_text='Εξετάστηκαν οι εικόνες;').get_by_role('button',name='Όχι ακόμη',exact=True).click()
                    expect(row).to_contain_text('Όχι ακόμη · καταγράφηκε')
                else:
                    expect(row).to_contain_text('Όχι ακόμη · καταγράφηκε')
                    row.get_by_role('button',name='Ναι',exact=True).click()
                    expect(row).to_contain_text('Ολοκληρώθηκε')
                points=page.locator('.patient-journey-point');expect(points).to_have_count(2)
                branches=page.locator('.patient-followthrough-branch');expect(branches).to_have_count(1)
                assert branches.get_attribute('id')=='followthrough-'+p['source_encounter_id']
                expect(branches.locator('..').locator('.patient-journey-point')).to_contain_text('Επανέλεγχος γόνατος')
                expect(branches).not_to_contain_text('Νεότερη επίσκεψη')
                expect(row.locator('.patient-followthrough-rail .pending-stage')).to_have_count(2)
                expect(row.locator('.patient-followthrough-end')).to_have_attribute('data-complete',str(width==390).lower())
                expect(points.filter(has_text='Παραλήφθηκε')).to_have_count(0)
                if width==390:
                    last=row.locator('.pending-stage').last
                    undo=last.get_by_role('button',name='Αναίρεση',exact=True).bounding_box()
                    delete=last.get_by_role('button',name='Διαγραφή υποχρέωσης',exact=True).bounding_box()
                    assert abs(undo['y']+undo['height']/2-delete['y']-delete['height']/2)<1
                    row.locator('.pending-stage').filter(has_text='✓ Εξετάστηκαν οι εικόνες').get_by_role('button',name='Αναίρεση',exact=True).click()
                    expect(row.locator('.patient-followthrough-end')).to_have_attribute('data-complete','false')
                    expect(row).to_contain_text('✓ Παραλήφθηκε το CD')
                    page.reload();expect(row.locator('.patient-followthrough-end')).to_have_attribute('data-complete','false')
                page.locator('.patient-obligation-link button').click()
                expect(page.locator('.patient-pending-dialog')).to_be_visible()
                page.locator('.patient-pending-dialog').get_by_role('button',name='Κλείσιμο',exact=True).click()
                expect(branches).to_have_attribute('open','')
                # Geometry, not just labels: the finite fork starts at its own primary dot.
                page.wait_for_function("document.querySelectorAll('.journey-fork').length===1")
                geometry=page.locator('.patient-journey-track').evaluate("""track=>{
                    const dots=[...track.querySelectorAll('.patient-journey-dot')].map(n=>n.getBoundingClientRect());
                    const line=track.querySelector('.journey-fork'),box=track.getBoundingClientRect();
                    const start=line.getPointAtLength(0),end=line.getPointAtLength(line.getTotalLength());
                    const stop=track.querySelector('.patient-followthrough-stop').getBoundingClientRect();
                    return {sameY:Math.abs(dots[0].y-dots[1].y)<1,
                      start:Math.abs(start.x-(dots[0].x-box.x+dots[0].width/2))<1 && Math.abs(start.y-(dots[0].y-box.y+dots[0].height/2))<1,
                      end:Math.abs(end.x-(stop.x-box.x+stop.width/2))<1 && Math.abs(end.y-(stop.y-box.y+stop.height/2))<1,
                      below:stop.y>dots[0].y,source:line.dataset.source};
                }""")
                assert geometry=={'sameY':True,'start':True,'end':True,'below':True,'source':p['source_encounter_id']}
                branches.locator(':scope > summary').click();expect(page.locator('.journey-fork')).to_have_count(0)
                branches.locator(':scope > summary').click();expect(page.locator('.journey-fork')).to_have_count(1)
                if width==390:
                    expect(page.locator('.patient-journey-scroll-hint')).to_be_visible()
                    points.last.click();expect(page.get_by_role('dialog')).to_contain_text('Νεότερη επίσκεψη')
                    page.get_by_role('dialog').get_by_role('button',name='Κλείσιμο',exact=True).click()
                    page.locator('.patient-journey-rail').evaluate('n=>n.scrollLeft=0')
                page.locator('.patient-timeline').screenshot(path=str(tmp_path/f'connected-journey-{width}.png'))
                points.filter(has_text='Επανέλεγχος γόνατος').click();expect(page.locator('.patient-journey-preview')).to_contain_text('Επανέλεγχος γόνατος')
                expect(page.get_by_role('dialog')).to_be_visible()
                page.screenshot(path=str(tmp_path/f'cp2-popup-{width}.png'),full_page=False)
                page.get_by_role('button',name='Άνοιγμα',exact=True).click()
                expect(page.get_by_role('dialog')).not_to_be_visible()
                expect(page.locator('#visit-'+p['source_encounter_id'])).to_be_visible()
                expect(page.locator('.patient-chart-overview')).not_to_be_visible()
                expect(page.locator('.patient-visit:visible')).to_have_count(1)
                expect(page.locator('.patient-visit:visible')).to_contain_text('Επανέλεγχος γόνατος')
                page.locator('#patientVisitBack').click()
                expect(page.locator('.patient-chart-overview')).to_be_visible()
                expect(page.locator('.patient-visit:visible')).to_have_count(0)
                points.filter(has_text='Νεότερη επίσκεψη').click()
                page.get_by_role('button',name='Άνοιγμα',exact=True).click()
                expect(page.locator('.patient-visit:visible')).to_have_count(1)
                expect(page.locator('.patient-visit:visible')).to_contain_text('Νεότερη επίσκεψη')
                expect(page.locator('#visit-'+p['source_encounter_id'])).not_to_be_visible()
                page.locator('#patientVisitBack').click()
                page.locator('.patient-journey-rail').evaluate('n=>n.scrollLeft=0')
                row.locator('.pending-open').first.click()
                page.get_by_role('dialog').get_by_role('button',name='Δημιουργήθηκε στην επίσκεψη',exact=False).click()
                expect(page.get_by_role('dialog')).not_to_be_visible()
                expect(page.locator('.patient-visit:visible')).to_have_count(1)
                expect(page.locator('#visit-'+p['source_encounter_id'])).to_be_visible()
                expect(page.locator('.patient-chart-overview')).not_to_be_visible()
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), page.evaluate("Array.from(document.querySelectorAll('body *')).filter(e=>e.getBoundingClientRect().right>innerWidth && e.getBoundingClientRect().width>0).slice(0,15).map(e=>[e.tagName,e.id,e.className,e.getBoundingClientRect().width])")
                page.locator('#patientChart').screenshot(path=str(tmp_path/f'cp2-{width}.png'))
                page.locator('#patientVisitBack').click()
                if width==390:
                    # A cancelled obligation closes its branch without claiming completion.
                    with Session(engine) as db:
                        db.get(ClinicalPendingORM,p['pending_id']).status='cancelled';db.commit()
                    page.reload()
                    expect(page.locator('.patient-followthrough-branch')).to_have_count(0)
                    expect(page.locator('.patient-deleted-obligations')).to_contain_text('Διαγραμμένες υποχρεώσεις')
                assert not errors;context.close()
            browser.close()
    finally:server.should_exit=True;thread.join(10);sock.close()


def real_seed(app,engine):
    c,p,path=seed(app,engine)
    with Session(engine) as db:
        row=db.get(ClinicalPendingORM,p['pending_id']);row.patient_id='outside-synthetic-scope'
        db.get(EncounterORM,row.source_encounter_id).patient_id=row.patient_id;db.commit()
    return c,p,path.replace(TEST_ID,'outside-synthetic-scope')


def test_real_patient_existing_key_staff_exclusion_and_honest_actor(setup,monkeypatch):
    from clinical_staff_auth import STAFF_COOKIE
    app,engine=setup;c,p,path=real_seed(app,engine)
    monkeypatch.setenv('COCKPIT_STAFF_AUTH_ENABLED','1')
    monkeypatch.setenv('COCKPIT_STAFF_CLINICIAN_KEY','C'*40)
    monkeypatch.setenv('COCKPIT_STAFF_SECRETARY_KEY','S'*40)
    monkeypatch.setenv('COCKPIT_STAFF_SESSION_SECRET','T'*40)
    assert c.get('/clinical/patient/outside-synthetic-scope/chart').json()['pending'][0]['checkpoint_write_allowed']
    action={'template':'material_images','first_observation':'yes_confirmed','idempotency_key':uuid4().hex}
    assert c.post(path+'/checkpoint-plan',json=action).status_code==200
    step={'checkpoint_key':'images_reviewed','observation':'yes_confirmed','expected_revision':1,'idempotency_key':uuid4().hex}
    for cookie in ['', 'invalid']:
        denied=login(app);denied.cookies.set(STAFF_COOKIE,cookie,domain='testserver.local',path='/clinical')
        assert denied.post(path+'/checkpoint',json=step,headers={'X-Clinical-Key':KEY}).status_code in (401,403)
    secretary=TestClient(app,base_url='https://testserver')
    assert secretary.post('/clinical/login',json={'key':'S'*40}).status_code==200
    assert secretary.post(path+'/checkpoint',json=step,headers={'X-Clinical-Key':KEY}).status_code==403
    assert secretary.post(path+'/lifecycle',json={'action':'cancel','expected_revision':1,'idempotency_key':uuid4().hex},headers={'X-Clinical-Key':KEY}).status_code==403
    assert TestClient(app).post(path+'/checkpoint',json=step).status_code==401
    clinician=TestClient(app,base_url='https://testserver')
    assert clinician.post('/clinical/login',json={'key':'C'*40}).status_code==200
    assert clinician.post(path+'/checkpoint',json=step).status_code==200
    monkeypatch.setenv('COCKPIT_STAFF_CLINICIAN_KEY','R'*40)
    assert clinician.post(path+'/checkpoint',json=step,headers={'X-Clinical-Key':KEY}).status_code==401
    with Session(engine) as db:
        events=db.scalars(select(PendingCheckpointEventORM).order_by(PendingCheckpointEventORM.revision)).all()
        assert [(e.actor_principal_id,e.actor_role) for e in events]==[('shared-clinical-key','shared_clinical_key')]*2+[('clinic-clinician','clinician')]


def test_real_lifecycle_cancel_restore_no_plan_and_active_plan(setup):
    app,engine=setup;c,p,path=real_seed(app,engine)
    with Session(engine) as db:source=dict(db.get(EncounterORM,p['source_encounter_id']).payload_json)
    def lifecycle(action,rev,cancel=None,key=None):
        return {'action':action,'expected_revision':rev,'cancellation_event_id':cancel,'idempotency_key':key or uuid4().hex}
    body=lifecycle('cancel',0);r=c.post(path+'/lifecycle',json=body);assert r.status_code==200,r.text
    cancel=r.json()['event_id'];assert r.json()['current'] is None
    assert c.post(path+'/lifecycle',json=body).json()['replayed']
    assert c.post(path+'/lifecycle',json={**body,'action':'restore'}).status_code==409
    plan={'template':'completion','first_observation':'yes_confirmed','idempotency_key':uuid4().hex}
    assert c.post(path+'/checkpoint-plan',json=plan).status_code==409
    assert c.post(path+'/lifecycle',json=lifecycle('restore',0,cancel)).status_code==409
    assert c.post(path+'/lifecycle',json=lifecycle('restore',1,str(uuid4()))).status_code==409
    restore=lifecycle('restore',1,cancel);assert c.post(path+'/lifecycle',json=restore).status_code==200
    assert c.post(path+'/lifecycle',json=restore).json()['replayed']
    assert c.post(path+'/checkpoint-plan',json=plan).status_code==409
    r=c.post(path+'/checkpoint-plan',json={**plan,'expected_revision':2});assert r.status_code==200,r.text
    assert r.json()['current']['completed'] and r.json()['current']['revision']==4
    # Undo completion, cancel, restore exact pre-cancel open state, then complete again.
    event=r.json()['current']['steps'][0]['event_id']
    undo={'checkpoint_key':'completed','observation':'retracted','retracts_event_id':event,'expected_revision':4,'idempotency_key':uuid4().hex}
    assert c.post(path+'/checkpoint',json=undo).status_code==200
    r=c.post(path+'/lifecycle',json=lifecycle('cancel',5));assert r.status_code==200
    second_cancel=r.json()['event_id'];assert r.json()['current']['completed'] is False
    assert c.post(path+'/checkpoint',json={**undo,'expected_revision':6,'idempotency_key':uuid4().hex}).status_code==409
    assert c.post(path+'/lifecycle',json=lifecycle('restore',6,cancel)).status_code==409
    r=c.post(path+'/lifecycle',json=lifecycle('restore',6,second_cancel));assert r.status_code==200
    assert r.json()['current']['status']=='open' and not r.json()['current']['completed']
    yes={'checkpoint_key':'completed','observation':'yes_confirmed','expected_revision':7,'idempotency_key':uuid4().hex}
    assert c.post(path+'/checkpoint',json=yes).status_code==200
    chart=login(app).get('/clinical/patient/outside-synthetic-scope/chart').json()['pending'][0]
    update=next(x for x in c.get('/clinical/clinical-updates').json()['items'] if x['pending_id']==p['pending_id'])
    assert chart['checkpoints']==update['checkpoints'] and chart['lifecycle']==update['lifecycle']
    assert len(chart['checkpoints']['history'])==3 and chart['status']=='resolved'
    assert c.post(path.replace('outside-synthetic-scope',TEST_ID)+'/lifecycle',json=lifecycle('cancel',8)).status_code==404
    with Session(engine) as db:
        assert db.get(EncounterORM,p['source_encounter_id']).payload_json==source
        assert db.scalar(select(func.count()).select_from(PendingCheckpointEventORM))==8


def test_browser_real_obligation_popup_complete_delete_restore(setup,tmp_path):
    import socket,threading,shutil
    import uvicorn
    from playwright.sync_api import sync_playwright,expect
    app,engine=setup;c,p,path=real_seed(app,engine)
    # Real-role chart contains a prior follow-up plus a second visit: attendance alone does not close it.
    with Session(engine) as db:
        row=db.get(ClinicalPendingORM,p['pending_id']);row.description='Επανεκτίμηση του γόνατος σε επόμενη επίσκεψη.';row.item_type='follow_up';db.commit()
    later=draft().model_dump(mode='json');later['encounter_date']='2026-10-19';later['pending']=[]
    assert c.post('/clinical/visit-capture/save',json={'context_id':ctx(c),'candidate':later}).status_code==200
    with Session(engine) as db:
        row=db.scalars(select(EncounterORM).where(EncounterORM.encounter_date=='2026-10-19')).one();row.patient_id='outside-synthetic-scope';db.commit()
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,log_level='error'));thread=threading.Thread(target=server.run,kwargs={'sockets':[sock]},daemon=True);thread.start()
    base=f'http://localhost:{port}';url=base+'/static/cockpit/index.html?chart=outside-synthetic-scope'
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(executable_path=shutil.which('google-chrome') or shutil.which('chromium'))
            for width in [1280,390]:
                context=browser.new_context(viewport={'width':width,'height':950});page=context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
                page.goto(url.split('?')[0]);page.locator('#ceGlobalKey').click();page.locator('#ceGlobalKeyInput').fill(KEY)
                page.locator('#ceGlobalLoginForm button[type=submit]').click();expect(page.locator('#ceGlobalAuthStatus')).to_contain_text('Συνδέθηκες');page.keyboard.press('Escape');page.goto(url)
                link=page.locator('.patient-chart-grid .patient-obligation-link button');link.click()
                popup=page.locator('.patient-pending-dialog');expect(popup).to_be_visible()
                expect(popup.locator('select')).to_have_count(0)
                expect(popup).not_to_contain_text('CD: παραλαβή')
                expect(popup).not_to_contain_text('Παρακολούθηση')
                expect(popup.get_by_role('button',name='Ναι',exact=True)).to_be_enabled()
                if width==1280:
                    popup.get_by_role('button',name='Όχι ακόμη',exact=True).click();expect(popup).to_have_count(0)
                    link.click();expect(popup).to_contain_text('Όχι ακόμη')
                popup.get_by_role('button',name='Ναι',exact=True).click()
                expect(page.locator('.patient-chart-grid .patient-obligation-link')).to_contain_text('Ολοκληρώθηκε')
                page.reload();link.click();expect(popup).to_contain_text('✓ Ολοκληρώθηκε η ενέργεια')
                expect(popup.get_by_role('button',name='Διαγραφή υποχρέωσης',exact=True)).to_be_visible()
                popup.screenshot(path=str(tmp_path/f'obligation-popup-{width}.png'))
                assert popup.bounding_box()['width']<=width
                popup.get_by_role('button',name='Κλείσιμο',exact=True).click()
                # Inline confirmation works without a dialog, including a completed task.
                branch=page.locator('.patient-followthrough-branch')
                branch.get_by_role('button',name='Διαγραφή υποχρέωσης',exact=True).click()
                branch.get_by_role('button',name='Διατήρηση',exact=True).click()
                expect(branch.get_by_role('button',name='Διαγραφή υποχρέωσης',exact=True)).to_be_visible()
                branch.get_by_role('button',name='Διαγραφή υποχρέωσης',exact=True).click()
                branch.get_by_role('button',name='Διαγραφή',exact=True).click()
                expect(branch).to_have_count(0)
                page.reload();page.locator('.patient-deleted-obligations summary').click();link.click()
                expect(popup.get_by_role('button',name='Ναι',exact=True)).to_have_count(0)
                popup.get_by_role('button',name='Επαναφορά υποχρέωσης',exact=True).click()
                expect(branch).to_have_count(1)
                link.click();expect(popup).to_contain_text('✓ Ολοκληρώθηκε η ενέργεια')
                # The same resolved cancellation/restore works through the popup.
                popup.get_by_role('button',name='Διαγραφή υποχρέωσης',exact=True).click()
                popup.get_by_role('button',name='Διατήρηση',exact=True).click();expect(popup).to_be_visible()
                popup.get_by_role('button',name='Διαγραφή υποχρέωσης',exact=True).click()
                popup.get_by_role('button',name='Διαγραφή',exact=True).click();expect(branch).to_have_count(0)
                page.locator('.patient-deleted-obligations summary').click();link.click()
                popup.get_by_role('button',name='Επαναφορά υποχρέωσης',exact=True).click()
                link.click();expect(popup).to_contain_text('✓ Ολοκληρώθηκε η ενέργεια')
                popup.get_by_role('button',name='Αναίρεση',exact=True).click()
                expect(popup).to_have_count(0)
                page.locator('#captureOpen').click();page.locator('#captureText').fill('Πρόχειρο που διατηρείται')
                expect(link).to_be_disabled();expect(page.locator('#captureText')).to_have_value('Πρόχειρο που διατηρείται')
                page.locator('#captureComposer').get_by_role('button',name='Κλείσιμο',exact=True).click()
                # Updates opens the same dialog, not merely a chart scroll.
                page.evaluate("""()=>{window._updates=document.createElement('section');document.body.append(window._updates);window._updates.id='proofUpdates';CockpitStoredChart.renderUpdates(window._updates,()=>true,()=>window._updates.remove(),'real')}""")
                page.locator('#proofUpdates .doctor-action-row').click();expect(popup).to_be_visible()
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                page.reload();expect(popup).to_have_count(0);expect(page.locator('.patient-followthrough-branch')).to_be_visible()
                assert not errors;context.close()
            browser.close()
    finally:server.should_exit=True;thread.join(10);sock.close()


@pytest.mark.parametrize('with_plan',[True,False])
def test_remove_resolved_preserves_confirmation_and_exact_resolution_pointer(setup,with_plan):
    from copy import deepcopy
    app,engine=setup;c,p,path=real_seed(app,engine)
    if with_plan:
        r=c.post(path+'/checkpoint-plan',json={'template':'completion','first_observation':'yes_confirmed','idempotency_key':uuid4().hex})
        assert r.status_code==200
        revision=r.json()['current']['revision'];confirmed=r.json()['current']['confirmed_keys']
    else:
        with Session(engine) as db:
            row=db.get(ClinicalPendingORM,p['pending_id']);row.status='resolved';row.resolution_event_id=str(uuid4());db.commit()
        revision=0;confirmed=None
    with Session(engine) as db:
        original_resolution=db.get(ClinicalPendingORM,p['pending_id']).resolution_event_id
        original_source=deepcopy(db.get(EncounterORM,p['source_encounter_id']).payload_json)
        count=db.scalar(select(func.count()).select_from(PendingCheckpointEventORM))
    cancel={'action':'cancel','expected_revision':revision,'idempotency_key':uuid4().hex}
    r=c.post(path+'/lifecycle',json=cancel);assert r.status_code==200,r.text
    event=r.json()['event_id'];assert c.post(path+'/lifecycle',json=cancel).json()['replayed']
    with Session(engine) as db:
        row=db.get(ClinicalPendingORM,p['pending_id']);assert row.status=='cancelled' and row.resolution_event_id is None
    restore={'action':'restore','expected_revision':revision+1,'cancellation_event_id':event,'idempotency_key':uuid4().hex}
    assert c.post(path+'/lifecycle',json={**restore,'expected_revision':revision}).status_code==409
    r=c.post(path+'/lifecycle',json=restore);assert r.status_code==200
    assert c.post(path+'/lifecycle',json=restore).json()['replayed']
    state=c.get('/clinical/patient/outside-synthetic-scope/chart').json()['pending'][0]
    assert state['status']=='resolved' and (state['checkpoints']['confirmed_keys'] if with_plan else state['checkpoints'])==confirmed
    with Session(engine) as db:
        assert db.get(ClinicalPendingORM,p['pending_id']).resolution_event_id==original_resolution
        assert db.get(EncounterORM,p['source_encounter_id']).payload_json==original_source
        assert db.scalar(select(func.count()).select_from(PendingCheckpointEventORM))==count+2


def test_patient_task_groups_and_incoming_sources_remain_separate(setup,tmp_path):
    import socket,threading,shutil
    import uvicorn
    from copy import deepcopy
    from playwright.sync_api import sync_playwright,expect
    app,engine=setup;c,p,path=real_seed(app,engine)
    extra=draft().model_dump(mode='json');extra['pending']=[deepcopy(extra['pending'][i%2]) for i in range(4)]
    for i,item in enumerate(extra['pending']):item['description']='Συνθετική ενέργεια '+str(i)
    assert c.post('/clinical/visit-capture/save',json={'context_id':ctx(c),'candidate':extra}).status_code==200
    with Session(engine) as db:
        for visit in db.scalars(select(EncounterORM).where(EncounterORM.patient_id==TEST_ID)):
            visit.patient_id='outside-synthetic-scope'
        added=db.scalars(select(ClinicalPendingORM).where(ClinicalPendingORM.patient_id==TEST_ID)).all()
        for item in added:item.patient_id='outside-synthetic-scope'
        next(x for x in added if x.description=='Συνθετική ενέργεια 3').status='resolved'
        next(x for x in added if x.description=='Συνθετική ενέργεια 2').status='cancelled';db.commit()
    # A namesake is a separate identity, never merged by the display name.
    other=c.post('/clinical/patients/create',json={'creation_id':str(uuid4()),'full_name':'Protected sentinel'}).json()['patient_id']
    another=draft().model_dump(mode='json');another['pending']=another['pending'][:1]
    assert c.post('/clinical/visit-capture/save',json={'context_id':ctx(c,other),'candidate':another}).status_code==200
    active=c.get('/clinical/clinical-updates?active_only=true').json()['items']
    assert len(active)==4 and {x['status'] for x in active} <= {'open','awaiting_patient','awaiting_result'}
    assert len(c.get('/clinical/clinical-updates').json()['items'])==6
    recent=c.get('/clinical/recent-encounters').json();assert len(recent)==2
    assert {x['patient_id'] for x in recent}=={'outside-synthetic-scope',other}
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,log_level='error'));thread=threading.Thread(target=server.run,kwargs={'sockets':[sock]},daemon=True);thread.start()
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(executable_path=shutil.which('google-chrome') or shutil.which('chromium'))
            for width in [1280,390]:
                context=browser.new_context(viewport={'width':width,'height':950});page=context.new_page();reads=[]
                page.on('request',lambda r:reads.append(r.url) if '/clinical-updates' in r.url else None)
                page.goto(f'http://localhost:{port}/static/cockpit/index.html')
                page.locator('#ceGlobalKey').click();page.locator('#ceGlobalKeyInput').fill(KEY)
                page.locator('#ceGlobalLoginForm button[type=submit]').click();expect(page.locator('#ceGlobalAuthStatus')).to_contain_text('Συνδέθηκες');page.keyboard.press('Escape')
                expect(page.locator('#visitRecentRows .clinical-patient-row')).to_have_count(2)
                page.locator('#doctorInboxButton').click();expect(page.locator('#doctorPopoverTitle')).to_have_text('Κλινικά εισερχόμενα')
                expect(page.locator('#doctorPopoverContent')).to_contain_text('Δεν έχει συνδεθεί')
                expect(page.locator('#doctorPopoverContent button.doctor-action-row')).to_have_count(0);assert not reads
                page.locator('#doctorTasksButton').click();expect(page.locator('#doctorPopoverTitle')).to_have_text('Υποχρεώσεις ασθενών')
                groups=page.locator('.patient-updates-group');expect(groups).to_have_count(2)
                expect(groups.locator('h3')).to_have_count(2);expect(groups.locator('.doctor-action-row')).to_have_count(4)
                assert sorted(groups.evaluate_all('gs=>gs.map(g=>g.querySelectorAll("button").length)'))==[1,3]
                assert len(set(groups.locator('button').evaluate_all('ns=>ns.map(n=>n.dataset.pendingId)')))==4
                expect(groups.filter(has_text='Συνθετική ενέργεια 3')).to_have_count(0)
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                page.locator('#doctorPopover').screenshot(path=str(tmp_path/f'task-groups-{width}.png'))
                selected=groups.locator('button').first;label=selected.locator('strong').text_content();selected.click()
                expect(page.locator('.patient-pending-dialog')).to_contain_text(label)
                context.close()
            browser.close()
    finally:server.should_exit=True;thread.join(10);sock.close()


@pytest.mark.parametrize('with_plan',[False,True])
def test_permanent_task_removal_is_terminal_and_keeps_signed_source(setup,with_plan):
    from copy import deepcopy
    app,engine=setup;c,p,path=real_seed(app,engine)
    _,neighbor,_=real_seed(app,engine)
    if with_plan:
        plan={'template':'completion','first_observation':'yes_confirmed','idempotency_key':uuid4().hex}
        assert c.post(path+'/checkpoint-plan',json=plan).status_code==200
    state=next(x for x in c.get('/clinical/patient/outside-synthetic-scope/chart').json()['pending'] if x['pending_id']==p['pending_id'])
    revision=(state['lifecycle'] or {})['revision']
    def body(action,rev,event=None):return {'action':action,'expected_revision':rev,'cancellation_event_id':event,'idempotency_key':uuid4().hex}
    assert c.post(path+'/lifecycle',json=body('delete_permanently',revision)).status_code==409
    cancelled=c.post(path+'/lifecycle',json=body('cancel',revision)).json();event=cancelled['event_id']
    permanent=body('delete_permanently',revision+1,event)
    assert c.post(path+'/lifecycle',json={**permanent,'expected_revision':revision}).status_code==409
    assert c.post(path+'/lifecycle',json={**permanent,'cancellation_event_id':str(uuid4())}).status_code==409
    with Session(engine) as db:
        original=deepcopy(db.get(EncounterORM,p['source_encounter_id']).payload_json)
        previous=[(e.id,e.kind,e.plan_json) for e in db.scalars(select(PendingCheckpointEventORM)).all()]
    r=c.post(path+'/lifecycle',json=permanent);assert r.status_code==200,r.text
    assert r.json()['current'] is None and c.post(path+'/lifecycle',json=permanent).json()['replayed']
    assert c.post(path+'/lifecycle',json={**permanent,'expected_revision':revision+2}).status_code==409
    assert c.post(path+'/lifecycle',json=body('restore',revision+2,event)).status_code==409
    assert c.post(path+'/lifecycle',json=body('cancel',revision+2)).status_code==409
    assert c.post(path+'/checkpoint-plan',json={'template':'completion','expected_revision':revision+2,'idempotency_key':uuid4().hex}).status_code==409
    assert c.post(path+'/checkpoint',json={'checkpoint_key':'completed','observation':'yes_confirmed','expected_revision':revision+2,'idempotency_key':uuid4().hex}).status_code==409
    assert c.post(path.replace('outside-synthetic-scope',TEST_ID)+'/lifecycle',json=permanent).status_code==404
    second=login(app)
    for route in ['/clinical/patient/outside-synthetic-scope/chart','/clinical/patient/outside-synthetic-scope/pending','/clinical/clinical-updates','/clinical/clinical-updates?active_only=true']:
        data=second.get(route).json();items=data if isinstance(data,list) else data.get('pending',data.get('items',[]))
        assert not any(x['pending_id']==p['pending_id'] for x in items)
        assert any(x['pending_id']==neighbor['pending_id'] for x in items)
    with Session(engine) as db:
        assert db.get(ClinicalPendingORM,p['pending_id']).status=='deleted'
        assert db.get(EncounterORM,p['source_encounter_id']).payload_json==original
        now=db.scalars(select(PendingCheckpointEventORM)).all()
        assert [(e.id,e.kind,e.plan_json) for e in now if e.kind!='permanently_deleted']==previous
        assert len([e for e in now if e.kind=='permanently_deleted'])==1


def test_legacy_writer_cannot_resurrect_permanent_task_with_current_timestamp(setup):
    from clinical_synthetic_161 import PREFIX,SYNTHETIC_161_PATIENT_IDS
    app,engine=setup;c=login(app);assert c.post(PREFIX+'/prepare').status_code==200
    pid=SYNTHETIC_161_PATIENT_IDS[0];pending=pid+'-pending'
    with Session(engine) as db:
        row=db.get(ClinicalPendingORM,pending);row.status='deleted';db.commit();stamp=row.updated_at.isoformat()+'Z'
    assert c.post(PREFIX+'/patient/'+pid+'/pending/'+pending,json={'status':'resolved','expected_updated_at':stamp}).status_code==409
    assert c.post(PREFIX+'/prepare').status_code==200
    with Session(engine) as db:assert db.get(ClinicalPendingORM,pending).status=='deleted'
    for route in ['/patient/'+pid+'/chart','/clinical-updates']:
        data=c.get(PREFIX+route).json();assert all(x['pending_id']!=pending for x in data.get('pending',data.get('items',[])))


def test_compact_branch_and_permanent_removal_browser(setup,tmp_path):
    import socket,threading,shutil
    import uvicorn
    from playwright.sync_api import sync_playwright,expect
    app,engine=setup
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,log_level='error'));thread=threading.Thread(target=server.run,kwargs={'sockets':[sock]},daemon=True);thread.start()
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(executable_path=shutil.which('google-chrome') or shutil.which('chromium'))
            for width in [1280,390]:
                c,p,path=real_seed(app,engine)
                with Session(engine) as db:
                    row=db.get(ClinicalPendingORM,p['pending_id']);row.description='Επανεκτίμηση γόνατος σε επόμενη επίσκεψη.';db.commit()
                assert c.post(path+'/checkpoint-plan',json={'template':'completion','first_observation':'yes_confirmed','idempotency_key':uuid4().hex}).status_code==200
                context=browser.new_context(viewport={'width':width,'height':950});page=context.new_page();page.goto(f'http://localhost:{port}/static/cockpit/index.html')
                page.locator('#ceGlobalKey').click();page.locator('#ceGlobalKeyInput').fill(KEY)
                page.locator('#ceGlobalLoginForm button[type=submit]').click();expect(page.locator('#ceGlobalAuthStatus')).to_contain_text('Συνδέθηκες');page.keyboard.press('Escape')
                page.goto(f'http://localhost:{port}/static/cockpit/index.html?chart=outside-synthetic-scope')
                row=page.locator('.patient-followthrough-branch article[data-pending-id="'+p['pending_id']+'"]')
                expect(row).to_be_visible();expect(row.locator('blockquote')).to_have_count(0)
                expect(row.get_by_role('button',name='Δημιουργήθηκε στην επίσκεψη',exact=False)).to_have_count(0)
                undo=row.get_by_role('button',name='Αναίρεση',exact=True).bounding_box();remove=row.get_by_role('button',name='Διαγραφή υποχρέωσης',exact=True).bounding_box()
                assert abs(undo['y']-remove['y'])<1 and remove['x']>undo['x']
                assert row.bounding_box()['height']<220
                page.locator('.patient-timeline').screenshot(path=str(tmp_path/f'compact-trajectory-{width}.png'))
                row.locator('.pending-open').first.click();popup=page.locator('.patient-pending-dialog')
                expect(popup.locator('blockquote')).to_have_count(1)
                expect(popup.get_by_role('button',name='Δημιουργήθηκε στην επίσκεψη',exact=False)).to_be_visible()
                expect(popup.get_by_role('button',name='Μόνιμη διαγραφή',exact=True)).to_have_count(0)
                popup.get_by_role('button',name='Διαγραφή υποχρέωσης',exact=True).click();popup.get_by_role('button',name='Διαγραφή',exact=True).click()
                page.locator('.patient-deleted-obligations summary').click()
                page.locator('.patient-deleted-obligations button').filter(has_text='Επανεκτίμηση γόνατος').click()
                popup.get_by_role('button',name='Μόνιμη διαγραφή',exact=True).click()
                expect(popup).to_contain_text('δεν θα μπορεί να επανέλθει');expect(popup).to_contain_text('Η αποθηκευμένη επίσκεψη διατηρείται')
                popup.get_by_role('button',name='Διατήρηση',exact=True).click()
                expect(popup.get_by_role('button',name='Μόνιμη διαγραφή',exact=True)).to_be_visible()
                popup.get_by_role('button',name='Μόνιμη διαγραφή',exact=True).click()
                popup.screenshot(path=str(tmp_path/f'permanent-confirmation-{width}.png'))
                popup.get_by_role('button',name='Οριστική διαγραφή',exact=True).click();expect(popup).to_have_count(0)
                page.reload();expect(page.locator('[data-pending-id="'+p['pending_id']+'"]')).to_have_count(0)
                expect(page.locator('.patient-deleted-obligations')).to_have_count(0)
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                assert c.get('/clinical/patient/outside-synthetic-scope/pending').json()==[]
                context.close()
            browser.close()
    finally:server.should_exit=True;thread.join(10);sock.close()
