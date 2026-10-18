"""Explicit per-booking chart choice, same clinical session, no name-inferred identity."""
from datetime import datetime,timedelta,timezone
from uuid import uuid4
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import select,func
from sqlalchemy.orm import Session
import clinical_visit_brief as brief
from clinical_calendar import ActualBooking
from clinical_data import PatientORM,EncounterORM
from clinical_visit_brief import BookingPatientLinkORM,BookingPatientLinkAuditORM
from test_cockpit_synthetic_161 import setup,login,KEY
from test_cockpit_cp2 import real_seed

PATH='/clinical/calendar/visit-brief/'

@pytest.fixture
def fixture(setup,monkeypatch):
    app,engine=setup;c,p,_=real_seed(app,engine)
    with Session(engine) as db:
        patient=db.get(PatientORM,'outside-synthetic-scope');patient.demographics_json={'full_name':'Συνθετικός Ραντεβού','dob':'1970-01-01'};db.commit()
    now=datetime.now(timezone.utc)
    source=ActualBooking(uid='cal.com:chart-fixture',start_at=now+timedelta(days=1),end_at=now+timedelta(days=1,minutes=30),clinic='test',patient_display_name='Συνθετικός Ραντεβού',reason='Συνθετικός επανέλεγχος',category='other')
    state={'source':source,'failure':None,'dates':[]}
    async def fresh(uid,target_date=None):
        state['dates'].append(target_date)
        if state['failure']:raise HTTPException(state['failure'],'Unavailable source')
        if uid not in ('chart-fixture',source.uid):raise HTTPException(409,'Missing booking')
        return state['source']
    monkeypatch.setattr(brief,'_fresh_booking',fresh)
    return app,engine,c,p,state


def selection(c,read=None,**extra):
    read=read or c.post(PATH+'chart-read',json={'booking_uid':'chart-fixture'}).json()
    return {'booking_uid':'chart-fixture','patient_id':'outside-synthetic-scope','expected_revision':read['revision'],
        'observed_booking':read['booking']['observation'],'replace_existing':read.get('replace_required',False),'request_id':str(uuid4()),**extra}


def test_explicit_link_replay_persistence_drift_cas_and_revoke(fixture):
    app,engine,c,p,state=fixture
    read=c.post(PATH+'chart-read',json={'booking_uid':'chart-fixture'}).json()
    assert read['mode']=='booking_only' and 'latest_visit' not in read and 'patient_id' not in read
    body=selection(c,read);r=c.post(PATH+'chart-select',json=body);assert r.status_code==200,r.text
    assert r.json()['latest_visit']['source_encounter_id']==p['source_encounter_id']
    assert c.post(PATH+'chart-select',json=body).status_code==200
    assert r.headers['cache-control']=='no-store'
    again=login(app).post(PATH+'chart-read',json={'booking_uid':'cal.com:chart-fixture'}).json();assert again['mode']=='chart_linked'
    assert c.post(PATH+'chart-select',json=selection(c,read)).status_code==409
    with Session(engine) as db:
        past=state['source'].model_copy(update={'start_at':datetime.now(timezone.utc)-timedelta(days=1)})
        assert brief._minimal_brief(db,'outside-synthetic-scope',past)['previous_visit'] is None
        assert brief._latest_chart_summary(db,'outside-synthetic-scope')['source_encounter_id']==p['source_encounter_id']
    # Rescheduling/reason updates keep an explicitly chosen chart; attendee drift does not.
    state['source']=state['source'].model_copy(update={'start_at':state['source'].start_at+timedelta(days=1),'reason':'Νέος λόγος'})
    assert c.post(PATH+'chart-read',json={'booking_uid':'chart-fixture'}).json()['mode']=='chart_linked'
    state['source']=state['source'].model_copy(update={'patient_display_name':'Άλλο πρόσωπο'})
    drift=c.post(PATH+'chart-read',json={'booking_uid':'chart-fixture'}).json();assert drift['mode']=='booking_only' and 'patient_id' not in drift
    assert c.post(PATH+'chart-select',json=body).status_code==409
    corrected=selection(c,drift);assert c.post(PATH+'chart-select',json=corrected).status_code==200
    unlink={'booking_uid':'chart-fixture','expected_revision':2,'request_id':str(uuid4())}
    state['failure']=503
    assert c.post(PATH+'chart-read',json={'booking_uid':'chart-fixture'}).status_code==503
    assert c.post(PATH+'chart-unlink',json=unlink).status_code==200
    assert c.post(PATH+'chart-unlink',json=unlink).status_code==200
    state['failure']=None
    read=c.post(PATH+'chart-read',json={'booking_uid':'chart-fixture'}).json();assert read['mode']=='booking_only'
    assert c.post(PATH+'chart-select',json=body).status_code==409
    assert c.post(PATH+'chart-select',json=selection(c,read)).status_code==200
    assert c.post(PATH+'chart-unlink',json=unlink).status_code==409
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(BookingPatientLinkORM))==1
        assert db.scalar(select(func.count()).select_from(BookingPatientLinkAuditORM))==4
        assert db.scalar(select(BookingPatientLinkAuditORM)).operator_type=='shared_clinical_key'
        assert db.get(EncounterORM,p['source_encounter_id']).patient_id=='outside-synthetic-scope'


def test_dated_authority_boundaries_and_latest_signed_visit(fixture,monkeypatch):
    from clinical_staff_auth import STAFF_COOKIE
    app,engine,c,p,state=fixture
    day=state['source'].start_at.date().isoformat()
    r=c.post(PATH+'chart-read',json={'booking_uid':'chart-fixture','target_date':day});assert r.status_code==200
    assert state['dates'][-1].isoformat()==day
    body=selection(c,r.json());assert c.post(PATH+'chart-select',json=body).status_code==200
    assert TestClient(app).post(PATH+'chart-read',json={'booking_uid':'chart-fixture'}).status_code==401
    for k,v in [('COCKPIT_STAFF_AUTH_ENABLED','1'),('COCKPIT_STAFF_CLINICIAN_KEY','C'*40),('COCKPIT_STAFF_SECRETARY_KEY','S'*40),('COCKPIT_STAFF_SESSION_SECRET','T'*40)]:monkeypatch.setenv(k,v)
    for key,status in [('C'*40,200),('S'*40,403)]:
        staff=TestClient(app,base_url='https://testserver');assert staff.post('/clinical/login',json={'key':key}).status_code==200
        assert staff.post(PATH+'chart-read',json={'booking_uid':'chart-fixture'},headers={'X-Clinical-Key':KEY}).status_code==status
    for token in ['', 'invalid']:
        bad=login(app);bad.cookies.set(STAFF_COOKIE,token,domain='testserver.local',path='/clinical')
        assert bad.post(PATH+'chart-read',json={'booking_uid':'chart-fixture'},headers={'X-Clinical-Key':KEY}).status_code==401
    # Existing staff-only route remains separately gated and does not gain shared-key authority.
    assert c.post(PATH+'read',json={'booking_uid':'chart-fixture'}).status_code==401
    state['failure']=409
    assert c.post(PATH+'chart-read',json={'booking_uid':'chart-fixture'}).status_code==409


def test_browser_single_login_booking_link_latest_and_independent_session(fixture,tmp_path):
    import socket,threading,shutil,json
    import uvicorn
    from playwright.sync_api import sync_playwright,expect
    app,engine,c,p,state=fixture
    for route in app.routes:
        if getattr(route,'path',None)=='/static':route.app.html=True
    c.post('/clinical/patients/create',json={'creation_id':str(uuid4()),'full_name':'Συνθετικός Ραντεβού','dob':'1980-01-01'})
    source=state['source'];record={'appointment_id':source.uid,'start_at':source.start_at.isoformat(),'end_at':source.end_at.isoformat(),'clinic':'test','category':'other','patient_display_name':source.patient_display_name,'reason':source.reason}
    home={'generated_at':datetime.now(timezone.utc).isoformat(),'source_updated_at':datetime.now(timezone.utc).isoformat(),'today_total':0,'previous':None,'current':None,'current_conflict_count':0,'next':record,'upcoming_today':[]}
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,log_level='error'));thread=threading.Thread(target=server.run,kwargs={'sockets':[sock]},daemon=True);thread.start()
    base=f'http://localhost:{port}';url=base+'/static/cockpit/'
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(executable_path=shutil.which('google-chrome') or shutil.which('chromium'))
            for width in [1280,390]:
                context=browser.new_context(viewport={'width':width,'height':950});page=context.new_page();page.set_default_timeout(8000);errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
                page.route('**/clinical/calendar/cockpit-context',lambda route:route.fulfill(json=home))
                page.goto(url);page.locator('#ceGlobalKey').click();page.locator('#ceGlobalKeyInput').fill(KEY)
                page.locator('#ceGlobalLoginForm button[type=submit]').click();expect(page.locator('#ceGlobalAuthStatus')).to_contain_text('Συνδέθηκες');page.keyboard.press('Escape')
                page.locator('#visitSidebarNext').click();overlay=page.locator('#visitBriefOverlay');expect(overlay).to_be_visible()
                if width==1280:
                    expect(page.locator('#visitBriefClinical')).not_to_be_visible()
                    page.locator('#visitBriefPatientEntry').click();expect(page.locator('#patientEntry')).to_be_visible()
                    expect(page.locator('.patient-entry-match')).to_have_count(2)
                    assert c.post(PATH+'chart-read',json={'booking_uid':source.uid}).json()['mode']=='booking_only'
                    page.locator('.patient-entry-match').filter(has_text='1970-01-01').get_by_role('button',name='Σύνδεση και άνοιγμα').click()
                    expect(page.locator('#patientChart')).to_contain_text('Κλινική συνέχεια')
                    page.locator('#visitBackHome').click();page.reload();page.locator('#visitSidebarNext').click()
                expect(page.locator('#visitBriefIdentity')).to_contain_text('Συνδεδεμένη καρτέλα')
                expect(page.locator('#visitBriefOpenLatest')).to_be_visible()
                expect(page.locator('#visitBriefPrevious')).to_contain_text('Επανέλεγχος γόνατος')
                overlay.screenshot(path=str(tmp_path/f'appointment-chart-{width}.png'))
                page.locator('#visitBriefOpenLatest').click();expect(page.locator('#visit-'+p['source_encounter_id'])).to_be_visible()
                if width==1280:
                    page.evaluate("CockpitHome.openAppointment({appointmentId:'cal.com:chart-fixture',name:'Συνθετικός Ραντεβού'})")
                    expect(page.locator('#visitBriefOpenLatest')).to_be_visible();page.locator('#visitBriefPatientEntry').click()
                    page.locator('#patientEntryName').fill('Νέα Συνθετική Σύνδεση')
                    page.route('**/visit-brief/chart-select',lambda route:route.fulfill(status=503,json={'detail':'Temporary link failure'}))
                    page.locator('#patientEntryCreate').click();expect(page.locator('#patientEntryStatus')).to_contain_text('Η καρτέλα υπάρχει')
                    assert len(c.get('/clinical/patients?query=Νέα Συνθετική Σύνδεση').json())==1
                    expect(page.locator('#patientEntryCreate')).to_be_disabled()
                    page.unroute('**/visit-brief/chart-select');page.get_by_role('button',name='Επανάληψη σύνδεσης',exact=True).click()
                    expect(page.locator('#patientEntry')).not_to_be_visible();expect(page.locator('#visitSelectedName')).to_have_text('Νέα Συνθετική Σύνδεση')
                    assert len(c.get('/clinical/patients?query=Νέα Συνθετική Σύνδεση').json())==1
                    # A selected link may commit after close, but must never reopen a closed selection/chart.
                    page.evaluate("CockpitHome.openAppointment({appointmentId:'cal.com:chart-fixture',name:'Συνθετικός Ραντεβού'})")
                    expect(page.locator('#visitBriefIdentity')).to_contain_text('Συνδεδεμένη καρτέλα');page.locator('#visitBriefPatientEntry').click()
                    held=[];page.route('**/visit-brief/chart-select',lambda route:held.append(route))
                    page.locator('.patient-entry-match').filter(has_text='1970-01-01').get_by_role('button',name='Σύνδεση και άνοιγμα').click()
                    page.wait_for_timeout(100);assert held
                    page.locator('#patientEntry').get_by_role('button',name='Κλείσιμο',exact=True).click()
                    response=held[0].fetch();held[0].fulfill(response=response);page.unroute('**/visit-brief/chart-select')
                    expect(page.locator('#patientEntry')).not_to_be_visible();expect(page.locator('#visitSelectedName')).to_have_text('Νέα Συνθετική Σύνδεση')
                    page.evaluate("CockpitHome.openAppointment({appointmentId:'cal.com:chart-fixture',name:'Συνθετικός Ραντεβού'})")
                    expect(page.locator('#visitBriefOpenLatest')).to_be_visible();page.locator('#visitBriefOpenChart').click()
                page.locator('#captureOpen').click();page.locator('#captureText').fill('Πρόχειρο')
                page.evaluate("CockpitHome.openAppointment({appointmentId:'cal.com:chart-fixture',name:'Συνθετικός Ραντεβού',date:'2026-10-19'})")
                expect(overlay).not_to_be_visible();expect(page.locator('#captureText')).to_have_value('Πρόχειρο')
                page.locator('#captureComposer').get_by_role('button',name='Κλείσιμο',exact=True).click()
                page.evaluate("CockpitHome.openAppointment({appointmentId:'cal.com:chart-fixture',name:'Συνθετικός Ραντεβού',date:'2026-10-19'})")
                expect(page.locator('#visitBriefOpenLatest')).to_be_visible()
                page.request.post(base+'/clinical/logout');page.evaluate("window.dispatchEvent(new Event('clinical:unauthenticated'))")
                expect(overlay).not_to_be_visible();expect(page.locator('#visitBriefPrevious')).to_be_empty()
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth');assert not errors,errors;context.close()
            browser.close()
    finally:server.should_exit=True;thread.join(10);sock.close()
