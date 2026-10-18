"""Normal patient entry: existing registry permission, explicit choice, no booking link."""
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from fastapi.testclient import TestClient
from sqlalchemy import select,func
from sqlalchemy.orm import Session
from clinical_data import PatientORM,EncounterORM,ClinicalPendingORM
from clinical_visit_brief import BookingPatientLinkORM
from test_cockpit_synthetic_161 import setup,login,KEY


def test_create_replay_race_conflict_and_no_clinical_side_effects(setup):
    app,engine=setup;c=login(app)
    body={'creation_id':str(uuid4()),'full_name':'  Δοκιμή   Καρτέλας  ','dob':'1970-04-05','phone':'  +357 12345678  '}
    assert TestClient(app).post('/clinical/patients/create',json=body).status_code==401
    def create(_):return login(app).post('/clinical/patients/create',json=body)
    with ThreadPoolExecutor(max_workers=2) as pool:responses=list(pool.map(create,range(2)))
    assert [r.status_code for r in responses]==[200,200]
    a,b=[r.json() for r in responses];assert a==b
    assert a['demographics']=={'full_name':'Δοκιμή Καρτέλας','dob':'1970-04-05','phone':'+357 12345678'}
    assert responses[0].headers['cache-control']=='no-store'
    assert c.post('/clinical/patients/create',json={**body,'full_name':'Other person'}).status_code==409
    assert c.get('/clinical/patient/'+a['patient_id']).json()['demographics']==a['demographics']
    # Namesakes must remain separate selectable profiles, never inferred to be the same person.
    same_name=c.post('/clinical/patients/create',json={**body,'creation_id':str(uuid4()),'dob':'1980-04-05'})
    assert same_name.status_code==200 and same_name.json()['patient_id']!=a['patient_id']
    assert len(c.get('/clinical/patients?query=Δοκιμή Καρτέλας').json())==2
    for invalid in [{**body,'patient_id':'outside-synthetic-scope'},{**body,'dob':'2999-01-01'},{**body,'full_name':' '},{**body,'creation_id':'not-uuid'}]:
        r=c.post('/clinical/patients/create',json=invalid);assert r.status_code==422 and body['full_name'] not in r.text
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(PatientORM).where(PatientORM.patient_id.like('patient-%')))==2
        for model in [EncounterORM,ClinicalPendingORM,BookingPatientLinkORM]:assert db.scalar(select(func.count()).select_from(model))==0


def test_create_uses_existing_staff_boundary(setup,monkeypatch):
    app,_=setup
    for k,v in [('COCKPIT_STAFF_AUTH_ENABLED','1'),('COCKPIT_STAFF_CLINICIAN_KEY','C'*40),('COCKPIT_STAFF_SECRETARY_KEY','S'*40),('COCKPIT_STAFF_SESSION_SECRET','T'*40)]:monkeypatch.setenv(k,v)
    body={'creation_id':str(uuid4()),'full_name':'Synthetic role fixture'}
    for key,expected in [('S'*40,401),('C'*40,200)]:
        c=TestClient(app,base_url='https://testserver');assert c.post('/clinical/login',json={'key':key}).status_code==200
        assert c.post('/clinical/patients/create',json=body).status_code==expected


def test_browser_shared_entry_and_late_results(setup,tmp_path):
    import socket,threading,shutil
    import uvicorn
    from playwright.sync_api import sync_playwright,expect
    app,engine=setup;c=login(app)
    # Match production directory-index serving (base fixture mounts file-only StaticFiles).
    for route in app.routes:
        if getattr(route,'path',None)=='/static':route.app.html=True
    old=c.post('/clinical/patients/create',json={'creation_id':str(uuid4()),'full_name':'Συνθετικός Είσοδος','dob':'1960-01-01'}).json()
    c.post('/clinical/patients/create',json={'creation_id':str(uuid4()),'full_name':'Συνθετικός Είσοδος','dob':'1980-01-01'})
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,log_level='error'));thread=threading.Thread(target=server.run,kwargs={'sockets':[sock]},daemon=True);thread.start()
    base=f'http://localhost:{port}';url=base+'/static/cockpit/'
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(executable_path=shutil.which('google-chrome') or shutil.which('chromium'))
            for width in [1280,390]:
                context=browser.new_context(viewport={'width':width,'height':950});page=context.new_page();page.set_default_timeout(8000);errors=[]
                page.on('pageerror',lambda e:errors.append(str(e)))
                page.goto(url);page.locator('#ceGlobalKey').click();page.locator('#ceGlobalKeyInput').fill(KEY)
                page.locator('#ceGlobalLoginForm button[type=submit]').click();expect(page.locator('#ceGlobalAuthStatus')).to_contain_text('Συνδέθηκες');page.keyboard.press('Escape')
                page.locator('#patientEntryHome').click();name=f'Νέα Συνθετική Καρτέλα {width}'
                page.locator('#patientEntryName').fill(name);page.locator('#patientEntryDob').fill('1972-02-03')
                page.locator('#patientEntryPhone').fill('12345678');page.locator('#patientEntryCreate').click()
                expect(page.locator('#patientEntry')).not_to_be_visible();expect(page.locator('#visitSelectedName')).to_have_text(name)
                expect(page.locator('#patientChart')).to_contain_text('Η πρώτη επίσκεψη')
                new_id=page.url.split('chart=')[1];page.reload();expect(page.locator('#visitSelectedName')).to_have_text(name)
                page.locator('#captureOpen').click();page.locator('#captureText').fill('Πρόχειρη καταγραφή που πρέπει να διατηρηθεί')
                page.evaluate("window.CockpitPatientEntry.open({name:'Other person'})")
                expect(page.locator('#patientEntry')).not_to_be_visible();expect(page.locator('#captureText')).to_have_value('Πρόχειρη καταγραφή που πρέπει να διατηρηθεί')
                page.locator('#visitBackHome').click();expect(page.locator('#captureText')).to_have_value('Πρόχειρη καταγραφή που πρέπει να διατηρηθεί')
                page.locator('#captureComposer').get_by_role('button',name='Κλείσιμο',exact=True).click();page.locator('#visitBackHome').click()
                # Provider booking boundary only is a fixture. Manual registry is real protected HTTP/DB.
                page.evaluate("""()=>{const p=document.getElementById('nextAppointmentPatient');p.textContent='Συνθετικός Είσοδος';p.dataset.appointmentId='patient-entry-fixture';document.getElementById('visitSidebarNext').disabled=false;document.getElementById('nextAppointmentTime').textContent='19/10/2026 · 09:40';}""")
                reads=[];page.on('request',lambda r:reads.append(r.url) if '/chart' in r.url else None)
                page.route('**/clinical/calendar/visit-brief/read',lambda route:route.fulfill(status=401 if width==1280 else 403,json={'detail':'Separate staff Brief authority unavailable'}))
                page.locator('#visitSidebarNext').click();expect(page.locator('#visitBriefIdentity')).to_contain_text('Η προεπισκόπηση δεν είναι διαθέσιμη',timeout=10000)
                expect(page.locator('#visitBriefName')).to_have_text('Συνθετικός Είσοδος')
                expect(page.locator('#visitBriefClinical')).not_to_be_visible()
                expect(page.locator('#visitBriefChoosePatient')).not_to_be_visible()
                page.locator('#visitBriefPatientEntry').click();expect(page.locator('#patientEntryName')).to_have_value('Συνθετικός Είσοδος')
                expect(page.locator('.patient-entry-match')).to_have_count(2);assert not reads
                page.locator('#patientEntry').screenshot(path=str(tmp_path/f'patient-entry-{width}.png'))
                page.locator('.patient-entry-match').filter(has_text='1960-01-01').get_by_role('button',name='Άνοιγμα καρτέλας').click()
                expect(page.locator('#visitSelectedName')).to_have_text('Συνθετικός Είσοδος');assert 'chart='+old['patient_id'] in page.url
                page.goto(base+'/static/cockpit/visit-capture/?patient_id='+new_id)
                expect(page.locator('#captureComposer')).to_be_visible();expect(page.locator('#captureText')).to_be_visible()
                assert '/visit-capture/' not in page.url;assert page.locator('#candidateInput').count()==0
                page.locator('#captureComposer').get_by_role('button',name='Κλείσιμο',exact=True).click()
                page.goto(base+'/static/cockpit/visit-capture/');expect(page.locator('#patientEntry')).to_be_visible()
                page.locator('#patientEntryName').fill(name);expect(page.locator('.patient-entry-match')).to_have_count(1)
                page.locator('.patient-entry-match').get_by_role('button',name='Καταγραφή επίσκεψης').click();expect(page.locator('#captureComposer')).to_be_visible()
                page.locator('#captureComposer').get_by_role('button',name='Κλείσιμο',exact=True).click()
                page.goto(base+'/static/cockpit/visit-capture/');expect(page.locator('#patientEntry')).to_be_visible()
                page.locator('#patientEntryName').fill(f'Από Καταγραφή {width}');page.locator('#patientEntryCreate').click();expect(page.locator('#captureComposer')).to_be_visible()
                page.locator('#captureComposer').get_by_role('button',name='Κλείσιμο',exact=True).click();page.locator('#visitBackHome').click()
                # A real create response arriving after close cannot reopen its chart.
                held=[];page.route('**/clinical/patients/create',lambda route:held.append(route))
                page.locator('#patientEntryHome').click();page.locator('#patientEntryName').fill('Καθυστερημένη Συνθετική Καρτέλα')
                page.locator('#patientEntryCreate').click();page.wait_for_function("document.getElementById('patientEntryCreate').disabled")
                page.locator('#patientEntry').get_by_role('button',name='Κλείσιμο',exact=True).click()
                response=held[0].fetch();held[0].fulfill(response=response);page.unroute('**/clinical/patients/create')
                expect(page.locator('#visitWorkspaceHome')).to_be_visible();expect(page.locator('#patientEntry')).not_to_be_visible()
                # A protected search response arriving after logout is ignored and hidden DOM cleared.
                searches=[];page.route('**/clinical/patients?query=*',lambda route:searches.append(route))
                page.locator('#patientEntryHome').click();page.locator('#patientEntryName').fill('Συνθετικός Είσοδος')
                page.wait_for_timeout(350);assert searches
                response=searches[0].fetch()
                page.request.post(base+'/clinical/logout');page.evaluate("window.dispatchEvent(new Event('clinical:unauthenticated'))")
                searches[0].fulfill(response=response);page.wait_for_timeout(50)
                expect(page.locator('#patientEntry')).not_to_be_visible();expect(page.locator('#patientEntryMatches')).to_be_empty();expect(page.locator('#patientEntryName')).to_have_value('')
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth');assert not errors,errors
                context.close()
            browser.close()
    finally:server.should_exit=True;thread.join(10);sock.close()
    with Session(engine) as db:
        for model in [EncounterORM,ClinicalPendingORM,BookingPatientLinkORM]:assert db.scalar(select(func.count()).select_from(model))==0
