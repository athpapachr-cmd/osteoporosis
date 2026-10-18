"""Bounded normal-entry live read/navigation proof. No patient/encounter creation."""
import os,sys
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
BASE='https://ortho-reception-backend.onrender.com';ROOT=BASE+'/static/cockpit/';ID='cockpit-cp1-test';stage='start'
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch();out=Path('live-161-proof');out.mkdir(exist_ok=True);appointments=0
  for width in [1280,390]:
   context=browser.new_context(viewport={'width':width,'height':1000});page=context.new_page();page.set_default_timeout(20000)
   stage='normal-root-login';page.goto(ROOT);page.locator('#ceGlobalKey').click();page.locator('#ceGlobalKeyInput').fill(os.environ['CLINICAL_DATA_KEY'])
   page.locator('#ceGlobalLoginForm button[type=submit]').click();expect(page.locator('#ceGlobalAuthStatus')).to_contain_text('Συνδέθηκες');page.keyboard.press('Escape')
   stage='create-adapter-present-private-validation'
   r=context.request.post(BASE+'/clinical/patients/create',data={'creation_id':'invalid-proof-no-write','full_name':'Συνθετικός έλεγχος χωρίς εγγραφή'})
   assert r.status==422 and 'Συνθετικός' not in r.text()
   stage='home-new-chart-form'
   page.locator('#patientEntryHome').click();page.locator('#patientEntryName').fill('Συνθετικός έλεγχος εισόδου χωρίς εγγραφή')
   expect(page.locator('#patientEntryMatches')).to_contain_text('Δεν βρέθηκε καρτέλα')
   expect(page.locator('#patientEntryCreate')).to_be_enabled();page.locator('#patientEntry').screenshot(path=str(out/f'patient-entry-home-{width}.png'))
   page.locator('#patientEntry').get_by_role('button',name='Κλείσιμο',exact=True).click()
   stage='real-displayed-appointment-manual-draft'
   page.wait_for_function("document.getElementById('nextAppointmentPatient').dataset.appointmentId || document.getElementById('calendarNote').textContent.length>20")
   if page.locator('#nextAppointmentPatient').get_attribute('data-appointment-id'):
    displayed=page.locator('#nextAppointmentPatient').inner_text();page.locator('#visitSidebarNext').click()
    expect(page.locator('#visitBriefPatientEntry')).to_be_visible();page.locator('#visitBriefPatientEntry').click()
    expect(page.locator('#patientEntryName')).to_have_value(displayed)
    # Do not save screenshots, names or registry results from real appointments.
    page.locator('#patientEntry').get_by_role('button',name='Κλείσιμο',exact=True).click();displayed='';appointments+=1
   stage='legacy-capture-existing-marked-chart'
   data=context.request.get(BASE+'/clinical/patient/'+ID+'/chart').json();assert data['demographics']['synthetic_scenario']=='cockpit-cp1-v1'
   page.goto(ROOT+'visit-capture/?patient_id='+ID)
   expect(page.locator('#captureComposer')).to_be_visible();assert '/visit-capture/' not in page.url
   assert page.locator('#candidateInput').count()==0
   expect(page.locator('#captureText')).to_be_visible();page.locator('#captureComposer').screenshot(path=str(out/f'patient-entry-natural-{width}.png'))
   page.locator('#captureComposer').get_by_role('button',name='Κλείσιμο',exact=True).click()
   stage='legacy-capture-new-selector'
   page.goto(ROOT+'visit-capture/');expect(page.locator('#patientEntry')).to_be_visible()
   expect(page.locator('#patientEntryCreate')).to_have_text('Δημιουργία και καταγραφή επίσκεψης')
   page.locator('#patientEntryName').fill('Συνθετικός έλεγχος εισόδου χωρίς εγγραφή')
   expect(page.locator('#patientEntryMatches')).to_contain_text('Δεν βρέθηκε καρτέλα')
   page.locator('#patientEntry').screenshot(path=str(out/f'patient-entry-capture-{width}.png'))
   page.locator('#patientEntry').get_by_role('button',name='Κλείσιμο',exact=True).click()
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   context.close()
  browser.close()
 print('PASS PATIENT ENTRY LIVE: normal root Home create form, strict no-write422 validation, displayed-appointment manual prefill in '+str(appointments)+' sessions, legacy capture existing marked chart and new selector, natural composer/no JSON, independent1280/390. No patient/encounter/Pending/link/provider writes; no real appointment data captured.')
except Exception:
 print('FAIL PATIENT ENTRY LIVE stage: '+stage);sys.exit(1)
