"""Normal real appointment read/navigation proof; no real associations or chart writes."""
import os,sys
from pathlib import Path
from urllib.parse import urlencode
from playwright.sync_api import sync_playwright,expect
BASE='https://ortho-reception-backend.onrender.com';ROOT=BASE+'/static/cockpit/';stage='start'
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch();out=Path('live-161-proof');out.mkdir(exist_ok=True)
  for width in [1280,390]:
   context=browser.new_context(viewport={'width':width,'height':1000});page=context.new_page();page.set_default_timeout(20000)
   logins=[];writes=[]
   def track(request):
    if request.method=='POST' and request.url.endswith('/clinical/login'):logins.append(1)
    if any(x in request.url for x in ['/chart-select','/chart-unlink','/patients/create','/visit-capture/save','/visit-capture/extract']):writes.append(1)
   page.on('request',track)
   stage='single-login';page.goto(ROOT);page.locator('#ceGlobalKey').click();page.locator('#ceGlobalKeyInput').fill(os.environ['CLINICAL_DATA_KEY'])
   page.locator('#ceGlobalLoginForm button[type=submit]').click();expect(page.locator('#ceGlobalAuthStatus')).to_contain_text('Συνδέθηκες');page.keyboard.press('Escape')
   stage='actual-displayed-appointment'
   page.wait_for_function("['next','current','previous'].some(p=>document.getElementById(p+'AppointmentPatient').dataset.appointmentId)")
   prefix=page.evaluate("['next','current','previous'].find(p=>document.getElementById(p+'AppointmentPatient').dataset.appointmentId)")
   uid=page.locator('#'+prefix+'AppointmentPatient').get_attribute('data-appointment-id')
   day=page.locator('#'+prefix+'AppointmentPatient').get_attribute('data-appointment-date')
   button='#visitSidebar'+prefix[0].upper()+prefix[1:]
   stage='normal-chart-read'
   with page.expect_response(lambda r:r.url.endswith('/visit-brief/chart-read')) as pending:page.locator(button).click()
   response=pending.value
   print('SAFE LIVE NORMAL READ:',response.status)
   assert response.status==200
   data=response.json();assert data['mode'] in ('booking_only','chart_linked')
   expect(page.locator('#visitBriefOverlay')).to_be_visible();expect(page.locator('#ceGlobalAccessPanel')).not_to_be_visible()
   assert response.headers.get('cache-control')=='no-store'
   stage='same-session-selector-no-write'
   page.locator('#visitBriefPatientEntry').click();expect(page.locator('#patientEntry')).to_be_visible()
   # Remove all actual appointment/person context before screenshots.
   page.locator('#patientEntryName').fill('');expect(page.locator('#patientEntryMatches')).to_be_empty()
   page.locator('#patientEntry').screenshot(path=str(out/f'appointment-selector-{width}.png'))
   page.locator('#patientEntry').get_by_role('button',name='Κλείσιμο',exact=True).click()
   stage='reopen-without-second-login'
   with page.expect_response(lambda r:r.url.endswith('/visit-brief/chart-read')) as pending:page.locator(button).click()
   assert pending.value.status==200
   page.locator('#visitBriefClose').click()
   stage='weekly-calendar-entry-same-session'
   page.goto(ROOT+'?'+urlencode({'booking':uid,'booking_date':day}))
   expect(page.locator('#visitBriefOverlay')).to_be_visible();expect(page.locator('#visitBriefPatientEntry')).to_be_visible()
   expect(page.locator('#visitBriefIdentity')).not_to_contain_text('Συνδέσου')
   # Ensure source completed rather than accepting the temporary loading label.
   expect(page.locator('#visitBriefIdentity')).to_contain_text('Συνδεδεμένη καρτέλα' if data['mode']=='chart_linked' else 'Επίλεξε την καρτέλα μία φορά')
   page.locator('#visitBriefClose').click()
   assert context.request.get(BASE+'/clinical/status').status==200
   assert len(logins)==1 and not writes
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   context.close()
  browser.close()
 print('PASS APPOINTMENT CHART LIVE: one login per independent1280/390 session; actual displayed appointment normal clinical-key read200/no-store; selector/reopen/weekly-link entry without second login. No actual names/history captured or logged, no real link/patient/visit/provider writes. Persistent selection/latest-visit behavior proven locally+CI, not claimed as a real production association write.')
except Exception as error:
 print('FAIL APPOINTMENT LIVE stage: '+stage+' type: '+type(error).__name__);sys.exit(1)
