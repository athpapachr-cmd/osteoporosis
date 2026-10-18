"""Read-only proof of focused stored visit navigation on the fixed marked chart."""
import os,sys
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
BASE='https://ortho-reception-backend.onrender.com';ROOT=BASE+'/static/cockpit/';ID='cockpit-cp1-test';stage='start'
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch();out=Path('live-161-proof');out.mkdir(exist_ok=True)
  for width in [1280,390]:
   context=browser.new_context(viewport={'width':width,'height':1000});page=context.new_page();page.set_default_timeout(20000);errors=[]
   page.on('pageerror',lambda e:errors.append(type(e).__name__))
   stage='login';page.goto(ROOT);page.locator('#ceGlobalKey').click();page.locator('#ceGlobalKeyInput').fill(os.environ['CLINICAL_DATA_KEY'])
   page.locator('#ceGlobalLoginForm button[type=submit]').click();expect(page.locator('#ceGlobalAuthStatus')).to_contain_text('Συνδέθηκες');page.keyboard.press('Escape')
   def read():
    r=context.request.get(BASE+'/clinical/patient/'+ID+'/chart');assert r.status==200
    d=r.json();assert d['demographics']['synthetic_scenario']=='cockpit-cp1-v1' and d['patient_id']==ID;return d
   stage='fixed-marked-baseline';before=read();visit=before['visits'][0];eid=visit['encounter_id'];title=visit['clinical']['reason_for_visit']['text']
   assert 'CP1' in title
   item=next(p for p in before['pending'] if p['source_encounter_id']==eid);pid=item['pending_id'];page.goto(ROOT+'?chart='+ID)
   def focused():
    expect(page.locator('.patient-chart-overview')).not_to_be_visible();expect(page.locator('.patient-visit-reader')).to_be_visible()
    expect(page.locator('.patient-visit:visible')).to_have_count(1);expect(page.locator('#visit-'+eid)).to_be_visible()
    expect(page.locator('#visit-'+eid+' > summary')).to_contain_text(title);expect(page.locator('#patientVisitBack')).to_be_visible()
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   def back():
    page.locator('#patientVisitBack').click();expect(page.locator('.patient-chart-overview')).to_be_visible()
    expect(page.locator('.patient-visit:visible')).to_have_count(0);expect(page.locator('.patient-timeline')).to_be_visible()
   stage='timeline-reader'
   page.locator('.patient-journey-node[data-encounter-id="'+eid+'"] .patient-journey-point').click()
   page.get_by_role('dialog').get_by_role('button',name='Άνοιγμα',exact=True).click();focused()
   page.locator('.patient-visit-reader').screenshot(path=str(out/f'focused-visit-{width}.png'));back()
   stage='obligation-source-reader';page.locator('.patient-followthrough-branch article[data-pending-id="'+pid+'"] .pending-open').first.click()
   page.get_by_role('dialog').get_by_role('button',name='Δημιουργήθηκε στην επίσκεψη',exact=False).click()
   expect(page.get_by_role('dialog')).not_to_be_visible();focused();back()
   stage='shared-public-reader';assert page.evaluate('async args=>await CockpitStoredChart.openVisit(...args)',[ID,eid]) is True
   focused();back()
   stage='unchanged-chart';assert read()==before;assert not errors
   context.close();print('PASS FOCUSED LIVE READ:',width,'one selected marked visit; overview hidden; timeline/source/shared caller; Back; unchanged chart')
  browser.close()
 print('PASS FOCUSED VISIT LIVE READ: fixed marked chart only1280/390; no clinical/task/provider write. Two-date selection and actual appointment caller verified local/CI.')
except Exception as error:
 print('FAIL FOCUSED LIVE stage: '+stage+' type: '+type(error).__name__);sys.exit(1)
