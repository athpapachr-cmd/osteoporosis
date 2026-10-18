"""Fixed marked chart edit/prefill/cancel proof, no clinical save or provider."""
import os,sys
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
BASE='https://ortho-reception-backend.onrender.com';ROOT=BASE+'/static/cockpit/';ID='cockpit-cp1-test';stage='start'
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch();out=Path('live-161-proof');out.mkdir(exist_ok=True)
  for width in [1280,390]:
   context=browser.new_context(viewport={'width':width,'height':1000});page=context.new_page();page.set_default_timeout(20000);forbidden=[];errors=[]
   page.on('request',lambda r:forbidden.append(True) if r.method=='POST' and any(s in r.url for s in ['/amend','/extract','/visit-capture/save']) else None)
   page.on('pageerror',lambda e:errors.append(type(e).__name__))
   stage='login';page.goto(ROOT);page.locator('#ceGlobalKey').click();page.locator('#ceGlobalKeyInput').fill(os.environ['CLINICAL_DATA_KEY'])
   page.locator('#ceGlobalLoginForm button[type=submit]').click();expect(page.locator('#ceGlobalAuthStatus')).to_contain_text('Συνδέθηκες');page.keyboard.press('Escape')
   def read():
    r=context.request.get(BASE+'/clinical/patient/'+ID+'/chart');assert r.status==200
    d=r.json();assert d['demographics']['synthetic_scenario']=='cockpit-cp1-v1' and d['patient_id']==ID;return d
   stage='fixed-marked-baseline';before=read();visit=before['visits'][0];eid=visit['encounter_id'];title=visit['clinical']['reason_for_visit']['text'];assert 'CP1' in title
   page.goto(ROOT+'?chart='+ID);page.locator('.patient-journey-node[data-encounter-id="'+eid+'"] .patient-journey-point').click()
   page.get_by_role('dialog').get_by_role('button',name='Άνοιγμα',exact=True).click()
   stage='edit-prefill';page.get_by_role('button',name='Επεξεργασία επίσκεψης',exact=True).click();composer=page.locator('#captureComposer')
   expect(composer).to_be_visible();expect(page.locator('#captureReview textarea').first).to_have_value(title)
   expect(page.locator('#captureDate')).to_have_value(visit['date'])
   expect(page.locator('#captureSave')).to_have_text('Αποθήκευση διορθώσεων')
   expect(page.locator('.patient-visit-reader')).not_to_be_visible()
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   composer.screenshot(path=str(out/f'visit-edit-prefill-{width}.png'))
   stage='cancel-unsaved-draft';page.locator('#captureReview textarea').first.fill('Δοκιμή ακύρωσης μη αποθηκευμένης διόρθωσης')
   composer.get_by_role('button',name='Κλείσιμο',exact=True).click()
   expect(composer).not_to_be_visible();expect(page.locator('.patient-visit:visible')).to_have_count(1)
   expect(page.locator('.patient-visit:visible')).to_contain_text(title)
   page.get_by_role('button',name='Επεξεργασία επίσκεψης',exact=True).click()
   expect(page.locator('#captureReview textarea').first).to_have_value(title)
   composer.get_by_role('button',name='Κλείσιμο',exact=True).click()
   stage='unchanged-chart';assert read()==before;assert not forbidden and not errors
   context.close();print('PASS VISIT AMENDMENT LIVE READ:',width,'prefilled existing visit/date; unsaved edit cancel/reopen; chart identical; no amendment/extraction/save request')
  browser.close()
 print('PASS VISIT AMENDMENT LIVE READ: fixed marked chart only1280/390. Actual amendment/task reconciliation/replay/as-of writes verified local/CI, never performed on production fixture.')
except Exception as error:
 print('FAIL VISIT AMENDMENT LIVE stage: '+stage+' type: '+type(error).__name__);sys.exit(1)
