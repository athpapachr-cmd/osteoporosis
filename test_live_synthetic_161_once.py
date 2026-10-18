"""Read/copy proof only, existing marked synthetic chart and static instruction."""
import os,sys
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
BASE='https://ortho-reception-backend.onrender.com';ROOT=BASE+'/static/cockpit/';ID='cockpit-cp1-test';stage='start'
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch();out=Path('live-161-proof');out.mkdir(exist_ok=True)
  for width in [1280,390]:
   context=browser.new_context(viewport={'width':width,'height':1000},permissions=['clipboard-read','clipboard-write']);page=context.new_page();page.set_default_timeout(20000)
   stage='login';page.goto(ROOT);page.locator('#ceGlobalKey').click();page.locator('#ceGlobalKeyInput').fill(os.environ['CLINICAL_DATA_KEY'])
   page.locator('#ceGlobalLoginForm button[type=submit]').click();expect(page.locator('#ceGlobalAuthStatus')).to_contain_text('Συνδέθηκες');page.keyboard.press('Escape')
   data=context.request.get(BASE+'/clinical/patient/'+ID+'/chart').json();assert data['demographics']['synthetic_scenario']=='cockpit-cp1-v1'
   stage='normal-composer-copy';page.goto(ROOT+'visit-capture/?patient_id='+ID);expect(page.locator('#captureComposer')).to_be_visible()
   page.locator('#captureCopyDiaPrompt').click();expect(page.locator('#captureDiaPromptStatus')).to_contain_text('Αντιγράφηκε')
   copied=page.evaluate('navigator.clipboard.readText()');display=page.locator('#captureDiaPromptText').text_content()
   assert copied==display and 'ενεργής επίσκεψης' in copied and data['demographics']['full_name'] not in copied and ID not in copied
   stage='readable-instruction-and-draft-preserved'
   page.locator('#captureText').fill('Συνθετικός έλεγχος διατήρησης πρόχειρου, χωρίς αποθήκευση.')
   page.locator('#captureCopyDiaPrompt').click();expect(page.locator('#captureText')).to_have_value('Συνθετικός έλεγχος διατήρησης πρόχειρου, χωρίς αποθήκευση.')
   page.locator('.capture-dia-prompt summary').click();expect(page.locator('#captureDiaPromptText')).to_be_visible()
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   page.locator('.capture-dia-prompt').screenshot(path=str(out/f'dia-prompt-{width}.png'))
   page.locator('#captureComposer').get_by_role('button',name='Κλείσιμο',exact=True).click();context.close()
  browser.close()
 print('PASS DIA PROMPT LIVE: normal natural composer; actual Clipboard write/read equals displayed static instruction; no patient identifiers; draft preserved; expandable manual text; independent1280/390/no overflow. No clinical/provider/patient writes.')
except Exception:
 print('FAIL DIA PROMPT LIVE stage: '+stage);sys.exit(1)
