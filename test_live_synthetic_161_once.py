"""Finite live CP1 proof, existing secret, synthetic data only, no error dumps."""
import os,sys
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
URL='https://ortho-reception-backend.onrender.com/static/cockpit/index.html'
TEXT='Συνθετική επίσκεψη επανελέγχου γόνατος. Ο ασθενής αναφέρει πόνο στην κάμψη. Κατά την εξέταση παρατηρήθηκε περιορισμός της κάμψης. Κλινική εκτίμηση: επίμονος πόνος γόνατος. Συμφωνήθηκε επανέλεγχος μετά την παραλαβή των προηγούμενων εξετάσεων. Ο ασθενής θα προσκομίσει το CD της μαγνητικής. Ο ασθενής θα στείλει το γραπτό πόρισμα. Δεν έχει ακόμη παραληφθεί κανένα από τα δύο. Δεν δόθηκε νέα αγωγή.'
REASON='Επανέλεγχος γόνατος — έλεγχος CP1'
stage='start'
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch();out=Path('live-161-proof');out.mkdir(exist_ok=True)
  for i in range(2):
   stage=f'browser-{i+1}-login'
   context=browser.new_context(viewport={'width':1280 if i==0 else 390,'height':1000})
   page=context.new_page();page.set_default_timeout(20000);page.goto(URL)
   page.locator('#ceGlobalKey').click();page.locator('#ceGlobalKeyInput').fill(os.environ['CLINICAL_DATA_KEY'])
   page.locator('#ceGlobalLoginForm button[type=submit]').click()
   expect(page.locator('#ceGlobalAuthStatus')).to_contain_text('Συνδέθηκες');page.keyboard.press('Escape')
   stage=f'browser-{i+1}-normal-test-chart'
   page.locator('#captureTestPatient').click();expect(page.locator('#captureOpen')).to_be_visible()
   if i==0:
    before=context.request.get('https://ortho-reception-backend.onrender.com/clinical/patient/cockpit-cp1-test/chart').json()
    prior=[v for v in before['visits'] if (v.get('clinical') or {}).get('reason_for_visit',{}).get('text')==REASON]
    if not prior:
     page.locator('#captureOpen').click();page.locator('#captureText').fill(TEXT)
     stage='real-provider-extraction'
     with page.expect_response('**/visit-capture/extract',timeout=120000) as reply:
      page.get_by_role('button',name='Προετοιμασία καταγραφής',exact=True).click()
     response=reply.value
     print('Extraction HTTP status:',response.status)
     assert response.status==200
     extracted=response.json()['candidate']
     print('Extraction identity:',extracted['source_identity_state'],'pending count:',len(extracted['pending']),'medication count:',len(extracted['medications']))
     assert extracted['source_identity_state']=='consistent' and len(extracted['pending'])>=2
     assert not extracted['medications']
     assert all(p['status'] in ('open','awaiting_patient','awaiting_result') and p['source_excerpt'] in TEXT for p in extracted['pending'])
     expect(page.locator('#captureSave')).to_be_visible()
     stage='edit-and-reject-proposal'
     reason=page.locator('#captureReview > fieldset').filter(has=page.locator('legend',has_text='Λόγος επίσκεψης'))
     reason.locator('textarea').fill(REASON)
     pending=page.locator('#captureReview > section').filter(has=page.get_by_role('heading',name='Προτεινόμενες υποχρεώσεις',exact=True))
     rows=pending.locator(':scope > .capture-field-group')
     # Reject the report proposal; retain only the CD obligation.
     removed=False
     for index in reversed(range(rows.count())):
      if 'CD' not in rows.nth(index).locator('textarea').first.input_value():
       rows.nth(index).get_by_role('button',name='Αφαίρεση').last.click();removed=True
     assert removed and rows.count()==1
     page.locator('#captureComposer').screenshot(path=str(out/'cp1-review.png'))
     stage='atomic-save'
     with page.expect_request('**/visit-capture/save') as sent:
      page.locator('#captureSave').click()
     body=sent.value.post_data_json
     expect(page.locator('#captureComposer')).to_be_hidden()
     assert context.request.post('https://ortho-reception-backend.onrender.com/clinical/visit-capture/save',data=body).status==409
    else:
     assert len(prior)==1
   stage=f'browser-{i+1}-reload-and-origin'
   page.reload();expect(page.locator('#patientChart')).to_contain_text(REASON)
   chart=context.request.get('https://ortho-reception-backend.onrender.com/clinical/patient/cockpit-cp1-test/chart').json()
   records=[v for v in chart['visits'] if (v.get('clinical') or {}).get('reason_for_visit',{}).get('text')==REASON]
   assert len(records)==1
   visit=records[0];items=[p for p in chart['pending'] if p['source_encounter_id']==visit['encounter_id']]
   assert len(items)==1 and 'CD' in items[0]['description'] and items[0]['source_excerpt'] in TEXT
   assert len(visit['clinical']['pending'])==1
   page.locator('#visit-'+visit['encounter_id']+' summary').click()
   page.locator('#visitWorkspacePatient').screenshot(path=str(out/f'cp1-chart-browser-{i+1}.png'))
   stage=f'browser-{i+1}-clinical-updates'
   page.locator('#doctorInboxButton').click();expect(page.locator('#doctorPopoverTitle')).to_have_text('Κλινικές ενημερώσεις')
   row=page.locator('#doctorPopoverContent .doctor-action-row').filter(has_text='CD').first
   expect(row).to_be_visible();page.locator('#doctorPopoverContent').screenshot(path=str(out/f'cp1-updates-browser-{i+1}.png'))
   row.click();expect(page.locator('#patientChart')).to_contain_text(REASON)
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   context.close()
  browser.close()
 print('PASS CP1: real provider, editable proposal, rejected obligation absent, atomic Save, duplicate refused, persistent chart/source and updates in two independent browsers')
except Exception:
 print('FAIL CP1 stage: '+stage);sys.exit(1)
