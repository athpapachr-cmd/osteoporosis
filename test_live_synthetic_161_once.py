"""Finite live CP2 lifecycle proof. Fixed synthetic patient only, existing secret."""
import os,sys
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
BASE='https://ortho-reception-backend.onrender.com'
URL=BASE+'/static/cockpit/index.html'
ID='cockpit-cp1-test';REASON='Επανέλεγχος γόνατος — έλεγχος CP1'
stage='start'
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch();out=Path('live-161-proof');out.mkdir(exist_ok=True)
  contexts=[];pages=[]
  def chart(context):
   r=context.request.get(BASE+'/clinical/patient/'+ID+'/chart');assert r.status==200
   data=r.json();assert data['patient_id']==ID and data['demographics']['synthetic_scenario']=='cockpit-cp1-v1'
   return data
  def item(context):
   data=chart(context);rows=[v for v in data['visits'] if (v.get('clinical') or {}).get('reason_for_visit',{}).get('text')==REASON]
   assert len(rows)==1
   items=[p for p in data['pending'] if p['source_encounter_id']==rows[0]['encounter_id']]
   assert len(items)==1 and 'CD' in items[0]['description'];assert items[0]['checkpoint_storage_ready'] and items[0]['checkpoint_write_allowed']
   return items[0]
  def login(width):
   context=browser.new_context(viewport={'width':width,'height':1000});page=context.new_page();page.set_default_timeout(20000)
   page.goto(URL);page.locator('#ceGlobalKey').click();page.locator('#ceGlobalKeyInput').fill(os.environ['CLINICAL_DATA_KEY'])
   page.locator('#ceGlobalLoginForm button[type=submit]').click();expect(page.locator('#ceGlobalAuthStatus')).to_contain_text('Συνδέθηκες');page.keyboard.press('Escape')
   page.locator('#captureTestPatient').click();expect(page.locator('#captureOpen')).to_be_visible()
   contexts.append(context);pages.append(page);return context,page
  stage='desktop-login';desktop,page=login(1280)
  initial=chart(desktop);signed_before=initial['visits'];row=page.locator('#patientChart .patient-followthrough-branch .synthetic-pending-row').filter(has_text='CD').first
  p=item(desktop)
  if not p.get('checkpoints'):
   stage='plan-and-explicit-notyet'
   expect(row).to_contain_text('Παραλήφθηκε το CD;')
   with page.expect_request('**/checkpoint-plan') as sent:row.get_by_role('button',name='Όχι ακόμη',exact=True).first.click()
   expect(row).to_contain_text('Όχι ακόμη · καταγράφηκε')
   assert desktop.request.post(sent.value.url,data=sent.value.post_data_json).json()['replayed'] is True
   p=item(desktop);assert p['checkpoints']['next_checkpoint']['state']=='not_yet'
  stage='receipt-confirmation'
  p=item(desktop)
  if 'received' not in p['checkpoints']['confirmed_keys']:
   row.locator('.pending-stage').filter(has_text='Παραλήφθηκε το CD;').get_by_role('button',name='Ναι',exact=True).click()
  expect(row).to_contain_text('✓ Παραλήφθηκε το CD')
  p=item(desktop);assert 'received' in p['checkpoints']['confirmed_keys']
  if 'images_reviewed' in p['checkpoints']['confirmed_keys']:
   row.locator('.pending-stage').filter(has_text='✓ Εξετάστηκαν οι εικόνες').get_by_role('button',name='Αναίρεση',exact=True).click()
  expect(row).to_contain_text('Εξετάστηκαν οι εικόνες;')
  stage='review-notyet'
  row.locator('.pending-stage').filter(has_text='Εξετάστηκαν οι εικόνες;').get_by_role('button',name='Όχι ακόμη',exact=True).click();expect(row).to_contain_text('Όχι ακόμη · καταγράφηκε')
  assert item(desktop)['status']!='resolved'
  stage='visit-hierarchy-and-finite-branch'
  expect(page.locator('.patient-journey-point')).to_have_count(len(initial['visits']))
  assert row.locator('xpath=ancestor::details[1]').get_attribute('id')=='followthrough-'+p['source_encounter_id']
  expect(row.locator('.patient-followthrough-rail .pending-stage')).to_have_count(2)
  expect(row.locator('.patient-followthrough-end')).to_have_attribute('data-complete','false')
  stage='connected-source-geometry'
  expect(page.locator('.journey-fork')).to_have_count(1)
  def connected(pg):
   return pg.locator('.patient-journey-track').evaluate('''track=>{
    const dot=track.querySelector('.patient-journey-dot').getBoundingClientRect(),box=track.getBoundingClientRect();
    const line=track.querySelector('.journey-fork'),stop=track.querySelector('.patient-followthrough-stop').getBoundingClientRect();
    const start=line.getPointAtLength(0),end=line.getPointAtLength(line.getTotalLength());
    return Math.abs(start.x-(dot.x-box.x+dot.width/2))<1 && Math.abs(start.y-(dot.y-box.y+dot.height/2))<1 && Math.abs(end.x-(stop.x-box.x+stop.width/2))<1 && Math.abs(end.y-(stop.y-box.y+stop.height/2))<1 && stop.y>dot.y;
   }''')
  assert connected(page)
  page.locator('.patient-timeline').screenshot(path=str(out/'connected-journey-desktop.png'))
  stage='desktop-timeline-popup'
  page.locator('.patient-journey-point').filter(has_text=REASON).click();expect(page.get_by_role('dialog')).to_be_visible()
  expect(page.get_by_role('dialog')).to_contain_text(REASON)
  page.get_by_role('dialog').screenshot(path=str(out/'cp2-popup-desktop.png'))
  page.get_by_role('button',name='Άνοιγμα',exact=True).click();expect(page.locator('#visit-'+p['source_encounter_id'])).to_be_visible()
  stage='mobile-independent-read'
  mobile,mp=login(390);mr=mp.locator('#patientChart .patient-followthrough-branch .synthetic-pending-row').filter(has_text='CD').first
  expect(mr).to_contain_text('✓ Παραλήφθηκε το CD');expect(mr).to_contain_text('Όχι ακόμη · καταγράφηκε')
  mp.locator('.patient-journey-point').filter(has_text=REASON).click();expect(mp.get_by_role('dialog')).to_be_visible()
  mp.get_by_role('dialog').screenshot(path=str(out/'cp2-popup-mobile.png'))
  mp.get_by_role('button',name='Άνοιγμα',exact=True).click()
  stage='review-confirmation-and-replay'
  with mp.expect_request('**/checkpoint') as sent:mr.get_by_role('button',name='Ναι',exact=True).click()
  expect(mr).to_contain_text('Ολοκληρώθηκε');expect(mr.locator('.patient-followthrough-end')).to_have_attribute('data-complete','true');state=item(mobile);assert state['status']=='resolved' and state['checkpoints']['completed']
  assert mobile.request.post(sent.value.url,data=sent.value.post_data_json).json()['replayed'] is True
  mp.locator('#visitWorkspacePatient').screenshot(path=str(out/'cp2-completed-mobile.png'))
  stage='exact-review-undo'
  mr.locator('.pending-stage').filter(has_text='✓ Εξετάστηκαν οι εικόνες').get_by_role('button',name='Αναίρεση',exact=True).click()
  expect(mr).to_contain_text('Εξετάστηκαν οι εικόνες;');expect(mr.locator('.patient-followthrough-end')).to_have_attribute('data-complete','false');state=item(mobile)
  assert state['status']=='open' and state['checkpoints']['confirmed_keys']==['received']
  stage='refresh-second-browser-and-source'
  for context,pg,width in [(desktop,page,1280),(mobile,mp,390)]:
   pg.reload();r=pg.locator('#patientChart .patient-followthrough-branch .synthetic-pending-row').filter(has_text='CD').first
   expect(r).to_contain_text('✓ Παραλήφθηκε το CD');expect(r).to_contain_text('Εξετάστηκαν οι εικόνες;')
   data=chart(context);assert data['visits']==signed_before
   updates=context.request.get(BASE+'/clinical/clinical-updates?scope=test').json()['items']
   linked=[i for i in updates if i['pending_id']==state['pending_id']];assert len(linked)==1
   assert linked[0]['checkpoints']==item(context)['checkpoints']
   expect(pg.locator('.journey-fork')).to_have_count(1);assert connected(pg)
   pg.locator('.patient-journey-rail').evaluate('n=>n.scrollLeft=0')
   pg.locator('.patient-timeline').screenshot(path=str(out/f'connected-journey-{width}.png'))
   pg.locator('#visitWorkspacePatient').screenshot(path=str(out/f'cp2-final-{width}.png'))
   pg.locator('#doctorInboxButton').click();expect(pg.locator('#doctorPopoverTitle')).to_have_text('Κλινικές ενημερώσεις')
   updates_row=pg.locator('#doctorPopoverContent .doctor-action-row').filter(has_text='CD').first
   expect(updates_row).to_contain_text('Εξετάστηκαν οι εικόνες;');pg.locator('#doctorPopoverContent').screenshot(path=str(out/f'cp2-updates-{width}.png'))
   updates_row.click();expect(pg.locator('#patientChart')).to_contain_text(REASON)
   assert pg.evaluate('document.documentElement.scrollWidth<=innerWidth')
  for context in contexts:context.close()
  browser.close()
 print('PASS CP2 LIVE: measured source-dot fork and finite terminal in one connected drawing; visit-only main trajectory; exact source branch and finite terminal; marked synthetic patient; PostgreSQL receipt/notyet/review/undo; idempotent replay; immutable signed visit; popup/open-source; same chart/updates in independent1280/390 sessions after refresh. Final fixture: received confirmed, image review pending. No provider request or new encounter.')
except Exception:
 print('FAIL CP2 LIVE stage: '+stage);sys.exit(1)
