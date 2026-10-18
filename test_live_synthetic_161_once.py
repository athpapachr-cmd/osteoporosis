"""Fixed marked synthetic only: resolved removal/restore plus two provider drafts, never Save."""
import os,sys,json
from uuid import uuid4
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
BASE='https://ortho-reception-backend.onrender.com';ROOT=BASE+'/static/cockpit/';ID='cockpit-cp1-test';stage='start'
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch();out=Path('live-161-proof');out.mkdir(exist_ok=True)
  for width in [1280,390]:
   context=browser.new_context(viewport={'width':width,'height':1000});page=context.new_page();page.set_default_timeout(20000)
   stage='normal-login';page.goto(ROOT);page.locator('#ceGlobalKey').click();page.locator('#ceGlobalKeyInput').fill(os.environ['CLINICAL_DATA_KEY'])
   page.locator('#ceGlobalLoginForm button[type=submit]').click();expect(page.locator('#ceGlobalAuthStatus')).to_contain_text('Συνδέθηκες');page.keyboard.press('Escape')
   def read():
    r=context.request.get(BASE+'/clinical/patient/'+ID+'/chart');assert r.status==200
    data=r.json();assert data['demographics']['synthetic_scenario']=='cockpit-cp1-v1' and data['patient_id']==ID
    return data
   stage='marked-resolved-baseline';data=read();visits=json.dumps(data['visits'],sort_keys=True)
   choices=[p for p in data['pending'] if {s['key'] for s in (p.get('checkpoints') or {}).get('steps',[])}=={'received','images_reviewed'}]
   assert len(choices)==1;item=choices[0];pid=item['pending_id']
   assert item['status']=='resolved' and set(item['checkpoints']['confirmed_keys'])=={'received','images_reviewed'} and item['checkpoint_write_allowed']
   def current():return next(p for p in read()['pending'] if p['pending_id']==pid)
   def link():return page.locator('.patient-chart-grid .patient-obligation-link button').filter(has_text=item['description'])
   stage='normal-chart';page.goto(ROOT+'?chart='+ID)
   assert 'Μην μετατρέπεις κενά ή μη καταγεγραμμένα πεδία' in page.locator('#visitDiaNaturalPrompt').evaluate('n=>n.content.textContent')
   popup=page.locator('.patient-pending-dialog');link().click();expect(popup).to_be_visible()
   expect(popup.locator('select')).to_have_count(0);expect(popup).to_contain_text('✓ Εξετάστηκαν οι εικόνες')
   expect(popup.get_by_role('button',name='Διαγραφή υποχρέωσης',exact=True)).to_be_visible()
   popup.screenshot(path=str(out/f'resolved-removal-{width}.png'));popup.get_by_role('button',name='Κλείσιμο',exact=True).click()
   branch=page.locator('.patient-followthrough-branch')
   stage='inline-resolved-keep';branch.get_by_role('button',name='Διαγραφή υποχρέωσης',exact=True).click()
   branch.get_by_role('button',name='Διατήρηση',exact=True).click()
   expect(branch.get_by_role('button',name='Διαγραφή υποχρέωσης',exact=True)).to_be_visible()
   stage='inline-resolved-delete';branch.get_by_role('button',name='Διαγραφή υποχρέωσης',exact=True).click()
   branch.get_by_role('button',name='Διαγραφή',exact=True).click();expect(branch).to_have_count(0)
   assert current()['status']=='cancelled'
   stage='persist-restore-exact-completion';page.reload();page.locator('.patient-deleted-obligations summary').click();link().click()
   popup.get_by_role('button',name='Επαναφορά υποχρέωσης',exact=True).click();expect(popup).to_have_count(0)
   restored=current();assert restored['status']=='resolved' and restored['checkpoints']['confirmed_keys']==item['checkpoints']['confirmed_keys']
   assert restored.get('resolution_event_id')==item.get('resolution_event_id')
   stage='persisted-approved-stages';page.reload();link().click();expect(popup).to_contain_text('✓ Παραλήφθηκε το CD');expect(popup).to_contain_text('✓ Εξετάστηκαν οι εικόνες')
   assert read()['visits']==data['visits'];assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   page.keyboard.press('Escape')
   print('PASS SYNTHETIC RESOLVED REMOVE/RESTORE:',width,'existing approved stages and source preserved')
   if width==390:
    cases=[('missing-field','Επανέλεγχος γόνατος. Αναφέρεται ήπιος πόνος στην κάμψη. Καταγραφή αρτηριακής πίεσης/σφυγμών (κενή έως τώρα). Δεν συμφωνήθηκε καμία επόμενη ενέργεια.',False),
      ('explicit-action','Επανέλεγχος γόνατος. Αναφέρεται ήπιος πόνος στην κάμψη. Συμφωνήθηκε να καταγραφεί η αρτηριακή πίεση και οι σφυγμοί στην επόμενη επίσκεψη.',True)]
    for label,source,wanted in cases:
     stage='provider-'+label
     r=context.request.post(BASE+'/clinical/visit-capture/context',data={'patient_id':ID,'client_session_id':'simple-obligations-'+uuid4().hex});assert r.status==200
     r=context.request.post(BASE+'/clinical/visit-capture/extract',data={'context_id':r.json()['context_id'],'text':source,'encounter_date':'2026-10-18'},timeout=110000)
     print('SAFE PROVIDER STATUS:',label,r.status,flush=True);assert r.status==200
     pending=r.json()['candidate']['pending']
     assert all(p['source_excerpt'] and p['source_excerpt'] in source for p in pending)
     if wanted:
      assert any('πίεσ' in p['description'] and 'σφυγμ' in p['description'] for p in pending)
     else:assert pending==[]
     print('PASS SYNTHETIC PROVIDER:',label,'source fidelity; no Save',flush=True)
    assert read()['visits']==data['visits']
    assert current()['status']=='resolved' and current()['checkpoints']['confirmed_keys']==item['checkpoints']['confirmed_keys']
   context.close()
  browser.close()
 print('PASS SIMPLE OBLIGATIONS LIVE: fixed marked synthetic only, resolved inline deletion/keep/persistence/exact-completion restore and approved stages at1280/390; missing BP creates no task, explicitly agreed BP preserved; no Save/new encounter/real-record writes.')
except Exception as error:
 print('FAIL SIMPLE OBLIGATIONS LIVE stage: '+stage+' type: '+type(error).__name__);sys.exit(1)
