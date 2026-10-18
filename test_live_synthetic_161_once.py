"""Normal clinical-key authority, fixed marked synthetic chart only, no provider/new visit."""
import os,sys,json
from uuid import uuid4
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
BASE='https://ortho-reception-backend.onrender.com';ROOT=BASE+'/static/cockpit/';ID='cockpit-cp1-test';stage='start'
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch();out=Path('live-161-proof');out.mkdir(exist_ok=True)
  baseline_visits=None
  for width in [1280,390]:
   context=browser.new_context(viewport={'width':width,'height':1000});page=context.new_page();page.set_default_timeout(20000)
   stage='login';page.goto(ROOT);page.locator('#ceGlobalKey').click();page.locator('#ceGlobalKeyInput').fill(os.environ['CLINICAL_DATA_KEY'])
   page.locator('#ceGlobalLoginForm button[type=submit]').click();expect(page.locator('#ceGlobalAuthStatus')).to_contain_text('Συνδέθηκες');page.keyboard.press('Escape')
   def read():
    r=context.request.get(BASE+'/clinical/patient/'+ID+'/chart');assert r.status==200
    data=r.json();assert data['demographics']['synthetic_scenario']=='cockpit-cp1-v1';assert data['patient_id']==ID
    return data
   stage='marked-baseline'
   data=read();baseline_visits=baseline_visits or json.dumps(data['visits'],sort_keys=True)
   choices=[p for p in data['pending'] if {s['key'] for s in (p.get('checkpoints') or {}).get('steps',[])}=={'received','images_reviewed'}]
   print('SAFE FIXED MARKED BASELINE:',[{'status':p['status'],'write':p['checkpoint_write_allowed'],'confirmed':(p.get('checkpoints') or {}).get('confirmed_keys'),'planned':p.get('checkpoints') is not None} for p in data['pending']])
   assert len(choices)==1
   item=choices[0]
   baseline_status=item['status'];baseline_keys=set(item['checkpoints']['confirmed_keys'])
   assert item['checkpoint_write_allowed'] and baseline_status in ('open','resolved')
   assert baseline_keys in ({'received'},{'received','images_reviewed'})
   pid=item['pending_id'];path=BASE+'/clinical/patient/'+ID+'/pending/'+pid
   def current():return next(p for p in read()['pending'] if p['pending_id']==pid)
   def link():return page.locator('.patient-chart-grid .patient-obligation-link button').filter(has_text=item['description'])
   popup=page.locator('.patient-pending-dialog')
   stage='normal-popup';page.goto(ROOT+'?chart='+ID);link().click();expect(popup).to_be_visible()
   if 'images_reviewed' in baseline_keys:
    stage='prepare-synthetic-baseline'
    popup.locator('.pending-stage').filter(has_text='✓ Εξετάστηκαν οι εικόνες').get_by_role('button',name='Αναίρεση',exact=True).click();expect(popup).to_have_count(0);link().click()
   expect(popup.get_by_role('button',name='Ναι',exact=True)).to_be_enabled()
   stage='notyet';popup.get_by_role('button',name='Όχι ακόμη',exact=True).click();expect(popup).to_have_count(0)
   link().click();expect(popup).to_contain_text('Όχι ακόμη')
   stage='complete';popup.get_by_role('button',name='Ναι',exact=True).click();expect(popup).to_have_count(0)
   assert current()['status']=='resolved';page.reload();link().click();expect(popup).to_contain_text('✓ Εξετάστηκαν οι εικόνες')
   stage='undo-image-only';popup.locator('.pending-stage').filter(has_text='✓ Εξετάστηκαν οι εικόνες').get_by_role('button',name='Αναίρεση',exact=True).click();expect(popup).to_have_count(0)
   assert current()['checkpoints']['confirmed_keys']==['received']
   stage='delete-cancel-keep';link().click();popup.get_by_role('button',name='Διαγραφή υποχρέωσης',exact=True).click();popup.get_by_role('button',name='Διατήρηση',exact=True).click();expect(popup).to_be_visible()
   stage='delete';popup.get_by_role('button',name='Διαγραφή υποχρέωσης',exact=True).click();popup.get_by_role('button',name='Διαγραφή',exact=True).click();expect(popup).to_have_count(0)
   expect(page.locator('.patient-followthrough-branch')).to_have_count(0);assert current()['status']=='cancelled'
   stage='persisted-deletion-and-restore';page.reload();page.locator('.patient-deleted-obligations summary').click();link().click()
   expect(popup.get_by_role('button',name='Ναι',exact=True)).to_have_count(0)
   popup.get_by_role('button',name='Επαναφορά υποχρέωσης',exact=True).click();expect(popup).to_have_count(0)
   assert current()['status']=='open' and current()['checkpoints']['confirmed_keys']==['received']
   stage='branch-popup';page.locator('.patient-followthrough-branch .pending-open').first.click();expect(popup).to_be_visible()
   assert popup.bounding_box()['width']<=width;popup.screenshot(path=str(out/f'obligation-actions-{width}.png'));popup.get_by_role('button',name='Κλείσιμο',exact=True).click()
   stage='updates-popup';page.locator('#doctorInboxButton').click()
   page.locator('#doctorPopoverContent .doctor-action-row').filter(has_text=item['description']).click();expect(popup).to_be_visible()
   if 'images_reviewed' in baseline_keys:
    stage='restore-original-completed-baseline'
    popup.get_by_role('button',name='Ναι',exact=True).click();expect(popup).to_have_count(0)
   else:page.keyboard.press('Escape')
   stage='independent-persistence-source';page.reload();expect(page.locator('.patient-followthrough-branch')).to_be_visible()
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   data=read();assert json.dumps(data['visits'],sort_keys=True)==baseline_visits
   assert set(current()['checkpoints']['confirmed_keys'])==baseline_keys and current()['status']==baseline_status
   context.close()
  browser.close()
 print('PASS REAL OBLIGATION OWNER LIVE: same normal clinical-key authority; popup via overview/branch/Updates; notyet/completion/undo, deletion/keep/persisted restore, independent1280/390, immutable signed source. Fixed marked synthetic only; no provider/new visit/real-patient writes. Actual original fixture completion state preserved.')
except Exception as error:
 print('FAIL OBLIGATION LIVE stage: '+stage+' type: '+type(error).__name__);sys.exit(1)
