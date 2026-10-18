"""Fixed marked chart read-only: compact branches and available controls; no destructive writes."""
import os,sys,json
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
    d=r.json();assert d['demographics']['synthetic_scenario']=='cockpit-cp1-v1' and d['patient_id']==ID;return d
   stage='fixed-marked-baseline';before=read();item=next(p for p in before['pending'] if set((p.get('checkpoints') or {}).get('confirmed_keys',[]))=={'received','images_reviewed'})
   assert item['status']=='resolved';pid=item['pending_id'];page.goto(ROOT+'?chart='+ID)
   stage='compact-inline';row=page.locator('.patient-followthrough-branch article[data-pending-id="'+pid+'"]')
   expect(row).to_be_visible();expect(row.locator('blockquote')).to_have_count(0)
   expect(row.get_by_role('button',name='Δημιουργήθηκε στην επίσκεψη',exact=False)).to_have_count(0)
   last=row.locator('.pending-stage').last;undo=last.get_by_role('button',name='Αναίρεση',exact=True).bounding_box();remove=last.get_by_role('button',name='Διαγραφή υποχρέωσης',exact=True).bounding_box()
   print('SAFE MARKED CONTROL GEOMETRY:',width,{'undo':undo,'remove':remove},flush=True)
   assert abs((undo['y']+undo['height']/2)-(remove['y']+remove['height']/2))<1 and remove['x']>undo['x']
   page.locator('.patient-timeline').screenshot(path=str(out/f'compact-timeline-{width}.png'))
   stage='source-retained-in-popup';row.locator('.pending-open').first.click();popup=page.locator('.patient-pending-dialog')
   expect(popup.locator('blockquote')).to_have_count(1)
   expect(popup.get_by_role('button',name='Δημιουργήθηκε στην επίσκεψη',exact=False)).to_be_visible()
   expect(popup.get_by_role('button',name='Μόνιμη διαγραφή',exact=True)).to_have_count(0)
   expect(popup).to_contain_text('✓ Παραλήφθηκε το CD');expect(popup).to_contain_text('✓ Εξετάστηκαν οι εικόνες')
   popup.screenshot(path=str(out/f'compact-popup-{width}.png'));page.keyboard.press('Escape')
   stage='read-only-persistence';assert read()==before;assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   context.close();print('PASS COMPACT LIVE READ:',width,'source omitted inline/retained popup; undo/delete same row; fixed completed stages unchanged')
  browser.close()
 print('PASS COMPACT OBLIGATIONS LIVE READ: fixed marked chart only1280/390; no patient/task/provider mutation. Permanent removal verified local/CI, not executed on production fixture.')
except Exception as error:
 print('FAIL COMPACT LIVE stage: '+stage+' type: '+type(error).__name__);sys.exit(1)
