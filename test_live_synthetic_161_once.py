"""Read-only actual source proof: no names/history logged, no patient/task/provider writes."""
import os,sys
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
BASE='https://ortho-reception-backend.onrender.com';ROOT=BASE+'/static/cockpit/';stage='start'
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch();out=Path('live-161-proof');out.mkdir(exist_ok=True)
  for width in [1280,390]:
   context=browser.new_context(viewport={'width':width,'height':1000});page=context.new_page();page.set_default_timeout(20000)
   stage='normal-login';page.goto(ROOT);page.locator('#ceGlobalKey').click();page.locator('#ceGlobalKeyInput').fill(os.environ['CLINICAL_DATA_KEY'])
   page.locator('#ceGlobalLoginForm button[type=submit]').click();expect(page.locator('#ceGlobalAuthStatus')).to_contain_text('Συνδέθηκες');page.keyboard.press('Escape')
   stage='unique-recent-projection';r=context.request.get(BASE+'/clinical/recent-encounters?limit=3');assert r.status==200
   rows=r.json();assert len(rows)<=3 and len({x['patient_id'] for x in rows})==len(rows)
   expect(page.locator('#visitRecentRows .clinical-patient-row')).to_have_count(len(rows))
   expect(page.locator('#visitSidebar h2')).to_have_text('Πρόσφατοι ασθενείς')
   stage='independent-source-inbox';reads=[];page.on('request',lambda r:reads.append(r.url) if '/clinical-updates' in r.url else None)
   page.locator('#doctorInboxButton').click();expect(page.locator('#doctorPopoverTitle')).to_have_text('Κλινικά εισερχόμενα')
   expect(page.locator('#doctorPopoverContent')).to_contain_text('Δεν έχει συνδεθεί')
   expect(page.locator('#doctorPopoverContent button.doctor-action-row')).to_have_count(0);assert not reads
   page.locator('#doctorPopover').screenshot(path=str(out/f'incoming-source-boundary-{width}.png'))
   stage='grouped-open-obligations';r=context.request.get(BASE+'/clinical/clinical-updates?active_only=true');assert r.status==200
   items=r.json()['items'];assert all(x['status'] in ('open','awaiting_patient','awaiting_result') for x in items)
   page.locator('#doctorTasksButton').click();expect(page.locator('#doctorPopoverTitle')).to_have_text('Υποχρεώσεις ασθενών')
   groups=page.locator('.patient-updates-group');expect(groups).to_have_count(len({x['patient_id'] for x in items}))
   expect(groups.locator('button.doctor-action-row')).to_have_count(len(items))
   actual=groups.locator('button').evaluate_all('ns=>ns.map(n=>n.dataset.pendingId)')
   assert set(actual)=={x['pending_id'] for x in items} and len(actual)==len(set(actual))
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   context.close();print('PASS READ-ONLY ATTENTION BOUNDARIES:',width,'unique recent identities, independent Inbox, grouped active Pending IDs; no clinical writes')
  browser.close()
 print('PASS ATTENTION SEPARATION LIVE: actual protected reads only1280/390; no real names/history logged or captured; no patient/task/provider writes.')
except Exception as error:
 print('FAIL ATTENTION SEPARATION LIVE stage: '+stage+' type: '+type(error).__name__);sys.exit(1)
