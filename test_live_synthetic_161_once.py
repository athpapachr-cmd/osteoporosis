"""Actual dated source counts only; safe navigation screenshots, no clinical writes."""
import os,sys
from pathlib import Path
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
from playwright.sync_api import sync_playwright,expect
BASE='https://ortho-reception-backend.onrender.com';ROOT=BASE+'/static/cockpit/';stage='start'
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch();out=Path('live-161-proof');out.mkdir(exist_ok=True)
  for width in [1280,390]:
   context=browser.new_context(viewport={'width':width,'height':1000});page=context.new_page();page.set_default_timeout(20000);errors=[];forbidden=[]
   page.on('pageerror',lambda e:errors.append(type(e).__name__))
   page.on('request',lambda r:forbidden.append(True) if r.method=='POST' and any(s in r.url for s in ['/amend','/extract','/visit-capture/save','chart-select','chart-unlink']) else None)
   stage='login';page.goto(ROOT);page.locator('#ceGlobalKey').click();page.locator('#ceGlobalKeyInput').fill(os.environ['CLINICAL_DATA_KEY'])
   page.locator('#ceGlobalLoginForm button[type=submit]').click();expect(page.locator('#ceGlobalAuthStatus')).to_contain_text('Συνδέθηκες');page.keyboard.press('Escape')
   stage='open-today'
   with page.expect_response(lambda r:'/clinical/calendar/agenda?' in r.url) as reply:page.locator('#doctorDateButton').click()
   assert reply.value.status==200;expect(page.locator('#doctorDayLabel')).to_contain_text('Σήμερα')
   stage='read-real-tomorrow'
   with page.expect_response(lambda r:'/clinical/calendar/agenda?' in r.url) as reply:page.locator('#doctorDayNext').click()
   assert reply.value.status==200;rows=reply.value.json();assert isinstance(rows,list)
   expect(page.locator('#doctorDayLabel')).to_contain_text('Αύριο')
   expect(page.locator('#doctorPopoverContent button.doctor-action-row')).to_have_count(len(rows))
   tomorrow=datetime.now(ZoneInfo('Asia/Nicosia')).date()+timedelta(days=1)
   for row in rows:
    when=datetime.fromisoformat(row['start_at'].replace('Z','+00:00'))
    if when.tzinfo is None:when=when.replace(tzinfo=ZoneInfo('UTC'))
    assert when.astimezone(ZoneInfo('Asia/Nicosia')).date()==tomorrow
   home=context.request.get(BASE+'/clinical/calendar/cockpit-context');assert home.status==200
   upcoming=home.json().get('next');matched=False
   if upcoming:
    when=datetime.fromisoformat(upcoming['start_at'].replace('Z','+00:00'))
    if when.astimezone(ZoneInfo('Asia/Nicosia')).date()==tomorrow:
     uid=upcoming['appointment_id'];uid=uid if uid.startswith('cal.com:') else 'cal.com:'+uid
     assert uid in {r['appointment_id'] for r in rows};matched=True
   # Crop to date/arrows only: never capture actual patient names/reasons.
   page.locator('.doctor-day-navigation').screenshot(path=str(out/f'tomorrow-navigation-{width}.png'))
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   stage='back-today'
   with page.expect_response(lambda r:'/clinical/calendar/agenda?' in r.url) as reply:page.locator('#doctorDayToday').click()
   assert reply.value.status==200;expect(page.locator('#doctorDayLabel')).to_contain_text('Σήμερα')
   page.locator('#doctorDateButton').click();expect(page.locator('#doctorPopover')).not_to_be_visible()
   assert not errors and not forbidden
   context.close();print('PASS DAY NAVIGATION LIVE READ:',width,'actual tomorrow rows:',len(rows),'correct Cyprus day; Home next inclusion:',matched,'Today return; zero clinical/link writes; names never logged/captured')
  browser.close()
 print('PASS DAY NAVIGATION LIVE READ: actual protected calendar source1280/390; navigation screenshots contain only date/arrows. Appointment caller fidelity and failure/stale/DST covered local/CI.')
except Exception as error:
 print('FAIL DAY NAVIGATION LIVE stage: '+stage+' type: '+type(error).__name__);sys.exit(1)
