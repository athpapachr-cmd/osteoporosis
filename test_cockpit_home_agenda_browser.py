"""Local-only browser checks, all names and records synthetic."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
from playwright.sync_api import sync_playwright, expect

SCRIPT = Path('static/cockpit/home-agenda.js').read_text()
CSS = Path('static/cockpit/doctor-shell.css').read_text()


def test_agenda_lifecycle_and_navigation():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=shutil.which('google-chrome') or shutil.which('chromium') or None, args=['--no-sandbox'])
        for width in (1280, 390):
            page = browser.new_page(viewport={'width': width, 'height': 850})
            page.clock.install(time=datetime(2026, 11, 18, 10, 52, tzinfo=timezone.utc))
            page.set_content('<style>' + CSS + '</style><button id="outside">Εκτός</button><main id="homeAgendaRows"></main>')
            page.evaluate('''() => {
              window.__selected=null;window.__reloads=0;window.__pending=[];
              window.CockpitVisitBrief={open(x){window.__selected=x;}};
              window.fetch=()=>new Promise(resolve=>window.__pending.push(resolve));
              window.addEventListener('clinical:unauthenticated',()=>{document.getElementById('homeAgendaRows').textContent='Απαιτείται σύνδεση';});
            }''')
            page.add_script_tag(content=SCRIPT)
            page.evaluate('''() => {window.__rows=[
              {appointment_id:'p',patient_display_name:'ΠΑΛΑΙΟ ΟΝΟΜΑ',start_at:'2026-11-18T09:00:00Z',end_at:'2026-11-18T09:20:00Z'},
              {appointment_id:'a',patient_display_name:'ΝΕΟ ΟΝΟΜΑ',start_at:'2026-11-18T11:40:00Z',end_at:'2026-11-18T12:00:00Z',reason:'Πόνος Ραντεβού για: ΝΕΟ ΟΝΟΜΑ'},
              {appointment_id:'b',patient_display_name:'Συνθετικός Δεύτερος',start_at:'2026-11-18T12:20:00Z',end_at:'2026-11-18T12:40:00Z'},
              {patient_display_name:'Ασαφές',start_at:'2026-11-18T12:30:00Z'}
            ];window.__reload=()=>{window.__reloads++;const keep=CockpitHomeAgenda.prepareReload();CockpitHomeAgenda.clear(true);
              const loading=document.createElement('p');loading.tabIndex=-1;loading.textContent='Φόρτωση';
              document.getElementById('homeAgendaRows').replaceChildren(loading);if(keep)CockpitHomeAgenda.markLoading(loading);
              window.__finish=()=>CockpitHomeAgenda.render(window.__rows,{day:'2026-11-18',reload:window.__reload});};
              CockpitHomeAgenda.render(window.__rows,{day:'2026-11-18',reload:window.__reload});}''')
            root=page.locator('#homeAgendaRows')
            expect(root).to_contain_text('1 περασμένα ραντεβού')
            expect(root).to_contain_text('Επόμενο με επιβεβαιωμένη ώρα σε 48 λεπτά')
            expect(root.locator('.hg-row.next')).to_contain_text('Νεο Ονομα')
            expect(root.locator('.hg-row.unknown')).to_be_disabled()
            root.locator('.hg-row.next').click()
            assert page.evaluate('window.__selected.uid') == 'a'
            assert page.evaluate('window.__selected.name') == 'ΝΕΟ ΟΝΟΜΑ'
            assert page.evaluate('window.__selected.reason') == 'Πόνος Ραντεβού για: ΝΕΟ ΟΝΟΜΑ'
            past=root.get_by_role('button',name='1 περασμένα ραντεβού')
            past.focus(); page.keyboard.press('Enter')
            expect(root.locator('.hg-row.past')).to_have_count(1)
            assert page.evaluate("document.activeElement.getAttribute('aria-expanded')") == 'true'
            page.clock.fast_forward('00:05:01')
            assert page.evaluate('window.__reloads') >= 1
            assert page.evaluate("document.activeElement.textContent") == 'Φόρτωση'
            page.evaluate('window.__finish()')
            assert page.evaluate("document.activeElement.getAttribute('aria-expanded')") == 'true'
            root.locator('.hg-row.next').focus()
            page.evaluate('window.__reload()')
            page.locator('#outside').focus();page.evaluate('window.__finish()')
            assert page.evaluate("document.activeElement.id") == 'outside', 'do not steal focus moved elsewhere'
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
            # Tomorrow succeeds, preserves all entries on expansion and opens the selected uid.
            page.evaluate("CockpitHomeAgenda.render([],{day:'2026-11-18',reload(){}})")
            page.evaluate("""() => window.__pending.at(-1)({ok:true,status:200,json:async()=>Array.from({length:5},(_,i)=>({appointment_id:'tom-'+i,patient_display_name:'Συνθετικός '+i,start_at:`2026-11-19T${String(8+i).padStart(2,'0')}:00:00Z`,end_at:`2026-11-19T${String(8+i).padStart(2,'0')}:20:00Z`}))})""")
            # Use valid ISO timestamps for all 5 appointments.
            page.wait_for_timeout(50)
            expect(root).to_contain_text('Αύριο')
            expect(root.get_by_role('button',name='Δες και τα 5')).to_be_visible()
            root.get_by_role('button',name='Δες και τα 5').click()
            expect(root.locator('.hg-row.tomorrow')).to_have_count(5)
            root.locator('.hg-row.tomorrow').first.click()
            assert page.evaluate('window.__selected.uid') == 'tom-0'
            # Failed reads are unavailable, not an empty calendar.
            page.evaluate("CockpitHomeAgenda.render([],{day:'2026-11-18',reload(){}})")
            page.evaluate("""() => window.dispatchEvent(new CustomEvent('cockpit:surgery-counts',{detail:{attention:[{date:'2026-11-19',name:'ΣΥΝΘΕΤΙΚΟ ΧΕΙΡΟΥΡΓΕΙΟ',procedure:'Δοκιμή',time:'10:00'}]}}))""")
            expect(root).to_contain_text('Συνθετικο Χειρουργειο')
            page.evaluate("window.__pending.at(-1)({ok:false,status:503})")
            expect(root).to_contain_text('δεν είναι διαθέσιμο')
            expect(root).to_contain_text('Συνθετικο Χειρουργειο')
            expect(root).not_to_contain_text('Δεν βρέθηκαν ραντεβού αύριο')
            # A tomorrow response arriving after logout must not restore data.
            page.evaluate('''() => {CockpitHomeAgenda.render([],{day:'2026-11-18',reload(){}});
              window.dispatchEvent(new Event('clinical:unauthenticated'));
              window.__pending.at(-1)({ok:true,status:200,json:async()=>[{appointment_id:'late',patient_display_name:'STALE-SYN',start_at:'2026-11-19T09:00:00Z',end_at:'2026-11-19T09:20:00Z'}]});}''')
            page.wait_for_timeout(50)
            expect(root).not_to_contain_text('STALE-SYN')
            expect(root).to_contain_text('Απαιτείται σύνδεση')
            page.close()
        browser.close()
