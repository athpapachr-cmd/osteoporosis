"""Synthetic real-browser proof of the three Office Cases tabs; no live service."""

import os
import re
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get("OFFICE_CASES_EVIDENCE_DIR", "/tmp/office-cases-browser-evidence-20261026"))
HTML = re.sub(r"<script\b[^>]*>.*?</script>", "", (ROOT / "static/cockpit/index.html").read_text(), flags=re.S)
HTML = re.sub(r"<link\b[^>]*rel=\"stylesheet\"[^>]*>", "", HTML)
CSS = "\n".join((ROOT / path).read_text() for path in (
    "static/cockpit/styles.css", "static/cockpit/doctor-shell.css",
))
JS = (ROOT / "static/cockpit/office-cases.js").read_text()


def _exercise(width: int):
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": 900})
        errors = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.set_content(HTML)
        page.add_style_tag(content=CSS)
        page.evaluate("""() => {
          document.querySelector('#doctorLibraryView').hidden=false;
          document.querySelector('#doctorMainView').hidden=true;
          document.querySelectorAll('[data-doctor-library-panel]').forEach(panel => {
            panel.hidden = panel.dataset.doctorLibraryPanel !== 'cases';
          });
          // about:blank is not a secure origin; production HTTPS supplies this API.
          Object.defineProperty(window.crypto,'randomUUID',
            {value:()=> '11111111-1111-4111-8111-111111111111'});
          window.fetch=async (url) => {
            const path=String(url);
            let data;
            if(path.endsWith('/accidents?offset=0')) data={cases:[{
              case_id:'synthetic-accident-1',patient_name:'<img src=x onerror=alert(1)> Συνθετικός',
              phone:'000000',accident_date:'2026-10-01',visit_date:'2026-10-02',
              writing_started_date:null,report_sent_date:null,amount_due_cents:10000,
              received_cents:3000,balance_cents:7000,revision:2
            }],has_more:false,next_offset:null};
            else if(path.endsWith('/accidents/synthetic-accident-1')) data={
              case_id:'synthetic-accident-1',patient_name:'Συνθετικός',phone:'000000',
              accident_date:'2026-10-01',visit_date:'2026-10-02',writing_started_date:null,
              report_sent_date:null,amount_due_cents:10000,received_cents:3000,
              balance_cents:7000,revision:2,payments:[{payment_id:'synthetic-pay-1',
                amount_cents:3000,paid_on:'2026-10-03',payer:'Συνθετικός πληρωτής',
                created_at:'2026-10-03T10:00:00',voided_at:null,void_reason:null}]};
            else if(path.includes('/office-cases/rf?')) data={applications:[{
              id:'synthetic-rf-1',revision:2,stage:'scheduled',patient_name:'RF Συνθετικός',
              identity_number:'SYN-001',laterality:'left',indication_code:'SYN',
              scheduled_date:'2026-11-01',surgery_id:null
            }],next_cursor:null,has_more:false};
            else if(path.includes('/accidents/synthetic-accident-1/payments'))
              return new Promise(resolve => { window.resolvePendingPayment = resolve; });
            else return {ok:false,status:401,json:async()=>({detail:'Unauthorized'})};
            return {ok:true,status:200,json:async()=>data};
          };
          window.CockpitSurgeryQueue={prefillFromRF(card){window.lastRF=card;
            window.CockpitOfficeCases.selectTab('surgery');}};
        }""")
        page.add_script_tag(content=JS)
        page.locator('[data-office-tab="accidents"]').click()
        expect(page.locator('#accidentList')).to_contain_text('Συνθετικός')
        assert page.locator('#accidentList img').count() == 0
        page.get_by_role('button', name='Άνοιγμα', exact=True).click()
        expect(page.locator('#accidentPaymentList')).to_contain_text('Συνθετικός πληρωτής')
        expect(page.locator('#accidentPaymentList')).to_contain_text('Καταχώριση (Κύπρος):')
        expect(page.locator('#accidentPaymentList')).to_contain_text('13:00')
        expect(page.locator('#accidentDetailSummary')).to_contain_text('70,00')
        OUT.mkdir(exist_ok=True)
        page.screenshot(path=str(OUT / f'office-accidents-{width}.png'))
        page.locator('#accidentList').get_by_role('button', name='Επεξεργασία').click()
        expect(page.locator('#accidentPatientName')).to_have_value('<img src=x onerror=alert(1)> Συνθετικός')
        page.locator('#paymentAmount').fill('30.00')
        page.locator('#paymentDate').fill('2026-10-04')
        page.locator('#paymentPayer').fill('Συνθετικός δεύτερος πληρωτής')
        page.locator('#accidentPaymentForm button[type=submit]').click()
        page.wait_for_function('window.resolvePendingPayment !== undefined')
        page.evaluate("window.dispatchEvent(new Event('clinical:unauthenticated'))")
        expect(page.locator('#accidentList')).to_be_empty()
        expect(page.locator('#accidentDetail')).to_be_hidden()
        expect(page.locator('#accidentPatientName')).to_have_value('')
        expect(page.locator('#paymentPayer')).to_have_value('')
        expect(page.locator('#accidentForm button[type=submit]')).to_be_disabled()
        expect(page.locator('#accidentPaymentForm button[type=submit]')).to_be_disabled()
        page.evaluate("window.dispatchEvent(new Event('clinical:authenticated'))")
        expect(page.locator('#accidentList')).to_contain_text('Συνθετικός')
        page.get_by_role('button', name='Άνοιγμα', exact=True).click()
        expect(page.locator('#paymentReviewConfirm')).to_be_visible()
        expect(page.locator('#accidentPaymentForm button[type=submit]')).to_be_disabled()
        page.evaluate("window.resolvePendingPayment({ok:true,status:200,json:async()=>({})})")
        expect(page.locator('#accidentPaymentForm button[type=submit]')).to_be_disabled()
        page.locator('#paymentReviewConfirm').click()
        expect(page.locator('#accidentPaymentForm button[type=submit]')).to_be_enabled()
        page.locator('[data-office-tab="rf"]').click()
        expect(page.locator('#officeRfList')).to_contain_text('RF Συνθετικός')
        page.screenshot(path=str(OUT / f'office-rf-{width}.png'))
        page.get_by_role('button', name='Προσθήκη στη λίστα χειρουργείων').click()
        expect(page.locator('[data-office-tab-panel="surgery"]')).to_be_visible()
        assert page.evaluate('window.lastRF.id') == 'synthetic-rf-1'
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 2')
        assert not errors, errors
        browser.close()


def test_office_cases_three_tabs_desktop_and_narrow():
    for width in (1280, 390):
        _exercise(width)
