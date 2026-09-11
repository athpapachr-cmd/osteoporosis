"""Actual Chromium + loopback HTTP + real CU-1 integration acceptance.

No mocked clinical engine. One test deliberately aborts network requests to
prove fail-closed export behavior. Safari/VoiceOver remain human acceptance.
"""
from __future__ import annotations
import json
import sys
import threading
import unittest
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from clinic_utilities.physio_referral_product.prototype import server as p
ARTIFACTS=Path('artifacts/knee-oa-prototype')


class BrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ARTIFACTS.mkdir(parents=True,exist_ok=True)
        cls.server=p.ThreadingHTTPServer(('127.0.0.1',0),p.Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.origin=f'http://127.0.0.1:{cls.server.server_port}'
        cls.playwright=sync_playwright().start()
        cls.browser=cls.playwright.chromium.launch(headless=True)
    @classmethod
    def tearDownClass(cls):
        cls.browser.close();cls.playwright.stop()
        cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def setUp(self):
        self.context=self.browser.new_context(viewport={'width':1280,'height':1000},reduced_motion='reduce',permissions=['clipboard-read','clipboard-write'])
        self.page=self.context.new_page();self.errors=[];self.requests=[]
        self.page.on('pageerror',lambda error:self.errors.append(str(error)))
        self.page.on('request',lambda r:self.requests.append(r.url))
        self.page.goto(self.origin)
        expect(self.page.locator('#plan .row')).to_have_count(3)
        self.page.wait_for_function("document.querySelector('#reviewStatus').textContent!=='Η παραπομπή ενημερώνεται'")
    def tearDown(self):
        self.assertEqual(self.errors,[])
        self.context.close()
    def ready(self):
        self.page.locator('#assertion').click();self.page.locator('[data-side=right]').click()
        expect(self.page.locator('#copy')).to_be_enabled()
    def group(self,title):
        if self.page.locator('#advancedToggle').get_attribute('aria-expanded')!='true':self.page.locator('#advancedToggle').click()
        self.page.locator('#advanced summary').filter(has_text=title).click()
    def select_adjunct(self,item):
        self.group('Συμπληρωματικά')
        self.page.locator(f'#advanced [data-select={item}]').click()
        expect(self.page.locator(f'#plan [data-select={item}]')).to_be_visible()
        expect(self.page.locator('#copy')).to_be_enabled()
    def test_01_routine_flow_and_clipboard(self):
        expect(self.page.locator('#copy')).to_be_disabled()
        self.ready();self.page.locator('[data-finding=pain]').click()
        self.page.locator('[data-phenotype=weakness_symptom_or_context]').click()
        self.page.locator('#functionToggle').click();self.page.locator('[data-function=stairs]').click()
        expect(self.page.locator('#referralText')).to_contain_text('δυσχέρεια στις σκάλες')
        expect(self.page.locator('#copy')).to_be_enabled()
        self.page.screenshot(path=str(ARTIFACTS/'desktop.png'),full_page=True)
        self.page.locator('#copy').click()
        copied=self.page.evaluate('navigator.clipboard.readText()')
        self.assertTrue(copied.startswith('ΔΟΚΙΜΑΣΤΙΚΟ · ΟΧΙ ΓΙΑ ΚΛΙΝΙΚΗ ΧΡΗΣΗ'))
        self.assertIn('δεξιού γόνατος',copied)
        self.assertNotIn('NICE',copied)
    def test_02_evidence_does_not_select_and_focus_returns(self):
        self.ready();before=self.page.locator('#referralText').inner_text()
        control=self.page.locator('#plan [data-evidence=therapeutic_exercise]');control.click()
        expect(self.page.locator('#sheet')).to_be_visible()
        expect(self.page.locator('#sheetTitle')).to_be_focused()
        expect(self.page.locator('#sheetBody')).to_contain_text('NICE NG226 · 2022')
        expect(self.page.locator('#sheetBody')).to_contain_text('11/09/2026')
        self.page.keyboard.press('Tab')
        self.assertTrue(self.page.evaluate("document.querySelector('#sheet').contains(document.activeElement)"))
        self.page.keyboard.press('Escape');expect(control).to_be_focused()
        self.assertEqual(self.page.locator('#referralText').inner_text(),before)
    def test_03_suggestion_is_explicit_and_traceable(self):
        self.ready();self.page.locator('#functionToggle').click();self.page.locator('[data-function=stairs]').click()
        expect(self.page.locator('#suggestions')).to_contain_text('Κλινική προσαρμογή')
        expect(self.page.locator('#referralText')).not_to_contain_text('λειτουργική επανεκπαίδευση')
        self.page.locator('#suggestions [data-add=functional_task_retraining]').click()
        expect(self.page.locator('#referralText')).to_contain_text('λειτουργική επανεκπαίδευση για τις σκάλες')
        expect(self.page.locator('#plan [data-select=functional_task_retraining]')).to_have_attribute('aria-pressed','true')
    def test_04_omission_dismissal_preserves_selection(self):
        self.ready();self.page.locator('#plan [data-select=therapeutic_exercise]').click()
        expect(self.page.locator('#suggestions')).to_contain_text('NICE NG226 · 2022')
        self.page.locator('#suggestions [data-dismiss-suggestion=therapeutic_exercise]').click()
        expect(self.page.locator('#suggestions .suggestion')).to_have_count(0)
        expect(self.page.locator('#plan [data-select=therapeutic_exercise]')).to_have_attribute('aria-pressed','false')
        expect(self.page.locator('#referralText')).not_to_contain_text('με θεραπευτική άσκηση')
    def test_05_all_conflicting_sources_and_collapsed_choices(self):
        self.ready();self.select_adjunct('acupuncture')
        expect(self.page.locator('.bubble')).to_have_count(1)
        self.page.locator('#plan [data-evidence=acupuncture]').click()
        expect(self.page.locator('#sheet .source-position')).to_have_count(5)
        expect(self.page.locator('#sheetBody')).to_contain_text('Συστήνει να μην προσφέρεται βελονισμός')
        expect(self.page.locator('#sheetBody')).to_contain_text('Υπό όρους σύσταση υπέρ')
        expect(self.page.locator('#sheetBody')).to_contain_text('Ανεπαρκή δεδομένα')
        self.page.screenshot(path=str(ARTIFACTS/'evidence-conflict.png'),full_page=True)
        self.page.keyboard.press('Escape');self.page.locator('[data-dismiss-bubble]').click()
        expect(self.page.locator('.bubble')).to_have_count(0)
        self.page.locator('#advancedToggle').click()
        expect(self.page.locator('#advancedLabel')).to_contain_text('1 ενεργά')
        expect(self.page.locator('#plan [data-select=acupuncture]')).to_have_attribute('aria-pressed','true')
        expect(self.page.locator('#referralText')).to_contain_text('βελονισμός')
        self.assertEqual(self.page.locator('[data-evidence=acupuncture]:visible').count(),1)
    def test_06_manual_edit_reconciliation(self):
        self.ready();self.page.locator('[data-menu]').first.click();self.page.locator('[data-edit]').click()
        self.page.locator('#manualText').fill('Δοκιμαστική χειροκίνητη παραπομπή.')
        self.page.locator('[data-confirm-manual]').click()
        expect(self.page.locator('#referralText')).to_have_text('Δοκιμαστική χειροκίνητη παραπομπή.')
        self.page.locator('[data-side=left]').click()
        expect(self.page.locator('#copy')).to_be_disabled()
        expect(self.page.locator('#manualReconcile')).to_be_visible()
        self.page.locator('#manualReconcile [data-manual-review]').click()
        expect(self.page.locator('#manualText')).to_have_value('Δοκιμαστική χειροκίνητη παραπομπή.')
        self.page.locator('[data-use-generated]').last.click()
        expect(self.page.locator('#referralText')).to_contain_text('αριστερού γόνατος')
        expect(self.page.locator('#copy')).to_be_enabled()
    def test_07_real_safety_blocks_copy_and_print(self):
        self.ready();self.group('Κλινικός έλεγχος')
        self.page.locator('[data-select=infection_or_septic_joint_concern]').click()
        expect(self.page.locator('#reviewStatus')).to_have_text('Απαιτείται κλινικός έλεγχος')
        expect(self.page.locator('#copy')).to_be_disabled()
        self.page.locator('#plan [data-evidence=therapeutic_exercise]').click()
        expect(self.page.locator('#sheetSafety')).to_be_visible();self.page.keyboard.press('Escape')
        expect(self.page.locator('#copy')).to_be_disabled()
        self.page.evaluate("window.dispatchEvent(new Event('beforeprint'))")
        self.assertNotIn('Παραπομπή για',self.page.locator('#printArea').text_content())
    def test_08_network_failure_cannot_copy_stale_text(self):
        self.ready()
        self.page.route('**/api/project',lambda route:route.abort())
        self.page.locator('[data-side=left]').click()
        expect(self.page.locator('#copy')).to_be_disabled()
        expect(self.page.locator('#reviewStatus')).to_contain_text('τοπική σύνδεση')
        expect(self.page.locator('#referralText')).not_to_contain_text('δεξιού γόνατος')
    def test_09_mobile_reflow_targets_and_text_enlargement(self):
        self.ready();self.page.locator('[data-finding=pain]').click()
        expect(self.page.locator('#copy')).to_be_enabled()
        for width in [320,390,800,1280]:
            self.page.set_viewport_size({'width':width,'height':900})
            self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth<=innerWidth'),width)
            small=self.page.locator('button:visible').evaluate_all('(nodes)=>nodes.map(n=>({text:n.textContent,w:n.getBoundingClientRect().width,h:n.getBoundingClientRect().height})).filter(n=>n.w<43.9||n.h<43.9)')
            self.assertEqual(small,[],width)
            if width==390:self.page.screenshot(path=str(ARTIFACTS/'mobile.png'))
        self.page.set_viewport_size({'width':320,'height':900})
        self.page.evaluate("document.documentElement.style.fontSize='32px'")
        self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth<=innerWidth'))
        self.page.locator('#openPreview').click();expect(self.page.locator('#sheet')).to_be_visible()
        self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth<=innerWidth'))
    def test_10_forced_colours_and_modal_keyboard(self):
        self.ready();self.page.emulate_media(forced_colors='active',reduced_motion='reduce')
        self.page.locator('#plan [data-evidence=therapeutic_exercise]').click()
        self.page.locator('#sheet .legend summary').click()
        cues=self.page.locator('#sheet .legend [data-cue]').evaluate_all('(nodes)=>nodes.map(n=>n.dataset.cue)')
        self.assertEqual(len(set(cues)),6)
        for _ in range(35):
            self.page.keyboard.press('Tab')
            self.assertTrue(self.page.evaluate("document.querySelector('#sheet').contains(document.activeElement)"))
    def test_11_no_storage_or_unrequested_external_requests(self):
        self.ready();self.page.locator('#plan [data-evidence=therapeutic_exercise]').click()
        self.assertEqual(self.page.evaluate('localStorage.length'),0)
        self.assertEqual(self.page.evaluate('sessionStorage.length'),0)
        self.assertTrue(all(url.startswith(self.origin+'/') or url==self.origin for url in self.requests),self.requests)
    def test_12_reset_and_simulated_bfcache_clear_state(self):
        self.ready();self.page.locator('[data-finding=pain]').click()
        self.page.locator('#reset').click();self.page.locator('[data-confirm-reset]').click()
        expect(self.page.locator('#assertion')).to_have_attribute('aria-pressed','false')
        expect(self.page.locator('#copy')).to_be_disabled()
        self.ready()
        self.page.evaluate("window.dispatchEvent(new PageTransitionEvent('pagehide',{persisted:true}));window.dispatchEvent(new PageTransitionEvent('pageshow',{persisted:true}));")
        expect(self.page.locator('#assertion')).to_have_attribute('aria-pressed','false')
        expect(self.page.locator('#copy')).to_be_disabled()


if __name__=='__main__':unittest.main(verbosity=2)
