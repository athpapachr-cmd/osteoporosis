"""Chromium acceptance for the scan-first Knee-OA `Περισσότερα` v3 redesign."""
from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from clinic_utilities.physio_referral_product.prototype import server as p

ARTIFACTS = Path('artifacts/knee-oa-prototype')


class MoreV3BrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        cls.server = p.ThreadingHTTPServer(("127.0.0.1", 0), p.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True); cls.thread.start()
        cls.origin = f"http://127.0.0.1:{cls.server.server_port}"
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close(); cls.playwright.stop()
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join()

    def setUp(self):
        self.context = self.browser.new_context(viewport={"width":1280,"height":1000}, reduced_motion="reduce")
        self.page = self.context.new_page(); self.errors=[]
        self.page.on('pageerror', lambda error: self.errors.append(str(error)))
        self.page.goto(self.origin)
        expect(self.page.locator('#reviewStatus')).not_to_have_text('Η παραπομπή ενημερώνεται')
        self.page.locator('#assertion').click(); self.page.locator('[data-side=right]').click()
        expect(self.page.locator('#copy')).to_be_enabled()

    def tearDown(self):
        self.assertEqual(self.errors, [])
        self.context.close()

    def open_more(self):
        if self.page.locator('#advancedToggle').get_attribute('aria-expanded') != 'true':
            self.page.locator('#advancedToggle').click()
        expect(self.page.locator('#advanced [data-v3-more]')).to_be_visible()

    def test_overview_is_scan_first_not_a_control_wall(self):
        self.open_more()
        expect(self.page.locator('#advanced .v3-category-row')).to_have_count(6)
        self.assertEqual(self.page.locator('#advanced .advanced-group').count(), 0)
        self.assertEqual(self.page.locator('#advanced [data-select]:visible').count(), 0)
        expect(self.page.locator('#v3Favorites')).to_contain_text('★ Συχνά')
        expect(self.page.locator('#v3Categories')).to_contain_text('Όλα')
        rows=self.page.locator('#advanced .v3-category-row:visible').evaluate_all('(nodes)=>nodes.map(n=>n.getBoundingClientRect().height)')
        self.assertTrue(all(h>=44 for h in rows), rows)
        self.page.screenshot(path=str(ARTIFACTS/'more-v3-overview.png'), full_page=True)

    def test_category_selection_updates_collapsed_summary_and_remains_reachable(self):
        self.open_more()
        self.page.locator('[data-v3-category=exam]').click()
        expect(self.page.locator('#sheetTitle')).to_have_text('Εξέταση')
        expect(self.page.locator('#sheet [data-select=effusion]')).to_be_visible()
        expect(self.page.locator('#sheet [data-q-ffd]')).to_be_visible()
        expect(self.page.locator('#sheet [data-q-tenderness=pes_anserine_region]')).to_be_visible()
        self.page.locator('#sheet [data-select=effusion]').click()
        self.page.keyboard.press('Escape')
        exam=self.page.locator('#advanced [data-v3-category=exam]')
        expect(exam).to_contain_text('1 ενεργά')
        exam.click(); expect(self.page.locator('#sheet [data-select=effusion]')).to_have_attribute('aria-pressed','true')
        self.page.keyboard.press('Escape')
        for category,title,sample in [
            ('function','Λειτουργία & στόχοι','graded_return_to_sport'),
            ('rehab','Αποκατάσταση','graded_activity_exposure'),
            ('adjuncts','Συμπληρωματικά','acupuncture'),
            ('safety','Κλινικός έλεγχος','infection_or_septic_joint_concern'),
        ]:
            self.page.locator(f'[data-v3-category={category}]').click()
            expect(self.page.locator('#sheetTitle')).to_have_text(title)
            expect(self.page.locator(f'#sheet [data-select={sample}]')).to_be_visible()
            self.page.keyboard.press('Escape')
        self.page.locator('[data-v3-category=notes]').click()
        expect(self.page.locator('#v3RestrictionId')).to_be_visible(); expect(self.page.locator('#v3ClinicalNote')).to_be_visible()

    def test_favorites_require_customization_and_pin_does_not_select(self):
        self.open_more()
        self.assertEqual(self.page.locator('#advanced .v3-pin:visible').count(),0)
        self.page.locator('[data-v3-customize-favorites]').click()
        expect(self.page.locator('[data-v3-customize-favorites]')).to_have_attribute('aria-pressed','true')
        self.page.locator('[data-v3-category=function]').click()
        pin=self.page.locator('#sheet [data-favorite-v3][data-favorite-item=graded_return_to_sport]')
        expect(pin).to_be_visible()
        choice=self.page.locator('#sheet [data-select=graded_return_to_sport]')
        expect(choice).to_have_attribute('aria-pressed','false')
        before=self.page.locator('#referralText').inner_text()
        pin.click(); expect(choice).to_have_attribute('aria-pressed','false')
        self.assertEqual(self.page.locator('#referralText').inner_text(),before)
        self.page.keyboard.press('Escape')
        favorites=self.page.locator('#v3Favorites')
        expect(favorites).to_contain_text('επιστροφή')
        self.assertEqual(self.page.locator('#advanced .v3-pin:visible').count(),0)
        shortcut=favorites.locator('[data-select=graded_return_to_sport]')
        expect(shortcut).to_have_attribute('aria-pressed','false')
        shortcut.click(); expect(shortcut).to_have_attribute('aria-pressed','true')
        self.page.screenshot(path=str(ARTIFACTS/'more-v3-favorites.png'), full_page=True)
        self.assertEqual(self.page.evaluate('localStorage.length'),0)
        self.assertEqual(self.page.evaluate('sessionStorage.length'),0)
        self.assertEqual(self.page.locator('button').filter(has_text='Απόκρυψη').count(),0)

    def test_relevant_now_is_contextual_and_never_selects_by_appearing(self):
        # v4 activates the reported weakness parent through its compact clinical
        # sheet. Merely opening/closing that sheet must not invent an objective
        # examination finding, preserving the original More-v3 invariant.
        self.page.locator('[data-clinical-v4=weakness]').click()
        expect(self.page.locator('#sheetTitle')).to_have_text('Αδυναμία')
        self.assertEqual(self.page.locator('#sheet [data-q-weakness][aria-pressed=true]').count(),0)
        self.page.locator('#closeSheet').click()
        expect(self.page.locator('#referralText')).to_contain_text('μυϊκή αδυναμία')
        self.open_more()
        relevant=self.page.locator('#v3Relevant')
        expect(relevant).to_be_visible(); expect(relevant).to_contain_text('Αδυναμία στην εξέταση')
        before=self.page.locator('#referralText').inner_text()
        self.assertEqual(self.page.locator('[data-q-weakness][aria-pressed=true]').count(),0)
        relevant.locator('[data-v3-focus=weakness]').click()
        expect(self.page.locator('#sheetTitle')).to_have_text('Εξέταση')
        expect(self.page.locator('#sheet [data-q-weakness=objective]')).to_have_attribute('aria-pressed','false')
        self.assertEqual(self.page.locator('#referralText').inner_text(),before)

    def test_mobile_overview_reflows_without_losing_hierarchy(self):
        self.page.set_viewport_size({"width":390,"height":900})
        self.open_more()
        self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth<=innerWidth'))
        expect(self.page.locator('#advanced .v3-category-row')).to_have_count(6)
        self.page.screenshot(path=str(ARTIFACTS/'more-v3-mobile.png'), full_page=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
