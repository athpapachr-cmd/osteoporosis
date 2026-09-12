"""Actual Chromium acceptance for Knee-OA usability refinement v2."""
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


class UsabilityV2BrowserTests(unittest.TestCase):
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
        self.context = self.browser.new_context(viewport={"width": 1280, "height": 1000}, reduced_motion="reduce")
        self.page = self.context.new_page(); self.errors=[]
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.goto(self.origin)
        expect(self.page.locator('#reviewStatus')).not_to_have_text('Η παραπομπή ενημερώνεται')
        self.page.locator('#assertion').click(); self.page.locator('[data-side=right]').click()
        expect(self.page.locator('#copy')).to_be_enabled()

    def tearDown(self):
        self.assertEqual(self.errors, [])
        self.context.close()

    def test_additional_suggestions_show_true_extra_count_and_titles(self):
        self.page.locator('#functionToggle').click()
        for item in ['walking_tolerance','stairs','sit_to_stand','sport_gym']:
            self.page.locator(f'[data-function={item}]').click()
        self.page.locator('#plan [data-select=therapeutic_exercise]').click()
        panel=self.page.locator('#suggestions .suggestions-more-panel')
        expect(panel).to_be_visible()
        expect(panel).to_contain_text('Άλλες 4 προτάσεις')
        expect(panel).to_contain_text('Σταδιακή δραστηριότητα')
        expect(panel).to_contain_text('Αντοχή & λειτουργική ικανότητα')
        expect(panel).to_contain_text('Επανεκπαίδευση βάδισης')
        expect(panel).to_contain_text('Λειτουργική επανεκπαίδευση')
        expect(self.page.locator('#plan [data-select=functional_task_retraining]')).to_have_count(0)
        panel.click()
        expect(self.page.locator('#sheet')).to_be_visible()
        expect(self.page.locator('#sheet .suggestion')).to_have_count(5)
        expect(self.page.locator('#sheet [data-add=functional_task_retraining]')).to_be_visible()
        self.page.screenshot(path=str(ARTIFACTS/'suggestions-v2.png'), full_page=True)

    def test_single_suggestion_has_no_secondary_panel(self):
        self.page.locator('#functionToggle').click(); self.page.locator('[data-function=stairs]').click()
        expect(self.page.locator('#suggestions .suggestion')).to_have_count(1)
        expect(self.page.locator('#suggestions .suggestions-more-panel')).to_have_count(0)

    def test_direct_edit_visible_desktop_and_mobile_and_preserves_manual_text(self):
        edit=self.page.locator('#directEditV2')
        expect(edit).to_be_visible(); expect(edit).to_be_enabled()
        edit.click(); expect(self.page.locator('#manualText')).to_be_visible()
        self.page.locator('#manualText').fill('Το δικό μου κείμενο παραπομπής.')
        self.page.locator('[data-confirm-manual]').click()
        expect(self.page.locator('#referralText')).to_have_text('Το δικό μου κείμενο παραπομπής.')
        self.page.locator('[data-side=left]').click()
        expect(self.page.locator('#referralText')).to_have_text('Το δικό μου κείμενο παραπομπής.')
        expect(self.page.locator('#manualReconcile')).to_be_visible()
        expect(self.page.locator('#copy')).to_be_disabled()
        self.page.set_viewport_size({"width":390,"height":900})
        self.page.locator('#openPreview').click()
        mobile_edit=self.page.locator('#sheet [data-edit].direct-edit')
        expect(mobile_edit).to_be_visible(); expect(mobile_edit).to_be_enabled()
        mobile_edit.click(); expect(self.page.locator('#manualText')).to_have_value('Το δικό μου κείμενο παραπομπής.')

    def test_favorite_pin_reorders_only_and_never_selects_by_itself(self):
        self.page.locator('#advancedToggle').click()
        self.page.locator('#advanced summary').filter(has_text='Στόχοι').click()
        star=self.page.locator('#advanced [data-favorite-v2][data-favorite-category=goals][data-favorite-item=graded_return_to_sport]').last
        expect(star).to_be_visible()
        before=self.page.locator('#referralText').inner_text()
        star.click()
        favorites=self.page.locator('#advancedFavoritesV2')
        expect(favorites).to_be_visible(); expect(favorites).to_contain_text('★ Συχνά')
        fav=favorites.locator('[data-select=graded_return_to_sport]')
        expect(fav).to_have_attribute('aria-pressed','false')
        self.assertEqual(self.page.locator('#referralText').inner_text(),before)
        self.page.screenshot(path=str(ARTIFACTS/'favorites-v2.png'), full_page=True)
        fav.click(); expect(fav).to_have_attribute('aria-pressed','true')
        favorites.locator('[data-favorite-v2][data-favorite-item=graded_return_to_sport]').click()
        expect(favorites).to_be_hidden()
        expect(self.page.locator('#advanced [data-select=graded_return_to_sport]').first).to_have_attribute('aria-pressed','true')
        self.assertEqual(self.page.evaluate('localStorage.length'),0)
        self.assertEqual(self.page.evaluate('sessionStorage.length'),0)
        self.assertEqual(self.page.locator('button').filter(has_text='Απόκρυψη').count(),0)

    def test_reset_and_bfcache_clear_ephemeral_favorites(self):
        self.page.locator('#advancedToggle').click(); self.page.locator('#advanced summary').filter(has_text='Στόχοι').click()
        self.page.locator('#advanced [data-favorite-v2][data-favorite-category=goals][data-favorite-item=graded_return_to_sport]').last.click()
        expect(self.page.locator('#advancedFavoritesV2')).to_be_visible()
        self.page.locator('#reset').click(); self.page.locator('[data-confirm-reset]').click()
        self.page.locator('#assertion').click(); self.page.locator('[data-side=right]').click(); self.page.locator('#advancedToggle').click()
        expect(self.page.locator('#advancedFavoritesV2')).to_be_hidden()
        self.page.locator('#advanced summary').filter(has_text='Στόχοι').click()
        self.page.locator('#advanced [data-favorite-v2][data-favorite-category=goals][data-favorite-item=graded_return_to_sport]').last.click()
        expect(self.page.locator('#advancedFavoritesV2')).to_be_visible()
        self.page.evaluate("window.dispatchEvent(new PageTransitionEvent('pagehide',{persisted:true}));window.dispatchEvent(new PageTransitionEvent('pageshow',{persisted:true}));")
        self.page.locator('#assertion').click(); self.page.locator('[data-side=right]').click(); self.page.locator('#advancedToggle').click()
        expect(self.page.locator('#advancedFavoritesV2')).to_be_hidden()
        self.assertEqual(self.page.evaluate('localStorage.length'),0)
        self.assertEqual(self.page.evaluate('sessionStorage.length'),0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
