"""Chromium acceptance for Knee-OA v5 post-use interaction corrections."""
from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from clinic_utilities.physio_referral_product.prototype import server as p


class V5BrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = p.ThreadingHTTPServer(("127.0.0.1", 0), p.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.origin = f"http://127.0.0.1:{cls.server.server_port}"
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close(); cls.playwright.stop()
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join()

    def setUp(self):
        self.context = self.browser.new_context(viewport={"width": 1280, "height": 950}, reduced_motion="reduce")
        self.page = self.context.new_page(); self.errors=[]
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.goto(self.origin)
        expect(self.page.locator('#reviewStatus')).not_to_have_text('Η παραπομπή ενημερώνεται')
        self.page.locator('#assertion').click(); self.page.locator('[data-side=right]').click()
        expect(self.page.locator('#copy')).to_be_enabled()

    def tearDown(self):
        self.assertEqual(self.errors, [])
        self.context.close()

    def assert_optional_parent(self, kind):
        button=self.page.locator(f'[data-clinical-v4={kind}]')
        expect(button).to_have_attribute('aria-pressed','false')
        button.click()
        expect(button).to_have_attribute('aria-pressed','true')
        self.assertFalse(self.page.locator('#sheet').evaluate('el=>el.open'))
        button.click()
        expect(self.page.locator('#sheet')).to_be_visible()
        expect(self.page.locator('#sheetTitle')).to_have_text({'pain':'Πόνος','stiffness':'Δυσκαμψία','weakness':'Αδυναμία'}[kind])
        self.page.locator('#closeSheet').click()

    def test_first_tap_selects_symptom_second_tap_refines(self):
        for kind in ('pain','stiffness','weakness'):
            with self.subTest(kind=kind):
                self.assert_optional_parent(kind)

    def test_generic_symptom_selection_needs_no_qualifier(self):
        self.page.locator('[data-clinical-v4=pain]').click()
        self.page.locator('[data-clinical-v4=stiffness]').click()
        self.page.locator('[data-clinical-v4=weakness]').click()
        expect(self.page.locator('#copy')).to_be_enabled()
        expect(self.page.locator('#referralText')).to_contain_text('πόνο')
        expect(self.page.locator('#referralText')).to_contain_text('δυσκαμψία')
        expect(self.page.locator('#referralText')).to_contain_text('μυϊκή αδυναμία')
        self.assertFalse(self.page.locator('#sheet').evaluate('el=>el.open'))

    def test_function_still_opens_chooser_on_first_tap(self):
        self.page.locator('[data-clinical-v4=function]').click()
        expect(self.page.locator('#sheet')).to_be_visible()
        expect(self.page.locator('#sheetTitle')).to_have_text('Λειτουργικότητα')
        expect(self.page.locator('#sheet [data-function]')).to_have_count(4)

    def test_weakness_refinement_has_three_unambiguous_exam_options(self):
        self.page.locator('[data-clinical-v4=weakness]').click()
        self.page.locator('[data-clinical-v4=weakness]').click()
        expect(self.page.locator('#sheet')).to_be_visible()
        expect(self.page.get_by_role('button', name='Μυϊκή αδυναμία στην εξέταση', exact=True)).to_have_count(1)
        expect(self.page.get_by_role('button', name='Αδυναμία τετρακεφάλου στην εξέταση', exact=True)).to_have_count(1)
        expect(self.page.get_by_role('button', name='Ατροφία τετρακεφάλου', exact=True)).to_have_count(1)
        self.assertEqual(self.page.get_by_role('button', name='Τετρακέφαλος', exact=True).count(), 0)
        self.assertEqual(self.page.get_by_role('button', name='Περιαρθρικά', exact=True).count(), 0)
        self.page.get_by_role('button', name='Ατροφία τετρακεφάλου', exact=True).click()
        expect(self.page.locator('[data-clinical-v4=weakness] [data-clinical-count-v4]')).to_have_text('· 1')
        expect(self.page.locator('#referralText')).to_contain_text('εμφανή ατροφία τετρακεφάλου')

    def test_more_exam_does_not_reintroduce_ambiguous_atrophy_labels(self):
        self.page.locator('[data-clinical-v4=weakness]').click()
        self.page.locator('#advancedToggle').click()
        self.page.locator('#advanced [data-v3-category=exam]').click()
        expect(self.page.locator('#sheet')).to_be_visible()
        expect(self.page.get_by_role('button', name='Μυϊκή αδυναμία στην εξέταση', exact=True)).to_have_count(1)
        expect(self.page.get_by_role('button', name='Αδυναμία τετρακεφάλου στην εξέταση', exact=True)).to_have_count(1)
        expect(self.page.get_by_role('button', name='Ατροφία τετρακεφάλου', exact=True)).to_have_count(1)
        self.assertEqual(self.page.get_by_role('button', name='Τετρακέφαλος', exact=True).count(), 0)
        self.assertEqual(self.page.get_by_role('button', name='Περιαρθρικά', exact=True).count(), 0)

    def test_refined_pain_output_has_no_legacy_joint_line_tail(self):
        self.page.locator('[data-clinical-v4=pain]').click()
        self.page.locator('[data-clinical-v4=pain]').click()
        self.page.locator('#sheet [data-q-pain=medial_joint_line]').click()
        self.page.locator('#sheet [data-q-pain=pes_anserine_region]').click()
        expect(self.page.locator('#referralText')).to_contain_text('πόνο στην έσω μεσάρθρια περιοχή και στην περιοχή του χηνείου ποδός')
        expect(self.page.locator('#referralText')).not_to_contain_text('χηνείου ποδός στη μεσάρθρια γραμμή')
        expect(self.page.locator('#referralText')).not_to_contain_text('κυρίως')


if __name__ == '__main__':
    unittest.main(verbosity=2)
