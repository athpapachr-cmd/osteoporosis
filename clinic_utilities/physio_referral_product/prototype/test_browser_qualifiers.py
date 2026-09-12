"""Actual Chromium acceptance for reviewed Knee-OA qualifier/clinical-picture UX."""
from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from clinic_utilities.physio_referral_product.prototype import server as p


class QualifierBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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

    def open_v3_category(self, category):
        if self.page.locator('#advancedToggle').get_attribute('aria-expanded') != 'true':
            self.page.locator('#advancedToggle').click()
        self.page.locator(f'#advanced [data-v3-category={category}]').click()
        expect(self.page.locator('#sheet')).to_be_visible()

    def open_clinical(self, kind):
        self.page.locator(f'[data-clinical-v4={kind}]').click()
        expect(self.page.locator('#sheet')).to_be_visible()
        expect(self.page.locator('#sheetTitle')).to_have_text({'pain':'Πόνος','stiffness':'Δυσκαμψία','weakness':'Αδυναμία','function':'Λειτουργικότητα'}[kind])

    def test_diagnosis_is_selection_not_checkbox_and_missing_state_is_specific(self):
        page = self.context.new_page(); errors=[]
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(self.origin)
        # Specific inline validation owns the unresolved-field wording; the
        # global review status may remain intentionally generic.
        expect(page.locator('#diagnosisRequiredHint')).to_be_visible()
        expect(page.locator('#diagnosisRequiredHint')).to_have_text('Απαιτείται επιλογή διάγνωσης.')
        expect(page.locator('#assertion')).to_have_class('diagnosis-choice required-missing')
        self.assertEqual(page.locator('#assertion .selection-mark').count(), 0)
        expect(page.locator('#jurisdictionProfile')).to_contain_text('Κύπρος · ΓεΣΥ')
        page.locator('#assertion').click()
        expect(page.locator('#diagnosisRequiredHint')).to_be_hidden()
        expect(page.locator('#sideRequiredHint')).to_be_visible()
        expect(page.locator('#lateralitySection')).to_have_class('required-missing')
        page.locator('[data-side=left]').click(); expect(page.locator('#copy')).to_be_enabled()
        expect(page.locator('#referralText')).to_contain_text('αριστερού γόνατος')
        self.assertEqual(errors, []); page.close()

    def test_compact_grid_and_pain_sheet_keep_parent_child_link_obvious(self):
        expect(self.page.locator('#phenotype [data-clinical-v4]')).to_have_count(4)
        self.open_clinical('pain')
        self.page.locator('#sheet [data-q-pain=medial_joint_line]').click()
        self.page.locator('#sheet [data-q-pain=pes_anserine_region]').click()
        expect(self.page.locator('[data-clinical-v4=pain] [data-clinical-count-v4]')).to_have_text('· 2')
        expect(self.page.locator('#referralText')).to_contain_text('έσω μεσάρθρια περιοχή')
        expect(self.page.locator('#referralText')).to_contain_text('περιοχή του χηνείου ποδός')
        expect(self.page.locator('#referralText')).not_to_contain_text('κυρίως')
        expect(self.page.locator('#referralText')).not_to_contain_text('θυλακ')

    def test_stiffness_sheet_nested_duration_and_review_clue(self):
        self.open_clinical('stiffness')
        self.page.locator('#sheet [data-q-stiffness=morning]').click()
        expect(self.page.locator('#v4MorningDuration')).to_be_visible()
        self.page.locator('#sheet [data-q-duration=gt_30]').click()
        expect(self.page.locator('#v4StiffnessClue')).to_be_visible()
        expect(self.page.locator('[data-clinical-v4=stiffness] [data-clinical-count-v4]')).to_have_text('· 2')
        expect(self.page.locator('#referralText')).to_contain_text('πρωινή δυσκαμψία άνω των 30 λεπτών')
        self.page.locator('#closeSheet').click()
        expect(self.page.locator('#reviewStatus')).to_contain_text('σημείο για έλεγχο')
        self.page.locator('#reviewStatus').click()
        expect(self.page.locator('#sheetBody')).to_contain_text('NICE NG226 · 2022')

    def test_quadriceps_exam_atrophy_stays_specific_and_human(self):
        self.open_clinical('weakness')
        quad=self.page.locator('#sheet [data-q-weakness=quadriceps_exam]')
        expect(quad).to_have_text('Τετρακέφαλος στην εξέταση'); quad.click()
        self.page.locator('#sheet [data-q-atrophy]').click()
        expect(self.page.locator('#v4AtrophyLocation')).to_be_visible()
        self.page.locator('#sheet [data-q-atrophy-location=quadriceps]').click()
        expect(self.page.locator('[data-clinical-v4=weakness] [data-clinical-count-v4]')).to_have_text('· 2')
        expect(self.page.locator('#referralText')).to_contain_text('αδυναμία του τετρακεφάλου κατά την εξέταση')
        expect(self.page.locator('#referralText')).to_contain_text('εμφανή ατροφία τετρακεφάλου')
        expect(self.page.locator('#referralText')).to_contain_text('με έμφαση σε')
        expect(self.page.locator('#referralText')).not_to_contain_text('ενδεικτικές προτεραιότητες')

    def test_function_sheet_and_explicit_remove(self):
        self.open_clinical('function')
        self.page.locator('#sheet [data-function=walking_tolerance]').click()
        expect(self.page.locator('[data-clinical-v4=function] [data-clinical-count-v4]')).to_have_text('· 1')
        expect(self.page.locator('#referralText')).to_contain_text('δυσχέρεια στη βάδιση')
        self.page.locator('#sheet [data-clinical-remove-v4=function]').click()
        expect(self.page.locator('[data-clinical-v4=function]')).to_have_attribute('aria-pressed','false')
        expect(self.page.locator('#referralText')).not_to_contain_text('δυσχέρεια στη βάδιση')

    def test_parent_remove_clears_hidden_pain_qualifier_meaning(self):
        self.open_clinical('pain')
        self.page.locator('#sheet [data-q-pain=pes_anserine_region]').click()
        expect(self.page.locator('#referralText')).to_contain_text('χηνείου ποδός')
        self.page.locator('#sheet [data-clinical-remove-v4=pain]').click()
        expect(self.page.locator('[data-clinical-v4=pain]')).to_have_attribute('aria-pressed','false')
        expect(self.page.locator('#referralText')).not_to_contain_text('χηνείου ποδός')
        self.open_clinical('pain')
        expect(self.page.locator('#sheet [data-q-pain][aria-pressed=true]')).to_have_count(0)

    def test_fixed_flexion_is_advanced_passive_exam_not_stiffness(self):
        self.open_v3_category('exam')
        ffd=self.page.locator('#sheet [data-q-ffd]'); expect(ffd).to_contain_text('Παθητικό έλλειμμα έκτασης'); ffd.click()
        expect(self.page.locator('#v3FfdDegreesWrap')).to_be_visible(); self.page.locator('#v3FfdDegrees').fill('10')
        expect(self.page.locator('#referralText')).to_contain_text('παθητικό έλλειμμα έκτασης 10°')
        expect(self.page.locator('#referralText')).not_to_contain_text('μόνιμο')
        expect(self.page.locator('#referralText')).not_to_contain_text('δυσκαμψία')
        self.page.keyboard.press('Escape'); expect(self.page.locator('#suggestions')).to_contain_text('Κινητικότητα')

    def test_supported_evidence_is_progressive_but_mixed_guidance_stays_visible(self):
        self.page.locator('#plan [data-evidence=therapeutic_exercise]').click()
        expect(self.page.locator('#sheet .evidence-deep')).to_have_count(1); self.page.keyboard.press('Escape')
        self.open_v3_category('adjuncts'); self.page.locator('#sheet [data-select=acupuncture]').click(); self.page.keyboard.press('Escape')
        self.page.locator('#plan [data-evidence=acupuncture]').click()
        expect(self.page.locator('#sheet .evidence-deep')).to_have_count(0)
        expect(self.page.locator('#sheet .source-position')).to_have_count(5)
        expect(self.page.locator('#sheetBody')).to_contain_text('Συστήνει να μην προσφέρεται βελονισμός')

    def test_mobile_grid_is_two_columns_and_reflows(self):
        self.page.set_viewport_size({"width":390,"height":900})
        columns=self.page.evaluate("getComputedStyle(document.querySelector('.clinical-grid-v4')).gridTemplateColumns.split(' ').length")
        self.assertEqual(columns,2)
        self.open_clinical('pain')
        self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth<=innerWidth'))
        expect(self.page.locator('#sheet')).to_be_visible()


if __name__ == '__main__':
    unittest.main(verbosity=2)
