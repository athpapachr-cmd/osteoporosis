"""Actual Chromium acceptance for reviewed Knee-OA amendments."""
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

    def test_diagnosis_is_selection_not_checkbox_and_missing_state_is_specific(self):
        page = self.context.new_page(); errors=[]
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(self.origin)
        expect(page.locator('#reviewStatus')).to_have_text('Επίλεξε διάγνωση')
        expect(page.locator('#assertion')).to_have_class('diagnosis-choice required-missing')
        self.assertEqual(page.locator('#assertion .selection-mark').count(), 0)
        expect(page.locator('#diagnosisRequiredHint')).to_be_visible()
        expect(page.locator('#jurisdictionProfile')).to_contain_text('Κύπρος · ΓεΣΥ')
        expect(page.locator('#jurisdictionProfile')).to_have_attribute('data-jurisdiction','CY_GESY')
        page.locator('#assertion').click()
        expect(page.locator('#reviewStatus')).to_have_text('Επίλεξε πλευρά')
        expect(page.locator('#diagnosisRequiredHint')).to_be_hidden()
        expect(page.locator('#sideRequiredHint')).to_be_visible()
        page.locator('[data-side=left]').click()
        expect(page.locator('#copy')).to_be_enabled()
        expect(page.locator('#referralText')).to_contain_text('αριστερού γόνατος')
        expect(page.locator('#referralText')).not_to_contain_text('Διάγνωση επιβεβαιωμένη')
        self.assertEqual(page.locator('[data-functional-baseline]').count(), 0)
        self.assertEqual(errors, [])
        page.close()

    def test_pain_location_progressive_disclosure_and_pes_anserine(self):
        expect(self.page.locator('[data-smart-root=pain]')).to_be_hidden()
        self.page.locator('[data-finding=pain]').click()
        expect(self.page.locator('[data-smart-root=pain]')).to_be_visible()
        self.page.locator('[data-smart-root=pain] [data-smart-summary]').click()
        self.page.locator('[data-q-pain=medial_joint_line]').click()
        self.page.locator('[data-q-pain=pes_anserine_region]').click()
        expect(self.page.locator('[data-smart-root=pain] [data-smart-summary]')).to_contain_text('Έσω μεσάρθρια')
        expect(self.page.locator('[data-smart-root=pain] [data-smart-summary]')).to_contain_text('Χήνειος πόδας')
        expect(self.page.locator('#referralText')).to_contain_text('έσω μεσάρθρια περιοχή')
        expect(self.page.locator('#referralText')).to_contain_text('περιοχή του χηνείου ποδός')
        expect(self.page.locator('#referralText')).not_to_contain_text('θυλακ')

    def test_qualifier_switching_collapses_prior_group_and_keeps_aria_consistent(self):
        self.page.locator('[data-finding=pain]').click()
        pain_summary=self.page.locator('[data-smart-root=pain] [data-smart-summary]')
        pain_summary.click(); expect(pain_summary).to_have_attribute('aria-expanded','true')
        self.page.locator('[data-q-pain=medial_joint_line]').click()
        self.page.locator('[data-phenotype=stiffness_symptom]').click()
        stiffness_summary=self.page.locator('[data-smart-root=stiffness] [data-smart-summary]')
        stiffness_summary.click()
        expect(stiffness_summary).to_have_attribute('aria-expanded','true')
        expect(pain_summary).to_have_attribute('aria-expanded','false')
        expect(self.page.locator('[data-smart-root=pain] [data-smart-panel]')).to_be_hidden()
        self.page.locator('[data-finding=pain]').click()
        expect(self.page.locator('[data-smart-root=pain]')).to_be_hidden()
        expect(pain_summary).to_have_attribute('aria-expanded','false')

    def test_stiffness_over_30_creates_review_clue_not_block(self):
        self.page.locator('[data-phenotype=stiffness_symptom]').click()
        self.page.locator('[data-smart-root=stiffness] [data-smart-summary]').click()
        self.page.locator('[data-q-stiffness=morning]').click()
        expect(self.page.locator('#morningDuration')).to_be_visible()
        self.page.locator('[data-q-duration=gt_30]').click()
        expect(self.page.locator('#stiffnessReviewClue')).to_be_visible()
        expect(self.page.locator('#reviewStatus')).to_contain_text('σημείο για έλεγχο')
        expect(self.page.locator('#copy')).to_be_enabled()
        expect(self.page.locator('#referralText')).to_contain_text('πρωινή δυσκαμψία άνω των 30 λεπτών')
        self.page.locator('#reviewStatus').click()
        expect(self.page.locator('#sheetBody')).to_contain_text('NICE NG226 · 2022')
        expect(self.page.locator('#sheetBody')).to_contain_text('χωρίς αυτόματη αλλαγή θεραπείας')

    def test_quadriceps_exam_atrophy_refines_summary_and_referral(self):
        self.page.locator('[data-phenotype=weakness_symptom_or_context]').click()
        self.page.locator('[data-smart-root=weakness] [data-smart-summary]').click()
        self.assertEqual(self.page.locator('[data-q-weakness=quadriceps]').count(),0)
        quad=self.page.locator('[data-q-weakness=quadriceps_exam]')
        expect(quad).to_have_text('Τετρακέφαλος στην εξέταση')
        quad.click()
        self.page.locator('[data-q-atrophy]').click()
        expect(self.page.locator('#atrophyLocation')).to_be_visible()
        self.page.locator('[data-q-atrophy-location=quadriceps]').click()
        expect(self.page.locator('[data-smart-root=weakness] [data-smart-summary]')).to_contain_text('Τετρακέφαλος · εξέταση')
        expect(self.page.locator('[data-smart-root=weakness] [data-smart-summary]')).to_contain_text('Ατροφία')
        expect(self.page.locator('#referralText')).to_contain_text('αδυναμία τετρακεφάλου με εμφανή ατροφία τετρακεφάλου')
        expect(self.page.locator('#referralText')).to_contain_text('έμφαση στον τετρακέφαλο')
        expect(self.page.locator('#referralText')).to_contain_text('ενδεικτικές προτεραιότητες')

    def test_fixed_flexion_is_advanced_passive_exam_not_stiffness(self):
        self.page.locator('#advancedToggle').click()
        self.page.locator('#examQualifierGroup summary').click()
        expect(self.page.locator('[data-q-ffd]')).to_have_text('Παθητικό έλλειμμα έκτασης')
        self.page.locator('[data-q-ffd]').click()
        expect(self.page.locator('#ffdDegreesWrap')).to_be_visible()
        self.page.locator('#ffdDegrees').fill('10')
        expect(self.page.locator('#ffdDegrees')).to_have_attribute('min','1')
        expect(self.page.locator('#referralText')).to_contain_text('παθητικό έλλειμμα έκτασης 10°')
        expect(self.page.locator('#referralText')).not_to_contain_text('μόνιμο')
        expect(self.page.locator('#referralText')).not_to_contain_text('fixed flexion deformity')
        expect(self.page.locator('#referralText')).not_to_contain_text('δυσκαμψία')
        expect(self.page.locator('#suggestions')).to_contain_text('Κινητικότητα')
        self.assertEqual(self.page.locator('#plan [data-select=mobility_exercise_when_restricted][aria-pressed=true]').count(), 0)

    def test_supported_evidence_is_progressive_but_mixed_guidance_stays_visible(self):
        self.page.locator('#plan [data-evidence=therapeutic_exercise]').click()
        expect(self.page.locator('#sheet .evidence-deep')).to_have_count(1)
        expect(self.page.locator('#sheet .evidence-deep')).not_to_have_attribute('open','')
        self.page.keyboard.press('Escape')
        self.page.locator('#advancedToggle').click(); self.page.locator('#advanced summary').filter(has_text='Συμπληρωματικά').click()
        self.page.locator('#advanced [data-select=acupuncture]').click()
        self.page.locator('#plan [data-evidence=acupuncture]').click()
        expect(self.page.locator('#sheet .evidence-deep')).to_have_count(0)
        expect(self.page.locator('#sheet .source-position')).to_have_count(5)
        expect(self.page.locator('#sheetBody')).to_contain_text('Συστήνει να μην προσφέρεται βελονισμός')
        expect(self.page.locator('#sheetBody')).to_contain_text('Ανεπαρκή δεδομένα')

    def test_pes_anserine_tenderness_and_mobile_reflow(self):
        self.page.locator('#advancedToggle').click(); self.page.locator('#examQualifierGroup summary').click()
        self.page.locator('[data-q-tenderness=pes_anserine_region]').click()
        expect(self.page.locator('#referralText')).to_contain_text('εντοπισμένη ευαισθησία')
        expect(self.page.locator('#referralText')).to_contain_text('χηνείου ποδός')
        expect(self.page.locator('#referralText')).not_to_contain_text('θυλακ')
        for width in [320, 390, 800]:
            self.page.set_viewport_size({"width": width, "height": 900})
            self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth<=innerWidth'), width)

    def test_parent_deselect_clears_hidden_qualifier_meaning(self):
        self.page.locator('[data-finding=pain]').click(); self.page.locator('[data-smart-root=pain] [data-smart-summary]').click()
        self.page.locator('[data-q-pain=pes_anserine_region]').click()
        expect(self.page.locator('#referralText')).to_contain_text('χηνείου ποδός')
        self.page.locator('[data-finding=pain]').click()
        expect(self.page.locator('[data-smart-root=pain]')).to_be_hidden()
        expect(self.page.locator('#referralText')).not_to_contain_text('χηνείου ποδός')
        expect(self.page.locator('[data-smart-root=pain] [data-smart-summary]')).to_have_attribute('aria-expanded','false')
        self.page.locator('[data-finding=pain]').click()
        expect(self.page.locator('[data-smart-root=pain] [data-smart-summary]')).to_have_text('Εντόπιση ›')


if __name__ == '__main__':
    unittest.main(verbosity=2)
