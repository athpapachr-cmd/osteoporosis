"""R2 information architecture acceptance using the five Product Owner real-use cases."""
from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from clinic_utilities.physio_referral_product.prototype import server as p


class R2BrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=p.ThreadingHTTPServer(("127.0.0.1",0),p.Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.origin=f"http://127.0.0.1:{cls.server.server_port}"
        cls.playwright=sync_playwright().start();cls.browser=cls.playwright.chromium.launch(headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close();cls.playwright.stop();cls.server.shutdown();cls.server.server_close();cls.thread.join()

    def setUp(self):
        self.context=self.browser.new_context(viewport={"width":390,"height":844},reduced_motion="reduce")
        self.page=self.context.new_page();self.errors=[]
        self.page.on("pageerror",lambda e:self.errors.append(str(e)))
        self.page.goto(self.origin)
        expect(self.page.locator("#r2ClinicalPicture")).to_be_visible()

    def tearDown(self):
        self.assertEqual(self.errors,[])
        self.context.close()

    def choose(self,kind,id):
        self.page.locator(f'[data-r2-kind="{kind}"][data-r2-id="{id}"]').click()

    def ready(self,side="right"):
        self.page.locator("#assertion").click();self.page.locator(f'[data-side="{side}"]').click()
        expect(self.page.locator("#copy")).to_be_enabled()

    def test_case1_routine_right_oa_and_single_function_owner(self):
        self.ready();self.choose("finding","pain")
        self.choose("function","walking_tolerance");self.choose("function","stairs")
        expect(self.page.locator("#referralText")).to_contain_text("βάδιση")
        expect(self.page.locator('#r2ClinicalPicture [data-r2-id="stairs"]')).to_have_count(0)
        expect(self.page.locator('#r2Functionality [data-r2-id="stairs"]')).to_have_count(1)
        self.choose("function","stairs")
        expect(self.page.locator('#r2Functionality [data-r2-id="stairs"]')).to_have_attribute("aria-pressed","false")
        self.assertFalse(self.page.locator("#advancedToggle").is_visible())

    def test_case2_bilateral_stiffness_and_chronicity(self):
        self.ready("bilateral");self.choose("finding","pain");self.choose("phenotype","stiffness_symptom")
        expect(self.page.locator("#r2StiffnessDetail")).to_be_visible()
        self.choose("stiffness","after_inactivity");self.choose("stiffness","morning")
        self.choose("duration","le_30")
        self.page.locator("#r2DurationValue").fill("2");self.page.locator("#r2DurationValue").blur()
        self.page.locator("#r2DurationUnit").select_option("years")
        self.choose("function","sit_to_stand")
        expect(self.page.locator("#referralText")).to_contain_text("2 ετών")
        expect(self.page.locator("#referralText")).to_contain_text("έγερση")
        self.assertFalse(self.page.locator("#sheet").evaluate("el=>el.open"))

    def test_case3_exam_details_without_retired_structured_fields(self):
        self.ready("left");self.choose("finding","pain")
        self.choose("function","sit_to_stand");self.choose("function","stairs")
        self.choose("function","walking_tolerance")
        self.choose("weakness","knee_extension_exam")
        self.choose("qualifier","active_flexion_restricted")
        expect(self.page.locator("#referralText")).to_contain_text("τετρακεφάλου")
        expect(self.page.locator("#referralText")).to_contain_text("περιορισμό ενεργητικής κάμψης")
        for retired in ("crepitus","tenderness","effusion","joint_line_pain"):
            self.assertEqual(self.page.locator(f'#r2Examination [data-r2-id="{retired}"]').count(),0)
        self.page.locator("#r2PlanAdditional summary").click()
        aid=self.page.locator('#r2PlanAdditional [data-row-item="walking_aid_assessment_and_training"] [data-select]')
        expect(aid).to_have_attribute("aria-pressed","false")
        self.assertNotIn("unavailable",aid.locator(".row-title").get_attribute("class"))
        aid.click()
        expect(aid).to_have_attribute("aria-pressed","true")
        self.assertEqual(self.page.locator('#plan [data-row-item="walking_aid_assessment_and_training"]').count(),0)
        expect(self.page.locator("#referralText")).to_contain_text("βοηθήματος βάδισης")
        aid.click()
        expect(aid).to_have_attribute("aria-pressed","false")
        expect(self.page.locator("#referralText")).not_to_contain_text("βοηθήματος βάδισης")

    def test_case4_rapid_worsening_without_diagnosis_inference(self):
        self.ready();self.choose("finding","pain")
        self.choose("observation","rapid_worsening_or_deformity")
        self.choose("function","walking_tolerance");self.choose("function","stairs")
        expect(self.page.locator("#copy")).to_be_enabled()
        expect(self.page.locator("#r2ReviewCues")).to_be_hidden()
        self.assertNotIn("SIFK",self.page.locator("#referralText").inner_text())

    def test_case5_acute_joint_and_swelling_boundary(self):
        self.ready("left");self.choose("finding","swelling")
        expect(self.page.locator("#copy")).to_be_enabled()
        expect(self.page.locator("#r2ReviewCues")).to_be_hidden()
        for id in ("hot_swollen_joint","acute_new_severe_pain","major_weight_bearing_or_movement_difficulty","acute_or_rapid_deterioration"):
            self.choose("observation",id)
        expect(self.page.locator("#r2ReviewCues")).to_be_visible()
        expect(self.page.locator("#copy")).to_be_disabled()
        self.page.locator('[data-r2-decision="defer"]').click()
        expect(self.page.locator("#copy")).to_be_disabled()
        expect(self.page.locator('[data-r2-decision="defer"]')).to_have_attribute("aria-pressed","true")
        self.page.locator('[data-r2-decision="continue"]').click()
        expect(self.page.locator("#copy")).to_be_enabled()
        self.choose("safety","infection_or_septic_joint_concern")
        expect(self.page.locator("#copy")).to_be_disabled()

    def test_additional_adjunct_hierarchy_and_incomplete_restriction(self):
        self.ready();self.page.locator("#r2PlanAdditional summary").click()
        adjunct=self.page.locator('#r2PlanAdditional [data-row-item="manual_therapy"] [data-select]')
        expect(self.page.locator('#r2PlanAdditional [data-row-item="manual_therapy"] [data-cue="guideline_conflict_or_mixed"]')).to_have_count(1)
        adjunct.click()
        expect(adjunct).to_have_attribute("aria-pressed","true")
        self.assertEqual(self.page.locator('#plan [data-row-item="manual_therapy"]').count(),0)
        expect(self.page.locator("#r2PlanSelectionSummary")).to_contain_text("Συμπληρωματικές επιλογές")
        self.page.locator("#r2RestrictionId").select_option("weight_bearing_status")
        expect(self.page.locator("#r2RestrictionHint")).to_be_visible()
        expect(self.page.locator("#copy")).to_be_disabled()
        self.assertNotIn("τοπικό prototype",self.page.locator("#reviewStatus").inner_text())
        self.page.locator("#r2RestrictionText").fill("Αποφυγή πλήρους φόρτισης για μία εβδομάδα.")
        expect(self.page.locator("#copy")).to_be_enabled()
        expect(self.page.locator("#referralText")).to_contain_text("Αποφυγή πλήρους φόρτισης")
        self.page.locator("#r2RestrictionText").fill("")
        expect(self.page.locator("#copy")).to_be_disabled()


if __name__=="__main__":
    unittest.main()
