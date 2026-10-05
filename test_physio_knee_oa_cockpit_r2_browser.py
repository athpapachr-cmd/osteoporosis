"""Production Cockpit smoke for the R2 presentation and CY_GESY evidence."""
from __future__ import annotations

import os
import socket
import threading
import unittest
from unittest.mock import patch

import uvicorn
from playwright.sync_api import expect, sync_playwright

KEY="physio-r2-browser-test-key"


class R2CockpitBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=patch.dict(os.environ,{"CLINICAL_DATA_KEY":KEY,"PHYSIO_REFERRAL_JURISDICTION_PROFILE":"CY_GESY"},clear=False)
        cls.env.start()
        from main import app
        cls.sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM);cls.sock.bind(("127.0.0.1",0));cls.sock.listen(128)
        cls.origin=f"http://127.0.0.1:{cls.sock.getsockname()[1]}"
        cls.server=uvicorn.Server(uvicorn.Config(app,log_level="error",access_log=False))
        cls.thread=threading.Thread(target=cls.server.run,kwargs={"sockets":[cls.sock]},daemon=True);cls.thread.start()
        for _ in range(200):
            if cls.server.started:break
            threading.Event().wait(.025)
        if not cls.server.started:raise RuntimeError("uvicorn_test_server_not_started")
        cls.playwright=sync_playwright().start();cls.browser=cls.playwright.chromium.launch(headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close();cls.playwright.stop();cls.server.should_exit=True;cls.thread.join(timeout=5)
        cls.sock.close();cls.env.stop()

    def setUp(self):
        self.context=self.browser.new_context(viewport={"width":390,"height":844},reduced_motion="reduce",
            extra_http_headers={"X-Clinical-Key":KEY},permissions=["clipboard-read","clipboard-write"])
        self.page=self.context.new_page();self.errors=[]
        self.page.on("pageerror",lambda error:self.errors.append(str(error)))
        response=self.page.goto(self.origin+"/clinical/clinic-utilities/physio-referral")
        self.assertEqual(response.status,200)
        expect(self.page.locator("#r2ClinicalPicture")).to_be_visible()

    def tearDown(self):
        self.assertEqual(self.errors,[]);self.context.close()

    def ready(self):
        self.page.locator("#assertion").click();self.page.locator('[data-side="right"]').click()
        expect(self.page.locator("#copy")).to_be_enabled()

    def test_live_r2_sections_copy_and_no_storage(self):
        self.ready()
        for section in ("r2ClinicalPicture","r2Functionality","r2Examination","r2ProposedPlan"):
            expect(self.page.locator("#"+section)).to_be_visible()
        expect(self.page.locator('#r2ClinicalPicture [data-r2-id="swelling"]')).to_have_attribute("aria-pressed","false")
        self.assertFalse(self.page.locator("#advancedToggle").is_visible())
        self.page.locator("#mobileDock [data-copy]").click()
        copied=self.page.evaluate("navigator.clipboard.readText()")
        self.assertIn("δεξιού γόνατος",copied)
        self.assertNotIn("ΔΟΚΙΜΑΣΤΙΚΟ",copied)
        self.assertEqual(self.page.evaluate("localStorage.length+sessionStorage.length"),0)
        self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth<=innerWidth"))

    def test_review_cue_blocks_export_until_disposition(self):
        self.ready()
        for id in ("hot_swollen_joint","acute_new_severe_pain","major_weight_bearing_or_movement_difficulty","acute_or_rapid_deterioration"):
            self.page.locator(f'[data-r2-kind="observation"][data-r2-id="{id}"]').click()
        expect(self.page.locator("#r2ReviewCues")).to_be_visible()
        expect(self.page.locator("#copy")).to_be_disabled()
        self.page.locator('[data-r2-decision="defer"]').click()
        expect(self.page.locator("#copy")).to_be_disabled()
        self.page.locator('[data-r2-decision="continue"]').click()
        expect(self.page.locator("#copy")).to_be_enabled()

    def test_gesy_context_remains_in_evidence(self):
        self.ready()
        self.page.locator('#plan [data-evidence="therapeutic_exercise"]').click()
        expect(self.page.locator("#sheet")).to_be_visible()
        expect(self.page.locator("#sheet [data-jurisdiction-position]")).to_contain_text("Κύπρος · ΓεΣΥ")

    def test_additional_plan_owner_and_restriction_draft(self):
        self.ready();self.page.locator("#r2PlanAdditional summary").click()
        aid=self.page.locator('#r2PlanAdditional [data-row-item="walking_aid_assessment_and_training"] [data-select]')
        self.assertNotIn("unavailable",aid.locator(".row-title").get_attribute("class"))
        aid.click();expect(aid).to_have_attribute("aria-pressed","true")
        self.assertEqual(self.page.locator('#plan [data-row-item="walking_aid_assessment_and_training"]').count(),0)
        self.page.locator("#r2RestrictionId").select_option("weight_bearing_status")
        expect(self.page.locator("#mobileDock [data-copy]")).to_be_disabled()
        expect(self.page.locator("#r2RestrictionHint")).to_be_visible()
        self.page.locator("#r2RestrictionText").fill("Αποφυγή πλήρους φόρτισης για μία εβδομάδα.")
        expect(self.page.locator("#mobileDock [data-copy]")).to_be_enabled()


if __name__=="__main__":
    unittest.main()
