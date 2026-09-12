from __future__ import annotations

import os
import socket
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import uvicorn
from playwright.sync_api import expect, sync_playwright


KEY = "physio-knee-oa-browser-test-key"
ARTIFACTS = Path("artifacts/knee-oa-cockpit")


class KneeOACockpitBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        cls.env = patch.dict(os.environ, {"CLINICAL_DATA_KEY": KEY}, clear=False)
        cls.env.start()
        from main import app

        cls.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cls.sock.bind(("127.0.0.1", 0))
        cls.sock.listen(128)
        port = cls.sock.getsockname()[1]
        cls.origin = f"http://127.0.0.1:{port}"
        config = uvicorn.Config(app, log_level="error", access_log=False)
        cls.server = uvicorn.Server(config)
        cls.thread = threading.Thread(target=cls.server.run, kwargs={"sockets": [cls.sock]}, daemon=True)
        cls.thread.start()
        for _ in range(200):
            if cls.server.started:
                break
            threading.Event().wait(0.025)
        if not cls.server.started:
            raise RuntimeError("uvicorn_test_server_not_started")

        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.server.should_exit = True
        cls.thread.join(timeout=5)
        try:
            cls.sock.close()
        except OSError:
            pass
        cls.env.stop()

    def setUp(self):
        self.context = self.browser.new_context(
            viewport={"width": 1280, "height": 1000},
            reduced_motion="reduce",
            permissions=["clipboard-read", "clipboard-write"],
            extra_http_headers={"X-Clinical-Key": KEY},
        )
        self.page = self.context.new_page()
        self.errors = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        response = self.page.goto(self.origin + "/clinical/clinic-utilities/physio-referral")
        self.assertEqual(response.status, 200)
        expect(self.page.locator("#reviewStatus")).not_to_have_text("Η παραπομπή ενημερώνεται")

    def tearDown(self):
        self.assertEqual(self.errors, [])
        self.context.close()

    def ready(self):
        self.page.locator("#assertion").click()
        self.page.locator("[data-side=right]").click()
        expect(self.page.locator("#copy")).to_be_enabled()
        expect(self.page.locator("#referralText")).to_contain_text("δεξιού γόνατος")

    def test_live_production_surface_has_no_generate_or_synthetic_stamp(self):
        self.assertEqual(self.page.get_by_text("Δημιουργία παραπεμπτικού", exact=True).count(), 0)
        self.assertEqual(self.page.get_by_text("ΔΟΚΙΜΑΣΤΙΚΟ ΚΕΙΜΕΝΟ", exact=True).count(), 0)
        self.ready()
        expect(self.page.locator("#directEditV2")).to_be_visible()
        self.page.locator("#advancedToggle").click()
        expect(self.page.locator("#advanced .v3-category-row")).to_have_count(6)
        expect(self.page.locator("#v3Favorites")).to_contain_text("★ Συχνά")
        self.page.screenshot(path=str(ARTIFACTS / "cockpit-knee-oa-desktop.png"), full_page=True)

    def test_copy_exports_clean_referral_without_prototype_prefix(self):
        self.ready()
        self.page.locator("#copy").click()
        copied = self.page.evaluate("navigator.clipboard.readText()")
        self.assertIn("δεξιού γόνατος", copied)
        self.assertFalse(copied.startswith("ΔΟΚΙΜΑΣΤΙΚΟ"), copied[:80])
        self.assertNotIn("ΟΧΙ ΓΙΑ ΚΛΙΝΙΚΗ ΧΡΗΣΗ", copied)

    def test_more_scan_first_and_context_shortcut_do_not_auto_select(self):
        self.ready()
        self.page.locator("[data-phenotype=weakness_symptom_or_context]").click()
        expect(self.page.locator("#referralText")).to_contain_text("μυϊκή αδυναμία")
        before = self.page.locator("#referralText").inner_text()
        self.page.locator("#advancedToggle").click()
        relevant = self.page.locator("#v3Relevant")
        expect(relevant).to_be_visible()
        expect(relevant).to_contain_text("Αδυναμία στην εξέταση")
        self.assertEqual(self.page.locator("[data-q-weakness][aria-pressed=true]").count(), 0)
        relevant.locator("[data-v3-focus=weakness]").click()
        expect(self.page.locator("#sheet [data-q-weakness=objective]")).to_have_attribute("aria-pressed", "false")
        self.assertEqual(self.page.locator("#referralText").inner_text(), before)

    def test_manual_edit_reconciliation_remains_fail_closed(self):
        self.ready()
        self.page.locator("#directEditV2").click()
        self.page.locator("#manualText").fill("Χειροκίνητο κείμενο παραπομπής.")
        self.page.locator("[data-confirm-manual]").click()
        expect(self.page.locator("#referralText")).to_have_text("Χειροκίνητο κείμενο παραπομπής.")
        self.page.locator("[data-side=left]").click()
        expect(self.page.locator("#manualReconcile")).to_be_visible()
        expect(self.page.locator("#copy")).to_be_disabled()
        expect(self.page.locator("#referralText")).to_have_text("Χειροκίνητο κείμενο παραπομπής.")

    def test_no_browser_storage_and_mobile_reflow(self):
        self.ready()
        self.assertEqual(self.page.evaluate("localStorage.length"), 0)
        self.assertEqual(self.page.evaluate("sessionStorage.length"), 0)
        self.page.set_viewport_size({"width": 390, "height": 900})
        self.page.locator("#advancedToggle").click()
        self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth<=innerWidth"))
        expect(self.page.locator("#advanced .v3-category-row")).to_have_count(6)
        self.page.screenshot(path=str(ARTIFACTS / "cockpit-knee-oa-mobile.png"), full_page=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
