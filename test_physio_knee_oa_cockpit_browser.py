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
PROFILE_ENV = "PHYSIO_REFERRAL_JURISDICTION_PROFILE"
ARTIFACTS = Path("artifacts/knee-oa-cockpit")


class KneeOACockpitBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        cls.env = patch.dict(os.environ,{"CLINICAL_DATA_KEY": KEY, PROFILE_ENV: "CY_GESY"},clear=False)
        cls.env.start()
        from main import app
        cls.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); cls.sock.bind(("127.0.0.1", 0)); cls.sock.listen(128)
        port = cls.sock.getsockname()[1]; cls.origin = f"http://127.0.0.1:{port}"
        config = uvicorn.Config(app, log_level="error", access_log=False); cls.server = uvicorn.Server(config)
        cls.thread = threading.Thread(target=cls.server.run, kwargs={"sockets": [cls.sock]}, daemon=True); cls.thread.start()
        for _ in range(200):
            if cls.server.started: break
            threading.Event().wait(0.025)
        if not cls.server.started: raise RuntimeError("uvicorn_test_server_not_started")
        cls.playwright = sync_playwright().start(); cls.browser = cls.playwright.chromium.launch(headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close(); cls.playwright.stop(); cls.server.should_exit = True; cls.thread.join(timeout=5)
        try: cls.sock.close()
        except OSError: pass
        cls.env.stop()

    def setUp(self):
        self.context = self.browser.new_context(viewport={"width": 1280, "height": 1000}, reduced_motion="reduce",
            permissions=["clipboard-read", "clipboard-write"], extra_http_headers={"X-Clinical-Key": KEY})
        self.page = self.context.new_page(); self.errors = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        response = self.page.goto(self.origin + "/clinical/clinic-utilities/physio-referral")
        self.assertEqual(response.status, 200)
        expect(self.page.locator("#reviewStatus")).not_to_have_text("Η παραπομπή ενημερώνεται")
        expect(self.page.locator("[data-clinical-v4]")).to_have_count(4)

    def tearDown(self):
        self.assertEqual(self.errors, []); self.context.close()

    def ready(self):
        self.page.locator("#assertion").click(); self.page.locator("[data-side=right]").click()
        expect(self.page.locator("#copy")).to_be_enabled(); expect(self.page.locator("#referralText")).to_contain_text("δεξιού γόνατος")

    def open_optional_refinement(self, kind):
        control=self.page.locator(f"[data-clinical-v4={kind}]")
        if control.get_attribute("aria-pressed") != "true":
            control.click(); expect(control).to_have_attribute("aria-pressed", "true")
            self.assertFalse(self.page.locator("#sheet").evaluate("el=>el.open"))
        control.click(); expect(self.page.locator("#sheet")).to_be_visible()

    def open_exam(self):
        if self.page.locator("#advancedToggle").get_attribute("aria-expanded") != "true": self.page.locator("#advancedToggle").click()
        self.page.locator("#advanced .v3-category-row[data-v3-category=exam]").click(); expect(self.page.locator("#sheet")).to_be_visible()

    def test_live_production_surface_has_no_generate_or_synthetic_stamp(self):
        self.assertEqual(self.page.get_by_text("Δημιουργία παραπεμπτικού", exact=True).count(), 0)
        self.assertEqual(self.page.get_by_text("ΔΟΚΙΜΑΣΤΙΚΟ ΚΕΙΜΕΝΟ", exact=True).count(), 0)
        self.ready(); expect(self.page.locator("#directEditV2")).to_be_visible(); expect(self.page.locator("#phenotype")).to_have_class("clinical-grid-v4")
        self.page.locator("#advancedToggle").click(); expect(self.page.locator("#advanced .v3-category-row")).to_have_count(6)
        expect(self.page.locator("#v3Favorites")).to_contain_text("★ Συχνά")
        self.page.screenshot(path=str(ARTIFACTS / "cockpit-knee-oa-desktop.png"), full_page=True)

    def test_copy_exports_clean_referral_without_prototype_prefix(self):
        self.ready(); self.page.locator("#copy").click(); copied = self.page.evaluate("navigator.clipboard.readText()")
        self.assertIn("δεξιού γόνατος", copied); self.assertFalse(copied.startswith("ΔΟΚΙΜΑΣΤΙΚΟ"), copied[:80]); self.assertNotIn("ΟΧΙ ΓΙΑ ΚΛΙΝΙΚΗ ΧΡΗΣΗ", copied)

    def test_first_tap_hint_is_visible_without_forced_sheet(self):
        self.ready(); pain=self.page.locator("[data-clinical-v4=pain]"); pain.click()
        expect(pain).to_have_attribute("aria-pressed","true"); self.assertFalse(self.page.locator("#sheet").evaluate("el=>el.open"))
        expect(self.page.locator("[data-v51-second-tap-hint=pain]")).to_have_text("Λεπτομέρειες ›")
        expect(self.page.locator("[data-v51-second-tap-hint=pain]")).to_be_visible()
        self.assertEqual(self.page.locator("[data-v51-second-tap-hint=function]").count(),0)

    def test_compact_clinical_parent_is_optional_and_more_context_does_not_auto_select(self):
        self.ready(); weakness=self.page.locator("[data-clinical-v4=weakness]"); weakness.click()
        expect(weakness).to_have_attribute("aria-pressed", "true"); self.assertFalse(self.page.locator("#sheet").evaluate("el=>el.open"))
        expect(self.page.locator("#referralText")).to_contain_text("μυϊκή αδυναμία")
        before = self.page.locator("#referralText").inner_text(); self.page.locator("#advancedToggle").click(); relevant = self.page.locator("#v3Relevant")
        expect(relevant).to_be_visible(); expect(relevant).to_contain_text("Αδυναμία στην εξέταση")
        self.assertEqual(self.page.locator("[data-q-weakness][aria-pressed=true]").count(), 0)
        relevant.locator("[data-v3-focus=weakness]").click()
        expect(self.page.locator("#sheet [data-q-weakness=knee_extension_exam]")).to_have_attribute("aria-pressed", "false")
        self.assertEqual(self.page.locator("#referralText").inner_text(), before)

    def test_weakness_atrophy_is_not_duplicated_between_second_tap_and_more_exam(self):
        self.ready(); weakness=self.page.locator("[data-clinical-v4=weakness]"); weakness.click(); weakness.click(); expect(self.page.locator("#sheet")).to_be_visible()
        self.assertEqual(self.page.get_by_role("button", name="Ατροφία τετρακεφάλου", exact=True).count(), 0)
        for label in ["Αδυναμία έκτασης γόνατος / τετρακεφάλου","Αδυναμία κάμψης γόνατος / ισχιοκνημιαίων","Αδυναμία κάμψης και έκτασης γόνατος"]:
            expect(self.page.get_by_role("button", name=label, exact=True)).to_have_count(1)
        self.page.locator("#closeSheet").click(); self.open_exam()
        expect(self.page.get_by_role("button", name="Ατροφία τετρακεφάλου", exact=True)).to_have_count(1)
        self.page.get_by_role("button", name="Ατροφία τετρακεφάλου", exact=True).click()
        expect(self.page.locator("#referralText")).to_contain_text("εμφανής ατροφία τετρακεφάλου")
        expect(weakness.locator("[data-clinical-count-v4]")).to_be_hidden()

    def test_specific_directional_weakness_and_pain_prose_on_production_transport(self):
        self.ready(); self.open_optional_refinement("pain")
        self.page.locator("#sheet [data-q-pain=medial_joint_line]").click(); self.page.locator("#sheet [data-q-pain=pes_anserine_region]").click()
        expect(self.page.locator("#referralText")).not_to_contain_text("κυρίως"); expect(self.page.locator("#referralText")).not_to_contain_text("χηνείου ποδός στη μεσάρθρια γραμμή")
        self.page.locator("#closeSheet").click(); self.open_optional_refinement("weakness")
        self.page.locator("#sheet [data-q-weakness=knee_extension_exam]").click()
        expect(self.page.locator("#referralText")).to_contain_text("αδυναμία έκτασης γόνατος / τετρακεφάλου κατά την εξέταση")
        expect(self.page.locator("#referralText")).to_contain_text("με έμφαση σε")

    def test_v51_exam_findings_on_protected_transport(self):
        self.ready(); self.open_exam(); expect(self.page.get_by_role("button",name="Κριγμός στην κίνηση",exact=True)).to_be_visible()
        self.page.get_by_role("button",name="Περιορισμός εύρους κίνησης",exact=True).click(); expect(self.page.locator("#v51RomDetails")).to_be_visible()
        self.page.get_by_role("button",name="Περιορισμός παθητικής κάμψης",exact=True).click(); expect(self.page.locator("#referralText")).to_contain_text("περιορισμό παθητικής κάμψης")
        self.page.get_by_role("button",name="Κριγμός στην κίνηση",exact=True).click(); expect(self.page.locator("#referralText")).to_contain_text("κριγμός κατά την κίνηση")
        self.page.locator("[data-q-tenderness=lateral_bony]").click(); expect(self.page.locator("#referralText")).to_contain_text("οστική ευαισθησία έξω")
        self.page.locator("[data-v51-stability=posterior_instability_pcl]").click(); expect(self.page.locator("#referralText")).to_contain_text("οπίσθια αστάθεια / ΟΧΣ")
        expect(self.page.locator("#v51ReviewBubble")).to_be_hidden()


    def test_cyprus_context_is_visible_for_core_items_without_mutating_selection(self):
        self.ready(); referral_before=self.page.locator("#referralText").inner_text()
        for item in ["therapeutic_exercise","progressive_strengthening","education_and_self_management"]:
            row=self.page.locator(f'#plan [data-row-item="{item}"]')
            expect(row.locator(".v51-jurisdiction-cue")).to_be_visible();expect(row.locator(".v51-jurisdiction-cue")).to_have_text("Κύπρος")
            row.locator(f'[data-evidence="{item}"]').click();expect(self.page.locator("#sheet")).to_be_visible()
            local=self.page.locator("#sheet [data-jurisdiction-position]");expect(local).to_have_count(1);expect(local).to_be_visible()
            expect(local).to_contain_text("Κύπρος · ΓεΣΥ");expect(local).to_contain_text("συμβατή με τη διεθνή θέση")
            expect(self.page.locator("#v51SourceShortcuts")).to_contain_text("Κύπρος · ΟΑΥ")
            self.page.locator("#closeSheet").click()
        self.assertEqual(self.page.locator("#referralText").inner_text(),referral_before)

    def test_cyprus_difference_is_progressive_and_source_links_are_direct(self):
        self.ready(); referral_before = self.page.locator("#referralText").inner_text(); self.assertEqual(self.page.get_by_text("Κύπρος · διαφέρει", exact=True).count(), 0)
        self.page.locator("#advancedToggle").click(); self.page.locator("#advanced [data-v3-category=adjuncts]").click(); expect(self.page.locator("#sheet")).to_be_visible()
        self.page.locator("#sheet [data-evidence=acupuncture]").click(); expect(self.page.locator("#sheet .sheet-state")).to_contain_text("Οι οδηγίες διαφέρουν")
        source_strip=self.page.locator("#v51SourceShortcuts"); expect(source_strip).to_be_visible(); expect(source_strip).to_contain_text("Κύπρος · ΟΑΥ")
        source_links=source_strip.locator("a.v51-source-shortcut"); self.assertGreaterEqual(source_links.count(),2)
        local = self.page.locator("#sheet [data-jurisdiction-position]"); expect(local).to_be_visible(); expect(local).to_contain_text("Κύπρος · διαφέρει")
        expect(local).to_contain_text("Δεν συνιστάται βελονισμός"); expect(local).to_contain_text("η διεθνής κατάσταση δεν αλλάζει")
        local_link=local.locator(".v51-source-link"); expect(local_link).to_have_count(1); self.assertTrue((local_link.get_attribute("href") or "").startswith("https://"))
        international=self.page.locator("#sheet .source-position:not(.jurisdiction-position-v1) .v51-source-link"); self.assertGreaterEqual(international.count(),1)
        self.assertEqual(self.page.get_by_text("Πληροφορία ΓεΣΥ", exact=True).count(), 0); self.assertEqual(self.page.locator("#referralText").inner_text(), referral_before)
        self.assertEqual(self.page.locator("#plan [data-select=acupuncture][aria-pressed=true]").count(), 0)

    def test_prolonged_stiffness_review_bubble_is_nonblocking(self):
        self.ready(); stiff=self.page.locator("[data-clinical-v4=stiffness]"); stiff.click(); stiff.click()
        self.page.locator("#sheet [data-q-stiffness=morning]").click(); self.page.locator("#sheet [data-q-duration=gt_30]").click(); self.page.locator("#closeSheet").click()
        expect(self.page.locator("#v51ReviewBubble")).to_be_visible(); expect(self.page.locator("#v51ReviewBubble")).to_contain_text("Πρωινή δυσκαμψία >30′")
        expect(self.page.locator("#copy")).to_be_enabled();self.open_exam();expect(self.page.get_by_role("heading",name="Κλινική επανεκτίμηση",exact=True)).to_be_visible();expect(self.page.locator("[data-v51-review-content]")).to_contain_text("Πρωινή δυσκαμψία >30′")

    def test_manual_edit_reconciliation_remains_fail_closed(self):
        self.ready(); self.page.locator("#directEditV2").click(); self.page.locator("#manualText").fill("Χειροκίνητο κείμενο παραπομπής.")
        self.page.locator("[data-confirm-manual]").click(); expect(self.page.locator("#referralText")).to_have_text("Χειροκίνητο κείμενο παραπομπής.")
        self.page.locator("[data-side=left]").click(); expect(self.page.locator("#manualReconcile")).to_be_visible(); expect(self.page.locator("#copy")).to_be_disabled()
        expect(self.page.locator("#referralText")).to_have_text("Χειροκίνητο κείμενο παραπομπής.")

    def test_no_browser_storage_and_mobile_reflow(self):
        self.ready(); self.assertEqual(self.page.evaluate("localStorage.length"), 0); self.assertEqual(self.page.evaluate("sessionStorage.length"), 0)
        self.page.set_viewport_size({"width": 390,"height": 900}); columns=self.page.evaluate("getComputedStyle(document.querySelector('.clinical-grid-v4')).gridTemplateColumns.split(' ').length")
        self.assertEqual(columns,2); self.page.locator("#advancedToggle").click(); self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth<=innerWidth"))
        expect(self.page.locator("#advanced .v3-category-row")).to_have_count(6); self.page.screenshot(path=str(ARTIFACTS / "cockpit-knee-oa-mobile.png"), full_page=True)


if __name__ == "__main__": unittest.main(verbosity=2)
