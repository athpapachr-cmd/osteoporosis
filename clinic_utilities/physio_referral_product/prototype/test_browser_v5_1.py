from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from clinic_utilities.physio_referral_product.prototype import server as p


class V51BrowserTests(unittest.TestCase):
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
        self.context=self.browser.new_context(viewport={"width":1280,"height":1000},reduced_motion="reduce")
        self.page=self.context.new_page();self.errors=[]
        self.page.on("pageerror",lambda e:self.errors.append(str(e)))
        self.page.goto(self.origin)
        expect(self.page.locator("[data-clinical-v4]")).to_have_count(4)
        expect(self.page.locator("#reviewStatus")).not_to_have_text("Η παραπομπή ενημερώνεται")

    def tearDown(self):
        self.assertEqual(self.errors,[]);self.context.close()

    def ready(self):
        self.page.locator("#assertion").click();self.page.locator("[data-side=right]").click()
        expect(self.page.locator("#copy")).to_be_enabled()

    def open_exam(self):
        if self.page.locator("#advancedToggle").get_attribute("aria-expanded")!="true":
            self.page.locator("#advancedToggle").click()
        self.page.locator("#advanced .v3-category-row[data-v3-category=exam]").click()
        expect(self.page.locator("#sheet")).to_be_visible()
        expect(self.page.locator("#sheetTitle")).to_have_text("Εξέταση")

    def open_safety(self):
        if self.page.locator("#advancedToggle").get_attribute("aria-expanded")!="true":
            self.page.locator("#advancedToggle").click()
        self.page.locator("#advanced .v3-category-row[data-v3-category=safety]").click()
        expect(self.page.locator("#sheet")).to_be_visible();expect(self.page.locator("#sheetTitle")).to_have_text("Κλινικός έλεγχος")

    def test_first_tap_hint_and_second_tap_disclosure(self):
        self.ready()
        pain=self.page.locator("[data-clinical-v4=pain]")
        pain.click();expect(pain).to_have_attribute("aria-pressed","true")
        self.assertFalse(self.page.locator("#sheet").evaluate("el=>el.open"))
        hint=self.page.locator("[data-v51-second-tap-hint=pain]")
        expect(hint).to_be_visible();expect(hint).to_have_text("Λεπτομέρειες ›");expect(hint).to_have_attribute("data-v51-open-details","pain")
        self.assertEqual(self.page.locator("[data-v51-second-tap-hint=function]").count(),0)
        hint.click();expect(self.page.locator("#sheet")).to_be_visible()

    def test_directional_weakness_choices_replace_generic_objective_wording(self):
        self.ready();weak=self.page.locator("[data-clinical-v4=weakness]")
        weak.click();expect(self.page.locator("[data-v51-second-tap-hint=weakness]")).to_be_visible();weak.click()
        expect(self.page.get_by_role("button",name="Αδυναμία έκτασης γόνατος / τετρακεφάλου",exact=True)).to_have_count(1)
        expect(self.page.get_by_role("button",name="Αδυναμία κάμψης γόνατος / ισχιοκνημιαίων",exact=True)).to_have_count(1)
        expect(self.page.get_by_role("button",name="Αδυναμία κάμψης και έκτασης γόνατος",exact=True)).to_have_count(1)
        self.assertEqual(self.page.get_by_role("button",name="Μυϊκή αδυναμία στην εξέταση",exact=True).count(),0)
        self.page.get_by_role("button",name="Αδυναμία κάμψης γόνατος / ισχιοκνημιαίων",exact=True).click()
        expect(self.page.locator("#referralText")).to_contain_text("αδυναμία κάμψης γόνατος / ισχιοκνημιαίων κατά την εξέταση")

    def test_exam_sheet_has_progressive_rom_crepitus_tenderness_stability_and_atrophy(self):
        self.ready();self.open_exam()
        expect(self.page.get_by_role("button",name="Ατροφία τετρακεφάλου",exact=True)).to_have_count(1)
        expect(self.page.get_by_role("button",name="Κριγμός στην κίνηση",exact=True)).to_have_count(1)
        expect(self.page.get_by_role("button",name="Περιορισμός εύρους κίνησης",exact=True)).to_have_count(1)
        expect(self.page.get_by_role("heading",name="Ευαισθησία στην ψηλάφηση",exact=True)).to_have_count(1)
        expect(self.page.get_by_role("heading",name="Σταθερότητα άρθρωσης",exact=True)).to_have_count(1)
        expect(self.page.get_by_role("heading",name="Κλινική επανεκτίμηση",exact=True)).to_have_count(1)
        expect(self.page.locator("[data-v51-review-content]")).to_contain_text("Η απουσία ένδειξης δεν αποτελεί φυσιολογικό έλεγχο")
        self.page.get_by_role("button",name="Περιορισμός εύρους κίνησης",exact=True).click()
        expect(self.page.locator("#v51RomDetails")).to_be_visible()
        for label in ["Υστέρηση ενεργητικής έκτασης","Παθητικό έλλειμμα έκτασης","Περιορισμός ενεργητικής κάμψης","Περιορισμός παθητικής κάμψης"]:
            expect(self.page.get_by_role("button",name=label,exact=True)).to_have_count(1)
        self.page.get_by_role("button",name="Περιορισμός ενεργητικής κάμψης",exact=True).click()
        expect(self.page.locator("#referralText")).to_contain_text("περιορισμό ενεργητικής κάμψης")
        self.page.get_by_role("button",name="Κριγμός στην κίνηση",exact=True).click()
        expect(self.page.locator("#referralText")).to_contain_text("κριγμός κατά την κίνηση")
        self.page.locator("[data-q-tenderness=medial_bony]").click()
        expect(self.page.locator("#referralText")).to_contain_text("οστική ευαισθησία έσω")
        self.page.locator("[data-q-tenderness=extensor_mechanism]").click()
        expect(self.page.locator("#referralText")).to_contain_text("ευαισθησία στον εκτατικό μηχανισμό")
        self.page.locator("[data-v51-stability=valgus_instability]").click()
        expect(self.page.locator("#referralText")).to_contain_text("αστάθεια σε βλαισότητα")
        self.assertNotIn("SIFK",self.page.locator("#referralText").inner_text())

    def test_atrophy_remains_when_generic_weakness_is_removed(self):
        self.ready();self.open_exam()
        self.page.get_by_role("button",name="Ατροφία τετρακεφάλου",exact=True).click()
        expect(self.page.locator("#referralText")).to_contain_text("εμφανής ατροφία τετρακεφάλου")
        self.page.locator("#closeSheet").click()
        weak=self.page.locator("[data-clinical-v4=weakness]");weak.click();weak.click()
        expect(self.page.locator("#sheet")).to_be_visible()
        self.page.locator("[data-clinical-remove-v4=weakness]").click()
        expect(weak).to_have_attribute("aria-pressed","false")
        expect(self.page.locator("#referralText")).to_contain_text("εμφανής ατροφία τετρακεφάλου")

    def test_bony_tenderness_does_not_create_review_bubble(self):
        self.ready();self.open_exam();self.page.locator("[data-q-tenderness=medial_bony]").click()
        expect(self.page.locator("#referralText")).to_contain_text("οστική ευαισθησία έσω")
        expect(self.page.locator("#v51ReviewBubble")).to_be_hidden()

    def test_prolonged_morning_stiffness_creates_nonblocking_review_bubble(self):
        self.ready();stiff=self.page.locator("[data-clinical-v4=stiffness]");stiff.click();stiff.click()
        self.page.locator("#sheet [data-q-stiffness=morning]").click();self.page.locator("#sheet [data-q-duration=gt_30]").click()
        self.page.locator("#closeSheet").click()
        expect(self.page.locator("#v51ReviewBubble")).to_be_visible()
        expect(self.page.locator("#v51ReviewBubble")).to_contain_text("Πρωινή δυσκαμψία >30′")
        expect(self.page.locator("#copy")).to_be_enabled()
        self.open_exam();expect(self.page.locator("[data-v51-review-content]")).to_contain_text("Πρωινή δυσκαμψία >30′")
        href=self.page.locator("#v51ReviewBubble a").get_attribute("href")
        self.assertTrue(href and href.startswith("https://www.nice.org.uk/"),href)

    def test_atypical_observations_are_progressive_nonblocking_and_aggregate_in_review(self):
        self.ready();before=self.page.locator("#referralText").inner_text();self.open_safety()
        details=self.page.locator("#sheet [data-v51-atypical-disclosure]");expect(details).to_be_visible()
        details.locator("summary").click()
        for key,label in [
            ("recent_trauma","Πρόσφατο τραύμα"),
            ("rapid_worsening_or_deformity","Ταχεία επιδείνωση συμπτωμάτων ή παραμόρφωση"),
            ("hot_swollen_joint","Θερμή και διογκωμένη άρθρωση"),
        ]:
            button=self.page.locator(f'#sheet [data-v51-atypical="{key}"]');expect(button).to_have_text(label);button.click()
        expect(self.page.locator("#copy")).to_be_enabled()
        expect(self.page.locator('#sheet [data-select="infection_or_septic_joint_concern"]')).to_have_attribute("aria-pressed","false")
        self.page.locator("#closeSheet").click();expect(self.page.locator("#v51ReviewBubble")).to_contain_text("3 χαρακτηριστικά")
        self.assertEqual(self.page.locator("#referralText").inner_text(),before)
        self.open_exam();review=self.page.locator("[data-v51-review-content]")
        for text in ["Πρόσφατο τραύμα","Ταχεία επιδείνωση συμπτωμάτων ή παραμόρφωση","Θερμή και διογκωμένη άρθρωση"]:expect(review).to_contain_text(text)
        self.assertGreaterEqual(review.locator('a[href^="https://www.gesy.org.cy/"]').count(),3)

    def test_explicit_infection_concern_blocks_while_hot_swollen_joint_alone_does_not(self):
        self.ready();self.open_safety();details=self.page.locator("#sheet [data-v51-atypical-disclosure]");details.locator("summary").click()
        self.page.locator('#sheet [data-v51-atypical="hot_swollen_joint"]').click();expect(self.page.locator("#copy")).to_be_enabled()
        infection=self.page.locator('#sheet [data-select="infection_or_septic_joint_concern"]');expect(infection).to_have_attribute("aria-pressed","false");infection.click()
        expect(infection).to_have_attribute("aria-pressed","true");expect(self.page.locator("#copy")).to_be_disabled();expect(self.page.locator("#sheetSafety")).to_be_visible()


    def test_more_notes_has_optional_chronicity_without_prior_physio_fields(self):
        self.ready()
        self.page.locator("#advancedToggle").click()
        self.page.locator("#advanced [data-v3-category=notes]").click();expect(self.page.locator("#sheet")).to_be_visible()
        expect(self.page.get_by_role("heading",name="Χρονιότητα συμπτωμάτων",exact=True)).to_have_count(1)
        self.assertEqual(self.page.get_by_text("Προηγούμενη φυσικοθεραπεία",exact=True).count(),0)
        self.assertEqual(self.page.get_by_text("Ανταπόκριση",exact=True).count(),0)
        self.page.locator("#v51SymptomDurationValue").fill("8")
        expect(self.page.locator("#referralText")).to_contain_text("Συμπτωματολογία διάρκειας 8 μηνών.")
        self.page.locator("#closeSheet").click()
        expect(self.page.locator('#advanced [data-v3-category="notes"] .v3-category-summary')).to_contain_text("Διάρκεια 8 μήνες")

    def test_functional_retraining_uses_receiver_compressed_wording(self):
        self.ready();self.page.locator("#advancedToggle").click()
        self.page.locator("#advanced [data-v3-category=function]").click()
        self.page.locator('#sheet [data-select="squat"]').click();self.page.locator('#sheet [data-select="kneeling"]').click();self.page.locator("#closeSheet").click()
        self.page.locator("#advanced [data-v3-category=rehab]").click();self.page.locator('#sheet [data-select="functional_task_retraining"]').click()
        expect(self.page.locator("#referralText")).to_contain_text("λειτουργική επανεκπαίδευση με έμφαση στις καταγεγραμμένες λειτουργικές δυσχέρειες")
        expect(self.page.locator("#referralText")).not_to_contain_text("λειτουργική επανεκπαίδευση για")

    def test_evidence_source_has_visible_direct_reviewed_shortcuts(self):
        self.ready();self.page.locator("#plan [data-evidence=therapeutic_exercise]").click()
        expect(self.page.locator("#sheet")).to_be_visible()
        strip=self.page.locator("#v51SourceShortcuts")
        expect(strip).to_be_visible();expect(strip).to_contain_text("Πηγές")
        links=strip.locator("a.v51-source-shortcut")
        self.assertGreaterEqual(links.count(),1)
        for index in range(links.count()):
            href=links.nth(index).get_attribute("href")
            self.assertTrue(href and href.startswith("https://"),href)
            self.assertEqual(links.nth(index).get_attribute("target"),"_blank")
            self.assertIn("noopener",links.nth(index).get_attribute("rel") or "")

    def test_no_storage_and_mobile_exam_reflow(self):
        self.ready();self.open_exam()
        self.assertEqual(self.page.evaluate("localStorage.length"),0);self.assertEqual(self.page.evaluate("sessionStorage.length"),0)
        self.page.set_viewport_size({"width":390,"height":900})
        self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth<=innerWidth"))
        expect(self.page.get_by_role("button",name="Αστάθεια σε βλαισότητα",exact=True)).to_be_visible()


if __name__=="__main__":unittest.main(verbosity=2)
