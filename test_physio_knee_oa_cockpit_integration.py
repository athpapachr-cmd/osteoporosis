from __future__ import annotations

import os
import uuid
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient


KEY = "physio-knee-oa-integration-test-key"
PROFILE_ENV = "PHYSIO_REFERRAL_JURISDICTION_PROFILE"


def _payload(package_version: str) -> dict:
    return {
        "draft_id": str(uuid.uuid4()),
        "revision": 0,
        "package_version": package_version,
        "synthetic_only": False,
        "state": {
            "laterality": "right",
            "formal_assertion_state": "yes",
            "findings": [],
            "functional_impairments": [],
            "rehab_directions": ["therapeutic_exercise","progressive_strengthening","education_and_self_management"],
            "adjunct_options": [],
            "goals": [],
            "phenotype": {},
            "explicit_restrictions": [],
            "clinician_free_text_optional": "",
            "safety_flags": [],
        },
        "dismissed": [],
    }


class KneeOACockpitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = patch.dict(os.environ,{"CLINICAL_DATA_KEY": KEY, PROFILE_ENV: "CY_GESY"},clear=False)
        cls.env.start()
        from main import app
        cls.client = TestClient(app); cls.headers = {"X-Clinical-Key": KEY}

    @classmethod
    def tearDownClass(cls):
        cls.client.close(); cls.env.stop()

    def test_protected_cockpit_page_rejects_missing_auth(self):
        response = self.client.get("/clinical/clinic-utilities/physio-referral"); self.assertEqual(response.status_code, 401)

    def test_protected_page_is_knee_oa_live_product_not_legacy_generate_form(self):
        response = self.client.get("/clinical/clinic-utilities/physio-referral", headers=self.headers); self.assertEqual(response.status_code, 200)
        html = response.text
        for marker in ["Οστεοαρθρίτιδα γόνατος",'id="advancedToggle"','id="referralText"',"product-more-v3.js","product-jurisdiction-v1.js","product-v5-1-exam.js","product-v5-1-exam.css"]:
            self.assertIn(marker, html)
        self.assertNotIn("Δημιουργία παραπεμπτικού", html); self.assertNotIn("ΔΟΚΙΜΑΣΤΙΚΟ ΚΕΙΜΕΝΟ", html)

    def test_product_api_is_protected_and_uses_reviewed_real_cu1_projection(self):
        unauthorized = self.client.get("/clinical/clinic-utilities/physio-referral/api/product/bootstrap"); self.assertEqual(unauthorized.status_code, 401)
        bootstrap = self.client.get("/clinical/clinic-utilities/physio-referral/api/product/bootstrap",headers=self.headers)
        self.assertEqual(bootstrap.status_code, 200); meta = bootstrap.json(); self.assertFalse(meta["synthetic_only"])
        self.assertEqual(meta["deployment_context"], "clinical_excellence_cockpit")
        self.assertEqual(meta["defaults"], ["therapeutic_exercise","progressive_strengthening","education_and_self_management"])
        self.assertEqual(meta["jurisdiction_profile"]["profile_id"], "CY_GESY")
        self.assertEqual(meta["jurisdiction_profile"]["selection_source"], "explicit_account_configuration")
        projected = self.client.post("/clinical/clinic-utilities/physio-referral/api/product/project",headers=self.headers,json=_payload(meta["package_version"]))
        self.assertEqual(projected.status_code, 200, projected.text); body = projected.json()
        self.assertTrue(body["gate"]["allowed"]); self.assertFalse(body["gate"]["blocked"]); self.assertIn("δεξιού γόνατος", body["text"]); self.assertNotIn("NICE", body["text"])
        self.assertEqual(body["jurisdiction_profile"]["profile_id"], "CY_GESY")
        self.assertEqual(body["evidence"]["acupuncture"]["evidence_state"], "guideline_conflict_or_mixed")
        self.assertEqual(body["evidence"]["acupuncture"]["jurisdiction"]["local_direction"], "against")
        self.assertEqual(body["evidence"]["manual_therapy"]["evidence_state"], "guideline_conflict_or_mixed")
        self.assertEqual(body["evidence"]["manual_therapy"]["jurisdiction"]["local_direction"], "conditional_for")

    def test_v51_product_local_exam_state_is_accepted_without_evidence_or_selection_mutation(self):
        meta=self.client.get("/clinical/clinic-utilities/physio-referral/api/product/bootstrap",headers=self.headers).json()
        payload=_payload(meta["package_version"])
        payload["state"]["phenotype"]={"weakness_symptom_or_context":True}
        payload["state"]["qualifiers"]={
            "weakness_detail":"knee_flexion_exam",
            "rom_restriction_present":True,
            "active_flexion_restricted":True,
            "crepitus":True,
            "focal_tenderness_locations":["medial_bony","extensor_mechanism"],
            "stability_findings":["valgus_instability","anterior_instability_acl"],
        }
        body=self.client.post("/clinical/clinic-utilities/physio-referral/api/product/project",headers=self.headers,json=payload).json()
        for phrase in ["αδυναμία κάμψης γόνατος / ισχιοκνημιαίων","περιορισμό ενεργητικής κάμψης","κριγμός κατά την κίνηση","οστική ευαισθησία έσω","αστάθεια σε βλαισότητα"]:
            self.assertIn(phrase,body["text"])
        self.assertEqual(body["state"]["rehab_directions"],["therapeutic_exercise","progressive_strengthening","education_and_self_management"])
        self.assertEqual(body["state"]["adjunct_options"],[])
        self.assertEqual(body["evidence"]["acupuncture"]["evidence_state"],"guideline_conflict_or_mixed")
        self.assertEqual(body["evidence"]["acupuncture"]["jurisdiction"]["local_direction"],"against")
        self.assertEqual(body["clinical_review_clues"],[])

    def test_jurisdiction_overlay_does_not_change_referral_or_selection(self):
        meta = self.client.get("/clinical/clinic-utilities/physio-referral/api/product/bootstrap",headers=self.headers).json(); payload = _payload(meta["package_version"])
        with patch.dict(os.environ, {PROFILE_ENV: ""}, clear=False):
            without_local = self.client.post("/clinical/clinic-utilities/physio-referral/api/product/project",headers=self.headers,json=payload).json()
        with patch.dict(os.environ, {PROFILE_ENV: "CY_GESY"}, clear=False):
            with_local = self.client.post("/clinical/clinic-utilities/physio-referral/api/product/project",headers=self.headers,json=payload).json()
        self.assertEqual(with_local["text"], without_local["text"]); self.assertEqual(with_local["state"], without_local["state"])
        self.assertEqual(with_local["gate"], without_local["gate"]); self.assertEqual(with_local["suggestions"], without_local["suggestions"])
        for item, view in without_local["evidence"].items():
            self.assertEqual(with_local["evidence"][item]["evidence_state"], view["evidence_state"])
            self.assertEqual(with_local["evidence"][item]["positions"], view["positions"])

    def test_unknown_profile_fails_closed_to_no_overlay(self):
        with patch.dict(os.environ, {PROFILE_ENV: "GR"}, clear=False):
            meta = self.client.get("/clinical/clinic-utilities/physio-referral/api/product/bootstrap",headers=self.headers).json()
            self.assertIsNone(meta["jurisdiction_profile"])
            response = self.client.post("/clinical/clinic-utilities/physio-referral/api/product/project",headers=self.headers,json=_payload(meta["package_version"]))
        self.assertEqual(response.status_code, 200); body = response.json(); self.assertIsNone(body["jurisdiction_profile"])
        self.assertFalse(any("jurisdiction" in view for view in body["evidence"].values()))

    def test_production_endpoint_rejects_synthetic_usage_marker(self):
        meta = self.client.get("/clinical/clinic-utilities/physio-referral/api/product/bootstrap",headers=self.headers).json(); payload = _payload(meta["package_version"]); payload["synthetic_only"] = True
        response = self.client.post("/clinical/clinic-utilities/physio-referral/api/product/project",headers=self.headers,json=payload)
        self.assertEqual(response.status_code, 400); self.assertEqual(response.json()["detail"], "invalid_or_stale_physio_product_request")

    def test_invalid_or_forged_product_state_fails_closed(self):
        meta = self.client.get("/clinical/clinic-utilities/physio-referral/api/product/bootstrap",headers=self.headers).json(); payload = _payload(meta["package_version"]); payload["gate"] = {"allowed": True}
        response = self.client.post("/clinical/clinic-utilities/physio-referral/api/product/project",headers=self.headers,json=payload)
        self.assertEqual(response.status_code, 400); self.assertEqual(response.json()["detail"], "invalid_or_stale_physio_product_request")

    def test_production_static_product_source_has_no_storage_or_demo_markup(self):
        for path in ["product-app.js","product-qualifiers.js","product-more-v3.js","product-jurisdiction-v1.js","product-v5-1-exam.js","production-env.js","production-finalize.js"]:
            response = self.client.get(f"/static/clinic-utilities/physio-referral/{path}"); self.assertEqual(response.status_code, 200, path)
            self.assertNotIn("localStorage", response.text, path); self.assertNotIn("sessionStorage", response.text, path)
        v51=self.client.get("/static/clinic-utilities/physio-referral/product-v5-1-exam.js").text
        for marker in ["knee_flexion_exam","data-v51-rom-parent","data-v51-crepitus","data-v51-stability","v51-source-link"]: self.assertIn(marker,v51)
        self.assertEqual(self.client.get("/static/clinic-utilities/physio-referral/product-v5-1-exam.css").status_code,200)
        finalizer = self.client.get("/static/clinic-utilities/physio-referral/production-finalize.js").text
        self.assertIn("navigator.clipboard.writeText(effectiveText())", finalizer); self.assertNotIn("DEMO+effectiveText", finalizer)
        bridge = self.client.get("/static/clinic-utilities/physio-referral/production-env.js").text; self.assertIn("body.synthetic_only = false", bridge)


if __name__ == "__main__": unittest.main(verbosity=2)
