from __future__ import annotations

import os
import uuid
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient


KEY = "physio-knee-oa-integration-test-key"


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
            "rehab_directions": [
                "therapeutic_exercise",
                "progressive_strengthening",
                "education_and_self_management",
            ],
            "adjunct_options": [],
            "goals": [],
            "phenotype": {},
            "qualifiers": {},
            "explicit_restrictions": [],
            "clinician_free_text_optional": "",
            "safety_flags": [],
        },
        "dismissed": [],
    }


class KneeOACockpitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = patch.dict(os.environ, {"CLINICAL_DATA_KEY": KEY}, clear=False)
        cls.env.start()
        from main import app
        cls.client = TestClient(app)
        cls.headers = {"X-Clinical-Key": KEY}

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.env.stop()

    def test_protected_cockpit_page_rejects_missing_auth(self):
        response = self.client.get("/clinical/clinic-utilities/physio-referral")
        self.assertEqual(response.status_code, 401)

    def test_protected_page_is_knee_oa_live_product_not_legacy_generate_form(self):
        response = self.client.get("/clinical/clinic-utilities/physio-referral", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        html = response.text
        self.assertIn("Οστεοαρθρίτιδα γόνατος", html)
        self.assertIn('id="advancedToggle"', html)
        self.assertIn('id="referralText"', html)
        self.assertIn("product-more-v3.js", html)
        self.assertNotIn("Δημιουργία παραπεμπτικού", html)
        self.assertNotIn("ΔΟΚΙΜΑΣΤΙΚΟ ΚΕΙΜΕΝΟ", html)

    def test_product_api_is_protected_and_uses_reviewed_real_cu1_projection(self):
        unauthorized = self.client.get("/clinical/clinic-utilities/physio-referral/api/product/bootstrap")
        self.assertEqual(unauthorized.status_code, 401)

        bootstrap = self.client.get(
            "/clinical/clinic-utilities/physio-referral/api/product/bootstrap",
            headers=self.headers,
        )
        self.assertEqual(bootstrap.status_code, 200)
        meta = bootstrap.json()
        self.assertFalse(meta["synthetic_only"])
        self.assertEqual(meta["deployment_context"], "clinical_excellence_cockpit")
        self.assertEqual(meta["defaults"], [
            "therapeutic_exercise",
            "progressive_strengthening",
            "education_and_self_management",
        ])

        projected = self.client.post(
            "/clinical/clinic-utilities/physio-referral/api/product/project",
            headers=self.headers,
            json=_payload(meta["package_version"]),
        )
        self.assertEqual(projected.status_code, 200, projected.text)
        body = projected.json()
        self.assertTrue(body["gate"]["allowed"])
        self.assertFalse(body["gate"]["blocked"])
        self.assertIn("δεξιού γόνατος", body["text"])
        self.assertNotIn("NICE", body["text"])

    def test_v5_production_presentation_separates_plan_and_reconciles_pain_overlap(self):
        meta = self.client.get(
            "/clinical/clinic-utilities/physio-referral/api/product/bootstrap",
            headers=self.headers,
        ).json()
        payload = _payload(meta["package_version"])
        payload["state"]["findings"] = ["pain", "joint_line_pain"]
        payload["state"]["functional_impairments"] = ["stairs"]
        payload["state"]["goals"] = ["maintain_or_regain_adl_independence"]
        payload["state"]["qualifiers"] = {"pain_locations": ["medial_joint_line", "pes_anserine_region"]}
        response = self.client.post(
            "/clinical/clinic-utilities/physio-referral/api/product/project",
            headers=self.headers,
            json=payload,
        )
        self.assertEqual(response.status_code, 200, response.text)
        text = response.json()["text"]
        self.assertIn("έσω μεσάρθρια περιοχή", text)
        self.assertIn("περιοχή του χηνείου ποδός", text)
        self.assertNotIn("χηνείου ποδός στη μεσάρθρια γραμμή", text)
        self.assertIn("\n\nΠαρακαλώ για φυσιοθεραπευτική αξιολόγηση", text)
        self.assertIn("Επιπρόσθετη λειτουργική προτεραιότητα:", text)
        self.assertNotIn("Επιπλέον στόχος:", text)

    def test_production_endpoint_rejects_synthetic_usage_marker(self):
        meta = self.client.get(
            "/clinical/clinic-utilities/physio-referral/api/product/bootstrap",
            headers=self.headers,
        ).json()
        payload = _payload(meta["package_version"])
        payload["synthetic_only"] = True
        response = self.client.post(
            "/clinical/clinic-utilities/physio-referral/api/product/project",
            headers=self.headers,
            json=payload,
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "invalid_or_stale_physio_product_request")

    def test_invalid_or_forged_product_state_fails_closed(self):
        meta = self.client.get(
            "/clinical/clinic-utilities/physio-referral/api/product/bootstrap",
            headers=self.headers,
        ).json()
        payload = _payload(meta["package_version"])
        payload["gate"] = {"allowed": True}
        response = self.client.post(
            "/clinical/clinic-utilities/physio-referral/api/product/project",
            headers=self.headers,
            json=payload,
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "invalid_or_stale_physio_product_request")

    def test_production_static_product_source_has_no_storage_or_demo_markup(self):
        for path in [
            "product-app.js",
            "product-qualifiers.js",
            "product-more-v3.js",
            "product-clinical-sheet-v5.js",
            "production-env.js",
            "production-finalize.js",
        ]:
            response = self.client.get(f"/static/clinic-utilities/physio-referral/{path}")
            self.assertEqual(response.status_code, 200, path)
            self.assertNotIn("localStorage", response.text, path)
            self.assertNotIn("sessionStorage", response.text, path)
        finalizer = self.client.get("/static/clinic-utilities/physio-referral/production-finalize.js").text
        self.assertIn("product-clinical-sheet-v5.js", finalizer)
        self.assertIn("navigator.clipboard.writeText(effectiveText())", finalizer)
        self.assertNotIn("DEMO+effectiveText", finalizer)
        bridge = self.client.get("/static/clinic-utilities/physio-referral/production-env.js").text
        self.assertIn("body.synthetic_only = false", bridge)


if __name__ == "__main__":
    unittest.main(verbosity=2)
