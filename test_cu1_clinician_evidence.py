from __future__ import annotations

import unittest
from pathlib import Path

from clinic_utilities.physio_evidence_api import contextual_evidence_summary
from clinic_utilities.physio_evidence_runtime import CU1ClinicianEvidenceResolver
from clinic_utilities.physio_referral_runtime import CU1ContractBundle
from clinic_utilities.physio_rich_referral import CU1RichReferralRenderer


ROOT = Path(__file__).resolve().parent


class CU1ClinicianEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolver = CU1ClinicianEvidenceResolver(ROOT)
        cls.renderer = CU1RichReferralRenderer(ROOT)
        cls.bundle = CU1ContractBundle(ROOT)

    def test_lateral_elbow_panel_resolves_reviewed_sources_and_scoped_claims(self):
        data = self.resolver.route_summary(
            profile_id="elbow",
            route_id="lateral_elbow_tendinopathy",
        )
        titles = {item.get("title") for item in data["sources"]}
        self.assertIn("Lateral Elbow Pain and Muscle Function Impairments - Clinical Practice Guidelines", titles)
        self.assertIn("Manual therapy and exercise for lateral elbow pain", titles)
        self.assertTrue(data["has_applicable_profile"])
        self.assertTrue(any(item.get("output_scope") == "referral_core" for item in data["claims"]))
        self.assertTrue(any(item.get("output_scope") == "clinician_ui_only" for item in data["claims"]))
        self.assertTrue(any("transition" in gap.lower() or "return" in gap.lower() for gap in data["evidence_gaps"]))
        for item in data["sources"]:
            self.assertNotIn("evidence_id", item)
        for item in data["claims"]:
            self.assertNotIn("claim_id", item)

    def test_promoted_knee_oa_evidence_is_visible(self):
        data = self.resolver.route_summary(profile_id="knee", route_id="knee_osteoarthritis")
        summaries = "\n".join(str(item.get("claim_summary") or "") for item in data["claims"])
        self.assertIn("AAOS", summaries)
        self.assertIn("EULAR", summaries)
        self.assertTrue(any(item.get("title") for item in data["sources"]))

    def test_achilles_subtype_scope_does_not_cross_leak(self):
        mid = self.resolver.route_summary(
            profile_id="ankle_foot",
            route_id="achilles_tendinopathy",
            subtype_id="midportion_achilles_tendinopathy",
        )
        insertional = self.resolver.route_summary(
            profile_id="ankle_foot",
            route_id="achilles_tendinopathy",
            subtype_id="insertional_achilles_tendinopathy",
        )
        mid_titles = {item.get("title") for item in mid["sources"]}
        insertional_titles = {item.get("title") for item in insertional["sources"]}
        self.assertTrue(any("Midportion Achilles" in str(title) for title in mid_titles))
        self.assertTrue(any("insertional Achilles" in str(title) for title in insertional_titles))
        self.assertFalse(any("Midportion Achilles" in str(title) for title in insertional_titles))

    def test_evidence_limited_route_keeps_gap_visible(self):
        data = self.resolver.route_summary(
            profile_id="lumbar",
            route_id="deep_gluteal_piriformis_presentation",
        )
        self.assertTrue(data["has_applicable_profile"])
        self.assertTrue(data["evidence_gaps"])
        status = str(data.get("sequence_status") or data.get("coverage_status") or "")
        self.assertTrue("blocked" in status or "incomplete" in status or bool(data["evidence_gaps"]))

    def test_every_registry_route_can_be_projected_without_generic_fallback(self):
        for profile_id, profile in (self.bundle.registry.get("profiles") or {}).items():
            for route_id in (profile.get("routes") or {}):
                with self.subTest(profile=profile_id, route=route_id):
                    data = self.resolver.route_summary(profile_id=profile_id, route_id=route_id)
                    self.assertEqual(data["profile_id"], profile_id)
                    self.assertEqual(data["route_id"], route_id)
                    if not data["has_applicable_profile"]:
                        self.assertEqual(data["sources"], [])
                        self.assertEqual(data["claims"], [])

    def test_shoulder_instability_evidence_is_context_scoped(self):
        anterior_context = {
            "shoulder_instability_direction": "anterior",
            "shoulder_instability_cause": "traumatic",
            "shoulder_instability_recurrence": "first_time",
            "shoulder_instability_management_context": "nonoperative_rehabilitation",
            "shoulder_instability_structural_protocol_context": "clear_for_selected_rehabilitation",
        }
        posterior_context = {
            "shoulder_instability_direction": "posterior",
            "shoulder_instability_cause": "traumatic",
            "shoulder_instability_recurrence": "recurrent",
            "shoulder_instability_management_context": "nonoperative_rehabilitation",
            "shoulder_instability_structural_protocol_context": "clear_for_selected_rehabilitation",
        }
        mdi_context = {
            "shoulder_instability_direction": "multidirectional",
            "shoulder_instability_cause": "atraumatic",
            "shoulder_instability_recurrence": "recurrent",
            "shoulder_instability_management_context": "nonoperative_rehabilitation",
            "shoulder_instability_structural_protocol_context": "clear_for_selected_rehabilitation",
        }

        anterior = contextual_evidence_summary(
            self.resolver,
            self.renderer,
            profile_id="shoulder",
            route_id="glenohumeral_instability_dislocation",
            wording_mode="presentation",
            context=anterior_context,
        )
        posterior = contextual_evidence_summary(
            self.resolver,
            self.renderer,
            profile_id="shoulder",
            route_id="glenohumeral_instability_dislocation",
            wording_mode="presentation",
            context=posterior_context,
        )
        mdi = contextual_evidence_summary(
            self.resolver,
            self.renderer,
            profile_id="shoulder",
            route_id="glenohumeral_instability_dislocation",
            wording_mode="presentation",
            context=mdi_context,
        )

        for data in (anterior, posterior, mdi):
            self.assertEqual(data["selection_state"], "resolved_context_profile")
            self.assertEqual(data["profile_count"], 1)
            self.assertTrue(data["sources"])
            self.assertTrue(data["claims"])

        anterior_text = "\n".join(str(item.get("claim_summary") or "") for item in anterior["claims"]).lower()
        posterior_text = "\n".join(str(item.get("claim_summary") or "") for item in posterior["claims"]).lower()
        mdi_text = "\n".join(str(item.get("claim_summary") or "") for item in mdi["claims"]).lower()

        self.assertIn("traumatic anterior", anterior_text)
        self.assertNotIn("posterior rotator-cuff", anterior_text)
        self.assertNotIn("multidirectional instability", anterior_text)

        self.assertIn("posterior", posterior_text)
        self.assertNotIn("traumatic anterior", posterior_text)
        self.assertNotIn("multidirectional instability", posterior_text)

        self.assertIn("multidirectional instability", mdi_text)
        self.assertIn("uncertain", mdi_text)
        self.assertNotIn("traumatic anterior", mdi_text)
        self.assertNotIn("posterior rotator-cuff", mdi_text)

    def test_shoulder_instability_unresolved_context_shows_no_mixed_evidence(self):
        data = contextual_evidence_summary(
            self.resolver,
            self.renderer,
            profile_id="shoulder",
            route_id="glenohumeral_instability_dislocation",
            wording_mode="presentation",
            context={"shoulder_instability_direction": "anterior"},
        )
        self.assertEqual(data["selection_state"], "context_required_for_evidence")
        self.assertFalse(data["has_applicable_profile"])
        self.assertEqual(data["sources"], [])
        self.assertEqual(data["claims"], [])
        self.assertIn("select_route_context_to_resolve_evidence", data["evidence_gaps"])

    def test_ghoa_evidence_is_management_context_scoped(self):
        nonop = contextual_evidence_summary(
            self.resolver,
            self.renderer,
            profile_id="shoulder",
            route_id="glenohumeral_osteoarthritis",
            wording_mode="formal_diagnosis",
            context={"ghoa_management_context": "nonoperative"},
        )
        preop = contextual_evidence_summary(
            self.resolver,
            self.renderer,
            profile_id="shoulder",
            route_id="glenohumeral_osteoarthritis",
            wording_mode="formal_diagnosis",
            context={"ghoa_management_context": "preoperative_TSA"},
        )
        self.assertEqual(nonop["profile_count"], 1)
        self.assertEqual(preop["profile_count"], 1)
        nonop_text = "\n".join(str(item.get("claim_summary") or "") for item in nonop["claims"]).lower()
        preop_text = "\n".join(str(item.get("claim_summary") or "") for item in preop["claims"]).lower()
        self.assertIn("nonoperative", nonop_text)
        self.assertNotIn("preoperative physical therapist services", nonop_text)
        self.assertIn("preoperative physical therapist services", preop_text)

        postop = contextual_evidence_summary(
            self.resolver,
            self.renderer,
            profile_id="shoulder",
            route_id="glenohumeral_osteoarthritis",
            wording_mode="formal_diagnosis",
            context={"ghoa_management_context": "postoperative_arthroplasty"},
        )
        self.assertEqual(postop["selection_state"], "context_required_for_evidence")
        self.assertEqual(postop["sources"], [])
        self.assertEqual(postop["claims"], [])

    def test_browser_evidence_panel_is_clinician_only_context_aware_and_separate_from_referral_text(self):
        js = (ROOT / "static/clinic-utilities/physio-referral/dynamic-subtype.js").read_text(encoding="utf-8")
        self.assertIn("Τεκμηρίωση / Παραπομπές", js)
        self.assertIn("/api/evidence", js)
        self.assertIn("clinicianEvidencePanel", js)
        self.assertIn("wording_mode_optional", js)
        self.assertIn("context_optional: currentRouteContext()", js)
        self.assertIn("context_required_for_evidence", js)
        self.assertIn("event.target?.dataset?.contextKey", js)
        self.assertNotIn("outputText.value +=", js)
        self.assertNotIn("navigator.clipboard.writeText", js)


if __name__ == "__main__":
    unittest.main()
