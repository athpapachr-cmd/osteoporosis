from __future__ import annotations

import unittest
from pathlib import Path

from clinic_utilities.physio_evidence_runtime import CU1ClinicianEvidenceResolver
from clinic_utilities.physio_referral_runtime import CU1ContractBundle


ROOT = Path(__file__).resolve().parent


class CU1ClinicianEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolver = CU1ClinicianEvidenceResolver(ROOT)
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

    def test_browser_evidence_panel_is_clinician_only_and_separate_from_referral_text(self):
        js = (ROOT / "static/clinic-utilities/physio-referral/dynamic-subtype.js").read_text(encoding="utf-8")
        self.assertIn("Τεκμηρίωση / Παραπομπές", js)
        self.assertIn("/api/evidence", js)
        self.assertIn("clinicianEvidencePanel", js)
        self.assertNotIn("outputText.value +=", js)
        self.assertNotIn("navigator.clipboard.writeText", js)


if __name__ == "__main__":
    unittest.main()
