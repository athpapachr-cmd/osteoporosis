from __future__ import annotations

import unittest
from pathlib import Path

from clinic_utilities.physio_evidence_runtime import CU1ClinicianEvidenceResolver
from clinic_utilities.physio_referral_formatter_el_v2 import CU1GreekReferralFormatter
from clinic_utilities.physio_referral_runtime import CONTRACT_VERSION, CU1ContractBundle
from clinic_utilities.physio_rich_referral import CU1RichReferralRenderer


ROOT = Path(__file__).resolve().parent


def lateral_elbow_draft():
    return {
        "contract_version": CONTRACT_VERSION,
        "patient_context": {
            "age_years_optional": 46,
            "skeletal_maturity_optional": None,
            "sport_or_work_demand_optional": "κομμώτρια",
            "relevant_medical_context_ids": [],
            "free_text_optional": None,
        },
        "body_region": "elbow",
        "primary_problem": {
            "problem_id": "synthetic-shared-rich",
            "profile_id": "elbow",
            "route_id": "lateral_elbow_tendinopathy",
            "wording_mode": "established_structural_diagnosis",
            "formal_assertion_state_optional": None,
            "subtype_id_optional": None,
            "laterality": "right",
            "chronicity_or_phase_optional": None,
            "context": {},
            "shared_target_optional": None,
            "source_route_optional": None,
        },
        "secondary_problems": [],
        "findings": [{"finding_id": "lateral_elbow_pain"}, {"finding_id": "pain_with_gripping"}],
        "functional_impairments": [{"id": "gripping"}, {"id": "manual_work"}],
        "precautions": [],
        "explicit_restrictions": [],
        "goals": [],
        "rehab_directions": [],
        "adjunct_options": [],
        "measurements": [],
        "safety": {"input_flags": [], "acknowledged_rule_ids": [], "clinician_disposition": "none_recorded"},
        "sessions_optional": None,
        "clinician_free_text_optional": None,
    }


class CU1SharedRichReferralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = CU1ContractBundle(ROOT)
        cls.renderer = CU1RichReferralRenderer(ROOT)
        cls.formatter = CU1GreekReferralFormatter(cls.bundle)
        cls.evidence = CU1ClinicianEvidenceResolver(ROOT)

    def test_formatter_and_renderer_have_no_lateral_elbow_treatment_branch(self):
        formatter_source = (ROOT / "clinic_utilities/physio_referral_formatter_el_v2.py").read_text(encoding="utf-8")
        renderer_source = (ROOT / "clinic_utilities/physio_rich_referral.py").read_text(encoding="utf-8")
        self.assertNotIn('route_id == "lateral_elbow_tendinopathy"', formatter_source)
        self.assertNotIn('_format_lateral_elbow', formatter_source)
        self.assertNotIn('lateral_elbow_tendinopathy', renderer_source)

    def test_every_configured_rich_route_exists_in_registry_and_evidence_profiles_resolve(self):
        registry_profiles = self.bundle.registry.get("profiles") or {}
        known_routes = {
            route_id
            for profile in registry_profiles.values()
            for route_id in ((profile or {}).get("routes") or {})
        }
        known_profiles = set(self.evidence.route_evidence_profiles)
        known_claims = set(self.evidence.claims)

        for route_id, spec in self.renderer.contract_route_specs().items():
            with self.subTest(route=route_id):
                self.assertIn(route_id, known_routes)
                self.assertTrue(spec.get("profile_ids"))
                for evidence_profile_id in spec.get("evidence_profile_ids") or []:
                    self.assertIn(evidence_profile_id, known_profiles)
                for stage in spec.get("stages") or []:
                    self.assertTrue(stage.get("goals_el"))
                    self.assertTrue(stage.get("intervention_directions_el"))
                    self.assertTrue(stage.get("progress_markers_el"))
                    for claim_id in stage.get("evidence_claim_ids") or []:
                        self.assertIn(claim_id, known_claims)

    def test_approved_let_meaning_survives_shared_renderer_migration(self):
        draft = lateral_elbow_draft()
        short = self.formatter.format(draft, "short")
        detailed = self.formatter.format(draft, "detailed")

        for text in (short, detailed):
            lower = text.lower()
            self.assertIn("ισομετρική", lower)
            self.assertIn("ομόκεντρη", lower)
            self.assertIn("έκκεντρη", lower)
            self.assertIn("επανένταξη", lower)
            self.assertIn("μείωση κινδύνου υποτροπής", lower)
            self.assertIn("διαχείριση φορτίου", lower)
            self.assertLessEqual(len(text.rstrip("\n")), self.renderer.max_chars)

        self.assertIn("ΣΤΑΔΙΟ 1", detailed)
        self.assertIn("ΣΤΑΔΙΟ 2", detailed)
        self.assertIn("ΣΤΑΔΙΟ 3", detailed)
        self.assertNotIn("ΣΤΑΔΙΟ 1", short)

    def test_rich_renderer_overflow_fails_closed_instead_of_clipping_safety_tail(self):
        with self.assertRaisesRegex(Exception, "exceeds 2000 characters"):
            self.renderer.render_detailed(
                profile_id="elbow",
                route_id="lateral_elbow_tendinopathy",
                subtype_id=None,
                clinical_context=["πολύ μεγάλο κλινικό πλαίσιο " * 200],
            )

    def test_absent_route_is_not_given_generic_rich_content(self):
        self.assertFalse(
            self.renderer.supports(
                profile_id="cervical",
                route_id="nonspecific_neck_pain",
                subtype_id=None,
            )
        )


if __name__ == "__main__":
    unittest.main()
