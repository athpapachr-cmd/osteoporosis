from __future__ import annotations

import unittest
from pathlib import Path

from clinic_utilities.physio_evidence_runtime import CU1ClinicianEvidenceResolver
from clinic_utilities.physio_referral_formatter_el_v2 import CU1GreekReferralFormatter
from clinic_utilities.physio_referral_runtime import CONTRACT_VERSION, CU1ContractBundle
from clinic_utilities.physio_route_context import CU1RouteContextEngine


ROOT = Path(__file__).resolve().parent


def draft(*, wording_mode: str = "presentation", assertion=None):
    return {
        "contract_version": CONTRACT_VERSION,
        "patient_context": {
            "age_years_optional": None,
            "skeletal_maturity_optional": None,
            "sport_or_work_demand_optional": "εργασία με επαναλαμβανόμενη χρήση του άνω άκρου",
            "relevant_medical_context_ids": [],
            "free_text_optional": None,
        },
        "body_region": "shoulder",
        "primary_problem": {
            "problem_id": "synthetic-rcrsp",
            "profile_id": "shoulder",
            "route_id": "rotator_cuff_related_shoulder_pain",
            "wording_mode": wording_mode,
            "formal_assertion_state_optional": assertion,
            "subtype_id_optional": None,
            "laterality": "right",
            "chronicity_or_phase_optional": None,
            "context": {},
            "shared_target_optional": None,
            "source_route_optional": None,
        },
        "secondary_problems": [],
        "findings": [],
        "functional_impairments": [],
        "precautions": [],
        "explicit_restrictions": [],
        "goals": [],
        "rehab_directions": [],
        "adjunct_options": [],
        "measurements": [],
        "safety": {
            "input_flags": [],
            "acknowledged_rule_ids": [],
            "clinician_disposition": "none_recorded",
        },
        "sessions_optional": None,
        "clinician_free_text_optional": None,
    }


class CU1ShoulderRotatorCuffRichTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = CU1ContractBundle(ROOT)
        cls.engine = CU1RouteContextEngine(cls.bundle)
        cls.formatter = CU1GreekReferralFormatter(cls.bundle)
        cls.renderer = cls.formatter.rich_renderer
        cls.evidence = CU1ClinicianEvidenceResolver(ROOT)

    def _format(self, mode: str, *, wording_mode: str = "presentation", assertion=None) -> str:
        result = self.engine.validate(draft(wording_mode=wording_mode, assertion=assertion))
        self.assertFalse(result.validation_errors)
        self.assertFalse(result.formatter_blocked)
        return self.formatter.format(result.normalized_draft, mode)

    def test_rcrsp_is_promoted_through_shared_renderer(self):
        self.assertEqual(
            self.renderer.rollout_state(profile_id="shoulder", route_id="rotator_cuff_related_shoulder_pain"),
            "rich_ready",
        )
        self.assertTrue(
            self.renderer.supports(profile_id="shoulder", route_id="rotator_cuff_related_shoulder_pain")
        )
        self.assertEqual(
            self.renderer.evidence_profile_ids(
                profile_id="shoulder", route_id="rotator_cuff_related_shoulder_pain"
            ),
            ["rep_rotator_cuff_related_pain_v1"],
        )

    def test_short_and_detailed_preserve_active_rehabilitation_truth(self):
        short = self._format("short")
        detailed = self._format("detailed")
        for text in (short, detailed):
            lowered = text.lower()
            self.assertIn("ενεργητική αποκατάσταση", lowered)
            self.assertIn("κινητικού ελέγχου", lowered)
            self.assertIn("αντίστασης", lowered)
            self.assertIn("αυτοδιαχείρι", lowered)
            self.assertIn("φόρτι", lowered)
            self.assertLessEqual(len(text), self.renderer.max_chars)
        self.assertIn("ΣΤΑΔΙΟ 1", detailed)
        self.assertIn("ΣΤΑΔΙΟ 2", detailed)
        self.assertNotIn("ΣΤΑΔΙΟ", short)
        self.assertLessEqual(len(detailed), self.renderer.standard_detailed_target_chars)

    def test_referral_does_not_invent_universal_dose_or_import_clinician_only_12_week_rule(self):
        detailed = self._format("detailed")
        lowered = detailed.lower()
        self.assertNotIn("12 εβδο", lowered)
        self.assertNotIn("3 x", lowered)
        self.assertNotIn("3x", lowered)
        self.assertIn("χωρίς καθολική συνταγή", lowered)
        self.assertIn("χωρίς καθολικό αριθμητικό", lowered)

    def test_adjuncts_remain_secondary_and_calcific_specific_care_is_not_borrowed(self):
        detailed = self._format("detailed")
        lowered = detailed.lower()
        self.assertIn("επικουρικά μέσα", lowered)
        self.assertIn("όχι ως υποκατάστατο", lowered)
        self.assertNotIn("eswt", lowered)
        self.assertNotIn("κρουστικ", lowered)
        self.assertIn("ασβεστοποιό τενοντοπάθεια", lowered)
        self.assertIn("ξεχωριστό", lowered)

    def test_full_thickness_tear_cannot_inherit_rcrsp_rich_authority(self):
        self.assertEqual(
            self.renderer.rollout_state(
                profile_id="shoulder",
                route_id="confirmed_full_thickness_rotator_cuff_tear_nonoperative",
            ),
            "evidence_limited",
        )
        self.assertFalse(
            self.renderer.supports(
                profile_id="shoulder",
                route_id="confirmed_full_thickness_rotator_cuff_tear_nonoperative",
            )
        )
        detailed = self._format("detailed")
        self.assertIn("Ρήξη πλήρους πάχους δεν αντιμετωπίζεται", detailed)

    def test_clinician_evidence_retains_scope_and_12_week_reassessment_claim(self):
        data = self.evidence.route_summary(
            profile_id="shoulder",
            route_id="rotator_cuff_related_shoulder_pain",
        )
        self.assertTrue(data["has_applicable_profile"])
        self.assertEqual(data["profile_count"], 1)
        summaries = "\n".join(str(item.get("claim_summary") or "") for item in data["claims"]).lower()
        self.assertIn("active rehabilitation", summaries)
        self.assertIn("12-week", summaries)
        self.assertTrue(any(item.get("output_scope") == "clinician_ui_only" for item in data["claims"]))

    def test_formal_diagnosis_mode_remains_valid_without_structural_overreach(self):
        text = self._format("short", wording_mode="formal_diagnosis", assertion="yes")
        self.assertIn("στροφ", text.lower())
        self.assertNotIn("ρήξη πλήρους πάχους", text.lower())


if __name__ == "__main__":
    unittest.main()
