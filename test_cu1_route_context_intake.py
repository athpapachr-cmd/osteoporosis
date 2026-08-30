from __future__ import annotations

import unittest
from pathlib import Path

from clinic_utilities.physio_referral_formatter_el_v2 import CU1GreekReferralFormatter
from clinic_utilities.physio_referral_runtime import CONTRACT_VERSION, CU1ContractBundle, CU1ContractError
from clinic_utilities.physio_route_context import CU1RouteContextEngine, route_context_contract_payload


ROOT = Path(__file__).resolve().parent


def cervical_draft(route_id: str, *, wording_mode: str = "presentation", context=None, assertion=None):
    return {
        "contract_version": CONTRACT_VERSION,
        "patient_context": {
            "age_years_optional": None,
            "skeletal_maturity_optional": None,
            "sport_or_work_demand_optional": None,
            "relevant_medical_context_ids": [],
            "free_text_optional": None,
        },
        "body_region": "cervical",
        "primary_problem": {
            "problem_id": "synthetic-route-context",
            "profile_id": "cervical",
            "route_id": route_id,
            "wording_mode": wording_mode,
            "formal_assertion_state_optional": assertion,
            "subtype_id_optional": None,
            "laterality": "not_stated",
            "chronicity_or_phase_optional": None,
            "context": context or {},
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


class CU1RouteContextIntakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = CU1ContractBundle(ROOT)
        cls.engine = CU1RouteContextEngine(cls.bundle)
        cls.formatter = CU1GreekReferralFormatter(cls.bundle)
        cls.payload = route_context_contract_payload(cls.bundle)

    def test_route_context_contract_is_closed_and_fully_labelled(self):
        routes = self.payload.get("routes") or {}
        self.assertIn("headache_with_cervical_msk_features", routes)
        self.assertIn("cervical_dizziness_presentation", routes)
        self.assertIn("post_traumatic_neck_pain", routes)
        for route_id, route in routes.items():
            fields = route.get("fields") or {}
            for key, field in fields.items():
                with self.subTest(route=route_id, field=key):
                    self.assertEqual(field.get("type"), "enum")
                    self.assertTrue(field.get("label_el"))
                    values = field.get("values") or []
                    labels = field.get("value_labels_el") or {}
                    self.assertTrue(values)
                    self.assertEqual(set(values), set(labels))
                    self.assertTrue(all(isinstance(labels[value], str) and labels[value].strip() for value in values))

    def test_formal_cgh_context_is_runtime_valid_and_reaches_rich_renderer(self):
        draft = cervical_draft(
            "headache_with_cervical_msk_features",
            wording_mode="formal_diagnosis",
            assertion="yes",
            context={"formal_cervicogenic_headache_diagnosis": "yes"},
        )
        result = self.engine.validate(draft)
        self.assertFalse(result.validation_errors)
        self.assertFalse(result.formatter_blocked)
        text = self.formatter.format(result.normalized_draft, "short")
        self.assertIn("ρητά διαγνωσμένη αυχενογενή κεφαλαλγία", text.lower())

    def test_clinician_established_cervical_dizziness_is_runtime_valid_and_rich(self):
        draft = cervical_draft(
            "cervical_dizziness_presentation",
            wording_mode="formal_diagnosis",
            assertion="yes",
            context={"clinician_diagnosis_cervicogenic_dizziness": "yes"},
        )
        result = self.engine.validate(draft)
        self.assertFalse(result.validation_errors)
        self.assertFalse(result.formatter_blocked)
        text = self.formatter.format(result.normalized_draft, "short")
        self.assertIn("χωρίς υπόσχεση επίλυσης της ζάλης", text.lower())

    def test_route_context_key_cannot_leak_to_another_route(self):
        draft = cervical_draft(
            "nonspecific_neck_pain",
            context={"formal_cervicogenic_headache_diagnosis": "yes"},
        )
        result = self.engine.validate(draft)
        self.assertTrue(any(error.error_id == "invalid_context_key" for error in result.validation_errors))
        self.assertTrue(result.formatter_blocked)

    def test_route_context_enum_is_exact(self):
        draft = cervical_draft(
            "cervical_dizziness_presentation",
            wording_mode="formal_diagnosis",
            assertion="yes",
            context={"clinician_diagnosis_cervicogenic_dizziness": "probably"},
        )
        result = self.engine.validate(draft)
        self.assertTrue(any(error.error_id == "invalid_context_enum_value" for error in result.validation_errors))
        self.assertTrue(result.formatter_blocked)

    def test_c5_context_omission_blocks_generation_instead_of_legacy_fallback(self):
        empty = self.engine.validate(cervical_draft("post_traumatic_neck_pain"))
        self.assertTrue(any(error.error_id == "rich_referral_context_required" for error in empty.validation_errors))
        self.assertTrue(empty.formatter_blocked)
        gate_error = next(error for error in empty.validation_errors if error.error_id == "rich_referral_context_required")
        self.assertEqual(gate_error.metadata.get("reason"), "no_applicable_reviewed_rich_variant")
        self.assertIn(
            "primary_problem.context.trauma_mechanism_context",
            gate_error.metadata.get("required_context_paths", []),
        )
        with self.assertRaises(CU1ContractError):
            self.formatter.format(empty.normalized_draft, "short")

        explicit = cervical_draft(
            "post_traumatic_neck_pain",
            context={
                "trauma_mechanism_context": "whiplash_acceleration_deceleration",
                "temporal_phase": "recent_or_acute_within_12_weeks",
                "structural_status": "no_material_structural_injury_identified_by_clinician",
                "material_neurological_or_other_safety_concern": "no",
                "physiotherapy_considered_appropriate_by_clinician": "yes",
            },
        )
        explicit_result = self.engine.validate(explicit)
        self.assertFalse(explicit_result.validation_errors)
        self.assertFalse(explicit_result.formatter_blocked)
        self.assertEqual(
            explicit_result.normalized_draft["primary_problem"]["context"]["temporal_phase"],
            "recent_or_acute_within_12_weeks",
        )

    def test_c5_invalid_phase_fails_closed(self):
        draft = cervical_draft(
            "post_traumatic_neck_pain",
            context={"temporal_phase": "about_two_months"},
        )
        result = self.engine.validate(draft)
        self.assertTrue(any(error.error_id == "invalid_context_enum_value" for error in result.validation_errors))
        self.assertTrue(result.formatter_blocked)


if __name__ == "__main__":
    unittest.main()
