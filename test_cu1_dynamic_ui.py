from __future__ import annotations

import unittest
from pathlib import Path

from clinic_utilities.physio_referral_formatter_el_v2 import CU1GreekReferralFormatter
from clinic_utilities.physio_referral_runtime import CONTRACT_VERSION, CU1ContractBundle, CU1Engine


ROOT = Path(__file__).resolve().parent


def minimal_draft(profile_id: str, route_id: str, wording_mode: str = "presentation"):
    return {
        "contract_version": CONTRACT_VERSION,
        "patient_context": {
            "age_years_optional": None,
            "skeletal_maturity_optional": None,
            "sport_or_work_demand_optional": None,
            "relevant_medical_context_ids": [],
            "free_text_optional": None,
        },
        "body_region": profile_id,
        "primary_problem": {
            "problem_id": "synthetic-dynamic-ui",
            "profile_id": profile_id,
            "route_id": route_id,
            "wording_mode": wording_mode,
            "formal_assertion_state_optional": None,
            "subtype_id_optional": None,
            "laterality": "not_stated",
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


class CU1DynamicUIScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = CU1ContractBundle(ROOT)
        cls.engine = CU1Engine(cls.bundle)
        cls.formatter = CU1GreekReferralFormatter(cls.bundle)
        cls.scope = cls.bundle.artifacts["ui_relevance_scope"]
        cls.options = cls.bundle.options

    def test_ui_scope_uses_only_existing_machine_option_ids(self):
        allowed = {
            "findings": set(self.options.get("common_findings", [])) | {
                item
                for values in (self.options.get("profile_findings", {}) or {}).values()
                for item in values
            },
            "functional_impairments": set(self.options.get("common_functional_impairments", [])),
            "goals": set(self.options.get("common_goal_ids", [])),
            "rehab_directions": set(self.options.get("common_rehab_direction_ids", [])),
            "adjuncts": set(self.options.get("adjunct_ids", [])),
        }
        for profile_id, profile in (self.scope.get("profiles") or {}).items():
            for section, valid_ids in allowed.items():
                for item in profile.get(section, []) or []:
                    with self.subTest(profile=profile_id, section=section, item=item):
                        self.assertIn(item, valid_ids)

    def test_elbow_scope_excludes_lower_limb_and_cervical_only_controls(self):
        elbow = self.scope["profiles"]["elbow"]
        self.assertNotIn("walking_tolerance", elbow["functional_impairments"])
        self.assertNotIn("stairs", elbow["functional_impairments"])
        self.assertNotIn("selected_cervical_traction", elbow["adjuncts"])
        self.assertNotIn("progressive_weight_bearing_within_restrictions", elbow["rehab_directions"])
        self.assertIn("gripping", elbow["functional_impairments"])
        self.assertIn("lifting_carrying", elbow["functional_impairments"])

    def test_knee_scope_is_lower_limb_relevant_not_hand_relevant(self):
        knee = self.scope["profiles"]["knee"]
        for expected in ("walking_tolerance", "stairs", "squat"):
            self.assertIn(expected, knee["functional_impairments"])
        for forbidden in ("gripping", "pinch", "dexterity"):
            self.assertNotIn(forbidden, knee["functional_impairments"])

    def test_cervical_scope_does_not_show_lower_limb_loading(self):
        cervical = self.scope["profiles"]["cervical"]
        self.assertNotIn("progressive_weight_bearing_within_restrictions", cervical["rehab_directions"])
        self.assertNotIn("stairs", cervical["functional_impairments"])

    def test_routine_elbow_referral_generates_without_optional_sections(self):
        draft = minimal_draft("elbow", "lateral_elbow_tendinopathy", "presentation")
        validation = self.engine.validate(draft)
        self.assertFalse(validation.formatter_blocked, validation.validation_errors)
        self.assertEqual(validation.validation_errors, [])
        text = self.formatter.format(validation.normalized_draft, "short")
        self.assertIn("φυσιοθεραπευτική", text.lower())
        self.assertNotIn("_", text)

    def test_optional_findings_goals_and_rehab_are_not_route_requirements(self):
        required = self.bundle.route_requirements["base_requirements"]["required_fields"]
        self.assertNotIn("findings", required)
        self.assertNotIn("functional_impairments", required)
        self.assertNotIn("goals", required)
        self.assertNotIn("rehab_directions", required)
        self.assertNotIn("adjunct_options", required)

    def test_structural_safety_requirement_is_not_removed(self):
        draft = minimal_draft(
            "shoulder",
            "confirmed_full_thickness_rotator_cuff_tear_nonoperative",
            "established_structural_diagnosis",
        )
        validation = self.engine.validate(draft)
        error_ids = {item.error_id for item in validation.validation_errors}
        self.assertIn("established_structural_diagnosis_source_required", error_ids)
        self.assertIn("established_nonoperative_management_context_required", error_ids)
        self.assertTrue(validation.formatter_blocked)

    def test_browser_source_has_progressive_disclosure_and_safe_wording_default(self):
        js = (ROOT / "static/clinic-utilities/physio-referral/app.js").read_text(encoding="utf-8")
        self.assertIn("ui_relevance_scope", js)
        self.assertIn("presentation','established_structural_diagnosis','postoperative','shared_structural", js)
        self.assertIn("els.contextCard.hidden = fields.length === 0", js)
        self.assertIn("clearSelectionsAndState", js)


if __name__ == "__main__":
    unittest.main()
