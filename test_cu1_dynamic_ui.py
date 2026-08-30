from __future__ import annotations

import unittest
from pathlib import Path

from clinic_utilities.physio_referral_formatter_el_v2 import CU1GreekReferralFormatter
from clinic_utilities.physio_referral_runtime import CONTRACT_VERSION, CU1ContractBundle, CU1Engine


ROOT = Path(__file__).resolve().parent
SECTIONS = ("findings", "functional_impairments", "goals", "rehab_directions", "adjuncts")


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
        cls.hierarchy = cls.bundle.artifacts["ui_relevance_hierarchy"]
        cls.options = cls.bundle.options
        cls.allowed = {
            "findings": set(cls.options.get("common_findings", [])) | {
                item
                for values in (cls.options.get("profile_findings", {}) or {}).values()
                for item in values
            },
            "functional_impairments": set(cls.options.get("common_functional_impairments", [])),
            "goals": set(cls.options.get("common_goal_ids", [])),
            "rehab_directions": set(cls.options.get("common_rehab_direction_ids", [])),
            "adjuncts": set(cls.options.get("adjunct_ids", [])),
        }

    def test_ui_scope_uses_only_existing_machine_option_ids(self):
        for profile_id, profile in (self.scope.get("profiles") or {}).items():
            for section, valid_ids in self.allowed.items():
                for item in profile.get(section, []) or []:
                    with self.subTest(profile=profile_id, section=section, item=item):
                        self.assertIn(item, valid_ids)

    def test_hierarchical_ui_rules_use_only_existing_option_ids(self):
        def validate_layer(name, layer):
            for operation in ("replace", "include", "exclude", "prioritize"):
                values = layer.get(operation) or {}
                for section, ids in values.items():
                    if section not in SECTIONS:
                        continue
                    for item in ids or []:
                        with self.subTest(layer=name, operation=operation, section=section, item=item):
                            self.assertIn(item, self.allowed[section])
            for subtype_id, subtype in (layer.get("subtypes") or {}).items():
                validate_layer(f"{name}.subtype.{subtype_id}", subtype)
            for index, variant in enumerate(layer.get("context_variants") or []):
                validate_layer(f"{name}.context.{index}", variant)

        self.assertEqual(
            self.hierarchy.get("resolution_order"),
            ["profile_base", "route", "subtype", "context_variant"],
        )
        for profile_id, routes in (self.hierarchy.get("routes") or {}).items():
            for route_id, route in routes.items():
                validate_layer(f"{profile_id}.{route_id}", route)

    def test_frozen_shoulder_is_narrowed_to_condition_relevant_controls(self):
        route = self.hierarchy["routes"]["shoulder"]["adhesive_capsulitis_frozen_shoulder"]
        replace = route["replace"]
        self.assertEqual(
            replace["findings"],
            [
                "pain",
                "night_or_sleep_disturbance",
                "active_rom_restricted",
                "passive_rom_restricted",
                "painful_active_rom",
                "painful_passive_rom",
            ],
        )
        self.assertEqual(
            replace["functional_impairments"],
            ["overhead_activity", "lifting_carrying", "driving", "patient_priority_activity"],
        )
        for forbidden in (
            "instability_apprehension_finding",
            "objective_abduction_weakness",
            "objective_external_rotation_weakness",
            "scapular_control_endurance_deficit",
        ):
            self.assertNotIn(forbidden, replace["findings"])
        self.assertEqual(replace["goals"], [])
        self.assertEqual(replace["rehab_directions"], [])
        self.assertEqual(replace["adjuncts"], [])

    def test_reviewed_shoulder_rich_routes_do_not_show_generic_goal_or_rehab_checklists(self):
        shoulder = self.hierarchy["routes"]["shoulder"]
        for route_id in (
            "adhesive_capsulitis_frozen_shoulder",
            "rotator_cuff_related_shoulder_pain",
            "glenohumeral_instability_dislocation",
            "glenohumeral_osteoarthritis",
        ):
            with self.subTest(route=route_id):
                replace = shoulder[route_id]["replace"]
                self.assertEqual(replace["goals"], [])
                self.assertEqual(replace["rehab_directions"], [])
                self.assertEqual(replace["adjuncts"], [])

    def test_frozen_irritability_context_changes_presentation_priority_not_clinical_authority(self):
        route = self.hierarchy["routes"]["shoulder"]["adhesive_capsulitis_frozen_shoulder"]
        variants = route["context_variants"]
        high = next(
            item for item in variants
            if item["match"]["context_equals"].get("frozen_shoulder_irritability") == "high"
        )
        low = next(
            item for item in variants
            if item["match"]["context_equals"].get("frozen_shoulder_irritability") == "low"
        )
        self.assertEqual(
            high["prioritize"]["findings"][:2],
            ["pain", "night_or_sleep_disturbance"],
        )
        self.assertEqual(
            low["prioritize"]["findings"][:2],
            ["active_rom_restricted", "passive_rom_restricted"],
        )
        self.assertNotIn("replace", high)
        self.assertNotIn("replace", low)

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
        self.assertIn("επικονδυλαλγία", text.lower())
        self.assertIn("ενεργητική", text.lower())
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

    def test_browser_source_has_progressive_disclosure_and_hierarchical_relevance(self):
        app_js = (ROOT / "static/clinic-utilities/physio-referral/app.js").read_text(encoding="utf-8")
        hierarchy_js = (ROOT / "static/clinic-utilities/physio-referral/hierarchical-relevance.js").read_text(encoding="utf-8")
        index = (ROOT / "static/clinic-utilities/physio-referral/index.html").read_text(encoding="utf-8")
        self.assertIn("ui_relevance_scope", app_js)
        self.assertIn("presentation','established_structural_diagnosis','postoperative','shared_structural", app_js)
        self.assertIn("els.contextCard.hidden = fields.length === 0", app_js)
        self.assertIn("clearSelectionsAndState", app_js)
        self.assertIn("ui_relevance_hierarchy", hierarchy_js)
        self.assertIn("route.subtypes", hierarchy_js)
        self.assertIn("route.context_variants", hierarchy_js)
        self.assertIn("subtypeSelect.addEventListener", hierarchy_js)
        self.assertIn("contextFields.addEventListener", hierarchy_js)
        self.assertIn("hierarchical-relevance.js", index)


if __name__ == "__main__":
    unittest.main()
