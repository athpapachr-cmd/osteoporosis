from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from clinic_utilities.physio_referral_runtime import (
    CONTRACT_VERSION,
    CU1ContractBundle,
    CU1Engine,
    _require_clinical_key,
)


ROOT = Path(__file__).resolve().parent


def draft(
    profile_id: str,
    route_id: str,
    *,
    wording_mode: str = "presentation",
    laterality: str = "not_applicable",
    context=None,
    formal_assertion_state_optional=None,
    subtype_id_optional=None,
    findings=None,
    rehab_directions=None,
    adjunct_options=None,
    safety_flags=None,
    acknowledged_rule_ids=None,
    disposition="none_recorded",
):
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
            "problem_id": "synthetic-problem-1",
            "profile_id": profile_id,
            "route_id": route_id,
            "wording_mode": wording_mode,
            "formal_assertion_state_optional": formal_assertion_state_optional,
            "subtype_id_optional": subtype_id_optional,
            "laterality": laterality,
            "chronicity_or_phase_optional": None,
            "context": context or {},
            "shared_target_optional": None,
            "source_route_optional": None,
        },
        "secondary_problems": [],
        "findings": findings or [],
        "functional_impairments": [],
        "precautions": [],
        "explicit_restrictions": [],
        "goals": [],
        "rehab_directions": rehab_directions or [],
        "adjunct_options": adjunct_options or [],
        "measurements": [],
        "safety": {
            "input_flags": safety_flags or [],
            "acknowledged_rule_ids": acknowledged_rule_ids or [],
            "clinician_disposition": disposition,
        },
        "sessions_optional": None,
        "clinician_free_text_optional": None,
    }


class CU1ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = CU1ContractBundle(ROOT)
        cls.engine = CU1Engine(cls.bundle)

    def error_ids(self, result):
        return [item.error_id for item in result.validation_errors]

    def rule_ids(self, result):
        return [item.rule_id for item in result.safety_results]

    def test_manifest_loads_all_normative_yaml_and_applies_correction(self):
        self.assertIn("effective_route_requirements", self.bundle.artifacts)
        effective = self.bundle.route_requirements
        muscle = effective["shared_context_requirements"]["shared_muscle_myotendinous"]
        self.assertEqual(muscle["conditional_requirements"], [])
        self.assertEqual(
            muscle["safety_boundary"]["unresolved_major_avulsion_or_rupture"],
            "acute_unresolved_tendon_rupture_or_major_avulsion_concern",
        )

    def test_alias_normalization_runs_before_validation(self):
        raw = draft(
            "knee",
            "ACL_injury_instability_rehabilitation",
            wording_mode="established_structural_diagnosis",
            laterality="right",
        )
        result = self.engine.validate(raw)
        self.assertEqual(result.normalized_draft["primary_problem"]["route_id"], "acl_injury_instability_rehabilitation")
        self.assertNotIn("invalid_route_or_subtype", self.error_ids(result))

    def test_calf_pain_and_swelling_do_not_infer_dvt(self):
        raw = draft(
            "shared_muscle_myotendinous",
            "acute_muscle_myotendinous_injury_rehabilitation",
            wording_mode="presentation",
            context={
                "muscle_group": "calf_muscle_myotendinous_injury",
                "injury_phase": "acute",
                "injury_type": "clinically_assessed_muscle_strain",
                "management_context": "conservative_rehabilitation",
            },
            findings=[{"finding_id": "pain"}, {"finding_id": "swelling"}],
        )
        result = self.engine.validate(raw)
        self.assertNotIn("unresolved_dvt_concern", self.rule_ids(result))

    def test_explicit_dvt_flag_triggers_urgent_rule_and_blocks(self):
        raw = draft(
            "shared_muscle_myotendinous",
            "acute_muscle_myotendinous_injury_rehabilitation",
            wording_mode="presentation",
            context={
                "muscle_group": "calf_muscle_myotendinous_injury",
                "injury_phase": "acute",
                "injury_type": "clinically_assessed_muscle_strain",
                "management_context": "conservative_rehabilitation",
            },
            safety_flags=["dvt_concern_unresolved"],
        )
        result = self.engine.validate(raw)
        self.assertIn("unresolved_dvt_concern", self.rule_ids(result))
        item = next(item for item in result.safety_results if item.rule_id == "unresolved_dvt_concern")
        self.assertEqual(item.severity, "urgent_reassessment")
        self.assertTrue(item.formatter_blocked)
        self.assertTrue(result.formatter_blocked)

    def test_urgent_rule_unblocks_only_with_allowed_disposition(self):
        raw = draft(
            "shared_muscle_myotendinous",
            "acute_muscle_myotendinous_injury_rehabilitation",
            wording_mode="presentation",
            context={
                "muscle_group": "calf_muscle_myotendinous_injury",
                "injury_phase": "acute",
                "injury_type": "clinically_assessed_muscle_strain",
                "management_context": "conservative_rehabilitation",
            },
            safety_flags=["dvt_concern_unresolved"],
            disposition="urgent_or_same_day_assessment_arranged",
        )
        result = self.engine.validate(raw)
        item = next(item for item in result.safety_results if item.rule_id == "unresolved_dvt_concern")
        self.assertFalse(item.formatter_blocked)

    def test_adjunct_without_core_rehab_requires_acknowledgement(self):
        raw = draft(
            "lumbar",
            "nonspecific_low_back_pain",
            wording_mode="presentation",
            adjunct_options=[{"adjunct_id": "acupuncture", "selected": True, "provenance": "clinician_selected"}],
        )
        blocked = self.engine.validate(raw)
        self.assertIn("adjunct_without_core_rehabilitation", self.rule_ids(blocked))
        self.assertTrue(blocked.formatter_blocked)
        raw["safety"]["acknowledged_rule_ids"] = ["adjunct_without_core_rehabilitation"]
        acknowledged = self.engine.validate(raw)
        item = next(item for item in acknowledged.safety_results if item.rule_id == "adjunct_without_core_rehabilitation")
        self.assertFalse(item.formatter_blocked)

    def test_formal_diagnosis_requires_assertion_yes(self):
        raw = draft(
            "cervical",
            "neck_pain_with_radiating_upper_limb_symptoms",
            wording_mode="formal_diagnosis",
            laterality="right",
            context={"neurological_screen": {"motor": "normal", "sensory": "normal", "reflexes": "normal"}},
            formal_assertion_state_optional="not_stated",
        )
        result = self.engine.validate(raw)
        self.assertIn("formal_diagnosis_assertion_required", self.error_ids(result))
        self.assertTrue(result.formatter_blocked)

    def test_structural_rotator_cuff_route_requires_diagnosis_source(self):
        raw = draft(
            "shoulder",
            "confirmed_full_thickness_rotator_cuff_tear_nonoperative",
            wording_mode="established_structural_diagnosis",
            laterality="left",
            context={"management_context": "nonoperative_confirmed"},
        )
        result = self.engine.validate(raw)
        self.assertIn("established_structural_diagnosis_source_required", self.error_ids(result))

    def test_postoperative_shoulder_requires_exact_context(self):
        raw = draft(
            "shoulder",
            "postoperative_shoulder_rehabilitation",
            wording_mode="postoperative",
            laterality="right",
            context={"procedure": "rotator_cuff_repair", "protocol_status": "available"},
        )
        result = self.engine.validate(raw)
        errors = self.error_ids(result)
        self.assertIn("required_field_missing", errors)
        self.assertIn("postoperative_protocol_source_required", errors)
        missing_paths = {item.metadata.get("path") for item in result.validation_errors if item.error_id == "required_field_missing"}
        self.assertIn("primary_problem.context.procedure_date_or_phase", missing_paths)
        self.assertIn("primary_problem.context.restrictions_review_status", missing_paths)

    def test_lower_limb_fracture_missing_weight_bearing_is_safety_managed(self):
        raw = draft(
            "shared_fracture",
            "fracture_rehabilitation_post_immobilization",
            wording_mode="established_structural_diagnosis",
            laterality="right",
            context={
                "fracture_site": "lateral_malleolus_fracture",
                "fracture_phase": "early_healing",
                "treatment": "orif",
                "healing_stability_status": "healing_confirmed_with_restrictions",
                "immobilization_status": "boot",
                "rom_status": "rom_allowed_with_specific_limits",
                "loading_strengthening_status": "strengthening_allowed_with_limits",
            },
        )
        result = self.engine.validate(raw)
        self.assertIn("lower_limb_weight_bearing_required_missing", self.error_ids(result))
        self.assertIn("fracture_weight_bearing_not_stated_when_required", self.rule_ids(result))
        self.assertTrue(result.formatter_blocked)

    def test_upper_limb_fracture_requires_use_status_not_weight_bearing(self):
        raw = draft(
            "shared_fracture",
            "fracture_rehabilitation_post_immobilization",
            wording_mode="established_structural_diagnosis",
            laterality="left",
            context={
                "fracture_site": "distal_radius_fracture",
                "fracture_phase": "post_immobilization",
                "treatment": "nonoperative_observation_or_functional_treatment",
                "healing_stability_status": "documented_stable_for_current_rehabilitation",
                "immobilization_status": "none_currently",
                "rom_status": "rom_allowed_with_specific_limits",
                "loading_strengthening_status": "strengthening_allowed_with_limits",
            },
        )
        result = self.engine.validate(raw)
        self.assertIn("upper_limb_use_status_required_missing", self.error_ids(result))
        self.assertNotIn("lower_limb_weight_bearing_required_missing", self.error_ids(result))

    def test_shared_muscle_structural_injury_does_not_require_imaging_after_correction(self):
        raw = draft(
            "shared_muscle_myotendinous",
            "acute_muscle_myotendinous_injury_rehabilitation",
            wording_mode="established_structural_diagnosis",
            laterality="right",
            context={
                "muscle_group": "hamstring_muscle_injury",
                "injury_phase": "progressive_loading",
                "injury_type": "confirmed_partial_muscle_or_myotendinous_tear",
                "management_context": "conservative_rehabilitation",
            },
        )
        result = self.engine.validate(raw)
        self.assertNotIn("structural_muscle_injury_confirmation_state_required", self.error_ids(result))

    def test_frailty_formal_wording_requires_established_yes(self):
        raw = draft(
            "shared_deconditioning_balance_gait",
            "functional_deconditioning_balance_gait_rehabilitation",
            wording_mode="formal_diagnosis",
            context={"functional_route_id": "frailty_associated_functional_decline", "frailty_established": "not_stated"},
        )
        result = self.engine.validate(raw)
        self.assertIn("frailty_must_be_clinician_established_for_formal_wording", self.error_ids(result))

    def test_acl_post_reconstruction_cannot_use_nonoperative_route(self):
        raw = draft(
            "knee",
            "acl_injury_instability_rehabilitation",
            wording_mode="established_structural_diagnosis",
            laterality="right",
            context={"management_context": "post_reconstruction"},
        )
        result = self.engine.validate(raw)
        self.assertIn("postoperative_case_has_wrong_primary_route", self.error_ids(result))

    def test_unknown_safety_flag_fails_closed(self):
        raw = draft("lumbar", "nonspecific_low_back_pain", safety_flags=["made_up_safe_flag"])
        result = self.engine.validate(raw)
        self.assertIn("invalid_safety_input_flag", self.error_ids(result))
        self.assertTrue(result.formatter_blocked)

    def test_incomplete_neuro_screen_does_not_generate_reassuring_negative(self):
        raw = draft(
            "cervical",
            "neck_pain_with_radiating_upper_limb_symptoms",
            wording_mode="presentation",
            laterality="right",
            context={"neurological_screen": {"motor": "not_assessed", "sensory": "normal", "reflexes": "normal"}},
            rehab_directions=[{"id": "therapeutic_exercise", "selected": True}],
        )
        result = self.engine.validate(raw)
        self.assertIn("incomplete_neurological_screen_in_radicular_presentation", self.rule_ids(result))
        raw["safety"]["acknowledged_rule_ids"] = ["incomplete_neurological_screen_in_radicular_presentation"]
        generated = self.engine.generate(raw, "detailed")
        self.assertIsNotNone(generated.text)
        self.assertNotIn("no neurological deficit", generated.text.lower())
        self.assertNotIn("no red flags", generated.text.lower())
        self.assertNotIn("not assessed", generated.text.lower())

    def test_short_and_detailed_formatter_are_deterministic(self):
        raw = draft(
            "cervical",
            "nonspecific_neck_pain",
            wording_mode="presentation",
            laterality="right",
            findings=[{"finding_id": "axial_cervical_pain"}],
            rehab_directions=[{"id": "therapeutic_exercise", "selected": True}],
        )
        short_a = self.engine.generate(raw, "short").text
        short_b = self.engine.generate(raw, "short").text
        detailed = self.engine.generate(raw, "detailed").text
        self.assertEqual(short_a, short_b)
        self.assertIn("Κύριο πρόβλημα:", short_a)
        self.assertIn("Κατευθύνσεις αποκατάστασης:", short_a)
        self.assertIsNotNone(detailed)

    def test_browser_runtime_has_no_referral_storage_calls(self):
        js = (ROOT / "static/clinic-utilities/physio-referral/app.js").read_text(encoding="utf-8")
        runtime = (ROOT / "clinic_utilities/physio_referral_runtime.py").read_text(encoding="utf-8")
        self.assertNotIn("localStorage", js)
        self.assertNotIn("sessionStorage", js)
        self.assertNotIn("Session(", runtime)
        self.assertNotIn("sqlalchemy", runtime.lower())

    def test_protected_dependency_rejects_wrong_key(self):
        with patch.dict(os.environ, {"CLINICAL_DATA_KEY": "correct-key"}, clear=False):
            with self.assertRaises(HTTPException) as ctx:
                _require_clinical_key("wrong-key")
            self.assertEqual(ctx.exception.status_code, 401)
            _require_clinical_key("correct-key")


if __name__ == "__main__":
    unittest.main()
