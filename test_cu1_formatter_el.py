from __future__ import annotations

import re
import unittest
from pathlib import Path

from clinic_utilities.physio_referral_formatter_el_v2 import CU1GreekReferralFormatter
from clinic_utilities.physio_referral_runtime import CONTRACT_VERSION, CU1ContractBundle, CU1ContractError


ROOT = Path(__file__).resolve().parent
GREEK_RE = re.compile(r"[Α-Ωα-ωΆΈΉΊΌΎΏάέήίόύώϊϋΐΰ]")


def base_draft(profile_id: str, route_id: str, *, laterality: str = "not_applicable", wording_mode: str = "presentation"):
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
            "problem_id": "synthetic-format-problem",
            "profile_id": profile_id,
            "route_id": route_id,
            "wording_mode": wording_mode,
            "formal_assertion_state_optional": None,
            "subtype_id_optional": None,
            "laterality": laterality,
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


class CU1GreekFormatterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = CU1ContractBundle(ROOT)
        cls.formatter = CU1GreekReferralFormatter(cls.bundle)
        cls.language = cls.formatter.language

    def assert_no_machine_leak(self, text: str):
        self.assertNotIn("_", text)
        self.assertTrue(GREEK_RE.search(text))
        for section in (
            "findings",
            "functional_impairments",
            "goals",
            "rehab_directions",
            "adjuncts",
            "measurements",
            "restrictions",
        ):
            mapping = self.language.get(section, {})
            if isinstance(mapping, dict):
                for canonical_id in mapping:
                    self.assertNotIn(canonical_id, text)

    def test_every_registry_route_has_greek_display(self):
        labels = self.formatter.contract_route_labels()
        profiles = self.bundle.registry["profiles"]
        for profile_id, profile in profiles.items():
            for route_id in profile.get("routes", {}):
                with self.subTest(profile=profile_id, route=route_id):
                    label = labels[profile_id][route_id]
                    self.assertTrue(GREEK_RE.search(label), label)
                    self.assertNotIn("_", label)

    def test_all_selectable_option_ids_have_greek_phrase(self):
        options = self.bundle.options
        expected = {
            "findings": set(options.get("common_findings", [])),
            "functional_impairments": set(options.get("common_functional_impairments", [])),
            "goals": set(options.get("common_goal_ids", [])),
            "rehab_directions": set(options.get("common_rehab_direction_ids", [])),
            "adjuncts": set(options.get("adjunct_ids", [])),
            "measurements": set(options.get("measurement_ids", [])),
            "restrictions": set(options.get("restriction_ids", [])),
        }
        for values in (options.get("profile_findings", {}) or {}).values():
            expected["findings"].update(values)

        for section, ids in expected.items():
            mapping = self.language.get(section, {})
            self.assertIsInstance(mapping, dict)
            for canonical_id in ids:
                with self.subTest(section=section, canonical_id=canonical_id):
                    phrase = mapping.get(canonical_id)
                    self.assertIsInstance(phrase, str)
                    self.assertTrue(phrase.strip())
                    self.assertTrue(GREEK_RE.search(phrase), phrase)
                    self.assertNotIn("_", phrase)

    def test_short_cervical_referral_is_compact_natural_greek(self):
        draft = base_draft("cervical", "nonspecific_neck_pain", laterality="right")
        draft["findings"] = [{"finding_id": "axial_cervical_pain"}]
        draft["functional_impairments"] = [{"id": "desk_or_computer_work"}]
        draft["goals"] = [{"id": "improve_strength"}, {"id": "restore_safe_functional_rom"}]
        draft["rehab_directions"] = [
            {"id": "therapeutic_exercise"},
            {"id": "progressive_strengthening"},
        ]
        text = self.formatter.format(draft, "short")
        lower = text.lower()
        self.assert_no_machine_leak(text)
        self.assertIn("αυχ", lower)
        self.assertIn("προτείνεται ενεργητική αποκατάσταση", lower)
        self.assertIn("αυτοδιαχείρι", lower)
        self.assertIn("δραστηρι", lower)
        self.assertNotIn("ΚΛΙΝΙΚΗ ΕΙΚΟΝΑ", text)
        self.assertNotIn("ΣΤΑΔΙΟ", text)
        self.assertLessEqual(len(text.rstrip("\n")), 850)

    def test_rich_knee_short_and_detailed_are_materially_different(self):
        draft = base_draft(
            "knee",
            "knee_osteoarthritis",
            laterality="right",
            wording_mode="established_structural_diagnosis",
        )
        draft["findings"] = [
            {"finding_id": "joint_line_pain"},
            {"finding_id": "quadriceps_weakness"},
        ]
        draft["functional_impairments"] = [{"id": "walking_tolerance"}, {"id": "stairs"}]
        draft["goals"] = [{"id": "improve_strength"}, {"id": "improve_walking_tolerance"}]
        draft["rehab_directions"] = [
            {"id": "progressive_strengthening"},
            {"id": "graded_activity_exposure"},
            {"id": "neuromuscular_proprioceptive_training"},
        ]
        draft["measurements"] = [{"measurement_id": "five_times_sit_to_stand", "value": 14.2, "unit_optional": "s"}]
        draft["clinician_free_text_optional"] = "Επανεκτίμηση ανάλογα με τη λειτουργική πρόοδο."

        short = self.formatter.format(draft, "short")
        detailed = self.formatter.format(draft, "detailed")
        self.assert_no_machine_leak(short)
        self.assert_no_machine_leak(detailed)
        self.assertNotEqual(short, detailed)
        self.assertNotIn("ΠΑΡΑΠΟΜΠΗ ΓΙΑ ΦΥΣΙΟΘΕΡΑΠΕΙΑ", short)
        self.assertIn("ΠΑΡΑΠΟΜΠΗ ΓΙΑ ΦΥΣΙΟΘΕΡΑΠΕΙΑ", detailed)
        self.assertIn("Κλινική εικόνα", detailed)
        self.assertIn("Στόχοι και κατευθύνσεις αποκατάστασης", detailed)
        self.assertIn("Μετρήσεις", detailed)
        self.assertGreater(len(detailed), len(short) + 80)

    def test_shared_fracture_restriction_visible_in_both_modes(self):
        draft = base_draft(
            "shared_fracture",
            "fracture_rehabilitation_post_immobilization",
            laterality="left",
            wording_mode="established_structural_diagnosis",
        )
        draft["primary_problem"]["context"] = {
            "fracture_site": "distal_radius_fracture",
            "fracture_phase": "post_immobilization",
            "healing_stability_status": "healing_confirmed_with_restrictions",
        }
        draft["findings"] = [{"finding_id": "post_immobilization_stiffness"}]
        draft["explicit_restrictions"] = [
            {
                "restriction_id": "upper_limb_use_status",
                "state_or_value": "light_use_with_explicit_limit",
                "source": "clinician_entered",
            }
        ]
        draft["rehab_directions"] = [{"id": "protected_rom_within_restrictions"}]
        short = self.formatter.format(draft, "short")
        detailed = self.formatter.format(draft, "detailed")
        for text in (short, detailed):
            self.assert_no_machine_leak(text)
            self.assertIn("περιορισ", text.lower())
            self.assertIn("ελαφρά χρήση", text.lower())
        self.assertIn("Πρόσθετα κλινικά στοιχεία", detailed)
        self.assertIn("κάταγμα περιφερικού άκρου κερκίδας", detailed.lower())

    def test_shared_muscle_context_is_greek_in_detailed_only(self):
        draft = base_draft(
            "shared_muscle_myotendinous",
            "acute_muscle_myotendinous_injury_rehabilitation",
            laterality="right",
            wording_mode="established_structural_diagnosis",
        )
        draft["primary_problem"]["context"] = {
            "muscle_group": "hamstring_muscle_injury",
            "injury_phase": "progressive_loading",
            "injury_type": "confirmed_partial_muscle_or_myotendinous_tear",
            "management_context": "conservative_rehabilitation",
        }
        draft["findings"] = [{"finding_id": "pain_with_resisted_contraction"}]
        draft["rehab_directions"] = [{"id": "graded_loading"}]
        short = self.formatter.format(draft, "short")
        detailed = self.formatter.format(draft, "detailed")
        self.assert_no_machine_leak(short)
        self.assert_no_machine_leak(detailed)
        self.assertNotIn("φάση προοδευτικής φόρτισης", short)
        self.assertIn("φάση προοδευτικής φόρτισης", detailed)
        self.assertIn("κάκωση οπίσθιων μηριαίων", detailed)

    def test_not_assessed_is_never_rendered_as_normal(self):
        draft = base_draft("cervical", "neck_pain_with_radiating_upper_limb_symptoms", laterality="right")
        draft["primary_problem"]["context"] = {
            "neurological_screen": {
                "motor": "not_assessed",
                "sensory": "normal",
                "reflexes": "normal",
            }
        }
        draft["findings"] = [{"finding_id": "radiating_upper_limb_pain"}]
        text = self.formatter.format(draft, "detailed")
        self.assert_no_machine_leak(text)
        self.assertNotIn("not assessed", text.lower())
        self.assertNotIn("δεν διαπιστώθηκε νευρολογικό έλλειμμα", text.lower())
        self.assertIn("αισθητικότητα φυσιολογικό", text.lower())

    def test_missing_required_greek_label_fails_closed(self):
        draft = base_draft("cervical", "nonspecific_neck_pain")
        draft["findings"] = [{"finding_id": "synthetic_unmapped_finding"}]
        with self.assertRaises(CU1ContractError):
            self.formatter.format(draft, "short")


if __name__ == "__main__":
    unittest.main()
