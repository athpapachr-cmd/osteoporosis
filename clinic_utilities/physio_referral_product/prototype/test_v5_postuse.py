"""Focused post-use regressions for Knee-OA v5 prose/state corrections."""
from __future__ import annotations

import sys
import unittest
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from clinic_utilities.physio_referral_product.prototype import server as p


def request() -> dict:
    return {
        "draft_id": str(uuid.uuid4()),
        "revision": 11,
        "package_version": p.PACKAGE,
        "synthetic_only": True,
        "state": {
            "laterality": "right",
            "formal_assertion_state": "yes",
            "findings": [],
            "functional_impairments": [],
            "rehab_directions": list(p.E["default_plan"]["selected"]),
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


def presented(req: dict) -> dict:
    return p.present_project_result(p.project(req), req)


class V5PostUseProjectionTests(unittest.TestCase):
    def test_low_information_output_stays_compact(self):
        result = presented(request())
        self.assertTrue(result["gate"]["allowed"])
        self.assertIn("Παρακαλώ για φυσιοθεραπευτική αξιολόγηση", result["text"])
        self.assertNotIn("\n\n", result["text"])
        self.assertNotIn("Επιπλέον στόχος:", result["text"])
        self.assertNotIn("Επιπλέον στόχοι:", result["text"])

    def test_generic_symptoms_remain_valid_without_refinement(self):
        req = request()
        req["state"]["findings"] = ["pain"]
        req["state"]["phenotype"] = {
            "stiffness_symptom": True,
            "weakness_symptom_or_context": True,
        }
        result = presented(req)
        self.assertTrue(result["gate"]["allowed"])
        self.assertIn("πόνο", result["text"])
        self.assertIn("δυσκαμψία", result["text"])
        self.assertIn("μυϊκή αδυναμία", result["text"])
        self.assertNotIn("κατά την εξέταση", result["text"])

    def test_specific_pain_qualifiers_own_location_and_remove_legacy_duplicate(self):
        req = request()
        req["state"]["findings"] = ["pain", "joint_line_pain"]
        req["state"]["qualifiers"] = {
            "pain_locations": ["medial_joint_line", "pes_anserine_region"],
        }
        result = presented(req)
        text = result["text"]
        self.assertIn("πόνο στην έσω μεσάρθρια περιοχή και στην περιοχή του χηνείου ποδός", text)
        self.assertNotIn("πόνο στη μεσάρθρια γραμμή", text)
        self.assertNotIn("χηνείου ποδός στη μεσάρθρια γραμμή", text)
        self.assertNotIn("κυρίως", text)
        self.assertNotIn("joint_line_pain", result["state"]["findings"])

    def test_user_reported_rich_case_has_two_paragraphs_and_human_goal_sentence(self):
        req = request()
        req["state"]["findings"] = [
            "pain", "joint_line_pain", "effusion", "tenderness",
            "subjective_giving_way", "recurrent_instability_episode",
        ]
        req["state"]["phenotype"] = {
            "stiffness_symptom": True,
            "weakness_symptom_or_context": True,
        }
        req["state"]["qualifiers"] = {
            "pain_locations": ["medial_joint_line", "pes_anserine_region"],
            "stiffness_patterns": ["morning", "after_inactivity"],
            "morning_stiffness_duration": "le_30",
            "visible_atrophy": True,
            "atrophy_location": "quadriceps",
            "focal_tenderness_locations": ["medial_joint_line"],
        }
        req["state"]["functional_impairments"] = [
            "walking_tolerance", "stairs", "sit_to_stand", "sport_gym",
        ]
        req["state"]["rehab_directions"] = [
            "therapeutic_exercise",
            "progressive_strengthening",
            "education_and_self_management",
            "graded_activity_exposure",
            "progressive_endurance_or_capacity_work",
            "gait_walking_practice",
            "functional_task_retraining",
        ]
        req["state"]["goals"] = ["maintain_or_regain_adl_independence"]

        result = presented(req)
        text = result["text"]

        self.assertIn("\n\nΠαρακαλώ για φυσιοθεραπευτική αξιολόγηση", text)
        first, second = text.split("\n\n", 1)
        self.assertIn("Η κλινική εικόνα", first)
        self.assertIn("Λειτουργικά,", first)
        self.assertNotIn("Παρακαλώ για φυσιοθεραπευτική αξιολόγηση", first)
        self.assertTrue(second.startswith("Παρακαλώ για φυσιοθεραπευτική αξιολόγηση"))
        self.assertIn("με έμφαση σε", second)
        self.assertIn(
            "Παράλληλα, στους λειτουργικούς στόχους περιλαμβάνεται και η διατήρηση ή ανάκτηση της ανεξαρτησίας στις καθημερινές δραστηριότητες.",
            second,
        )
        self.assertNotIn("Επιπλέον στόχος:", text)
        self.assertNotIn("Επιπλέον στόχοι:", text)
        self.assertNotIn("χηνείου ποδός στη μεσάρθρια γραμμή", text)
        self.assertNotIn("κυρίως", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
