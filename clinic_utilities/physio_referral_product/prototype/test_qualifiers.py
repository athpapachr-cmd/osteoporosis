"""Focused tests for post-review Knee-OA amendments."""
from __future__ import annotations

import copy
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
        "revision": 7,
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


class QualifierProjectionTests(unittest.TestCase):
    def test_low_information_output_is_compact_and_preserves_core_priorities(self):
        req = request()
        base = p.render(p.T, p.clean_request(req)["state"], p.LANG)
        result = p.project(req)
        self.assertLess(len(result["text"]), len(base))
        self.assertIn("φυσιοθεραπευτική αξιολόγηση", result["text"])
        self.assertIn("θεραπευτική άσκηση", result["text"])
        self.assertIn("προοδευτική ενδυνάμωση", result["text"])
        self.assertIn("εκπαίδευση για αυτοδιαχείριση", result["text"])
        self.assertNotIn("Η κλινική εικόνα περιλαμβάνει", result["text"])

    def test_patient_specific_output_is_richer_than_low_information_output(self):
        minimal = p.project(request())["text"]
        req = request()
        req["state"]["findings"] = ["pain"]
        req["state"]["functional_impairments"] = ["stairs"]
        rich = p.project(req)["text"]
        self.assertGreater(len(rich), len(minimal))
        self.assertIn("Η κλινική εικόνα περιλαμβάνει", rich)
        self.assertIn("Λειτουργικά", rich)
        self.assertIn("ενδεικτικές προτεραιότητες", rich)
        self.assertIn("ανάλογα με τα ευρήματα της αξιολόγησης", rich)

    def test_pes_anserine_is_location_not_bursitis_diagnosis(self):
        req = request()
        req["state"]["findings"] = ["pain"]
        req["state"]["qualifiers"] = {"pain_locations": ["medial_joint_line", "pes_anserine_region"]}
        result = p.project(req)
        self.assertIn("έσω μεσάρθρια περιοχή", result["text"])
        self.assertIn("περιοχή του χηνείου ποδός", result["text"])
        self.assertNotIn("θυλακ", result["text"].lower())
        self.assertNotIn("burs", result["text"].lower())

    def test_stiffness_over_30_is_review_clue_not_block_or_treatment(self):
        req = request()
        req["state"]["phenotype"] = {"stiffness_symptom": True}
        req["state"]["qualifiers"] = {
            "stiffness_patterns": ["morning", "after_inactivity"],
            "morning_stiffness_duration": "gt_30",
        }
        before_plan = copy.deepcopy(req["state"]["rehab_directions"])
        result = p.project(req)
        self.assertTrue(result["gate"]["allowed"])
        self.assertFalse(result["gate"]["blocked"])
        self.assertEqual(result["state"]["rehab_directions"], before_plan)
        self.assertIn("πρωινή δυσκαμψία άνω των 30 λεπτών", result["text"])
        self.assertEqual(result["clinical_review_clues"][0]["clue_id"], "morning_stiffness_over_30")
        self.assertIn("σημείο για έλεγχο", result["readiness"]["label"])

    def test_quadriceps_exam_and_atrophy_are_explicit_and_refine_strengthening(self):
        req = request()
        req["state"]["phenotype"] = {"weakness_symptom_or_context": True}
        req["state"]["qualifiers"] = {
            "weakness_detail": "quadriceps_exam",
            "visible_atrophy": True,
            "atrophy_location": "quadriceps",
        }
        result = p.project(req)
        self.assertIn("quadriceps_weakness", result["state"]["findings"])
        self.assertIn("αδυναμία τετρακεφάλου με εμφανή ατροφία τετρακεφάλου", result["text"])
        self.assertIn("έμφαση στον τετρακέφαλο", result["text"])

    def test_generic_weakness_does_not_become_objective(self):
        req = request()
        req["state"]["phenotype"] = {"weakness_symptom_or_context": True}
        result = p.project(req)
        self.assertNotIn("objective_weakness", result["state"]["findings"])
        self.assertNotIn("quadriceps_weakness", result["state"]["findings"])
        self.assertIn("μυϊκή αδυναμία", result["text"])
        self.assertNotIn("αντικειμενικά", result["text"])

    def test_old_quadriceps_localisation_value_fails_closed(self):
        req = request()
        req["state"]["phenotype"] = {"weakness_symptom_or_context": True}
        req["state"]["qualifiers"] = {"weakness_detail": "quadriceps"}
        with self.assertRaises(ValueError):
            p.project(req)

    def test_fixed_flexion_is_passive_exam_finding_and_only_suggests_mobility(self):
        req = request()
        req["state"]["qualifiers"] = {"fixed_flexion_deformity": True, "fixed_flexion_deformity_deg": 10}
        result = p.project(req)
        self.assertIn("παθητικό έλλειμμα έκτασης 10°", result["text"])
        self.assertNotIn("μόνιμο", result["text"])
        self.assertNotIn("fixed flexion deformity", result["text"].lower())
        self.assertNotIn("δυσκαμψία", result["text"])
        mobility = [c for c in result["suggestions"] if c["item_id"] == "mobility_exercise_when_restricted"]
        self.assertEqual(len(mobility), 1)
        self.assertNotIn("mobility_exercise_when_restricted", result["state"]["rehab_directions"])

    def test_fixed_flexion_degree_zero_is_rejected_but_unknown_degree_is_allowed(self):
        req = request()
        req["state"]["qualifiers"] = {"fixed_flexion_deformity": True, "fixed_flexion_deformity_deg": 0}
        with self.assertRaises(ValueError):
            p.project(req)
        req = request()
        req["state"]["qualifiers"] = {"fixed_flexion_deformity": True, "fixed_flexion_deformity_deg": None}
        result = p.project(req)
        self.assertIn("παθητικό έλλειμμα έκτασης", result["text"])
        self.assertNotIn(" 0°", result["text"])

    def test_pes_anserine_tenderness_is_specific_without_diagnosis(self):
        req = request()
        req["state"]["qualifiers"] = {"focal_tenderness_locations": ["pes_anserine_region"]}
        result = p.project(req)
        self.assertIn("tenderness", result["state"]["findings"])
        self.assertIn("ευαισθησία στην περιοχή του χηνείου ποδός", result["text"])
        self.assertNotIn("θυλακ", result["text"].lower())
        self.assertNotIn("burs", result["text"].lower())

    def test_inconsistent_or_impossible_qualifiers_fail_closed(self):
        mutations = [
            {"pain_locations": ["diffuse", "pes_anserine_region"]},
            {"stiffness_patterns": [], "morning_stiffness_duration": "gt_30"},
            {"fixed_flexion_deformity": False, "fixed_flexion_deformity_deg": 15},
            {"visible_atrophy": False, "atrophy_location": "quadriceps"},
        ]
        for qualifiers in mutations:
            with self.subTest(qualifiers=qualifiers):
                req = request(); req["state"]["qualifiers"] = qualifiers
                with self.assertRaises(ValueError):
                    p.project(req)


if __name__ == "__main__":
    unittest.main(verbosity=2)
