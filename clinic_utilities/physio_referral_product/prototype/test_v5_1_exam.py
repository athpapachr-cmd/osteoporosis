from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from clinic_utilities.physio_referral_product.prototype import server as p
from clinic_utilities.physio_referral_product.prototype.test_server import request


def projected(*, qualifiers=None, findings=None, phenotype=None):
    req=request()
    req["state"]["qualifiers"]=copy.deepcopy(qualifiers or {})
    req["state"]["findings"]=list(findings or [])
    req["state"]["phenotype"]=copy.deepcopy(phenotype or {})
    return req,p.present_project_result(p.project(req),req)


class V51ExamSemanticsTests(unittest.TestCase):
    def test_three_directional_weakness_choices_are_distinct(self):
        cases={
            "knee_extension_exam":"αδυναμία έκτασης γόνατος / τετρακεφάλου κατά την εξέταση",
            "knee_flexion_exam":"αδυναμία κάμψης γόνατος / ισχιοκνημιαίων κατά την εξέταση",
            "knee_extension_flexion_exam":"αδυναμία κάμψης και έκτασης γόνατος κατά την εξέταση",
        }
        for value,expected in cases.items():
            with self.subTest(value=value):
                _,result=projected(
                    qualifiers={"weakness_detail":value},
                    phenotype={"weakness_symptom_or_context":True},
                )
                self.assertIn(expected,result["text"])
                self.assertTrue(result["gate"]["allowed"])
        _,extension=projected(qualifiers={"weakness_detail":"knee_extension_exam"},phenotype={"weakness_symptom_or_context":True})
        self.assertNotIn("ισχιοκνημιαίων",extension["text"])

    def test_directional_weakness_requires_explicit_weakness_context(self):
        req=request();req["state"]["qualifiers"]={"weakness_detail":"knee_flexion_exam"}
        with self.assertRaises(ValueError):
            p.project(req)

    def test_quadriceps_atrophy_is_independent_objective_finding(self):
        _,result=projected(qualifiers={"visible_atrophy":True,"atrophy_location":"quadriceps"})
        self.assertIn("εμφανής ατροφία τετρακεφάλου",result["text"])
        self.assertNotIn("μυϊκή αδυναμία",result["text"])

    def test_generic_rom_parent_does_not_invent_direction(self):
        _,result=projected(qualifiers={"rom_restriction_present":True})
        self.assertIn("περιορισμένο εύρος κίνησης",result["text"])
        self.assertNotIn("ενεργητικής κάμψης",result["text"])
        self.assertNotIn("παθητικής κάμψης",result["text"])
        self.assertNotIn("υστέρηση",result["text"])

    def test_active_and_passive_flexion_remain_distinct(self):
        _,active=projected(qualifiers={"rom_restriction_present":True,"active_flexion_restricted":True})
        self.assertIn("περιορισμό ενεργητικής κάμψης",active["text"])
        self.assertNotIn("περιορισμό παθητικής κάμψης",active["text"])
        _,passive=projected(qualifiers={"rom_restriction_present":True,"passive_flexion_restricted":True})
        self.assertIn("περιορισμό παθητικής κάμψης",passive["text"])
        self.assertNotIn("περιορισμό ενεργητικής κάμψης",passive["text"])
        _,both=projected(qualifiers={"rom_restriction_present":True,"active_flexion_restricted":True,"passive_flexion_restricted":True})
        self.assertIn("περιορισμό ενεργητικής και παθητικής κάμψης",both["text"])

    def test_extension_lag_and_passive_extension_deficit_do_not_collapse(self):
        _,lag=projected(qualifiers={"rom_restriction_present":True},findings=["extension_lag"])
        self.assertIn("υστέρηση έκτασης",lag["text"])
        _,ffd=projected(qualifiers={"rom_restriction_present":True,"fixed_flexion_deformity":True,"fixed_flexion_deformity_deg":12})
        self.assertIn("παθητικό έλλειμμα έκτασης 12°",ffd["text"])
        self.assertNotIn("υστέρηση έκτασης",ffd["text"])

    def test_crepitus_tenderness_and_objective_stability_render_without_diagnosis_inference(self):
        q={
            "crepitus":True,
            "focal_tenderness_locations":["medial_joint_line","lateral_bony","extensor_mechanism"],
            "stability_findings":["valgus_instability","anterior_instability_acl"],
        }
        _,result=projected(qualifiers=q)
        for phrase in [
            "κριγμός κατά την κίνηση",
            "αρθρική ευαισθησία έσω",
            "οστική ευαισθησία έξω",
            "ευαισθησία στον εκτατικό μηχανισμό",
            "αστάθεια σε βλαισότητα",
            "πρόσθια αστάθεια / ΠΧΣ",
        ]:
            self.assertIn(phrase,result["text"])
        self.assertNotIn("SIFK",result["text"])
        self.assertNotIn("SONK",result["text"])
        self.assertNotIn("giving",result["text"].lower())
        self.assertEqual(result["clinical_review_clues"],[])

    def test_bony_tenderness_alone_is_not_an_atypical_feature_trigger(self):
        _,result=projected(qualifiers={"focal_tenderness_locations":["medial_bony"]})
        self.assertIn("οστική ευαισθησία έσω",result["text"])
        self.assertEqual(result["clinical_review_clues"],[])
        self.assertTrue(result["gate"]["allowed"])

    def test_prolonged_morning_stiffness_remains_nonblocking_review_clue(self):
        _,result=projected(
            qualifiers={"stiffness_patterns":["morning"],"morning_stiffness_duration":"gt_30"},
            phenotype={"stiffness_symptom":True},
        )
        self.assertTrue(result["gate"]["allowed"])
        self.assertFalse(result["gate"]["blocked"])
        self.assertEqual([c["clue_id"] for c in result["clinical_review_clues"]],["morning_stiffness_over_30"])
        self.assertIn("πιθανό πρόσθετο ή εναλλακτικό αίτιο",result["clinical_review_clues"][0]["detail"])

    def test_new_exam_detail_does_not_auto_select_treatment(self):
        q={
            "rom_restriction_present":True,
            "active_flexion_restricted":True,
            "crepitus":True,
            "stability_findings":["varus_instability"],
        }
        _,result=projected(qualifiers=q)
        self.assertEqual(result["state"]["rehab_directions"],list(p.E["default_plan"]["selected"]))
        self.assertNotIn("mobility_exercise_when_restricted",result["state"]["rehab_directions"])
        self.assertEqual(result["state"]["adjunct_options"],[])


if __name__=="__main__":
    unittest.main(verbosity=2)
