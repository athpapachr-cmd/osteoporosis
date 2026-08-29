from __future__ import annotations

import unittest
from pathlib import Path

from clinic_utilities.physio_referral_formatter_el_v2 import CU1GreekReferralFormatter
from clinic_utilities.physio_referral_runtime import CONTRACT_VERSION, CU1ContractBundle


ROOT = Path(__file__).resolve().parent
MAX_REFERRAL_CHARS = 2000


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
            "problem_id": "synthetic-let-rich-referral",
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
        "findings": [
            {"finding_id": "lateral_elbow_pain"},
            {"finding_id": "pain_with_gripping"},
        ],
        "functional_impairments": [
            {"id": "gripping"},
            {"id": "manual_work"},
        ],
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


class CU1RichLateralElbowReferralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = CU1ContractBundle(ROOT)
        cls.formatter = CU1GreekReferralFormatter(cls.bundle)

    def test_detailed_let_has_locked_stage_grammar_and_active_rehab(self):
        text = self.formatter.format(lateral_elbow_draft(), "detailed")

        for stage in ("ΣΤΑΔΙΟ 1", "ΣΤΑΔΙΟ 2", "ΣΤΑΔΙΟ 3"):
            self.assertIn(stage, text)
        self.assertGreaterEqual(text.count("Στόχοι:"), 3)
        self.assertGreaterEqual(text.count("Κατευθύνσεις:"), 3)
        self.assertGreaterEqual(text.count("Δείκτες:"), 3)
        self.assertGreaterEqual(text.count("Μετάβαση:"), 2)

        self.assertIn("ισομετρική", text.lower())
        self.assertIn("ομόκεντρη", text.lower())
        self.assertIn("έκκεντρη", text.lower())
        self.assertIn("ενεργητική κινητοποίηση", text.lower())
        self.assertIn("κρυοθεραπεία/tens", text.lower())
        self.assertIn("δεν υποκαθιστούν", text.lower())
        self.assertIn("λειτουργ", text.lower())
        self.assertIn("κομμώτρια", text.lower())
        self.assertLessEqual(len(text.rstrip("\n")), MAX_REFERRAL_CHARS)

    def test_detailed_let_does_not_encode_universal_dose_or_false_transition_threshold(self):
        text = self.formatter.format(lateral_elbow_draft(), "detailed")
        lower = text.lower()

        forbidden = (
            "2 lbs",
            "5 lbs",
            "10 lbs",
            "20 επαναλή",
            "3 σετ",
            "1–3 εβδο",
            "3–6 εβδο",
            "4–8 εβδο",
            "≤3/10",
            "90%",
            "25%",
        )
        for fragment in forbidden:
            self.assertNotIn(fragment.lower(), lower)
        self.assertNotRegex(lower, r"\b\d+\s*(kg|κιλ(?:ό|ά)|lbs)\b")
        self.assertNotRegex(lower, r"\b\d+\s*(σετ|επαναλήψεις)\b")
        self.assertIn("χωρίς καθολικά αριθμητικά κριτήρια", lower)

    def test_short_let_is_compact_but_preserves_methods_progression_and_passive_boundary(self):
        text = self.formatter.format(lateral_elbow_draft(), "short")
        lower = text.lower()

        self.assertNotIn("ΣΤΑΔΙΟ 1", text)
        self.assertIn("ισομετρική", lower)
        self.assertIn("ομόκεντρη", lower)
        self.assertIn("έκκεντρη", lower)
        self.assertIn("κρυοθεραπεία/tens", lower)
        self.assertIn("δεν υποκαθιστούν", lower)
        self.assertIn("επανένταξη", lower)
        self.assertIn("χωρίς προκαθορισμένα καθολικά αριθμητικά κριτήρια", lower)
        self.assertLessEqual(len(text.rstrip("\n")), MAX_REFERRAL_CHARS)

    def test_lateral_elbow_output_ceiling_survives_long_work_context(self):
        draft = lateral_elbow_draft()
        draft["patient_context"]["sport_or_work_demand_optional"] = "κομμώτρια με επαναλαμβανόμενη χρήση άνω άκρου " * 20
        for mode in ("short", "detailed"):
            text = self.formatter.format(draft, mode)
            self.assertLessEqual(len(text.rstrip("\n")), MAX_REFERRAL_CHARS)

    def test_non_let_routes_keep_existing_formatter_path(self):
        draft = lateral_elbow_draft()
        draft["body_region"] = "cervical"
        draft["primary_problem"].update(
            {
                "profile_id": "cervical",
                "route_id": "nonspecific_neck_pain",
                "laterality": "not_applicable",
                "wording_mode": "presentation",
            }
        )
        draft["findings"] = []
        draft["functional_impairments"] = []
        text = self.formatter.format(draft, "detailed")
        self.assertNotIn("ΣΤΑΔΙΟ 1", text)
        self.assertNotIn("κρυοθεραπεία/TENS", text)
        self.assertIn("Κλινική εικόνα", text)


if __name__ == "__main__":
    unittest.main()
