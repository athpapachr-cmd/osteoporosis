from __future__ import annotations

import unittest
from pathlib import Path

from clinic_utilities.physio_referral_formatter_el_v2 import CU1GreekReferralFormatter
from clinic_utilities.physio_referral_runtime import CONTRACT_VERSION, CU1ContractBundle
from clinic_utilities.physio_route_context import CU1RouteContextEngine


ROOT = Path(__file__).resolve().parent


def frozen_draft(*, irritability="high", findings=None, functional_impairments=None):
    return {
        "contract_version": CONTRACT_VERSION,
        "patient_context": {
            "age_years_optional": None,
            "skeletal_maturity_optional": None,
            "sport_or_work_demand_optional": None,
            "relevant_medical_context_ids": [],
            "free_text_optional": None,
        },
        "body_region": "shoulder",
        "primary_problem": {
            "problem_id": "synthetic-frozen-shoulder-composition",
            "profile_id": "shoulder",
            "route_id": "adhesive_capsulitis_frozen_shoulder",
            "wording_mode": "formal_diagnosis",
            "formal_assertion_state_optional": "yes",
            "subtype_id_optional": None,
            "laterality": "right",
            "chronicity_or_phase_optional": None,
            "context": {
                "frozen_shoulder_scope": "primary_frozen_shoulder",
                "frozen_shoulder_irritability": irritability,
            },
            "shared_target_optional": None,
            "source_route_optional": None,
        },
        "secondary_problems": [],
        "findings": [
            {
                "finding_id": item,
                "state_optional": None,
                "laterality_optional": None,
                "value_optional": None,
                "unit_optional": None,
                "free_text_optional": None,
            }
            for item in (findings or [])
        ],
        "functional_impairments": [
            {"id": item, "selected": True, "notes_optional": None}
            for item in (functional_impairments or [])
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


class CU1ClinicalCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = CU1ContractBundle(ROOT)
        cls.engine = CU1RouteContextEngine(cls.bundle)
        cls.formatter = CU1GreekReferralFormatter(cls.bundle)

    def _validated(self, **kwargs):
        result = self.engine.validate(frozen_draft(**kwargs))
        self.assertFalse(result.validation_errors)
        self.assertFalse(result.formatter_blocked)
        return result.normalized_draft

    def test_realistic_frozen_case_has_exact_natural_short_and_detailed_output(self):
        draft = self._validated(
            findings=[
                "pain",
                "active_rom_restricted",
                "passive_rom_restricted",
                "painful_active_rom",
                "painful_passive_rom",
            ],
            functional_impairments=["overhead_activity", "lifting_carrying", "driving"],
        )

        expected_short = (
            "Ασθενής με συμφυτική θυλακίτιδα / παγωμένο ώμο δεξιά, με επώδυνο και περιορισμένο ενεργητικό "
            "και παθητικό εύρος κίνησης. Λειτουργικά αναφέρεται δυσκολία σε δραστηριότητες πάνω από το ύψος "
            "του ώμου, στην άρση ή μεταφορά φορτίου και στην οδήγηση. Κλινική ερεθιστικότητα: υψηλή. "
            "Παρακαλώ για εξατομικευμένη φυσιοθεραπευτική αποκατάσταση με έμφαση στη βελτίωση της κινητικότητας, "
            "του διαθέσιμου εύρους κίνησης και της λειτουργικής χρήσης του άνω άκρου. Η ένταση της άσκησης και "
            "των τεχνικών κινητοποίησης να προσαρμόζεται στην ανοχή των συμπτωμάτων και στην κλινική ανταπόκριση.\n"
        )
        expected_detailed = (
            "ΚΛΙΝΙΚΗ ΕΙΚΟΝΑ\n"
            "Ασθενής με συμφυτική θυλακίτιδα / παγωμένο ώμο δεξιά, με επώδυνο και περιορισμένο ενεργητικό και "
            "παθητικό εύρος κίνησης. Λειτουργικά αναφέρεται δυσκολία σε δραστηριότητες πάνω από το ύψος του ώμου, "
            "στην άρση ή μεταφορά φορτίου και στην οδήγηση. Κλινική ερεθιστικότητα: υψηλή.\n\n"
            "ΣΤΟΧΟΙ ΑΠΟΚΑΤΑΣΤΑΣΗΣ\n"
            "Βελτίωση του πόνου, του διαθέσιμου εύρους κίνησης και της λειτουργικής χρήσης του άνω άκρου, με σταδιακή "
            "επάνοδο στις περιορισμένες καθημερινές δραστηριότητες.\n\n"
            "ΚΑΤΕΥΘΥΝΣΗ ΦΥΣΙΟΘΕΡΑΠΕΙΑΣ\n"
            "Παρακαλώ για εξατομικευμένο πρόγραμμα ενεργητικής αποκατάστασης και ασκήσεων κινητικότητας. "
            "Τεχνικές κινητοποίησης μπορούν να χρησιμοποιηθούν συμπληρωματικά όταν κρίνεται κατάλληλο. "
            "Η ένταση και η πρόοδος της αποκατάστασης να προσαρμόζονται στην ανοχή των συμπτωμάτων, στο διαθέσιμο "
            "εύρος κίνησης και στη λειτουργική βελτίωση.\n\n"
            "ΕΠΑΝΕΚΤΙΜΗΣΗ\n"
            "Παρότι έχει προγραμματιστεί ιατρική επανεκτίμηση, συνιστάται επικοινωνία με τον θεράποντα ιατρό για "
            "νωρίτερη επανεκτίμηση σε περίπτωση επιδείνωσης, εμφάνισης νέων κλινικών ή τραυματικών στοιχείων ή "
            "άλλης ουσιώδους μεταβολής της κλινικής εικόνας.\n"
        )

        self.assertEqual(self.formatter.format(draft, "short"), expected_short)
        self.assertEqual(self.formatter.format(draft, "detailed"), expected_detailed)

    def test_partial_rom_selection_does_not_invent_passive_or_painful_rom(self):
        draft = self._validated(
            findings=["pain", "active_rom_restricted"],
            functional_impairments=["overhead_activity"],
        )
        short = self.formatter.format(draft, "short")
        self.assertIn("Ασθενής με συμφυτική θυλακίτιδα / παγωμένο ώμο δεξιά.", short)
        self.assertIn("Στην κλινική εικόνα καταγράφονται πόνος και περιορισμός ενεργητικού εύρους κίνησης.", short)
        self.assertIn("Λειτουργικά αναφέρεται δυσκολία σε δραστηριότητες πάνω από το ύψος του ώμου.", short)
        self.assertNotIn("παθητικού εύρους κίνησης", short)
        self.assertNotIn("επώδυνο και περιορισμένο ενεργητικό και παθητικό", short)
        self.assertNotIn("άρση ή μεταφορά φορτίου", short)
        self.assertNotIn("οδήγηση", short)

    def test_reviewed_fusion_subsumes_generic_pain_without_duplicate_serialization(self):
        draft = self._validated(
            findings=[
                "pain",
                "active_rom_restricted",
                "passive_rom_restricted",
                "painful_active_rom",
                "painful_passive_rom",
            ],
            functional_impairments=["overhead_activity", "lifting_carrying"],
        )
        short = self.formatter.format(draft, "short")
        self.assertEqual(short.count("επώδυνο και περιορισμένο ενεργητικό και παθητικό εύρος κίνησης"), 1)
        self.assertNotIn("περιορισμός ενεργητικού εύρους κίνησης", short)
        self.assertNotIn("περιορισμός παθητικού εύρους κίνησης", short)
        self.assertNotIn("επώδυνο ενεργητικό εύρος κίνησης", short)
        self.assertNotIn("επώδυνο παθητικό εύρος κίνησης", short)
        self.assertNotIn("με πόνο", short)
        self.assertIn(
            "Λειτουργικά αναφέρεται δυσκολία σε δραστηριότητες πάνω από το ύψος του ώμου και στην άρση ή μεταφορά φορτίου.",
            short,
        )

    def test_composer_is_shared_and_contains_no_frozen_route_branch(self):
        source = (ROOT / "clinic_utilities/physio_clinical_composition.py").read_text(encoding="utf-8")
        self.assertNotIn("adhesive_capsulitis_frozen_shoulder", source)
        self.assertNotIn("frozen_shoulder", source)
        contract = self.bundle.artifacts["clinical_composition_el"]
        self.assertTrue(contract["governance"]["selected_facts_only"])
        self.assertTrue(contract["governance"]["route_specific_python_branches_forbidden"])


if __name__ == "__main__":
    unittest.main()
