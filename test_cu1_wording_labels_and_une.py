from __future__ import annotations

import copy
import unittest
from pathlib import Path

from clinic_utilities.physio_referral_formatter_el_v2 import CU1GreekReferralFormatter
from clinic_utilities.physio_referral_runtime import CONTRACT_VERSION, CU1ContractBundle
from clinic_utilities.physio_route_context import CU1RouteContextEngine, route_context_contract_payload


ROOT = Path(__file__).resolve().parent


def draft_for(
    profile_id: str,
    route_id: str,
    *,
    wording_mode: str = "presentation",
    context=None,
    assertion=None,
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
            "problem_id": f"synthetic-{route_id}",
            "profile_id": profile_id,
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


MILD_UNE_CONTEXT = {
    "une_clinician_severity_context": "mild",
    "une_objective_ulnar_motor_status": "assessed_without_material_motor_deficit",
    "une_intrinsic_atrophy_or_clawing": "absent_when_assessed",
    "une_unresolved_alternative_localization_or_structural_owner": "no",
}


class CU1WordingLabelsAndUNETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = CU1ContractBundle(ROOT)
        cls.engine = CU1RouteContextEngine(cls.bundle)
        cls.formatter = CU1GreekReferralFormatter(cls.bundle)
        cls.renderer = cls.formatter.rich_renderer
        cls.route_context_payload = route_context_contract_payload(cls.bundle)

    def test_c3_presentation_and_formal_labels_are_distinct(self):
        presentation = draft_for("cervical", "headache_with_cervical_msk_features")
        presentation_result = self.engine.validate(presentation)
        self.assertFalse(presentation_result.validation_errors)
        presentation_text = self.formatter.format(presentation_result.normalized_draft, "short")
        self.assertIn("Κεφαλαλγία με αυχενικά μυοσκελετικά χαρακτηριστικά", presentation_text)
        self.assertNotIn("Αυχενογενής κεφαλαλγία.", presentation_text)

        formal = draft_for(
            "cervical",
            "headache_with_cervical_msk_features",
            wording_mode="formal_diagnosis",
            assertion="yes",
            context={"formal_cervicogenic_headache_diagnosis": "yes"},
        )
        formal_result = self.engine.validate(formal)
        self.assertFalse(formal_result.validation_errors)
        formal_text = self.formatter.format(formal_result.normalized_draft, "short")
        self.assertIn("Αυχενογενής κεφαλαλγία.", formal_text)

    def test_c4_presentation_fallback_stays_non_diagnostic_and_formal_is_diagnostic(self):
        presentation = draft_for("cervical", "cervical_dizziness_presentation")
        presentation_result = self.engine.validate(presentation)
        self.assertFalse(presentation_result.validation_errors)
        self.assertFalse(
            self.renderer.supports(
                profile_id="cervical",
                route_id="cervical_dizziness_presentation",
                context={"__wording_mode": "presentation"},
            )
        )
        presentation_text = self.formatter.format(presentation_result.normalized_draft, "short")
        self.assertIn("Ζάλη με αυχενικά μυοσκελετικά χαρακτηριστικά", presentation_text)
        self.assertNotIn("Αυχενογενής / αυχενικής προέλευσης ζάλη", presentation_text)

        formal = draft_for(
            "cervical",
            "cervical_dizziness_presentation",
            wording_mode="formal_diagnosis",
            assertion="yes",
            context={"clinician_diagnosis_cervicogenic_dizziness": "yes"},
        )
        formal_result = self.engine.validate(formal)
        self.assertFalse(formal_result.validation_errors)
        formal_text = self.formatter.format(formal_result.normalized_draft, "short")
        self.assertIn("Αυχενογενής / αυχενικής προέλευσης ζάλη.", formal_text)

    def test_une_context_fields_are_closed_and_browser_labelled(self):
        route = (self.route_context_payload.get("routes") or {}).get("ulnar_neuropathy_at_elbow") or {}
        fields = route.get("fields") or {}
        self.assertEqual(
            set(fields),
            {
                "une_clinician_severity_context",
                "une_objective_ulnar_motor_status",
                "une_intrinsic_atrophy_or_clawing",
                "une_unresolved_alternative_localization_or_structural_owner",
            },
        )
        for key, spec in fields.items():
            with self.subTest(field=key):
                self.assertEqual(spec.get("type"), "enum")
                self.assertTrue(spec.get("label_el"))
                values = spec.get("values") or []
                labels = spec.get("value_labels_el") or {}
                self.assertEqual(set(values), set(labels))

    def test_mild_une_presentation_uses_non_diagnostic_label_and_rich_sequence(self):
        draft = draft_for("elbow", "ulnar_neuropathy_at_elbow", context=copy.deepcopy(MILD_UNE_CONTEXT))
        result = self.engine.validate(draft)
        self.assertFalse(result.validation_errors)
        self.assertFalse(result.formatter_blocked)
        context = dict(MILD_UNE_CONTEXT, __wording_mode="presentation")
        self.assertTrue(
            self.renderer.supports(
                profile_id="elbow",
                route_id="ulnar_neuropathy_at_elbow",
                context=context,
            )
        )
        short = self.formatter.format(result.normalized_draft, "short")
        detailed = self.formatter.format(result.normalized_draft, "detailed")
        self.assertIn("Συμπτωματολογία ωλενίου νεύρου στην περιοχή του αγκώνα.", short)
        self.assertNotIn("Ωλένια νευροπάθεια στον αγκώνα / σύνδρομο κυβοειδούς σωλήνα", short)
        self.assertIn("πραγματικά καταγεγραμμένες θέσεις ή κινήσεις", short)
        self.assertIn("Δεν καθιερώνεται αυτόματα νυχτερινός νάρθηκας", short)
        self.assertLessEqual(len(detailed), self.renderer.standard_detailed_target_chars)

    def test_mild_une_formal_diagnosis_uses_clinician_asserted_label(self):
        draft = draft_for(
            "elbow",
            "ulnar_neuropathy_at_elbow",
            wording_mode="formal_diagnosis",
            assertion="yes",
            context=copy.deepcopy(MILD_UNE_CONTEXT),
        )
        result = self.engine.validate(draft)
        self.assertFalse(result.validation_errors)
        self.assertFalse(result.formatter_blocked)
        short = self.formatter.format(result.normalized_draft, "short")
        self.assertIn("Ωλένια νευροπάθεια στον αγκώνα / σύνδρομο κυβοειδούς σωλήνα.", short)
        self.assertNotIn("Συμπτωματολογία ωλενίου νεύρου στην περιοχή του αγκώνα", short)

    def test_une_incomplete_or_nonmild_context_never_receives_rich_sequence(self):
        cases = []
        for key in MILD_UNE_CONTEXT:
            candidate = copy.deepcopy(MILD_UNE_CONTEXT)
            candidate.pop(key)
            cases.append((f"missing_{key}", candidate))
        for key, value in (
            ("une_clinician_severity_context", "nonmild"),
            ("une_objective_ulnar_motor_status", "material_motor_deficit_present"),
            ("une_intrinsic_atrophy_or_clawing", "present"),
            ("une_unresolved_alternative_localization_or_structural_owner", "yes"),
        ):
            candidate = copy.deepcopy(MILD_UNE_CONTEXT)
            candidate[key] = value
            cases.append((f"unsafe_{key}", candidate))

        for name, context in cases:
            with self.subTest(case=name):
                rich_context = dict(context, __wording_mode="presentation")
                self.assertFalse(
                    self.renderer.supports(
                        profile_id="elbow",
                        route_id="ulnar_neuropathy_at_elbow",
                        context=rich_context,
                    )
                )

    def test_une_presentation_fallback_remains_non_diagnostic_when_rich_context_is_incomplete(self):
        draft = draft_for(
            "elbow",
            "ulnar_neuropathy_at_elbow",
            context={"une_clinician_severity_context": "mild"},
        )
        result = self.engine.validate(draft)
        self.assertFalse(result.validation_errors)
        self.assertFalse(result.formatter_blocked)
        text = self.formatter.format(result.normalized_draft, "short")
        self.assertIn("Συμπτωματολογία ωλενίου νεύρου στην περιοχή του αγκώνα", text)
        self.assertNotIn("Ωλένια νευροπάθεια στον αγκώνα / σύνδρομο κυβοειδούς σωλήνα", text)
        self.assertNotIn("Δεν καθιερώνεται αυτόματα νυχτερινός νάρθηκας", text)

    def test_une_unknown_context_enum_fails_closed(self):
        draft = draft_for(
            "elbow",
            "ulnar_neuropathy_at_elbow",
            context={"une_clinician_severity_context": "probably_mild"},
        )
        result = self.engine.validate(draft)
        self.assertTrue(any(error.error_id == "invalid_context_enum_value" for error in result.validation_errors))
        self.assertTrue(result.formatter_blocked)


if __name__ == "__main__":
    unittest.main()
