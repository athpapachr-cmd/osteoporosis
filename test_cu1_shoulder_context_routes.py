from __future__ import annotations

import copy
import unittest
from pathlib import Path

from clinic_utilities.physio_referral_formatter_el_v2 import CU1GreekReferralFormatter
from clinic_utilities.physio_referral_runtime import CONTRACT_VERSION, CU1ContractBundle
from clinic_utilities.physio_route_context import CU1RouteContextEngine, route_context_contract_payload


ROOT = Path(__file__).resolve().parent


def shoulder_draft(
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
        "body_region": "shoulder",
        "primary_problem": {
            "problem_id": f"synthetic-{route_id}",
            "profile_id": "shoulder",
            "route_id": route_id,
            "wording_mode": wording_mode,
            "formal_assertion_state_optional": assertion,
            "subtype_id_optional": None,
            "laterality": "right",
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


def instability_context(
    *,
    direction: str,
    cause: str,
    recurrence: str,
    management: str,
    structural: str = "clear_for_selected_rehabilitation",
):
    return {
        "shoulder_instability_direction": direction,
        "shoulder_instability_cause": cause,
        "shoulder_instability_recurrence": recurrence,
        "shoulder_instability_management_context": management,
        "shoulder_instability_structural_protocol_context": structural,
    }


class CU1ShoulderContextRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = CU1ContractBundle(ROOT)
        cls.engine = CU1RouteContextEngine(cls.bundle)
        cls.formatter = CU1GreekReferralFormatter(cls.bundle)
        cls.renderer = cls.formatter.rich_renderer
        cls.context_payload = route_context_contract_payload(cls.bundle)

    def _validate_and_format(self, draft, mode="detailed"):
        result = self.engine.validate(draft)
        self.assertFalse(result.validation_errors)
        self.assertFalse(result.formatter_blocked)
        return result, self.formatter.format(result.normalized_draft, mode)

    def test_shoulder_context_contract_is_browser_labelled_and_closed(self):
        routes = self.context_payload.get("routes") or {}
        instability = (routes.get("glenohumeral_instability_dislocation") or {}).get("fields") or {}
        self.assertEqual(
            set(instability),
            {
                "shoulder_instability_direction",
                "shoulder_instability_cause",
                "shoulder_instability_recurrence",
                "shoulder_instability_management_context",
                "shoulder_instability_structural_protocol_context",
            },
        )
        ghoa = (routes.get("glenohumeral_osteoarthritis") or {}).get("fields") or {}
        self.assertEqual(set(ghoa), {"ghoa_management_context"})
        for fields in (instability, ghoa):
            for key, spec in fields.items():
                with self.subTest(field=key):
                    self.assertEqual(spec.get("type"), "enum")
                    self.assertTrue(spec.get("label_el"))
                    values = spec.get("values") or []
                    labels = spec.get("value_labels_el") or {}
                    self.assertTrue(values)
                    self.assertEqual(set(values), set(labels))

    def test_anterior_traumatic_first_time_nonoperative_is_rich_and_presentation_safe(self):
        context = instability_context(
            direction="anterior",
            cause="traumatic",
            recurrence="first_time",
            management="nonoperative_rehabilitation",
        )
        result, detailed = self._validate_and_format(
            shoulder_draft("glenohumeral_instability_dislocation", context=context)
        )
        self.assertTrue(
            self.renderer.supports(
                profile_id="shoulder",
                route_id="glenohumeral_instability_dislocation",
                context=dict(context, __wording_mode="presentation"),
            )
        )
        short = self.formatter.format(result.normalized_draft, "short")
        self.assertIn("Συμπτωματολογία αστάθειας γληνοβραχιόνιας άρθρωσης", short)
        self.assertNotIn("Εξάρθρημα / αστάθεια γληνοβραχιόνιας άρθρωσης — αποκατάσταση", short)
        self.assertIn("πρώτο τραυματικό πρόσθιο επεισόδιο", short)
        self.assertLessEqual(len(detailed), self.renderer.standard_detailed_target_chars)

    def test_anterior_traumatic_recurrent_preoperative_selected_rehab_is_rich(self):
        context = instability_context(
            direction="anterior",
            cause="traumatic",
            recurrence="recurrent",
            management="preoperative_selected_rehabilitation",
        )
        _, short = self._validate_and_format(
            shoulder_draft("glenohumeral_instability_dislocation", context=context),
            mode="short",
        )
        self.assertIn("υποτροπιάζουσα τραυματική πρόσθια αστάθεια", short.lower())
        self.assertIn("δεν παρουσιάζεται ως υποκατάστατο", short.lower())

    def test_posterior_nonoperative_is_rich_for_explicit_resolved_context(self):
        context = instability_context(
            direction="posterior",
            cause="traumatic",
            recurrence="recurrent",
            management="nonoperative_rehabilitation",
        )
        _, detailed = self._validate_and_format(
            shoulder_draft("glenohumeral_instability_dislocation", context=context)
        )
        self.assertIn("οπίσθιου", detailed.lower())
        self.assertIn("στροφικού πετάλου", detailed.lower())
        self.assertIn("ψυχολογική ετοιμότητα", detailed.lower())
        self.assertLessEqual(len(detailed), self.renderer.standard_detailed_target_chars)

    def test_anterior_atraumatic_nonoperative_is_rich(self):
        context = instability_context(
            direction="anterior",
            cause="atraumatic",
            recurrence="recurrent",
            management="nonoperative_rehabilitation",
        )
        _, short = self._validate_and_format(
            shoulder_draft("glenohumeral_instability_dislocation", context=context),
            mode="short",
        )
        self.assertIn("ατραυματική πρόσθια αστάθεια", short.lower())
        self.assertIn("όχι με καθολικό χρονοδιάγραμμα", short.lower())

    def test_mdi_nonoperative_preserves_evidence_uncertainty(self):
        context = instability_context(
            direction="multidirectional",
            cause="atraumatic",
            recurrence="recurrent",
            management="nonoperative_rehabilitation",
        )
        _, detailed = self._validate_and_format(
            shoulder_draft("glenohumeral_instability_dislocation", context=context)
        )
        self.assertIn("αβεβαιότητα", detailed.lower())
        self.assertIn("αποτελεσματικότητας", detailed.lower())
        self.assertIn("αβέβαι", detailed.lower())
        self.assertLessEqual(len(detailed), self.renderer.standard_detailed_target_chars)

    def test_instability_incomplete_unsafe_or_wrong_owner_contexts_never_match_rich(self):
        valid = instability_context(
            direction="posterior",
            cause="traumatic",
            recurrence="recurrent",
            management="nonoperative_rehabilitation",
        )
        cases = []
        for key in valid:
            candidate = copy.deepcopy(valid)
            candidate.pop(key)
            cases.append((f"missing_{key}", candidate))
        for key, value in (
            ("shoulder_instability_direction", "not_stated"),
            ("shoulder_instability_cause", "not_stated"),
            ("shoulder_instability_recurrence", "not_stated"),
            ("shoulder_instability_management_context", "pending_specialist_assessment"),
            ("shoulder_instability_management_context", "postoperative_rehabilitation"),
            ("shoulder_instability_structural_protocol_context", "material_restriction_or_unresolved_structural_context"),
        ):
            candidate = copy.deepcopy(valid)
            candidate[key] = value
            cases.append((f"blocked_{key}_{value}", candidate))

        for name, context in cases:
            with self.subTest(case=name):
                self.assertFalse(
                    self.renderer.supports(
                        profile_id="shoulder",
                        route_id="glenohumeral_instability_dislocation",
                        context=dict(context, __wording_mode="presentation"),
                    )
                )

    def test_posterior_preoperative_context_does_not_borrow_anterior_authority(self):
        context = instability_context(
            direction="posterior",
            cause="traumatic",
            recurrence="recurrent",
            management="preoperative_selected_rehabilitation",
        )
        self.assertFalse(
            self.renderer.supports(
                profile_id="shoulder",
                route_id="glenohumeral_instability_dislocation",
                context=dict(context, __wording_mode="presentation"),
            )
        )

    def test_ghoa_nonoperative_is_rich_and_evidence_gap_aware(self):
        draft = shoulder_draft(
            "glenohumeral_osteoarthritis",
            wording_mode="formal_diagnosis",
            assertion="yes",
            context={"ghoa_management_context": "nonoperative"},
        )
        _, detailed = self._validate_and_format(draft)
        self.assertIn("Οστεοαρθρίτιδα γληνοβραχιόνιας άρθρωσης", detailed)
        self.assertIn("δεν τεκμηριώνουν υπεροχή συγκεκριμένης", detailed.lower())
        self.assertLessEqual(len(detailed), self.renderer.standard_detailed_target_chars)
        self.assertEqual(
            self.renderer.evidence_profile_ids(
                profile_id="shoulder",
                route_id="glenohumeral_osteoarthritis",
                context={"ghoa_management_context": "nonoperative", "__wording_mode": "formal_diagnosis"},
            ),
            ["rep_glenohumeral_oa_nonoperative_v1"],
        )

    def test_ghoa_preop_TSA_is_rich_but_does_not_generate_postop_protocol(self):
        draft = shoulder_draft(
            "glenohumeral_osteoarthritis",
            wording_mode="formal_diagnosis",
            assertion="yes",
            context={"ghoa_management_context": "preoperative_TSA"},
        )
        _, detailed = self._validate_and_format(draft)
        self.assertIn("ΠΡΟΕΓΧΕΙΡΗΤΙΚΗ ΛΕΙΤΟΥΡΓΙΚΗ ΠΡΟΕΤΟΙΜΑΣΙΑ", detailed)
        self.assertIn("δεν μετατρέπεται σε συγκεκριμένο υποχρεωτικό πρόγραμμα", detailed.lower())
        self.assertIn("postoperative shoulder route", detailed.lower())
        self.assertLessEqual(len(detailed), self.renderer.standard_detailed_target_chars)

    def test_ghoa_postoperative_or_unspecified_management_is_not_rich(self):
        for management in ("postoperative_arthroplasty", "not_stated"):
            with self.subTest(management=management):
                self.assertFalse(
                    self.renderer.supports(
                        profile_id="shoulder",
                        route_id="glenohumeral_osteoarthritis",
                        context={
                            "ghoa_management_context": management,
                            "__wording_mode": "formal_diagnosis",
                        },
                    )
                )

    def test_shoulder_context_enum_is_exact(self):
        draft = shoulder_draft(
            "glenohumeral_instability_dislocation",
            context={"shoulder_instability_direction": "mostly_anterior"},
        )
        result = self.engine.validate(draft)
        self.assertTrue(any(error.error_id == "invalid_context_enum_value" for error in result.validation_errors))
        self.assertTrue(result.formatter_blocked)


if __name__ == "__main__":
    unittest.main()
