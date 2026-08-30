from __future__ import annotations

import os
import unicodedata
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from clinic_utilities.physio_evidence_api import contextual_evidence_summary
from clinic_utilities.physio_evidence_runtime import CU1ClinicianEvidenceResolver
from clinic_utilities.physio_referral_api import build_cu1_physio_referral_router
from clinic_utilities.physio_referral_formatter_el_v2 import CU1GreekReferralFormatter
from clinic_utilities.physio_referral_runtime import CONTRACT_VERSION, CU1ContractBundle, CU1ContractError
from clinic_utilities.physio_rich_referral import CU1RichReferralRenderer
from clinic_utilities.physio_route_context import CU1RouteContextEngine, route_context_contract_payload


ROOT = Path(__file__).resolve().parent


def fold_el(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text.casefold())
        if unicodedata.category(char) != "Mn"
    )


def frozen_draft(
    *,
    wording_mode="formal_diagnosis",
    assertion="yes",
    scope="primary_frozen_shoulder",
    irritability=None,
    findings=None,
    functional_impairments=None,
):
    context = {} if scope is None else {"frozen_shoulder_scope": scope}
    if irritability is not None:
        context["frozen_shoulder_irritability"] = irritability
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
            "problem_id": "synthetic-frozen-shoulder",
            "profile_id": "shoulder",
            "route_id": "adhesive_capsulitis_frozen_shoulder",
            "wording_mode": wording_mode,
            "formal_assertion_state_optional": assertion,
            "subtype_id_optional": None,
            "laterality": "right",
            "chronicity_or_phase_optional": None,
            "context": context,
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


class CU1PrimaryFrozenShoulderRichTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = CU1ContractBundle(ROOT)
        cls.engine = CU1RouteContextEngine(cls.bundle)
        cls.formatter = CU1GreekReferralFormatter(cls.bundle)
        cls.renderer = CU1RichReferralRenderer(ROOT)
        cls.evidence = CU1ClinicianEvidenceResolver(ROOT)
        cls.context_contract = route_context_contract_payload(cls.bundle)

    def _validate(self, **kwargs):
        result = self.engine.validate(frozen_draft(**kwargs))
        self.assertFalse(result.validation_errors)
        self.assertFalse(result.formatter_blocked)
        return result

    def test_frozen_shoulder_context_fields_are_closed_and_browser_labelled(self):
        fields = self.context_contract["routes"]["adhesive_capsulitis_frozen_shoulder"]["fields"]
        self.assertEqual(set(fields), {"frozen_shoulder_scope", "frozen_shoulder_irritability"})

        scope = fields["frozen_shoulder_scope"]
        self.assertEqual(scope["type"], "enum")
        self.assertEqual(
            scope["values"],
            ["primary_frozen_shoulder", "secondary_or_other_stiff_shoulder", "not_stated"],
        )
        self.assertEqual(set(scope["values"]), set(scope["value_labels_el"]))
        self.assertEqual(scope["show_when"]["wording_modes"], ["formal_diagnosis"])

        irritability = fields["frozen_shoulder_irritability"]
        self.assertEqual(irritability["type"], "enum")
        self.assertEqual(
            irritability["values"],
            ["high", "moderate", "low", "uncertain_or_not_assessed"],
        )
        self.assertEqual(set(irritability["values"]), set(irritability["value_labels_el"]))
        self.assertFalse(irritability.get("required", False))
        self.assertFalse(irritability.get("required_for_rich_variant", False))

    def test_primary_formal_context_is_the_only_rich_variant(self):
        result = self._validate()
        rich_context = {"frozen_shoulder_scope": "primary_frozen_shoulder", "__wording_mode": "formal_diagnosis"}
        self.assertEqual(
            self.renderer.rollout_state(profile_id="shoulder", route_id="adhesive_capsulitis_frozen_shoulder"),
            "context_gated",
        )
        self.assertTrue(
            self.renderer.supports(
                profile_id="shoulder",
                route_id="adhesive_capsulitis_frozen_shoulder",
                context=rich_context,
            )
        )
        self.assertEqual(
            self.renderer.evidence_profile_ids(
                profile_id="shoulder",
                route_id="adhesive_capsulitis_frozen_shoulder",
                context=rich_context,
            ),
            ["rep_adhesive_capsulitis_v1"],
        )
        short = self.formatter.format(result.normalized_draft, "short")
        detailed = self.formatter.format(result.normalized_draft, "detailed")
        for text in (short, detailed):
            folded = fold_el(text)
            self.assertIn(fold_el("παγωμένος ώμος"), folded)
            self.assertIn("κινητικ", folded)
            self.assertIn(fold_el("εύρους κίνησης"), folded)
            self.assertLessEqual(len(text), self.renderer.max_chars)
        self.assertEqual(detailed.count("ΣΤΑΔΙΟ "), 0)
        for heading in ("ΚΛΙΝΙΚΗ ΕΙΚΟΝΑ", "ΣΤΟΧΟΙ ΑΠΟΚΑΤΑΣΤΑΣΗΣ", "ΚΑΤΕΥΘΥΝΣΗ ΦΥΣΙΟΘΕΡΑΠΕΙΑΣ", "ΕΠΑΝΕΚΤΙΜΗΣΗ"):
            self.assertIn(heading, detailed)
        self.assertLessEqual(len(detailed), self.renderer.standard_detailed_target_chars)

    def test_realistic_frozen_case_is_clinical_synthesis_not_checkbox_serializer(self):
        result = self._validate(
            irritability="high",
            findings=[
                "pain",
                "active_rom_restricted",
                "passive_rom_restricted",
                "painful_active_rom",
                "painful_passive_rom",
            ],
            functional_impairments=["overhead_activity", "lifting_carrying", "driving"],
        )
        short = self.formatter.format(result.normalized_draft, "short")
        detailed = self.formatter.format(result.normalized_draft, "detailed")
        for text in (short, detailed):
            folded = fold_el(text)
            self.assertIn(fold_el("Κλινική ερεθιστικότητα: υψηλή"), folded)
            self.assertIn(fold_el("επώδυνο και περιορισμένο ενεργητικό και παθητικό εύρος κίνησης"), folded)
            self.assertNotIn(fold_el("περιορισμός ενεργητικού εύρους κίνησης"), folded)
            self.assertNotIn(fold_el("περιορισμός παθητικού εύρους κίνησης"), folded)
            self.assertIn(fold_el("δραστηριότητες πάνω από το ύψος του ώμου"), folded)
            self.assertIn(fold_el("άρση ή μεταφορά φορτίου"), folded)
            self.assertIn(fold_el("οδήγηση"), folded)
            self.assertIn(fold_el("Παρακαλώ για"), folded)
            self.assertNotIn(fold_el("προοδευτική ενδυνάμωση"), folded)
            self.assertNotIn(fold_el("φυσικής πορείας"), folded)
            for english_leak in ("freezing", "thawing", "routine", "fixed", "frozen shoulder"):
                self.assertNotIn(english_leak, folded)
        self.assertIn("Ασθενής με συμφυτική θυλακίτιδα / παγωμένο ώμο δεξιά", short)
        self.assertIn("Παρακαλώ για εξατομικευμένη φυσιοθεραπευτική αποκατάσταση", short)
        self.assertIn("Παρακαλώ για εξατομικευμένο πρόγραμμα ενεργητικής αποκατάστασης και ασκήσεων κινητικότητας", detailed)
        self.assertIn("Η επιλογή και δοσολογία των επιμέρους ασκήσεων και τεχνικών εξατομικεύονται από τον φυσιοθεραπευτή", detailed)
        self.assertNotIn("ΣΤΑΔΙΟ ", detailed)

    def test_uncertain_irritability_is_not_rendered_in_referral(self):
        result = self._validate(irritability="uncertain_or_not_assessed")
        for mode in ("short", "detailed"):
            text = fold_el(self.formatter.format(result.normalized_draft, mode))
            self.assertNotIn(fold_el("Κλινική ερεθιστικότητα"), text)
            self.assertNotIn(fold_el("αβέβαι"), text)
            self.assertNotIn(fold_el("δεν αξιολογήθηκε"), text)

    def test_presentation_secondary_and_unresolved_scope_block_generation(self):
        cases = [
            ("presentation", "yes", "primary_frozen_shoulder"),
            ("formal_diagnosis", "yes", "secondary_or_other_stiff_shoulder"),
            ("formal_diagnosis", "yes", "not_stated"),
            ("formal_diagnosis", "yes", None),
        ]
        for wording, assertion, scope in cases:
            with self.subTest(wording=wording, scope=scope):
                draft = frozen_draft(wording_mode=wording, assertion=assertion, scope=scope)
                result = self.engine.validate(draft)
                self.assertTrue(result.formatter_blocked)
                gate_errors = [error for error in result.validation_errors if error.error_id == "rich_referral_context_required"]
                self.assertEqual(len(gate_errors), 1)
                context = {} if scope is None else {"frozen_shoulder_scope": scope}
                context["__wording_mode"] = wording
                self.assertFalse(
                    self.renderer.supports(
                        profile_id="shoulder",
                        route_id="adhesive_capsulitis_frozen_shoulder",
                        context=context,
                    )
                )
                with self.assertRaises(CU1ContractError):
                    self.formatter.format(result.normalized_draft, "detailed")

    def test_missing_scope_reports_the_exact_required_context_path(self):
        result = self.engine.validate(frozen_draft(scope=None))
        gate = next(error for error in result.validation_errors if error.error_id == "rich_referral_context_required")
        self.assertEqual(gate.metadata.get("reason"), "no_applicable_reviewed_rich_variant")
        self.assertEqual(
            gate.metadata.get("required_context_paths"),
            ["primary_problem.context.frozen_shoulder_scope"],
        )
        self.assertTrue(result.formatter_blocked)

    def test_missing_scope_generate_endpoint_returns_no_referral_text(self):
        app = FastAPI()
        app.include_router(build_cu1_physio_referral_router())
        with patch.dict(os.environ, {"CLINICAL_DATA_KEY": "cu1-test-key"}, clear=False):
            with TestClient(app) as client:
                response = client.post(
                    "/clinical/clinic-utilities/physio-referral/api/generate",
                    headers={"X-Clinical-Key": "cu1-test-key"},
                    json={"draft": frozen_draft(scope=None), "mode": "detailed"},
                )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["formatter_blocked"])
        self.assertIsNone(payload["text"])
        self.assertIn(
            "rich_referral_context_required",
            [item["error_id"] for item in payload["validation_errors"]],
        )

    def test_rich_output_does_not_invent_fixed_stages_dose_or_mandatory_strengthening(self):
        result = self._validate()
        detailed = self.formatter.format(result.normalized_draft, "detailed")
        folded = fold_el(detailed)
        self.assertNotIn("freezing", folded)
        self.assertNotIn("thawing", folded)
        self.assertNotIn("routine", folded)
        self.assertNotIn("fixed", folded)
        self.assertNotIn("frozen shoulder", folded)
        self.assertNotIn("σταδιο", folded)
        self.assertNotIn("3x", folded)
        self.assertNotIn("3 x", folded)
        self.assertNotIn(fold_el("φυσικής πορείας"), folded)
        self.assertNotIn(fold_el("ενδυνάμωση"), folded)
        self.assertNotIn(fold_el("καθολικό αριθμητικό"), folded)
        self.assertNotIn(fold_el("προκαθορισμένο κανόνα μετάβασης"), folded)
        self.assertIn(fold_el("ΣΤΟΧΟΙ ΑΠΟΚΑΤΑΣΤΑΣΗΣ"), folded)
        self.assertIn(fold_el("ΚΑΤΕΥΘΥΝΣΗ ΦΥΣΙΟΘΕΡΑΠΕΙΑΣ"), folded)
        self.assertIn(fold_el("ΕΠΑΝΕΚΤΙΜΗΣΗ"), folded)

    def test_self_stretching_and_strengthening_scope_remain_outside_mandatory_referral_core(self):
        self.assertEqual(
            self.evidence.claims["adhesive_capsulitis_self_stretching_cpg2025"].get("output_scope"),
            "therapist_execution_detail",
        )
        strengthening = self.evidence.claims["adhesive_capsulitis_strengthening_inconclusive_cpg2025"]
        self.assertEqual(strengthening.get("recommendation_direction"), "insufficient_evidence")
        self.assertEqual(strengthening.get("output_scope"), "clinician_ui_only")

    def test_contextual_evidence_panel_resolves_only_primary_formal_context(self):
        resolved = contextual_evidence_summary(
            self.evidence,
            self.renderer,
            profile_id="shoulder",
            route_id="adhesive_capsulitis_frozen_shoulder",
            wording_mode="formal_diagnosis",
            context={"frozen_shoulder_scope": "primary_frozen_shoulder"},
        )
        self.assertEqual(resolved["selection_state"], "resolved_context_profile")
        self.assertEqual(resolved["profile_count"], 1)
        self.assertTrue(resolved["sources"])
        self.assertTrue(resolved["claims"])
        summaries = "\n".join(str(item.get("claim_summary") or "") for item in resolved["claims"]).lower()
        self.assertIn("primary frozen shoulder", summaries)
        self.assertIn("insufficient", summaries)

        for wording, context in (
            ("presentation", {}),
            ("formal_diagnosis", {"frozen_shoulder_scope": "secondary_or_other_stiff_shoulder"}),
            ("formal_diagnosis", {"frozen_shoulder_scope": "not_stated"}),
        ):
            with self.subTest(wording=wording, context=context):
                blocked = contextual_evidence_summary(
                    self.evidence,
                    self.renderer,
                    profile_id="shoulder",
                    route_id="adhesive_capsulitis_frozen_shoulder",
                    wording_mode=wording,
                    context=context,
                )
                self.assertEqual(blocked["selection_state"], "context_required_for_evidence")
                self.assertEqual(blocked["sources"], [])
                self.assertEqual(blocked["claims"], [])

    def test_logical_coverage_has_one_complete_primary_context_and_blocked_others(self):
        route = self.evidence.coverage["profiles"]["shoulder"]["adhesive_capsulitis_frozen_shoulder"]
        self.assertEqual(route["sequence_status"], "sequence_incomplete")
        contexts = route["contexts"]
        self.assertEqual(contexts["clinician_established_primary_frozen_shoulder"]["sequence_status"], "sequence_complete")
        self.assertEqual(contexts["presentation_only_or_primary_scope_unresolved"]["sequence_status"], "blocked_evidence_gap")
        self.assertEqual(contexts["secondary_or_other_stiff_shoulder"]["sequence_status"], "blocked_evidence_gap")
        self.assertEqual(route["fixture_extension"], "cu1_frozen_shoulder_fixtures_v1.yaml")

    def test_unknown_frozen_shoulder_context_enums_fail_closed(self):
        for key, draft in (
            ("scope", frozen_draft(scope="probably_primary")),
            ("irritability", frozen_draft(irritability="very_high")),
        ):
            with self.subTest(key=key):
                result = self.engine.validate(draft)
                self.assertTrue(any(error.error_id == "invalid_context_enum_value" for error in result.validation_errors))
                self.assertTrue(result.formatter_blocked)


if __name__ == "__main__":
    unittest.main()
