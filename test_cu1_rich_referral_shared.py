from __future__ import annotations

import unicodedata
import unittest
from pathlib import Path

from clinic_utilities.physio_evidence_runtime import CU1ClinicianEvidenceResolver
from clinic_utilities.physio_referral_formatter_el_v2 import CU1GreekReferralFormatter
from clinic_utilities.physio_referral_runtime import CONTRACT_VERSION, CU1ContractBundle
from clinic_utilities.physio_rich_referral import CU1RichReferralRenderer


ROOT = Path(__file__).resolve().parent


def greek_fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


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
            "problem_id": "synthetic-shared-rich",
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
        "findings": [{"finding_id": "lateral_elbow_pain"}, {"finding_id": "pain_with_gripping"}],
        "functional_impairments": [{"id": "gripping"}, {"id": "manual_work"}],
        "precautions": [],
        "explicit_restrictions": [],
        "goals": [],
        "rehab_directions": [],
        "adjunct_options": [],
        "measurements": [],
        "safety": {"input_flags": [], "acknowledged_rule_ids": [], "clinician_disposition": "none_recorded"},
        "sessions_optional": None,
        "clinician_free_text_optional": None,
    }


class CU1SharedRichReferralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = CU1ContractBundle(ROOT)
        cls.renderer = CU1RichReferralRenderer(ROOT)
        cls.formatter = CU1GreekReferralFormatter(cls.bundle)
        cls.evidence = CU1ClinicianEvidenceResolver(ROOT)

    def test_formatter_and_renderer_have_no_lateral_elbow_treatment_branch(self):
        formatter_source = (ROOT / "clinic_utilities/physio_referral_formatter_el_v2.py").read_text(encoding="utf-8")
        renderer_source = (ROOT / "clinic_utilities/physio_rich_referral.py").read_text(encoding="utf-8")
        self.assertNotIn('route_id == "lateral_elbow_tendinopathy"', formatter_source)
        self.assertNotIn('_format_lateral_elbow', formatter_source)
        self.assertNotIn('lateral_elbow_tendinopathy', renderer_source)

    def test_every_registry_route_has_exactly_one_rollout_classification(self):
        rollout = self.renderer.contract_rollout_entries()
        registry_profiles = self.bundle.registry.get("profiles") or {}
        allowed_states = {"rich_ready", "context_gated", "evidence_limited", "pending_evidence", "protocol_owned"}

        self.assertEqual(set(rollout), set(registry_profiles))
        for profile_id, profile_spec in registry_profiles.items():
            registry_routes = set(((profile_spec or {}).get("routes") or {}))
            rollout_routes = set(rollout.get(profile_id) or {})
            with self.subTest(profile=profile_id):
                self.assertEqual(rollout_routes, registry_routes)
            for route_id, entry in (rollout.get(profile_id) or {}).items():
                with self.subTest(profile=profile_id, route=route_id):
                    self.assertIn(entry.get("state"), allowed_states)

    def test_rollout_states_do_not_overstate_authoritative_sequence_coverage(self):
        rollout = self.renderer.contract_rollout_entries()
        coverage_profiles = self.evidence.coverage.get("profiles") or {}
        for profile_id, routes in rollout.items():
            profile_coverage = coverage_profiles.get(profile_id) or {}
            for route_id, entry in routes.items():
                coverage = profile_coverage.get(route_id) or {}
                state = entry.get("state")
                sequence_status = coverage.get("sequence_status")
                with self.subTest(profile=profile_id, route=route_id, state=state):
                    if state == "rich_ready":
                        self.assertEqual(sequence_status, "sequence_complete")
                    elif state == "context_gated":
                        if sequence_status == "sequence_complete":
                            continue
                        branches = coverage.get("contexts") or coverage.get("variants") or {}
                        complete_branches = [
                            branch
                            for branch in branches.values()
                            if isinstance(branch, dict) and branch.get("sequence_status") == "sequence_complete"
                        ] if isinstance(branches, dict) else []
                        self.assertTrue(complete_branches)
                    elif state == "evidence_limited" and sequence_status:
                        self.assertNotEqual(sequence_status, "sequence_complete")

    def test_only_rich_ready_routes_can_be_supported_without_context(self):
        rollout = self.renderer.contract_rollout_entries()
        for profile_id, routes in rollout.items():
            for route_id, entry in routes.items():
                if entry.get("state") == "rich_ready":
                    continue
                with self.subTest(profile=profile_id, route=route_id, state=entry.get("state")):
                    self.assertFalse(self.renderer.supports(profile_id=profile_id, route_id=route_id, subtype_id=None))

    def test_context_gated_variant_resolution_is_exact_and_fail_closed(self):
        renderer = CU1RichReferralRenderer(ROOT)
        renderer.routes = dict(renderer.routes)
        renderer.routes["glenohumeral_osteoarthritis"] = {
            "profile_ids": ["shoulder"],
            "variants": [
                {
                    "variant_id": "synthetic_nonoperative",
                    "match": {"context_equals": {"management_context": "nonoperative"}},
                    "evidence_profile_ids": ["rep_glenohumeral_oa_nonoperative_v1"],
                    "short_flow_el": ["Εξατομικευμένη συντηρητική αποκατάσταση."],
                    "stages": [
                        {
                            "stage_id": "functional_return_self_management",
                            "label_el": "ΣΤΑΔΙΟ 1 — ΕΞΑΤΟΜΙΚΕΥΜΕΝΗ ΑΠΟΚΑΤΑΣΤΑΣΗ",
                            "goals_el": ["βελτίωση λειτουργίας"],
                            "intervention_directions_el": ["εξατομικευμένη αποκατάσταση βάσει κλινικών ελλειμμάτων"],
                            "progress_markers_el": ["λειτουργική βελτίωση"],
                            "evidence_claim_ids": ["ghoa_individualized_PT_selection_2023"],
                        }
                    ],
                }
            ],
        }
        self.assertEqual(
            renderer.rollout_state(profile_id="shoulder", route_id="glenohumeral_osteoarthritis"),
            "context_gated",
        )
        self.assertFalse(renderer.supports(profile_id="shoulder", route_id="glenohumeral_osteoarthritis"))
        self.assertFalse(
            renderer.supports(
                profile_id="shoulder",
                route_id="glenohumeral_osteoarthritis",
                context={"management_context": "postoperative"},
            )
        )
        self.assertTrue(
            renderer.supports(
                profile_id="shoulder",
                route_id="glenohumeral_osteoarthritis",
                context={"management_context": "nonoperative"},
            )
        )
        text = renderer.render_detailed(
            profile_id="shoulder",
            route_id="glenohumeral_osteoarthritis",
            subtype_id=None,
            context={"management_context": "nonoperative"},
            clinical_context=["Οστεοαρθρίτιδα γληνοβραχιόνιας"],
        )
        self.assertIn("ΣΤΑΔΙΟ 1", text)

        renderer.routes["glenohumeral_osteoarthritis"]["variants"].append(
            {
                "variant_id": "synthetic_ambiguous_duplicate",
                "match": {"context_equals": {"management_context": "nonoperative"}},
                "evidence_profile_ids": ["rep_glenohumeral_oa_nonoperative_v1"],
                "short_flow_el": ["Δεν πρέπει να επιλυθεί."],
                "stages": [
                    {
                        "stage_id": "functional_return_self_management",
                        "label_el": "ΣΤΑΔΙΟ 1 — DUPLICATE",
                        "goals_el": ["στόχος"],
                        "intervention_directions_el": ["κατεύθυνση"],
                        "progress_markers_el": ["πρόοδος"],
                    }
                ],
            }
        )
        self.assertFalse(
            renderer.supports(
                profile_id="shoulder",
                route_id="glenohumeral_osteoarthritis",
                context={"management_context": "nonoperative"},
            )
        )

    def test_every_configured_rich_route_is_rollout_ready_and_evidence_profiles_resolve(self):
        registry_profiles = self.bundle.registry.get("profiles") or {}
        known_routes = {
            route_id
            for profile in registry_profiles.values()
            for route_id in ((profile or {}).get("routes") or {})
        }
        known_profiles = set(self.evidence.route_evidence_profiles)
        known_claims = set(self.evidence.claims)

        for route_id, spec in self.renderer.contract_route_specs().items():
            with self.subTest(route=route_id):
                self.assertIn(route_id, known_routes)
                self.assertTrue(spec.get("profile_ids"))
                state_values = {
                    self.renderer.rollout_state(profile_id=profile_id, route_id=route_id)
                    for profile_id in spec.get("profile_ids") or []
                }
                self.assertTrue(state_values.issubset({"rich_ready", "context_gated"}))
                if "variants" in spec:
                    variants = spec.get("variants") or []
                    self.assertTrue(variants)
                    evidence_sets = [variant.get("evidence_profile_ids") or [] for variant in variants if isinstance(variant, dict)]
                else:
                    evidence_sets = [spec.get("evidence_profile_ids") or []]
                    for profile_id in spec.get("profile_ids") or []:
                        rollout_entry = self.renderer.rollout_entry(profile_id=profile_id, route_id=route_id) or {}
                        self.assertTrue(set(spec.get("evidence_profile_ids") or []).issubset(set(rollout_entry.get("evidence_profile_ids") or [])))
                for evidence_profile_ids in evidence_sets:
                    for evidence_profile_id in evidence_profile_ids:
                        self.assertIn(evidence_profile_id, known_profiles)
                stages_to_check = []
                if "variants" in spec:
                    for variant in spec.get("variants") or []:
                        if isinstance(variant, dict):
                            stages_to_check.extend(variant.get("stages") or [])
                else:
                    stages_to_check.extend(spec.get("stages") or [])
                for stage in stages_to_check:
                    self.assertTrue(stage.get("goals_el"))
                    self.assertTrue(stage.get("intervention_directions_el"))
                    self.assertTrue(stage.get("progress_markers_el"))
                    for claim_id in stage.get("evidence_claim_ids") or []:
                        self.assertIn(claim_id, known_claims)

    def test_every_active_ungated_rich_route_renders_short_and_detailed_within_limits(self):
        for route_id, spec in self.renderer.contract_route_specs().items():
            profile_id = (spec.get("profile_ids") or [None])[0]
            if self.renderer.rollout_state(profile_id=profile_id, route_id=route_id) != "rich_ready":
                continue
            with self.subTest(profile=profile_id, route=route_id):
                self.assertIsNotNone(profile_id)
                short = self.renderer.render_short(
                    profile_id=profile_id,
                    route_id=route_id,
                    subtype_id=None,
                    clinical_context=["Κλινική διάγνωση", "Σχετικός λειτουργικός περιορισμός"],
                )
                detailed = self.renderer.render_detailed(
                    profile_id=profile_id,
                    route_id=route_id,
                    subtype_id=None,
                    clinical_context=["Κλινική διάγνωση", "Σχετικός λειτουργικός περιορισμός"],
                )
                self.assertTrue(short.strip())
                self.assertTrue(detailed.strip())
                self.assertNotIn("ΣΤΑΔΙΟ", short)
                self.assertIn("Στόχοι:", detailed)
                self.assertIn("Κατευθύνσεις:", detailed)
                self.assertIn("Πρόοδος", detailed)
                self.assertLessEqual(len(short), self.renderer.max_chars)
                self.assertLessEqual(len(detailed), self.renderer.standard_detailed_target_chars)

    def test_approved_let_meaning_survives_shared_renderer_migration(self):
        draft = lateral_elbow_draft()
        short = self.formatter.format(draft, "short")
        detailed = self.formatter.format(draft, "detailed")

        for text in (short, detailed):
            folded = greek_fold(text)
            self.assertIn(greek_fold("ισομετρική"), folded)
            self.assertIn(greek_fold("ομόκεντρη"), folded)
            self.assertIn(greek_fold("έκκεντρη"), folded)
            self.assertIn(greek_fold("επανένταξη"), folded)
            self.assertIn(greek_fold("μείωση κινδύνου υποτροπής"), folded)
            self.assertIn(greek_fold("διαχείριση φορτίου"), folded)
            self.assertLessEqual(len(text.rstrip("\n")), self.renderer.max_chars)

        self.assertIn("ΣΤΑΔΙΟ 1", detailed)
        self.assertIn("ΣΤΑΔΙΟ 2", detailed)
        self.assertIn("ΣΤΑΔΙΟ 3", detailed)
        self.assertNotIn("ΣΤΑΔΙΟ 1", short)

    def test_nonspecific_neck_pain_is_first_non_let_shared_rich_route(self):
        self.assertTrue(
            self.renderer.supports(
                profile_id="cervical",
                route_id="nonspecific_neck_pain",
                subtype_id=None,
            )
        )
        short = self.renderer.render_short(
            profile_id="cervical",
            route_id="nonspecific_neck_pain",
            subtype_id=None,
            clinical_context=["Μη ειδική αυχεναλγία", "Περιορισμός σε παρατεταμένη εργασία σε υπολογιστή"],
        )
        detailed = self.renderer.render_detailed(
            profile_id="cervical",
            route_id="nonspecific_neck_pain",
            subtype_id=None,
            clinical_context=["Μη ειδική αυχεναλγία", "Περιορισμός σε παρατεταμένη εργασία σε υπολογιστή"],
        )
        for text in (short, detailed):
            lower = text.lower()
            self.assertIn("ενεργ", lower)
            self.assertIn("αυτοδιαχείρι", lower)
            self.assertIn("δραστηρι", lower)
            self.assertNotIn("traction", lower)
            self.assertNotIn("υπέρηχ", lower)
            self.assertLessEqual(len(text), self.renderer.max_chars)
        self.assertIn("ΣΤΑΔΙΟ 1", detailed)
        self.assertIn("ΣΤΑΔΙΟ 2", detailed)
        self.assertNotIn("ΣΤΑΔΙΟ 3", detailed)
        self.assertLessEqual(len(detailed), self.renderer.standard_detailed_target_chars)

    def test_rich_renderer_overflow_fails_closed_instead_of_clipping_safety_tail(self):
        with self.assertRaisesRegex(Exception, "exceeds 2000 characters"):
            self.renderer.render_detailed(
                profile_id="elbow",
                route_id="lateral_elbow_tendinopathy",
                subtype_id=None,
                clinical_context=["πολύ μεγάλο κλινικό πλαίσιο " * 200],
            )

    def test_evidence_limited_route_cannot_gain_rich_authority_from_content_presence(self):
        self.assertEqual(
            self.renderer.rollout_state(profile_id="lumbar", route_id="nonspecific_low_back_pain"),
            "evidence_limited",
        )
        self.assertFalse(
            self.renderer.supports(
                profile_id="lumbar",
                route_id="nonspecific_low_back_pain",
                subtype_id=None,
            )
        )


if __name__ == "__main__":
    unittest.main()
