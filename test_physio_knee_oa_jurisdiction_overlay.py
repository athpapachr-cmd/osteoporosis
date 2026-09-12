from __future__ import annotations

import copy
import os
import unittest
import uuid
from unittest.mock import patch

from clinic_utilities.physio_referral_product import jurisdiction_overlay as j
from clinic_utilities.physio_referral_product import knee_oa_projection as p


def request() -> dict:
    return {
        "draft_id": str(uuid.uuid4()),
        "revision": 0,
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


class JurisdictionOverlayTests(unittest.TestCase):
    def test_reviewed_cy_gesy_profile_loads_only_when_explicitly_configured(self):
        profile = j.load_profile("CY_GESY")
        self.assertIsNotNone(profile)
        self.assertEqual(profile["profile"]["profile_id"], "CY_GESY")
        self.assertEqual(profile["profile"]["selection_source"], "explicit_account_configuration")
        self.assertIsNone(j.load_profile("GR"))
        self.assertIsNone(j.load_profile("UK_ENGLAND"))
        with patch.dict(os.environ, {j.PROFILE_ENV: ""}, clear=False):
            self.assertIsNone(j.configured_profile())
        with patch.dict(os.environ, {j.PROFILE_ENV: "CY_GESY"}, clear=False):
            self.assertEqual(j.configured_profile()["profile"]["profile_id"], "CY_GESY")

    def test_profile_contract_rejects_core_mutation_and_invalid_policy_semantics(self):
        profile = j.load_profile("CY_GESY")
        self.assertIsNotNone(profile)
        bad = copy.deepcopy(profile)
        bad["positions"][0]["core_reference"]["core_state_mutated_by_overlay"] = True
        with self.assertRaises(ValueError):
            j._validate_profile(bad)
        bad = copy.deepcopy(profile)
        bad["positions"][0]["policy_class"] = "stronger_local_evidence"
        with self.assertRaises(ValueError):
            j._validate_profile(bad)
        bad = copy.deepcopy(profile)
        bad["positions"][0]["display_policy"]["may_change_selection_automatically"] = True
        with self.assertRaises(ValueError):
            j._validate_profile(bad)

    def test_overlay_attaches_local_context_without_mutating_international_state(self):
        core = p.project(request())
        before = copy.deepcopy(core)
        profile = j.load_profile("CY_GESY")
        resolved = j.apply_evidence_overlay(core, profile)

        self.assertEqual(core, before, "input projection must not be mutated")
        self.assertEqual(resolved["jurisdiction_profile"]["profile_id"], "CY_GESY")
        self.assertEqual(
            resolved["evidence"]["acupuncture"]["evidence_state"],
            "guideline_conflict_or_mixed",
        )
        local = resolved["evidence"]["acupuncture"]["jurisdiction"]
        self.assertEqual(local["local_direction"], "against")
        self.assertEqual(local["relationship_to_core"], "local_position_within_international_conflict")
        self.assertEqual(local["display_policy"]["local_label"], "Κύπρος · διαφέρει")
        self.assertEqual(
            resolved["evidence"]["manual_therapy"]["evidence_state"],
            "guideline_conflict_or_mixed",
        )
        self.assertEqual(
            resolved["evidence"]["manual_therapy"]["jurisdiction"]["local_direction"],
            "conditional_for",
        )

        for item, view in before["evidence"].items():
            self.assertEqual(resolved["evidence"][item]["evidence_state"], view["evidence_state"])
            self.assertEqual(resolved["evidence"][item]["positions"], view["positions"])

    def test_admin_and_local_only_positions_never_become_clinical_evidence_items(self):
        profile = j.load_profile("CY_GESY")
        core = p.project(request())
        resolved = j.apply_evidence_overlay(core, profile)
        self.assertNotIn("electrotherapy", resolved["evidence"])
        self.assertNotIn("radiofrequency_nerve_ablation", resolved["evidence"])
        self.assertNotIn("GESY_ADMIN_PHYSIO_ACCESS", resolved["evidence"])

        operational = j.operational_positions(profile)
        ids = {row["local_position_id"] for row in operational}
        self.assertIn("GESY_ADMIN_PHYSIO_ACCESS", ids)
        self.assertIn("GESY_ADMIN_PHYS02", ids)
        self.assertIn("GESY_OA_IT_INTEGRATION", ids)
        self.assertTrue(all(row["policy_class"] != "clinical_guidance" for row in operational))

    def test_core_state_mismatch_fails_closed_to_no_overlay(self):
        profile = j.load_profile("CY_GESY")
        core = p.project(request())
        forged = copy.deepcopy(profile)
        for row in forged["positions"]:
            if row.get("intervention_id") == "acupuncture":
                row["core_reference"]["international_evidence_state"] = "recommended_or_supported"
                break
        resolved = j.apply_evidence_overlay(core, forged)
        self.assertNotIn("jurisdiction_profile", resolved)
        self.assertFalse(any("jurisdiction" in view for view in resolved["evidence"].values()))

    def test_referral_text_and_selection_are_identical_with_overlay_on_or_off(self):
        core = p.project(request())
        profile = j.load_profile("CY_GESY")
        resolved = j.apply_evidence_overlay(core, profile)
        self.assertEqual(resolved["text"], core["text"])
        self.assertEqual(resolved["state"], core["state"])
        self.assertEqual(resolved["suggestions"], core["suggestions"])
        self.assertEqual(resolved["gate"], core["gate"])
        self.assertEqual(resolved["safety"], core["safety"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
