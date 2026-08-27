from __future__ import annotations

import unittest

from clinic_utilities.physio_referral_api import _gateway_target_is_canonical
from clinic_utilities.physio_referral_runtime import CU1ContractBundle, CU1Engine


class CU1GatewayRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = CU1ContractBundle()
        cls.engine = CU1Engine(cls.bundle)

    def test_every_frozen_gateway_resolves_to_exact_semantic_owner(self):
        gateways = self.bundle.registry.get("gateways", {})
        self.assertTrue(gateways)

        for gateway_id, gateway in gateways.items():
            with self.subTest(gateway=gateway_id):
                source_profile = gateway["source_profile"]
                source_route_alias = gateway["source_route_alias"]
                target_profile = gateway["target_profile"]
                target_route = gateway["target_route"]
                target_detail = gateway.get("target_subtype_or_site")

                raw = {
                    "contract_version": "cu1_referral_draft_v1",
                    "body_region": source_profile,
                    "primary_problem": {
                        "problem_id": f"synthetic-{gateway_id}",
                        "profile_id": source_profile,
                        "route_id": source_route_alias,
                        "wording_mode": "presentation",
                        "formal_assertion_state_optional": None,
                        "subtype_id_optional": None,
                        "laterality": "not_stated",
                        "chronicity_or_phase_optional": None,
                        "context": {},
                        "shared_target_optional": {
                            "profile_id": target_profile,
                            "route_id": target_route,
                            "subtype_or_site_id_optional": target_detail,
                        },
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
                }

                self.assertTrue(_gateway_target_is_canonical(raw))
                normalized = self.engine.normalize(raw)
                self.assertEqual(normalized["body_region"], target_profile)
                self.assertEqual(normalized["primary_problem"]["profile_id"], target_profile)
                self.assertEqual(normalized["primary_problem"]["route_id"], target_route)
                self.assertEqual(normalized["primary_problem"]["source_route_optional"], source_route_alias)

                context = normalized["primary_problem"]["context"]
                if target_detail is not None and target_profile == "shared_fracture":
                    self.assertEqual(context.get("fracture_site"), target_detail)
                elif target_detail is not None and target_profile == "shared_muscle_myotendinous":
                    self.assertEqual(context.get("muscle_group"), target_detail)
                elif target_detail is not None and target_profile == "shared_deconditioning_balance_gait":
                    self.assertEqual(context.get("functional_route_id"), target_detail)


if __name__ == "__main__":
    unittest.main()
