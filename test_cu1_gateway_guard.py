from __future__ import annotations

import unittest

from clinic_utilities.physio_referral_api import _gateway_target_is_canonical


class CU1GatewayGuardTests(unittest.TestCase):
    def test_exact_frozen_gateway_is_accepted(self):
        draft = {
            "primary_problem": {
                "profile_id": "hip_groin",
                "route_id": "acute_hip_groin_muscle_injury_shared",
                "shared_target_optional": {
                    "profile_id": "shared_muscle_myotendinous",
                    "route_id": "acute_muscle_myotendinous_injury_rehabilitation",
                    "subtype_or_site_id_optional": None,
                },
            }
        }
        self.assertTrue(_gateway_target_is_canonical(draft))

    def test_forged_gateway_target_route_is_rejected(self):
        draft = {
            "primary_problem": {
                "profile_id": "hip_groin",
                "route_id": "acute_hip_groin_muscle_injury_shared",
                "shared_target_optional": {
                    "profile_id": "shared_fracture",
                    "route_id": "fracture_rehabilitation_post_immobilization",
                    "subtype_or_site_id_optional": None,
                },
            }
        }
        self.assertFalse(_gateway_target_is_canonical(draft))

    def test_forged_gateway_source_alias_is_rejected(self):
        draft = {
            "primary_problem": {
                "profile_id": "hip_groin",
                "route_id": "invented_gateway_alias",
                "shared_target_optional": {
                    "profile_id": "shared_muscle_myotendinous",
                    "route_id": "acute_muscle_myotendinous_injury_rehabilitation",
                    "subtype_or_site_id_optional": None,
                },
            }
        }
        self.assertFalse(_gateway_target_is_canonical(draft))

    def test_direct_shared_profile_needs_no_gateway_target(self):
        draft = {
            "primary_problem": {
                "profile_id": "shared_fracture",
                "route_id": "fracture_rehabilitation_post_immobilization",
                "shared_target_optional": None,
            }
        }
        self.assertTrue(_gateway_target_is_canonical(draft))


if __name__ == "__main__":
    unittest.main()
