from __future__ import annotations

import unittest

from clinic_utilities.physio_referral_api import _safety_state_is_canonical
from clinic_utilities.physio_referral_runtime import get_cu1_bundle


class CU1SafetyTrustBoundaryTests(unittest.TestCase):
    def test_empty_acknowledgements_and_none_recorded_are_valid(self):
        draft = {
            "safety": {
                "input_flags": [],
                "acknowledged_rule_ids": [],
                "clinician_disposition": "none_recorded",
            }
        }
        self.assertTrue(_safety_state_is_canonical(draft))

    def test_unknown_acknowledged_rule_id_is_rejected(self):
        draft = {
            "safety": {
                "input_flags": [],
                "acknowledged_rule_ids": ["invented_rule_id"],
                "clinician_disposition": "none_recorded",
            }
        }
        self.assertFalse(_safety_state_is_canonical(draft))

    def test_known_rule_id_is_accepted(self):
        rule_ids = list((get_cu1_bundle().rules.get("rules") or {}).keys())
        self.assertTrue(rule_ids)
        draft = {
            "safety": {
                "input_flags": [],
                "acknowledged_rule_ids": [rule_ids[0]],
                "clinician_disposition": "none_recorded",
            }
        }
        self.assertTrue(_safety_state_is_canonical(draft))

    def test_unknown_clinician_disposition_is_rejected(self):
        draft = {
            "safety": {
                "input_flags": [],
                "acknowledged_rule_ids": [],
                "clinician_disposition": "invented_disposition",
            }
        }
        self.assertFalse(_safety_state_is_canonical(draft))

    def test_every_frozen_disposition_is_accepted(self):
        bundle = get_cu1_bundle()
        dispositions = (
            bundle.artifacts["typed_supplement"]["safety_result_completion"]["fields"]["clinician_disposition"]
        )
        self.assertTrue(dispositions)
        for disposition in dispositions:
            with self.subTest(disposition=disposition):
                draft = {
                    "safety": {
                        "input_flags": [],
                        "acknowledged_rule_ids": [],
                        "clinician_disposition": disposition,
                    }
                }
                self.assertTrue(_safety_state_is_canonical(draft))


if __name__ == "__main__":
    unittest.main()
