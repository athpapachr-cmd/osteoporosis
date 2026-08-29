from __future__ import annotations

import unittest
from pathlib import Path

from clinic_utilities.physio_rich_referral import CU1RichReferralRenderer


ROOT = Path(__file__).resolve().parent


class CU1RichReferralContextRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = CU1RichReferralRenderer(ROOT)

    def test_cervical_headache_presentation_and_formal_cgh_remain_distinct(self):
        route_id = "headache_with_cervical_msk_features"
        profile_id = "cervical"

        self.assertEqual(self.renderer.rollout_state(profile_id=profile_id, route_id=route_id), "context_gated")
        self.assertFalse(self.renderer.supports(profile_id=profile_id, route_id=route_id))
        self.assertFalse(
            self.renderer.supports(
                profile_id=profile_id,
                route_id=route_id,
                context={"__wording_mode": "formal_diagnosis"},
            )
        )
        self.assertFalse(
            self.renderer.supports(
                profile_id=profile_id,
                route_id=route_id,
                context={
                    "__wording_mode": "formal_diagnosis",
                    "formal_cervicogenic_headache_diagnosis": "no",
                },
            )
        )

        presentation_context = {"__wording_mode": "presentation"}
        self.assertTrue(
            self.renderer.supports(
                profile_id=profile_id,
                route_id=route_id,
                context=presentation_context,
            )
        )
        presentation_short = self.renderer.render_short(
            profile_id=profile_id,
            route_id=route_id,
            subtype_id=None,
            context=presentation_context,
            clinical_context=["Κεφαλαλγία με αυχενικά μυοσκελετικά χαρακτηριστικά", "Περιορισμός στην εργασία σε οθόνη"],
        )
        presentation_detailed = self.renderer.render_detailed(
            profile_id=profile_id,
            route_id=route_id,
            subtype_id=None,
            context=presentation_context,
            clinical_context=["Κεφαλαλγία με αυχενικά μυοσκελετικά χαρακτηριστικά", "Περιορισμός στην εργασία σε οθόνη"],
        )
        self.assertNotIn("ρητά διαγνωσμένη αυχενογενή κεφαλαλγία", presentation_short.lower())
        self.assertIn("δεν ισοδυναμούν από μόνα τους με διάγνωση αυχενογενούς κεφαλαλγίας", presentation_short.lower())
        self.assertLessEqual(len(presentation_detailed), self.renderer.standard_detailed_target_chars)

        formal_context = {
            "__wording_mode": "formal_diagnosis",
            "formal_cervicogenic_headache_diagnosis": "yes",
        }
        self.assertTrue(
            self.renderer.supports(
                profile_id=profile_id,
                route_id=route_id,
                context=formal_context,
            )
        )
        formal_short = self.renderer.render_short(
            profile_id=profile_id,
            route_id=route_id,
            subtype_id=None,
            context=formal_context,
            clinical_context=["Αυχενογενής κεφαλαλγία", "Περιορισμός στην εργασία σε οθόνη"],
        )
        formal_detailed = self.renderer.render_detailed(
            profile_id=profile_id,
            route_id=route_id,
            subtype_id=None,
            context=formal_context,
            clinical_context=["Αυχενογενής κεφαλαλγία", "Περιορισμός στην εργασία σε οθόνη"],
        )
        self.assertIn("ρητά διαγνωσμένη αυχενογενή κεφαλαλγία", formal_short.lower())
        self.assertIn("χωρίς ένα καθολικό πρόγραμμα ή δόση", formal_detailed.lower())
        self.assertLessEqual(len(formal_detailed), self.renderer.standard_detailed_target_chars)

    def test_cervical_dizziness_requires_explicit_clinician_diagnosis(self):
        route_id = "cervical_dizziness_presentation"
        profile_id = "cervical"

        self.assertEqual(self.renderer.rollout_state(profile_id=profile_id, route_id=route_id), "context_gated")
        self.assertFalse(self.renderer.supports(profile_id=profile_id, route_id=route_id))
        self.assertFalse(
            self.renderer.supports(
                profile_id=profile_id,
                route_id=route_id,
                context={"__wording_mode": "presentation"},
            )
        )
        self.assertFalse(
            self.renderer.supports(
                profile_id=profile_id,
                route_id=route_id,
                context={
                    "__wording_mode": "formal_diagnosis",
                    "clinician_diagnosis_cervicogenic_dizziness": "no",
                },
            )
        )

        formal_context = {
            "__wording_mode": "formal_diagnosis",
            "clinician_diagnosis_cervicogenic_dizziness": "yes",
        }
        self.assertTrue(
            self.renderer.supports(
                profile_id=profile_id,
                route_id=route_id,
                context=formal_context,
            )
        )
        short = self.renderer.render_short(
            profile_id=profile_id,
            route_id=route_id,
            subtype_id=None,
            context=formal_context,
            clinical_context=["Αυχενογενής / αυχενικής προέλευσης ζάλη", "Περιορισμός σε βάδιση και εργασία"],
        )
        detailed = self.renderer.render_detailed(
            profile_id=profile_id,
            route_id=route_id,
            subtype_id=None,
            context=formal_context,
            clinical_context=["Αυχενογενής / αυχενικής προέλευσης ζάλη", "Περιορισμός σε βάδιση και εργασία"],
        )
        self.assertIn("χωρίς υπόσχεση επίλυσης της ζάλης", short.lower())
        self.assertIn("δεν προστίθεται αυτόματα αιθουσαία", detailed.lower())
        self.assertIn("ιατρική επανεκτίμηση", detailed.lower())
        self.assertLessEqual(len(detailed), self.renderer.standard_detailed_target_chars)


if __name__ == "__main__":
    unittest.main()
