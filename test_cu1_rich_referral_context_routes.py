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

    def test_posttraumatic_neck_rich_output_requires_exact_wad_context(self):
        route_id = "post_traumatic_neck_pain"
        profile_id = "cervical"
        base = {
            "__wording_mode": "presentation",
            "trauma_mechanism_context": "whiplash_acceleration_deceleration",
            "structural_status": "no_material_structural_injury_identified_by_clinician",
            "material_neurological_or_other_safety_concern": "no",
            "physiotherapy_considered_appropriate_by_clinician": "yes",
        }

        self.assertEqual(self.renderer.rollout_state(profile_id=profile_id, route_id=route_id), "context_gated")
        self.assertFalse(self.renderer.supports(profile_id=profile_id, route_id=route_id))

        blocked_contexts = [
            {**base, "temporal_phase": "not_stated"},
            {**base, "trauma_mechanism_context": "other_cervical_trauma", "temporal_phase": "recent_or_acute_within_12_weeks"},
            {**base, "structural_status": "material_structural_injury_or_restriction_present", "temporal_phase": "recent_or_acute_within_12_weeks"},
            {**base, "material_neurological_or_other_safety_concern": "yes", "temporal_phase": "recent_or_acute_within_12_weeks"},
            {**base, "physiotherapy_considered_appropriate_by_clinician": "no", "temporal_phase": "recent_or_acute_within_12_weeks"},
            {**base, "__wording_mode": "established_structural_diagnosis", "temporal_phase": "recent_or_acute_within_12_weeks"},
        ]
        for context in blocked_contexts:
            with self.subTest(context=context):
                self.assertFalse(
                    self.renderer.supports(
                        profile_id=profile_id,
                        route_id=route_id,
                        context=context,
                    )
                )

        recent_context = {**base, "temporal_phase": "recent_or_acute_within_12_weeks"}
        persistent_context = {**base, "temporal_phase": "persistent_over_3_months"}
        for context, expected in (
            (recent_context, "πρόσφατη μη επιπλεγμένη whiplash-associated αυχεναλγία"),
            (persistent_context, "επίμονη whiplash-associated αυχεναλγία"),
        ):
            with self.subTest(expected=expected):
                self.assertTrue(
                    self.renderer.supports(
                        profile_id=profile_id,
                        route_id=route_id,
                        context=context,
                    )
                )
                short = self.renderer.render_short(
                    profile_id=profile_id,
                    route_id=route_id,
                    subtype_id=None,
                    context=context,
                    clinical_context=["Μετατραυματική / whiplash-associated αυχεναλγία", "Περιορισμός στην εργασία και οδήγηση"],
                )
                detailed = self.renderer.render_detailed(
                    profile_id=profile_id,
                    route_id=route_id,
                    subtype_id=None,
                    context=context,
                    clinical_context=["Μετατραυματική / whiplash-associated αυχεναλγία", "Περιορισμός στην εργασία και οδήγηση"],
                )
                self.assertIn(expected, short.lower())
                self.assertIn("χωρίς καθολ", detailed.lower())
                self.assertIn("ιατρική επανεκτίμηση", detailed.lower())
                self.assertLessEqual(len(detailed), self.renderer.standard_detailed_target_chars)


if __name__ == "__main__":
    unittest.main()
