from __future__ import annotations

import unittest

from fastapi import HTTPException
from pydantic import ValidationError

from clinic_utilities.rf.api import RFApplicationDraft, _validate_exact_location
from clinic_utilities.rf.catalog import INDICATIONS, LATERALITY_LABELS


def draft(**overrides):
    data = {
        "pathway": "A1",
        "patient_name": "Synthetic Patient",
        "identity_number": "SYN-UNI",
        "gesy_number": "GESY-SYN",
        "age": 67,
        "product_key": "medikey",
        "indication_code": "KNEE_OA_KL34",
        "laterality": "left",
        "exact_location": "Αριστερό γόνατο",
    }
    data.update(overrides)
    return RFApplicationDraft(**data)


class RFUnilateralContractTests(unittest.TestCase):
    def test_only_left_and_right_are_valid_lateralities(self):
        self.assertEqual(LATERALITY_LABELS, {"left": "Αριστερά", "right": "Δεξιά"})
        self.assertEqual(draft(laterality="left").laterality, "left")
        self.assertEqual(draft(laterality="right").laterality, "right")

    def test_bilateral_none_and_midline_are_rejected(self):
        for value in ("bilateral", "none", "midline", ""):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                draft(laterality=value)

    def test_knee_location_suggestion_is_side_specific(self):
        labels = INDICATIONS["KNEE_OA_KL34"]["location_labels"]
        self.assertEqual(labels["left"], "Αριστερό γόνατο")
        self.assertEqual(labels["right"], "Δεξί γόνατο")
        self.assertEqual(set(labels), {"left", "right"})

    def test_fixed_location_must_match_selected_side(self):
        indication = INDICATIONS["KNEE_OA_KL34"]
        good = draft(laterality="right", exact_location="Δεξί γόνατο")
        self.assertEqual(_validate_exact_location(good, indication), "Δεξί γόνατο")
        bad = draft(laterality="right", exact_location="Αριστερό γόνατο")
        with self.assertRaises(HTTPException) as caught:
            _validate_exact_location(bad, indication)
        self.assertEqual(caught.exception.status_code, 422)

    def test_all_fixed_indications_expose_only_single_side_location_labels(self):
        for code, item in INDICATIONS.items():
            labels = item.get("location_labels", {})
            if code == "OTHER_CUSTOM":
                self.assertEqual(labels, {})
            else:
                self.assertEqual(set(labels), {"left", "right"}, code)


if __name__ == "__main__":
    unittest.main()
