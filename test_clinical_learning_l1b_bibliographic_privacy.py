from __future__ import annotations

import unittest

from clinical_learning.privacy import scan_persistable_strings


class ClinicalLearningL1BBibliographicPrivacyTests(unittest.TestCase):
    def test_reference_title_allows_bibliographic_numeric_patterns(self):
        payload = {
            "references": [
                {
                    "title": (
                        "KDIGO 2017 Clinical Practice Guideline Update for CKD-MBD. "
                        "Kidney Int Suppl. 2017;7:1-59. PMID 30675420"
                    )
                },
                {
                    "title": (
                        "Management of Osteoporosis in Chronic Kidney Disease. "
                        "J Clin Endocrinol Metab. 2024;109:1234-1248"
                    )
                },
            ]
        }
        findings = scan_persistable_strings(payload)
        self.assertEqual(findings, [])

    def test_reference_title_still_blocks_explicit_phone_phrase(self):
        findings = scan_persistable_strings(
            {"references": [{"title": "Contact author phone: 99123456"}]}
        )
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].code, "explicit_phone_phrase_detected")
        self.assertEqual(findings[0].path, "references[0].title")

    def test_reference_title_still_blocks_email(self):
        findings = scan_persistable_strings(
            {"references": [{"title": "Correspondence test@example.com"}]}
        )
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].code, "email_address_detected")
        self.assertEqual(findings[0].path, "references[0].title")

    def test_generic_nonreference_phone_like_sequence_remains_blocked(self):
        findings = scan_persistable_strings(
            {"observations": [{"statement": "Call 99123456 for details"}]}
        )
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].code, "phone_number_like_sequence_detected")
        self.assertEqual(findings[0].path, "observations[0].statement")


if __name__ == "__main__":
    unittest.main()
