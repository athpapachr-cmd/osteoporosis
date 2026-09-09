from __future__ import annotations

from pathlib import Path
import unittest

from clinical_learning.challenge_completion_protocol import classify_handoff_receipt


ROOT = Path(__file__).resolve().parent


class ClinicalLearningL1CChallengeCompletionTransportTests(unittest.TestCase):
    def test_project_instruction_static_copy_matches_versioned_owner(self):
        canonical = (
            ROOT / "clinical_learning" / "chatgpt_project_instructions_v1.txt"
        ).read_text(encoding="utf-8")
        browser_copy = (
            ROOT / "static" / "clinical-learning" / "chatgpt-project-instructions-v1.txt"
        ).read_text(encoding="utf-8")
        self.assertEqual(browser_copy, canonical)

    def test_project_instruction_makes_handoff_part_of_definition_of_done(self):
        text = (
            ROOT / "clinical_learning" / "chatgpt_project_instructions_v1.txt"
        ).read_text(encoding="utf-8")
        self.assertIn("DEFINITION OF DONE", text)
        self.assertIn("Cockpit handoff attempt", text)
        self.assertIn("state=pending_review", text)
        self.assertIn("import_id", text)
        self.assertIn("source_event_id", text)
        self.assertIn("COCKPIT_HANDOFF_PENDING", text)
        self.assertIn("HANDOFF_SUCCEEDED_PENDING_COCKPIT_REVIEW", text)
        self.assertIn("VERBATIM REASONING", text)
        self.assertIn("ΓΕΦΥΡΩΣΗ ΝΗΣΙΔΩΝ", text)
        self.assertIn("repeated consolidation", text)

    def test_setup_page_is_truthful_about_current_transport_state(self):
        html = (
            ROOT / "static" / "clinical-learning" / "project-setup.html"
        ).read_text(encoding="utf-8")
        self.assertIn("Project instruction", html)
        self.assertIn("READY", html)
        self.assertIn("Native Cockpit write tool", html)
        self.assertIn("NOT CONNECTED", html)
        self.assertIn("Advanced/manual fallback", html)
        self.assertIn("Copy Project Instructions", html)
        self.assertIn("Project settings → Project instructions", html)
        self.assertNotIn("CLINICAL_LEARNING_INGEST_KEY=", html)
        self.assertNotIn("X-Learning-Ingest-Key:", html)

    def test_no_debrief_means_in_progress_even_if_receipt_is_supplied(self):
        decision = classify_handoff_receipt(
            {
                "state": "pending_review",
                "import_id": "11111111-1111-4111-8111-111111111111",
                "source_event_id": "22222222-2222-4222-8222-222222222222",
                "source_format": "rich_challenge_export_v1",
            },
            transport_attempted=True,
            debrief_complete=False,
        )
        self.assertEqual(decision.state, "IN_PROGRESS")

    def test_debrief_without_transport_is_handoff_pending(self):
        decision = classify_handoff_receipt(
            None,
            transport_attempted=False,
            debrief_complete=True,
        )
        self.assertEqual(decision.state, "DEBRIEF_COMPLETE_HANDOFF_PENDING")
        self.assertEqual(decision.reason, "trusted_transport_not_attempted")

    def test_tool_invocation_without_receipt_cannot_claim_success(self):
        decision = classify_handoff_receipt(
            None,
            transport_attempted=True,
            debrief_complete=True,
        )
        self.assertEqual(decision.state, "HANDOFF_FAILED_MANUAL_FALLBACK_READY")
        self.assertEqual(decision.reason, "transport_receipt_missing")

    def test_invalid_receipt_cannot_claim_success(self):
        invalid_cases = [
            {},
            {"state": "accepted"},
            {
                "state": "pending_review",
                "import_id": "not-a-uuid",
                "source_event_id": "22222222-2222-4222-8222-222222222222",
                "source_format": "rich_challenge_export_v1",
            },
            {
                "state": "pending_review",
                "import_id": "11111111-1111-4111-8111-111111111111",
                "source_event_id": "22222222-2222-4222-8222-222222222222",
                "source_format": "unexpected_format",
            },
        ]
        for receipt in invalid_cases:
            with self.subTest(receipt=receipt):
                decision = classify_handoff_receipt(
                    receipt,
                    transport_attempted=True,
                    debrief_complete=True,
                )
                self.assertEqual(
                    decision.state,
                    "HANDOFF_FAILED_MANUAL_FALLBACK_READY",
                )

    def test_valid_pending_review_receipt_is_success_pending_review_only(self):
        decision = classify_handoff_receipt(
            {
                "state": "pending_review",
                "import_id": "11111111-1111-4111-8111-111111111111",
                "source_event_id": "22222222-2222-4222-8222-222222222222",
                "source_format": "rich_challenge_export_v1",
            },
            transport_attempted=True,
            debrief_complete=True,
        )
        self.assertEqual(
            decision.state,
            "HANDOFF_SUCCEEDED_PENDING_COCKPIT_REVIEW",
        )
        self.assertEqual(decision.receipt["state"], "pending_review")
        self.assertNotEqual(decision.state, "accepted")


if __name__ == "__main__":
    unittest.main()
