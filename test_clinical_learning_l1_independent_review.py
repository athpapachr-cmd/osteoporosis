from __future__ import annotations

import copy
import unittest
from pathlib import Path

from clinical_learning.service import ClinicalLearningService
from test_clinical_learning_l1_runtime import (
    ACTION_ID,
    CHALLENGE_ID,
    FACT_ID,
    FOUNDATION_NODE,
    REF_ID,
    challenge,
    engine,
)


ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "static" / "clinical-learning" / "index.html"
JS_PATH = ROOT / "static" / "clinical-learning" / "app.js"


class ClinicalLearningL1IndependentReviewRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ClinicalLearningService(engine())

    def assert_preview_issue(self, payload: dict, code: str) -> None:
        result = self.service.preview_challenge(payload)
        self.assertFalse(result["valid"], result)
        self.assertIn(code, {item["code"] for item in result["issues"]})

    def test_frozen_nested_unique_items_are_enforced(self) -> None:
        payload = challenge()
        payload["progressive_disclosures"] = [
            {
                "disclosure_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                "sequence": 1,
                "label": "Synthetic disclosure",
                "narrative": None,
                "fact_ids": [FACT_ID, FACT_ID],
                "released_after_response_id": None,
            }
        ]
        self.assert_preview_issue(payload, "duplicate_disclosure_fact_id")

        payload = challenge()
        payload["observations"][0]["linked_fact_ids"] = [FACT_ID, FACT_ID]
        self.assert_preview_issue(payload, "duplicate_observation_fact_id")

        payload = challenge()
        payload["observations"][0]["linked_reference_ids"] = [REF_ID, REF_ID]
        self.assert_preview_issue(payload, "duplicate_observation_reference_id")

        payload = challenge()
        payload["observations"][0]["gap_classes"] = ["knowledge", "knowledge"]
        self.assert_preview_issue(payload, "duplicate_observation_gap_class")

        payload = challenge()
        payload["learning_actions"][0]["reference_ids"] = [REF_ID, REF_ID]
        self.assert_preview_issue(payload, "duplicate_action_reference_id")

        payload = challenge()
        payload["learning_actions"][0]["foundation_node_ids"] = [
            FOUNDATION_NODE,
            FOUNDATION_NODE,
        ]
        self.assert_preview_issue(payload, "duplicate_action_foundation_node_id")

        payload = challenge()
        payload["foundation_node_ids"] = [FOUNDATION_NODE, FOUNDATION_NODE]
        self.assert_preview_issue(payload, "duplicate_foundation_node_id")

        payload = challenge()
        payload["gap_classes"] = ["reasoning", "reasoning"]
        self.assert_preview_issue(payload, "duplicate_gap_class")

        payload = challenge()
        payload["linked_signal_ids"] = ["untrusted-signal", "untrusted-signal"]
        self.assert_preview_issue(payload, "duplicate_linked_signal_id")

    def test_saved_revision_reimports_as_exact_duplicate_after_authority_reset(self) -> None:
        accepted = challenge(disposition="accepted")
        saved = self.service.create_challenge(accepted, confirm_save=True)
        self.assertEqual(saved["challenge_id"], CHALLENGE_ID)

        preview = self.service.preview_challenge(accepted)
        self.assertTrue(preview["valid"], preview)
        self.assertEqual(preview["duplicate_state"], "exact_idempotent_duplicate")
        self.assertEqual(
            preview["normalized_summary"]["observations"][0]["clinician_disposition"],
            "pending",
        )

        changed = copy.deepcopy(accepted)
        changed["title"] = "Different content under the same revision"
        conflict = self.service.preview_challenge(changed)
        self.assertFalse(conflict["valid"], conflict)
        self.assertEqual(conflict["duplicate_state"], "same_revision_conflict")

    def test_history_ui_completes_frozen_filters_and_visibility(self) -> None:
        html = HTML_PATH.read_text(encoding="utf-8")
        js = JS_PATH.read_text(encoding="utf-8")
        self.assertIn('id="filterDate"', html)
        self.assertIn('id="filterReview"', html)
        self.assertIn("params.set('challenge_date', $('filterDate').value)", js)
        self.assertIn("params.set('review_state', $('filterReview').value)", js)
        self.assertIn("item.foundation_node_ids", js)
        self.assertIn("item.due.due_status", js)
        self.assertIn("Foundation: ${esc(foundation)} · due: ${esc(dueState)}", js)

    def test_challenge_save_requires_fresh_preview_of_current_textarea(self) -> None:
        js = JS_PATH.read_text(encoding="utf-8")
        self.assertIn("previewInputFingerprint", js)
        self.assertIn("stableJson(currentChallenge) !== state.previewInputFingerprint", js)
        self.assertIn("Preview stale · revalidate required", js)

    def test_foundation_attempt_payload_is_stable_during_open_assessment(self) -> None:
        js = JS_PATH.read_text(encoding="utf-8")
        self.assertIn("state.foundationAssessedAt = new Date().toISOString()", js)
        self.assertIn("state.foundationEvidenceIds = {}", js)
        self.assertIn(
            "if (!state.foundationEvidenceIds[method]) state.foundationEvidenceIds[method] = crypto.randomUUID()",
            js,
        )
        self.assertIn("evidence_id: state.foundationEvidenceIds[method]", js)
        self.assertIn("assessed_at: state.foundationAssessedAt", js)
        self.assertNotIn("evidence_id: crypto.randomUUID()", js)
        self.assertNotIn("assessed_at: new Date().toISOString()", js)


if __name__ == "__main__":
    unittest.main()
