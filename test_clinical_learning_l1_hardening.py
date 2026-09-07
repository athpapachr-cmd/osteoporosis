from __future__ import annotations

import copy
import unittest
from datetime import date
from uuid import uuid4

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from clinical_learning.contracts import LearningContractError
from clinical_learning.persistence import ChallengeRevisionORM, FoundationAttemptORM
from clinical_learning.service import ClinicalLearningService, LearningServiceError
from test_clinical_learning_l1_runtime import (
    CHALLENGE_ID,
    FOUNDATION_NODE,
    REF_ID,
    challenge,
    foundation_attempt,
)


def engine():
    return create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


class ClinicalLearningL1HardeningTests(unittest.TestCase):
    def setUp(self):
        self.engine = engine()
        self.service = ClinicalLearningService(self.engine)

    def test_aware_challenge_created_at_is_normalized_to_utc_before_storage(self):
        payload = challenge(disposition="accepted")
        payload["created_at"] = "2026-09-07T15:00:00+03:00"
        self.service.create_challenge(payload, confirm_save=True)
        with Session(self.engine) as session:
            row = session.get(ChallengeRevisionORM, (CHALLENGE_ID, 1))
            self.assertEqual(row.created_at.isoformat(), "2026-09-07T12:00:00")

    def test_aware_foundation_assessed_at_is_normalized_to_utc(self):
        payload = foundation_attempt(formal=True)
        payload["assessed_at"] = "2026-09-07T16:00:00+03:00"
        self.service.save_foundation_assessment(
            FOUNDATION_NODE,
            payload,
            next_review_due=date(2026, 10, 1),
            confirm_save=True,
        )
        with Session(self.engine) as session:
            row = session.execute(select(FoundationAttemptORM)).scalar_one()
            self.assertEqual(row.assessed_at.isoformat(), "2026-09-07T13:00:00")

    def test_older_foundation_attempt_cannot_overwrite_newer_materialized_state(self):
        newer = foundation_attempt(formal=True)
        newer["attempt_id"] = str(uuid4())
        newer["assessed_at"] = "2026-09-08T10:00:00Z"
        self.service.save_foundation_assessment(
            FOUNDATION_NODE,
            newer,
            next_review_due=date(2026, 10, 8),
            confirm_save=True,
        )

        older = foundation_attempt(formal=True)
        older["attempt_id"] = str(uuid4())
        older["assessed_at"] = "2026-09-07T10:00:00Z"
        older["clinician_final_state"] = "FRAGMENTED"
        older["proposed_state"] = "FRAGMENTED"
        with self.assertRaises(LearningServiceError) as caught:
            self.service.save_foundation_assessment(
                FOUNDATION_NODE,
                older,
                next_review_due=date(2026, 9, 20),
                confirm_save=True,
            )
        self.assertEqual(caught.exception.code, "foundation_assessment_older_than_current_state")
        current = next(
            item for item in self.service.foundation_registry()
            if item["node"]["node_id"] == FOUNDATION_NODE
        )
        self.assertEqual(current["state"]["state"], "FORMAL_SOLID")
        self.assertEqual(current["state"]["next_review_due"], "2026-10-08")

    def test_due_api_materialization_retains_occurrence_and_source_provenance(self):
        self.service.create_challenge(challenge(disposition="accepted"), confirm_save=True)
        due = self.service.list_due()
        self.assertEqual(len(due), 2)
        for item in due:
            self.assertEqual(item["occurrence"], 1)
            self.assertEqual(item["source_artifact_type"], "challenge")
            self.assertEqual(item["source_artifact_id"], CHALLENGE_ID)
            self.assertEqual(item["source_revision"], 1)

    def test_reference_verification_note_phi_is_rejected_without_persisting_overlay(self):
        self.service.create_challenge(challenge(disposition="accepted"), confirm_save=True)
        with self.assertRaises(LearningContractError) as caught:
            self.service.set_reference_verification(
                challenge_id=CHALLENGE_ID,
                revision=1,
                reference_id=REF_ID,
                verification_state="verified_locator",
                verification_note="contact learner@example.com",
            )
        self.assertIn("email_address_detected", repr(caught.exception.issues))
        detail = self.service.get_challenge(CHALLENGE_ID)
        self.assertEqual(detail["revisions"][0]["reference_verification"], [])

    def test_tombstoned_challenge_identity_is_blocked_by_preview_and_save(self):
        accepted = challenge(disposition="accepted")
        self.service.create_challenge(accepted, confirm_save=True)
        self.service.delete_challenge(CHALLENGE_ID, confirm_delete=True)
        preview = self.service.preview_challenge(accepted)
        self.assertFalse(preview["valid"])
        self.assertEqual(preview["duplicate_state"], "tombstoned_identity_rejected")
        with self.assertRaises(LearningServiceError) as caught:
            self.service.create_challenge(accepted, confirm_save=True)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.code, "tombstoned_identity_rejected")

    def test_confirmation_is_required_for_content_save_and_delete(self):
        accepted = challenge(disposition="accepted")
        with self.assertRaises(LearningServiceError) as caught:
            self.service.create_challenge(accepted, confirm_save=False)
        self.assertEqual(caught.exception.code, "confirm_save_required")
        self.service.create_challenge(accepted, confirm_save=True)
        with self.assertRaises(LearningServiceError) as caught_delete:
            self.service.delete_challenge(CHALLENGE_ID, confirm_delete=False)
        self.assertEqual(caught_delete.exception.code, "confirm_delete_required")
        self.assertFalse(self.service.get_challenge(CHALLENGE_ID)["deleted"])

    def test_foundation_attempt_idempotency_and_conflict(self):
        first = foundation_attempt(formal=True)
        saved = self.service.save_foundation_assessment(
            FOUNDATION_NODE,
            first,
            next_review_due=date(2026, 10, 1),
            confirm_save=True,
        )
        self.assertFalse(saved["idempotent"])
        repeated = self.service.save_foundation_assessment(
            FOUNDATION_NODE,
            first,
            next_review_due=date(2026, 10, 1),
            confirm_save=True,
        )
        self.assertTrue(repeated["idempotent"])

        changed = copy.deepcopy(first)
        changed["clinician_note"] = "Different content under same attempt id."
        with self.assertRaises(LearningServiceError) as caught:
            self.service.save_foundation_assessment(
                FOUNDATION_NODE,
                changed,
                next_review_due=date(2026, 10, 1),
                confirm_save=True,
            )
        self.assertEqual(caught.exception.code, "foundation_attempt_id_conflict")


if __name__ == "__main__":
    unittest.main()
