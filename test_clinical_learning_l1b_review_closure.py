from __future__ import annotations

import copy
import os
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from clinical_learning.api import build_learning_router
from clinical_learning.l1b_runtime import LearningLoopRuntimeError, LearningLoopRuntimeService
from clinical_learning.service import ClinicalLearningService
from test_clinical_learning_l1b_learning_loop import automatic_rich_episode, rich_episode, reviewed_candidate


def engine():
    return create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


class ClinicalLearningL1BReviewClosureTests(unittest.TestCase):
    def setUp(self):
        self.engine = engine()
        self.learning = ClinicalLearningService(self.engine)
        self.loop = LearningLoopRuntimeService(self.engine)

    def _accepted_cycle(self):
        pending = self.loop.ingest_episode(rich_episode())
        accepted = self.learning.create_challenge(reviewed_candidate(pending), confirm_save=True)
        self.loop.accept_pending_import(
            pending["import_id"],
            challenge_id=accepted["challenge_id"],
            revision=accepted["revision"],
        )
        return pending, accepted, self.loop.list_learning_loops()[0]

    def test_manual_summary_only_legacy_export_remains_salvageable_with_warning(self):
        pending = self.loop.ingest_episode(rich_episode())
        self.assertEqual(pending["state"], "pending_review")
        self.assertIn("source_reasoning_was_summary_not_verbatim", pending["adapter_warnings"])

    def test_automatic_ingress_rejects_summary_only_reasoning(self):
        app = FastAPI()
        app.include_router(build_learning_router(engine()))
        with patch.dict(os.environ, {"CLINICAL_LEARNING_INGEST_KEY": "ingest-key"}, clear=True):
            client = TestClient(app)
            response = client.post(
                "/clinical/learning/api/ingress/episodes",
                headers={"X-Learning-Ingest-Key": "ingest-key"},
                json={"episode": rich_episode()},
            )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["code"], "automatic_ingress_requires_verbatim_reasoning")

    def test_automatic_ingress_accepts_explicit_synthetic_verbatim_reasoning(self):
        app = FastAPI()
        app.include_router(build_learning_router(engine()))
        with patch.dict(os.environ, {"CLINICAL_LEARNING_INGEST_KEY": "ingest-key"}, clear=True):
            client = TestClient(app)
            response = client.post(
                "/clinical/learning/api/ingress/episodes",
                headers={"X-Learning-Ingest-Key": "ingest-key"},
                json={"episode": automatic_rich_episode()},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["state"], "pending_review")

    def test_rich_export_revision_greater_than_one_fails_closed(self):
        episode = rich_episode()
        episode["revision"] = 2
        episode["supersedes_revision"] = 1
        with self.assertRaises(LearningLoopRuntimeError) as ctx:
            self.loop.ingest_episode(episode)
        self.assertEqual(ctx.exception.code, "rich_learning_episode_revision_1_only")

    def test_retention_label_requires_explicit_clinician_review(self):
        _, _, cycle = self._accepted_cycle()
        occurrence = cycle["plan"]["consolidation_occurrences"][0]
        with self.assertRaises(LearningLoopRuntimeError) as ctx:
            self.loop.add_consolidation_attempt(
                cycle["cycle_id"],
                occurrence["occurrence_id"],
                {
                    "response_text": "Synthetic retrieval response.",
                    "result": "retained",
                    "clinician_reviewed": False,
                },
            )
        self.assertEqual(ctx.exception.code, "consolidation_result_requires_clinician_review")

    def test_same_consolidation_retry_is_idempotent_and_not_duplicated(self):
        _, _, cycle = self._accepted_cycle()
        occurrence = cycle["plan"]["consolidation_occurrences"][0]
        payload = {
            "response_text": "Synthetic retrieval response integrating renal-bone physiology and treatment safety.",
            "result": "retained",
            "clinician_reviewed": True,
        }
        first = self.loop.add_consolidation_attempt(cycle["cycle_id"], occurrence["occurrence_id"], payload)
        second = self.loop.add_consolidation_attempt(cycle["cycle_id"], occurrence["occurrence_id"], payload)
        self.assertEqual(first["attempt_id"], second["attempt_id"])
        self.assertFalse(first["idempotent"])
        self.assertTrue(second["idempotent"])
        updated = self.loop.get_learning_loop(cycle["cycle_id"])
        matching = [
            item for item in updated["attempts"]
            if item["occurrence_id"] == occurrence["occurrence_id"]
        ]
        self.assertEqual(len(matching), 1)

    def test_different_second_attempt_after_completion_fails_closed(self):
        _, _, cycle = self._accepted_cycle()
        occurrence = cycle["plan"]["consolidation_occurrences"][0]
        self.loop.add_consolidation_attempt(
            cycle["cycle_id"],
            occurrence["occurrence_id"],
            {
                "response_text": "First reviewed response.",
                "result": "partially_retained",
                "clinician_reviewed": True,
            },
        )
        with self.assertRaises(LearningLoopRuntimeError) as ctx:
            self.loop.add_consolidation_attempt(
                cycle["cycle_id"],
                occurrence["occurrence_id"],
                {
                    "response_text": "Different later response.",
                    "result": "retained",
                    "clinician_reviewed": True,
                },
            )
        self.assertEqual(ctx.exception.code, "consolidation_occurrence_already_completed")

    def test_bridge_targets_retain_observation_provenance(self):
        pending = self.loop.ingest_episode(rich_episode())
        objectives = pending["loop_plan"]["objectives"]
        self.assertTrue(any(item["source_observation_ids"] for item in objectives))
        bridges = pending["loop_plan"]["bridge_targets"]
        self.assertTrue(bridges)
        self.assertTrue(any(item["source_observation_ids"] for item in bridges))


if __name__ == "__main__":
    unittest.main()
