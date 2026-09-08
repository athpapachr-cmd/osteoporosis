from __future__ import annotations

import copy
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from clinical_learning.l1b_runtime import (
    LearningLoopRuntimeError,
    LearningLoopRuntimeService,
    _validate_loop_plan,
)
from clinical_learning.service import ClinicalLearningService
from test_clinical_learning_l1b_learning_loop import rich_episode, reviewed_candidate


ROOT = Path(__file__).resolve().parent


def engine():
    return create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


class ClinicalLearningL1BHardeningTests(unittest.TestCase):
    def setUp(self):
        self.engine = engine()
        self.learning = ClinicalLearningService(self.engine)
        self.loop = LearningLoopRuntimeService(self.engine)

    def test_acceptance_reanchors_repetition_to_actual_accept_date(self):
        pending = self.loop.ingest_episode(rich_episode())
        accepted = self.learning.create_challenge(reviewed_candidate(pending), confirm_save=True)
        self.loop.accept_pending_import(
            pending["import_id"],
            challenge_id=accepted["challenge_id"],
            revision=accepted["revision"],
        )
        cycle = self.loop.list_learning_loops()[0]
        today = datetime.now(timezone.utc).date()
        expected = [(today + timedelta(days=value)).isoformat() for value in (3, 7, 14, 30)]
        actual = [item["due_on"] for item in cycle["plan"]["consolidation_occurrences"]]
        self.assertEqual(actual, expected)

    def test_loop_plan_unknown_foundation_node_fails_closed(self):
        pending = self.loop.ingest_episode(rich_episode())
        plan = copy.deepcopy(pending["loop_plan"])
        plan["objectives"][0]["foundation_node_ids"] = ["ost.foundation.not_real"]
        with self.assertRaises(LearningLoopRuntimeError) as ctx:
            _validate_loop_plan(plan)
        self.assertEqual(ctx.exception.code, "unknown_foundation_node")

    def test_loop_plan_phi_like_content_returns_specific_sanitized_error(self):
        pending = self.loop.ingest_episode(rich_episode())
        plan = copy.deepcopy(pending["loop_plan"])
        plan["consolidation_occurrences"][0]["prompt"] = "Contact learner at test@example.com"
        with self.assertRaises(LearningLoopRuntimeError) as ctx:
            _validate_loop_plan(plan)
        self.assertEqual(ctx.exception.code, "email_address_detected")
        self.assertIn("prompt", ctx.exception.path)

    def test_browser_contract_auto_links_saved_inbox_challenge(self):
        html = (ROOT / "static/clinical-learning/index.html").read_text(encoding="utf-8")
        js = (ROOT / "static/clinical-learning/l1b.js").read_text(encoding="utf-8")
        self.assertIn('data-view="inbox"', html)
        self.assertIn('data-view="loop"', html)
        self.assertIn('data-view="import">Advanced', html)
        self.assertIn('/static/clinical-learning/l1b.js', html)
        self.assertIn("pendingSaveImportId", js)
        self.assertIn("window.fetch = async", js)
        self.assertIn("/accepted`,", js)
        self.assertIn("confirm_link: true", js)
        self.assertIn("later repetitions preserved", js)
        self.assertIn("Learning Loop link needs retry", js)


if __name__ == "__main__":
    unittest.main()
