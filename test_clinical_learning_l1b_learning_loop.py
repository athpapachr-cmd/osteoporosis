from __future__ import annotations

import copy
import os
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from clinical_learning.api import build_learning_router
from clinical_learning.ingress import adapt_learning_episode
from clinical_learning.l1b_runtime import LearningLoopRuntimeError, LearningLoopRuntimeService
from clinical_learning.persistence import PendingImportORM
from clinical_learning.service import ClinicalLearningService


def engine():
    return create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def rich_episode():
    return {
        "schema_type": "ClinicalLearningChallengeV1",
        "schema_version": "1.0",
        "challenge_id": "synthetic-ckd-fracture-learning-episode",
        "revision": 1,
        "supersedes_revision": None,
        "source_artifact_status": "frozen_external_candidate",
        "raw_source_sentinel": "THIS_RAW_FIELD_MUST_NOT_PERSIST",
        "module": "osteoporosis",
        "challenge_mode": "synthetic",
        "deidentification": {
            "contains_patient_identifiers": False,
            "clinician_attestation_required_on_import": False,
        },
        "topic_tags": ["ckd-mbd", "hip-fracture", "evidence-appraisal"],
        "foundation_node_ids": [
            "secondary-osteoporosis",
            "antiresorptive-pharmacology",
            "safety-contraindications-transitions",
            "evidence-appraisal-directness-guideline-disagreement",
        ],
        "session": {
            "title": "Synthetic CKD-MBD fracture challenge",
            "format": "advanced progressive disclosure",
            "started_on": "2026-09-08",
            "completed_on": "2026-09-08",
        },
        "initial_case": {
            "prompt": "Choose the disease model and treatment approach.",
            "facts": [
                "A synthetic older adult has a low-trauma fracture and CKD G3b.",
                "Calcium is normal and PTH is elevated.",
            ],
        },
        "fact_ledger": [
            {
                "fact_id": "f01",
                "fact_class": "synthetic_initial_case_fact",
                "stage": "initial",
                "content": "Synthetic low-trauma fracture with CKD G3b.",
                "authoritative_for_patient": False,
            },
            {
                "fact_id": "f02",
                "fact_class": "synthetic_progressive_disclosure_fact",
                "stage": "disclosure_1",
                "content": "PTH and phosphate remain elevated and bone-specific alkaline phosphatase is high.",
                "authoritative_for_patient": False,
            },
            {
                "fact_id": "f03",
                "fact_class": "clinician_hypothesis",
                "stage": "post_disclosure_response",
                "content": "The biochemical pattern changes drug-selection reasoning.",
                "authoritative_for_patient": False,
            },
        ],
        "progressive_disclosures": [
            {
                "sequence": 1,
                "consequential_new_datum": "Persistent biochemical CKD-MBD abnormalities are disclosed.",
                "why_it_matters": "Routine osteoporosis pharmacology can no longer be applied in isolation.",
            }
        ],
        "clinician_reasoning_responses": [
            {
                "stage": "initial_response",
                "summary": [
                    "Recognised the fracture as a treatment-defining event.",
                    "Initially considered an antiresorptive strategy.",
                ],
                "confidence": 70,
                "final_decision": None,
            },
            {
                "stage": "post_disclosure_response",
                "summary": [
                    "Updated the model after the biochemical disclosure.",
                    "Recognised the need to integrate renal-bone physiology and drug safety.",
                ],
                "confidence": 75,
                "final_decision": "Use specialist co-management before committing to a long-term drug pathway.",
            },
        ],
        "mentor_observations": {
            "strengths": ["Correctly prioritised the fragility fracture."],
            "clear_errors": [
                {
                    "issue": "Initially used an incomplete mechanism for CKD secondary hyperparathyroidism.",
                    "correction": "Integrate phosphate retention, calcitriol and parathyroid physiology.",
                }
            ],
            "defensible_disagreements": [
                {
                    "issue": "Choice of antiresorptive after biochemical optimisation.",
                    "assessment": "Requires explicit safety and transition reasoning.",
                }
            ],
            "evidence_gaps": ["Direct evidence for the exact phenotype is limited."],
            "blind_spots": ["Renal osteodystrophy and biopsy decision thresholds need reinforcement."],
            "reasoning_patterns": ["Willingness to update the model after consequential new information."],
            "clinical_insight": "Do not treat active CKD-MBD as routine primary osteoporosis.",
        },
        "evidence": [
            {
                "citation": "Synthetic CKD-MBD guideline learning reference.",
                "source_type": "guideline",
                "evidence_level": "guideline/consensus",
                "relevance": "Supports integrated CKD-MBD treatment reasoning.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
                "verification_status": "unverified_external_reference",
            }
        ],
        "gap_classes": [
            {
                "class": "knowledge_and_mechanism",
                "gap": "CKD secondary hyperparathyroidism and renal osteodystrophy physiology.",
            },
            {
                "class": "decision_safety",
                "gap": "Integrate medication choice with CKD-MBD safety and transition constraints.",
            },
            {
                "class": "evidence_appraisal",
                "gap": "Separate direct evidence from extrapolated evidence.",
            },
        ],
        "learning_actions": [
            {
                "action_id": "la01",
                "action": "Review the synthetic CKD-MBD guideline resource.",
                "reason": "Targets the mechanism and decision-safety gap.",
                "source_type": "guideline/consensus",
                "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
            }
        ],
        "spaced_repetition": {
            "occurrence": 1,
            "due_on": "2026-09-20",
            "status": "scheduled_candidate",
            "adaptive_algorithm_applied": False,
            "prompt": "State why CKD-MBD changes the drug-selection problem after a fragility fracture.",
        },
        "clinician_review": {
            "state": "pending_clinician_review",
            "reviewed_at": None,
        },
    }


def automatic_rich_episode():
    episode = rich_episode()
    episode["clinician_reasoning_responses"][0]["text"] = (
        "I recognise the fragility fracture as treatment-defining and would integrate CKD before selecting therapy."
    )
    episode["clinician_reasoning_responses"][1]["text"] = (
        "The persistent biochemical abnormalities change my disease model and require renal-bone and medication-safety reasoning together."
    )
    return episode


def reviewed_candidate(pending):
    challenge = copy.deepcopy(pending["normalized_challenge"])
    challenge["privacy"]["deidentification_attested"] = True
    for observation in challenge.get("observations") or []:
        observation["clinician_disposition"] = "accepted"
    return challenge


class ClinicalLearningL1BLearningLoopTests(unittest.TestCase):
    def setUp(self):
        self.engine = engine()
        self.learning = ClinicalLearningService(self.engine)
        self.loop = LearningLoopRuntimeService(self.engine)

    def test_rich_export_adapts_to_frozen_challenge_contract(self):
        first = adapt_learning_episode(rich_episode())
        second = adapt_learning_episode(rich_episode())
        self.assertEqual(first.source_format, "rich_challenge_export_v1")
        self.assertEqual(first.normalized_challenge["challenge_id"], second.normalized_challenge["challenge_id"])
        self.assertEqual(first.normalized_challenge["fact_ledger"][0]["fact_id"], second.normalized_challenge["fact_ledger"][0]["fact_id"])
        self.assertEqual(first.normalized_challenge["schema_version"], "clinical_learning_challenge_v1")
        self.assertFalse(first.normalized_challenge["privacy"]["deidentification_attested"])
        self.assertTrue(all(item["authoritative_for_patient"] is False for item in first.normalized_challenge["fact_ledger"]))
        self.assertTrue(all(item["verification_state"] == "unverified" for item in first.normalized_challenge["references"]))
        self.assertTrue(all(item["clinician_disposition"] == "pending" for item in first.normalized_challenge["observations"]))
        self.assertEqual(len(first.loop_plan["consolidation_occurrences"]), 4)
        self.assertEqual(
            [item["kind"] for item in first.loop_plan["consolidation_occurrences"]],
            ["retrieval", "discrimination", "transfer", "bridge_transfer"],
        )
        self.assertGreaterEqual(len(first.loop_plan["bridge_targets"]), 1)
        self.assertIn("source_reasoning_was_summary_not_verbatim", first.warnings)

    def test_pending_import_is_idempotent_without_raw_payload_persistence(self):
        first = self.loop.ingest_episode(rich_episode())
        second = self.loop.ingest_episode(rich_episode())
        self.assertEqual(first["import_id"], second["import_id"])
        self.assertTrue(second["idempotent"])
        with Session(self.engine) as session:
            row = session.execute(select(PendingImportORM)).scalar_one()
            stored = repr(
                {
                    "challenge": row.normalized_challenge_json,
                    "loop": row.loop_plan_json,
                    "resources": row.resources_json,
                    "warnings": row.warnings_json,
                }
            )
        self.assertNotIn("THIS_RAW_FIELD_MUST_NOT_PERSIST", stored)
        self.assertNotIn("raw_source_sentinel", stored)

    def test_same_source_event_id_with_changed_content_fails_closed(self):
        source_id = "10101010-1010-4010-8010-101010101010"
        self.loop.ingest_episode(rich_episode(), source_event_id=source_id)
        changed = rich_episode()
        changed["session"]["title"] = "Different normalized candidate"
        with self.assertRaises(LearningLoopRuntimeError) as ctx:
            self.loop.ingest_episode(changed, source_event_id=source_id)
        self.assertEqual(ctx.exception.code, "source_event_id_content_conflict")

    def test_accept_materializes_four_repetitions_and_first_success_preserves_later_repeats(self):
        pending = self.loop.ingest_episode(rich_episode())
        accepted = self.learning.create_challenge(reviewed_candidate(pending), confirm_save=True)
        linked = self.loop.accept_pending_import(
            pending["import_id"],
            challenge_id=accepted["challenge_id"],
            revision=accepted["revision"],
        )
        self.assertEqual(linked["state"], "accepted")
        loops = self.loop.list_learning_loops()
        self.assertEqual(len(loops), 1)
        cycle = loops[0]
        occurrences = cycle["plan"]["consolidation_occurrences"]
        self.assertEqual(len(occurrences), 4)
        self.assertEqual(len([item for item in self.learning.list_due() if item["item_type"] == "consolidation_test"]), 4)

        first = occurrences[0]
        attempt = self.loop.add_consolidation_attempt(
            cycle["cycle_id"],
            first["occurrence_id"],
            {
                "response_text": "Synthetic retrieval response integrating CKD-MBD and medication safety.",
                "result": "retained",
                "clinician_reviewed": True,
            },
        )
        self.assertEqual(attempt["result"], "retained")
        updated = self.loop.get_learning_loop(cycle["cycle_id"])
        statuses = [item["due_state"]["due_status"] for item in updated["plan"]["consolidation_occurrences"]]
        self.assertEqual(statuses[0], "completed")
        self.assertTrue(all(value != "completed" for value in statuses[1:]))

    def test_bridge_requires_reviewed_bridge_transfer_before_demonstrated(self):
        pending = self.loop.ingest_episode(rich_episode())
        accepted = self.learning.create_challenge(reviewed_candidate(pending), confirm_save=True)
        self.loop.accept_pending_import(pending["import_id"], challenge_id=accepted["challenge_id"], revision=1)
        cycle = self.loop.list_learning_loops()[0]
        bridge_occurrence = cycle["plan"]["consolidation_occurrences"][-1]
        bridge_before = cycle["plan"]["bridge_targets"][0]["state"]
        self.assertEqual(bridge_before, "planned")
        with self.assertRaises(LearningLoopRuntimeError) as ctx:
            self.loop.add_consolidation_attempt(
                cycle["cycle_id"],
                bridge_occurrence["occurrence_id"],
                {
                    "response_text": "Synthetic joint transfer response.",
                    "result": "retained",
                    "clinician_reviewed": False,
                },
            )
        self.assertEqual(ctx.exception.code, "consolidation_result_requires_clinician_review")
        self.assertEqual(self.loop.get_learning_loop(cycle["cycle_id"])["plan"]["bridge_targets"][0]["state"], "planned")

    def test_resource_overlay_update_does_not_change_challenge_hash(self):
        pending = self.loop.ingest_episode(rich_episode())
        accepted = self.learning.create_challenge(reviewed_candidate(pending), confirm_save=True)
        immutable_hash = accepted["content_hash"]
        self.loop.accept_pending_import(pending["import_id"], challenge_id=accepted["challenge_id"], revision=1)
        resource = self.loop.list_learning_loops()[0]["resources"][0]
        self.loop.update_resource_status(resource["recommendation_id"], {"status": "completed"})
        detail = self.learning.get_challenge(accepted["challenge_id"])
        self.assertEqual(detail["revisions"][0]["content_hash"], immutable_hash)

    def test_challenge_delete_cleans_loop_attempts_resources_and_due(self):
        pending = self.loop.ingest_episode(rich_episode())
        accepted = self.learning.create_challenge(reviewed_candidate(pending), confirm_save=True)
        self.loop.accept_pending_import(pending["import_id"], challenge_id=accepted["challenge_id"], revision=1)
        cycle = self.loop.list_learning_loops()[0]
        first = cycle["plan"]["consolidation_occurrences"][0]
        self.loop.add_consolidation_attempt(
            cycle["cycle_id"],
            first["occurrence_id"],
            {"response_text": "Synthetic response.", "result": "not_assessed"},
        )
        self.learning.delete_challenge(accepted["challenge_id"], confirm_delete=True)
        self.assertEqual(self.loop.list_learning_loops(), [])
        self.assertEqual(self.learning.list_due(), [])

    def test_external_ingress_uses_dedicated_key_and_is_synthetic_only(self):
        app = FastAPI()
        app.include_router(build_learning_router(engine()))
        with patch.dict(os.environ, {"CLINICAL_DATA_KEY": "clinical-key"}, clear=True):
            client = TestClient(app)
            unconfigured = client.post(
                "/clinical/learning/api/ingress/episodes",
                json={"episode": automatic_rich_episode()},
            )
            self.assertEqual(unconfigured.status_code, 503)
            self.assertEqual(unconfigured.json()["detail"]["code"], "learning_ingress_not_configured")

        with patch.dict(
            os.environ,
            {"CLINICAL_DATA_KEY": "clinical-key", "CLINICAL_LEARNING_INGEST_KEY": "ingest-key"},
            clear=True,
        ):
            client = TestClient(app)
            wrong = client.post(
                "/clinical/learning/api/ingress/episodes",
                headers={"X-Learning-Ingest-Key": "wrong"},
                json={"episode": automatic_rich_episode()},
            )
            self.assertEqual(wrong.status_code, 401)
            ok = client.post(
                "/clinical/learning/api/ingress/episodes",
                headers={"X-Learning-Ingest-Key": "ingest-key"},
                json={"episode": automatic_rich_episode()},
            )
            self.assertEqual(ok.status_code, 200)
            self.assertEqual(ok.json()["state"], "pending_review")

            real_case = automatic_rich_episode()
            real_case["challenge_mode"] = "deidentified_real_case"
            blocked = client.post(
                "/clinical/learning/api/ingress/episodes",
                headers={"X-Learning-Ingest-Key": "ingest-key"},
                json={"episode": real_case},
            )
            self.assertEqual(blocked.status_code, 422)
            self.assertEqual(blocked.json()["detail"]["code"], "external_ingress_synthetic_only")


if __name__ == "__main__":
    unittest.main()
