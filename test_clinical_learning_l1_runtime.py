from __future__ import annotations

import copy
import os
import unittest
from datetime import date

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from clinical_auth import ClinicalCookieMiddleware, build_auth_router
from clinical_learning.api import build_learning_router
from clinical_learning.service import ClinicalLearningService


CHALLENGE_ID = "11111111-1111-4111-8111-111111111111"
FACT_ID = "22222222-2222-4222-8222-222222222222"
RESPONSE_ID = "33333333-3333-4333-8333-333333333333"
OBS_ID = "44444444-4444-4444-8444-444444444444"
REF_ID = "55555555-5555-4555-8555-555555555555"
ACTION_ID = "66666666-6666-4666-8666-666666666666"
FOUNDATION_NODE = "ost.foundation.denosumab_rebound"


def engine():
    return create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def challenge(*, disposition: str = "pending"):
    return {
        "challenge_id": CHALLENGE_ID,
        "schema_version": "clinical_learning_challenge_v1",
        "revision": 1,
        "supersedes_revision": None,
        "module": "osteoporosis",
        "created_at": "2026-09-07T12:00:00Z",
        "title": "Synthetic denosumab transition challenge",
        "challenge_mode": "synthetic",
        "topics": ["Denosumab", "  denosumab  ", "Sequencing"],
        "foundation_node_ids": [FOUNDATION_NODE],
        "difficulty_label": "advanced",
        "initial_case": "Synthetic teaching case without patient identifiers.",
        "fact_ledger": [
            {
                "fact_id": FACT_ID,
                "statement": "Synthetic fact for a denosumab transition exercise.",
                "fact_scope": "synthetic_case_fact",
                "introduced_via": "initial_case",
                "authoritative_for_patient": False,
                "source": "synthetic_fixture",
                "certainty": "high",
                "introduced_at_stage": "initial",
                "status": "active",
                "supersedes_fact_id": None,
            }
        ],
        "progressive_disclosures": [],
        "reasoning_responses": [
            {
                "response_id": RESPONSE_ID,
                "stage": "initial",
                "text": "I would first verify the treatment timeline and transition constraints.",
                "confidence_percent": 75,
                "created_at": "2026-09-07T12:05:00Z",
            }
        ],
        "final_clinician_decision": "Synthetic final decision for learning only.",
        "observations": [
            {
                "observation_id": OBS_ID,
                "category": "reasoning_pattern",
                "statement": "The reasoning explicitly prioritized sequencing safety.",
                "importance": "moderate",
                "linked_fact_ids": [FACT_ID],
                "linked_reference_ids": [REF_ID],
                "gap_classes": [],
                "clinician_disposition": disposition,
                "clinician_modified_statement": None,
                "disposition_note": None,
            }
        ],
        "references": [
            {
                "reference_id": REF_ID,
                "title": "Synthetic reference locator fixture",
                "evidence_type": "guideline",
                "framework_or_guideline": "Synthetic framework",
                "pmid": "12345678",
                "doi": None,
                "url": None,
                "relation": "contextualizes",
                "verification_state": "verified_content",
                "verification_note": "Imported claim must be reset by server",
            }
        ],
        "gap_classes": [],
        "learning_actions": [
            {
                "action_id": ACTION_ID,
                "action_type": "retrieval_test",
                "title": "Repeat sequencing retrieval test",
                "rationale": "Synthetic spaced retrieval",
                "foundation_node_ids": [FOUNDATION_NODE],
                "reference_ids": [REF_ID],
                "due_on": "2026-09-10",
                "status": "planned",
                "completed_at": None,
            }
        ],
        "next_challenge_topic": "sequencing",
        "spaced_repetition_due": "2026-09-12",
        "linked_signal_ids": ["external-untrusted-signal"],
        "record_review_state": "clinician_reviewed",
        "reviewed_at": "2026-09-07T12:10:00Z",
        "privacy": {
            "contains_direct_identifiers": False,
            "deidentification_attested": True,
            "source_case_deidentified": None,
        },
    }


def foundation_attempt(*, formal=True):
    evidence = [
        {
            "evidence_id": "77777777-7777-4777-8777-777777777777",
            "method": "mechanistic_explanation" if formal else "self_rating_only",
            "result": "demonstrated" if formal else "not_assessed",
            "clinician_reviewed": True,
            "note": "Synthetic assessment evidence.",
            "source_artifact_type": "foundation_assessment",
            "source_artifact_id": None,
        }
    ]
    if formal:
        evidence.append(
            {
                "evidence_id": "88888888-8888-4888-8888-888888888888",
                "method": "boundary_or_exception_recognition",
                "result": "demonstrated",
                "clinician_reviewed": True,
                "note": "Synthetic transfer evidence.",
                "source_artifact_type": "foundation_assessment",
                "source_artifact_id": None,
            }
        )
    return {
        "attempt_id": "99999999-9999-4999-8999-999999999999",
        "schema_version": "foundation_assessment_attempt_v1",
        "module": "osteoporosis",
        "foundation_node_id": FOUNDATION_NODE,
        "assessed_at": "2026-09-07T13:00:00Z",
        "evidence": evidence,
        "proposed_state": "FORMAL_SOLID" if formal else "FORMAL_SOLID",
        "clinician_final_state": "FORMAL_SOLID",
        "clinician_note": "Synthetic assessment only.",
    }


class ClinicalLearningL1RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.engine = engine()
        self.service = ClinicalLearningService(self.engine)

    def test_preview_normalizes_external_authority_and_topics(self):
        result = self.service.preview_challenge(challenge())
        self.assertTrue(result["valid"])
        normalized = result["normalized_summary"]
        self.assertEqual(normalized["record_review_state"], "imported_pending_review")
        self.assertIsNone(normalized["reviewed_at"])
        self.assertEqual(normalized["linked_signal_ids"], [])
        self.assertEqual(normalized["references"][0]["verification_state"], "unverified")
        self.assertIsNone(normalized["references"][0]["verification_note"])
        self.assertEqual(normalized["observations"][0]["clinician_disposition"], "pending")
        self.assertEqual(normalized["topics"], ["Denosumab", "Sequencing"])

    def test_recursive_unknown_field_rejected_without_echo(self):
        payload = challenge()
        payload["fact_ledger"][0]["patient_name"] = "Synthetic Person"
        result = self.service.preview_challenge(payload)
        self.assertFalse(result["valid"])
        text = repr(result)
        self.assertIn("direct_identifier_field_forbidden", text)
        self.assertNotIn("Synthetic Person", text)

    def test_free_text_phi_rejected_but_pmid_numeric_locator_allowed(self):
        ok = self.service.preview_challenge(challenge())
        self.assertTrue(ok["valid"])
        payload = challenge()
        payload["initial_case"] = "Contact test@example.com for this synthetic example."
        blocked = self.service.preview_challenge(payload)
        self.assertFalse(blocked["valid"])
        self.assertIn("email_address_detected", repr(blocked))
        self.assertNotIn("test@example.com", repr(blocked))

    def test_save_revision_idempotency_due_overlay_and_delete(self):
        accepted = challenge(disposition="accepted")
        first = self.service.create_challenge(accepted, confirm_save=True)
        self.assertEqual(first["revision"], 1)
        self.assertFalse(first["idempotent"])
        self.assertEqual(len(self.service.list_due()), 2)

        repeat = self.service.create_challenge(accepted, confirm_save=True)
        self.assertTrue(repeat["idempotent"])
        self.assertEqual(repeat["revision"], 1)

        changed = copy.deepcopy(accepted)
        changed["title"] = "Revised synthetic denosumab challenge"
        revised = self.service.revise_challenge(CHALLENGE_ID, changed, confirm_save=True)
        self.assertEqual(revised["revision"], 2)
        self.assertEqual(revised["payload"]["supersedes_revision"], 1)
        immutable_hash = revised["content_hash"]

        overlay = self.service.set_reference_verification(
            challenge_id=CHALLENGE_ID,
            revision=2,
            reference_id=REF_ID,
            verification_state="verified_locator",
            verification_note=None,
        )
        self.assertEqual(overlay["verification_state"], "verified_locator")
        detail = self.service.get_challenge(CHALLENGE_ID)
        latest = detail["revisions"][-1]
        self.assertEqual(latest["content_hash"], immutable_hash)
        self.assertEqual(latest["payload"]["references"][0]["verification_state"], "unverified")
        self.assertEqual(latest["reference_verification"][0]["verification_state"], "verified_locator")

        deleted = self.service.delete_challenge(CHALLENGE_ID, confirm_delete=True)
        self.assertTrue(deleted["deleted"])
        tombstone = self.service.get_challenge(CHALLENGE_ID)
        self.assertTrue(tombstone["deleted"])
        self.assertEqual(tombstone["max_deleted_revision"], 2)
        self.assertEqual(self.service.list_due(), [])
        self.assertNotIn("Revised synthetic", repr(tombstone))

    def test_foundation_formal_solid_requires_formal_plus_transfer_evidence(self):
        blocked = self.service.preview_foundation_assessment(
            FOUNDATION_NODE,
            foundation_attempt(formal=False),
            next_review_due=date(2026, 10, 1),
        )
        self.assertFalse(blocked["valid"])
        self.assertIn("formal_solid_requires_formal_evidence", repr(blocked))

        valid = self.service.preview_foundation_assessment(
            FOUNDATION_NODE,
            foundation_attempt(formal=True),
            next_review_due=date(2026, 10, 1),
        )
        self.assertTrue(valid["valid"])
        saved = self.service.save_foundation_assessment(
            FOUNDATION_NODE,
            foundation_attempt(formal=True),
            next_review_due=date(2026, 10, 1),
            confirm_save=True,
        )
        self.assertEqual(saved["state"]["state"], "FORMAL_SOLID")
        self.assertEqual(saved["attempt_count"], 1)
        self.assertIsNotNone(saved["due"])

    def test_api_header_and_browser_session_auth(self):
        old = os.environ.get("CLINICAL_DATA_KEY")
        os.environ["CLINICAL_DATA_KEY"] = "synthetic-test-key"
        try:
            app = FastAPI()
            app.add_middleware(ClinicalCookieMiddleware)
            app.include_router(build_auth_router())
            app.include_router(build_learning_router(self.engine))
            client = TestClient(app, base_url="https://testserver")

            no_auth = client.get("/clinical/learning/api/foundation")
            self.assertEqual(no_auth.status_code, 401)

            header = client.get(
                "/clinical/learning/api/foundation",
                headers={"X-Clinical-Key": "synthetic-test-key"},
            )
            self.assertEqual(header.status_code, 200)
            self.assertEqual(len(header.json()["items"]), 14)

            login = client.post("/clinical/login", json={"key": "synthetic-test-key"})
            self.assertEqual(login.status_code, 200)
            cookie = client.get("/clinical/learning/api/foundation")
            self.assertEqual(cookie.status_code, 200)

            invalid_envelope = client.post(
                "/clinical/learning/api/challenges/preview",
                headers={"X-Clinical-Key": "synthetic-test-key"},
                json={"challenge": challenge(), "unexpected": "private-value"},
            )
            self.assertEqual(invalid_envelope.status_code, 422)
            self.assertNotIn("private-value", invalid_envelope.text)
        finally:
            if old is None:
                os.environ.pop("CLINICAL_DATA_KEY", None)
            else:
                os.environ["CLINICAL_DATA_KEY"] = old


if __name__ == "__main__":
    unittest.main()
