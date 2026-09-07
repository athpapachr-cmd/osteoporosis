from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).resolve().parent


def load(name):
    with (ROOT / "schemas" / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class ClinicalLearningL0BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.boundary = load("clinical_learning_l1_boundary_v1.yaml")
        cls.core = load("clinical_learning_core_v1.yaml")

    def test_external_import_cannot_self_certify_review_signal_or_reference_verification(self):
        authority = self.boundary["external_challenge_import_authority"]
        untrusted = set(authority["untrusted_inbound_fields"])
        self.assertTrue({"record_review_state", "reviewed_at", "linked_signal_ids", "references[].verification_state"}.issubset(untrusted))
        normalized = authority["preview_normalization"]
        self.assertEqual(normalized["record_review_state"], "imported_pending_review")
        self.assertIsNone(normalized["reviewed_at"])
        self.assertEqual(normalized["linked_signal_ids"], [])
        self.assertEqual(normalized["references_verification_state"], "unverified")
        final = set(authority["final_save_rules"])
        self.assertIn("record_review_state_set_by_server_to_clinician_reviewed", final)
        self.assertIn("linked_signal_ids_remain_empty_in_L1", final)

    def test_l1_topics_are_tags_not_second_ontology(self):
        semantics = self.boundary["topic_semantics"]
        self.assertEqual(semantics["controlled_structure"], "foundation_node_ids")
        self.assertEqual(semantics["descriptive_topics"]["type"], "normalized_free_tags")
        self.assertNotIn("osteoporosis_topic_taxonomy", self.boundary["owners"]["module_01"]["owns"])

    def test_privacy_scans_title_and_does_not_numeric_scan_bibliographic_locators(self):
        privacy = self.boundary["privacy_guard"]
        self.assertIn("title", privacy["free_text_scan_paths"])
        excluded = set(privacy["bibliographic_paths_excluded_from_numeric_identity_heuristics"])
        self.assertEqual(excluded, {"references[].pmid", "references[].doi", "references[].url"})

    def test_due_items_have_repeatable_occurrence_semantics(self):
        table = self.boundary["persistence"]["tables"]["clinical_learning_due_items"]
        self.assertIn("occurrence", table["columns"])
        self.assertEqual(table["unique_logical_occurrence"], ["item_type", "target_id", "occurrence"])
        rules = set(table["rules"])
        self.assertIn("completed_occurrence_is_historical_and_not_reopened", rules)
        self.assertIn("scheduling_after_completed_occurrence_creates_occurrence_plus_1", rules)
        due = self.boundary["l1_due_semantics"]
        self.assertEqual(due["occurrence_starts_at"], 1)

    def test_l1_foundation_assessment_is_explicit_only(self):
        mutation = self.boundary["foundation_state_mutation"]
        self.assertEqual(mutation["l1_allowed_assessment_source_artifact_types"], ["foundation_assessment"])
        self.assertEqual(mutation["l1_authoritative_source"], "explicit_foundation_assessment")
        table = self.boundary["persistence"]["tables"]["clinical_learning_foundation_attempts"]
        self.assertIn("explicit foundation_assessment evidence only", table["l1_source_rule"])

    def test_delete_tombstone_has_no_learning_content(self):
        deletion = self.boundary["challenge_delete_semantics"]
        self.assertEqual(deletion["operation"], "content_purge_plus_noncontent_tombstone")
        retained = set(deletion["retained_tombstone_fields"])
        self.assertEqual(retained, {"challenge_id", "deleted_at", "max_deleted_revision"})
        forbidden = set(deletion["forbidden_tombstone_content"])
        self.assertTrue({"title", "topics", "facts", "reasoning", "observations", "references"}.issubset(forbidden))

    def test_daily_case_record_exists_only_for_eligible_case(self):
        invariants = set(self.core["objects"]["DailyCaseReviewV1"]["invariants"])
        self.assertIn("persisted_daily_case_review_requires_eligibility_state_eligible", invariants)
        self.assertIn("decision_evidence_summary_must_not_be_verbatim_raw_transcript", invariants)


if __name__ == "__main__":
    unittest.main()
