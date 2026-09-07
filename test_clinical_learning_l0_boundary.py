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
        self.assertTrue({
            "record_review_state",
            "reviewed_at",
            "linked_signal_ids",
            "references[].verification_state",
            "references[].verification_note",
        }.issubset(untrusted))
        self.assertEqual(authority["unknown_field_policy"], "reject_recursively")
        normalized = authority["preview_normalization"]
        self.assertEqual(normalized["record_review_state"], "imported_pending_review")
        self.assertIsNone(normalized["reviewed_at"])
        self.assertEqual(normalized["linked_signal_ids"], [])
        self.assertEqual(normalized["references_verification_state"], "unverified")
        self.assertIsNone(normalized["references_verification_note"])
        final = set(authority["final_save_rules"])
        self.assertIn("record_review_state_set_by_server_to_clinician_reviewed", final)
        self.assertIn("linked_signal_ids_remain_empty_in_L1", final)
        self.assertIn("current_reference_verification_changes_only_through_separate_server_clinician_overlay_action", final)

    def test_reference_verification_overlay_preserves_revision_immutability(self):
        tables = self.boundary["persistence"]["tables"]
        self.assertIn("clinical_learning_reference_verification", tables)
        overlay = tables["clinical_learning_reference_verification"]
        self.assertEqual(overlay["primary_key"], ["challenge_id", "revision", "reference_id"])
        self.assertIn("overlay_changes_never_mutate_challenge_payload_or_content_hash", overlay["invariants"])
        endpoints = self.boundary["api_boundary_l1"]["endpoints"]
        verification = [item for item in endpoints if item.get("purpose") == "server_clinician_owned_reference_verification_overlay_without_mutating_revision_payload"]
        self.assertEqual(len(verification), 1)
        self.assertEqual(verification[0]["method"], "POST")

    def test_l1_topics_are_tags_not_second_ontology(self):
        semantics = self.boundary["topic_semantics"]
        self.assertEqual(semantics["controlled_structure"], "foundation_node_ids")
        self.assertEqual(semantics["descriptive_topics"]["type"], "normalized_free_tags")
        self.assertNotIn("osteoporosis_topic_taxonomy", self.boundary["owners"]["module_01"]["owns"])

    def test_privacy_guard_is_fail_closed_for_schema_and_persistable_text(self):
        privacy = self.boundary["privacy_guard"]
        self.assertEqual(privacy["challenge_unknown_field_policy"], "reject_recursively")
        self.assertEqual(privacy["foundation_assessment_unknown_field_policy"], "reject_recursively")
        self.assertIn("Every user/import-supplied string field", privacy["untrusted_string_scan_rule"])
        paths = set(privacy["minimum_challenge_free_text_scan_paths"])
        self.assertTrue({
            "title",
            "topics[]",
            "fact_ledger[].source",
            "progressive_disclosures[].label",
            "references[].title",
            "references[].framework_or_guideline",
            "references[].verification_note",
        }.issubset(paths))
        excluded = set(privacy["bibliographic_paths_excluded_from_numeric_identity_heuristics"])
        self.assertEqual(excluded, {"references[].pmid", "references[].doi", "references[].url"})

    def test_due_items_have_repeatable_occurrence_and_source_provenance(self):
        table = self.boundary["persistence"]["tables"]["clinical_learning_due_items"]
        self.assertIn("occurrence", table["columns"])
        self.assertIn("source_artifact_type", table["columns"])
        self.assertIn("source_artifact_id", table["columns"])
        self.assertEqual(table["unique_logical_occurrence"], ["item_type", "target_id", "occurrence"])
        rules = set(table["rules"])
        self.assertIn("completed_occurrence_is_historical_and_not_reopened", rules)
        self.assertIn("scheduling_after_completed_occurrence_creates_occurrence_plus_1", rules)
        self.assertIn("challenge_delete_removes_rows_whose_target_or_source_is_the_deleted_challenge", rules)
        due = self.boundary["l1_due_semantics"]
        self.assertEqual(due["occurrence_starts_at"], 1)
        self.assertTrue(due["target_id_required_for_l1_materialized_items"])
        self.assertEqual(due["source_artifact_fields_required"], ["source_artifact_type", "source_artifact_id"])

    def test_challenge_delete_purges_nested_action_due_rows_and_reference_overlay(self):
        deletion = self.boundary["challenge_delete_semantics"]
        steps = set(deletion["transactional_steps"])
        self.assertIn("identify_all_due_items_whose_target_or_source_artifact_is_the_challenge", steps)
        self.assertIn("delete_all_due_items_targeting_or_sourced_from_challenge_id", steps)
        self.assertIn("delete_all_reference_verification_overlay_rows_for_challenge_id", steps)
        self.assertIn("learning_action due items targeted by nested action_id", deletion["due_referential_rule"])

    def test_l1_foundation_assessment_is_explicit_only(self):
        mutation = self.boundary["foundation_state_mutation"]
        self.assertEqual(mutation["l1_allowed_assessment_source_artifact_types"], ["foundation_assessment"])
        self.assertEqual(mutation["l1_authoritative_source"], "explicit_foundation_assessment")
        table = self.boundary["persistence"]["tables"]["clinical_learning_foundation_attempts"]
        self.assertIn("explicit foundation_assessment evidence only", table["l1_source_rule"])
        self.assertIn("direct patient identifiers are rejected", table["privacy_rule"])

    def test_delete_tombstone_has_no_learning_content(self):
        deletion = self.boundary["challenge_delete_semantics"]
        self.assertEqual(deletion["operation"], "content_purge_plus_noncontent_tombstone")
        retained = set(deletion["retained_tombstone_fields"])
        self.assertEqual(retained, {"challenge_id", "deleted_at", "max_deleted_revision"})
        forbidden = set(deletion["forbidden_tombstone_content"])
        self.assertTrue({"title", "topics", "facts", "reasoning", "observations", "references"}.issubset(forbidden))

    def test_daily_case_record_exists_only_for_eligible_case_and_references_resolve(self):
        invariants = set(self.core["objects"]["DailyCaseReviewV1"]["invariants"])
        self.assertIn("persisted_daily_case_review_requires_eligibility_state_eligible", invariants)
        self.assertIn("decision_evidence_summary_must_not_be_verbatim_raw_transcript", invariants)
        self.assertIn("every_decision_linked_fact_id_resolves_within_fact_ledger", invariants)
        self.assertIn("every_observation_fact_reference_resolves_within_fact_ledger", invariants)
        self.assertIn("every_observation_reference_id_resolves_within_references", invariants)
        self.assertIn("every_learning_action_reference_id_resolves_within_references", invariants)

    def test_signal_links_do_not_mutate_immutable_learning_payloads(self):
        linkage = self.boundary["signal_linkage_boundary"]
        self.assertEqual(linkage["authoritative_owner"], "future_shared_signal_engine")
        self.assertIn("must not mutate an accepted Challenge or Daily Case Review", linkage["immutable_revision_rule"])
        self.assertEqual(linkage["l1_challenge_linked_signal_ids"], "normalized_empty_non_authoritative")


if __name__ == "__main__":
    unittest.main()
