from __future__ import annotations

import re
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
SCHEMAS = ROOT / "schemas"


def load_yaml(name: str):
    with (SCHEMAS / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def validate_challenge_semantics(payload: dict) -> list[str]:
    errors: list[str] = []
    privacy = payload.get("privacy") or {}
    if privacy.get("contains_direct_identifiers") is not False:
        errors.append("direct identifiers are forbidden")
    if privacy.get("deidentification_attested") is not True:
        errors.append("de-identification attestation must be true")
    mode = payload.get("challenge_mode")
    if mode in {"deidentified_real_case", "mixed"} and privacy.get("source_case_deidentified") is not True:
        errors.append("real/mixed source case must be de-identified")

    facts = {item["fact_id"]: item for item in payload.get("fact_ledger", [])}
    if any(item.get("authoritative_for_patient") is not False for item in facts.values()):
        errors.append("learning facts cannot be patient-authoritative")

    for disclosure in payload.get("progressive_disclosures", []):
        for fact_id in disclosure.get("fact_ids", []):
            if fact_id not in facts:
                errors.append("progressive disclosure references missing fact")

    reference_ids = {item["reference_id"] for item in payload.get("references", [])}
    for observation in payload.get("observations", []):
        for fact_id in observation.get("linked_fact_ids", []):
            if fact_id not in facts:
                errors.append("observation references missing fact")
        for reference_id in observation.get("linked_reference_ids", []):
            if reference_id not in reference_ids:
                errors.append("observation references missing reference")

    revision = payload.get("revision")
    supersedes = payload.get("supersedes_revision")
    if revision == 1 and supersedes is not None:
        errors.append("revision 1 cannot supersede a revision")
    if isinstance(revision, int) and revision > 1 and supersedes != revision - 1:
        errors.append("revision must supersede immediately previous revision")
    return errors


def valid_foundation_attempt(attempt: dict) -> bool:
    evidence = attempt.get("evidence", [])
    reviewed = [item for item in evidence if item.get("clinician_reviewed")]
    non_self = [item for item in reviewed if item.get("method") != "self_rating_only"]
    state = attempt.get("clinician_final_state")
    if state != "UNKNOWN_UNTESTED" and not non_self:
        return False
    if state == "FORMAL_SOLID":
        formal = any(
            item.get("result") == "demonstrated"
            and item.get("method") in {"unaided_explanation", "mechanistic_explanation"}
            for item in reviewed
        )
        transfer = any(
            item.get("result") == "demonstrated"
            and item.get("method")
            in {
                "novel_case_transfer",
                "boundary_or_exception_recognition",
                "evidence_directness_calibration",
            }
            for item in reviewed
        )
        return formal and transfer
    return True


class ClinicalLearningL0ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = load_yaml("clinical_learning_core_v1.yaml")
        cls.foundation = load_yaml("osteoporosis_foundation_map_v1.yaml")
        cls.boundary = load_yaml("clinical_learning_l1_boundary_v1.yaml")
        cls.manifest = load_yaml("clinical_learning_contract_manifest_v1.yaml")
        cls.fixtures = load_yaml("clinical_learning_design_fixtures_v1.yaml")["fixtures"]

    def test_all_contracts_parse_and_have_expected_identity(self):
        self.assertEqual(self.core["schema"], "clinical_learning_core_v1")
        self.assertEqual(self.foundation["schema"], "osteoporosis_foundation_map_v1")
        self.assertEqual(self.boundary["schema"], "clinical_learning_l1_boundary_v1")
        self.assertEqual(self.manifest["schema"], "clinical_learning_contract_manifest_v1")

    def test_all_object_references_resolve(self):
        objects = self.core["objects"]
        refs = []
        for node in walk(self.core):
            ref = node.get("$ref") if isinstance(node, dict) else None
            if ref:
                refs.append(ref)
        self.assertTrue(refs)
        self.assertEqual([], sorted({ref for ref in refs if ref not in objects}))

    def test_foundation_graph_resolves_and_is_unique(self):
        nodes = self.foundation["nodes"]
        ids = [node["node_id"] for node in nodes]
        self.assertEqual(len(ids), len(set(ids)))
        known = set(ids)
        for node in nodes:
            for prereq in node.get("prerequisites", []):
                self.assertIn(prereq["node_id"], known)
            for related in node.get("related_nodes", []):
                self.assertIn(related, known)
        self.assertIn("ost.foundation.evidence_appraisal", known)
        self.assertIn("ost.foundation.denosumab_rebound", known)

    def test_manifest_owner_separation_is_explicit(self):
        owners = self.manifest["ownership_matrix"]
        self.assertEqual(owners["transcript_extraction"], "PR1_PR2_owner_not_learning")
        self.assertEqual(owners["practice_review_interpretation"], "PR3_owner_not_L1")
        self.assertEqual(owners["patient_record_truth"], "clinical_encounter_owner_not_learning")
        self.assertIn("every_learning_fact_authoritative_for_patient_false", self.manifest["hard_invariants"])

    def test_valid_challenge_fixtures_pass(self):
        for name in ("valid_synthetic_challenge", "valid_deidentified_real_case_challenge"):
            payload = self.fixtures[name]["payload"]
            self.assertEqual([], validate_challenge_semantics(payload), name)

    def test_invalid_patient_authority_fixture_fails(self):
        payload = yaml.safe_load(yaml.safe_dump(self.fixtures["valid_synthetic_challenge"]["payload"]))
        payload["fact_ledger"][0]["authoritative_for_patient"] = True
        self.assertIn("learning facts cannot be patient-authoritative", validate_challenge_semantics(payload))

    def test_missing_progressive_fact_fixture_fails(self):
        payload = yaml.safe_load(yaml.safe_dump(self.fixtures["valid_synthetic_challenge"]["payload"]))
        payload["progressive_disclosures"][0]["fact_ids"] = ["99999999-9999-4999-8999-999999999999"]
        self.assertIn("progressive disclosure references missing fact", validate_challenge_semantics(payload))

    def test_real_case_requires_deidentification_attestation(self):
        payload = yaml.safe_load(yaml.safe_dump(self.fixtures["valid_deidentified_real_case_challenge"]["payload"]))
        payload["privacy"]["deidentification_attested"] = False
        self.assertIn("de-identification attestation must be true", validate_challenge_semantics(payload))

    def test_revision_semantics_are_frozen(self):
        rules = self.boundary["challenge_revision_semantics"]["rules"]
        self.assertIn("new_nonduplicate_revision_must_equal_latest_revision_plus_1", rules)
        self.assertIn("same_challenge_id_and_revision_and_same_server_normalized_content_hash_is_idempotent_noop", rules)
        self.assertIn("same_challenge_id_and_revision_but_different_content_is_conflict_and_requires_new_revision", rules)
        self.assertIn("accepted_revision_payload_is_never_updated_in_place", rules)
        self.assertIn("tombstoned_challenge_id_cannot_be_recreated_in_L1", rules)

    def test_delete_is_content_purge_with_noncontent_tombstone(self):
        deletion = self.boundary["challenge_delete_semantics"]
        self.assertEqual(deletion["operation"], "content_purge_plus_noncontent_tombstone")
        self.assertEqual(set(deletion["retained_tombstone_fields"]), {"challenge_id", "deleted_at", "max_deleted_revision"})
        forbidden = set(deletion["forbidden_tombstone_content"])
        self.assertTrue({"facts", "reasoning", "observations", "references"}.issubset(forbidden))
        self.assertIn("clinical_learning_challenge_tombstones", self.boundary["persistence"]["tables"])

    def test_privacy_scanning_is_path_scoped_and_bibliographic_ids_are_excluded(self):
        guard = self.boundary["privacy_guard"]
        paths = set(guard["free_text_scan_paths"])
        self.assertIn("fact_ledger[].statement", paths)
        self.assertIn("reasoning_responses[].text", paths)
        excluded = set(guard["bibliographic_paths_excluded_from_numeric_identity_heuristics"])
        self.assertEqual(excluded, {"references[].pmid", "references[].doi", "references[].url"})
        self.assertTrue(guard["clinician_attestation"]["required_for_challenge_import"])
        self.assertTrue(guard["clinician_attestation"]["required_value"])

    def test_imported_references_default_unverified(self):
        self.assertEqual(self.boundary["reference_handling"]["imported_reference_default"], "unverified")
        ref_field = self.core["objects"]["LearningReferenceV1"]["fields"]["verification_state"]
        self.assertEqual(ref_field["default"], "unverified")

    def test_foundation_self_rating_cannot_promote_state(self):
        attempt = self.fixtures["foundation_self_rating_only"]["attempt"]
        self.assertFalse(valid_foundation_attempt(attempt))

    def test_foundation_formal_solid_requires_formal_plus_transfer_or_calibration(self):
        attempt = self.fixtures["foundation_formal_solid_valid"]["attempt"]
        self.assertTrue(valid_foundation_attempt(attempt))
        invariants = self.core["objects"]["FoundationAssessmentAttemptV1"]["invariants"]
        self.assertTrue(any("formal_solid" in item for item in invariants))

    def test_daily_case_review_is_not_persisted_when_ineligible(self):
        invariants = self.core["objects"]["DailyCaseReviewV1"]["invariants"]
        self.assertIn("persisted_daily_case_review_requires_eligibility_state_eligible", invariants)
        self.assertFalse(self.fixtures["daily_case_no_eligible_case"]["expect_review_record_created"])
        self.assertEqual(self.fixtures["daily_case_no_eligible_case"]["due_state"]["due_status"], "not_applicable")

    def test_daily_case_review_does_not_store_raw_transcript(self):
        descriptor = self.core["objects"]["ReviewEvidenceDescriptorV1"]
        self.assertFalse(descriptor["fields"]["contains_raw_transcript"]["const"])
        invariants = self.core["objects"]["DailyCaseReviewV1"]["invariants"]
        self.assertIn("raw_heidi_transcript_is_not_persisted_in_this_object", invariants)
        self.assertIn("decision_evidence_summary_must_not_be_verbatim_raw_transcript", invariants)

    def test_baseline_daily_case_due_state_can_be_shadow_hidden(self):
        fixture = self.fixtures["daily_case_baseline_shadow"]["due_state"]
        self.assertEqual(fixture["due_status"], "due")
        self.assertEqual(fixture["delivery_mode"], "shadow_hidden")
        due_rules = self.core["objects"]["LearningDueStateV1"]["deterministic_rules"]
        self.assertIn("baseline_phase_may_change_delivery_mode_to_shadow_hidden_without_changing_due_status", due_rules)

    def test_l1_scope_excludes_daily_case_transcript_and_signal_runtime(self):
        forbidden = set(self.boundary["api_boundary_l1"]["forbidden_l1_endpoints"])
        self.assertTrue({"transcript_storage_endpoint", "signal_promotion_endpoint", "daily_case_review_endpoint"}.issubset(forbidden))
        future_tables = set(self.boundary["persistence"]["future_l2_tables_not_created_in_l1"])
        self.assertIn("clinical_learning_daily_case_review_revisions", future_tables)

    def test_no_direct_patient_or_transcript_owner_is_assigned_to_learning(self):
        owner_text = yaml.safe_dump(self.boundary["owners"], sort_keys=False)
        self.assertIn("patient_record_mutation", owner_text)
        self.assertIn("transcript_provider_extraction", owner_text)
        self.assertIn("does_not_own", owner_text)


if __name__ == "__main__":
    unittest.main()
