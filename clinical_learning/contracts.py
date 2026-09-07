from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml
from pydantic import ValidationError

from .models import ClinicalLearningChallengeV1, FoundationAssessmentAttemptV1
from .privacy import PrivacyFinding, find_forbidden_structured_fields, scan_persistable_strings


@dataclass(frozen=True)
class ContractIssue:
    code: str
    path: str


class LearningContractError(ValueError):
    def __init__(self, issues: Iterable[ContractIssue]):
        self.issues = list(issues)
        super().__init__("Clinical Learning contract validation failed")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid learning contract artifact: {path}")
    return data


class FoundationRegistry:
    def __init__(self, root: Path | None = None):
        path = (root or _repo_root()) / "schemas/osteoporosis_foundation_map_v1.yaml"
        data = _load_yaml(path)
        if data.get("schema") != "osteoporosis_foundation_map_v1":
            raise RuntimeError("Unexpected osteoporosis Foundation Map schema")
        nodes = data.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            raise RuntimeError("Osteoporosis Foundation Map has no nodes")
        self.data = data
        self.nodes: dict[str, dict[str, Any]] = {}
        for node in nodes:
            if not isinstance(node, dict) or not node.get("node_id"):
                raise RuntimeError("Invalid Foundation node")
            node_id = str(node["node_id"])
            if node_id in self.nodes:
                raise RuntimeError(f"Duplicate Foundation node: {node_id}")
            self.nodes[node_id] = copy.deepcopy(node)
        for node_id, node in self.nodes.items():
            for prereq in node.get("prerequisites") or []:
                target = prereq.get("node_id") if isinstance(prereq, dict) else None
                if target not in self.nodes:
                    raise RuntimeError(f"Foundation node {node_id} references unknown prerequisite {target}")
            for related in node.get("related_nodes") or []:
                if related not in self.nodes:
                    raise RuntimeError(f"Foundation node {node_id} references unknown related node {related}")

    def contains(self, node_id: str) -> bool:
        return node_id in self.nodes

    def public_nodes(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(node) for node in self.nodes.values()]


_REGISTRY: FoundationRegistry | None = None


def get_foundation_registry() -> FoundationRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = FoundationRegistry()
    return _REGISTRY


def _loc_to_path(loc: tuple[Any, ...]) -> str:
    out = ""
    for item in loc:
        if isinstance(item, int):
            out += f"[{item}]"
        else:
            out += ("." if out else "") + str(item)
    return out


def sanitized_schema_issues(exc: ValidationError) -> list[ContractIssue]:
    return [
        ContractIssue(code=f"schema_{err.get('type', 'invalid')}", path=_loc_to_path(tuple(err.get("loc") or ())))
        for err in exc.errors(include_url=False, include_context=False, include_input=False)
    ]


def _privacy_issues(findings: Iterable[PrivacyFinding]) -> list[ContractIssue]:
    return [ContractIssue(code=f.code, path=f.path) for f in findings]


def _duplicates(values: Iterable[Any]) -> set[Any]:
    seen: set[Any] = set()
    dupes: set[Any] = set()
    for value in values:
        if value in seen:
            dupes.add(value)
        seen.add(value)
    return dupes


def _validate_challenge_integrity(challenge: ClinicalLearningChallengeV1, registry: FoundationRegistry) -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    facts = {str(item.fact_id): item for item in challenge.fact_ledger}
    fact_ids = [str(item.fact_id) for item in challenge.fact_ledger]
    if _duplicates(fact_ids):
        issues.append(ContractIssue("duplicate_fact_id", "fact_ledger"))
    for idx, fact in enumerate(challenge.fact_ledger):
        if fact.supersedes_fact_id is not None:
            supersedes = str(fact.supersedes_fact_id)
            if supersedes == str(fact.fact_id) or supersedes not in facts:
                issues.append(ContractIssue("invalid_fact_supersedes_reference", f"fact_ledger[{idx}].supersedes_fact_id"))

    response_ids = [str(item.response_id) for item in challenge.reasoning_responses]
    if _duplicates(response_ids):
        issues.append(ContractIssue("duplicate_reasoning_response_id", "reasoning_responses"))
    response_set = set(response_ids)

    disclosure_ids = [str(item.disclosure_id) for item in challenge.progressive_disclosures]
    if _duplicates(disclosure_ids):
        issues.append(ContractIssue("duplicate_disclosure_id", "progressive_disclosures"))
    sequences = [item.sequence for item in challenge.progressive_disclosures]
    if len(sequences) != len(set(sequences)) or sorted(sequences) != list(range(1, len(sequences) + 1)):
        issues.append(ContractIssue("invalid_disclosure_sequence", "progressive_disclosures"))
    for idx, disclosure in enumerate(challenge.progressive_disclosures):
        for fact_id in disclosure.fact_ids:
            if str(fact_id) not in facts:
                issues.append(ContractIssue("unresolved_disclosure_fact_id", f"progressive_disclosures[{idx}].fact_ids"))
        if disclosure.released_after_response_id is not None and str(disclosure.released_after_response_id) not in response_set:
            issues.append(ContractIssue("unresolved_disclosure_response_id", f"progressive_disclosures[{idx}].released_after_response_id"))

    reference_ids = [str(item.reference_id) for item in challenge.references]
    reference_set = set(reference_ids)
    if _duplicates(reference_ids):
        issues.append(ContractIssue("duplicate_reference_id", "references"))

    observation_ids = [str(item.observation_id) for item in challenge.observations]
    if _duplicates(observation_ids):
        issues.append(ContractIssue("duplicate_observation_id", "observations"))
    for idx, observation in enumerate(challenge.observations):
        for fact_id in observation.linked_fact_ids:
            if str(fact_id) not in facts:
                issues.append(ContractIssue("unresolved_observation_fact_id", f"observations[{idx}].linked_fact_ids"))
        for reference_id in observation.linked_reference_ids:
            if str(reference_id) not in reference_set:
                issues.append(ContractIssue("unresolved_observation_reference_id", f"observations[{idx}].linked_reference_ids"))
        if observation.clinician_disposition == "modified" and not (observation.clinician_modified_statement or "").strip():
            issues.append(ContractIssue("modified_observation_requires_statement", f"observations[{idx}].clinician_modified_statement"))

    action_ids = [str(item.action_id) for item in challenge.learning_actions]
    if _duplicates(action_ids):
        issues.append(ContractIssue("duplicate_learning_action_id", "learning_actions"))
    for idx, action in enumerate(challenge.learning_actions):
        for reference_id in action.reference_ids:
            if str(reference_id) not in reference_set:
                issues.append(ContractIssue("unresolved_action_reference_id", f"learning_actions[{idx}].reference_ids"))
        for node_id in action.foundation_node_ids:
            if not registry.contains(node_id):
                issues.append(ContractIssue("unknown_foundation_node", f"learning_actions[{idx}].foundation_node_ids"))

    for idx, node_id in enumerate(challenge.foundation_node_ids):
        if not registry.contains(node_id):
            issues.append(ContractIssue("unknown_foundation_node", f"foundation_node_ids[{idx}]"))

    if challenge.revision == 1 and challenge.supersedes_revision is not None:
        issues.append(ContractIssue("revision_1_requires_null_supersedes", "supersedes_revision"))
    if challenge.revision > 1 and challenge.supersedes_revision != challenge.revision - 1:
        issues.append(ContractIssue("revision_must_supersede_immediate_predecessor", "supersedes_revision"))

    if not challenge.privacy.deidentification_attested:
        issues.append(ContractIssue("deidentification_attestation_required", "privacy.deidentification_attested"))
    if challenge.challenge_mode in {"deidentified_real_case", "mixed"} and challenge.privacy.source_case_deidentified is not True:
        issues.append(ContractIssue("real_case_requires_source_deidentified", "privacy.source_case_deidentified"))
    return issues


def validate_challenge_payload(raw_challenge: Any) -> ClinicalLearningChallengeV1:
    issues = _privacy_issues(find_forbidden_structured_fields(raw_challenge))
    if issues:
        raise LearningContractError(issues)
    try:
        challenge = ClinicalLearningChallengeV1.model_validate(raw_challenge)
    except ValidationError as exc:
        raise LearningContractError(sanitized_schema_issues(exc)) from None
    normalized = challenge.model_dump(mode="json")
    issues.extend(_privacy_issues(scan_persistable_strings(normalized)))
    issues.extend(_validate_challenge_integrity(challenge, get_foundation_registry()))
    if issues:
        raise LearningContractError(issues)
    return challenge


def normalize_topics(topics: Iterable[str]) -> tuple[list[str], list[str]]:
    display: list[str] = []
    keys: list[str] = []
    seen: set[str] = set()
    for raw in topics:
        label = " ".join(str(raw).split()).strip()
        key = label.casefold()
        if not label or key in seen:
            continue
        seen.add(key)
        display.append(label)
        keys.append(key)
    return display, keys


def normalize_import_preview(challenge: ClinicalLearningChallengeV1) -> dict[str, Any]:
    payload = challenge.model_dump(mode="json")
    payload["topics"] = normalize_topics(payload.get("topics") or [])[0]
    payload["record_review_state"] = "imported_pending_review"
    payload["reviewed_at"] = None
    payload["linked_signal_ids"] = []
    for reference in payload.get("references") or []:
        reference["verification_state"] = "unverified"
        reference["verification_note"] = None
    for observation in payload.get("observations") or []:
        observation["clinician_disposition"] = "pending"
        observation["clinician_modified_statement"] = None
        observation["disposition_note"] = None
    return payload


def normalize_for_save(challenge: ClinicalLearningChallengeV1, *, reviewed_at_iso: str) -> dict[str, Any]:
    payload = challenge.model_dump(mode="json")
    payload["topics"] = normalize_topics(payload.get("topics") or [])[0]
    payload["record_review_state"] = "clinician_reviewed"
    payload["reviewed_at"] = reviewed_at_iso
    payload["linked_signal_ids"] = []
    for reference in payload.get("references") or []:
        reference["verification_state"] = "unverified"
        reference["verification_note"] = None
    for idx, observation in enumerate(payload.get("observations") or []):
        if observation.get("clinician_disposition") == "pending":
            raise LearningContractError([ContractIssue("observation_disposition_required", f"observations[{idx}].clinician_disposition")])
        if observation.get("clinician_disposition") == "modified" and not str(observation.get("clinician_modified_statement") or "").strip():
            raise LearningContractError([ContractIssue("modified_observation_requires_statement", f"observations[{idx}].clinician_modified_statement")])
    findings = scan_persistable_strings(payload)
    if findings:
        raise LearningContractError(_privacy_issues(findings))
    return payload


def canonical_content_hash(payload: dict[str, Any]) -> str:
    material = copy.deepcopy(payload)
    material.pop("reviewed_at", None)
    material.pop("record_review_state", None)
    material.pop("linked_signal_ids", None)
    for reference in material.get("references") or []:
        reference["verification_state"] = "unverified"
        reference["verification_note"] = None
    encoded = json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_foundation_attempt_payload(raw_attempt: Any, *, expected_node_id: str) -> FoundationAssessmentAttemptV1:
    issues = _privacy_issues(find_forbidden_structured_fields(raw_attempt))
    if issues:
        raise LearningContractError(issues)
    try:
        attempt = FoundationAssessmentAttemptV1.model_validate(raw_attempt)
    except ValidationError as exc:
        raise LearningContractError(sanitized_schema_issues(exc)) from None
    issues.extend(_privacy_issues(scan_persistable_strings(attempt.model_dump(mode="json"))))
    registry = get_foundation_registry()
    if attempt.foundation_node_id != expected_node_id or not registry.contains(attempt.foundation_node_id):
        issues.append(ContractIssue("foundation_node_mismatch_or_unknown", "foundation_node_id"))
    ids = [str(item.evidence_id) for item in attempt.evidence]
    if _duplicates(ids):
        issues.append(ContractIssue("duplicate_foundation_evidence_id", "evidence"))
    for idx, evidence in enumerate(attempt.evidence):
        if evidence.source_artifact_type != "foundation_assessment":
            issues.append(ContractIssue("l1_foundation_source_must_be_explicit_assessment", f"evidence[{idx}].source_artifact_type"))
    non_self_reviewed = [e for e in attempt.evidence if e.method != "self_rating_only" and e.clinician_reviewed]
    if attempt.clinician_final_state != "UNKNOWN_UNTESTED" and not non_self_reviewed:
        issues.append(ContractIssue("non_unknown_state_requires_reviewed_evidence", "clinician_final_state"))
    if attempt.clinician_final_state == "FORMAL_SOLID":
        formal = any(e.clinician_reviewed and e.result == "demonstrated" and e.method in {"unaided_explanation", "mechanistic_explanation"} for e in attempt.evidence)
        transfer = any(e.clinician_reviewed and e.result == "demonstrated" and e.method in {"novel_case_transfer", "boundary_or_exception_recognition", "evidence_directness_calibration"} for e in attempt.evidence)
        if not formal:
            issues.append(ContractIssue("formal_solid_requires_formal_evidence", "evidence"))
        if not transfer:
            issues.append(ContractIssue("formal_solid_requires_transfer_or_boundary_evidence", "evidence"))
    if issues:
        raise LearningContractError(issues)
    return attempt
