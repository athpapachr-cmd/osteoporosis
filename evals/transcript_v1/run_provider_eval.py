from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from clinical_excellence.core.providers.openai_transcript import OpenAITranscriptProvider, provider_status
from clinical_excellence.core.transcript_contracts import TranscriptExtractRequestV1
from clinical_excellence.core.transcript_service import extract_candidates

CASES = Path(__file__).with_name("cases.json")


def _norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().casefold()


def _candidate_signature(candidate) -> str:
    return json.dumps(
        {
            "semantic_type": candidate.semantic_type,
            "components": [component.model_dump(mode="json") for component in candidate.components],
            "source_assertion": candidate.source_assertion.model_dump(mode="json"),
            "evidence_snippet": candidate.evidence_snippet,
            "confidence": candidate.confidence,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _dict_subset(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        if key not in actual:
            return False
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            if not isinstance(actual_value, dict) or not _dict_subset(actual_value, expected_value):
                return False
        elif actual_value != expected_value:
            return False
    return True


def _component_matches_rule(candidate, component, rule: dict[str, Any]) -> bool:
    semantic_type = rule.get("semantic_type")
    if semantic_type is not None and candidate.semantic_type != semantic_type:
        return False

    source_rule = rule.get("source_assertion")
    if source_rule is not None and not _dict_subset(candidate.source_assertion.model_dump(mode="json"), source_rule):
        return False

    concept_key = rule.get("concept_key")
    if concept_key is not None and component.concept_key != concept_key:
        return False

    value_rule = rule.get("value")
    if value_rule is not None and not _dict_subset(component.value.model_dump(mode="json"), value_rule):
        return False

    mapping_rule = rule.get("mapping")
    if mapping_rule is not None:
        mappings = [
            mapping
            for mapping in candidate.target_mappings
            if component.concept_key in mapping.component_keys
        ]
        if not mappings or not any(
            _dict_subset(mapping.model_dump(mode="json"), mapping_rule)
            for mapping in mappings
        ):
            return False

    evidence_contains = rule.get("evidence_contains")
    if evidence_contains is not None:
        snippet = _norm_text(candidate.evidence_snippet)
        if not snippet or _norm_text(str(evidence_contains)) not in snippet:
            return False

    return True


def _candidate_matches(candidate, rule: dict[str, Any]) -> bool:
    semantic_type = rule.get("semantic_type")
    if semantic_type is not None and candidate.semantic_type != semantic_type:
        return False

    source_rule = rule.get("source_assertion")
    if source_rule is not None and not _dict_subset(candidate.source_assertion.model_dump(mode="json"), source_rule):
        return False

    concept_key = rule.get("concept_key")
    if concept_key is None:
        return True

    components = [component for component in candidate.components if component.concept_key == concept_key]
    if len(components) != 1:
        return False
    return _component_matches_rule(candidate, components[0], rule)


def _group_rule_component(group: dict[str, Any], component_rule: dict[str, Any]) -> dict[str, Any]:
    merged = dict(component_rule)
    if "semantic_type" in group and "semantic_type" not in merged:
        merged["semantic_type"] = group["semantic_type"]
    if "source_assertion" in group and "source_assertion" not in merged:
        merged["source_assertion"] = group["source_assertion"]
    return merged


def _candidate_group_matches(candidate, group: dict[str, Any]) -> bool:
    candidate_rule = {
        key: group[key]
        for key in ("semantic_type", "source_assertion")
        if key in group
    }
    if not _candidate_matches(candidate, candidate_rule):
        return False
    components = group.get("components", [])
    if not components:
        return False
    return all(
        _candidate_matches(candidate, _group_rule_component(group, component_rule))
        for component_rule in components
    )


def _evaluate_case(item: dict[str, Any], result) -> list[str]:
    failures: list[str] = []

    if result.meta.authoritative_write or result.meta.raw_persisted or result.meta.candidates_persisted:
        failures.append("non_ephemeral_response_meta")

    semantics = {candidate.semantic_type for candidate in result.candidates}
    concepts = {
        component.concept_key
        for candidate in result.candidates
        for component in candidate.components
    }

    missing_semantics = set(item.get("expect_semantics", [])) - semantics
    if missing_semantics:
        failures.append("missing_expected_semantics")
    missing_concepts = set(item.get("expect_concepts", [])) - concepts
    if missing_concepts:
        failures.append("missing_expected_concepts")

    required_assertions = item.get("required_assertions", [])
    for index, rule in enumerate(required_assertions):
        if not any(_candidate_matches(candidate, rule) for candidate in result.candidates):
            failures.append(f"required_assertion_{index}_missing")

    required_groups = item.get("required_candidate_groups", [])
    for index, group in enumerate(required_groups):
        if not any(_candidate_group_matches(candidate, group) for candidate in result.candidates):
            failures.append(f"required_candidate_group_{index}_missing")

    for index, rule in enumerate(item.get("forbidden_assertions", [])):
        if any(_candidate_matches(candidate, rule) for candidate in result.candidates):
            failures.append(f"forbidden_assertion_{index}_present")

    for concept_key in item.get("forbidden_concepts", []):
        if concept_key in concepts:
            failures.append(f"forbidden_concept_{concept_key}_present")

    # Promotion evidence is default-deny. Every returned component must be covered
    # by an explicit required/allowed/group rule with a concept key. This prevents
    # expected facts from masking clinically material hallucinated extras.
    authorization_rules = [rule for rule in required_assertions if rule.get("concept_key")]
    authorization_rules.extend(item.get("allowed_assertions", []))
    for group in required_groups:
        authorization_rules.extend(
            _group_rule_component(group, component_rule)
            for component_rule in group.get("components", [])
            if component_rule.get("concept_key")
        )

    for candidate in result.candidates:
        for component in candidate.components:
            if not any(
                rule.get("concept_key") and _component_matches_rule(candidate, component, rule)
                for rule in authorization_rules
            ):
                failures.append(f"unexpected_assertion_{component.concept_key}")

    seen_signatures: set[str] = set()
    for candidate in result.candidates:
        signature = _candidate_signature(candidate)
        if signature in seen_signatures:
            failures.append("duplicate_candidate")
            break
        seen_signatures.add(signature)

    exact_date_forbidden = set(item.get("forbid_exact_date_concepts", []))
    for candidate in result.candidates:
        for component in candidate.components:
            if component.concept_key not in exact_date_forbidden:
                continue
            value = component.value.model_dump(mode="json")
            if value.get("kind") == "date" and value.get("normalized") is not None:
                failures.append(f"invented_exact_date_{component.concept_key}")

    for semantic_type, expected_count in item.get("semantic_counts", {}).items():
        actual_count = sum(1 for candidate in result.candidates if candidate.semantic_type == semantic_type)
        if actual_count != expected_count:
            failures.append(f"semantic_count_{semantic_type}_mismatch")

    if "EVIDENCE_NOT_VERIFIABLE" in result.warnings:
        failures.append("evidence_not_verifiable")

    if not item.get("allow_low_confidence", False):
        if "LOW_SOURCE_CLARITY" in result.warnings or any(candidate.confidence == "low" for candidate in result.candidates):
            failures.append("low_confidence_output")

    for warning in item.get("forbidden_warnings", []):
        if warning in result.warnings:
            failures.append(f"forbidden_warning_{warning}")

    return list(dict.fromkeys(failures))


def main() -> int:
    status = provider_status("synthetic_eval")
    if not status["configured"]:
        print("provider-eval BLOCKED: synthetic transcript provider gate not configured")
        return 2

    cases = json.loads(CASES.read_text(encoding="utf-8"))
    provider = OpenAITranscriptProvider(purpose="synthetic_eval")
    failed = 0
    for item in cases:
        request = TranscriptExtractRequestV1.model_validate({
            "schema_version": "clinical_transcript_extract_request_v1",
            "source_type": "heidi_transcript",
            "module": "osteoporosis",
            "encounter_phase": "during_visit",
            "language": item["language"],
            "transcript": item["transcript"],
            "context": {"encounter_archetype": None},
        })
        result = extract_candidates(request, provider)
        failures = _evaluate_case(item, result)
        ok = not failures
        if ok:
            print(f"{item['id']}: PASS candidates={len(result.candidates)}")
        else:
            print(f"{item['id']}: FAIL checks={','.join(failures)} candidates={len(result.candidates)}")
        failed += 0 if ok else 1

    print(f"provider-eval summary: total={len(cases)} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
