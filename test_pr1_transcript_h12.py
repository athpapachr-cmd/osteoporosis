from __future__ import annotations

import json
from pathlib import Path

from clinical_excellence.core.transcript_contracts import ProviderTranscriptExtractionV1
from clinical_excellence.core.transcript_service import extract_candidates
from clinical_excellence.modules.osteoporosis.transcript_target_guard import map_candidate
from evals.transcript_v1.run_provider_eval import _component_matches_rule, _evaluate_case
from test_pr1_transcript_h09 import _candidate, _multi_candidate
from test_pr1_transcript_h10 import StaticProvider, _request


CASES = Path("evals/transcript_v1/cases.json")


def _mappings(components: list[dict], *, semantic_type: str, temporality: str):
    candidate = _multi_candidate(components, semantic_type=semantic_type, temporality=temporality)
    return {mapping.component_keys[0]: mapping for mapping in map_candidate(candidate)}


def test_h12_future_done_administration_fails_closed_as_a_whole_event():
    mappings = _mappings(
        [
            {"concept_key": "administration.agent", "value": {"kind": "code", "code": "denosumab"}},
            {"concept_key": "administration.status", "value": {"kind": "code", "code": "done"}},
            {"concept_key": "administration.scheduled_date", "value": {"kind": "date", "normalized": "2026-10-20", "precision": "day"}},
        ],
        semantic_type="followup_task",
        temporality="future",
    )
    assert {item.status for item in mappings.values()} == {"ambiguous"}
    assert {item.reason_code for item in mappings.values()} == {"TEMPORAL_ADMINISTRATION_CONFLICT"}


def test_h12_future_actual_date_fails_closed_as_a_whole_event():
    mappings = _mappings(
        [
            {"concept_key": "administration.agent", "value": {"kind": "code", "code": "denosumab"}},
            {"concept_key": "administration.actual_date", "value": {"kind": "date", "normalized": "2026-09-01", "precision": "day"}},
        ],
        semantic_type="followup_task",
        temporality="planned",
    )
    assert {item.status for item in mappings.values()} == {"ambiguous"}
    assert {item.reason_code for item in mappings.values()} == {"TEMPORAL_ADMINISTRATION_CONFLICT"}


def test_h12_future_active_treatment_cannot_become_exposure():
    mappings = _mappings(
        [
            {"concept_key": "treatment.agent", "value": {"kind": "code", "code": "denosumab"}},
            {"concept_key": "treatment.status", "value": {"kind": "code", "code": "active"}},
        ],
        semantic_type="patient_history_fact",
        temporality="future",
    )
    assert {item.status for item in mappings.values()} == {"ambiguous"}
    assert {item.reason_code for item in mappings.values()} == {"TEMPORAL_TREATMENT_EPISODE_CONFLICT"}


def test_h12_valid_planned_followup_and_historical_actual_events_remain_mapped():
    planned = _mappings(
        [
            {"concept_key": "administration.agent", "value": {"kind": "code", "code": "denosumab"}},
            {"concept_key": "administration.scheduled_date", "value": {"kind": "date", "normalized": "2026-10-20", "precision": "day"}},
            {"concept_key": "administration.status", "value": {"kind": "code", "code": "planned"}},
        ],
        semantic_type="followup_task",
        temporality="future",
    )
    assert {item.status for item in planned.values()} == {"mapped"}
    assert planned["administration.status"].proposed_value == "planned"

    actual = _mappings(
        [
            {"concept_key": "administration.agent", "value": {"kind": "code", "code": "denosumab"}},
            {"concept_key": "administration.actual_date", "value": {"kind": "date", "normalized": "2026-09-01", "precision": "day"}},
            {"concept_key": "administration.status", "value": {"kind": "code", "code": "done"}},
        ],
        semantic_type="objective_result",
        temporality="past",
    )
    assert {item.status for item in actual.values()} == {"mapped"}

    treatment = _mappings(
        [
            {"concept_key": "treatment.agent", "value": {"kind": "code", "code": "alendronate"}},
            {"concept_key": "treatment.status", "value": {"kind": "code", "code": "active"}},
        ],
        semantic_type="patient_history_fact",
        temporality="current",
    )
    assert {item.status for item in treatment.values()} == {"mapped"}


def test_h12_h09_negation_and_recommendation_guards_remain_authoritative():
    negated = map_candidate(_candidate("administration.status", {"kind": "code", "code": "done"}, polarity="negative", temporality="future"))[0]
    assert negated.status == "ambiguous"
    assert negated.reason_code == "NEGATED_ASSERTION_NOT_POSITIVE_RUNTIME_VALUE"

    recommendation = map_candidate(_candidate("treatment.status", {"kind": "code", "code": "active"}, semantic_type="clinician_recommendation", temporality="future"))[0]
    assert recommendation.status == "ambiguous"
    assert recommendation.reason_code == "SEMANTIC_TYPE_NOT_ALLOWED_FOR_TREATMENT_EPISODE"


def test_h12_authentic_evidence_plus_invented_narrative_fails_promotion():
    case = next(item for item in json.loads(CASES.read_text(encoding="utf-8")) if item["id"] == "referral_not_completed_result")
    required = {
        "semantic_type": "followup_task",
        "components": [{"concept_key": "followup.task_type", "value": {"kind": "code", "code": "DXA"}}],
        "source_assertion": {"speaker": "clinician", "polarity": "positive", "temporality": "future", "certainty": "explicit"},
        "evidence_snippet": "Παραπέμπω για DXA",
        "confidence": "high",
    }

    def result_with(value_text: str):
        narrative = {
            "semantic_type": "clinician_interpretation",
            "components": [{"concept_key": "clinical.unmapped_narrative", "value": {"kind": "text", "text": value_text}}],
            "source_assertion": {"speaker": "clinician", "polarity": "positive", "temporality": "current", "certainty": "explicit"},
            "evidence_snippet": "Δεν υπάρχει ακόμη αποτέλεσμα DXA",
            "confidence": "high",
        }
        return extract_candidates(_request(case["transcript"]), StaticProvider({"candidates": [required, narrative], "warnings": []}))

    invented = _evaluate_case(case, result_with("Χορηγήθηκε denosumab σήμερα."))
    assert "unexpected_assertion_clinical.unmapped_narrative" in invented
    assert "narrative_value_anchor_mismatch" in invented
    assert "narrative_value_source_span_mismatch" in invented

    appended_invention = _evaluate_case(case, result_with("Δεν υπάρχει ακόμη αποτέλεσμα DXA. Χορηγήθηκε denosumab σήμερα."))
    assert "unexpected_assertion_clinical.unmapped_narrative" in appended_invention
    assert "narrative_value_source_span_mismatch" in appended_invention

    supported = _evaluate_case(case, result_with("Δεν υπάρχει ακόμη αποτέλεσμα DXA."))
    assert supported == []


def test_h12_all_narrative_authorizations_bind_value_and_source():
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    assert len(cases) == 22
    for case in cases:
        for category in ("required_assertions", "allowed_assertions"):
            for rule in case.get(category, []):
                if rule.get("concept_key") == "clinical.unmapped_narrative":
                    assert rule.get("evidence_contains")
                    assert rule.get("value_text_contains")

    for case_id in ("referral_not_completed_result", "prescription_not_administration"):
        case = next(item for item in cases if item["id"] == case_id)
        rule = next(item for item in case["allowed_assertions"] if item.get("concept_key") == "clinical.unmapped_narrative")
        assert rule.get("value_text_source_span") is True


def test_h12_frax_rule_accepts_source_supported_risk_phrase_but_rejects_unrelated_value():
    case = next(item for item in json.loads(CASES.read_text(encoding="utf-8")) if item["id"] == "frax_original_adjusted")
    rule = next(item for item in case["allowed_assertions"] if item.get("concept_key") == "clinical.unmapped_narrative")

    def matches(value_text: str) -> bool:
        candidate = ProviderTranscriptExtractionV1.model_validate({"candidates": [{
            "semantic_type": "clinician_interpretation",
            "components": [{"concept_key": "clinical.unmapped_narrative", "value": {"kind": "text", "text": value_text}}],
            "source_assertion": {"speaker": "clinician", "polarity": "positive", "temporality": "current", "certainty": "explicit"},
            "evidence_snippet": "Με κλινική προσαρμογή εκτιμώ τον κίνδυνο υψηλότερο",
            "confidence": "high",
        }]}).candidates[0]
        return _component_matches_rule(candidate, candidate.components[0], rule, case["transcript"])

    assert matches("εκτιμώ τον κίνδυνο υψηλότερο")
    assert matches("Η κλινική εκτίμηση: κίνδυνος αυξημένος")
    assert not matches("Χορηγήθηκε denosumab σήμερα")
    assert not matches("εκτιμώ τον κίνδυνο υψηλότερο. Χορηγήθηκε denosumab σήμερα")


def test_h12_shoulder_paraphrase_cannot_carry_unrelated_administration_claim():
    case = next(item for item in json.loads(CASES.read_text(encoding="utf-8")) if item["id"] == "unrelated_general_clinical_text")
    rule = next(item for item in case["allowed_assertions"] if item.get("concept_key") == "clinical.unmapped_narrative")

    def matches(value_text: str) -> bool:
        payload = {"candidates": [{
            "semantic_type": "clinician_interpretation",
            "components": [{"concept_key": "clinical.unmapped_narrative", "value": {"kind": "text", "text": value_text}}],
            "source_assertion": {"speaker": "clinician", "polarity": "positive", "temporality": "future", "certainty": "explicit"},
            "evidence_snippet": "Θα το εξετάσουμε ξεχωριστά",
            "confidence": "high",
        }]}
        candidate = extract_candidates(_request(case["transcript"]), StaticProvider(payload)).candidates[0]
        return _component_matches_rule(candidate, candidate.components[0], rule, case["transcript"])

    assert matches("Θα το αξιολογήσουμε ξεχωριστά")
    assert not matches("Θα το αξιολογήσουμε ξεχωριστά και χορηγήθηκε denosumab")
