from __future__ import annotations

import json
from pathlib import Path

from clinical_excellence.core.transcript_contracts import ProviderTranscriptExtractionV1, TranscriptExtractRequestV1
from clinical_excellence.core.transcript_service import extract_candidates
from clinical_excellence.modules.osteoporosis.transcript_profile import provider_profile
from evals.transcript_v1.run_provider_eval import _evaluate_case


CASES = Path("evals/transcript_v1/cases.json")


def _cases():
    items = json.loads(CASES.read_text(encoding="utf-8"))
    return items, {item["id"]: item for item in items}


def _request(transcript: str) -> TranscriptExtractRequestV1:
    return TranscriptExtractRequestV1.model_validate(
        {
            "schema_version": "clinical_transcript_extract_request_v1",
            "source_type": "heidi_transcript",
            "module": "osteoporosis",
            "encounter_phase": "during_visit",
            "language": "el",
            "transcript": transcript,
            "context": {"encounter_archetype": None},
        }
    )


class StaticProvider:
    def __init__(self, payload):
        self.payload = ProviderTranscriptExtractionV1.model_validate(payload)

    def extract(self, request, provider_profile):
        return self.payload


def test_h10_speaker_ambiguity_fixture_matches_h09_fail_closed_mapping():
    items, by_id = _cases()
    assert len(items) == 22
    rule = by_id["speaker_ambiguity"]["required_assertions"][0]
    assert rule["semantic_type"] == "uncertain_needs_review"
    assert rule["concept_key"] == "treatment.agent"
    assert rule["mapping"] == {
        "status": "ambiguous",
        "reason_code": "SEMANTIC_TYPE_NOT_ALLOWED_FOR_TREATMENT_EPISODE",
        "target_path": "step4.treatment_episodes[].agent",
        "proposed_value": "denosumab",
    }


def test_h10_profile_freezes_unrelated_deferral_and_no_result_semantics():
    profile = provider_profile()
    assert 'generic unrelated clinician deferral such as "we will examine it separately" is NOT a clinician_recommendation' in profile
    assert "NOT an osteoporosis followup_task" in profile
    assert "no result exists yet" in profile
    assert "clinician_interpretation + clinical.unmapped_narrative" in profile
    assert "never emit DXA objective-result concepts from absence-of-result wording" in profile


def test_h10_optional_free_text_allowances_are_source_bound():
    _, by_id = _cases()
    expected = {
        "garbled_speech": "το χάπι",
        "frax_original_adjusted": "Με κλινική προσαρμογή εκτιμώ τον κίνδυνο υψηλότερο",
        "embedded_instruction_untrusted": "Δεν αφορά το ιατρικό μου ιστορικό",
        "referral_not_completed_result": "Δεν υπάρχει ακόμη αποτέλεσμα DXA",
        "unrelated_general_clinical_text": "Θα το εξετάσουμε ξεχωριστά",
    }
    for case_id, needle in expected.items():
        rules = [
            rule
            for rule in by_id[case_id].get("allowed_assertions", [])
            if rule.get("concept_key") == "clinical.unmapped_narrative"
        ]
        assert rules
        assert any(rule.get("evidence_contains") == needle for rule in rules)


def test_h10_evidence_contains_rejects_blank_or_unrelated_optional_narrative():
    transcript = "Ιατρός: Παραπέμπω για DXA τον επόμενο μήνα. Δεν υπάρχει ακόμη αποτέλεσμα DXA."
    case = {
        "required_assertions": [
            {
                "semantic_type": "followup_task",
                "concept_key": "followup.task_type",
                "source_assertion": {"speaker": "clinician"},
                "value": {"kind": "code", "code": "DXA"},
            }
        ],
        "allowed_assertions": [
            {
                "semantic_type": "clinician_interpretation",
                "concept_key": "clinical.unmapped_narrative",
                "source_assertion": {"speaker": "clinician"},
                "value": {"kind": "text"},
                "evidence_contains": "Δεν υπάρχει ακόμη αποτέλεσμα DXA",
            }
        ],
    }

    def result_with(snippet: str):
        return extract_candidates(
            _request(transcript),
            StaticProvider(
                {
                    "candidates": [
                        {
                            "semantic_type": "followup_task",
                            "components": [{"concept_key": "followup.task_type", "value": {"kind": "code", "code": "DXA"}}],
                            "source_assertion": {"speaker": "clinician", "polarity": "positive", "temporality": "future", "certainty": "explicit"},
                            "evidence_snippet": "Παραπέμπω για DXA",
                            "confidence": "high",
                        },
                        {
                            "semantic_type": "clinician_interpretation",
                            "components": [{"concept_key": "clinical.unmapped_narrative", "value": {"kind": "text", "text": "Δεν υπάρχει ακόμη αποτέλεσμα DXA."}}],
                            "source_assertion": {"speaker": "clinician", "polarity": "positive", "temporality": "current", "certainty": "explicit"},
                            "evidence_snippet": snippet,
                            "confidence": "high",
                        },
                    ],
                    "warnings": [],
                }
            ),
        )

    blank_failures = _evaluate_case(case, result_with(""))
    assert "unexpected_assertion_clinical.unmapped_narrative" in blank_failures

    unrelated_failures = _evaluate_case(case, result_with("Παραπέμπω για DXA"))
    assert "unexpected_assertion_clinical.unmapped_narrative" in unrelated_failures

    matching_failures = _evaluate_case(case, result_with("Δεν υπάρχει ακόμη αποτέλεσμα DXA"))
    assert "unexpected_assertion_clinical.unmapped_narrative" not in matching_failures


def test_h10_exact_duplicate_authorized_candidate_fails_promotion():
    transcript = "Ασθενής: Δεν καπνίζω."
    candidate = {
        "semantic_type": "patient_history_fact",
        "components": [{"concept_key": "risk.current_smoking", "value": {"kind": "boolean", "value": False}}],
        "source_assertion": {"speaker": "patient", "polarity": "negative", "temporality": "current", "certainty": "explicit"},
        "evidence_snippet": "Δεν καπνίζω",
        "confidence": "high",
    }
    result = extract_candidates(
        _request(transcript),
        StaticProvider({"candidates": [candidate, candidate], "warnings": []}),
    )
    case = {
        "required_assertions": [
            {
                "semantic_type": "patient_history_fact",
                "concept_key": "risk.current_smoking",
                "source_assertion": {"speaker": "patient", "polarity": "negative"},
                "value": {"kind": "boolean", "value": False},
            }
        ]
    }
    assert "duplicate_candidate" in _evaluate_case(case, result)
