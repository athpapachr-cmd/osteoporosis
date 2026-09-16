import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from clinical_excellence.core.providers.openai_transcript import OpenAITranscriptProvider
from clinical_excellence.core.transcript_contracts import (
    DateValueV1,
    ProviderCandidateV1,
    ProviderTranscriptExtractionV1,
    TranscriptExtractRequestV1,
)
from clinical_excellence.core.transcript_provider import ProviderInvalidOutput, ProviderRefusal, ProviderUnavailable
from clinical_excellence.core.transcript_service import extract_candidates
from clinical_excellence.modules.osteoporosis.transcript_target_guard import map_candidate
from evals.transcript_v1.run_provider_eval import _evaluate_case


def test_eval_fixture_is_synthetic_and_covers_required_cases():
    cases = json.loads(Path("evals/transcript_v1/cases.json").read_text(encoding="utf-8"))
    assert len(cases) >= 13
    ids = {item["id"] for item in cases}
    assert {
        "fracture_relative_time",
        "explicit_negative_smoking",
        "dxa_objective",
        "labs_objective",
        "options_one_final",
        "preference_only",
        "followup_vague",
        "garbled_speech",
        "frax_original_adjusted",
        "speaker_ambiguity",
        "negative_history_vs_negative_investigation",
        "followup_exact",
        "unrelated_general_clinical_text",
    }.issubset(ids)
    assert all(item.get("required_assertions") for item in cases)
    assert any(item.get("forbidden_assertions") or item.get("forbidden_concepts") for item in cases)
    joined = json.dumps(cases, ensure_ascii=False).lower()
    for forbidden in ("gesy id", "@gmail.com", "+357 9"):
        assert forbidden not in joined


def test_provider_eval_runner_is_fail_closed_and_does_not_print_transcript_content():
    runner = Path("evals/transcript_v1/run_provider_eval.py").read_text(encoding="utf-8")
    assert "provider-eval BLOCKED" in runner
    assert 'status["phi_provider_approved"]' in runner
    assert "item['transcript']" not in runner
    assert "result.candidates" in runner
    assert "required_assertions" in runner
    assert "forbidden_assertions" in runner
    assert "non_ephemeral_response_meta" in runner


def _candidate(concept_key, value):
    return ProviderCandidateV1.model_validate({
        "semantic_type": "patient_history_fact",
        "components": [{"concept_key": concept_key, "value": value}],
        "source_assertion": {
            "speaker": "clinician",
            "polarity": "positive",
            "temporality": "current",
            "certainty": "explicit",
        },
        "evidence_snippet": "",
        "confidence": "high",
    })


def _request():
    return TranscriptExtractRequestV1.model_validate({
        "schema_version": "clinical_transcript_extract_request_v1",
        "source_type": "heidi_transcript",
        "module": "osteoporosis",
        "encounter_phase": "during_visit",
        "language": "el",
        "transcript": "SYNTHETIC ONLY",
        "context": {"encounter_archetype": None},
    })


def _enable_provider(monkeypatch):
    monkeypatch.setenv("CLINICAL_TRANSCRIPT_AI_ENABLED", "true")
    monkeypatch.setenv("CLINICAL_TRANSCRIPT_PHI_PROVIDER_APPROVED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic")


def test_step4_status_mapping_rejects_provider_codes_outside_runtime_enums():
    valid_episode = map_candidate(_candidate("treatment.status", {"kind": "code", "code": "active"}))[0]
    assert valid_episode.status == "mapped"
    assert valid_episode.proposed_value == "active"

    invalid_episode = map_candidate(_candidate("treatment.status", {"kind": "code", "code": "paused_forever"}))[0]
    assert invalid_episode.status == "ambiguous"
    assert invalid_episode.reason_code == "UNSUPPORTED_TREATMENT_STATUS"

    valid_admin = map_candidate(_candidate("administration.status", {"kind": "code", "code": "overdue"}))[0]
    assert valid_admin.status == "mapped"
    assert valid_admin.proposed_value == "overdue"

    invalid_admin = map_candidate(_candidate("administration.status", {"kind": "text", "text": "administered maybe"}))[0]
    assert invalid_admin.status == "ambiguous"
    assert invalid_admin.reason_code == "UNSUPPORTED_ADMINISTRATION_STATUS"


def test_treatment_duration_requires_numeric_runtime_range():
    valid = map_candidate(_candidate("treatment.duration_years", {"kind": "number", "value": 4.5}))[0]
    assert valid.status == "mapped"
    assert valid.proposed_value == 4.5

    wrong_type = map_candidate(_candidate("treatment.duration_years", {"kind": "code", "code": "four years"}))[0]
    assert wrong_type.status == "ambiguous"
    assert wrong_type.reason_code == "TYPE_MISMATCH"

    out_of_range = map_candidate(_candidate("treatment.duration_years", {"kind": "number", "value": 51}))[0]
    assert out_of_range.status == "ambiguous"
    assert out_of_range.reason_code == "OUT_OF_RUNTIME_RANGE"


def test_fixed_runtime_code_targets_are_whitelisted_locally():
    assert map_candidate(_candidate("fracture.site", {"kind": "code", "code": "hip"}))[0].status == "mapped"
    bad_site = map_candidate(_candidate("fracture.site", {"kind": "code", "code": "ankle"}))[0]
    assert bad_site.status == "ambiguous"
    assert bad_site.reason_code == "UNSUPPORTED_FRACTURE_SITE"

    assert map_candidate(_candidate("frax.tool_name", {"kind": "code", "code": "fraxplus"}))[0].status == "mapped"
    bad_tool = map_candidate(_candidate("frax.tool_name", {"kind": "code", "code": "magic_score"}))[0]
    assert bad_tool.status == "ambiguous"
    assert bad_tool.reason_code == "UNSUPPORTED_FRAX_TOOL"

    assert map_candidate(_candidate("risk.resulting_category", {"kind": "code", "code": "very_high"}))[0].status == "mapped"
    bad_risk = map_candidate(_candidate("risk.resulting_category", {"kind": "code", "code": "extreme"}))[0]
    assert bad_risk.status == "ambiguous"
    assert bad_risk.reason_code == "UNSUPPORTED_RISK_CATEGORY"


def test_vfa_runtime_enums_are_validated_and_boolean_indication_is_normalized():
    indicated = map_candidate(_candidate("vfa.indicated", {"kind": "boolean", "value": True}))[0]
    assert indicated.status == "mapped"
    assert indicated.proposed_value == "yes"

    uncertain = map_candidate(_candidate("vfa.indicated", {"kind": "code", "code": "uncertain"}))[0]
    assert uncertain.status == "mapped"
    assert uncertain.proposed_value == "uncertain"

    bad_action = map_candidate(_candidate("vfa.action", {"kind": "code", "code": "probably_later"}))[0]
    assert bad_action.status == "ambiguous"
    assert bad_action.reason_code == "UNSUPPORTED_VFA_ACTION"

    modality = map_candidate(_candidate("vfa.modality", {"kind": "code", "code": "MRI"}))[0]
    assert modality.status == "mapped"

    bad_modality = map_candidate(_candidate("vfa.modality", {"kind": "code", "code": "ultrasound"}))[0]
    assert bad_modality.status == "ambiguous"
    assert bad_modality.reason_code == "UNSUPPORTED_VFA_MODALITY"


def test_exact_date_contract_rejects_impossible_calendar_values():
    for normalized, precision in (
        ("2026-02-31", "day"),
        ("2026-13", "month"),
        ("0000", "year"),
    ):
        with pytest.raises(ValidationError):
            DateValueV1.model_validate({
                "kind": "date",
                "normalized": normalized,
                "precision": precision,
                "date_text": normalized,
            })

    leap_day = DateValueV1.model_validate({
        "kind": "date",
        "normalized": "2024-02-29",
        "precision": "day",
        "date_text": "29/02/2024",
    })
    assert leap_day.normalized == "2024-02-29"


def test_provider_eval_gate_rejects_dangerous_extra_assertion_even_when_legacy_sets_match():
    provider_output = ProviderTranscriptExtractionV1.model_validate({
        "candidates": [
            {
                "semantic_type": "followup_task",
                "components": [
                    {"concept_key": "followup.timeframe_text", "value": {"kind": "text", "text": "σε περίπου έξι μήνες"}},
                    {"concept_key": "followup.due_date", "value": {"kind": "date", "normalized": "2027-03-16", "precision": "day", "date_text": "σε περίπου έξι μήνες"}},
                ],
                "source_assertion": {
                    "speaker": "clinician",
                    "polarity": "positive",
                    "temporality": "future",
                    "certainty": "explicit",
                },
                "evidence_snippet": "",
                "confidence": "high",
            }
        ]
    })

    class StaticProvider:
        def extract(self, request, provider_profile):
            return provider_output

    result = extract_candidates(_request(), StaticProvider())
    case = {
        "expect_semantics": ["followup_task"],
        "expect_concepts": ["followup.timeframe_text"],
        "required_assertions": [
            {"semantic_type": "followup_task", "concept_key": "followup.timeframe_text"}
        ],
        "forbidden_concepts": ["followup.due_date"],
    }
    failures = _evaluate_case(case, result)
    assert "forbidden_concept_followup.due_date_present" in failures


def test_openai_adapter_classifies_structured_validation_as_invalid_output(monkeypatch):
    _enable_provider(monkeypatch)

    class InvalidResponses:
        def parse(self, **kwargs):
            return ProviderTranscriptExtractionV1.model_validate({
                "candidates": [{"semantic_type": "not-a-real-semantic-type"}],
                "warnings": [],
            })

    client = type("Client", (), {"responses": InvalidResponses()})()
    with pytest.raises(ProviderInvalidOutput):
        OpenAITranscriptProvider(client=client).extract(_request(), "PROFILE")


def test_openai_adapter_preserves_refusal_and_unavailable_failure_classes(monkeypatch):
    _enable_provider(monkeypatch)

    refusal_content = type("Content", (), {"type": "refusal"})()
    refusal_item = type("Item", (), {"output": None, "content": [refusal_content]})()
    refusal_response = type("Response", (), {"output": [refusal_item], "output_parsed": None})()

    class RefusalResponses:
        def parse(self, **kwargs):
            return refusal_response

    refusal_client = type("Client", (), {"responses": RefusalResponses()})()
    with pytest.raises(ProviderRefusal):
        OpenAITranscriptProvider(client=refusal_client).extract(_request(), "PROFILE")

    APITimeoutError = type("APITimeoutError", (Exception,), {})

    class TimeoutResponses:
        def parse(self, **kwargs):
            raise APITimeoutError("synthetic timeout")

    timeout_client = type("Client", (), {"responses": TimeoutResponses()})()
    with pytest.raises(ProviderUnavailable):
        OpenAITranscriptProvider(client=timeout_client).extract(_request(), "PROFILE")
