import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from clinical_excellence.core.providers.openai_transcript import (
    OpenAITranscriptProvider,
    provider_status,
)
from clinical_excellence.core.transcript_contracts import (
    DateValueV1,
    ProviderCandidateV1,
    ProviderTranscriptExtractionV1,
    TranscriptExtractRequestV1,
)
from clinical_excellence.core.transcript_provider import (
    ProviderInvalidOutput,
    ProviderNotConfigured,
    ProviderRefusal,
    ProviderUnavailable,
)
from clinical_excellence.core.transcript_service import extract_candidates
from clinical_excellence.modules.osteoporosis.transcript_target_guard import map_candidate
from evals.transcript_v1.run_provider_eval import _evaluate_case


EXPANDED_CASE_IDS = {
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
    "repeated_fracture_event_grouping",
    "embedded_instruction_untrusted",
    "referral_not_completed_result",
    "prescription_not_administration",
    "self_correction_date",
    "third_party_treatment_history",
    "out_of_range_runtime_values",
    "planned_not_done_administration",
    "negated_treatment_exposure",
}


def test_eval_fixture_is_synthetic_and_covers_required_cases():
    cases = json.loads(Path("evals/transcript_v1/cases.json").read_text(encoding="utf-8"))
    assert len(cases) >= 22
    ids = {item["id"] for item in cases}
    assert EXPANDED_CASE_IDS.issubset(ids)
    assert all(item.get("required_assertions") or item.get("allowed_assertions") for item in cases)
    assert any(item.get("required_candidate_groups") for item in cases)
    assert any(item.get("forbidden_assertions") or item.get("forbidden_concepts") for item in cases)
    joined = json.dumps(cases, ensure_ascii=False).lower()
    for forbidden in ("gesy id", "@gmail.com", "+357 9"):
        assert forbidden not in joined


def test_provider_eval_runner_is_fail_closed_and_does_not_print_transcript_content():
    runner = Path("evals/transcript_v1/run_provider_eval.py").read_text(encoding="utf-8")
    assert "provider-eval BLOCKED" in runner
    assert 'provider_status("synthetic_eval")' in runner
    assert 'OpenAITranscriptProvider(purpose="synthetic_eval")' in runner
    assert "unexpected_assertion_" in runner
    assert "required_candidate_groups" in runner
    assert "low_confidence_output" in runner
    assert "item['transcript']" not in runner
    assert "result.candidates" in runner
    assert "required_assertions" in runner
    assert "allowed_assertions" in runner
    assert "forbidden_assertions" in runner
    assert "non_ephemeral_response_meta" in runner


def _candidate(
    concept_key,
    value,
    *,
    semantic_type="patient_history_fact",
    speaker="clinician",
    polarity="positive",
    temporality="current",
    confidence="high",
):
    return ProviderCandidateV1.model_validate({
        "semantic_type": semantic_type,
        "components": [{"concept_key": concept_key, "value": value}],
        "source_assertion": {
            "speaker": speaker,
            "polarity": polarity,
            "temporality": temporality,
            "certainty": "explicit",
        },
        "evidence_snippet": "",
        "confidence": confidence,
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


def test_provider_candidate_rejects_duplicate_concept_keys_but_allows_repeated_events_across_candidates():
    duplicate = {
        "semantic_type": "patient_history_fact",
        "components": [
            {"concept_key": "fracture.site", "value": {"kind": "code", "code": "hip"}},
            {"concept_key": "fracture.site", "value": {"kind": "code", "code": "other"}},
        ],
        "source_assertion": {
            "speaker": "patient",
            "polarity": "positive",
            "temporality": "past",
            "certainty": "explicit",
        },
        "evidence_snippet": "",
        "confidence": "high",
    }
    with pytest.raises(ValidationError):
        ProviderCandidateV1.model_validate(duplicate)

    first = _candidate("fracture.site", {"kind": "code", "code": "distal_radius"}, speaker="patient")
    second = _candidate("fracture.site", {"kind": "code", "code": "hip"}, speaker="patient")
    result = ProviderTranscriptExtractionV1.model_validate({
        "candidates": [first.model_dump(mode="json"), second.model_dump(mode="json")],
        "warnings": [],
    })
    assert len(result.candidates) == 2


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


@pytest.mark.parametrize(
    ("concept", "value", "semantic_type"),
    [
        ("anthropometrics.weight", {"kind": "quantity", "value": 301, "unit": "kg"}, "patient_history_fact"),
        ("anthropometrics.current_height", {"kind": "quantity", "value": 99, "unit": "cm"}, "patient_history_fact"),
        ("frax.mof_percent", {"kind": "number", "value": 101}, "objective_result"),
        ("frax.hip_percent", {"kind": "number", "value": -0.1}, "objective_result"),
        ("dxa.spine_bmd", {"kind": "quantity", "value": 3.1, "unit": "g/cm²"}, "objective_result"),
        ("dxa.total_hip_t_score", {"kind": "number", "value": -8.1}, "objective_result"),
        ("risk.falls_last_12_months", {"kind": "integer", "value": 51}, "patient_history_fact"),
        ("risk.cfs_score", {"kind": "integer", "value": 10}, "patient_history_fact"),
    ],
)
def test_runtime_numeric_targets_fail_closed_outside_exact_ranges(concept, value, semantic_type):
    mapping = map_candidate(_candidate(concept, value, semantic_type=semantic_type))[0]
    assert mapping.status == "ambiguous"
    assert mapping.reason_code == "OUT_OF_RUNTIME_RANGE"


def test_integer_runtime_targets_reject_fractional_provider_values():
    falls = map_candidate(_candidate("risk.falls_last_12_months", {"kind": "number", "value": 2.5}))[0]
    assert falls.status == "ambiguous"
    assert falls.reason_code == "TYPE_MISMATCH"

    cfs = map_candidate(_candidate("risk.cfs_score", {"kind": "number", "value": 4.5}))[0]
    assert cfs.status == "ambiguous"
    assert cfs.reason_code == "TYPE_MISMATCH"


def test_original_frax_percentages_require_objective_result_semantics():
    interpretation = map_candidate(
        _candidate("frax.mof_percent", {"kind": "number", "value": 18}, semantic_type="clinician_interpretation")
    )[0]
    assert interpretation.status == "ambiguous"
    assert interpretation.reason_code == "OBJECTIVE_RESULT_REQUIRED"

    objective = map_candidate(
        _candidate("frax.mof_percent", {"kind": "number", "value": 18}, semantic_type="objective_result")
    )[0]
    assert objective.status == "mapped"
    assert objective.proposed_value == 18


def test_third_party_and_negated_treatment_agents_fail_closed_before_patient_mapping():
    third_party = map_candidate(
        _candidate(
            "treatment.agent",
            {"kind": "code", "code": "denosumab"},
            speaker="third_party",
            polarity="positive",
        )
    )[0]
    assert third_party.status == "ambiguous"
    assert third_party.reason_code == "THIRD_PARTY_SOURCE_NOT_PATIENT"

    negated = map_candidate(
        _candidate(
            "treatment.agent",
            {"kind": "code", "code": "denosumab"},
            speaker="patient",
            polarity="negative",
        )
    )[0]
    assert negated.status == "ambiguous"
    assert negated.reason_code == "NEGATED_ASSERTION_NOT_POSITIVE_RUNTIME_VALUE"


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


def test_provider_eval_default_deny_rejects_unexpected_extra_assertion_without_fixture_foreknowledge():
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
    }
    failures = _evaluate_case(case, result)
    assert "unexpected_assertion_followup.due_date" in failures


def test_provider_eval_candidate_grouping_detects_cross_paired_repeated_events():
    provider_output = ProviderTranscriptExtractionV1.model_validate({
        "candidates": [
            {
                "semantic_type": "patient_history_fact",
                "components": [
                    {"concept_key": "fracture.site", "value": {"kind": "code", "code": "distal_radius"}},
                    {"concept_key": "fracture.date", "value": {"kind": "date", "normalized": "2025-03", "precision": "month", "date_text": "Μάρτιο 2025"}},
                ],
                "source_assertion": {"speaker": "patient", "polarity": "positive", "temporality": "past", "certainty": "explicit"},
                "evidence_snippet": "",
                "confidence": "high",
            },
            {
                "semantic_type": "patient_history_fact",
                "components": [
                    {"concept_key": "fracture.site", "value": {"kind": "code", "code": "hip"}},
                    {"concept_key": "fracture.date", "value": {"kind": "date", "normalized": "2022-01", "precision": "month", "date_text": "Ιανουάριο 2022"}},
                ],
                "source_assertion": {"speaker": "patient", "polarity": "positive", "temporality": "past", "certainty": "explicit"},
                "evidence_snippet": "",
                "confidence": "high",
            },
        ],
        "warnings": [],
    })

    class StaticProvider:
        def extract(self, request, provider_profile):
            return provider_output

    result = extract_candidates(_request(), StaticProvider())
    case = {
        "required_assertions": [
            {"semantic_type": "patient_history_fact", "concept_key": "fracture.site", "value": {"kind": "code", "code": "distal_radius"}},
            {"semantic_type": "patient_history_fact", "concept_key": "fracture.date", "value": {"kind": "date", "normalized": "2022-01"}},
            {"semantic_type": "patient_history_fact", "concept_key": "fracture.site", "value": {"kind": "code", "code": "hip"}},
            {"semantic_type": "patient_history_fact", "concept_key": "fracture.date", "value": {"kind": "date", "normalized": "2025-03"}},
        ],
        "required_candidate_groups": [
            {"semantic_type": "patient_history_fact", "components": [
                {"concept_key": "fracture.site", "value": {"kind": "code", "code": "distal_radius"}},
                {"concept_key": "fracture.date", "value": {"kind": "date", "normalized": "2022-01"}},
            ]},
            {"semantic_type": "patient_history_fact", "components": [
                {"concept_key": "fracture.site", "value": {"kind": "code", "code": "hip"}},
                {"concept_key": "fracture.date", "value": {"kind": "date", "normalized": "2025-03"}},
            ]},
        ],
    }
    failures = _evaluate_case(case, result)
    assert "required_candidate_group_0_missing" in failures
    assert "required_candidate_group_1_missing" in failures


def test_provider_eval_low_confidence_fails_clean_case_unless_explicitly_allowed():
    provider_output = ProviderTranscriptExtractionV1.model_validate({
        "candidates": [
            {
                "semantic_type": "patient_history_fact",
                "components": [{"concept_key": "clinical.unmapped_narrative", "value": {"kind": "text", "text": "synthetic"}}],
                "source_assertion": {"speaker": "patient", "polarity": "positive", "temporality": "current", "certainty": "explicit"},
                "evidence_snippet": "",
                "confidence": "low",
            }
        ],
        "warnings": [],
    })

    class StaticProvider:
        def extract(self, request, provider_profile):
            return provider_output

    result = extract_candidates(_request(), StaticProvider())
    case = {
        "required_assertions": [
            {"semantic_type": "patient_history_fact", "concept_key": "clinical.unmapped_narrative"}
        ]
    }
    assert "low_confidence_output" in _evaluate_case(case, result)
    assert "low_confidence_output" not in _evaluate_case({**case, "allow_low_confidence": True}, result)


def test_synthetic_eval_gate_is_separate_from_identifiable_phi_approval(monkeypatch):
    monkeypatch.setenv("CLINICAL_TRANSCRIPT_AI_ENABLED", "true")
    monkeypatch.setenv("CLINICAL_TRANSCRIPT_SYNTHETIC_EVAL_ENABLED", "true")
    monkeypatch.setenv("CLINICAL_TRANSCRIPT_PHI_PROVIDER_APPROVED", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic")

    assert provider_status()["configured"] is False
    assert provider_status("synthetic_eval")["configured"] is True

    parsed = ProviderTranscriptExtractionV1.model_validate({"candidates": [], "warnings": []})

    class SyntheticResponses:
        def parse(self, **kwargs):
            return type("Response", (), {"output_parsed": parsed, "output": []})()

    client = type("Client", (), {"responses": SyntheticResponses()})()
    with pytest.raises(ProviderNotConfigured):
        OpenAITranscriptProvider(client=client).extract(_request(), "PROFILE")

    result = OpenAITranscriptProvider(client=client, purpose="synthetic_eval").extract(_request(), "PROFILE")
    assert result == parsed


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
