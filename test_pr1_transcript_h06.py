from __future__ import annotations

import pytest
from pydantic import ValidationError

from clinical_excellence.core.providers.openai_transcript import OpenAITranscriptProvider
from clinical_excellence.core.transcript_contracts import (
    BooleanValueV1,
    CodeValueV1,
    DateValueV1,
    IntegerValueV1,
    NumberValueV1,
    ProviderCandidateV1,
    ProviderTranscriptExtractionV1,
    QuantityValueV1,
    TextValueV1,
    TranscriptExtractRequestV1,
)
from clinical_excellence.core.transcript_provider import ProviderInvalidOutput


def _request() -> TranscriptExtractRequestV1:
    return TranscriptExtractRequestV1.model_validate({
        "schema_version": "clinical_transcript_extract_request_v1",
        "source_type": "heidi_transcript",
        "module": "osteoporosis",
        "encounter_phase": "during_visit",
        "language": "el",
        "transcript": "SYNTHETIC ONLY",
        "context": {"encounter_archetype": None},
    })


def _enable_provider(monkeypatch) -> None:
    monkeypatch.setenv("CLINICAL_TRANSCRIPT_AI_ENABLED", "true")
    monkeypatch.setenv("CLINICAL_TRANSCRIPT_PHI_PROVIDER_APPROVED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic")


def _contains_key(value, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def test_provider_response_schema_does_not_emit_unsupported_oneof():
    schema = ProviderTranscriptExtractionV1.model_json_schema()
    assert not _contains_key(schema, "oneOf")


@pytest.mark.parametrize(
    ("payload", "expected_type"),
    [
        ({"kind": "text", "text": "history"}, TextValueV1),
        ({"kind": "code", "code": "hip"}, CodeValueV1),
        ({"kind": "number", "value": -2.8}, NumberValueV1),
        ({"kind": "integer", "value": 2}, IntegerValueV1),
        ({"kind": "boolean", "value": True}, BooleanValueV1),
        ({"kind": "quantity", "value": 24.0, "unit": "ng/mL"}, QuantityValueV1),
        ({"kind": "date", "normalized": "2026-09-10", "precision": "day", "date_text": "10/09/2026"}, DateValueV1),
    ],
)
def test_plain_value_union_preserves_exact_typed_kind_validation(payload, expected_type):
    candidate = ProviderCandidateV1.model_validate({
        "semantic_type": "patient_history_fact",
        "components": [{"concept_key": "clinical.unmapped_narrative", "value": payload}],
        "source_assertion": {
            "speaker": "patient",
            "polarity": "positive",
            "temporality": "current",
            "certainty": "explicit",
        },
        "evidence_snippet": "",
        "confidence": "high",
    })
    assert isinstance(candidate.components[0].value, expected_type)


def test_plain_value_union_still_rejects_kind_shape_mismatch_and_impossible_date():
    base = {
        "semantic_type": "patient_history_fact",
        "source_assertion": {
            "speaker": "patient",
            "polarity": "positive",
            "temporality": "current",
            "certainty": "explicit",
        },
        "evidence_snippet": "",
        "confidence": "high",
    }
    with pytest.raises(ValidationError):
        ProviderCandidateV1.model_validate({
            **base,
            "components": [{"concept_key": "clinical.unmapped_narrative", "value": {"kind": "code", "text": "not-a-code"}}],
        })
    with pytest.raises(ValidationError):
        ProviderCandidateV1.model_validate({
            **base,
            "components": [{"concept_key": "followup.due_date", "value": {"kind": "date", "normalized": "2026-02-31", "precision": "day", "date_text": "31/02/2026"}}],
        })


def test_h06_preserves_duplicate_concept_rejection():
    with pytest.raises(ValidationError):
        ProviderCandidateV1.model_validate({
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
        })


def test_http_400_invalid_schema_is_deterministic_invalid_output_not_unavailable(monkeypatch):
    _enable_provider(monkeypatch)

    BadRequestError = type("BadRequestError", (Exception,), {})
    error = BadRequestError("synthetic invalid schema")
    error.status_code = 400
    error.body = {"code": "invalid_json_schema", "param": "text.format.schema"}

    class BadResponses:
        def parse(self, **kwargs):
            raise error

    client = type("Client", (), {"responses": BadResponses()})()
    with pytest.raises(ProviderInvalidOutput):
        OpenAITranscriptProvider(client=client).extract(_request(), "PROFILE")
