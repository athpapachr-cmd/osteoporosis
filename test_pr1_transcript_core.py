from __future__ import annotations

import json
import os
import sys
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from clinical_auth import ClinicalCookieMiddleware, build_auth_router
from clinical_excellence.core.providers.openai_transcript import (
    DEFAULT_TIMEOUT_SECONDS,
    OpenAITranscriptProvider,
)
from clinical_excellence.core.transcript_contracts import (
    ProviderTranscriptExtractionV1,
    TranscriptExtractRequestV1,
)
from clinical_excellence.core.transcript_provider import (
    ProviderInvalidOutput,
    ProviderRefusal,
    ProviderUnavailable,
)
from clinical_excellence.core.transcript_router import MAX_BODY_BYTES, build_transcript_router
from clinical_excellence.core.transcript_service import extract_candidates
from clinical_excellence.modules.osteoporosis.transcript_targets import map_candidate


def req(transcript: str = "Ασθενής: Έπεσα πριν 2 μήνες.") -> TranscriptExtractRequestV1:
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


def provider_result(candidates=None, warnings=None) -> ProviderTranscriptExtractionV1:
    return ProviderTranscriptExtractionV1.model_validate(
        {"candidates": candidates or [], "warnings": warnings or []}
    )


class FakeProvider:
    def __init__(self, result=None, error: Exception | None = None):
        self.result = result or provider_result([])
        self.error = error
        self.calls = 0

    def extract(self, request, provider_profile):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


def payload(transcript: str = "Ασθενής: κάταγμα ισχίου.") -> dict:
    return {
        "schema_version": "clinical_transcript_extract_request_v1",
        "source_type": "heidi_transcript",
        "module": "osteoporosis",
        "encounter_phase": "during_visit",
        "language": "el",
        "transcript": transcript,
        "context": {"encounter_archetype": None},
    }


def make_client(result=None, error: Exception | None = None, *, cookie_auth: bool = False):
    fake = FakeProvider(result, error)
    app = FastAPI()
    if cookie_auth:
        app.add_middleware(ClinicalCookieMiddleware)
        app.include_router(build_auth_router())
    app.include_router(build_transcript_router(lambda: fake))
    os.environ["CLINICAL_DATA_KEY"] = "synth-key"
    return TestClient(app, base_url="https://testserver"), fake


def provider_candidate(*, semantic_type="patient_history_fact", concept="fracture.site", value=None, evidence=""):
    return {
        "semantic_type": semantic_type,
        "components": [
            {
                "concept_key": concept,
                "value": value or {"kind": "code", "code": "hip"},
            }
        ],
        "source_assertion": {
            "speaker": "patient",
            "polarity": "positive",
            "temporality": "past",
            "certainty": "explicit",
        },
        "evidence_snippet": evidence,
        "confidence": "high",
    }


def test_request_contract_rejects_unknown_fields_and_provider_target_path_injection():
    item = req().model_dump()
    item["surprise"] = "x"
    with pytest.raises(ValidationError):
        TranscriptExtractRequestV1.model_validate(item)

    candidate = provider_candidate()
    candidate["target_path"] = "step4.decision.selected_agent"
    with pytest.raises(ValidationError):
        provider_result([candidate])


def test_relative_timing_cannot_be_normalized_into_exact_date():
    candidate = provider_candidate(
        concept="fracture.date",
        value={"kind": "date", "normalized": None, "precision": "relative", "date_text": "πριν 2 μήνες"},
        evidence="πριν 2 μήνες",
    )
    candidate["source_assertion"].update(
        {
            "temporality": "relative",
            "normalized_date": "2026-07-16",
            "date_precision": "day",
            "date_text": "πριν 2 μήνες",
        }
    )
    with pytest.raises(ValidationError):
        provider_result([candidate])


def test_service_forces_review_and_flags_unverifiable_evidence_without_persistence():
    raw = provider_candidate(evidence="Αυτό δεν υπάρχει στο transcript")
    provider = FakeProvider(provider_result([raw]))
    result = extract_candidates(req("Ασθενής: κάταγμα ισχίου."), provider)
    assert provider.calls == 1
    candidate = result.candidates[0]
    assert candidate.requires_clinician_review is True
    assert candidate.status == "proposed"
    assert "EVIDENCE_NOT_VERIFIABLE" in candidate.warnings
    assert candidate.target_mappings[0].target_path == "fracture_history.events[].site"
    assert result.meta.raw_persisted is False
    assert result.meta.candidates_persisted is False
    assert result.meta.authoritative_write is False


def test_mapper_preserves_adjusted_frax_and_negative_default_semantics():
    adjusted = provider_result(
        [
            provider_candidate(
                semantic_type="objective_result",
                concept="frax.adjusted_mof_percent",
                value={"kind": "number", "value": 25.0},
            )
        ]
    ).candidates[0]
    mapped = map_candidate(adjusted)[0]
    assert mapped.status == "unmapped"
    assert mapped.reason_code == "ADJUSTED_RISK_MUST_NOT_OVERWRITE_ORIGINAL_FRAX"

    smoking = provider_result(
        [
            provider_candidate(
                concept="risk.current_smoking",
                value={"kind": "boolean", "value": False},
            )
        ]
    ).candidates[0]
    mapped = map_candidate(smoking)[0]
    assert mapped.status == "ambiguous"
    assert mapped.reason_code == "NEGATIVE_DEFAULT_SEMANTICS"


def test_mapper_units_and_final_decision_semantics():
    dxa = provider_result(
        [
            provider_candidate(
                semantic_type="objective_result",
                concept="dxa.spine_bmd",
                value={"kind": "quantity", "value": 0.812, "unit": "g/cm²"},
            )
        ]
    ).candidates[0]
    assert map_candidate(dxa)[0].status == "mapped"

    wrong_unit = provider_result(
        [
            provider_candidate(
                semantic_type="objective_result",
                concept="labs.ctx",
                value={"kind": "quantity", "value": 180, "unit": "pg/mL"},
            )
        ]
    ).candidates[0]
    assert map_candidate(wrong_unit)[0].status == "ambiguous"

    option = provider_result(
        [
            provider_candidate(
                semantic_type="option_discussed",
                concept="decision.selected_agent",
                value={"kind": "code", "code": "denosumab"},
            )
        ]
    ).candidates[0]
    assert map_candidate(option)[0].status == "ambiguous"
    assert map_candidate(option)[0].reason_code == "FINAL_DECISION_REQUIRED"

    final = provider_result(
        [
            provider_candidate(
                semantic_type="final_decision",
                concept="decision.selected_agent",
                value={"kind": "code", "code": "denosumab"},
            )
        ]
    ).candidates[0]
    assert map_candidate(final)[0].target_path == "step4.decision.selected_agent"


def test_endpoint_auth_sanitized_errors_and_exactly_one_provider_call(caplog):
    sentinel = "SECRET-PHI-SENTINEL-987"
    client, fake = make_client(provider_result([]))
    assert client.post("/clinical/transcript/extract", json=payload()).status_code == 401

    bad = client.post(
        "/clinical/transcript/extract",
        headers={"X-Clinical-Key": "synth-key"},
        json={**payload(sentinel), "unknown": sentinel},
    )
    assert bad.status_code == 422
    assert bad.json()["detail"]["code"] == "INVALID_REQUEST"
    assert sentinel not in bad.text
    assert sentinel not in caplog.text

    ok = client.post(
        "/clinical/transcript/extract",
        headers={"X-Clinical-Key": "synth-key"},
        json=payload(),
    )
    assert ok.status_code == 200
    assert fake.calls == 1
    assert ok.json()["meta"]["authoritative_write"] is False


def test_cookie_session_authenticates_transcript_endpoint_without_browser_key_header():
    client, fake = make_client(provider_result([]), cookie_auth=True)
    login = client.post("/clinical/login", json={"key": "synth-key"})
    assert login.status_code == 200
    response = client.post("/clinical/transcript/extract", json=payload())
    assert response.status_code == 200
    assert fake.calls == 1


def test_endpoint_body_character_and_module_limits_are_fail_closed():
    client, fake = make_client(provider_result([]))
    headers = {"X-Clinical-Key": "synth-key", "Content-Type": "application/json"}

    empty = client.post("/clinical/transcript/extract", headers=headers, json=payload("   "))
    assert empty.status_code == 422
    assert empty.json()["detail"]["code"] == "EMPTY_TRANSCRIPT"

    too_many_chars = client.post(
        "/clinical/transcript/extract",
        headers=headers,
        json=payload("x" * 120_001),
    )
    assert too_many_chars.status_code == 413
    assert too_many_chars.json()["detail"]["code"] == "TRANSCRIPT_TOO_LARGE"

    too_large_body = client.post(
        "/clinical/transcript/extract",
        headers=headers,
        content=b"x" * (MAX_BODY_BYTES + 1),
    )
    assert too_large_body.status_code == 413
    assert too_large_body.json()["detail"]["code"] == "REQUEST_TOO_LARGE"

    unsupported = payload()
    unsupported["module"] = "cardiology"
    unsupported_response = client.post(
        "/clinical/transcript/extract",
        headers={"X-Clinical-Key": "synth-key"},
        json=unsupported,
    )
    assert unsupported_response.status_code == 422
    assert unsupported_response.json()["detail"]["code"] == "UNSUPPORTED_MODULE"
    assert fake.calls == 0


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code"),
    [
        (ProviderUnavailable(), 503, "PROVIDER_UNAVAILABLE"),
        (ProviderRefusal(), 422, "PROVIDER_REFUSAL"),
        (ProviderInvalidOutput(), 502, "PROVIDER_INVALID_OUTPUT"),
    ],
)
def test_provider_failures_map_to_sanitized_public_codes(error, expected_status, expected_code):
    client, _ = make_client(error=error)
    response = client.post(
        "/clinical/transcript/extract",
        headers={"X-Clinical-Key": "synth-key"},
        json=payload("SENSITIVE-SYNTHETIC-TEXT"),
    )
    assert response.status_code == expected_status
    assert response.json()["detail"]["code"] == expected_code
    assert "SENSITIVE-SYNTHETIC-TEXT" not in response.text


def test_default_provider_is_fail_closed_until_transcript_specific_gate(monkeypatch):
    monkeypatch.setenv("CLINICAL_DATA_KEY", "synth-key")
    monkeypatch.delenv("CLINICAL_TRANSCRIPT_AI_ENABLED", raising=False)
    monkeypatch.delenv("CLINICAL_TRANSCRIPT_PHI_PROVIDER_APPROVED", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-not-used")
    app = FastAPI()
    app.include_router(build_transcript_router())
    client = TestClient(app)
    response = client.post(
        "/clinical/transcript/extract",
        headers={"X-Clinical-Key": "synth-key"},
        json=payload("NO-REAL-PATIENT-DATA"),
    )
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "PROVIDER_NOT_CONFIGURED"
    assert "NO-REAL-PATIENT-DATA" not in response.text


class FakeResponses:
    def __init__(self, parsed):
        self.parsed = parsed
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return type("Resp", (), {"output_parsed": self.parsed, "output": []})()


class FakeClient:
    def __init__(self, parsed):
        self.responses = FakeResponses(parsed)


def test_openai_adapter_contract(monkeypatch):
    monkeypatch.setenv("CLINICAL_TRANSCRIPT_AI_ENABLED", "true")
    monkeypatch.setenv("CLINICAL_TRANSCRIPT_PHI_PROVIDER_APPROVED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic")
    parsed = provider_result([])
    client = FakeClient(parsed)
    output = OpenAITranscriptProvider(client=client).extract(req(), "PROFILE")
    assert output == parsed
    kwargs = client.responses.kwargs
    assert kwargs["store"] is False
    assert kwargs["text_format"] is ProviderTranscriptExtractionV1
    assert "tools" not in kwargs
    assert kwargs["model"] == "gpt-5.6"
    assert DEFAULT_TIMEOUT_SECONDS == 120.0


def test_openai_client_creation_disables_retries_and_bounds_timeout(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    provider = OpenAITranscriptProvider(timeout_seconds=45.0)
    client = provider._client_or_create()
    assert isinstance(client, FakeOpenAI)
    assert captured == {"max_retries": 0, "timeout": 45.0}
