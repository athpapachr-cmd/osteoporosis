"""Synthetic web-audio safety checks; no real recording or provider request."""
from __future__ import annotations

import io
import logging
import asyncio
from dataclasses import replace
from datetime import datetime, timezone

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update
from sqlalchemy.orm import Session

import clinical_visit_audio as audio
from clinical_data import PatientORM, VisitCaptureContextORM
from test_cockpit_synthetic_161 import setup
from test_visit_capture import HEADERS, _client_with_engine, _context, _patient


WEBM = b"\x1aE\xdf\xa3" + b"\x00" * 128
TRANSCRIPT = (
    "Επανέλεγχος οστεοπόρωσης. DEXA T-score -2.8 στο ισχίο. "
    "Ξεκινά alendronate 70 mg εβδομαδιαίως. "
    "Η ασθενής προτιμά να περιμένει."
)


def enabled(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-provider-key")
    monkeypatch.setenv("VISIT_AUDIO_OPENAI_PHI_APPROVED", "1")
    monkeypatch.setenv("VISIT_AUDIO_RETENTION_MODE", "standard")
    monkeypatch.setenv("VISIT_AUDIO_OPENAI_PROJECT_ID", "proj_synthetic_standard")
    monkeypatch.setenv("VISIT_AUDIO_RECOVERY_SECRET", "synthetic-recovery-secret-" + "x" * 40)


def labels(batch, *, identity="consistent", bad_id=None):
    result = {"identity_state": identity, "facts": [], "suggested_next_checks": [],
              "unclassified_ids": []}
    for identifier, text in batch:
        if text.startswith("Επανέλεγχος"):
            kind = "reason"
        elif text.startswith("DEXA"):
            kind = "finding"
        elif text.startswith("Ξεκινά"):
            kind = "therapy"
        else:
            kind = "preference_undecided"
        result["facts"].append({"source_id": identifier, "excerpt": text, "kind": kind})
        if text.startswith("Ξεκινά"):
            result["suggested_next_checks"].append({
                "intent": "therapy_tolerance",
                "source_id": identifier,
            })
    if bad_id is not None:
        result["facts"].append({"source_id": bad_id, "excerpt": "άγνωστο", "kind": "finding"})
    return result


def considerations(facts, items=None):
    return {"contextual_clinical_considerations": items or []}


def one_consideration(facts):
    source_ids = [item["source_id"] for item in facts if item["kind"] in ("finding", "therapy")]
    return considerations(facts, [{
        "category": "management_review",
        "text": "Να εξεταστεί αν χρειάζεται συνολική επανεκτίμηση του καταγματικού κινδύνου.",
        "rationale": "Συνδυάζει το ρητό εύρημα οστικής πυκνότητας με τη ρητά αναφερόμενη θεραπεία.",
        "source_ids": source_ids[:2],
        "confidence": "moderate",
    }])


def draft(client, context_id):
    return client.post("/clinical/visit-capture/audio/draft", headers={
        **HEADERS, "X-Visit-Context": context_id,
        "X-Encounter-Date": "2026-10-27", "Content-Type": "audio/webm",
        "X-Visit-Retention-Acknowledged": "standard",
    }, content=WEBM)


def reprocess(client, context_id, transcript, recovery_binding, encounter_date="2026-10-27"):
    return client.post("/clinical/visit-capture/audio/reprocess", headers={
        **HEADERS, "X-Visit-Retention-Acknowledged": "standard",
    }, json={"context_id": context_id, "encounter_date": encounter_date,
             "transcript": transcript, "recovery_binding": recovery_binding})


def binding_claims(token):
    encoded = token.split(".", 1)[0]
    return audio.json.loads(audio.base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))


def test_gate_auth_and_request_format(monkeypatch):
    monkeypatch.delenv("VISIT_AUDIO_RETENTION_MODE", raising=False)
    client, _ = _client_with_engine(monkeypatch)
    _patient(client, "SYN-WEB")
    context = _context(client, "SYN-WEB")
    assert client.get("/clinical/visit-capture/audio/capability").status_code == 401
    assert client.get("/clinical/visit-capture/audio/capability", headers=HEADERS).json() == {"enabled": False}
    assert draft(client, context["context_id"]).status_code == 503
    enabled(monkeypatch)
    assert client.get("/clinical/visit-capture/audio/capability", headers=HEADERS).json() == {"enabled": True}
    missing_ack = client.post("/clinical/visit-capture/audio/draft", headers={
        **HEADERS, "X-Visit-Context": context["context_id"],
        "X-Encounter-Date": "2026-10-27", "Content-Type": "audio/webm",
    }, content=WEBM)
    assert missing_ack.status_code == 422
    bad = client.post("/clinical/visit-capture/audio/draft", headers={
        **HEADERS, "X-Visit-Context": context["context_id"],
        "X-Encounter-Date": "2026-10-27", "Content-Type": "audio/webm",
        "X-Visit-Retention-Acknowledged": "standard",
    }, content=b"wrong" * 30)
    assert bad.status_code == 422
    monkeypatch.delenv("VISIT_AUDIO_OPENAI_PROJECT_ID")
    assert client.get("/clinical/visit-capture/audio/capability", headers=HEADERS).json() == {"enabled": False}
    monkeypatch.setenv("VISIT_AUDIO_OPENAI_PROJECT_ID", "invalid-project")
    assert client.get("/clinical/visit-capture/audio/capability", headers=HEADERS).json() == {"enabled": False}
    monkeypatch.setenv("VISIT_AUDIO_OPENAI_PROJECT_ID", "proj_synthetic_standard")
    monkeypatch.setenv("VISIT_AUDIO_RETENTION_MODE", "other")
    assert client.get("/clinical/visit-capture/audio/capability", headers=HEADERS).json() == {"enabled": False}
    monkeypatch.setenv("VISIT_AUDIO_RETENTION_MODE", "standard")
    monkeypatch.delenv("VISIT_AUDIO_RECOVERY_SECRET")
    assert client.get("/clinical/visit-capture/audio/capability", headers=HEADERS).json() == {"enabled": False}


def test_provider_multipart_and_private_responses_contract(monkeypatch):
    enabled(monkeypatch)
    settings = audio.AudioSettings.from_env()
    requests = []

    async def receive(request):
        body = await request.aread()
        requests.append((request, body))
        if request.url.path.endswith("/transcriptions"):
            return httpx.Response(200, json={"text": TRANSCRIPT})
        request_json = __import__("json").loads(body)
        if request_json["text"]["format"]["name"] == "visit_audio_contextual_considerations_v1":
            output = considerations([])
        else:
            output = labels([(0, "Επανέλεγχος")])
        return httpx.Response(200, json={"status": "completed", "output": [{
            "content": [{"type": "output_text", "text": __import__("json").dumps(output)}]
        }]})

    real = httpx.AsyncClient
    monkeypatch.setattr(audio.httpx, "AsyncClient", lambda **kwargs: real(
        transport=httpx.MockTransport(receive), **kwargs))

    import asyncio
    async def run():
        assert await audio.OpenAITranscriber(settings).transcribe(io.BytesIO(WEBM), "audio/webm") == TRANSCRIPT
        assert (await audio.OpenAISummarizer(settings).classify(
            [(0, "Επανέλεγχος")], {"full_name": "Synthetic"}))["facts"][0]["kind"] == "reason"
        assert (await audio.OpenAISummarizer(settings).classify_retry(
            [(0, "Επανέλεγχος")], {"full_name": "Synthetic"},
            "classification_overlap"))["facts"][0]["kind"] == "reason"
        assert await audio.OpenAISummarizer(settings).consider([
            {"source_id": 0, "excerpt": "Η Synthetic ήρθε για επανέλεγχο", "kind": "reason"}
        ]) == considerations([])
    asyncio.run(run())
    transcription = requests[0][1]
    assert transcription.count(b'name="languages[]"') == 2
    assert b'gpt-transcribe' in transcription and b'name="keywords[]"' in transcription
    assert all(request.headers["OpenAI-Project"] == settings.project_id for request, _ in requests)
    assert all("OpenAI-Organization" not in request.headers for request, _ in requests)
    assert audio.provider_headers(audio.AudioSettings.from_env())["OpenAI-Project"] == settings.project_id
    monkeypatch.setenv("VISIT_AUDIO_OPENAI_ORG_ID", "org-synthetic")
    assert audio.provider_headers(audio.AudioSettings.from_env())["OpenAI-Organization"] == "org-synthetic"
    response_json = __import__("json").loads(requests[1][1])
    assert response_json["model"] == "gpt-6-luna"
    assert response_json["store"] is False
    assert response_json["text"]["format"]["strict"] is True
    suggestion_schema = response_json["text"]["format"]["schema"]["properties"]["suggested_next_checks"]["items"]
    assert set(suggestion_schema["properties"]) == {"intent", "source_id"}
    assert "text" not in suggestion_schema["properties"] and "rationale" not in suggestion_schema["properties"]
    assert "tools" not in response_json
    retry_json = __import__("json").loads(requests[2][1])
    assert "classification_overlap" in retry_json["input"][0]["content"]
    assert retry_json["input"][1]["content"] == response_json["input"][1]["content"]
    assert retry_json["store"] is False and "tools" not in retry_json
    consideration_json = __import__("json").loads(requests[3][1])
    assert consideration_json["model"] == "gpt-6-luna" and consideration_json["store"] is False
    assert consideration_json["text"]["format"]["name"] == "visit_audio_contextual_considerations_v1"
    consideration_item = consideration_json["text"]["format"]["schema"]["properties"][
        "contextual_clinical_considerations"]["items"]
    assert set(consideration_item["properties"]) == {
        "category", "text", "rationale", "source_ids", "confidence"}
    assert "selected_identity" not in consideration_json["input"][1]["content"]
    assert "source_units" not in consideration_json["input"][1]["content"]
    consideration_source = __import__("json").loads(consideration_json["input"][1]["content"])
    assert consideration_source == {"validated_facts": [{
        "source_id": 0,
        "excerpt": "Η Synthetic ήρθε για επανέλεγχο",
        "kind": "reason",
    }]}
    assert "tools" not in consideration_json


def test_provider_rejection_retains_status_only(monkeypatch):
    enabled(monkeypatch)
    settings = audio.AudioSettings.from_env()

    async def reject(request):
        return httpx.Response(403, json={"error": {"message": "synthetic private provider text"}})

    real = httpx.AsyncClient
    monkeypatch.setattr(audio.httpx, "AsyncClient", lambda **kwargs: real(
        transport=httpx.MockTransport(reject), **kwargs))

    import asyncio
    async def run():
        with pytest.raises(audio.ProviderHTTPFailure) as transcription:
            await audio.OpenAITranscriber(settings).transcribe(io.BytesIO(WEBM), "audio/webm")
        with pytest.raises(audio.ProviderHTTPFailure) as classification:
            await audio.OpenAISummarizer(settings).classify([(0, "Επανέλεγχος")], {})
        for failure in (transcription.value, classification.value):
            assert failure.status_code == 403
            assert "synthetic private provider text" not in str(failure)
    asyncio.run(run())


def test_multi_clause_source_builds_treatment_plan_and_obligation_without_inference():
    units = [
        "Επανέλεγχος οστεοπόρωσης με θωρακικό πόνο και σχεδιάστηκε ακτινογραφία.",
        "Ξεκινά αλενδρονάτη 70 mg εβδομαδιαίως και επανέλεγχος σε τρεις μήνες.",
    ]
    result = {
        "identity_state": "consistent",
        "facts": [
            {"source_id": 0, "excerpt": "Επανέλεγχος οστεοπόρωσης", "kind": "reason"},
            {"source_id": 0, "excerpt": "θωρακικό πόνο", "kind": "symptom"},
            {"source_id": 0, "excerpt": "σχεδιάστηκε ακτινογραφία", "kind": "obligation_investigation"},
            {"source_id": 1, "excerpt": "Ξεκινά αλενδρονάτη 70 mg εβδομαδιαίως", "kind": "therapy"},
            {"source_id": 1, "excerpt": "επανέλεγχος σε τρεις μήνες", "kind": "follow_up"},
        ],
        "suggested_next_checks": [{"intent": "therapy_adherence", "source_id": 1}],
        "unclassified_ids": [],
    }
    identity, facts, suggestions, unclassified = audio.checked_analysis(
        result, list(enumerate(units)))
    assert identity == "consistent"
    candidate, review = audio.build_candidate(units, facts, unclassified, "2026-10-28")
    assert candidate.reason_for_visit.text == "Επανέλεγχος οστεοπόρωσης"
    assert [item.text for item in candidate.history] == ["θωρακικό πόνο"]
    assert candidate.recommendations[0].text == "Ξεκινά αλενδρονάτη 70 mg εβδομαδιαίως"
    assert candidate.recommendations[0].type == "therapy"
    assert candidate.medications == []
    assert candidate.pending[0].description == "σχεδιάστηκε ακτινογραφία"
    assert candidate.pending[0].item_type == "investigation"
    assert candidate.pending[0].responsible_role == "unspecified"
    assert candidate.pending[0].review_date is None and candidate.pending[0].trigger is None
    assert candidate.follow_up[0].text == "επανέλεγχος σε τρεις μήνες"
    assert suggestions[0]["source_id"] == 1
    assert suggestions[0]["text"].startswith("Έλεγχος αν η αναφερόμενη θεραπεία")
    assert {item["hint"] for item in review} >= {"therapy", "obligation_investigation"}


@pytest.mark.parametrize("facts", [
    [
        {"source_id": 0, "excerpt": "Μπορούμε να κάνουμε μαγνητική", "kind": "option"},
        {"source_id": 0, "excerpt": "Μπορούμε να κάνουμε μαγνητική", "kind": "obligation_investigation"},
    ],
    [
        {"source_id": 0, "excerpt": "Μπορούμε να κάνουμε μαγνητική", "kind": "option"},
        {"source_id": 0, "excerpt": "κάνουμε μαγνητική", "kind": "completed"},
    ],
])
def test_conflicting_or_overlapping_source_spans_fail_closed(facts):
    result = {"identity_state": "consistent", "facts": facts,
              "suggested_next_checks": [], "unclassified_ids": []}
    with pytest.raises(ValueError, match="classification_overlap"):
        audio.checked_analysis(result, [(0, "Μπορούμε να κάνουμε μαγνητική")])


def test_ambiguous_excerpt_and_free_text_or_incompatible_suggestion_fail_closed():
    repeated = {"identity_state": "consistent", "facts": [
        {"source_id": 0, "excerpt": "πόνος", "kind": "symptom"}],
        "suggested_next_checks": [], "unclassified_ids": []}
    with pytest.raises(ValueError, match="classification_excerpt"):
        audio.checked_analysis(repeated, [(0, "πόνος και ξανά πόνος")])
    for suggestion in (
        {"text": "Να γίνει μαγνητική τομογραφία;", "rationale": "Επίμονος πόνος.", "source_ids": [0]},
        {"text": "Πιθανή νέα διάγνωση οστεοπόρωσης.", "rationale": "Εύρημα.", "source_ids": [0]},
        {"intent": "diagnose_osteoporosis", "source_id": 0},
        {"intent": "planned_result_review", "source_id": 0},
    ):
        unsafe = {"identity_state": "consistent", "facts": [
            {"source_id": 0, "excerpt": "Επανέλεγχος", "kind": "reason"}],
            "suggested_next_checks": [suggestion], "unclassified_ids": []}
        with pytest.raises(ValueError, match="classification_suggestions"):
            audio.checked_analysis(unsafe, [(0, "Επανέλεγχος")])


def test_contextual_considerations_require_validated_sources_and_bounded_shape():
    facts = [
        {"source_id": 2, "excerpt": "πόνος στη ράχη", "kind": "symptom"},
        {"source_id": 8, "excerpt": "απώλεια ύψους", "kind": "finding"},
    ]
    valid = {"contextual_clinical_considerations": [{
        "category": "investigation_consideration",
        "text": "Να εξεταστεί αν χρειάζεται περαιτέρω διερεύνηση της ράχης.",
        "rationale": "Ο συνδυασμός πόνου και απώλειας ύψους μπορεί να χρειάζεται κλινική συσχέτιση.",
        "source_ids": [2, 8],
        "confidence": "moderate",
    }]}
    assert audio.checked_considerations(valid, facts)[0]["source_ids"] == [2, 8]

    invalid_items = [
        {**valid["contextual_clinical_considerations"][0], "source_ids": [2, 99]},
        {**valid["contextual_clinical_considerations"][0], "source_ids": [2, 2]},
        {**valid["contextual_clinical_considerations"][0], "confidence": "certain"},
        {**valid["contextual_clinical_considerations"][0], "category": "diagnosis"},
        {**valid["contextual_clinical_considerations"][0], "text": "α\x00β"},
        {**valid["contextual_clinical_considerations"][0], "extra": "field"},
    ]
    for item in invalid_items:
        with pytest.raises(ValueError):
            audio.checked_considerations({"contextual_clinical_considerations": [item]}, facts)
    with pytest.raises(ValueError, match="duplicate"):
        audio.checked_considerations({
            "contextual_clinical_considerations": [
                valid["contextual_clinical_considerations"][0],
                {**valid["contextual_clinical_considerations"][0],
                 "text": valid["contextual_clinical_considerations"][0]["text"].upper()},
            ]}, facts)


def test_contextual_deadline_helpers_preserve_finalization_reserve():
    assert audio.optional_timeout_budget(100.0, 90.0, 90) == 5.0
    assert audio.optional_timeout_budget(100.0, 94.1, 90) is None
    audio.ensure_before_deadline(94.9, 95.0)
    with pytest.raises(TimeoutError):
        audio.ensure_before_deadline(95.0, 95.0)


def test_structural_classification_failure_retries_once_and_recovers(monkeypatch):
    enabled(monkeypatch)
    client, _ = _client_with_engine(monkeypatch)
    _patient(client, "SYN-WEB")
    context = _context(client, "SYN-WEB")
    calls = []

    class Transcriber:
        async def transcribe(self, file, mime):
            return TRANSCRIPT

    class Summarizer:
        async def classify(self, batch, identity):
            calls.append("first")
            return labels(batch, bad_id=999)

        async def classify_retry(self, batch, identity, validation_code):
            calls.append(validation_code)
            return labels(batch)

        async def consider(self, facts):
            return considerations(facts)

    monkeypatch.setattr(audio, "make_providers", lambda settings: (Transcriber(), Summarizer()))
    response = draft(client, context["context_id"])
    assert response.status_code == 200, response.text
    assert response.json()["candidate"]["reason_for_visit"]["text"] == "Επανέλεγχος οστεοπόρωσης."
    assert calls == ["first", "classification_values"]


def test_repeated_structural_failure_logs_only_fixed_code(monkeypatch, caplog):
    enabled(monkeypatch)
    client, _ = _client_with_engine(monkeypatch)
    _patient(client, "SYN-WEB")
    context = _context(client, "SYN-WEB")
    calls = []

    class Transcriber:
        async def transcribe(self, file, mime):
            return TRANSCRIPT

    class Summarizer:
        async def classify(self, batch, identity):
            calls.append("first")
            return labels(batch, bad_id=999)

        async def classify_retry(self, batch, identity, validation_code):
            calls.append(validation_code)
            return labels(batch, bad_id=999)

    monkeypatch.setattr(audio, "make_providers", lambda settings: (Transcriber(), Summarizer()))
    with caplog.at_level(logging.WARNING, logger="clinical_visit_audio"):
        response = draft(client, context["context_id"])
    assert response.status_code == 200
    body = response.json()
    assert body["processing_status"] == "transcript_only"
    assert body["transcript"] == TRANSCRIPT
    assert body["processing_failure"] == {
        "stage": "classification", "code": "classification_values",
        "provider_http_status": None, "retryable": True,
    }
    assert "candidate" not in body and body["recovery_binding"]
    assert calls == ["first", "classification_values"]
    assert "validation_code=classification_values" in caplog.text
    assert TRANSCRIPT not in caplog.text and "SYN-WEB" not in caplog.text


def test_bound_transcript_reprocess_skips_transcription_and_recovers(monkeypatch):
    enabled(monkeypatch)
    client, _ = _client_with_engine(monkeypatch)
    _patient(client, "SYN-WEB")
    context = _context(client, "SYN-WEB")
    calls = {"transcribe": 0, "classify": 0}

    class Transcriber:
        async def transcribe(self, file, mime):
            calls["transcribe"] += 1
            return TRANSCRIPT

    class Summarizer:
        async def classify(self, batch, identity):
            calls["classify"] += 1
            return labels(batch, bad_id=999) if calls["classify"] <= 2 else labels(batch)

        async def classify_retry(self, batch, identity, validation_code):
            calls["classify"] += 1
            return labels(batch, bad_id=999) if calls["classify"] <= 2 else labels(batch)

        async def consider(self, facts):
            return considerations(facts)

    monkeypatch.setattr(audio, "make_providers", lambda settings: (Transcriber(), Summarizer()))
    partial = draft(client, context["context_id"])
    assert partial.status_code == 200
    partial_body = partial.json()
    assert partial_body["processing_status"] == "transcript_only"

    recovered = reprocess(client, context["context_id"], partial_body["transcript"],
                          partial_body["recovery_binding"])
    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["processing_status"] == "complete"
    assert recovered.json()["candidate"]["reason_for_visit"]["text"] == "Επανέλεγχος οστεοπόρωσης."
    assert calls["transcribe"] == 1


def test_reprocess_binding_rejects_substitution_and_new_contexts(monkeypatch):
    enabled(monkeypatch)
    client, _ = _client_with_engine(monkeypatch)
    _patient(client, "SYN-A")
    _patient(client, "SYN-B")
    session_id = "synthetic-browser-session-0001"
    context = _context(client, "SYN-A", session_id)
    provider_calls = []

    class Transcriber:
        async def transcribe(self, file, mime):
            return TRANSCRIPT

    class BrokenThenStable:
        def __init__(self):
            self.fail = True

        async def classify(self, batch, identity):
            provider_calls.append("classify")
            if self.fail:
                return labels(batch, bad_id=999)
            return labels(batch)

        async def classify_retry(self, batch, identity, validation_code):
            provider_calls.append("retry")
            return labels(batch, bad_id=999)

        async def consider(self, facts):
            return considerations(facts)

    summarizer = BrokenThenStable()
    monkeypatch.setattr(audio, "make_providers", lambda settings: (Transcriber(), summarizer))
    partial = draft(client, context["context_id"]).json()
    baseline_calls = len(provider_calls)
    token = partial["recovery_binding"]

    for changed_transcript, changed_date, changed_token in (
        (TRANSCRIPT + " x", "2026-10-27", token),
        (TRANSCRIPT, "2026-10-28", token),
        (TRANSCRIPT, "2026-10-27", token[:-1] + ("0" if token[-1] != "0" else "1")),
    ):
        rejected = reprocess(client, context["context_id"], changed_transcript,
                             changed_token, changed_date)
        assert rejected.status_code == 409
        assert "transcript" not in rejected.text and "recovery_binding" not in rejected.text
        assert len(provider_calls) == baseline_calls

    renewed = client.post("/clinical/visit-capture/context/renew", headers=HEADERS, json={
        "prior_context_id": context["context_id"], "patient_id": "SYN-A",
        "client_session_id": session_id,
    })
    assert renewed.status_code == 200
    before_renewed = len(provider_calls)
    rejected = reprocess(client, renewed.json()["context_id"], TRANSCRIPT, token)
    assert rejected.status_code == 409
    assert "transcript" not in rejected.text and len(provider_calls) == before_renewed

    fresh_same_patient = _context(client, "SYN-A", session_id)
    before_fresh = len(provider_calls)
    rejected = reprocess(client, fresh_same_patient["context_id"], TRANSCRIPT, token)
    assert rejected.status_code == 409
    assert "transcript" not in rejected.text and len(provider_calls) == before_fresh

    other = _context(client, "SYN-B", session_id)
    before_other = len(provider_calls)
    rejected = reprocess(client, other["context_id"], TRANSCRIPT, token)
    assert rejected.status_code == 409
    assert "transcript" not in rejected.text and len(provider_calls) == before_other


def test_repeated_processing_failure_preserves_original_binding(monkeypatch):
    enabled(monkeypatch)
    client, _ = _client_with_engine(monkeypatch)
    _patient(client, "SYN-WEB")
    context = _context(client, "SYN-WEB")
    provider_calls = []

    class Transcriber:
        async def transcribe(self, file, mime):
            return TRANSCRIPT

    class AlwaysInvalid:
        async def classify(self, batch, identity):
            provider_calls.append("classify")
            return labels(batch, bad_id=999)

        async def classify_retry(self, batch, identity, validation_code):
            provider_calls.append("retry")
            return labels(batch, bad_id=999)

    monkeypatch.setattr(audio, "make_providers", lambda settings: (Transcriber(), AlwaysInvalid()))
    partial = draft(client, context["context_id"]).json()
    original_binding = partial["recovery_binding"]
    retried = reprocess(client, context["context_id"], TRANSCRIPT, original_binding)
    assert retried.status_code == 200, retried.text
    assert retried.json()["processing_status"] == "transcript_only"
    successor_binding = retried.json()["recovery_binding"]
    assert successor_binding != original_binding
    original_claims = binding_claims(original_binding)
    successor_claims = binding_claims(successor_binding)
    for key in ("original_context_id", "issued_at", "expires_at", "encounter_date",
                "transcript_sha256", "patient_id", "patient_updated_at"):
        assert successor_claims[key] == original_claims[key]

    calls_before_replay = len(provider_calls)
    replayed = reprocess(client, context["context_id"], TRANSCRIPT, original_binding)
    assert replayed.status_code == 409
    assert len(provider_calls) == calls_before_replay


def test_identity_conflict_during_reprocess_permanently_consumes_retry(monkeypatch):
    enabled(monkeypatch)
    client, _ = _client_with_engine(monkeypatch)
    _patient(client, "SYN-WEB")
    context = _context(client, "SYN-WEB")
    calls = []

    class Transcriber:
        async def transcribe(self, file, mime):
            return TRANSCRIPT

    class StructuralThenIdentityConflict:
        async def classify(self, batch, identity):
            calls.append("classify")
            if len(calls) <= 2:
                return labels(batch, bad_id=999)
            return labels(batch, identity="conflict")

        async def classify_retry(self, batch, identity, validation_code):
            calls.append("retry")
            return labels(batch, bad_id=999)

    monkeypatch.setattr(
        audio, "make_providers", lambda settings: (Transcriber(), StructuralThenIdentityConflict()))
    partial = draft(client, context["context_id"]).json()
    original_binding = partial["recovery_binding"]
    conflicted = reprocess(client, context["context_id"], TRANSCRIPT, original_binding)
    assert conflicted.status_code == 200, conflicted.text
    assert conflicted.json()["processing_failure"]["code"] == "source_identity_unresolved"
    assert conflicted.json()["processing_failure"]["retryable"] is False
    before_replay = len(calls)
    replayed = reprocess(client, context["context_id"], TRANSCRIPT, original_binding)
    assert replayed.status_code == 409
    assert "transcript" not in replayed.text and len(calls) == before_replay


def test_nonretryable_identity_binding_rejects_reprocess(monkeypatch):
    enabled(monkeypatch)
    client, _ = _client_with_engine(monkeypatch)
    _patient(client, "SYN-WEB")
    context = _context(client, "SYN-WEB")
    calls = []

    class Transcriber:
        async def transcribe(self, file, mime):
            return TRANSCRIPT

    class Summarizer:
        async def classify(self, batch, identity):
            calls.append("classify")
            return labels(batch, identity="uncertain")

    monkeypatch.setattr(audio, "make_providers", lambda settings: (Transcriber(), Summarizer()))
    partial = draft(client, context["context_id"]).json()
    assert partial["processing_failure"]["retryable"] is False
    before = len(calls)
    rejected = reprocess(client, context["context_id"], TRANSCRIPT,
                         partial["recovery_binding"])
    assert rejected.status_code == 409
    assert "transcript" not in rejected.text and len(calls) == before


def test_expired_recovery_binding_fails_before_provider(monkeypatch):
    enabled(monkeypatch)
    client, _ = _client_with_engine(monkeypatch)
    _patient(client, "SYN-WEB")
    context = _context(client, "SYN-WEB")
    calls = []

    class Transcriber:
        async def transcribe(self, file, mime):
            return TRANSCRIPT

    class Summarizer:
        async def classify(self, batch, identity):
            calls.append("classify")
            return labels(batch, bad_id=999)

        async def classify_retry(self, batch, identity, validation_code):
            calls.append("retry")
            return labels(batch, bad_id=999)

    monkeypatch.setattr(audio, "make_providers", lambda settings: (Transcriber(), Summarizer()))
    partial = draft(client, context["context_id"]).json()
    before = len(calls)
    issued_now = audio.time.time()
    monkeypatch.setattr(audio.time, "time", lambda: issued_now + audio.RECOVERY_BINDING_SECONDS + 1)
    rejected = reprocess(client, context["context_id"], TRANSCRIPT,
                         partial["recovery_binding"])
    assert rejected.status_code == 409
    assert "transcript" not in rejected.text and len(calls) == before


def test_identity_conflict_is_not_retried(monkeypatch):
    enabled(monkeypatch)
    client, _ = _client_with_engine(monkeypatch)
    _patient(client, "SYN-WEB")
    context = _context(client, "SYN-WEB")
    retried = []

    class Transcriber:
        async def transcribe(self, file, mime):
            return TRANSCRIPT

    class Summarizer:
        async def classify(self, batch, identity):
            return labels(batch, identity="conflict")

        async def classify_retry(self, batch, identity, validation_code):
            retried.append(validation_code)
            return labels(batch)

    monkeypatch.setattr(audio, "make_providers", lambda settings: (Transcriber(), Summarizer()))
    body = draft(client, context["context_id"]).json()
    assert body["processing_status"] == "transcript_only"
    assert body["processing_failure"]["code"] == "source_identity_unresolved"
    assert body["processing_failure"]["retryable"] is False
    assert retried == []


@pytest.mark.parametrize("identity_state", ["conflict", "uncertain"])
def test_unresolved_identity_with_structural_error_is_not_retried(monkeypatch, identity_state):
    enabled(monkeypatch)
    client, _ = _client_with_engine(monkeypatch)
    _patient(client, "SYN-WEB")
    context = _context(client, "SYN-WEB")
    retried = []

    class Transcriber:
        async def transcribe(self, file, mime):
            return TRANSCRIPT

    class Summarizer:
        async def classify(self, batch, identity):
            return labels(batch, identity=identity_state, bad_id=999)

        async def classify_retry(self, batch, identity, validation_code):
            retried.append(validation_code)
            return labels(batch)

    monkeypatch.setattr(audio, "make_providers", lambda settings: (Transcriber(), Summarizer()))
    body = draft(client, context["context_id"]).json()
    assert body["processing_status"] == "transcript_only"
    assert body["processing_failure"]["code"] == "source_identity_unresolved"
    assert body["processing_failure"]["retryable"] is False
    assert retried == []


def test_whole_visit_contextual_pass_combines_different_batches(monkeypatch):
    enabled(monkeypatch)
    client, _ = _client_with_engine(monkeypatch)
    _patient(client, "SYN-WEB")
    context = _context(client, "SYN-WEB")
    transcript = (
        "Επανέλεγχος οστεοπόρωσης " + "α" * 3200 + ". "
        "Υπάρχει θωρακικός πόνος " + "β" * 3200 + ". "
        "Αναφέρθηκε απώλεια ύψους " + "γ" * 3200 + "."
    )
    assert len(audio.batches(audio.source_units(transcript))) > 1

    class Transcriber:
        async def transcribe(self, file, mime):
            return transcript

    class Summarizer:
        async def classify(self, batch, identity):
            result = {"identity_state": "consistent", "facts": [],
                      "suggested_next_checks": [], "unclassified_ids": []}
            for identifier, text in batch:
                if identifier == 0:
                    excerpt, kind = "Επανέλεγχος οστεοπόρωσης", "reason"
                elif identifier == 1:
                    excerpt, kind = "Υπάρχει θωρακικός πόνος", "symptom"
                else:
                    excerpt, kind = "Αναφέρθηκε απώλεια ύψους", "finding"
                result["facts"].append({"source_id": identifier, "excerpt": excerpt, "kind": kind})
            return result

        async def consider(self, facts):
            assert {item["source_id"] for item in facts} == {0, 1, 2}
            return considerations(facts, [{
                "category": "investigation_consideration",
                "text": "Να εξεταστεί αν ο θωρακικός πόνος και η απώλεια ύψους χρειάζονται συσχέτιση.",
                "rationale": "Η σκέψη βασίζεται στον συνδυασμό δύο ρητών ευρημάτων από διαφορετικά τμήματα.",
                "source_ids": [1, 2],
                "confidence": "moderate",
            }])

    monkeypatch.setattr(audio, "make_providers", lambda settings: (Transcriber(), Summarizer()))
    response = draft(client, context["context_id"])
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["contextual_considerations_status"] == "complete"
    assert body["contextual_clinical_considerations"][0]["source_ids"] == [1, 2]
    assert body["contextual_clinical_considerations"][0]["text"] not in str(body["candidate"])


def test_contextual_input_limit_preserves_exact_source_candidate(monkeypatch):
    enabled(monkeypatch)
    client, _ = _client_with_engine(monkeypatch)
    _patient(client, "SYN-WEB")
    context = _context(client, "SYN-WEB")
    called = []

    class Transcriber:
        async def transcribe(self, file, mime):
            return TRANSCRIPT

    class Summarizer:
        async def classify(self, batch, identity):
            return labels(batch)

        async def consider(self, facts):
            called.append(True)
            return one_consideration(facts)

    monkeypatch.setattr(audio, "make_providers", lambda settings: (Transcriber(), Summarizer()))
    monkeypatch.setattr(audio, "MAX_CONTEXTUAL_FACT_BYTES", 1)
    response = draft(client, context["context_id"])
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["candidate"]["reason_for_visit"]["text"] == "Επανέλεγχος οστεοπόρωσης."
    assert body["contextual_considerations_status"] == "input_limit"
    assert body["contextual_clinical_considerations"] == []
    assert called == []


def test_candidate_near_core_cutoff_survives_when_optional_budget_is_too_small(monkeypatch):
    enabled(monkeypatch)
    client, _ = _client_with_engine(monkeypatch)
    _patient(client, "SYN-WEB")
    context = _context(client, "SYN-WEB")
    settings = replace(audio.AudioSettings.from_env(), pipeline_timeout=1.1, summarize_timeout=5)
    monkeypatch.setattr(audio.AudioSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(audio, "FINALIZATION_RESERVE_SECONDS", 0.4)
    considered = []

    class Transcriber:
        async def transcribe(self, file, mime):
            await asyncio.sleep(0.5)
            return TRANSCRIPT

    class Summarizer:
        async def classify(self, batch, identity):
            return labels(batch)

        async def consider(self, facts):
            considered.append(True)
            return one_consideration(facts)

    monkeypatch.setattr(audio, "make_providers", lambda settings: (Transcriber(), Summarizer()))
    response = draft(client, context["context_id"])
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["candidate"]["reason_for_visit"]["text"] == "Επανέλεγχος οστεοπόρωσης."
    assert body["contextual_considerations_status"] == "unavailable"
    assert body["contextual_clinical_considerations"] == []
    assert considered == []


def test_optional_contextual_timeout_preserves_timely_candidate(monkeypatch):
    enabled(monkeypatch)
    client, _ = _client_with_engine(monkeypatch)
    _patient(client, "SYN-WEB")
    context = _context(client, "SYN-WEB")
    settings = replace(audio.AudioSettings.from_env(), pipeline_timeout=2.0, summarize_timeout=5)
    monkeypatch.setattr(audio.AudioSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(audio, "FINALIZATION_RESERVE_SECONDS", 0.3)
    considered = []

    class Transcriber:
        async def transcribe(self, file, mime):
            return TRANSCRIPT

    class Summarizer:
        async def classify(self, batch, identity):
            return labels(batch)

        async def consider(self, facts):
            considered.append(True)
            await asyncio.sleep(3.0)
            return one_consideration(facts)

    monkeypatch.setattr(audio, "make_providers", lambda settings: (Transcriber(), Summarizer()))
    response = draft(client, context["context_id"])
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["candidate"]["reason_for_visit"]["text"] == "Επανέλεγχος οστεοπόρωσης."
    assert body["contextual_considerations_status"] == "unavailable"
    assert body["contextual_clinical_considerations"] == []
    assert considered == [True]


def test_source_copy_provisional_review_and_confirmed_save(monkeypatch):
    enabled(monkeypatch)
    client, engine = _client_with_engine(monkeypatch)
    _patient(client, "SYN-WEB")
    context = _context(client, "SYN-WEB")

    class Transcriber:
        async def transcribe(self, file, mime):
            assert file.read(4) == WEBM[:4]
            return TRANSCRIPT

    class Summarizer:
        async def classify(self, batch, identity):
            return labels(batch)

        async def consider(self, facts):
            return one_consideration(facts)

    monkeypatch.setattr(audio, "make_providers", lambda settings: (Transcriber(), Summarizer()))
    result = draft(client, context["context_id"])
    assert result.status_code == 200, result.text
    body = result.json()
    candidate = body["candidate"]
    assert body["patient_id"] == "SYN-WEB"
    assert body["processing_status"] == "complete"
    assert body["transcript"] == TRANSCRIPT
    assert body["processing_failure"] is None and body["recovery_binding"]
    assert candidate["reason_for_visit"]["text"] == "Επανέλεγχος οστεοπόρωσης."
    assert candidate["findings"][0]["text"] == "DEXA T-score -2.8 στο ισχίο."
    assert candidate["decisions"] == [] and candidate["medications"] == []
    assert candidate["pending"] == []
    assert candidate["recommendations"][0]["text"] == "Ξεκινά alendronate 70 mg εβδομαδιαίως."
    assert candidate["patient_preferences"][0]["text"] == "Η ασθενής προτιμά να περιμένει."
    assert candidate["uncertainties"] == []
    assert candidate["follow_up"] == []
    assert body["suggested_next_checks"][0]["text"] == "Έλεγχος ανοχής της αναφερόμενης θεραπείας."
    assert body["contextual_considerations_status"] == "complete"
    assert body["contextual_clinical_considerations"][0]["source_ids"] == [1, 2]
    assert body["contextual_clinical_considerations"][0]["text"] not in str(candidate)
    assert {item["hint"] for item in body["review_units"]} >= {"finding", "therapy", "preference_undecided"}
    payload = {"context_id": context["context_id"], "candidate": candidate}
    preview = client.post("/clinical/visit-capture/preview", headers=HEADERS, json=payload)
    assert preview.status_code == 200 and preview.json()["can_save"] is False
    assert client.post("/clinical/visit-capture/save", headers=HEADERS, json=payload).status_code == 409
    payload["recording_patient_confirmed"] = True
    assert client.post("/clinical/visit-capture/preview", headers=HEADERS, json=payload).json()["can_save"] is True
    assert client.post("/clinical/visit-capture/save", headers=HEADERS, json=payload).status_code == 200
    assert client.post("/clinical/visit-capture/save", headers=HEADERS, json=payload).status_code == 409
    with Session(engine) as db:
        assert db.query(VisitCaptureContextORM).count() == 0


def test_patient_change_and_malformed_classification_fail_closed(monkeypatch):
    enabled(monkeypatch)
    client, engine = _client_with_engine(monkeypatch)
    _patient(client, "SYN-WEB")
    context = _context(client, "SYN-WEB")

    class Transcriber:
        async def transcribe(self, file, mime):
            with Session(engine) as db:
                db.execute(update(PatientORM).where(PatientORM.patient_id == "SYN-WEB")
                           .values(updated_at=datetime.now(timezone.utc)))
                db.commit()
            return TRANSCRIPT

    class Summarizer:
        async def classify(self, batch, identity):
            raise AssertionError("Provider must not receive stale patient context")

    monkeypatch.setattr(audio, "make_providers", lambda settings: (Transcriber(), Summarizer()))
    changed_after_transcription = draft(client, context["context_id"])
    assert changed_after_transcription.status_code == 409
    assert TRANSCRIPT not in changed_after_transcription.text
    assert "recovery_binding" not in changed_after_transcription.text

    fresh = _context(client, "SYN-WEB")
    class StableTranscriber:
        async def transcribe(self, file, mime):
            return TRANSCRIPT
    class BadSummarizer:
        async def classify(self, batch, identity):
            return labels(batch, bad_id=999)
    monkeypatch.setattr(audio, "make_providers", lambda settings: (StableTranscriber(), BadSummarizer()))
    failed = draft(client, fresh["context_id"])
    assert failed.status_code == 200
    assert failed.json()["processing_status"] == "transcript_only"
    assert failed.json()["transcript"] == TRANSCRIPT
    assert "candidate" not in failed.json()


def test_patient_change_after_classification_returns_no_transcript_or_token(monkeypatch):
    enabled(monkeypatch)
    client, engine = _client_with_engine(monkeypatch)
    _patient(client, "SYN-WEB")
    context = _context(client, "SYN-WEB")

    class Transcriber:
        async def transcribe(self, file, mime):
            return TRANSCRIPT

    class Summarizer:
        async def classify(self, batch, identity):
            with Session(engine) as db:
                db.execute(update(PatientORM).where(PatientORM.patient_id == "SYN-WEB")
                           .values(updated_at=datetime.now(timezone.utc)))
                db.commit()
            return labels(batch)

    monkeypatch.setattr(audio, "make_providers", lambda settings: (Transcriber(), Summarizer()))
    response = draft(client, context["context_id"])
    assert response.status_code == 409
    assert TRANSCRIPT not in response.text and "recovery_binding" not in response.text


def test_temp_audio_closed_on_failed_provider_and_oversize(monkeypatch, caplog):
    enabled(monkeypatch)
    client, _ = _client_with_engine(monkeypatch)
    _patient(client, "SYN-WEB")
    context = _context(client, "SYN-WEB")
    files = []
    real = audio.tempfile.TemporaryFile
    def temporary(*args, **kwargs):
        file = real(*args, **kwargs)
        files.append(file)
        return file
    monkeypatch.setattr(audio.tempfile, "TemporaryFile", temporary)

    class BrokenTranscriber:
        async def transcribe(self, file, mime):
            raise RuntimeError("secret patient words")
    monkeypatch.setattr(audio, "make_providers", lambda settings: (BrokenTranscriber(), None))
    with caplog.at_level(logging.WARNING, logger="clinical_visit_audio"):
        response = draft(client, context["context_id"])
    assert response.status_code == 502 and "secret patient words" not in response.text
    assert "stage=transcription kind=unexpected provider_http_status=None" in caplog.text
    assert "secret patient words" not in caplog.text
    assert files and all(file.closed for file in files)
    monkeypatch.setattr(audio, "MAX_AUDIO_BYTES", 100)
    assert draft(client, context["context_id"]).status_code == 413
    assert all(file.closed for file in files)


def test_private_failure_stage_and_http_status_without_clinical_content(monkeypatch, caplog):
    enabled(monkeypatch)
    client, _ = _client_with_engine(monkeypatch)
    _patient(client, "SYN-WEB")
    context = _context(client, "SYN-WEB")

    class RejectedTranscriber:
        async def transcribe(self, file, mime):
            raise audio.ProviderHTTPFailure(403)

    class StableTranscriber:
        async def transcribe(self, file, mime):
            return TRANSCRIPT

    class RejectedSummarizer:
        async def classify(self, batch, identity):
            raise audio.ProviderHTTPFailure(429)

    with caplog.at_level(logging.WARNING, logger="clinical_visit_audio"):
        monkeypatch.setattr(audio, "make_providers", lambda settings: (RejectedTranscriber(), None))
        response = draft(client, context["context_id"])
        assert response.status_code == 502
        assert response.json() == {"detail": "The recording could not be processed safely"}
        assert "stage=transcription kind=provider_http provider_http_status=403" in caplog.text
        assert TRANSCRIPT not in caplog.text and "SYN-WEB" not in caplog.text
        caplog.clear()

        monkeypatch.setattr(audio, "make_providers", lambda settings: (StableTranscriber(), RejectedSummarizer()))
        response = draft(client, context["context_id"])
        assert response.status_code == 200
        assert response.json()["processing_status"] == "transcript_only"
        assert response.json()["processing_failure"] == {
            "stage": "classification", "code": "provider_http",
            "provider_http_status": 429, "retryable": True,
        }
        assert "stage=classification kind=provider_http provider_http_status=429" in caplog.text
        assert TRANSCRIPT not in caplog.text and "SYN-WEB" not in caplog.text


def test_identity_refusal_and_missing_clinical_content_are_rejected(monkeypatch):
    enabled(monkeypatch)
    client, _ = _client_with_engine(monkeypatch)
    _patient(client, "SYN-WEB")
    context = _context(client, "SYN-WEB")

    class Transcriber:
        async def transcribe(self, file, mime):
            return TRANSCRIPT

    class Summarizer:
        def __init__(self, mode):
            self.mode = mode

        async def classify(self, batch, identity):
            if self.mode == "conflict":
                return labels(batch, identity="conflict")
            if self.mode == "refusal":
                return {"refusal": "untrusted source text"}
            result = {"identity_state": "consistent", "facts": [{"source_id": batch[0][0],
                      "excerpt": batch[0][1], "kind": "reason"}],
                      "suggested_next_checks": [], "unclassified_ids": [item[0] for item in batch[1:]]}
            return result

    for mode in ("conflict", "refusal", "empty"):
        monkeypatch.setattr(audio, "make_providers", lambda settings, m=mode: (Transcriber(), Summarizer(m)))
        response = draft(client, context["context_id"])
        assert response.status_code == 200
        assert response.json()["processing_status"] == "transcript_only"
        assert response.json()["transcript"] == TRANSCRIPT
        assert "candidate" not in response.json()


def test_browser_web_audio_review_and_confirmed_save(setup, monkeypatch):
    import socket
    import shutil
    import threading
    import uvicorn
    from playwright.sync_api import expect, sync_playwright
    from test_cockpit_synthetic_161 import KEY

    enabled(monkeypatch)
    app, _ = setup
    transcribed = []
    classifications = []

    class Transcriber:
        async def transcribe(self, file, mime):
            transcribed.append(True)
            assert file.read(4) == WEBM[:4]
            return TRANSCRIPT

    class Summarizer:
        async def classify(self, batch, identity):
            classifications.append("classify")
            return labels(batch, bad_id=999) if len(classifications) <= 2 else labels(batch)

        async def classify_retry(self, batch, identity, validation_code):
            classifications.append("retry")
            return labels(batch, bad_id=999) if len(classifications) <= 2 else labels(batch)

        async def consider(self, facts):
            return one_consideration(facts)

    monkeypatch.setattr(audio, "make_providers", lambda settings: (Transcriber(), Summarizer()))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=(
                shutil.which("google-chrome") or shutil.which("chromium")
                or "/Applications/Dia.app/Contents/MacOS/Dia"))
            page = browser.new_page(viewport={"width": 1280, "height": 950}, timezone_id="Asia/Nicosia")
            errors = []
            save_payloads = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("request", lambda request: save_payloads.append(request.post_data_json)
                    if request.method == "POST" and request.url.endswith("/clinical/visit-capture/save") else None)
            page.add_init_script("""
                window._holdMic=false;window._releaseMic=null;window._audioStarts=0;window._trackStops=0;
                window._downloadClicks=0;
                URL.createObjectURL=()=>"blob:synthetic-visit-recording";URL.revokeObjectURL=()=>{};
                HTMLAnchorElement.prototype.click=function(){window._downloadClicks++;};
                Object.defineProperty(navigator, 'mediaDevices', {configurable:true, value:{
                  getUserMedia: async()=>{if(window._holdMic)await new Promise(resolve=>{window._releaseMic=resolve;});
                    return {getTracks:()=>[{stop(){window._trackStops++;}}]};}
                }});
                window.MediaRecorder=class {
                  static isTypeSupported(type){return type==='audio/webm;codecs=opus'}
                  constructor(stream,opts){this.state='inactive';this.mimeType=opts.mimeType}
                  start(){this.state='recording';window._audioStarts++;}
                  stop(){this.state='inactive';
                    const bytes=new Uint8Array(132);bytes.set([0x1a,0x45,0xdf,0xa3]);
                    this.ondataavailable({data:new Blob([bytes],{type:'audio/webm'})});this.onstop();}
                };
            """)
            page.goto(f"http://localhost:{port}/static/cockpit/index.html")
            page.locator("#ceGlobalKey").click()
            page.locator("#ceGlobalKeyInput").fill(KEY)
            page.locator("#ceGlobalLoginForm button[type=submit]").click()
            expect(page.locator("#ceGlobalAuthStatus")).to_contain_text("Συνδέθηκες")
            page.keyboard.press("Escape")
            page.locator("#captureTestPatient").click()
            expect(page.locator("#captureOpen")).to_be_visible()
            page.locator("#captureOpen").click()
            expect(page.locator("#captureAudioStatus")).to_contain_text("Έτοιμο")
            expect(page.locator("#captureAudioStart")).to_be_disabled()
            expect(page.locator("#captureWebAudio")).to_contain_text("ημερομηνία γέννησης")
            expect(page.locator("#captureWebAudio")).to_contain_text("30 ημέρες")
            page.locator("#capturePatientInformed").check()
            page.evaluate("window._holdMic=true")
            page.locator("#captureAudioStart").click()
            expect(page.locator("#captureAudioStatus")).to_contain_text("Άνοιγμα μικροφώνου")
            page.locator("#capturePatientInformed").uncheck()
            page.evaluate("window._holdMic=false;window._releaseMic()")
            expect(page.locator("#captureAudioStatus")).to_contain_text("αποσύρθηκε")
            assert page.evaluate("window._audioStarts") == 0
            assert page.evaluate("window._trackStops") >= 1
            page.locator("#capturePatientInformed").check()
            page.locator("#captureAudioStart").click()
            expect(page.locator("#captureAudioStatus")).to_contain_text("Ηχογράφηση")
            page.locator("#capturePatientInformed").uncheck()
            expect(page.locator("#captureAudioStatus")).to_contain_text("απορρίφθηκαν")
            page.locator("#capturePatientInformed").check()
            page.locator("#captureAudioStart").click()
            expect(page.locator("#captureAudioStatus")).to_contain_text("Ηχογράφηση")
            page.locator("#captureAudioStop").click()
            expect(page.locator("#captureAudioStatus")).to_contain_text("μεταγραφή διατηρήθηκε", timeout=15000)
            expect(page.locator("#captureAudioTranscriptTab")).to_be_visible()
            expect(page.locator("#captureAudioProcessedTab")).to_be_visible()
            expect(page.locator("#captureAudioTranscriptText")).to_contain_text(TRANSCRIPT)
            expect(page.locator("#captureSave")).to_be_hidden()
            expect(page.locator("#captureAudioRetryProcess")).to_be_visible()
            expect(page.locator("#captureAudioDownload")).to_be_visible()
            page.locator("#captureAudioProcessedTab").click()
            expect(page.locator("#captureAudioProcessedUnavailable")).to_be_visible()
            expect(page.locator("#captureAudioProcessedUnavailable")).to_contain_text("δεν είναι διαθέσιμο")
            page.locator("#captureAudioTranscriptTab").click()
            page.locator("#captureAudioDownload").click()
            assert page.evaluate("window._downloadClicks") == 1
            page.locator("#captureAudioRetryProcess").click()
            expect(page.locator("#captureAudioStatus")).to_contain_text(
                "έτοιμα για ξεχωριστό έλεγχο", timeout=15000)
            expect(page.locator("#captureAudioDownload")).to_be_hidden()
            expect(page.locator("#captureAudioTranscriptText")).to_contain_text(TRANSCRIPT)
            expect(page.locator("#captureAudioProcessedTab")).to_have_attribute("aria-pressed", "true")
            expect(page.locator("#captureSave")).to_be_disabled()
            expect(page.locator("#captureReview .capture-audio-compact-section")).to_have_count(4)
            expect(page.locator("#captureReview section[data-field=findings] textarea")).to_have_value("DEXA T-score -2.8 στο ισχίο.")
            expect(page.locator("#captureReview section[data-field=recommendations] textarea")).to_have_value(
                "Ξεκινά alendronate 70 mg εβδομαδιαίως.")
            expect(page.locator("#captureReview section[data-field=patient_preferences] textarea")).to_have_value(
                "Η ασθενής προτιμά να περιμένει.")
            expect(page.locator("#captureReview section[data-field=uncertainties]")).to_have_count(0)
            expect(page.locator("#captureReview section[data-field=medications]")).to_have_count(0)
            expect(page.locator("#captureReview").get_by_role("button", name="Προσθήκη ·", exact=False)).to_have_count(0)
            expect(page.locator("#captureReview").get_by_role("button", name="Αφαίρεση", exact=True)).to_have_count(0)
            expect(page.get_by_role("button", name="Επιβεβαιώνω ότι ολοκληρώθηκε", exact=True)).to_have_count(0)
            expect(page.get_by_role("button", name="Μεταφορά στη σωστή κλινική ενότητα", exact=True)).to_have_count(0)
            attention = page.locator(".capture-audio-focused details")
            expect(attention).not_to_have_attribute("open", "")
            expect(attention.locator("summary")).to_contain_text("3 σημείων")
            expect(attention.locator("li").first).not_to_be_visible()
            assert page.locator("#captureAudioIdentity").count() == 1
            suggestions = page.locator("#captureReview .capture-audio-suggestions")
            expect(suggestions.locator("textarea")).to_have_value("Έλεγχος ανοχής της αναφερόμενης θεραπείας.")
            expect(suggestions.locator("details")).not_to_have_attribute("open", "")
            ai_considerations = page.locator("#captureReview .capture-ai-considerations")
            expect(ai_considerations).to_contain_text("Κλινικές σκέψεις AI από τα συμφραζόμενα")
            expect(ai_considerations).to_contain_text("δεν ειπώθηκαν στην επίσκεψη")
            expect(ai_considerations).to_contain_text("συνολική επανεκτίμηση του καταγματικού κινδύνου")
            expect(ai_considerations).to_contain_text("Βεβαιότητα AI: Μέτρια")
            expect(ai_considerations.locator("details")).not_to_have_attribute("open", "")
            expect(ai_considerations.locator("button")).to_have_count(0)
            expect(page.locator("#captureReview section[data-field=follow_up]")).to_have_count(0)
            suggestions.get_by_role("button", name="Προσθήκη στο πλάνο", exact=True).click()
            expect(page.locator("#captureReview .capture-audio-suggestions")).to_have_count(0)
            expect(page.locator("#captureReview section[data-field=follow_up] textarea")).to_have_value(
                "Έλεγχος ανοχής της αναφερόμενης θεραπείας.")
            expect(page.locator("#captureAudioStatus")).not_to_contain_text(TRANSCRIPT)
            expect(page.locator("#captureAudioStatus")).to_contain_text("στον server")
            finding = page.locator("#captureReview section[data-field=findings] textarea")
            finding.fill("DEXA T-score -2.7 στο ισχίο.")
            page.locator("#captureAudioIdentity").check()
            finding.fill("DEXA T-score -2.6 στο ισχίο.")
            expect(page.locator("#captureAudioIdentity")).not_to_be_checked()
            expect(page.locator("#captureSave")).to_be_disabled()
            page.locator("#captureAudioIdentity").check()
            try:
                expect(page.locator("#captureSave")).to_be_enabled()
            except AssertionError as exc:
                raise AssertionError(page.locator("#captureStatus").inner_text()) from exc
            page.locator("#captureSave").click()
            expect(page.locator("#captureComposer")).to_be_hidden(timeout=15000)
            expect(page.locator("#patientChart")).to_contain_text("DEXA T-score -2.6")
            assert save_payloads[-1]["candidate"]["follow_up"][0]["text"] == "Έλεγχος ανοχής της αναφερόμενης θεραπείας."
            assert "clinician_edit" in save_payloads[-1]["candidate"]["follow_up"][0]["provenance"]
            assert "συνολική επανεκτίμηση του καταγματικού κινδύνου" not in str(save_payloads[-1])
            assert TRANSCRIPT not in str(save_payloads[-1])
            assert not errors, errors
            assert len(transcribed) == 1  # transcript retry never retranscribes the retained audio
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
