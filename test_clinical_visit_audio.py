"""Synthetic web-audio safety checks; no real recording or provider request."""
from __future__ import annotations

import io
import logging
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


def draft(client, context_id):
    return client.post("/clinical/visit-capture/audio/draft", headers={
        **HEADERS, "X-Visit-Context": context_id,
        "X-Encounter-Date": "2026-10-27", "Content-Type": "audio/webm",
        "X-Visit-Retention-Acknowledged": "standard",
    }, content=WEBM)


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


def test_provider_multipart_and_private_responses_contract(monkeypatch):
    enabled(monkeypatch)
    settings = audio.AudioSettings.from_env()
    requests = []

    async def receive(request):
        body = await request.aread()
        requests.append((request, body))
        if request.url.path.endswith("/transcriptions"):
            return httpx.Response(200, json={"text": TRANSCRIPT})
        return httpx.Response(200, json={"status": "completed", "output": [{
            "content": [{"type": "output_text", "text": __import__("json").dumps(labels([(0, "Επανέλεγχος")]))}]
        }]})

    real = httpx.AsyncClient
    monkeypatch.setattr(audio.httpx, "AsyncClient", lambda **kwargs: real(
        transport=httpx.MockTransport(receive), **kwargs))

    import asyncio
    async def run():
        assert await audio.OpenAITranscriber(settings).transcribe(io.BytesIO(WEBM), "audio/webm") == TRANSCRIPT
        assert (await audio.OpenAISummarizer(settings).classify(
            [(0, "Επανέλεγχος")], {"full_name": "Synthetic"}))["facts"][0]["kind"] == "reason"
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

    monkeypatch.setattr(audio, "make_providers", lambda settings: (Transcriber(), Summarizer()))
    result = draft(client, context["context_id"])
    assert result.status_code == 200, result.text
    body = result.json()
    candidate = body["candidate"]
    assert body["patient_id"] == "SYN-WEB"
    assert "transcript" not in body
    assert candidate["reason_for_visit"]["text"] == "Επανέλεγχος οστεοπόρωσης."
    assert candidate["findings"][0]["text"] == "DEXA T-score -2.8 στο ισχίο."
    assert candidate["decisions"] == [] and candidate["medications"] == []
    assert candidate["pending"] == []
    assert candidate["recommendations"][0]["text"] == "Ξεκινά alendronate 70 mg εβδομαδιαίως."
    assert candidate["patient_preferences"][0]["text"] == "Η ασθενής προτιμά να περιμένει."
    assert candidate["uncertainties"] == []
    assert candidate["follow_up"] == []
    assert body["suggested_next_checks"][0]["text"] == "Έλεγχος ανοχής της αναφερόμενης θεραπείας."
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
    assert draft(client, context["context_id"]).status_code == 409

    fresh = _context(client, "SYN-WEB")
    class StableTranscriber:
        async def transcribe(self, file, mime):
            return TRANSCRIPT
    class BadSummarizer:
        async def classify(self, batch, identity):
            return labels(batch, bad_id=999)
    monkeypatch.setattr(audio, "make_providers", lambda settings: (StableTranscriber(), BadSummarizer()))
    failed = draft(client, fresh["context_id"])
    assert failed.status_code == 502
    assert TRANSCRIPT not in failed.text


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
        assert response.status_code == 502
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
        assert response.status_code == 502
        assert TRANSCRIPT not in response.text


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

    class Transcriber:
        async def transcribe(self, file, mime):
            transcribed.append(True)
            assert file.read(4) == WEBM[:4]
            return TRANSCRIPT

    class Summarizer:
        async def classify(self, batch, identity):
            return labels(batch)

    monkeypatch.setattr(audio, "make_providers", lambda settings: (Transcriber(), Summarizer()))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=shutil.which("google-chrome") or shutil.which("chromium"))
            page = browser.new_page(viewport={"width": 1280, "height": 950}, timezone_id="Asia/Nicosia")
            errors = []
            save_payloads = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("request", lambda request: save_payloads.append(request.post_data_json)
                    if request.method == "POST" and request.url.endswith("/clinical/visit-capture/save") else None)
            page.add_init_script("""
                window._holdMic=false;window._releaseMic=null;window._audioStarts=0;window._trackStops=0;
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
            expect(page.locator("#captureAudioStatus")).to_contain_text("απορρίφθηκε")
            page.locator("#capturePatientInformed").check()
            page.locator("#captureAudioStart").click()
            expect(page.locator("#captureAudioStatus")).to_contain_text("Ηχογράφηση")
            page.locator("#captureAudioStop").click()
            expect(page.locator("#captureAudioStatus")).to_contain_text("έτοιμο για έλεγχο", timeout=15000)
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
            assert not errors, errors
            assert len(transcribed) == 1  # withdrawn recording never reaches the provider
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
