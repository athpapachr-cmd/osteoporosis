"""Synthetic web-audio safety checks; no real recording or provider request."""
from __future__ import annotations

import io
from datetime import datetime, timezone

import httpx
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
    "Συζητήθηκε alendronate 70 mg εβδομαδιαίως. "
    "Η ασθενής προτιμά να περιμένει."
)


def enabled(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-provider-key")
    monkeypatch.setenv("VISIT_AUDIO_OPENAI_PHI_APPROVED", "1")
    monkeypatch.setenv("VISIT_AUDIO_RETENTION_MODE", "standard")
    monkeypatch.setenv("VISIT_AUDIO_OPENAI_PROJECT_ID", "proj_synthetic_standard")


def labels(batch, *, identity="consistent", bad_id=None):
    result = {name: [] for name in audio.CATEGORIES}
    result["identity_state"] = identity
    for identifier, text in batch:
        if text.startswith("Επανέλεγχος"):
            result["reason"].append(identifier)
        elif text.startswith("DEXA"):
            result["finding"].append(identifier)
        elif text.startswith("Συζητήθηκε"):
            result["recommendation"].append(identifier)
        else:
            result["preference"].append(identifier)
    if bad_id is not None:
        result["finding"].append(bad_id)
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
            [(0, "Επανέλεγχος")], {"full_name": "Synthetic"}))["reason"] == [0]
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
    assert "tools" not in response_json


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
    assert {item["text"] for item in candidate["uncertainties"]} == {
        "Συζητήθηκε alendronate 70 mg εβδομαδιαίως.",
        "Η ασθενής προτιμά να περιμένει.",
    }
    assert {item["hint"] for item in body["review_units"]} >= {"finding", "recommendation", "preference"}
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


def test_temp_audio_closed_on_failed_provider_and_oversize(monkeypatch):
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
    response = draft(client, context["context_id"])
    assert response.status_code == 502 and "secret patient words" not in response.text
    assert files and all(file.closed for file in files)
    monkeypatch.setattr(audio, "MAX_AUDIO_BYTES", 100)
    assert draft(client, context["context_id"]).status_code == 413
    assert all(file.closed for file in files)


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
            result = {name: [] for name in audio.CATEGORIES}
            result["identity_state"] = "consistent"
            result["reason"].append(batch[0][0])
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
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.add_init_script("""
                Object.defineProperty(navigator, 'mediaDevices', {configurable:true, value:{
                  getUserMedia: async()=>({getTracks:()=>[{stop(){}}]})
                }});
                window.MediaRecorder=class {
                  static isTypeSupported(type){return type==='audio/webm;codecs=opus'}
                  constructor(stream,opts){this.state='inactive';this.mimeType=opts.mimeType}
                  start(){this.state='recording'}
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
            expect(page.locator("#captureReview textarea").nth(1)).to_have_value("DEXA T-score -2.8 στο ισχίο.")
            expect(page.locator("#captureAudioStatus")).not_to_contain_text(TRANSCRIPT)
            expect(page.locator("#captureAudioStatus")).to_contain_text("στον server")
            page.locator("#captureAudioIdentity").check()
            try:
                expect(page.locator("#captureSave")).to_be_enabled()
            except AssertionError as exc:
                raise AssertionError(page.locator("#captureStatus").inner_text()) from exc
            page.locator("#captureSave").click()
            expect(page.locator("#captureComposer")).to_be_hidden(timeout=15000)
            expect(page.locator("#patientChart")).to_contain_text("DEXA T-score -2.8")
            assert not errors, errors
            assert len(transcribed) == 1  # withdrawn recording never reaches the provider
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
