from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
import http.client
import json
import re

import pytest

import local_visit_audio.pipeline as pipeline_module
from local_visit_audio.pipeline import (
    DISPOSITIONS,
    MODEL_KEYS,
    LocalAudioPipeline,
    PipelineConfig,
    _batches,
    classify,
    risk_flags,
    segment_source_units,
)
from local_visit_audio.service import CompanionServer, CompanionState


def _config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        support_dir=tmp_path,
        ffmpeg=tmp_path / "ffmpeg",
        whisper_cli=tmp_path / "whisper-cli",
        whisper_repo=tmp_path / "whisper.cpp",
        whisper_model=tmp_path / "model.bin",
        decoder_manifest=tmp_path / "manifest.json",
    )


def _empty_model_result() -> dict[str, list[int]]:
    return {key: [] for key in MODEL_KEYS}


def test_classifier_accounts_for_every_id_and_never_generates_text(monkeypatch, tmp_path):
    units = [
        "Πόνος στον δεξιό ώμο.",
        "Δεν αναφέρει πρόσφατο τραυματισμό.",
        "T-score -2.8 και alendronate 70 mg.",
    ]

    def fake_generate(_config, _batch, _cancelled):
        result = _empty_model_result()
        result["reason_ids"] = [0]
        result["patient_history_ids"] = [1]
        # ID 2 intentionally omitted: deterministic accounting must surface it.
        return result

    monkeypatch.setattr(pipeline_module, "_ollama_generate", fake_generate)
    mapped = classify(_config(tmp_path), units)
    assert mapped == {0: {"reason"}, 1: {"patient_history"}, 2: {"unclassified"}}
    assert "digit_or_measurement" in risk_flags(units[2], mapped[2])
    assert "medication" in risk_flags(units[2], mapped[2])
    assert "unclassified" in risk_flags(units[2], mapped[2])
    assert "laterality" in risk_flags(units[0], mapped[0])
    assert "negation" in risk_flags(units[1], mapped[1])


def test_batching_never_splits_units_and_stays_below_byte_ceiling():
    units = [("Ελληνικό κείμενο " + str(i) + " ") * 40 for i in range(20)]
    batches = _batches(units)
    flattened = [identifier for batch in batches for identifier, _ in batch]
    assert flattened == list(range(len(units)))
    assert len(batches) > 1
    for batch in batches:
        assert len(pipeline_module._instruction(batch).encode("utf-8")) <= 8192


def test_whisper_segments_split_only_on_explicit_sentence_boundaries():
    units = segment_source_units([
        " Πόνος δεξιά.  Δεν αναφέρει τραυματισμό; Προτάθηκε φυσιοθεραπεία. ",
        "Follow up σε τρεις μήνες",
    ])
    assert units == [
        "Πόνος δεξιά.",
        "Δεν αναφέρει τραυματισμό;",
        "Προτάθηκε φυσιοθεραπεία.",
        "Follow up σε τρεις μήνες",
    ]


def test_mutually_inconsistent_semantic_statuses_are_explicitly_flagged():
    flags = risk_flags("Προτάθηκε και αποφασίστηκε φυσιοθεραπεία.", {
        "clinician_recommendation", "final_decision", "pending_obligation",
    })
    assert "semantic_status_conflict" in flags
    assert "semantic_status_conflict" not in risk_flags(
        "Εκκρεμεί επανέλεγχος.", {"follow_up", "pending_obligation"})


def test_pipeline_result_is_closed_exact_source_inventory(monkeypatch, tmp_path):
    config = _config(tmp_path)
    encoded = tmp_path / "audio.webm"
    encoded.write_bytes(b"synthetic")

    monkeypatch.setattr(pipeline_module, "decode_audio", lambda *_args: None)
    monkeypatch.setattr(pipeline_module, "transcribe", lambda *_args: ["T-score -2.8", "Προτάθηκε φυσιοθεραπεία"])
    monkeypatch.setattr(pipeline_module, "classify", lambda *_args: {0: {"objective_result"}, 1: {"clinician_recommendation"}})
    processor = LocalAudioPipeline(config, environment_check=lambda _config: None)
    result = processor.process(
        encoded_path=encoded,
        request_dir=tmp_path,
        local_session_id="A" * 22,
        request_id="B" * 22,
        duration_ms=20_000,
    )
    assert set(result) == {"schema_version", "protocol", "local_session_id", "request_id", "source_ref", "captured_at", "duration_ms", "items"}
    assert [item["text"] for item in result["items"]] == ["T-score -2.8", "Προτάθηκε φυσιοθεραπεία"]
    assert result["items"][0]["proposed_dispositions"] == ["objective_result"]
    assert all(set(item["proposed_dispositions"]).issubset(DISPOSITIONS) for item in result["items"])
    assert "response" not in json.dumps(result)


class _FakePipeline:
    def __init__(self, support_dir: Path):
        self.config = _config(support_dir)
        self.request_dir: Path | None = None
        self.called = 0

    def process(
        self, *, encoded_path, request_dir, local_session_id, request_id, duration_ms, cancelled
    ):
        self.called += 1
        self.request_dir = request_dir
        assert encoded_path.read_bytes() == b"synthetic-audio"
        assert not cancelled()
        (request_dir / "transcript-secret.txt").write_text("never persists", encoding="utf-8")
        return {
            "schema_version": "local_visit_audio_result_v1",
            "protocol": "local_visit_audio_v1",
            "local_session_id": local_session_id,
            "request_id": request_id,
            "source_ref": "local-audio:" + request_id,
            "captured_at": "2026-10-26T10:00:00Z",
            "duration_ms": duration_ms,
            "items": [{
                "id": 0,
                "text": "Συνθετικό κείμενο",
                "proposed_dispositions": ["reason"],
                "risk_flags": [],
                "whisper_flags": [],
            }],
        }


def test_loopback_service_rejects_cross_site_and_cleans_every_content_path(tmp_path):
    fake = _FakePipeline(tmp_path)
    server = CompanionServer(CompanionState(fake))
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", 8765, timeout=5)
        connection.request("GET", "/")
        response = connection.getresponse()
        html = response.read().decode()
        cookie = response.getheader("Set-Cookie").split(";", 1)[0]
        csrf = re.search(r'name="csrf-token" content="([^"]+)"', html).group(1)
        base_headers = {
            "Cookie": cookie,
            "Content-Type": "audio/webm;codecs=opus",
            "X-CSRF-Token": csrf,
            "X-Local-Session-Id": "A" * 22,
            "X-Request-Id": "B" * 22,
            "X-Duration-Ms": "1000",
        }

        connection.request("POST", "/api/process", body=b"synthetic-audio", headers={**base_headers, "Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"})
        denied = connection.getresponse()
        denied.read()
        assert denied.status == 403
        assert fake.called == 0
        connection.close()

        connection = http.client.HTTPConnection("127.0.0.1", 8765, timeout=5)
        connection.request("POST", "/api/process", body=b"synthetic-audio", headers={**base_headers, "Origin": "http://127.0.0.1:8765", "Sec-Fetch-Site": "same-origin"})
        accepted = connection.getresponse()
        payload = json.loads(accepted.read())
        assert accepted.status == 200
        assert payload["items"][0]["text"] == "Συνθετικό κείμενο"
        assert fake.called == 1
        assert fake.request_dir is not None and not fake.request_dir.exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class _BlockingPipeline:
    def __init__(self, support_dir: Path):
        self.config = _config(support_dir)
        self.started = Event()
        self.request_dir: Path | None = None

    def process(
        self, *, encoded_path, request_dir, local_session_id, request_id, duration_ms, cancelled
    ):
        del encoded_path, local_session_id, request_id, duration_ms
        self.request_dir = request_dir
        (request_dir / "transcript-secret.txt").write_text("never persists", encoding="utf-8")
        self.started.set()
        assert self.started.wait(timeout=2)
        for _ in range(100):
            if cancelled():
                raise pipeline_module.LocalAudioError("cancelled")
            Event().wait(0.01)
        raise AssertionError("cancellation was not delivered")


def test_loopback_cancel_is_same_session_and_cleans_before_terminal_response(tmp_path):
    fake = _BlockingPipeline(tmp_path)
    cancelled_backend = Event()
    recovered_backend = Event()
    state = CompanionState(
        fake,
        cancel_hook=cancelled_backend.set,
        recover_hook=recovered_backend.set,
    )
    server = CompanionServer(state)
    server_thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    server_thread.start()
    try:
        setup = http.client.HTTPConnection("127.0.0.1", 8765, timeout=5)
        setup.request("GET", "/")
        setup_response = setup.getresponse()
        html = setup_response.read().decode()
        cookie = setup_response.getheader("Set-Cookie").split(";", 1)[0]
        csrf = re.search(r'name="csrf-token" content="([^"]+)"', html).group(1)
        setup.close()
        headers = {
            "Cookie": cookie,
            "Origin": "http://127.0.0.1:8765",
            "Sec-Fetch-Site": "same-origin",
            "X-CSRF-Token": csrf,
            "X-Local-Session-Id": "A" * 22,
            "X-Request-Id": "B" * 22,
        }
        process_result: dict[str, object] = {}

        def run_process() -> None:
            connection = http.client.HTTPConnection("127.0.0.1", 8765, timeout=5)
            connection.request(
                "POST", "/api/process", body=b"synthetic-audio",
                headers={
                    **headers,
                    "Content-Type": "audio/webm;codecs=opus",
                    "X-Duration-Ms": "1000",
                },
            )
            response = connection.getresponse()
            process_result["status"] = response.status
            process_result["payload"] = json.loads(response.read())
            connection.close()

        process_thread = Thread(target=run_process, daemon=True)
        process_thread.start()
        assert fake.started.wait(timeout=2)

        cancel = http.client.HTTPConnection("127.0.0.1", 8765, timeout=5)
        cancel.request("POST", "/api/cancel", body=b"", headers=headers)
        cancel_response = cancel.getresponse()
        cancel_payload = json.loads(cancel_response.read())
        cancel.close()
        assert cancel_response.status == 202
        assert cancel_payload == {"code": "cancelling", "request_id": "B" * 22}

        process_thread.join(timeout=5)
        assert not process_thread.is_alive()
        assert process_result == {
            "status": 422,
            "payload": {"code": "cancelled", "request_id": "B" * 22},
        }
        assert cancelled_backend.is_set()
        assert recovered_backend.is_set()
        assert fake.request_dir is not None and not fake.request_dir.exists()
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)


def test_companion_total_processing_deadline_cancels_backend(tmp_path):
    fake = _FakePipeline(tmp_path)
    stopped = Event()
    state = CompanionState(fake, cancel_hook=stopped.set, processing_timeout=0.02)
    cancel_event, timeout_event = state.begin_job("browser", "A" * 22, "B" * 22)
    assert cancel_event.wait(timeout=1)
    assert timeout_event.is_set()
    assert stopped.is_set()
    assert state.finish_job(cancel_event) == (True, True)
