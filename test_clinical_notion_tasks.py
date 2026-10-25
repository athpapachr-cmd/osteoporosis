"""Synthetic contract tests. No Notion/OpenAI call or real task content."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from uuid import uuid4
import asyncio
import json

import httpx

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

import clinical_notion_tasks as tasks

TASK_ID = "00000000-0000-4000-8000-000000000001"
REVISION = "2026-10-25T08:00:00.000Z"


def source_page(status="Not started"):
    return {
        "object": "page", "id": TASK_ID, "last_edited_time": REVISION,
        "parent": {"type": "data_source_id", "data_source_id": tasks.DATA_SOURCE_ID},
        "url": "https://www.notion.so/00000000000040008000000000000001",
        "properties": {
            "Task name": {"type": "title", "title": [{"plain_text": "Συνθετική εργασία"}]},
            "Status": {"type": "status", "status": {"name": status}},
            "Due": {"type": "date", "date": {"start": "2026-10-30"}},
            "Κατηγορία": {"type": "select", "select": {"name": "Ιατρείο"}},
            "Σημειώσεις": {"id": "notes%3Aid", "type": "rich_text", "rich_text": []},
        },
    }


def app_and_calls(monkeypatch, tmp_path):
    engine = create_engine("sqlite:///" + str(tmp_path / "attempts.sqlite"))
    monkeypatch.setenv("CLINICAL_DATA_KEY", "synthetic-key")
    monkeypatch.setattr(tasks, "_settings", lambda _engine, voice=False: "synthetic-token")
    page = source_page()
    calls = []

    async def notion(_token, method, path, *, payload=None):
        calls.append((method, path, deepcopy(payload)))
        if method == "GET" and path.startswith("/pages/") and "/properties/" in path:
            return {"object": "list", "has_more": False, "results": [{
                "type": "rich_text", "rich_text": {"type": "text",
                    "text": {"content": "Υπάρχουσα σημείωση"}, "plain_text": "Υπάρχουσα σημείωση"}}]}
        if method == "GET":
            return deepcopy(page)
        if method == "POST" and path.endswith("/query"):
            return {"results": [deepcopy(page)]}
        if method == "PATCH":
            page["last_edited_time"] = "2026-10-25T08:01:00.000Z"
            return deepcopy(page)
        if method == "POST" and path == "/pages":
            followup = source_page()
            followup["id"] = "00000000-0000-4000-8000-000000000002"
            return followup
        raise AssertionError((method, path))

    monkeypatch.setattr(tasks, "_notion", notion)
    app = FastAPI()
    app.include_router(tasks.build_notion_tasks_router(engine))
    client = TestClient(app)
    return client, engine, page, calls


def confirm_payload(**overrides):
    result = {"attempt_id": str(uuid4()), "task_id": TASK_ID, "revision": REVISION,
              "mark_done": True, "completion_note": "Ολοκληρώθηκε συνθετικά",
              "followup_title": "Συνθετικό follow-up", "followup_notes": "Συνθετική ενέργεια"}
    result.update(overrides)
    return result


def test_peek_auth_and_confirm_preserves_notes(monkeypatch, tmp_path):
    client, _engine, _page, calls = app_and_calls(monkeypatch, tmp_path)
    assert client.get("/clinical/notion-tasks/peek").status_code == 401
    headers = {"X-Clinical-Key": "synthetic-key"}
    peek = client.get("/clinical/notion-tasks/peek", headers=headers)
    assert peek.status_code == 200
    assert peek.headers["cache-control"] == "private, no-store"
    assert len(peek.json()["tasks"]) == 1
    assert peek.json()["tasks"][0]["url"].startswith("https://www.notion.so/")
    assert calls[-1][2]["page_size"] == 4
    result = client.post("/clinical/notion-tasks/confirm", headers=headers,
                         json=confirm_payload())
    assert result.status_code == 200, result.text
    assert result.json()["followup_created"] is True
    patches = [payload for method, _path, payload in calls if method == "PATCH"]
    creates = [payload for method, path, payload in calls if method == "POST" and path == "/pages"]
    assert len(patches) == len(creates) == 1
    assert patches[0]["properties"]["Status"]["status"]["name"] == "Done"
    rich = patches[0]["properties"]["Σημειώσεις"]["rich_text"]
    assert rich[0]["text"]["content"] == "Υπάρχουσα σημείωση"
    assert rich[1]["text"]["content"] == "\nΟλοκληρώθηκε συνθετικά"
    assert any("/properties/notes%3Aid?" in path for method, path, _ in calls if method == "GET")
    assert not any("%253A" in path for _, path, _ in calls)
    assert creates[0]["parent"]["data_source_id"] == tasks.DATA_SOURCE_ID


def test_replay_and_competing_attempt_are_rejected(monkeypatch, tmp_path):
    client, engine, _page, calls = app_and_calls(monkeypatch, tmp_path)
    payload = confirm_payload()
    body = tasks.ConfirmTask.model_validate(payload)
    tasks._ensure_guard(engine)
    def claim(attempt):
        try:
            tasks._claim(engine, tasks.ConfirmTask.model_validate({**payload, "attempt_id": attempt}))
            return 200
        except HTTPException as exc:
            return exc.status_code
    with ThreadPoolExecutor(max_workers=2) as pool:
        verdicts = list(pool.map(claim, [str(body.attempt_id), str(uuid4())]))
    assert sorted(verdicts) == [200, 409]
    headers = {"X-Clinical-Key": "synthetic-key"}
    assert client.post("/clinical/notion-tasks/confirm", headers=headers, json=payload).status_code == 409
    assert not any(method in {"PATCH"} or path == "/pages" for method, path, _ in calls)


def test_unknown_patch_freezes_attempt_and_never_creates_followup(monkeypatch, tmp_path):
    client, _engine, _page, calls = app_and_calls(monkeypatch, tmp_path)
    original = tasks._notion
    async def uncertain(token, method, path, *, payload=None):
        if method == "PATCH":
            calls.append((method, path, payload))
            raise HTTPException(503, "synthetic timeout")
        return await original(token, method, path, payload=payload)
    monkeypatch.setattr(tasks, "_notion", uncertain)
    headers = {"X-Clinical-Key": "synthetic-key"}
    body = confirm_payload()
    first = client.post("/clinical/notion-tasks/confirm", headers=headers, json=body)
    assert first.status_code == 503
    second = client.post("/clinical/notion-tasks/confirm", headers=headers,
                         json=confirm_payload())
    assert second.status_code == 409
    assert len([c for c in calls if c[0] == "PATCH"]) == 1
    assert not any(c[1] == "/pages" for c in calls)


def test_notes_too_large_or_parent_mismatch_fail_before_write(monkeypatch, tmp_path):
    client, _engine, page, calls = app_and_calls(monkeypatch, tmp_path)
    page["parent"]["data_source_id"] = "00000000-0000-4000-8000-000000000099"
    response = client.post("/clinical/notion-tasks/confirm", headers={"X-Clinical-Key": "synthetic-key"},
                           json=confirm_payload())
    assert response.status_code == 409
    assert not any(c[0] == "PATCH" for c in calls)


def test_paginated_notes_fail_before_claim_or_write(monkeypatch, tmp_path):
    client, _engine, _page, calls = app_and_calls(monkeypatch, tmp_path)
    original = tasks._notion
    async def paginated(token, method, path, *, payload=None):
        if "/properties/" in path:
            return {"object": "list", "has_more": True, "results": []}
        return await original(token, method, path, payload=payload)
    monkeypatch.setattr(tasks, "_notion", paginated)
    headers = {"X-Clinical-Key": "synthetic-key"}
    assert client.post("/clinical/notion-tasks/confirm", headers=headers,
                       json=confirm_payload()).status_code == 409
    assert not any(c[0] == "PATCH" for c in calls)
    # A safe preflight failure did not consume this revision's one-shot guard.
    monkeypatch.setattr(tasks, "_notion", original)
    assert client.post("/clinical/notion-tasks/confirm", headers=headers,
                       json=confirm_payload()).status_code == 200


def test_followup_unknown_reports_partial_and_does_not_retry(monkeypatch, tmp_path):
    client, _engine, _page, calls = app_and_calls(monkeypatch, tmp_path)
    original = tasks._notion
    async def uncertain(token, method, path, *, payload=None):
        if method == "POST" and path == "/pages":
            calls.append((method, path, payload))
            raise HTTPException(503, "synthetic timeout")
        return await original(token, method, path, payload=payload)
    monkeypatch.setattr(tasks, "_notion", uncertain)
    result = client.post("/clinical/notion-tasks/confirm",
                         headers={"X-Clinical-Key": "synthetic-key"}, json=confirm_payload())
    assert result.status_code == 503
    assert "Original task updated" in result.json()["detail"]
    assert len([c for c in calls if c[0] == "PATCH"]) == 1
    assert len([c for c in calls if c[1] == "/pages"]) == 1


def test_voice_route_uses_current_task_and_bounded_audio(monkeypatch, tmp_path):
    client, _engine, _page, calls = app_and_calls(monkeypatch, tmp_path)
    invoked = []
    async def synthetic_proposal(audio, mime, task):
        invoked.append((len(audio), mime, task["id"]))
        return {"transcript": "Συνθετική υπαγόρευση", "suggestion": {
            "mark_done": True, "completion_note": "Συνθετική ολοκλήρωση",
            "followup_title": "", "followup_notes": ""}}
    monkeypatch.setattr(tasks, "_proposal", synthetic_proposal)
    headers = {"X-Clinical-Key": "synthetic-key", "X-Task-Id": TASK_ID,
               "X-Task-Revision": REVISION, "X-Audio-Duration-Ms": "2500",
               "Content-Type": "audio/webm"}
    sent = []
    def oversized_chunks():
        for index in range(4):
            sent.append(index)
            yield b"x" * 1_000_000
    too_large = client.post("/clinical/notion-tasks/propose", headers=headers,
                            content=oversized_chunks())
    assert too_large.status_code == 413
    assert len(sent) >= 3
    assert not invoked
    result = client.post("/clinical/notion-tasks/propose", headers=headers, content=b"x" * 100)
    assert result.status_code == 200
    assert result.json()["transcript"] == "Συνθετική υπαγόρευση"
    assert invoked == [(100, "audio/webm", TASK_ID)]
    assert not any(c[0] in {"PATCH", "POST"} and c[1] == "/pages" for c in calls)


def test_voice_provider_contract_has_no_write_tool_or_storage(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-only")
    calls = []
    class FakeClient:
        def __init__(self, **_kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_args):
            pass
        async def post(self, url, **kwargs):
            calls.append((url, kwargs))
            if url.endswith("/audio/transcriptions"):
                return httpx.Response(200, json={"text": "Ολοκληρώθηκε συνθετική εργασία"})
            return httpx.Response(200, json={"output": [{"content": [{"type": "output_text",
                "text": json.dumps({"mark_done": True, "completion_note": "Συνθετική ολοκλήρωση",
                                    "followup_title": "Συνθετικό follow-up",
                                    "followup_notes": "Συνθετικό επόμενο βήμα"})}]}]})
    monkeypatch.setattr(tasks.httpx, "AsyncClient", FakeClient)
    result = asyncio.run(tasks._proposal(b"synthetic-webm", "audio/webm", {
        "title": "Συνθετική εργασία", "id": TASK_ID})
    )
    assert result["suggestion"]["mark_done"] is True
    assert len(calls) == 2
    assert calls[0][1]["data"]["model"] == "gpt-transcribe"
    assert calls[1][1]["json"]["model"] == "gpt-6-luna"
    assert calls[1][1]["json"]["store"] is False
    assert "tools" not in calls[1][1]["json"]
