"""Offline tests of a real-useful but default-OFF Reception call read projection."""
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

import clinical_reception_calls as calls


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("CLINICAL_DATA_KEY", "local-test-clinical-key")
    monkeypatch.setenv("RECEPTION_CALLS_READ_ENABLED", "1")
    monkeypatch.setenv("RECEPTION_CALLS_DASHBOARD_KEY", "local-test-reception-key")
    app = FastAPI()
    app.include_router(calls.build_reception_calls_router())
    return TestClient(app)


def test_requires_existing_clinical_key(client):
    assert client.get("/clinical/reception-calls/peek").status_code == 401
    assert client.get("/clinical/reception-calls/peek", headers={"X-Clinical-Key": "wrong"}).status_code == 401


def test_default_off_and_missing_source_503(client, monkeypatch):
    monkeypatch.delenv("RECEPTION_CALLS_READ_ENABLED")
    response = client.get("/clinical/reception-calls/peek", headers={"X-Clinical-Key": "local-test-clinical-key"})
    assert response.status_code == 503
    monkeypatch.setenv("RECEPTION_CALLS_READ_ENABLED", "1")
    monkeypatch.delenv("RECEPTION_CALLS_DASHBOARD_KEY")
    assert client.get("/clinical/reception-calls/peek",
                      headers={"X-Clinical-Key": "local-test-clinical-key"}).status_code == 503


def test_source_mapping_minimizes_response_and_uses_real_coverage(client, monkeypatch):
    async def fake_source(key):
        assert key == "local-test-reception-key"
        return {"calls": [
            {"call_id": "RM-old", "logged_at": "2026-10-24T09:00:00+03:00",
             "first_name": "Δοκιμαστικός", "last_name": "Καλών",
             "call_narrative_status": "complete", "call_narrative": "Παλαιότερη δοκιμαστική κλήση",
             "transcript": "MUST-NOT-LEAK"},
            {"call_id": "RM-new", "logged_at": "2026-10-25T07:00:00+02:00",
             "first_name": "Δοκιμή", "last_name": "Ασθενής",
             "phone": "0000000000", "type": "message", "caller_wants_callback": True,
             "call_narrative_status": "complete", "call_narrative": "Ζήτησε τηλεφώνημα",
             "runtime_observability": {"secret": "MUST-NOT-LEAK"}},
        ]}
    monkeypatch.setattr(calls, "_fetch_calls", fake_source)
    response = client.get("/clinical/reception-calls/peek",
                          headers={"X-Clinical-Key": "local-test-clinical-key"})
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "private, no-store"
    assert response.json()["is_unread_count"] is False
    rows = response.json()["calls"]
    assert [row["call_id"] for row in rows] == ["RM-new", "RM-old"]
    assert rows[0]["summary"] == "Ζήτησε τηλεφώνημα"
    assert rows[0]["requires_action"] is True
    assert "MUST-NOT-LEAK" not in response.text
    assert "transcript" not in response.text


def test_invalid_source_does_not_silently_become_empty(client, monkeypatch):
    async def invalid(_): return {"not_calls": []}
    monkeypatch.setattr(calls, "_fetch_calls", invalid)
    response = client.get("/clinical/reception-calls/peek",
                          headers={"X-Clinical-Key": "local-test-clinical-key"})
    assert response.status_code == 503


def test_pending_narrative_is_not_reported_as_finished(client, monkeypatch):
    async def source(_):
        return {"calls": [{
            "call_id": "RM-123", "logged_at": "2026-10-25T07:00:00+02:00",
            "call_narrative_status": "pending", "call_narrative": "NOT-YET-VALID",
            "display_message": "Έγινε κλήση", "type": "information",
        }]}
    monkeypatch.setattr(calls, "_fetch_calls", source)
    response = client.get("/clinical/reception-calls/peek",
                          headers={"X-Clinical-Key": "local-test-clinical-key"})
    assert response.status_code == 200
    row = response.json()["calls"][0]
    assert row["summary_pending"] is True
    assert row["summary"] == "Έγινε κλήση"
    assert "NOT-YET-VALID" not in response.text
