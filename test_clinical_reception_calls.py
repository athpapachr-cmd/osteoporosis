"""Offline tests of the scoped Reception call projection and explicit handling."""
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

import clinical_reception_calls as calls


def source_snapshot(rows, *, count=None, coverage="available_call_log_max_200"):
    source_rows = [{**row, "handled": row.get("handled", False),
                    "todo_id": row.get("todo_id", "td_" + row["call_id"] if row.get("call_id") else None)}
                   for row in rows]
    return {"calls": source_rows, "total": len(source_rows),
            "unhandled_count": len(source_rows) if count is None and coverage != "ambiguous_call_identity" else count,
            "coverage": coverage}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("CLINICAL_DATA_KEY", "local-test-clinical-key")
    monkeypatch.setenv("RECEPTION_CALLS_READ_ENABLED", "1")
    monkeypatch.setenv("RECEPTION_CALLS_SOURCE_KEY", "local-test-reception-key")
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
    monkeypatch.delenv("RECEPTION_CALLS_SOURCE_KEY")
    assert client.get("/clinical/reception-calls/peek",
                      headers={"X-Clinical-Key": "local-test-clinical-key"}).status_code == 503


def test_source_mapping_minimizes_response_and_uses_real_coverage(client, monkeypatch):
    async def fake_source(key):
        assert key == "local-test-reception-key"
        return source_snapshot([
            {"call_id": "RM-old", "logged_at": "2026-10-24T09:00:00+03:00",
             "first_name": "Δοκιμαστικός", "last_name": "Καλών",
             "call_narrative_status": "complete", "call_narrative": "Παλαιότερη δοκιμαστική κλήση",
             "transcript": "MUST-NOT-LEAK"},
            {"call_id": "RM-new", "logged_at": "2026-10-25T07:00:00+02:00",
             "first_name": "Δοκιμή", "last_name": "Ασθενής",
             "phone": "0000000000", "type": "message", "caller_wants_callback": True,
             "call_narrative_status": "complete", "call_narrative": "Ζήτησε τηλεφώνημα",
             "runtime_observability": {"secret": "MUST-NOT-LEAK"}},
        ])
    monkeypatch.setattr(calls, "_fetch_calls", fake_source)
    response = client.get("/clinical/reception-calls/peek",
                          headers={"X-Clinical-Key": "local-test-clinical-key"})
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "private, no-store"
    assert response.json()["is_unread_count"] is False
    assert response.json()["unhandled_count"] == 2
    assert response.json()["coverage"] == "available_call_log_max_200"
    rows = response.json()["calls"]
    assert [row["call_id"] for row in rows] == ["RM-new", "RM-old"]
    assert rows[0]["summary"] == "Ζήτησε τηλεφώνημα"
    assert "requires_action" not in rows[0]
    assert rows[0]["kind"] == "message"
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
        return source_snapshot([{
            "call_id": "RM-123", "logged_at": "2026-10-25T07:00:00+02:00",
            "call_narrative_status": "pending", "call_narrative": "NOT-YET-VALID",
            "display_message": "Έγινε κλήση", "type": "information",
        }])
    monkeypatch.setattr(calls, "_fetch_calls", source)
    response = client.get("/clinical/reception-calls/peek",
                          headers={"X-Clinical-Key": "local-test-clinical-key"})
    assert response.status_code == 200
    row = response.json()["calls"][0]
    assert row["summary_pending"] is True
    assert row["summary"] == "Έγινε κλήση"
    assert "NOT-YET-VALID" not in response.text


def test_real_call_without_call_id_stays_visible_with_null_identity(client, monkeypatch):
    """Reception records can lack a usable call ID; do not omit actual calls."""
    async def source(_):
        return source_snapshot([
            {"call_id": "", "logged_at": "2026-10-25T08:00:00+02:00",
             "caller_name": "Χωρίς αναγνωριστικό", "phone": "0000000001",
             "display_message": "Πραγματική συνθετική κλήση"},
            {"logged_at": "2026-10-25T09:00:00+02:00",
             "first_name": "Δεύτερη", "last_name": "Κλήση",
             "type": "message", "message_clean": "Απλό μήνυμα"},
            {"call_id": "RM-older", "logged_at": "2026-10-25T07:00:00+02:00",
             "display_message": "Παλαιότερη κλήση"},
        ])
    monkeypatch.setattr(calls, "_fetch_calls", source)
    response = client.get("/clinical/reception-calls/peek",
                          headers={"X-Clinical-Key": "local-test-clinical-key"})
    assert response.status_code == 200
    rows = response.json()["calls"]
    assert len(rows) == 3
    assert [r["call_id"] for r in rows] == [None, None, "RM-older"]
    assert rows[0]["name"] == "Δεύτερη Κλήση"
    assert rows[1]["summary"] == "Πραγματική συνθετική κλήση"
    assert all("requires_action" not in r for r in rows)
    assert "0000000001" in response.text


def test_message_or_callback_uses_only_real_handled_state(client, monkeypatch):
    """The call type never overrides the Reception ledger."""
    async def source(_):
        return source_snapshot([{
            "call_id": "RM-already-handled",
            "logged_at": "2026-10-25T09:00:00+02:00",
            "type": "message", "is_urgent": True,
            "caller_wants_callback": True,
            "native_message_operation_id": "synthetic-operation",
            "handled": True,
            "display_message": "Η κατάσταση διεκπεραίωσης δεν είναι γνωστή.",
        }], count=0)
    monkeypatch.setattr(calls, "_fetch_calls", source)
    response = client.get("/clinical/reception-calls/peek",
                          headers={"X-Clinical-Key": "local-test-clinical-key"})
    assert response.status_code == 200
    row = response.json()["calls"][0]
    assert row["kind"] == "message"
    assert row["handled"] is True
    assert response.json()["unhandled_count"] == 0
    assert "requires_action" not in row
    assert "is_urgent" not in row
    assert "native_message_operation_id" not in row


def test_anonymous_rows_still_obey_four_call_cap(client, monkeypatch):
    async def source(_):
        return source_snapshot([{
            "logged_at": f"2026-10-25T{hour:02d}:00:00+02:00",
            "display_message": f"Κλήση {hour}",
        } for hour in range(5, 11)])
    monkeypatch.setattr(calls, "_fetch_calls", source)
    response = client.get("/clinical/reception-calls/peek",
                          headers={"X-Clinical-Key": "local-test-clinical-key"})
    assert response.status_code == 200
    rows = response.json()["calls"]
    assert len(rows) == 4
    assert response.json()["unhandled_count"] == 6
    assert [r["summary"] for r in rows] == ["Κλήση 10", "Κλήση 9", "Κλήση 8", "Κλήση 7"]
    assert all(r["call_id"] is None for r in rows)


def test_ambiguous_source_hides_count_and_completion_identity(client, monkeypatch):
    async def source(_):
        return source_snapshot([{
            "logged_at": "2026-10-25T09:00:00+02:00", "phone": "0000000001",
            "display_message": "Συνθετική ασαφής κλήση", "handled": None,
            "todo_id": None,
        }], coverage="ambiguous_call_identity")
    monkeypatch.setattr(calls, "_fetch_calls", source)
    body = client.get("/clinical/reception-calls/peek",
                      headers={"X-Clinical-Key": "local-test-clinical-key"}).json()
    assert body["unhandled_count"] is None
    assert body["calls"][0]["handled"] is None
    assert body["calls"][0]["todo_id"] is None


def test_handled_write_requires_explicit_action_and_source_confirmation(client, monkeypatch):
    sent = []
    async def set_handled(key, todo_id):
        sent.append((key, todo_id))
    monkeypatch.setattr(calls, "_set_handled", set_handled)
    assert client.post("/clinical/reception-calls/handled",
                       json={"todo_id": "td_RM-test", "handled": True}).status_code == 401
    assert client.post("/clinical/reception-calls/handled",
                       headers={"X-Clinical-Key": "local-test-clinical-key"},
                       json={"todo_id": "td_RM-test", "handled": False}).status_code == 422
    assert sent == []
    response = client.post("/clinical/reception-calls/handled",
                           headers={"X-Clinical-Key": "local-test-clinical-key"},
                           json={"todo_id": "td_RM-test", "handled": True})
    assert response.status_code == 200
    assert response.json() == {"handled": True}
    assert sent == [("local-test-reception-key", "td_RM-test")]
