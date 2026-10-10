from __future__ import annotations

from copy import deepcopy

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from clinical_data import build_clinical_router


CLINICAL_KEY = "synthetic-clinical-key"
HEADERS = {"X-Clinical-Key": CLINICAL_KEY}


def _client(monkeypatch):
    monkeypatch.setenv("CLINICAL_DATA_KEY", CLINICAL_KEY)
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    app = FastAPI()
    app.include_router(build_clinical_router(engine))
    return TestClient(app)


def _patient(client: TestClient, patient_id: str):
    response = client.post(
        "/clinical/patients",
        headers=HEADERS,
        json={"patient_id": patient_id, "demographics": {}},
    )
    assert response.status_code == 200


def _context(client: TestClient, patient_id: str, session_id: str = "synthetic-browser-session-0001"):
    response = client.post(
        "/clinical/visit-capture/context",
        headers=HEADERS,
        json={"patient_id": patient_id, "client_session_id": session_id},
    )
    assert response.status_code == 200
    return response.json()


def _candidate():
    return {
        "schema_version": "visit_capture_candidate_v1",
        "source_identity_state": "consistent",
        "encounter_date": "2026-10-07",
        "visit_type": "Ορθοπαιδική επανεξέταση",
        "specialty": "Ορθοπαιδική",
        "reason_for_visit": {
            "text": "Μετατραυματική αυχεναλγία μετά τροχαίο",
            "provenance": ["heidi_today"],
            "verification": "source_stated",
            "certainty": "certain",
        },
        "source_bindings": [
            {"source_ref": "heidi_today", "role": "heidi_today"},
            {"source_ref": "gesy_today", "role": "gesy_today"},
            {"source_ref": "gesy_previous", "role": "gesy_previous"},
        ],
        "comparison_basis": {
            "type": "previous_gesy_visit",
            "source_ref": "gesy_previous",
        },
        "what_changed": [
            {
                "text": "Η ζάλη υποχώρησε",
                "direction": "resolved",
                "provenance": ["heidi_today", "gesy_previous"],
                "verification": "source_stated",
                "certainty": "certain",
            }
        ],
        "findings": [
            {
                "text": "Επώδυνο εύρος κίνησης αυχένα",
                "region": "Αυχενική μοίρα",
                "provenance": ["heidi_today"],
                "verification": "source_stated",
                "certainty": "certain",
            }
        ],
        "clinical_impression": [
            {
                "text": "Μυϊκή θλάση αυχενικής/θωρακικής μοίρας",
                "provenance": ["heidi_today"],
                "verification": "source_stated",
                "certainty": "certain",
            }
        ],
        "coding": {
            "gesy_coded": [
                {
                    "system": "ICD10",
                    "code": "S13.4",
                    "title": "Διάστρεμμα/διάταση αυχενικής μοίρας",
                    "provenance": ["gesy_today"],
                }
            ],
            "coding_complete": False,
        },
        "decisions": [
            {
                "text": "6 συνεδρίες φυσιοθεραπείας",
                "type": "referral",
                "responsible_role": "clinician",
                "provenance": ["heidi_today"],
                "verification": "source_stated",
                "certainty": "certain",
            }
        ],
        "medications": [
            {
                "name": "Narox",
                "action": "prescribed",
                "dose": None,
                "frequency": None,
                "route": None,
                "duration": {"value": 10, "unit": "days", "approximate": True},
                "provenance": ["heidi_today"],
                "verification": "source_stated",
                "certainty": "certain",
            }
        ],
        "pending": [
            {
                "item_type": "investigation",
                "description": "MRI αυχενικής/θωρακικής",
                "trigger": "αν δεν υπάρξει βελτίωση μετά τις 6 φυσιοθεραπείες",
                "status": "not_yet_indicated",
                "responsible_role": "clinician",
                "external_dependency": None,
                "review_date": "2026-10-30",
                "provenance": ["heidi_today"],
            }
        ],
        "next_contact": {
            "when": "2026-10-30",
            "time": "09:00",
            "purpose": "επανεκτίμηση",
            "assess": ["ανταπόκριση στη φυσιοθεραπεία", "ανάγκη MRI"],
            "provenance": ["heidi_today", "gesy_today"],
        },
        "safety_net": [
            {
                "text": "τηλέφωνο σε επιδείνωση",
                "provenance": ["heidi_today"],
                "verification": "source_stated",
                "certainty": "certain",
            }
        ],
        "uncertainties": [],
    }


def _preview(client, context_id, candidate):
    return client.post(
        "/clinical/visit-capture/preview",
        headers=HEADERS,
        json={"context_id": context_id, "candidate": candidate},
    )


def _save(client, context_id, candidate):
    return client.post(
        "/clinical/visit-capture/save",
        headers=HEADERS,
        json={"context_id": context_id, "candidate": candidate},
    )


def test_case_1_straightforward_capture_and_three_projections(monkeypatch):
    client = _client(monkeypatch)
    _patient(client, "SYN-A")
    context = _context(client, "SYN-A")

    preview = _preview(client, context["context_id"], _candidate())
    assert preview.status_code == 200
    body = preview.json()
    assert body["patient_id"] == "SYN-A"
    assert body["can_save"] is True
    assert "ΣΗΜΕΡΑ:" in body["snapshot"]
    assert "ΑΠΟΦΑΣΗ:" in body["brief"]
    assert "ΠΗΓΕΣ:" in body["detail"]

    saved = _save(client, context["context_id"], _candidate())
    assert saved.status_code == 200
    record = saved.json()
    assert record["encounter"]["status"] == "completed"
    assert record["encounter"]["patient_id"] == "SYN-A"
    assert len(record["pending"]) == 1
    payload = record["encounter"]["payload"]
    assert payload["_visit_capture_v1"]["signed"] is True
    assert "snapshot" not in payload["_visit_capture_v1"]
    assert "brief" not in payload["_visit_capture_v1"]
    assert "detail" not in payload["_visit_capture_v1"]


def test_case_2_no_safe_comparison_omits_change_and_rejects_invented_delta(monkeypatch):
    client = _client(monkeypatch)
    _patient(client, "SYN-B")
    context = _context(client, "SYN-B")
    candidate = _candidate()
    candidate["comparison_basis"] = {"type": "unavailable", "source_ref": None}
    candidate["what_changed"] = []

    assert _preview(client, context["context_id"], candidate).status_code == 200

    unsafe = deepcopy(candidate)
    unsafe["what_changed"] = [{
        "text": "Υποτίθεται ότι βελτιώθηκε",
        "direction": "improved",
        "provenance": ["heidi_today"],
        "verification": "source_stated",
        "certainty": "uncertain",
    }]
    response = _preview(client, context["context_id"], unsafe)
    assert response.status_code == 422
    assert "comparison basis" in response.json()["detail"]


def test_case_3_cross_patient_source_conflict_cannot_save(monkeypatch):
    client = _client(monkeypatch)
    _patient(client, "SYN-C")
    context = _context(client, "SYN-C")
    candidate = _candidate()
    candidate["source_identity_state"] = "conflict"
    candidate["uncertainties"] = [{
        "text": "Πιθανό περιεχόμενο άλλου ασθενούς στην πηγή",
        "provenance": ["heidi_today"],
        "verification": "source_stated",
        "certainty": "conflicting",
    }]

    preview = _preview(client, context["context_id"], candidate)
    assert preview.status_code == 200
    assert preview.json()["can_save"] is False
    assert _save(client, context["context_id"], candidate).status_code == 409


def test_case_4_stale_a_to_b_context_fails_closed(monkeypatch):
    client = _client(monkeypatch)
    _patient(client, "SYN-A")
    _patient(client, "SYN-B")
    session_id = "synthetic-browser-session-switch"
    context_a = _context(client, "SYN-A", session_id)
    context_b = _context(client, "SYN-B", session_id)

    assert _save(client, context_a["context_id"], _candidate()).status_code == 409

    saved_b = _save(client, context_b["context_id"], _candidate())
    assert saved_b.status_code == 200
    assert saved_b.json()["encounter"]["patient_id"] == "SYN-B"


def test_case_5_partial_medication_stays_partial(monkeypatch):
    client = _client(monkeypatch)
    _patient(client, "SYN-MED")
    context = _context(client, "SYN-MED")
    saved = _save(client, context["context_id"], _candidate())
    assert saved.status_code == 200

    medication = saved.json()["encounter"]["payload"]["_visit_capture_v1"]["candidate"]["medications"][0]
    assert medication["name"] == "Narox"
    assert medication["duration"] == {"value": 10.0, "unit": "days", "approximate": True}
    assert medication["dose"] is None
    assert medication["frequency"] is None
    assert medication["route"] is None


def test_case_6_later_event_does_not_mutate_signed_encounter(monkeypatch):
    client = _client(monkeypatch)
    _patient(client, "SYN-LONG")
    context = _context(client, "SYN-LONG")
    saved = _save(client, context["context_id"], _candidate()).json()
    encounter_id = saved["encounter"]["encounter_id"]
    before = client.get(f"/clinical/encounter/{encounter_id}", headers=HEADERS).json()["payload"]

    lab = client.post(
        "/clinical/patient/SYN-LONG/labs",
        headers=HEADERS,
        json={
            "lab_date": "2026-10-10",
            "source_encounter_id": encounter_id,
            "values": {"ctx": 180},
        },
    )
    assert lab.status_code == 200
    after = client.get(f"/clinical/encounter/{encounter_id}", headers=HEADERS).json()["payload"]
    assert after == before


def test_case_7_signed_visit_capture_cannot_be_silently_overwritten(monkeypatch):
    client = _client(monkeypatch)
    _patient(client, "SYN-LOCK")
    context = _context(client, "SYN-LOCK")
    saved = _save(client, context["context_id"], _candidate()).json()["encounter"]

    mutated = deepcopy(saved["payload"])
    mutated["_visit_capture_v1"]["candidate"]["reason_for_visit"]["text"] = "Silent overwrite"
    response = client.put(
        f"/clinical/encounter/{saved['encounter_id']}",
        headers=HEADERS,
        json={"payload": mutated, "status": "amended"},
    )
    assert response.status_code == 409
    assert "immutable" in response.json()["detail"]


def test_case_8_pending_is_separate_and_provenance_must_be_bound(monkeypatch):
    client = _client(monkeypatch)
    _patient(client, "SYN-PENDING")
    context = _context(client, "SYN-PENDING")
    saved = _save(client, context["context_id"], _candidate())
    assert saved.status_code == 200
    encounter_id = saved.json()["encounter"]["encounter_id"]

    pending = client.get("/clinical/patient/SYN-PENDING/pending", headers=HEADERS)
    assert pending.status_code == 200
    item = pending.json()[0]
    assert item["source_encounter_id"] == encounter_id
    assert item["responsible_role"] == "clinician"
    assert item["status"] == "not_yet_indicated"

    new_context = _context(client, "SYN-PENDING", "synthetic-browser-session-unbound")
    bad = _candidate()
    bad["decisions"][0]["provenance"] = ["unknown-source"]
    response = _preview(client, new_context["context_id"], bad)
    assert response.status_code == 422
    assert "unbound provenance" in response.json()["detail"]


def test_p2_f2_01_dependency_declared_fields_roundtrip(monkeypatch):
    client = _client(monkeypatch)
    _patient(client, "SYN-DEPENDENCY")
    context = _context(client, "SYN-DEPENDENCY")
    candidate = _candidate()
    candidate["pending"][0]["external_dependency"] = {
        "actor": "lawyer",
        "condition": "request_received",
    }

    preview = _preview(client, context["context_id"], candidate)
    assert preview.status_code == 200
    assert preview.json()["can_save"] is True
    assert preview.json()["normalized_candidate"]["pending"][0]["external_dependency"] == {
        "actor": "lawyer",
        "condition": "request_received",
    }

    saved = _save(client, context["context_id"], candidate)
    assert saved.status_code == 200
    encounter = saved.json()["encounter"]
    assert encounter["payload"]["_visit_capture_v1"]["candidate"]["pending"][0]["external_dependency"] == {
        "actor": "lawyer",
        "condition": "request_received",
    }
    pending = client.get("/clinical/patient/SYN-DEPENDENCY/pending", headers=HEADERS)
    assert pending.status_code == 200
    assert pending.json()[0]["external_dependency"] == {
        "actor": "lawyer",
        "condition": "request_received",
    }


def test_p2_f2_01_dependency_rejects_undeclared_nested_fields_without_persistence(monkeypatch):
    client = _client(monkeypatch)
    _patient(client, "SYN-DEPENDENCY-REJECT")
    context = _context(client, "SYN-DEPENDENCY-REJECT")
    candidate = _candidate()
    candidate["pending"][0]["external_dependency"] = {
        "actor": "lawyer",
        "condition": "request_received",
        "raw_source_body": {"patient_document": "do not persist"},
    }

    for endpoint in (_preview, _save):
        rejected = endpoint(client, context["context_id"], candidate)
        assert rejected.status_code == 422
        assert any(
            "external_dependency" in str(item.get("loc", []))
            and item.get("type") == "extra_forbidden"
            for item in rejected.json()["detail"]
        )

    assert client.get("/clinical/patient/SYN-DEPENDENCY-REJECT/encounters", headers=HEADERS).json() == []
    assert client.get("/clinical/patient/SYN-DEPENDENCY-REJECT/pending", headers=HEADERS).json() == []

    for field, value in (("actor", "a" * 81), ("condition", "c" * 241)):
        too_long = _candidate()
        too_long["pending"][0]["external_dependency"] = {
            "actor": "lawyer",
            "condition": "request_received",
        }
        too_long["pending"][0]["external_dependency"][field] = value
        assert _preview(client, context["context_id"], too_long).status_code == 422

    allowed = _candidate()
    allowed["pending"][0]["external_dependency"] = {
        "actor": "lawyer",
        "condition": "request_received",
    }
    assert _save(client, context["context_id"], allowed).status_code == 200


def test_patient_registry_lookup_reaches_all_patients_not_only_latest_100(monkeypatch):
    client = _client(monkeypatch)

    # Oldest patient will fall out of the legacy last-100 response.
    for number in range(151):
        patient_id = f"SYN-{number:03d}"
        name = "Αθανάσιος Παπαχρήστου" if number == 0 else "Κοινός Δοκιμαστικός"
        stored = client.post(
            "/clinical/patients",
            headers=HEADERS,
            json={
                "patient_id": patient_id,
                "demographics": {
                    "full_name": name,
                    "phone": f"NOT-SEARCHABLE-{number:03d}",
                },
            },
        )
        assert stored.status_code == 200

    latest_100 = client.get("/clinical/patients?limit=100", headers=HEADERS)
    assert latest_100.status_code == 200
    assert len(latest_100.json()) == 100
    assert "SYN-000" not in {row["patient_id"] for row in latest_100.json()}

    # The new search examines the entire registry, including the oldest row.
    by_id = client.get("/clinical/patients?query=SYN-000&limit=20&offset=0", headers=HEADERS)
    assert by_id.status_code == 200
    assert [row["patient_id"] for row in by_id.json()] == ["SYN-000"]
    by_name = client.get("/clinical/patients", params={"query": "ΠΑΠΑΧΡΗΣΤΟΥ αθανασιος"}, headers=HEADERS)
    assert [row["patient_id"] for row in by_name.json()] == ["SYN-000"]

    # Search matches only the allowed name/id fields, never arbitrary demographics.
    assert client.get(
        "/clinical/patients?query=NOT-SEARCHABLE-000", headers=HEADERS
    ).json() == []

    # All 150 common-name records can be retrieved with explicit pagination.
    ids = []
    for offset in range(0, 160, 20):
        result = client.get(
            "/clinical/patients",
            params={"query": "κοινος", "limit": 20, "offset": offset},
            headers=HEADERS,
        )
        assert result.status_code == 200
        ids.extend(row["patient_id"] for row in result.json())
    assert len(ids) == 150
    assert len(set(ids)) == 150
    assert "SYN-150" in ids and "SYN-001" in ids
    assert client.get("/clinical/patients?query=SYN-000").status_code == 401

    # Retrieval is still read-only; protected patient-context confirmation is
    # the existing explicit next step, not an automatic identity association.
    selected = _context(client, "SYN-000", "synthetic-search-selection")
    assert selected["patient_id"] == "SYN-000"
    assert client.get("/clinical/patient/SYN-000/encounters", headers=HEADERS).json() == []


def test_cockpit_recent_encounters_protected_three_minimal(monkeypatch):
    client = _client(monkeypatch)
    records = [
        ("R1", "2026-10-07", "completed", {"_visit_capture_v1": {
            "signed": True, "candidate": {"visit_type": "Επανέλεγχος"},
        }}),
        ("R2", "2026-10-08", "completed", {"private_clinical_note": "DO-NOT-RETURN"}),
        ("R3", "2026-10-09", "amended", {"_visit_capture_v1": {
            "signed": True, "candidate": {"visit_type": "Νέα εκτίμηση"},
        }}),
        ("R4", "2026-10-10", "draft", {"private_clinical_note": "DRAFT-SECRET"}),
        ("R5", "2026-10-06", "completed", {"private_clinical_note": "OLDER-SECRET"}),
    ]
    for patient_id, day, status, payload in records:
        name = "Μαρία Δοκιμαστική" if patient_id in {"R1", "R3"} else f"Δοκιμαστικός {patient_id}"
        patient = client.post("/clinical/patients", headers=HEADERS, json={
            "patient_id": patient_id, "demographics": {
                "full_name": name,
                "date_of_birth": "1970-01-01",
                "phone": "SECRET-PHONE",
            }
        })
        assert patient.status_code == 200
        saved = client.post(f"/clinical/patient/{patient_id}/encounters", headers=HEADERS, json={
            "encounter_date": day, "status": status, "payload": payload
        })
        assert saved.status_code == 200

    assert client.get("/clinical/recent-encounters?limit=3").status_code == 401
    response = client.get("/clinical/recent-encounters?limit=3", headers=HEADERS)
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 3
    assert [row["patient_id"] for row in rows] == ["R3", "R2", "R1"]
    assert [row["visit_type"] for row in rows] == [
        "Νέα εκτίμηση", "Κλινική επίσκεψη", "Επανέλεγχος"
    ]
    assert rows[0]["patient_display_name"] == rows[2]["patient_display_name"]
    assert all(set(row) == {"patient_id", "patient_display_name", "encounter_date", "visit_type"}
               for row in rows)
    assert "DO-NOT-RETURN" not in response.text
    assert "SECRET-PHONE" not in response.text
    assert "date_of_birth" not in response.text
    assert client.get("/clinical/recent-encounters?limit=1", headers=HEADERS).status_code == 200
    assert client.get("/clinical/recent-encounters?limit=4", headers=HEADERS).status_code == 422
