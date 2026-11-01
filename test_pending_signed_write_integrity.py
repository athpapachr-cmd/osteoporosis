"""Existing clinical owner; disposable SQLite/PostgreSQL only, no provider calls."""
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Event
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from clinical_calendar import CalendarBase
from clinical_data import ClinicalBase, ClinicalPendingORM, EncounterORM, PatientORM, utcnow
from clinical_pending_one_tap import PendingEventBase, PendingCheckpointEventORM
from clinical_visit_brief import VisitBriefBase
from test_cockpit_cp2 import seed
from test_cockpit_synthetic_161 import application, login, KEY
from test_cockpit_visit_amendments import edit


@pytest.fixture(params=["sqlite", "postgresql"])
def case(request, tmp_path, monkeypatch):
    monkeypatch.setenv("CLINICAL_DATA_KEY", KEY)
    monkeypatch.delenv("COCKPIT_STAFF_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("CLINICAL_VISIT_BRIEF_LINKS_ENABLED", raising=False)
    admin = None
    if request.param == "postgresql":
        url = os.environ.get("P2_TEST_POSTGRES_URL")
        if not url:
            pytest.skip("Set P2_TEST_POSTGRES_URL for disposable local PostgreSQL evidence")
        parsed = make_url(url)
        if parsed.host not in ("localhost", "127.0.0.1") or parsed.database not in ("p2_test", "cockpit_p2_test"):
            pytest.fail("P2 evidence requires a disposable loopback test database")
        schema = "p2_" + uuid4().hex
        admin = create_engine(url)
        with admin.begin() as db:
            db.execute(text('CREATE SCHEMA "' + schema + '"'))
        engine = create_engine(url, connect_args={"options": "-csearch_path=" + schema + " -clock_timeout=1000ms"})
    else:
        engine = create_engine("sqlite:///" + str(tmp_path / "p2.db"),
            connect_args={"check_same_thread": False, "timeout": 1})
    try:
        for base in (ClinicalBase, CalendarBase, VisitBriefBase, PendingEventBase):
            base.metadata.create_all(engine)  # isolated test schema only
        with Session(engine) as db:
            db.add(PatientORM(patient_id="outside-synthetic-scope", demographics_json={"full_name": "Synthetic sentinel"},
                created_at=utcnow(), updated_at=utcnow()))
            db.commit()
        app = application(engine)
        client, pending, path = seed(app, engine)
        yield app, engine, client, pending, path
    finally:
        engine.dispose()
        if admin is not None:
            with admin.begin() as db:
                db.execute(text('DROP SCHEMA "' + schema + '" CASCADE'))
            admin.dispose()


def plan(revision=0, observed=None):
    return {"template": "completion", "expected_revision": revision, "idempotency_key": uuid4().hex,
        **({"first_observation": observed} if observed else {})}


def action(revision=0, observation="yes_confirmed", retract=None):
    return {"checkpoint_key": "completed", "observation": observation,
        "expected_revision": revision, "idempotency_key": uuid4().hex, "retracts_event_id": retract}


def lifecycle(kind, revision, cancellation=None):
    return {"action": kind, "expected_revision": revision,
        "cancellation_event_id": cancellation, "idempotency_key": uuid4().hex}


def snapshot(engine, pending):
    with Session(engine) as db:
        row = db.get(ClinicalPendingORM, pending["pending_id"])
        source = db.get(EncounterORM, pending["source_encounter_id"])
        events = db.scalars(select(PendingCheckpointEventORM).order_by(PendingCheckpointEventORM.revision)).all()
        return deepcopy((row.status, row.item_type, row.review_date, row.updated_at, row.resolution_event_id,
            source.payload_json if source else None,
            [(e.id, e.kind, e.revision, e.plan_json, e.retracts_event_id) for e in events]))


@pytest.mark.parametrize("defect", ["absent", "duplicate", "other_duplicate", "length", "bad_id", "bad_item", "bad_payload", "unsigned", "no_timestamp", "foreign", "missing_source"])
@pytest.mark.parametrize("operation", ["plan", "checkpoint", "retract", "restore"])
def test_invalid_current_source_never_creates_progress(case, defect, operation):
    _, engine, c, p, path = case
    revision = 0
    retract = cancellation = None
    if operation in ("checkpoint", "retract"):
        response = c.post(path + "/checkpoint-plan", json=plan(observed="yes_confirmed" if operation == "retract" else None))
        assert response.status_code == 200, response.text
        revision = response.json()["current"]["revision"]
        if operation == "retract":
            retract = response.json()["current"]["steps"][0]["event_id"]
    if operation == "restore":
        response = c.post(path + "/lifecycle", json=lifecycle("cancel", 0))
        assert response.status_code == 200
        cancellation = response.json()["event_id"]
        revision = 1
    with Session(engine) as db:
        source = db.get(EncounterORM, p["source_encounter_id"])
        payload = deepcopy(source.payload_json)
        signed = payload["_visit_capture_v1"]
        if defect == "absent":
            signed["pending_ids"] = []; signed["candidate"]["pending"] = []
        elif defect == "duplicate":
            signed["pending_ids"] *= 2; signed["candidate"]["pending"] *= 2
        elif defect == "other_duplicate":
            signed["pending_ids"] += ["other", "other"]
            signed["candidate"]["pending"] *= 3
        elif defect == "length": signed["candidate"]["pending"] = []
        elif defect == "bad_id": signed["pending_ids"] = [None]
        elif defect == "bad_item": signed["candidate"]["pending"] = [{}]
        elif defect == "bad_payload": payload = []
        elif defect == "unsigned": signed["signed"] = False
        elif defect == "no_timestamp": signed["signed_at"] = ""
        elif defect == "foreign": source.patient_id = "outside-synthetic-scope"
        elif defect == "missing_source": db.delete(source)
        if defect != "missing_source": source.payload_json = payload
        db.commit()
    before = snapshot(engine, p)
    if operation == "plan": response = c.post(path + "/checkpoint-plan", json=plan(observed="yes_confirmed"))
    elif operation == "restore": response = c.post(path + "/lifecycle", json=lifecycle("restore", revision, cancellation))
    else: response = c.post(path + "/checkpoint", json=action(revision, "retracted" if retract else "yes_confirmed", retract))
    assert response.status_code == 409, response.text
    assert snapshot(engine, p) == before


def test_real_amendment_membership_replay_and_historical_deletion(case):
    _, engine, c, p, path = case
    approved = plan(observed="not_yet_observed")
    r = c.post(path + "/checkpoint-plan", json=approved); assert r.status_code == 200
    observed = action(1)
    r = c.post(path + "/checkpoint", json=observed); assert r.status_code == 200
    visit = f"/clinical/patient/{p['patient_id']}/visit/{p['source_encounter_id']}"
    body = edit(c, visit)
    body["candidate"]["pending"] = []; body["pending_bindings"] = []
    r = c.post(visit + "/amend", json=body); assert r.status_code == 200, r.text
    before = snapshot(engine, p)
    assert before[0] == "cancelled"
    for endpoint, payload in (("/checkpoint-plan", approved), ("/checkpoint", observed)):
        r = c.post(path + endpoint, json=payload)
        assert r.status_code == 200 and r.json()["replayed"]
    assert snapshot(engine, p) == before
    chart = c.get(f"/clinical/patient/{p['patient_id']}/chart").json()["pending"][0]
    rev = chart["lifecycle"]["revision"]; cancelled = chart["lifecycle"]["cancellation_event_id"]
    assert c.post(path + "/lifecycle", json=lifecycle("restore", rev, cancelled)).status_code == 409
    assert snapshot(engine, p) == before
    delete = lifecycle("delete_permanently", rev, cancelled)
    assert c.post(path + "/lifecycle", json=delete).status_code == 200
    deleted = snapshot(engine, p)
    assert c.post(path + "/lifecycle", json=delete).json()["replayed"]
    assert c.post(path + "/checkpoint", json=observed).status_code == 409
    assert snapshot(engine, p) == deleted


def test_metadata_a_b_a_replay_keeps_current_projection_and_guard_identity(case):
    _, engine, c, p, path = case
    with Session(engine) as db:
        row = db.get(ClinicalPendingORM, p["pending_id"]); row.item_type = "legacy_category"
        source = db.get(EncounterORM, p["source_encounter_id"])
        payload = deepcopy(source.payload_json)
        payload["_visit_capture_v1"]["candidate"]["pending"][0]["item_type"] = row.item_type
        source.payload_json = payload; db.commit()
    a = {"item_type": "legacy_category", "review_date": "2027-01-18", "expected_revision": 0, "idempotency_key": uuid4().hex}
    first = c.post(path + "/metadata", json=a); assert first.status_code == 200
    b = {**a, "item_type": "result", "review_date": "2027-03-18", "expected_revision": 1, "idempotency_key": uuid4().hex}
    assert c.post(path + "/metadata", json=b).status_code == 200
    before = snapshot(engine, p)
    r = c.post(path + "/metadata", json=a)
    assert r.status_code == 200 and r.json()["replayed"] and r.json()["event_id"] == first.json()["event_id"]
    assert r.json()["current"] == {"item_type": "result", "review_date": b["review_date"], "status": p["status"], "revision": 2}
    assert snapshot(engine, p) == before
    for changed in ({"item_type": "document"}, {"expected_revision": 1}, {"review_date": "2028-01-01"}):
        assert c.post(path + "/metadata", json={**a, **changed}).status_code == 409
    assert c.post(path + "/metadata", json={**a, "expected_revision": 2, "idempotency_key": uuid4().hex}).status_code == 422
    assert c.post(path.replace(p["patient_id"], "outside-synthetic-scope") + "/metadata", json=a).status_code == 404
    # Simulate a different actor's committed event; actor identity cannot replay.
    with Session(engine) as db:
        ev = db.scalar(select(PendingCheckpointEventORM).where(PendingCheckpointEventORM.idempotency_key == a["idempotency_key"]))
        ev.actor_principal_id = "different-synthetic-actor"; db.commit()
    assert c.post(path + "/metadata", json=a).status_code == 409


def test_current_member_metadata_and_unchanged_amendment_then_undo_restore(case):
    _, engine, c, p, path = case
    original = snapshot(engine, p)[5]
    metadata = {"item_type": "result", "review_date": "2027-04-18", "expected_revision": 0, "idempotency_key": uuid4().hex}
    assert c.post(path + "/metadata", json=metadata).status_code == 200
    r = c.post(path + "/checkpoint-plan", json=plan(1)); assert r.status_code == 200
    r = c.post(path + "/checkpoint", json=action(2)); assert r.status_code == 200
    event_id = r.json()["event_id"]
    assert snapshot(engine, p)[5] == original
    r = c.post(path + "/checkpoint", json=action(3, "retracted", event_id)); assert r.status_code == 200
    r = c.post(path + "/lifecycle", json=lifecycle("cancel", 4)); assert r.status_code == 200
    restore = lifecycle("restore", 5, r.json()["event_id"])
    assert c.post(path + "/lifecycle", json=restore).status_code == 200
    before = snapshot(engine, p)
    assert c.post(path + "/lifecycle", json=restore).json()["replayed"]
    assert snapshot(engine, p) == before
    visit = f"/clinical/patient/{p['patient_id']}/visit/{p['source_encounter_id']}"
    body = edit(c, visit); body["candidate"]["uncertainties"] = []
    assert c.post(visit + "/amend", json=body).status_code == 200
    r = c.post(path + "/checkpoint", json=action(6)); assert r.status_code == 200, r.text


def test_noop_cancelled_replays_and_terminal_deletion_boundaries(case):
    _, engine, c, p, path = case
    metadata = {"item_type": "result", "review_date": "2027-01-18", "expected_revision": 0, "idempotency_key": uuid4().hex}
    assert c.post(path + "/metadata", json=metadata).status_code == 200
    before = snapshot(engine, p)
    noop = {**metadata, "expected_revision": 1, "idempotency_key": uuid4().hex}
    assert c.post(path + "/metadata", json=noop).json()["unchanged"]
    assert snapshot(engine, p) == before
    cancelled = c.post(path + "/lifecycle", json=lifecycle("cancel", 1)).json()
    restore = lifecycle("restore", 2, cancelled["event_id"])
    assert c.post(path + "/lifecycle", json=restore).status_code == 200
    visit = f"/clinical/patient/{p['patient_id']}/visit/{p['source_encounter_id']}"
    body = edit(c, visit); body["candidate"]["pending"] = []; body["pending_bindings"] = []
    assert c.post(visit + "/amend", json=body).status_code == 200
    before = snapshot(engine, p)
    assert before[0] == "cancelled"
    for endpoint, payload in (("/metadata", metadata), ("/lifecycle", restore)):
        replay = c.post(path + endpoint, json=payload)
        assert replay.status_code == 200 and replay.json()["replayed"]
    assert snapshot(engine, p) == before
    row = c.get(f"/clinical/patient/{p['patient_id']}/chart").json()["pending"][0]
    delete = lifecycle("delete_permanently", 4, row["lifecycle"]["cancellation_event_id"])
    assert c.post(path + "/lifecycle", json=delete).status_code == 200
    before = snapshot(engine, p)
    assert c.post(path + "/metadata", json=metadata).status_code == 409
    assert c.post(path + "/lifecycle", json=restore).status_code == 409
    assert c.post(path + "/lifecycle", json=delete).json()["replayed"]
    assert snapshot(engine, p) == before


def test_historical_orphan_can_be_cancelled_and_deleted_without_reactivation(case):
    _, engine, c, p, path = case
    with Session(engine) as db:
        source = db.get(EncounterORM, p["source_encounter_id"])
        payload = deepcopy(source.payload_json)
        payload["_visit_capture_v1"]["pending_ids"] = []
        payload["_visit_capture_v1"]["candidate"]["pending"] = []
        source.payload_json = payload; db.commit()
    signed = snapshot(engine, p)[5]
    response = c.post(path + "/lifecycle", json=lifecycle("cancel", 0))
    assert response.status_code == 200
    assert c.post(path + "/lifecycle", json=lifecycle("restore", 1, response.json()["event_id"])).status_code == 409
    assert c.post(path + "/lifecycle", json=lifecycle("delete_permanently", 1, response.json()["event_id"])).status_code == 200
    assert snapshot(engine, p)[5] == signed


@pytest.mark.parametrize("first", ["checkpoint", "amendment"])
@pytest.mark.parametrize("keep", [False, True])
def test_two_connections_serialize_amendment_and_checkpoint(case, monkeypatch, first, keep):
    import clinical_pending_one_tap as checkpoints
    import clinical_visit_amendments as amendments
    app, engine, c, p, path = case
    assert c.post(path + "/checkpoint-plan", json=plan()).status_code == 200
    visit = f"/clinical/patient/{p['patient_id']}/visit/{p['source_encounter_id']}"
    body = edit(c, visit)
    if not keep: body["candidate"]["pending"] = []; body["pending_bindings"] = []
    holder, release, attempting, acquired = Event(), Event(), Event(), Event()
    contender_queries = set()
    # Pause the holder AFTER it owns the relevant locks and before mutation.
    if first == "checkpoint":
        original = checkpoints._current_member
        def pause(*args):
            holder.set(); assert release.wait(5); return original(*args)
        monkeypatch.setattr(checkpoints, "_current_member", pause)
    else:
        original = amendments.token
        def pause(*args):
            holder.set(); assert release.wait(5); return original(*args)
        monkeypatch.setattr(amendments, "token", pause)
    # A driver hook signals actual SQL execution on the contender connection.
    def seen(conn, cursor, statement, parameters, context, executemany):
        # TestClient runs handlers on AnyIO threads, not the client thread.
        # While the holder is paused, only the contender can issue SQL.
        if holder.is_set() and not release.is_set():
            locking = ((statement == "BEGIN IMMEDIATE" or statement.startswith("UPDATE clinical_encounters"))
                if engine.dialect.name == "sqlite" else "clinical_encounters" in statement and "FOR UPDATE" in statement)
            if locking:
                contender_queries.add(id(context)); attempting.set()
    def completed(conn, cursor, statement, parameters, context, executemany):
        if id(context) in contender_queries: acquired.set()
    event.listen(engine, "before_cursor_execute", seen)
    event.listen(engine, "after_cursor_execute", completed)
    cp_client, amendment_client = login(app), login(app)
    def cp(): return cp_client.post(path + "/checkpoint", json=action())
    def amend(): return amendment_client.post(visit + "/amend", json=body)
    try:
        with ThreadPoolExecutor(1, thread_name_prefix="p2-holder") as pool, ThreadPoolExecutor(1, thread_name_prefix="p2-contender") as other:
            held = pool.submit(cp if first == "checkpoint" else amend)
            assert holder.wait(5)
            waiting = other.submit(amend if first == "checkpoint" else cp)
            try:
                assert attempting.wait(5)
                assert not acquired.wait(.05), "Contender bypassed the transaction lock"
                assert not waiting.done()
            finally: release.set()
            a, b = held.result(5), waiting.result(5)
        assert a.status_code == 200, a.text
        expected = 409 if first == "checkpoint" or not keep else 200
        assert b.status_code == expected, b.text
        state = snapshot(engine, p)
        if first == "checkpoint":
            assert state[0] == "resolved" and state[5]["_visit_capture_v1"]["pending_ids"] == [p["pending_id"]]
            assert [e[1] for e in state[6]] == ["plan_approved", "yes_confirmed"]
        elif not keep:
            assert state[0] == "cancelled" and [e[1] for e in state[6]] == ["plan_approved", "cancelled"]
        else: assert state[0] == "resolved"
    finally:
        release.set(); event.remove(engine, "before_cursor_execute", seen)
        event.remove(engine, "after_cursor_execute", completed)


def test_real_lock_acquisition_conflict_rolls_back(case):
    _, engine, c, p, path = case
    before = snapshot(engine, p)
    with engine.connect() as blocker:
        if engine.dialect.name == "sqlite": blocker.exec_driver_sql("BEGIN IMMEDIATE")
        else: blocker.execute(select(EncounterORM).where(EncounterORM.id == p["source_encounter_id"]).with_for_update())
        try:
            r = c.post(path + "/metadata", json={"item_type": "result", "review_date": None,
                "expected_revision": 0, "idempotency_key": uuid4().hex})
            assert r.status_code == 409, r.text
        finally: blocker.rollback()
    assert snapshot(engine, p) == before


def test_sqlite_real_commit_conflict_rolls_back(case):
    _, engine, c, p, path = case
    if engine.dialect.name != "sqlite": pytest.skip("SQLite rollback-journal commit conflict")
    before = snapshot(engine, p)
    with engine.connect() as reader:
        reader.exec_driver_sql("BEGIN")
        reader.exec_driver_sql("SELECT * FROM clinical_pending_items").all()
        try:
            r = c.post(path + "/checkpoint-plan", json=plan(observed="yes_confirmed"))
            assert r.status_code == 409, r.text
        finally: reader.rollback()
    assert snapshot(engine, p) == before


def test_write_phase_lock_failure_rolls_back_partial_flush(case):
    _, engine, c, p, path = case
    before = snapshot(engine, p)
    def fail(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("INSERT INTO clinical_pending_checkpoint_events"):
            # Deterministic driver failure injection, not claimed as a live race.
            if engine.dialect.name == "sqlite":
                error = sqlite3.OperationalError("synthetic locked write"); error.sqlite_errorcode = sqlite3.SQLITE_BUSY
            else:
                class Conflict(Exception): sqlstate = "55P03"
                error = Conflict("synthetic locked write")
            raise OperationalError(statement, parameters, error)
    event.listen(engine, "before_cursor_execute", fail)
    try:
        r = c.post(path + "/checkpoint-plan", json=plan(observed="yes_confirmed"))
        assert r.status_code == 409
    finally: event.remove(engine, "before_cursor_execute", fail)
    assert snapshot(engine, p) == before
