"""Real disposable DB transactions, synthetic read-only IMAP, no live clinical data."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import os
from threading import Barrier
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, inspect, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

import clinical_icloud_mail as mail_source
import clinical_icloud_preferences as pref
from test_clinical_icloud_mail import client as source_client, mail


@pytest.fixture(params=["sqlite", "postgresql"])
def store(request, tmp_path):
    admin = None
    if request.param == "postgresql":
        url = os.getenv("P2_TEST_POSTGRES_URL")
        if not url:
            pytest.skip("Disposable loopback PostgreSQL URL required")
        parsed = make_url(url)
        assert parsed.host in ("127.0.0.1", "localhost") and parsed.database in ("p2_test", "cockpit_p2_test")
        admin = create_engine(url)
        schema = "triage_" + uuid4().hex
        with admin.begin() as db:
            db.execute(text('CREATE SCHEMA "' + schema + '"'))
        engine = create_engine(url, connect_args={"options": "-csearch_path=" + schema})
    else:
        engine = create_engine("sqlite:///" + str(tmp_path / "prefs.db"), connect_args={"timeout": .15, "check_same_thread": False})
    if admin:
        # Execute the actual offline provisioning candidate in the isolated schema.
        with engine.begin() as db:
            db.exec_driver_sql((Path(__file__).parent / "cockpit/migrations/20261101_icloud_triage_preferences.sql").read_text())
    else:
        pref.PreferenceBase.metadata.create_all(engine)  # Only this disposable fixture provisions.
    try:
        yield engine
        assert inspect(engine).get_table_names() == ["icloud_triage_preferences"]
    finally:
        engine.dispose()
        if admin:
            with admin.begin() as db:
                db.execute(text('DROP SCHEMA "' + schema + '" CASCADE'))
            admin.dispose()


@pytest.fixture()
def api(store, source_client):
    _, fake = source_client
    app = FastAPI()
    app.include_router(pref.build_preferences_router(store))
    app.include_router(mail_source.build_icloud_mail_router(store))
    c = TestClient(app)
    c.headers.update({"X-Clinical-Key": "synthetic-clinical-key", "Origin": "http://testserver"})
    return c, fake


BASE = "/clinical/icloud-mail"


def change(revision=0, **values):
    return {"kind": "category", "operation_id": str(uuid4()), "expected_revision": revision,
            "item_id": "765-3", "category": "administrative", **values}


def post(c, data):
    r = c.post(BASE + "/preferences", json=data)
    assert r.status_code == 200, r.text
    return r.json()


def state(engine):
    with Session(engine) as db:
        return deepcopy([(r.mailbox_key, r.revision, r.state_json) for r in db.scalars(select(pref.PreferenceORM))])


def test_category_matrix_reload_reset_and_undo(api, store):
    c, fake = api
    assert c.get(BASE + "/preferences").json()["triage"]["revision"] == 0
    assert state(store) == []  # GET does not initialize storage.
    rev = 0
    for before in ("clinical", "administrative", "other", "review"):
        for after in ("clinical", "administrative", "other", "review"):
            for value in (before, after):
                rev = post(c, change(rev, category=value))["triage"]["revision"]
            row = c.get(BASE + "/items/765-3").json()["item"]
            assert (row["category"], row["category_origin"]) == (after, "manual")
    saved = post(c, change(rev, category="administrative"))
    assert "765-3" not in [r["id"] for r in c.get(BASE).json()["items"]]
    assert "765-3" not in [r["id"] for r in c.get(BASE + "/peek").json()["items"]]
    reset = post(c, change(saved["triage"]["revision"], category=None))
    assert c.get(BASE + "/items/765-3").json()["item"]["category_origin"] == "suggestion"
    post(c, {"kind": "undo", "operation_id": str(uuid4()), "expected_revision": reset["triage"]["revision"], "undo_id": reset["operation_id"]})
    assert c.get(BASE + "/items/765-3").json()["item"]["category"] == "administrative"
    assert all(op in ("FETCH", "SEARCH") for op, _ in fake.operations)


def test_rules_future_scope_conflict_manual_and_edit(api):
    c, fake = api
    rule = {"sender": "noreply@wetransfer.com", "terms": ["αξονικί"], "category": "administrative"}
    a = post(c, change(rule=rule, category="review"))
    assert c.get(BASE + "/items/765-3").json()["item"]["category"] == "review"
    fake.rows["5"] = mail(subject="Αξονικί δοκιμή.rar", sender=rule["sender"])
    row = c.get(BASE + "/items/765-5").json()["item"]
    assert (row["category"], row["category_origin"]) == ("administrative", "rule")
    b = post(c, {"kind": "rule", "operation_id": str(uuid4()), "expected_revision": 1, "rule": {**rule, "category": "clinical"}})
    fake.rows["6"] = fake.rows["5"]
    assert c.get(BASE + "/items/765-6").json()["item"]["category_origin"] == "conflict"
    manual = post(c, change(2, item_id="765-6", category="other"))
    assert c.get(BASE + "/items/765-6").json()["item"]["category"] == "other"
    r = b["rules"][0]
    edited = post(c, {"kind": "rule", "operation_id": str(uuid4()), "expected_revision": 3,
                     "rule": {**rule, "rule_id": r["rule_id"], "terms": ["MRI"]}})
    assert edited["rules"][-1]["after_uid"] == 6
    post(c, {"kind": "disable", "operation_id": str(uuid4()), "expected_revision": 4, "rule_id": r["rule_id"]})
    assert c.get(BASE + "/items/765-6").json()["item"]["category"] == "other"
    # Changing epoch makes old per-message IDs and rule watermarks incomparable.
    @contextmanager
    def epoch(*args):
        yield fake, "999"
    from unittest.mock import patch
    with patch.object(mail_source, "_session", epoch):
        assert all(not r["active"] for r in c.get(BASE + "/preferences").json()["rules"])
        assert c.get(BASE + "/items/999-6").json()["item"]["category_origin"] == "suggestion"
    assert pref.scope_key("one", "a") != pref.scope_key("one", "b") != pref.scope_key("two", "b")


def test_replay_current_projection_eviction_and_conditional_undo(api, store):
    c, _ = api
    original = change()
    a = post(c, original)
    b = post(c, change(1, category="clinical"))
    snapshot = state(store)
    replay = post(c, original)
    assert replay["replayed"] and replay["triage"]["revision"] == 2 and replay["committed_revision"] == 1
    assert state(store) == snapshot
    assert c.post(BASE + "/preferences", json={**original, "category": "other"}).status_code == 409
    assert c.post(BASE + "/preferences", json={"kind": "undo", "operation_id": str(uuid4()), "expected_revision": 2, "undo_id": a["operation_id"]}).status_code == 409
    for rev in range(2, 67):
        post(c, change(rev))
    assert len(state(store)[0][2]["receipts"]) == 64
    assert c.post(BASE + "/preferences", json=original).status_code == 409


def test_denials_source_capacity_and_commit_rollback(api, store, monkeypatch):
    c, fake = api
    assert c.post(BASE + "/preferences", json=change(), headers={"Origin": "https://foreign.invalid"}).status_code == 403
    assert c.post(BASE + "/preferences", json=change(), headers={"X-Clinical-Key": "wrong"}).status_code == 401
    assert c.post(BASE + "/preferences", json=change(item_id="765-1")).status_code == 404
    assert c.post(BASE + "/preferences", json=change(item_id="999-3")).status_code == 409
    assert c.post(BASE + "/preferences", json=change(rule={"sender": "noreply@wetransfer.com", "terms": [], "category": "clinical"})).status_code == 422
    monkeypatch.setattr(pref, "MAX_OVERRIDES", 0)
    assert c.post(BASE + "/preferences", json=change()).status_code == 409
    assert state(store) == []
    monkeypatch.setattr(pref, "MAX_OVERRIDES", 5000)
    def fail_commit(connection):
        raise SQLAlchemyError("synthetic driver commit failure")
    event.listen(store, "commit", fail_commit)
    try:
        assert c.post(BASE + "/preferences", json=change()).status_code == 503
    finally:
        event.remove(store, "commit", fail_commit)
    assert state(store) == []  # Explicit injection; not claimed as a physical race.


@pytest.mark.parametrize("initial", [False, True])
def test_real_two_connection_compare_and_swap(store, initial):
    scope = pref.scope_key("synthetic", "alias")
    if initial:
        pref.mutate(store, scope, "765", pref.Change(**change()), lambda *_: 0)
    barrier = Barrier(2)
    def write(value):
        def qualify(*args):
            barrier.wait(timeout=5)
            return 0
        try:
            result = pref.mutate(store, scope, "765", pref.Change(**change(int(initial), category=value)), qualify)
            return result["triage"]["revision"]
        except HTTPException as error:
            return error.status_code
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(write, ("clinical", "administrative")))
    assert sorted(results) == [int(initial) + 1, 409]
    assert state(store)[0][1] == int(initial) + 1


def test_store_failure_never_hides_unknown_categories(api, monkeypatch):
    c, _ = api
    monkeypatch.setattr(pref, "read_preferences", lambda *_: (_ for _ in ()).throw(pref.unavailable()))
    data = c.get(BASE).json()
    assert data["triage"]["available"] is False and data["triage"]["revision"] is None
    assert {i["id"] for i in data["items"]} == {"765-2", "765-3", "765-4"}
    assert all(i["category_origin"] == "unavailable" for i in data["items"])
    assert c.post(BASE + "/preferences", json=change()).status_code == 503


def test_unprovisioned_store_has_no_implicit_ddl(source_client, tmp_path):
    _, fake = source_client
    engine = create_engine("sqlite:///" + str(tmp_path / "unprovisioned.db"))
    app = FastAPI()
    app.include_router(pref.build_preferences_router(engine))
    app.include_router(mail_source.build_icloud_mail_router(engine))
    c = TestClient(app)
    c.headers.update({"X-Clinical-Key": "synthetic-clinical-key", "Origin": "http://testserver"})
    assert c.get(BASE + "/preferences").status_code == 503
    data = c.get(BASE).json()
    assert data["triage"]["available"] is False and len(data["items"]) == 3
    assert c.post(BASE + "/preferences", json=change()).status_code == 503
    assert inspect(engine).get_table_names() == []
    engine.dispose()


def test_real_sqlite_reader_held_commit_conflict(store):
    if store.dialect.name != "sqlite":
        pytest.skip("SQLite reader-held commit condition is not PostgreSQL MVCC")
    scope = pref.scope_key("synthetic", "alias")
    pref.mutate(store, scope, "765", pref.Change(**change()), lambda *_: 0)
    before = state(store)
    with store.connect() as reader:
        reader.exec_driver_sql("BEGIN")
        reader.exec_driver_sql("SELECT * FROM icloud_triage_preferences").fetchall()
        with pytest.raises(HTTPException) as failure:
            pref.mutate(store, scope, "765", pref.Change(**change(1, category="clinical")), lambda *_: 0)
        assert failure.value.status_code == 409
        reader.rollback()
    assert state(store) == before


@pytest.mark.parametrize("term", ["Αξονική", "αξονική", "ΑΞΟΝΙΚΗ", "αξονικί"])
def test_greek_ct_variants_are_hints_only(term):
    from email.parser import BytesParser
    from email import policy
    from clinical_icloud_triage_rules import suggest_category
    message = BytesParser(policy=policy.default).parsebytes(mail(subject=term + " δοκιμή.rar", sender="noreply@wetransfer.com"))
    assert suggest_category(message) == "clinical"
