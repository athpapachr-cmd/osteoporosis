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


def test_existing_rule_scope_legacy_hash_manual_conflict_and_inverse(api, store):
    c, fake = api
    rule = {"sender": "synthetic.invalid", "domain": True, "terms": [], "category": "clinical"}
    fake.rows["5"] = mail(subject="Older notice", sender="auto@sub.synthetic.invalid")
    legacy = {"kind": "rule", "operation_id": str(uuid4()), "expected_revision": 0, "rule": rule}
    a = post(c, legacy)
    assert c.get(BASE + "/items/765-5").json()["item"]["category_origin"] == "suggestion"
    # Reconstruct the exact pre-feature canonical receipt (no new default field).
    old_payload = pref.Change(**legacy).model_dump(mode="json")
    old_payload["rule"].pop("apply_existing")
    assert state(store)[0][2]["receipts"][0]["hash"] == pref._hash(old_payload)
    snapshot = state(store)
    assert post(c, {**legacy, "rule": {**rule, "apply_existing": False}})["replayed"]
    assert state(store) == snapshot
    assert c.post(BASE + "/preferences", json={**legacy, "rule": {**rule, "apply_existing": True}}).status_code == 409
    for invalid in (1, "true", None):
        assert c.post(BASE + "/preferences", json={**legacy, "rule": {**rule, "apply_existing": invalid}}).status_code == 422
    # Deliberate edit changes coverage, not sender criteria, same current epoch only.
    rule_id = a["rules"][0]["rule_id"]
    edit = post(c, {**legacy, "operation_id": str(uuid4()), "expected_revision": 1,
                   "rule": {**rule, "rule_id": rule_id, "apply_existing": True}})
    assert edit["rules"][0]["after_uid"] == 5
    assert c.get(BASE + "/items/765-5").json()["item"]["category_origin"] == "rule"
    fake.rows["6"] = fake.rows["5"]
    assert c.get(BASE + "/items/765-6").json()["item"]["category"] == "clinical"
    post(c, change(2, item_id="765-5", category="other"))
    post(c, {**legacy, "operation_id": str(uuid4()), "expected_revision": 3,
             "rule": {**rule, "category": "administrative", "apply_existing": True}})
    assert c.get(BASE + "/items/765-5").json()["item"]["category"] == "other"
    assert c.get(BASE + "/items/765-6").json()["item"]["category_origin"] == "conflict"
    disable = post(c, {"kind": "disable", "operation_id": str(uuid4()), "expected_revision": 4, "rule_id": rule_id})
    assert c.get(BASE + "/items/765-6").json()["item"]["category"] == "administrative"
    post(c, {"kind": "undo", "operation_id": str(uuid4()), "expected_revision": 5, "undo_id": disable["operation_id"]})
    assert c.get(BASE + "/items/765-6").json()["item"]["category_origin"] == "conflict"
    @contextmanager
    def epoch(*args):
        yield fake, "999"
    from unittest.mock import patch
    with patch.object(mail_source, "_session", epoch):
        assert c.get(BASE + "/items/999-6").json()["item"]["category_origin"] == "suggestion"
    # Mixed transfer domains still cannot become a broad rule in either scope.
    assert c.post(BASE + "/preferences", json={**legacy, "expected_revision": 6,
        "rule": {**rule, "sender": "wetransfer.com", "apply_existing": True}}).status_code == 422


def captured(c):
    data = c.get(BASE + "?view=all").json()
    return [{"id": item["id"], "projection_ref": item["projection_ref"]} for item in data["items"]]


def test_projection_current_complete_snapshot_no_source_fetch_or_write(api, store, monkeypatch):
    c, fake = api
    rows = captured(c)
    before = state(store)
    fake.operations.clear()
    first = c.post(BASE + "/category-projection", json={"items": rows})
    assert first.status_code == 200, first.text
    assert "no-store" in first.headers["cache-control"]
    initial = first.json()
    assert state(store) == before == [] and fake.operations == []
    assert {r["id"] for r in initial["items"]} == {"765-2", "765-3", "765-4"}
    # Commit a new rule plus manual override, then retry OLD captured source facts.
    post(c, change(category="other", rule={"sender": "sender@example.com", "terms": [],
        "category": "clinical", "apply_existing": True}))
    fake.operations.clear()
    before = state(store)
    result = c.post(BASE + "/category-projection", json={"items": rows}).json()
    assert result["triage"]["revision"] == 1 and fake.operations == [] and state(store) == before
    by_id = {r["id"]: r for r in result["items"]}
    assert (by_id["765-3"]["category"], by_id["765-3"]["category_origin"]) == ("other", "manual")
    assert [r["source_captured_at"] for r in result["items"]] == [r["source_captured_at"] for r in initial["items"]]
    assert [r["projection_ref"] for r in result["items"]] == [r["projection_ref"] for r in initial["items"]]
    # One frozen DB snapshot, even when a real second transaction advances mid-projection.
    original_classify = pref.classify
    fired = []
    def advance(state_arg, *args):
        if not fired:
            fired.append(True)
            pref.mutate(store, before[0][0], "765", pref.Change(**change(1, item_id="765-4", category="other")), lambda *_: 0)
        return original_classify(state_arg, *args)
    monkeypatch.setattr(pref, "classify", advance)
    result = c.post(BASE + "/category-projection", json={"items": rows}).json()
    assert result["triage"]["revision"] == 1
    assert next(r for r in result["items"] if r["id"] == "765-4")["category"] != "other"
    assert state(store)[0][1] == 2
    assert fake.operations == []


def test_projection_cold_headers_visibility_and_historical_source_limits(api, store):
    import base64
    import json
    c, fake = api
    fake.rows["5"] = mail(subject="A" * 220 + " hidden clinical suffix", body="SECRET BODY", sender="sender@example.com")
    listed = c.get(BASE + "?view=all").json()["items"]
    assert "projection_ref" not in next(r for r in listed if r["id"] == "765-5")
    normal = next(r for r in listed if r["id"] == "765-3")
    payload = json.loads(base64.urlsafe_b64decode(normal["projection_ref"].split(".")[0] + "=="))
    assert set(payload) == {"v", "scope", "id", "captured_at", "display_from", "display_subject", "date", "sender", "subject", "suggested"}
    assert payload["display_from"] == normal["from"] and payload["display_subject"] == normal["subject"]
    assert "SECRET BODY" not in str(payload) and "To" not in payload
    fake.operations.clear()
    response = c.post(BASE + "/category-projection", json={"items": [{"id": "765-3"}, {"id": "765-5"}]})
    assert response.status_code == 200, response.text
    assert len(fake.operations) == 2 and all(op == "FETCH" and "HEADER.FIELDS" in args[1] for op, args in fake.operations)
    assert response.json()["items"][1]["projection_ref"] is None
    assert state(store) == []
    # Captured facts don't assert current membership: removed mail can be reclassified
    # without being fetched or presented as freshly read. Explicit Refresh owns existence.
    del fake.rows["3"]
    fake.operations.clear()
    response = c.post(BASE + "/category-projection", json={"items": [{"id": normal["id"], "projection_ref": normal["projection_ref"]}]})
    assert response.status_code == 200 and fake.operations == []
    assert response.json()["items"][0]["source_captured_at"] == payload["captured_at"]
    assert c.post(BASE + "/category-projection", json={"items": [{"id": "765-1"}]}).status_code == 404  # Non-alias.


def test_projection_denials_bounds_foreign_refs_epoch_and_unavailable(api, store, monkeypatch):
    c, fake = api
    rows = captured(c)
    original = rows[0]
    path = BASE + "/category-projection"
    fake.operations.clear()
    for body in ({"items": []}, {"items": rows * 9}, {"items": [original, original]},
                 {"items": [{"id": "765-3", "projection_ref": "x" * 8193}]},
                 {"items": [{"id": 3}]}, {"items": rows, "scope": "foreign"}):
        assert c.post(path, json=body).status_code == 422
    assert c.post(path, content=b' ' * (256 * 1024 + 1), headers={"Content-Type": "application/json"}).status_code == 413
    for raw in (b'{"items":[],"items":[]}', b'{', b'\xff'):
        assert c.post(path, content=raw, headers={"Content-Type": "application/json"}).status_code == 422
    assert c.post(path, json={"items": rows}, headers={"Origin": "https://foreign.invalid"}).status_code == 403
    assert c.post(path, json={"items": rows}, headers={"X-Clinical-Key": "wrong"}).status_code == 401
    ref = original["projection_ref"]
    bad = {**original, "projection_ref": ref[:-1] + ("a" if ref[-1] != "a" else "b")}
    result = c.post(path, json={"items": [bad]})
    assert result.status_code == 422 and ref not in result.text
    other = {**original, "id": "765-999"}
    assert c.post(path, json={"items": [other]}).status_code == 409
    monkeypatch.setenv("ICLOUD_CLINICAL_ALIAS", "different@icloud.com")
    assert c.post(path, json={"items": rows}).status_code == 409
    monkeypatch.setenv("ICLOUD_CLINICAL_ALIAS", "clinic@icloud.com")
    @contextmanager
    def epoch(*args):
        yield fake, "999"
    monkeypatch.setattr(mail_source, "_session", epoch)
    assert c.post(path, json={"items": rows}).status_code == 409
    assert fake.operations == [] and state(store) == []
    monkeypatch.setattr(pref, "read_preferences", lambda *_: (_ for _ in ()).throw(pref.unavailable()))
    assert c.post(path, json={"items": rows}).status_code == 503
    assert fake.operations == []


def test_projection_signed_shape_key_rotation_and_stream_limits(api, monkeypatch):
    import base64
    import hashlib
    import hmac
    import json
    import time
    c, fake=api
    rows=captured(c)
    entry=rows[0]
    encoded=entry["projection_ref"].split(".")[0]
    facts=json.loads(base64.urlsafe_b64decode(encoded+"="*(-len(encoded)%4)))
    def signed(value):
        raw=json.dumps(value,separators=(",",":")).encode()
        body=base64.urlsafe_b64encode(raw).rstrip(b"=")
        return body.decode()+"."+hmac.new(b"synthetic-clinical-key",mail_source._PROJECTION_NAMESPACE+body,hashlib.sha256).hexdigest()
    fake.operations.clear()
    path=BASE+"/category-projection"
    for invalid in ({**facts,"v":2},{**facts,"captured_at":True},{**facts,"captured_at":int(time.time())+61},
                    {**facts,"display_subject":"x"*221},{**facts,"body":"Never allowed"},
                    {**facts,"sender":"x"*255},{**facts,"suggested":"unknown"}):
        response=c.post(path,json={"items":[{**entry,"projection_ref":signed(invalid)}]})
        assert response.status_code==422 and "Never allowed" not in response.text
    # A correctly authenticated new key still cannot reuse a source proof from the old key.
    monkeypatch.setenv("CLINICAL_DATA_KEY","rotated-synthetic-key")
    response=c.post(path,json={"items":rows},headers={"X-Clinical-Key":"rotated-synthetic-key"})
    assert response.status_code==422
    monkeypatch.setenv("CLINICAL_DATA_KEY","synthetic-clinical-key")
    response=c.post(path,content=iter([b" "*65536]*5),headers={"Content-Type":"application/json"})
    assert response.status_code==413
    response=c.post(path,content=b"["*2000+b"]"*2000,headers={"Content-Type":"application/json"})
    assert response.status_code==422
    assert fake.operations==[]
