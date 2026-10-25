"""Bounded B1 synthetic PostgreSQL 17 integration tests; never connect to Aiven.

Run only in fresh disposable CI postgres with B1_SYNTHETIC_TEST_ONLY=yes.
No real identifiers, medical data, mail transport, DDL at application boot.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import psycopg
from psycopg import errors
from sqlalchemy import create_engine

from clinical_staff_auth import StaffActor
from clinical_attention_b1 import (
    B1Conflict, B1Disabled, B1Forbidden, B1NotFound,
    SyntheticAttentionPostgresB1,
)

BASE = Path(__file__).resolve().parent
MIGRATION = BASE / "cockpit/migrations/20261025_clinical_attention_b1_empty.sql"
PASSWORD = os.getenv("B1_SYNTHETIC_PASSWORD", "")


def connect(db="osteoporosis", user="postgres", password=None):
    return psycopg.connect(
        host="127.0.0.1", port=5432, dbname=db,
        user=user, password=password or PASSWORD, autocommit=True,
    )


def expect(exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"Expected {exc_type.__name__}")


def event_count(admin):
    return admin.execute("SELECT count(*) FROM clinical_attention_events").fetchone()[0]


def check():
    if (os.getenv("B1_SYNTHETIC_TEST_ONLY") != "yes"
            or not PASSWORD or not MIGRATION.is_file()):
        raise RuntimeError("STOP: isolated synthetic B1 execution gate missing")
    # Never conflate the new B1 store with any running production router/ORM.
    assert "clinical_attention_b1" not in (BASE / "main.py").read_text(encoding="utf-8")
    assert "ClinicalAttentionItemB1" not in (BASE / "clinical_data.py").read_text(encoding="utf-8")

    with connect("postgres") as admin:
        if admin.execute("SELECT 1 FROM pg_database WHERE datname='osteoporosis'").fetchone():
            raise RuntimeError("STOP: pre-existing database")
        if admin.execute("SELECT 1 FROM pg_roles WHERE rolname='cockpit_app'").fetchone():
            raise RuntimeError("STOP: pre-existing app role")
        admin.execute(
            "CREATE ROLE cockpit_app LOGIN PASSWORD 'synthetic-b1-only' "
            "NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION"
        )
        admin.execute("CREATE DATABASE osteoporosis")
    engine = create_engine(
        "postgresql+psycopg://cockpit_app:synthetic-b1-only@127.0.0.1:5432/osteoporosis"
    )
    try:
        # Initialization cannot perform boot DDL or paper over absent migration.
        expect(B1Disabled, lambda: SyntheticAttentionPostgresB1(engine))
        with connect() as admin:
            assert admin.execute(
                "SELECT to_regclass('public.clinical_attention_items')"
            ).fetchone()[0] is None
            sql = MIGRATION.read_text(encoding="utf-8")
            grant = ("GRANT SELECT, INSERT ON TABLE "
                     "public.clinical_attention_events TO cockpit_app;")
            assert sql.count(grant) == 1
            broken = sql.replace(
                grant, "GRANT SELECT ON TABLE public.b1_missing TO cockpit_app;", 1
            )
            try:
                admin.execute(broken, prepare=False)
            except psycopg.Error:
                admin.execute("ROLLBACK")
            else:
                raise AssertionError("Bad migration did not roll back")
            for table in ("clinical_attention_items", "clinical_attention_events"):
                assert admin.execute("SELECT to_regclass(%s)", ("public." + table,)
                                     ).fetchone()[0] is None
            admin.execute(sql, prepare=False)
            for table in ("clinical_attention_items", "clinical_attention_events"):
                assert admin.execute(
                    "SELECT has_table_privilege('cockpit_app',%s,'SELECT')",
                    ("public." + table,),
                ).fetchone()[0]
                assert admin.execute(
                    "SELECT has_table_privilege('cockpit_app',%s,'INSERT')",
                    ("public." + table,),
                ).fetchone()[0]
                assert not admin.execute(
                    "SELECT has_table_privilege('cockpit_app',%s,'DELETE')",
                    ("public." + table,),
                ).fetchone()[0]
            assert not admin.execute(
                "SELECT has_table_privilege('cockpit_app',"
                "'public.clinical_attention_events','UPDATE')"
            ).fetchone()[0]
            assert not admin.execute(
                "SELECT has_schema_privilege('cockpit_app','public','CREATE')"
            ).fetchone()[0]
            assert admin.execute("SELECT count(*) FROM clinical_attention_items"
                                 ).fetchone()[0] == 0

        os.environ.pop("B1_SYNTHETIC_TEST_ONLY", None)
        expect(B1Disabled, lambda: SyntheticAttentionPostgresB1(engine))
        os.environ["B1_SYNTHETIC_TEST_ONLY"] = "yes"
        store = SyntheticAttentionPostgresB1(engine)
        doctor = StaffActor("b1-synthetic-person-clinician1", "clinician")
        second = StaffActor("b1-synthetic-person-clinician2", "clinician")
        secretary = StaffActor("b1-synthetic-person-secretary1", "secretary")
        legacy = StaffActor("clinic-clinician", "clinician")

        expect(B1Forbidden, lambda: store.create_fixture(
            legacy, item_id="SYN-B1-A", source_key="SYN-B1-MESSAGE-A",
            source_type="lab_notice",
        ))
        expect(B1Forbidden, lambda: store.create_fixture(
            doctor, item_id="REAL-PATIENT-000", source_key="SYN-B1-MESSAGE-A",
            source_type="lab_notice",
        ))
        item = store.create_fixture(
            doctor, item_id="SYN-B1-A", source_key="SYN-B1-MESSAGE-A",
            source_type="lab_notice",
        )
        assert item["version"] == 0 and item["patient_id"] is None
        assert item["review_state"] == "unreviewed"
        prior_events = 1
        assert store.create_fixture(
            doctor, item_id="SYN-B1-A", source_key="SYN-B1-MESSAGE-A",
            source_type="lab_notice",
        ) == item

        with connect() as admin:
            assert event_count(admin) == prior_events
        expect(B1Conflict, lambda: store.create_fixture(
            doctor, item_id="SYN-B1-B", source_key="SYN-B1-MESSAGE-A",
            source_type="lab_notice",
        ))

        # Secretary and static shared principal cannot mutate/read an item.
        expect(B1Forbidden, lambda: store.read(secretary, "SYN-B1-A"))
        expect(B1Forbidden, lambda: store.mutate(
            legacy, item_id="SYN-B1-A", action="inspect_source",
            expected_version=0, idempotency_key=str(uuid4()),
        ))
        with connect() as admin:
            assert event_count(admin) == 2  # one attributable secretary denial

        before = store.read(doctor, "SYN-B1-A")
        assert before["identity_state"] == "unlinked"
        expect(B1NotFound, lambda: store.read(doctor, "SYN-B1-NOT-FOUND"))

        k = str(uuid4())
        viewed = store.mutate(
            doctor, item_id="SYN-B1-A", action="inspect_source",
            expected_version=0, idempotency_key=k,
        )
        assert viewed["version"] == 1 and viewed["review_state"] == "source_inspected"
        assert viewed["patient_id"] is None and viewed["disposition"] == "undecided"
        with connect() as admin:
            count = event_count(admin)
        assert store.mutate(
            doctor, item_id="SYN-B1-A", action="inspect_source",
            expected_version=0, idempotency_key=k,
        ) == viewed
        with connect() as admin:
            assert event_count(admin) == count  # exact replay has no additional effect
        expect(B1Conflict, lambda: store.mutate(
            doctor, item_id="SYN-B1-A", action="correct_source",
            expected_version=0, idempotency_key=k,
            new_revision="r2", source_scope="single_subject",
        ))

        correction = store.mutate(
            doctor, item_id="SYN-B1-A", action="correct_source",
            expected_version=1, idempotency_key=str(uuid4()),
            new_revision="r2", source_scope="single_subject",
        )
        assert correction["version"] == 2
        assert correction["active_source_revision"] == "r2"
        assert correction["review_state"] == "needs_re_review"
        assert correction["patient_id"] is None
        expect(B1Conflict, lambda: store.mutate(
            doctor, item_id="SYN-B1-A", action="inspect_source",
            expected_version=0, idempotency_key=k,
        ))
        expect(B1Conflict, lambda: store.mutate(
            doctor, item_id="SYN-B1-A", action="inspect_source",
            expected_version=1, idempotency_key=str(uuid4()),
        ))

        # New instance/connection retains state; no in-memory or session truth.
        restarted = SyntheticAttentionPostgresB1(engine)
        assert restarted.read(second, "SYN-B1-A")["version"] == 2
        assert restarted.read(second, "SYN-B1-A")["review_state"] == "needs_re_review"

        unknown = restarted.create_fixture(
            doctor, item_id="SYN-B1-MIXED", source_key="SYN-B1-PDF-MIXED",
            source_type="imaging_notice", source_scope="mixed_subjects",
        )
        assert unknown["patient_id"] is None and unknown["identity_state"] == "unlinked"
        assert "link_patient" not in dir(store)  # no real/synthetic chart mutation path

        # Independent attempts to act on the same source version are serialized.
        concurrent = restarted.create_fixture(
            doctor, item_id="SYN-B1-RACE", source_key="SYN-B1-SOURCE-RACE",
            source_type="gesy_availability",
        )
        assert concurrent["version"] == 0
        def race(who):
            try:
                restarted.mutate(
                    who, item_id="SYN-B1-RACE", action="inspect_source",
                    expected_version=0, idempotency_key=str(uuid4()),
                )
                return "allowed"
            except B1Conflict:
                return "conflict"
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(race, (doctor, second)))
        assert sorted(outcomes) == ["allowed", "conflict"]
        assert restarted.read(doctor, "SYN-B1-RACE")["version"] == 1

        # Database hard fence: even a buggy client cannot insert PHI or alter audit.
        with connect(user="cockpit_app", password="synthetic-b1-only") as app:
            try:
                app.execute(
                    "UPDATE clinical_attention_items SET patient_id='REAL-ID' "
                    "WHERE item_id='SYN-B1-A'"
                )
            except errors.CheckViolation:
                pass
            else:
                raise AssertionError("B1 database accepted real patient link")
            for sql_stmt in (
                "UPDATE clinical_attention_events SET outcome='allowed'",
                "DELETE FROM clinical_attention_events",
                "DELETE FROM clinical_attention_items",
                "CREATE TABLE public.unauthorized_b1(id integer)",
            ):
                try:
                    app.execute(sql_stmt)
                except errors.InsufficientPrivilege:
                    pass
                else:
                    raise AssertionError("Excess grant: " + sql_stmt)
        with connect() as admin:
            assert admin.execute(
                "SELECT count(*) FROM clinical_attention_items"
            ).fetchone()[0] == 3
            assert admin.execute(
                "SELECT count(*) FROM clinical_attention_events "
                "WHERE outcome='denied_409'"
            ).fetchone()[0] >= 3
            assert admin.execute(
                "SELECT count(*) FROM clinical_attention_events "
                "WHERE event_type='view' AND outcome='allowed'"
            ).fetchone()[0] >= 4
        print("PASS B1: PostgreSQL 17 synthetic-only durable CAS, source correction, "
              "idempotency, individual actor/denials, concurrency and least grants")
    finally:
        engine.dispose()


if __name__ == "__main__":
    check()
