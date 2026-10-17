"""Ephemeral PostgreSQL A5 Visit Brief grants test; no real patients/providers/services.

Requires a new synthetic postgres:17 service on localhost; never use Aiven.
"""
from __future__ import annotations

import os
from pathlib import Path

import psycopg
from psycopg import errors

BASE = Path(__file__).resolve().parent
FIRST = BASE / "cockpit/migrations/20261017_visit_brief_staff_access.sql"
GRANTS = BASE / "cockpit/migrations/20261017_visit_brief_cockpit_app_grants.sql"
PASSWORD = os.getenv("A5_SYNTHETIC_PASSWORD", "")


def connect(db="osteoporosis", user="postgres", password=None):
    return psycopg.connect(
        host="127.0.0.1", port=5432, dbname=db,
        user=user, password=password or PASSWORD, autocommit=True
    )


def apply(conn, file):
    # Without bound params psycopg sends the transaction via simple protocol.
    conn.execute(file.read_text(encoding="utf-8"), prepare=False)


def denied(conn, sql):
    try:
        conn.execute(sql)
    except errors.InsufficientPrivilege:
        return
    raise AssertionError("Forbidden operation succeeded: " + sql[:50])


def test():
    if os.getenv("A5_SYNTHETIC_ONLY") != "yes" or not PASSWORD:
        raise RuntimeError("STOP: synthetic isolated Postgres gate not set")
    if not FIRST.is_file() or not GRANTS.is_file():
        raise RuntimeError("STOP: missing reviewed SQL fixture")
    with connect("postgres") as admin:
        # Never operate against a pre-existing db or role, even with test opt-in.
        if admin.execute("SELECT 1 FROM pg_database WHERE datname='osteoporosis'").fetchone():
            raise RuntimeError("STOP: pre-existing database")
        if admin.execute("SELECT 1 FROM pg_roles WHERE rolname='cockpit_app'").fetchone():
            raise RuntimeError("STOP: pre-existing role")
        admin.execute(
            "CREATE ROLE cockpit_app LOGIN PASSWORD 'synthetic-a5-only' "
            "NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION"
        )
        admin.execute("CREATE DATABASE osteoporosis")
    with connect() as admin:
        apply(admin, FIRST)
        apply(admin, GRANTS)
        apply(admin, GRANTS)  # idempotent exact-schema re-run
        assert admin.execute(
            "SELECT count(*) FROM pg_tables WHERE schemaname='public'"
        ).fetchone()[0] == 3
        table_names = (
            "clinical_booking_patient_links",
            "clinical_booking_patient_link_audit",
            "clinical_visit_brief_staff_access_audit",
        )
        for name in table_names:
            for operation in ("SELECT", "INSERT"):
                assert admin.execute(
                    "SELECT has_table_privilege('cockpit_app',%s,%s)",
                    ("public." + name, operation),
                ).fetchone()[0]
            assert not admin.execute(
                "SELECT has_table_privilege('cockpit_app',%s,'DELETE')",
                ("public." + name,),
            ).fetchone()[0]
        assert admin.execute(
            "SELECT has_table_privilege('cockpit_app',"
            "'public.clinical_booking_patient_links','UPDATE')"
        ).fetchone()[0]
        for name in table_names[1:]:
            assert not admin.execute(
                "SELECT has_table_privilege('cockpit_app',%s,'UPDATE')",
                ("public." + name,),
            ).fetchone()[0]
        assert not admin.execute(
            "SELECT has_schema_privilege('cockpit_app','public','CREATE')"
        ).fetchone()[0]

    with connect(user="cockpit_app", password="synthetic-a5-only") as app:
        app.execute("""INSERT INTO clinical_booking_patient_links
          (namespace,booking_uid,patient_id,state,revision,evidence_type,confirmed_at,updated_at)
          VALUES ('synthetic','UID','SYN-PATIENT','needs_reconfirmation',1,
                  'protected_chart_identifiers',now(),now())""")
        app.execute("UPDATE clinical_booking_patient_links SET state='revoked' WHERE booking_uid='UID'")
        assert app.execute(
            "SELECT state FROM clinical_booking_patient_links WHERE booking_uid='UID'"
        ).fetchone()[0] == "revoked"
        app.execute("""INSERT INTO clinical_booking_patient_link_audit
          (id,namespace,booking_uid,event,revision,operator_type,occurred_at)
          VALUES ('SYN-LINK','synthetic','UID','revoked',2,'clinician',now())""")
        app.execute("""INSERT INTO clinical_visit_brief_staff_access_audit
          (id,principal_id,role,booking_uid,event,outcome,occurred_at)
          VALUES ('SYN-AUDIT','clinic-clinician','clinician','UID','link_revoke','allowed',now())""")
        assert app.execute(
            "SELECT count(*) FROM clinical_visit_brief_staff_access_audit"
        ).fetchone()[0] == 1
        denied(app, "DELETE FROM clinical_booking_patient_links WHERE booking_uid='UID'")
        denied(app, "UPDATE clinical_booking_patient_link_audit SET event='wrong'")
        denied(app, "DELETE FROM clinical_visit_brief_staff_access_audit")
        denied(app, "CREATE TABLE unauthorized_a5 (id int)")

    with connect() as admin:
        # A changed table must fail before any grant commits.
        admin.execute("REVOKE SELECT, INSERT, UPDATE ON clinical_booking_patient_links FROM cockpit_app")
        admin.execute("ALTER TABLE clinical_booking_patient_links ADD COLUMN extra_synthetic text")
        try:
            apply(admin, GRANTS)
        except psycopg.Error as exc:
            # The deliberate negative case leaves BEGIN aborted; explicitly
            # end it before verifying that no privilege grant was committed.
            admin.execute("ROLLBACK")
            assert "A5 STOP: Visit Brief table schema drift" in str(exc), str(exc)
        else:
            raise AssertionError("Schema drift was silently accepted")
        assert not admin.execute(
            "SELECT has_table_privilege('cockpit_app',"
            "'public.clinical_booking_patient_links','SELECT')"
        ).fetchone()[0]
    print("PASS A5: isolated PostgreSQL 17, least privilege, DML, idempotence, schema drift")


if __name__ == "__main__":
    test()
