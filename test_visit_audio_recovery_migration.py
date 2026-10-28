"""Least-privilege contract for the Visit Audio recovery migration."""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest

import clinical_visit_audio as audio
from clinical_data import ClinicalBase


SQL = (Path(__file__).resolve().parent /
       "cockpit/migrations/20261028_visit_audio_recovery_aiven.sql").read_text()


def test_recovery_schema_is_migration_owned_and_content_free():
    assert audio.VisitAudioRecoveryORM.metadata is audio.VisitAudioRecoveryBase.metadata
    assert audio.VisitAudioRecoveryORM.metadata is not ClinicalBase.metadata
    assert set(audio.VisitAudioRecoveryORM.__table__.columns.keys()) == {
        "id", "state", "expires_at"}
    assert ".create(" not in inspect.getsource(audio.attach_visit_audio)

    normalized = " ".join(SQL.split()).lower()
    assert "create table public.clinical_visit_audio_recovery" in normalized
    assert "create index ix_clinical_visit_audio_recovery_state" in normalized
    assert "create index ix_clinical_visit_audio_recovery_expires_at" in normalized
    assert "grant select, insert, delete on table" in normalized
    assert "grant update (state)" in normalized
    assert "has_schema_privilege('cockpit_app', 'public', 'create')" in normalized
    assert "grant create" not in normalized
    for forbidden in ("audio", "transcript", "patient_id", "clinical_content"):
        assert forbidden not in " ".join(
            column.name for column in audio.VisitAudioRecoveryORM.__table__.columns)


@pytest.mark.skipif(os.environ.get("VISIT_AUDIO_RECOVERY_PG_TEST") != "synthetic-only",
                    reason="requires disposable PostgreSQL service")
def test_recovery_admin_migration_privileges_and_one_shot_state():
    psycopg = pytest.importorskip("psycopg")
    from psycopg import errors

    dsn = ("host=127.0.0.1 port=" + os.environ.get("VISIT_AUDIO_RECOVERY_PG_PORT", "5432")
           + " user=postgres dbname=postgres password="
           + os.environ["VISIT_AUDIO_RECOVERY_PG_PASSWORD"])
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute("CREATE ROLE cockpit_app LOGIN")
        admin.execute("CREATE DATABASE osteoporosis")
    with psycopg.connect(dsn.replace("dbname=postgres", "dbname=osteoporosis"),
                         autocommit=True) as db:
        db.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
        db.execute("GRANT USAGE ON SCHEMA public TO cockpit_app")
        db.execute(SQL)
        assert db.execute(
            "SELECT has_table_privilege('cockpit_app',"
            "'public.clinical_visit_audio_recovery','DELETE')").fetchone()[0]
        assert db.execute(
            "SELECT has_column_privilege('cockpit_app',"
            "'public.clinical_visit_audio_recovery','state','UPDATE')").fetchone()[0]
        assert not db.execute(
            "SELECT has_column_privilege('cockpit_app',"
            "'public.clinical_visit_audio_recovery','expires_at','UPDATE')").fetchone()[0]
        assert not db.execute(
            "SELECT has_schema_privilege('cockpit_app','public','CREATE')").fetchone()[0]

        db.execute("SET ROLE cockpit_app")
        db.execute("INSERT INTO public.clinical_visit_audio_recovery "
                   "VALUES ('synthetic-capability','active',now() + interval '5 minutes')")
        assert db.execute("UPDATE public.clinical_visit_audio_recovery SET state='consumed' "
                          "WHERE id='synthetic-capability'").rowcount == 1
        with pytest.raises(errors.InsufficientPrivilege):
            db.execute("UPDATE public.clinical_visit_audio_recovery SET expires_at=now()")
        db.execute("DELETE FROM public.clinical_visit_audio_recovery "
                   "WHERE expires_at <= now() + interval '10 minutes'")
        db.execute("RESET ROLE")
        with pytest.raises(errors.RaiseException):
            db.execute(SQL)
