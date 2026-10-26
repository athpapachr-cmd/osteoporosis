"""Disposable PostgreSQL proof for the one-time Priority 3 admin migration."""

import os
from pathlib import Path

import pytest


SQL = (Path(__file__).resolve().parent / "cockpit/migrations/20261026_office_cases_aiven.sql").read_text()


@pytest.mark.skipif(os.environ.get("OFFICE_CASES_PG_TEST") != "synthetic-only",
                    reason="requires disposable PostgreSQL service")
def test_office_cases_admin_migration_privileges_and_constraints():
    psycopg = pytest.importorskip("psycopg")
    from psycopg import errors
    dsn = ("host=127.0.0.1 port=" + os.environ.get("OFFICE_CASES_PG_PORT", "5432")
           + " user=postgres dbname=postgres password=" + os.environ["OFFICE_CASES_PG_PASSWORD"])
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute("CREATE ROLE cockpit_app LOGIN")
        admin.execute("CREATE DATABASE osteoporosis")
    with psycopg.connect(dsn.replace("dbname=postgres", "dbname=osteoporosis"),
                         autocommit=True) as db:
        db.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
        db.execute("GRANT USAGE ON SCHEMA public TO cockpit_app")
        db.execute("CREATE TABLE public.clinical_surgery_queue(id varchar PRIMARY KEY)")
        db.execute("CREATE TABLE public.clinic_rf_applications(id varchar PRIMARY KEY)")
        db.execute(SQL)
        assert db.execute("SELECT to_regclass('public.clinical_accident_cases') IS NOT NULL").fetchone()[0]
        assert db.execute("SELECT has_table_privilege('cockpit_app','public.clinical_accident_payments','INSERT')").fetchone()[0]
        assert not db.execute("SELECT has_table_privilege('cockpit_app','public.clinical_accident_payments','DELETE')").fetchone()[0]
        assert not db.execute("SELECT has_column_privilege('cockpit_app','public.clinical_surgery_details','rf_application_id','UPDATE')").fetchone()[0]
        db.execute("INSERT INTO public.clinical_surgery_queue VALUES ('synthetic-surgery')")
        db.execute("INSERT INTO public.clinic_rf_applications VALUES ('synthetic-rf')")
        db.execute("INSERT INTO public.clinical_surgery_details VALUES ('synthetic-surgery','synthetic-rf','09:30',now(),now())")
        # PostgreSQL 17 reports 23503; PostgreSQL 18 uses 23001 for RESTRICT.
        with pytest.raises((errors.ForeignKeyViolation, errors.RestrictViolation)):
            db.execute("DELETE FROM public.clinic_rf_applications WHERE id='synthetic-rf'")
        db.execute("INSERT INTO public.clinical_surgery_queue VALUES ('another-surgery')")
        with pytest.raises(errors.UniqueViolation):
            db.execute("INSERT INTO public.clinical_surgery_details VALUES ('another-surgery','synthetic-rf',null,now(),now())")
        db.execute("INSERT INTO public.clinic_rf_applications VALUES ('another-rf')")
        db.execute("SET ROLE cockpit_app")
        db.execute("INSERT INTO public.clinical_surgery_details VALUES ('another-surgery','another-rf',null,now(),now())")
        db.execute("UPDATE public.clinical_surgery_details SET room_entry_time='10:15' WHERE surgery_id='another-surgery'")
        db.execute("INSERT INTO public.clinical_accident_cases VALUES ('synthetic-case','Synthetic','000000','2026-10-01',null,null,null,10000,1,now(),now())")
        db.execute("INSERT INTO public.clinical_accident_case_audit VALUES ('synthetic-audit','synthetic-case',1,'create','{}','{}',now())")
        db.execute("INSERT INTO public.clinical_accident_payments VALUES ('synthetic-payment','synthetic-case','synthetic-request',2500,'2026-10-03','Synthetic payer',now(),null,null)")
        with pytest.raises(errors.InsufficientPrivilege):
            db.execute("DELETE FROM public.clinical_accident_payments WHERE id='synthetic-payment'")
        db.execute("RESET ROLE")
        with pytest.raises(errors.RaiseException):
            db.execute(SQL)
