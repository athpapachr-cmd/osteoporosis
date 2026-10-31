"""Disposable PostgreSQL 17 contract test for the OFF-production follow-through store.

This file uses only the GitHub Actions localhost postgres service with a
fixed invented database, username and password. Never use a clinic URL.
No clinical patient records, IMAP provider or real recipient is touched.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from clinical_followthrough_cycle import Actor, CycleConflict, CycleForbidden
from clinical_followthrough_durable_synthetic import (
    DurableFixtureBase, DurableSyntheticFollowthrough,
    VisitObligationFixture, ObligationEventFixture, SourceClaimFixture,
)

pytestmark = pytest.mark.skipif(
    os.getenv("FOLLOWTHROUGH_PG_SYNTHETIC") != "ci-postgres17",
    reason="Disposable localhost PostgreSQL 17 CI only",
)

PG_URL = (
    "postgresql+psycopg://postgres:synthetic-ci-only"
    "@127.0.0.1:5432/synthetic_followthrough_ci"
)
DOCTOR = Actor("synthetic-clinician-pg01", "clinician")
STAFF = Actor("synthetic-secretary-pg01", "secretary")
SOURCE = Actor("synthetic-source-pg01", "source")
VISIT = "synthetic-pg-visit-01"
PENDING = "synthetic-pg-pending-01"


@pytest.fixture
def pg():
    if os.getenv("FOLLOWTHROUGH_PG_SYNTHETIC") != "ci-postgres17":
        pytest.skip("PostgreSQL fixture is CI-only")
    engine = create_engine(
        PG_URL, pool_size=8, max_overflow=0,
        connect_args={"connect_timeout": 5}, pool_pre_ping=True,
    )
    try:
        with engine.connect() as db:
            assert db.scalar(text("SELECT current_database()")) == "synthetic_followthrough_ci"
            assert db.scalar(text("SELECT current_user")) == "postgres"
            assert db.scalar(text("SELECT inet_server_port()")) == 5432
        # Guaranteed ephemeral GitHub service. Explicit test-only DDL.
        DurableFixtureBase.metadata.create_all(engine)
        yield engine
    finally:
        DurableFixtureBase.metadata.drop_all(engine)
        engine.dispose()


def store(engine):
    return DurableSyntheticFollowthrough(
        engine,
        qualified_signed_pending=lambda visit, pending: (
            visit, pending
        ) == (VISIT, PENDING),
    )


def signed(owner, required=("report", "images")):
    return owner.register(
        DOCTOR, visit_ref=VISIT, request_ref="synthetic-pg-mri-01",
        required=required, mode="signed_pending", pending_ref=PENDING,
    )["case_id"]


def secondary(owner):
    return owner.register(
        DOCTOR, visit_ref="synthetic-pg-no-chart-visit-02",
        request_ref="synthetic-pg-mri-second-02", required=("report",),
        mode="provisional_visit",
    )["case_id"]


def act(owner, case, actor, action, n, version=None, **attrs):
    state = owner.view(case, DOCTOR)
    return owner.act(
        case, actor, action=action,
        expected_version=state["version"] if version is None else version,
        idempotency_key=f"synthetic-postgres-idem-{n:05d}",
        **attrs,
    )


def receive(owner, case, n, part="report", uid="synthetic-pg-mail-uid-01",
            version="revision-01"):
    return act(
        owner, case, SOURCE, "received", n,
        part=part, source_ref=uid, source_version=version,
        proof="qualified_order",
    )


def counts(engine):
    with Session(engine) as db:
        return (
            db.scalar(select(func.count()).select_from(VisitObligationFixture)),
            db.scalar(select(func.count()).select_from(ObligationEventFixture)),
            db.scalar(select(func.count()).select_from(SourceClaimFixture)),
        )


def test_pg_restart_reuses_signed_pending_and_versions(pg):
    owner = store(pg)
    case = signed(owner)
    assert counts(pg) == (1, 0, 0)
    receive(owner, case, 1)
    act(owner, case, DOCTOR, "identity_verified", 2,
        evidence="qualified_referral",
        evidence_ref="synthetic-pg-referral-verified")
    reviewed = act(
        owner, case, DOCTOR, "review_notify", 3,
        part="report", source_version="revision-01",
    )
    assert reviewed["contact_status"] == "to_notify"
    assert owner.secretary_view(case, STAFF)["doctor_reviewed"] is True
    assert counts(pg) == (1, 3, 1)

    # Different SQLAlchemy engine/pool simulates process restart.
    another_engine = create_engine(
        PG_URL, pool_size=4, max_overflow=0,
        connect_args={"connect_timeout": 5},
    )
    try:
        reopened = store(another_engine)
        assert reopened.view(case, DOCTOR) == reviewed
        assert signed(reopened) == case
        assert reopened.history(case, DOCTOR) == owner.history(case, DOCTOR)
        receive(
            reopened, case, 4, part="images",
            uid="synthetic-pg-mail-uid-02",
        )
        act(
            reopened, case, DOCTOR, "review_no_contact", 5,
            part="images", source_version="revision-01",
        )
        assert reopened.view(case, DOCTOR)["stage"] == "awaiting_secretary"
        act(reopened, case, STAFF, "contact_attempt", 6)
        informed = act(reopened, case, STAFF, "patient_notified", 7)
        assert not informed["complete"] and informed["stage"] == "awaiting_callback"
        done = act(reopened, case, DOCTOR, "callback_complete", 8)
        assert done["complete"]
        assert counts(pg) == (1, 8, 2)
    finally:
        another_engine.dispose()


def test_pg_same_uid_concurrently_arrives_for_only_one_visit(pg):
    owner = store(pg)
    a = signed(owner, required=("report",))
    b = secondary(owner)
    uid = "synthetic-pg-collision-uid-01"

    def race(one, no):
        try:
            return receive(owner, one, no, uid=uid)
        except CycleConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(race, a, 10),
            pool.submit(race, b, 11),
        ]
        results = [future.result(timeout=15) for future in futures]
    assert sum(value is not None for value in results) == 1
    assert sorted((owner.view(a, DOCTOR)["version"],
                   owner.view(b, DOCTOR)["version"])) == [0, 1]
    assert counts(pg) == (2, 1, 1)
    with Session(pg) as db:
        claim = db.get(SourceClaimFixture, uid)
        assert claim is not None
        assert claim.obligation_id in (a, b)


def test_pg_parallel_duplicate_poll_is_one_event_and_idempotent(pg):
    owner = store(pg)
    case = signed(owner, required=("report",))
    def read_again(n):
        return receive(owner, case, n, uid="synthetic-pg-replayed-uid-01")
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(read_again, range(20, 24)))
    assert all(x["version"] == 1 for x in rows)
    assert counts(pg) == (1, 1, 1)
    assert receive(owner, case, 24, uid="synthetic-pg-replayed-uid-01") == rows[0]


def test_pg_stale_cas_and_replay_payload_conflict_leave_audit_untouched(pg):
    owner = store(pg)
    case = signed(owner, required=("report",))
    first = receive(owner, case, 30)
    assert first["version"] == 1
    assert receive(owner, case, 30) == first
    with pytest.raises(CycleConflict):
        act(
            owner, case, DOCTOR, "review_no_contact", 31, version=0,
            part="report", source_version="revision-01",
        )
    with pytest.raises(CycleConflict):
        receive(owner, case, 30, version="revision-02")
    assert counts(pg) == (1, 1, 1)


def test_pg_corrected_source_requires_new_explicit_review_and_no_stale_contact(pg):
    owner = store(pg)
    case = signed(owner, required=("report",))
    receive(owner, case, 40)
    act(
        owner, case, DOCTOR, "identity_verified", 41,
        evidence="clinician_verified",
        evidence_ref="synthetic-pg-contact-proof-01",
    )
    act(
        owner, case, DOCTOR, "review_notify", 42,
        part="report", source_version="revision-01",
    )
    act(owner, case, STAFF, "patient_notified", 43)
    assert owner.view(case, DOCTOR)["stage"] == "awaiting_callback"
    latest = receive(owner, case, 44, version="revision-02")
    assert latest["stage"] == "awaiting_doctor_review"
    assert latest["reviewed"] == {}
    assert latest["contact_status"] == "not_requested"
    with pytest.raises(CycleForbidden):
        owner.secretary_view(case, STAFF)
    with pytest.raises(CycleConflict):
        act(
            owner, case, DOCTOR, "review_no_contact", 45,
            part="report", source_version="revision-01",
        )
    assert counts(pg) == (1, 5, 1)


def test_pg_db_rejects_non_synthetic_refs_and_foreign_event(pg):
    owner = store(pg)
    signed(owner, required=("report",))
    stamp = datetime(2026, 10, 31)
    # Raw SQL cannot bypass fixture-only marker and FK constraints.
    with pytest.raises(IntegrityError):
        with Session(pg) as db, db.begin():
            db.add(VisitObligationFixture(
                id=str(uuid4()), visit_ref="real-patient-identity",
                request_ref="synthetic-pg-fake-request",
                mode="provisional_visit", pending_ref=None,
                required_parts=["report"], verified_contact=False,
                version=0, created_at=stamp, updated_at=stamp,
            ))
            db.flush()
    with pytest.raises(IntegrityError):
        with Session(pg) as db, db.begin():
            db.add(SourceClaimFixture(
                source_ref="synthetic-pg-foreign-key",
                obligation_id=str(uuid4()), part="report",
            ))
            db.flush()
    assert counts(pg) == (1, 0, 0)


def test_pg_no_chart_cycle_stays_separate_and_staff_cannot_review(pg):
    owner = store(pg)
    case = owner.register(
        DOCTOR,
        visit_ref="synthetic-pg-no-chart-visit-03",
        request_ref="synthetic-pg-labs-request-03",
        mode="provisional_visit", required=("labs",),
    )["case_id"]
    with pytest.raises(CycleForbidden):
        owner.view(case, STAFF)
    with pytest.raises(CycleForbidden):
        owner.register(
            STAFF, visit_ref="synthetic-pg-fake-visit-01",
            request_ref="synthetic-pg-fake-request-01",
            mode="provisional_visit", required=("labs",),
        )
    receive(
        owner, case, 60, part="labs",
        uid="synthetic-pg-labs-uid-03",
    )
    done = act(
        owner, case, DOCTOR, "review_no_contact", 61,
        part="labs", source_version="revision-01",
    )
    assert done["complete"] and done["pending_ref"] is None
    assert counts(pg) == (1, 2, 1)
