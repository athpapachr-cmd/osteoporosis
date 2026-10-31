"""Offline synthetic durable follow-through: restart, CAS, dedupe and role tests."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session
import pytest

from clinical_followthrough_cycle import Actor, CycleConflict, CycleForbidden, CycleNotFound
from clinical_followthrough_durable_synthetic import (
    DurableFixtureBase, DurableSyntheticFollowthrough,
    VisitObligationFixture, ObligationEventFixture, SourceClaimFixture,
)

DOCTOR = Actor("synthetic-clinician-01", "clinician")
STAFF = Actor("synthetic-secretary-01", "secretary")
SOURCE = Actor("synthetic-mail-source-01", "source")
UNTRUSTED = Actor("synthetic-anonymous-01", "unknown")
REAL_ROLE = Actor("clinic-clinician", "clinician")

VISIT = "synthetic-visit-mri-01"
PENDING = "synthetic-pending-mri-01"
REQ = "synthetic-mri-request-01"


def new_engine(path):
    e = create_engine(
        "sqlite:///" + str(path),
        connect_args={"check_same_thread": False, "timeout": 10},
    )
    @event.listens_for(e, "connect")
    def enforce_fk(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")
    return e


@pytest.fixture
def setup(tmp_path):
    path = tmp_path / "durable-synthetic.sqlite"
    engine = new_engine(path)
    # Disposable isolated fixture ONLY. No production boot or migration call.
    DurableFixtureBase.metadata.create_all(engine)
    qualify = lambda v, p: (v, p) == (VISIT, PENDING)
    store = DurableSyntheticFollowthrough(
        engine, qualified_signed_pending=qualify
    )
    yield store, engine, path, qualify
    engine.dispose()


def signed(store, required=("report", "images")):
    return store.register(
        DOCTOR, visit_ref=VISIT, request_ref=REQ, required=required,
        mode="signed_pending", pending_ref=PENDING,
    )["case_id"]


def provisional(store, key="labs"):
    return store.register(
        DOCTOR, visit_ref="synthetic-unrecorded-visit-01",
        request_ref="synthetic-request-" + key, required=(key,),
        mode="provisional_visit",
    )["case_id"]


def apply(store, case, actor, action, no, version=None, **attrs):
    return store.act(
        case, actor, action=action,
        expected_version=store.view(case, DOCTOR)["version"] if version is None else version,
        idempotency_key="synthetic-idem-" + str(no).zfill(8),
        **attrs,
    )


def receipt(store, case, part="report", version="r1", no=1, ref=None, actor=SOURCE):
    return apply(
        store, case, actor, "received", no, part=part,
        source_ref=ref or "synthetic-mail-" + part,
        source_version=version,
        proof="qualified_order" if actor.role == "source" else "clinician_verified",
    )


def test_restart_retains_one_signed_pending_and_original_event_history(setup):
    store, engine, path, qualify = setup
    case = signed(store)
    start = store.view(case, DOCTOR)
    assert start["pending_ref"] == PENDING and start["mode"] == "signed_pending"
    assert start["stage"] == "awaiting_result"

    stage1 = receipt(store, case, "report", "r1", 1)
    assert stage1["stage"] == "partially_received"
    apply(store, case, DOCTOR, "identity_verified", 2,
          evidence="qualified_referral", evidence_ref="synthetic-qualified-referral")
    after_review = apply(store, case, DOCTOR, "review_notify", 3,
                         part="report", source_version="r1")
    assert after_review["stage"] == "partially_received"
    assert store.secretary_view(case, STAFF)["doctor_reviewed"] is True
    assert "source_ref" not in str(store.secretary_view(case, STAFF))
    history = store.history(case, DOCTOR)
    assert len(history) == 3

    # New SQLAlchemy Engine: real file-backed DB persistence, not in-memory.
    engine.dispose()
    reopened_engine = new_engine(path)
    restarted = DurableSyntheticFollowthrough(
        reopened_engine, qualified_signed_pending=qualify)
    assert restarted.view(case, DOCTOR) == after_review
    assert restarted.history(case, DOCTOR) == history
    assert signed(restarted) == case

    images = receipt(restarted, case, "images", "r1", 4)
    assert images["stage"] == "awaiting_doctor_review"
    apply(restarted, case, DOCTOR, "review_no_contact", 5,
          part="images", source_version="r1")
    assert restarted.view(case, DOCTOR)["stage"] == "awaiting_secretary"
    assert not restarted.view(case, DOCTOR)["complete"]
    apply(restarted, case, STAFF, "contact_attempt", 6)
    notified = apply(restarted, case, STAFF, "patient_notified", 7)
    assert notified["contact_status"] == "notified"
    assert notified["stage"] == "awaiting_callback"
    finish = apply(restarted, case, DOCTOR, "callback_complete", 8)
    assert finish["stage"] == "complete"
    with Session(reopened_engine) as db:
        assert db.scalar(select(func.count()).select_from(VisitObligationFixture)) == 1
        assert db.scalar(select(func.count()).select_from(ObligationEventFixture)) == 8
        assert db.scalar(select(func.count()).select_from(SourceClaimFixture)) == 2
    reopened_engine.dispose()


def test_no_chart_lab_close_requires_no_fabricated_encounter_or_extra_tap(setup):
    store, _engine, _path, _qualify = setup
    case = provisional(store)
    assert store.view(case, DOCTOR)["pending_ref"] is None
    receipt(store, case, "labs", "r1", 10)
    result = apply(store, case, DOCTOR, "review_no_contact", 11,
                   part="labs", source_version="r1")
    assert result["complete"] and result["version"] == 2
    assert result["verified_contact"] is False
    with pytest.raises(CycleForbidden):
        store.secretary_view(case, STAFF)


def test_one_canonical_pending_and_one_visit_request(setup):
    store, _engine, _path, _qualify = setup
    a = signed(store)
    assert signed(store) == a
    with pytest.raises(CycleConflict):
        store.register(DOCTOR, visit_ref=VISIT, request_ref=REQ,
                       required=("report",), mode="signed_pending",
                       pending_ref=PENDING)
    with pytest.raises(CycleConflict):
        store.register(DOCTOR, visit_ref=VISIT, request_ref="synthetic-mri-request-02",
                       required=("report", "images"), mode="signed_pending",
                       pending_ref=PENDING)
    with pytest.raises(CycleForbidden):
        store.register(DOCTOR, visit_ref=VISIT, request_ref="synthetic-wrong-pending",
                       required=("report",), mode="signed_pending",
                       pending_ref="synthetic-fake-pending")
    with pytest.raises(CycleForbidden):
        store.register(DOCTOR, visit_ref="synthetic-other-visit",
                       request_ref="synthetic-unverified-signed",
                       required=("report",), mode="signed_pending", pending_ref=PENDING)


def test_claimed_source_reused_only_for_same_case_and_part(setup):
    store, _engine, _path, _qualify = setup
    case = signed(store)
    other = store.register(DOCTOR, visit_ref="synthetic-visit-mri-02",
                           request_ref="synthetic-request-mri-02", required=("report",),
                           mode="provisional_visit")["case_id"]
    receipt(store, case, "report", "r1", 20, ref="synthetic-shared-uid-01")
    with pytest.raises(CycleConflict):
        receipt(store, other, "report", "r1", 21, ref="synthetic-shared-uid-01")
    with pytest.raises(CycleConflict):
        receipt(store, case, "images", "r1", 22, ref="synthetic-shared-uid-01")
    assert store.view(other, DOCTOR)["version"] == 0


def test_duplicate_poll_and_actor_scoped_idempotency_do_not_write_extra_event(setup):
    store, engine, _path, _qualify = setup
    case = signed(store, required=("report",))
    data = receipt(store, case, "report", "r1", 30)
    replay = receipt(store, case, "report", "r1", 30)
    another_poll = receipt(store, case, "report", "r1", 31)
    assert data == replay == another_poll
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(ObligationEventFixture)) == 1
    with pytest.raises(CycleConflict):
        receipt(store, case, "report", "r2", 30)
    with pytest.raises(CycleConflict):
        apply(store, case, DOCTOR, "review_no_contact", 32, version=0,
              part="report", source_version="r1")
    assert store.view(case, DOCTOR)["version"] == 1


def test_corrected_report_reopens_review_and_revokes_old_secretary_instruction(setup):
    store, engine, _path, _qualify = setup
    case = signed(store, required=("report",))
    receipt(store, case, "report", "r1", 40)
    apply(store, case, DOCTOR, "identity_verified", 41,
          evidence="clinician_verified", evidence_ref="synthetic-verified-contact")
    apply(store, case, DOCTOR, "review_notify", 42, part="report", source_version="r1")
    apply(store, case, STAFF, "patient_notified", 43)
    assert store.view(case, DOCTOR)["stage"] == "awaiting_callback"
    new = receipt(store, case, "report", "r2", 44)
    assert new["stage"] == "awaiting_doctor_review"
    assert not new["reviewed"] and new["contact_status"] == "not_requested"
    with pytest.raises(CycleForbidden):
        store.secretary_view(case, STAFF)
    with pytest.raises(CycleConflict):
        apply(store, case, DOCTOR, "review_no_contact", 45,
              part="report", source_version="r1")
    assert not store.view(case, DOCTOR)["complete"]
    fresh = apply(store, case, DOCTOR, "review_no_contact", 46,
                  part="report", source_version="r2")
    assert fresh["complete"]
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(ObligationEventFixture)) == 6


def test_actor_guards_refuse_real_roles_extra_attributes_and_unproven_contact(setup):
    store, engine, _path, _qualify = setup
    case = signed(store, required=("report",))
    with pytest.raises(CycleForbidden):
        store.view(case, REAL_ROLE)
    with pytest.raises(CycleForbidden):
        store.secretary_view(case, REAL_ROLE)
    with pytest.raises(CycleForbidden):
        receipt(store, case, "report", "r1", 50, actor=UNTRUSTED)
    with pytest.raises(CycleConflict):
        apply(store, case, DOCTOR, "received", 51, part="report",
              source_ref="synthetic-uid-51", source_version="r1",
              proof="clinician_verified", patient_name="REAL NAME DISALLOWED")
    with pytest.raises(CycleForbidden):
        apply(store, case, DOCTOR, "identity_verified", 52,
              evidence="name_match", evidence_ref="synthetic-name-only")
    with pytest.raises(CycleForbidden):
        apply(store, case, STAFF, "review_no_contact", 53,
              part="report", source_version="r1")
    with pytest.raises(CycleForbidden):
        store.history(case, STAFF)
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(ObligationEventFixture)) == 0


def test_no_contact_notification_or_callback_without_doctor_review(setup):
    store, _engine, _path, _qualify = setup
    case = signed(store, required=("report",))
    receipt(store, case, "report", "r1", 60)
    with pytest.raises(CycleForbidden):
        apply(store, case, STAFF, "patient_notified", 61)
    with pytest.raises(CycleConflict):
        apply(store, case, DOCTOR, "callback_complete", 62)
    with pytest.raises(CycleForbidden):
        apply(store, case, DOCTOR, "review_notify", 63,
              part="report", source_version="r1")
    result = apply(store, case, DOCTOR, "review_no_contact", 64,
                   part="report", source_version="r1")
    assert result["complete"] and result["contact_status"] == "not_requested"


def test_same_source_two_concurrent_opaque_visits_never_both_claim_it(setup):
    store, engine, _path, _qualify = setup
    a = signed(store, required=("report",))
    b = store.register(DOCTOR, visit_ref="synthetic-unrecorded-visit-02",
                       request_ref="synthetic-request-report-02", required=("report",),
                       mode="provisional_visit")["case_id"]

    def arrive(case_id, no):
        try:
            return receipt(store, case_id, "report", "r1", no,
                           ref="synthetic-uid-unique-crosscase-01")
        except CycleConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda x: arrive(*x), [(a, 71), (b, 72)]))
    assert sum(x is not None for x in results) == 1
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(SourceClaimFixture)) == 1
        assert db.scalar(select(func.count()).select_from(ObligationEventFixture)) == 1


def test_untrusted_no_chart_pending_and_undefined_actor_are_rejected(setup):
    store, _engine, _path, _qualify = setup
    with pytest.raises(CycleForbidden):
        store.register(DOCTOR, visit_ref="synthetic-missing-chart",
                       request_ref="synthetic-nonexistent-pending", required=("labs",),
                       mode="signed_pending", pending_ref="synthetic-not-signed")
    with pytest.raises(CycleConflict):
        store.register(DOCTOR, visit_ref="synthetic-no-chart",
                       request_ref="synthetic-impossible", required=("labs",),
                       mode="provisional_visit", pending_ref="synthetic-fake-pending")
    with pytest.raises(CycleForbidden):
        store.register(REAL_ROLE, visit_ref="synthetic-visit-03",
                       request_ref="synthetic-request-03", required=("labs",),
                       mode="provisional_visit")
    with pytest.raises(CycleNotFound):
        store.view("synthetic-not-existing-case", DOCTOR)
