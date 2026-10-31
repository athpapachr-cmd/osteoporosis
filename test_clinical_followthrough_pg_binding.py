"""Prove exact signed visit and durable obligations in the SAME disposable PG17 DB.

Strictly synthetic fixtures. No production database or real patient contact.
"""
from datetime import datetime
import os

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from clinical_data import ClinicalBase, ClinicalPendingORM, EncounterORM, PatientORM
from clinical_followthrough_cycle import Actor, CycleConflict, CycleForbidden
from clinical_followthrough_durable_synthetic import (
    DurableFixtureBase, DurableSyntheticFollowthrough, VisitObligationFixture,
    ObligationEventFixture,
)
from clinical_followthrough_signed_pending_synthetic import (
    SyntheticSignedPendingBinding, SCENARIO,
)

pytestmark = pytest.mark.skipif(
    os.getenv("FOLLOWTHROUGH_PG_SYNTHETIC") != "ci-postgres17",
    reason="Requires localhost disposable GitHub Actions PostgreSQL 17",
)

PG_URL = (
    "postgresql+psycopg://postgres:synthetic-ci-only"
    "@127.0.0.1:5432/synthetic_followthrough_ci"
)
DOCTOR = Actor("synthetic-clinician-postgres01", "clinician")
SECRETARY = Actor("synthetic-secretary-postgres01", "secretary")
SOURCE = Actor("synthetic-imap-qualified-postgres01", "source")
PATIENT = "synthetic-pg-patient-01"
VISIT = "synthetic-pg-signed-visit-01"
PENDING = "synthetic-pg-signed-pending-01"
STAMP = datetime(2026, 10, 31)


@pytest.fixture
def single_db():
    if os.getenv("FOLLOWTHROUGH_PG_SYNTHETIC") != "ci-postgres17":
        pytest.skip("Local synthetic PostgreSQL only")
    engine = create_engine(
        PG_URL, pool_size=8, max_overflow=0,
        connect_args={"connect_timeout": 5},
    )
    try:
        # DDL only in ephemeral, named GitHub Actions CI database.
        ClinicalBase.metadata.create_all(engine)
        DurableFixtureBase.metadata.create_all(engine)
        yield engine
    finally:
        DurableFixtureBase.metadata.drop_all(engine)
        ClinicalBase.metadata.drop_all(engine)
        engine.dispose()


def seed_source(engine, *, patient=PATIENT, visit=VISIT, pending=PENDING):
    source = {
        "_visit_capture_v1": {
            "signed": True,
            "signed_at": "2026-10-31T11:00:00+00:00",
            "pending_ids": [pending],
            "candidate": {"pending": [{
                "item_type": "imaging",
                "description": "Invented MRI",
                "status": "awaiting_result",
            }]},
        },
    }
    with Session(engine) as db, db.begin():
        db.add(PatientORM(
            patient_id=patient,
            demographics_json={"synthetic_scenario": SCENARIO,
                               "full_name": "SYNTHETIC ONLY"},
            created_at=STAMP, updated_at=STAMP,
        ))
        db.add(EncounterORM(
            id=visit, patient_id=patient,
            encounter_date="2026-10-31", status="completed",
            payload_json=source, created_at=STAMP, updated_at=STAMP,
        ))
        db.add(ClinicalPendingORM(
            id=pending, patient_id=patient, source_encounter_id=visit,
            item_type="imaging", description="Invented MRI",
            status="awaiting_result", responsible_role="patient",
            provenance_json=["synthetic-pg-fixture"],
            created_at=STAMP, updated_at=STAMP,
        ))


def owner(engine):
    bridge = SyntheticSignedPendingBinding(engine)
    store = DurableSyntheticFollowthrough(
        engine, qualified_signed_pending=bridge.qualify_refs
    )
    return bridge, store


def register(bridge, store):
    return bridge.register_existing(
        store, DOCTOR, patient_id=PATIENT,
        encounter_id=VISIT, pending_id=PENDING, required=("report", "images"),
    )


def act(store, case, actor, action, no, **attrs):
    return store.act(
        case, actor, action=action,
        expected_version=store.view(case, DOCTOR)["version"],
        idempotency_key="synthetic-pg-binding-idem-" + str(no).zfill(3),
        **attrs,
    )


def test_pg_canonical_signed_pending_and_full_followthrough_with_immutable_note(single_db):
    seed_source(single_db)
    bridge, store = owner(single_db)
    with Session(single_db) as db:
        original = db.get(EncounterORM, VISIT).payload_json.copy()
    case = register(bridge, store)
    assert register(bridge, store)["case_id"] == case["case_id"]
    assert case["pending_ref"] == "synthetic-" + PENDING
    case_id = case["case_id"]
    r = act(
        store, case_id, SOURCE, "received", 1,
        part="report", source_ref="synthetic-pg-imap-uvalidity1-uid101",
        source_version="revision-01", proof="qualified_order",
    )
    assert r["stage"] == "partially_received"
    act(store, case_id, DOCTOR, "identity_verified", 2,
        evidence="qualified_referral", evidence_ref="synthetic-pg-verified-ref-01")
    act(store, case_id, DOCTOR, "review_notify", 3,
        part="report", source_version="revision-01")
    task = store.secretary_view(case_id, SECRETARY)
    assert task["doctor_reviewed"] and "source_ref" not in str(task)
    act(
        store, case_id, SOURCE, "received", 4,
        part="images", source_ref="synthetic-pg-imap-uvalidity1-uid102",
        source_version="revision-01", proof="qualified_order",
    )
    act(store, case_id, DOCTOR, "review_no_contact", 5,
        part="images", source_version="revision-01")
    act(store, case_id, SECRETARY, "contact_attempt", 6)
    informed = act(store, case_id, SECRETARY, "patient_notified", 7)
    assert not informed["complete"] and informed["stage"] == "awaiting_callback"
    closed = act(store, case_id, DOCTOR, "callback_complete", 8)
    assert closed["complete"]
    projection = bridge.project_pending(
        store, DOCTOR, patient_id=PATIENT, pending_id=PENDING,
        encounter_id=VISIT,
    )
    assert projection["case_id"] == case_id and projection["complete"]
    with Session(single_db) as db:
        assert db.get(EncounterORM, VISIT).payload_json == original
        assert db.get(ClinicalPendingORM, PENDING).status == "awaiting_result"
        assert db.scalar(select(func.count()).select_from(PatientORM)) == 1
        assert db.scalar(select(func.count()).select_from(ClinicalPendingORM)) == 1
        assert db.scalar(select(func.count()).select_from(VisitObligationFixture)) == 1
        assert db.scalar(select(func.count()).select_from(ObligationEventFixture)) == 8


def test_pg_signed_pending_revocation_blocks_future_visit_projection(single_db):
    seed_source(single_db)
    bridge, store = owner(single_db)
    record = register(bridge, store)
    with Session(single_db) as db, db.begin():
        pending = db.get(ClinicalPendingORM, PENDING)
        pending.status = "cancelled"
    with pytest.raises(CycleForbidden):
        bridge.project_pending(
            store, DOCTOR, patient_id=PATIENT,
            pending_id=PENDING, encounter_id=VISIT,
        )
    with pytest.raises(CycleForbidden):
        register(bridge, store)
    assert store.view(record["case_id"], DOCTOR)["stage"] == "awaiting_result"


def test_pg_source_signature_or_pending_membership_cannot_be_faked(single_db):
    seed_source(single_db)
    bridge, store = owner(single_db)
    with Session(single_db) as db, db.begin():
        source = db.get(EncounterORM, VISIT)
        payload = dict(source.payload_json)
        signed = dict(payload["_visit_capture_v1"])
        signed["pending_ids"] = []
        signed["candidate"] = {"pending": []}
        payload["_visit_capture_v1"] = signed
        source.payload_json = payload
    with pytest.raises(CycleForbidden):
        register(bridge, store)
    with Session(single_db) as db:
        assert db.scalar(select(func.count()).select_from(VisitObligationFixture)) == 0
    # Name or calendar UID does NOT qualify missing signed-source membership.
    with pytest.raises(CycleForbidden):
        bridge.resolve(DOCTOR, patient_id=PATIENT,
                       pending_id=PENDING, encounter_id=VISIT)


def test_pg_no_chart_stays_independent_of_clinical_patient_table(single_db):
    bridge, store = owner(single_db)
    record = store.register(
        DOCTOR, visit_ref="synthetic-pg-no-chart-visit",
        request_ref="synthetic-pg-labs-request",
        mode="provisional_visit", required=("labs",),
    )
    assert record["pending_ref"] is None
    with Session(single_db) as db:
        assert db.scalar(select(func.count()).select_from(PatientORM)) == 0
        assert db.scalar(select(func.count()).select_from(EncounterORM)) == 0
    with pytest.raises(CycleForbidden):
        store.secretary_view(record["case_id"], SECRETARY)
    completed = act(
        store, record["case_id"], DOCTOR, "received", 10,
        part="labs", source_ref="synthetic-pg-manual-lab-uid",
        source_version="revision-01", proof="clinician_verified",
    )
    assert completed["stage"] == "awaiting_doctor_review"
    reviewed = act(store, record["case_id"], DOCTOR, "review_no_contact", 11,
                   part="labs", source_version="revision-01")
    assert reviewed["complete"]
    with Session(single_db) as db:
        assert db.scalar(select(func.count()).select_from(PatientORM)) == 0
        assert db.scalar(select(func.count()).select_from(EncounterORM)) == 0


def test_pg_guard_rejects_remote_or_non_disposable_database():
    engine = create_engine("postgresql+psycopg://postgres:wrong@invalid.internal/osteoporosis")
    try:
        with pytest.raises(CycleForbidden):
            SyntheticSignedPendingBinding(engine)
    finally:
        engine.dispose()


def test_pg_atomic_signed_source_guard_rechecks_inside_durable_transaction(single_db):
    seed_source(single_db)
    bridge, store = owner(single_db)
    original = bridge.qualify_in_transaction
    seen = []

    def invalidate_while_registering(db, visit_ref, pending_ref):
        # Simulate a source amendment/cancellation after the earlier
        # preliminary qualified resolve(), before durable event creation.
        assert db.get_bind() is single_db
        pending = db.get(ClinicalPendingORM, PENDING)
        pending.status = "cancelled"
        db.flush()
        seen.append("rechecked-inside-transaction")
        return original(db, visit_ref, pending_ref)

    bridge.qualify_in_transaction = invalidate_while_registering
    with pytest.raises(CycleForbidden):
        register(bridge, store)
    assert seen == ["rechecked-inside-transaction"]
    with Session(single_db) as db:
        assert db.get(ClinicalPendingORM, PENDING).status == "awaiting_result"
        assert db.scalar(select(func.count()).select_from(VisitObligationFixture)) == 0
        assert db.scalar(select(func.count()).select_from(ObligationEventFixture)) == 0


def test_pg_signed_qualification_cannot_be_moved_to_another_patient(single_db):
    seed_source(single_db)
    bridge, store = owner(single_db)
    first = register(bridge, store)
    with Session(single_db) as db, db.begin():
        pending = db.get(ClinicalPendingORM, PENDING)
        pending.patient_id = "synthetic-cross-patient-identifier"
    with pytest.raises(CycleForbidden):
        bridge.project_pending(
            store, DOCTOR, patient_id=PATIENT,
            pending_id=PENDING, encounter_id=VISIT,
        )
    assert store.view(first["case_id"], DOCTOR)["pending_ref"] == "synthetic-" + PENDING


def test_pg_transactional_qualification_for_provisional_visit_is_rejected(single_db):
    bridge, store = owner(single_db)
    with pytest.raises(CycleForbidden):
        store.register(
            DOCTOR,
            visit_ref="synthetic-pg-provisional-source",
            request_ref="synthetic-pg-test-request",
            mode="provisional_visit", required=("labs",),
            transactional_qualification=lambda db, v, p: True,
        )
    with Session(single_db) as db:
        assert db.scalar(select(func.count()).select_from(VisitObligationFixture)) == 0
