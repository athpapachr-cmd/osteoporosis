"""Existing signed ClinicalPending ↔ durable follow-through bridge (offline only).

Uses actual clinical table models but ONLY invented fixtures in a disposable DB.
Never contacts clinic PostgreSQL, Apple iCloud or any patient messaging provider.
"""
from __future__ import annotations

from datetime import datetime
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import Session
import pytest

from clinical_data import ClinicalBase, PatientORM, EncounterORM, ClinicalPendingORM
from clinical_followthrough_cycle import Actor, CycleConflict, CycleForbidden
from clinical_followthrough_durable_synthetic import (
    DurableFixtureBase, DurableSyntheticFollowthrough,
    VisitObligationFixture, ObligationEventFixture,
)
from clinical_followthrough_signed_pending_synthetic import (
    SyntheticSignedPendingBinding, SCENARIO,
)

DOCTOR = Actor("synthetic-clinician-unique01", "clinician")
SECRETARY = Actor("synthetic-secretary-unique01", "secretary")
SOURCE = Actor("synthetic-mail-source-unique01", "source")
REAL_ROLE = Actor("clinic-clinician", "clinician")
PATIENT = "synthetic-patient-01"
VISIT = "synthetic-signed-visit-01"
PENDING = "synthetic-signed-pending-01"
STAMP = datetime(2026, 10, 1, 9, 10, 0)


@pytest.fixture()
def setup(tmp_path):
    clinical = create_engine("sqlite:///" + str(tmp_path / "clinical-source.sqlite"),
                             connect_args={"check_same_thread": False})
    durable = create_engine("sqlite:///" + str(tmp_path / "durable-events.sqlite"),
                            connect_args={"check_same_thread": False})
    ClinicalBase.metadata.create_all(clinical)  # Disposable tests only, NEVER runtime.
    DurableFixtureBase.metadata.create_all(durable)
    source = {
        "signed": True,
        "signed_at": "2026-10-01T09:10:00+00:00",
        "pending_ids": [PENDING],
        "candidate": {
            "schema_version": "visit_capture_candidate_v1",
            "pending": [{
                "item_type": "imaging",
                "description": "Synthetic MRI images + report",
                "status": "awaiting_result",
            }],
        },
    }
    with Session(clinical) as db, db.begin():
        db.add(PatientORM(
            patient_id=PATIENT,
            demographics_json={
                "full_name": "Synthetic Fixture Only",
                "synthetic_scenario": SCENARIO,
            }, created_at=STAMP, updated_at=STAMP,
        ))
        db.add(EncounterORM(
            id=VISIT, patient_id=PATIENT, encounter_date="2026-10-01",
            status="completed",
            payload_json={"_visit_capture_v1": source},
            created_at=STAMP, updated_at=STAMP,
        ))
        db.add(ClinicalPendingORM(
            id=PENDING, patient_id=PATIENT, source_encounter_id=VISIT,
            item_type="imaging", description="Synthetic MRI images + report",
            trigger_text="Awaiting requested MRI",
            status="awaiting_result", responsible_role="patient",
            provenance_json=["synthetic-clinician-note"],
            created_at=STAMP, updated_at=STAMP,
        ))
    binding = SyntheticSignedPendingBinding(clinical)
    store = DurableSyntheticFollowthrough(
        durable, qualified_signed_pending=binding.qualify_refs
    )
    yield clinical, durable, binding, store
    clinical.dispose()
    durable.dispose()


def enroll(binding, store):
    return binding.register_existing(
        store, DOCTOR, patient_id=PATIENT, pending_id=PENDING,
        encounter_id=VISIT, required=("report", "images")
    )


def snapshot(clinical):
    with Session(clinical) as db:
        patient = db.get(PatientORM, PATIENT)
        visit = db.get(EncounterORM, VISIT)
        pending = db.get(ClinicalPendingORM, PENDING)
        return (dict(patient.demographics_json), dict(visit.payload_json),
                pending.status, pending.updated_at,
                db.scalar(select(func.count()).select_from(PatientORM)),
                db.scalar(select(func.count()).select_from(EncounterORM)),
                db.scalar(select(func.count()).select_from(ClinicalPendingORM)))


def act(store, case, actor, action, number, **attrs):
    return store.act(
        case, actor, action=action,
        expected_version=store.view(case, DOCTOR)["version"],
        idempotency_key="synthetic-binding-event-" + str(number).zfill(3),
        **attrs,
    )


def test_existing_signed_visit_registered_once_and_projections_share_pending(setup):
    clinical, durable, binding, store = setup
    original = snapshot(clinical)
    assert binding.project_pending(store, DOCTOR, patient_id=PATIENT,
                                   pending_id=PENDING, encounter_id=VISIT) is None
    case = enroll(binding, store)
    assert case["pending_ref"] == "synthetic-" + PENDING
    assert case["visit_ref"] == "synthetic-" + VISIT
    assert case["mode"] == "signed_pending"
    assert enroll(binding, store)["case_id"] == case["case_id"]

    with Session(durable) as db:
        assert db.scalar(select(func.count()).select_from(VisitObligationFixture)) == 1
    assert binding.project_pending(store, DOCTOR, patient_id=PATIENT,
                                   pending_id=PENDING, encounter_id=VISIT) == case
    assert snapshot(clinical) == original, "Signed patient record and Pending are never rewritten"


def test_existing_signed_pending_results_and_notification_roundtrip(setup):
    clinical, durable, binding, store = setup
    original = snapshot(clinical)
    case = enroll(binding, store)["case_id"]
    received = act(store, case, SOURCE, "received", 1, part="report",
                   source_ref="synthetic-imap-account1-validity1-uid1",
                   source_version="revision-01", proof="qualified_order")
    assert received["stage"] == "partially_received"
    assert "images" not in received["received"]
    assert received["reviewed"] == {}
    act(store, case, DOCTOR, "identity_verified", 2,
        evidence="qualified_referral", evidence_ref="synthetic-verified-referral")
    reviewed = act(store, case, DOCTOR, "review_notify", 3,
                   part="report", source_version="revision-01")
    assert not reviewed["complete"]
    assert store.secretary_view(case, SECRETARY)["doctor_reviewed"]
    act(store, case, STAFF := SECRETARY, "contact_attempt", 4)
    notified = act(store, case, STAFF, "patient_notified", 5)
    assert not notified["complete"] and notified["contact_status"] == "notified"

    act(store, case, SOURCE, "received", 6, part="images",
        source_ref="synthetic-imap-account1-validity1-uid2",
        source_version="revision-01", proof="qualified_order")
    act(store, case, DOCTOR, "review_no_contact", 7,
        part="images", source_version="revision-01")
    assert store.view(case, DOCTOR)["stage"] == "awaiting_callback"
    finished = act(store, case, DOCTOR, "callback_complete", 8)
    assert finished["complete"]
    projection = binding.project_pending(
        store, DOCTOR, patient_id=PATIENT,
        pending_id=PENDING, encounter_id=VISIT,
    )
    assert projection["case_id"] == case and projection["complete"]
    assert snapshot(clinical) == original
    with Session(durable) as db:
        assert db.scalar(select(func.count()).select_from(ObligationEventFixture)) == 8


@pytest.mark.parametrize("change", [
    "not_signed", "missing_pending_membership", "duplicate_pending_id",
    "missing_candidate", "wrong_signed_item_type", "unsigned_draft",
    "cancelled_pending", "cross_patient_pending", "incorrect_patient_marker",
    "malformed_id_array", "malformed_signed_at", "malformed_payload",
])
def test_fail_closed_on_missing_or_conflicting_signed_visit_evidence(setup, change):
    clinical, durable, binding, store = setup
    with Session(clinical) as db, db.begin():
        visit = db.get(EncounterORM, VISIT)
        pending = db.get(ClinicalPendingORM, PENDING)
        patient = db.get(PatientORM, PATIENT)
        payload = dict(visit.payload_json)
        signed = dict(payload["_visit_capture_v1"])
        if change == "not_signed":
            signed["signed"] = False
        elif change == "missing_pending_membership":
            signed["pending_ids"] = []
            signed["candidate"] = {"pending": []}
        elif change == "duplicate_pending_id":
            signed["pending_ids"] = [PENDING, PENDING]
            signed["candidate"] = {"pending": [{"item_type": "imaging"}] * 2}
        elif change == "missing_candidate":
            signed.pop("candidate")
        elif change == "wrong_signed_item_type":
            signed["candidate"] = {"pending": [{"item_type": "laboratory"}]}
        elif change == "unsigned_draft":
            visit.status = "draft"
        elif change == "cancelled_pending":
            pending.status = "cancelled"
        elif change == "cross_patient_pending":
            pending.patient_id = "synthetic-unrelated-patient"
        elif change == "incorrect_patient_marker":
            patient.demographics_json = {"synthetic_scenario": "another-scenario"}
        elif change == "malformed_id_array":
            signed["pending_ids"] = [{"id": PENDING}]
        elif change == "malformed_signed_at":
            signed["signed_at"] = "unverifiable"
        elif change == "malformed_payload":
            visit.payload_json = {"_visit_capture_v1": "not a signed source"}
        if change != "malformed_payload":
            payload["_visit_capture_v1"] = signed
            visit.payload_json = payload
    with pytest.raises(CycleForbidden):
        enroll(binding, store)
    with Session(durable) as db:
        assert db.scalar(select(func.count()).select_from(VisitObligationFixture)) == 0


def test_wrong_patient_visit_or_actor_must_not_create_obligation(setup):
    _clinical, durable, binding, store = setup
    with pytest.raises(CycleForbidden):
        binding.resolve(DOCTOR, patient_id="synthetic-other-patient",
                        pending_id=PENDING, encounter_id=VISIT)
    with pytest.raises(CycleForbidden):
        binding.resolve(DOCTOR, patient_id=PATIENT, pending_id=PENDING,
                        encounter_id="synthetic-other-visit")
    for actor in (SECRETARY, REAL_ROLE):
        with pytest.raises(CycleForbidden):
            binding.resolve(actor, patient_id=PATIENT, pending_id=PENDING,
                            encounter_id=VISIT)
    with pytest.raises(CycleConflict):
        binding.register_existing(
            store, DOCTOR, patient_id=PATIENT, pending_id=PENDING,
            encounter_id=VISIT, required=("labs",)
        )
    with Session(durable) as db:
        assert db.scalar(select(func.count()).select_from(VisitObligationFixture)) == 0


def test_no_synthetic_fixture_no_auto_link_even_with_matching_name(setup):
    clinical, durable, binding, store = setup
    with Session(clinical) as db, db.begin():
        patient = db.get(PatientORM, PATIENT)
        patient.demographics_json = {"full_name": "Synthetic Fixture Only"}
    with pytest.raises(CycleForbidden):
        enroll(binding, store)
    with Session(durable) as db:
        assert db.scalar(select(func.count()).select_from(VisitObligationFixture)) == 0


def test_post_signature_reassignment_blocks_future_projection(setup):
    clinical, _durable, binding, store = setup
    case = enroll(binding, store)
    assert binding.project_pending(
        store, DOCTOR, patient_id=PATIENT,
        pending_id=PENDING, encounter_id=VISIT,
    ) == case
    with Session(clinical) as db, db.begin():
        source = db.get(EncounterORM, VISIT)
        payload = dict(source.payload_json)
        signed = dict(payload["_visit_capture_v1"])
        signed["pending_ids"] = []
        signed["candidate"] = {"pending": []}
        payload["_visit_capture_v1"] = signed
        source.payload_json = payload
    with pytest.raises(CycleForbidden):
        binding.project_pending(store, DOCTOR, patient_id=PATIENT,
                                pending_id=PENDING, encounter_id=VISIT)
    # A source amendment must also reject new events in the durable owner.
    with pytest.raises(CycleForbidden):
        act(store, case["case_id"], SOURCE, "received", 44, part="report",
            source_ref="synthetic-revoked-mail-source",
            source_version="revision-01", proof="qualified_order")
    assert store.view(case["case_id"], DOCTOR)["version"] == 0
    # Evidence remains in audit; it is NOT silently moved to another visit.
    assert store.view(case["case_id"], DOCTOR)["pending_ref"] == "synthetic-" + PENDING


def test_wrong_verifier_cannot_bypass_synthetic_existing_pending(setup):
    _clinical, durable, binding, store = setup
    impostor = DurableSyntheticFollowthrough(
        durable, qualified_signed_pending=lambda v, p: True
    )
    with pytest.raises(CycleForbidden):
        binding.register_existing(
            impostor, DOCTOR, patient_id=PATIENT, pending_id=PENDING,
            encounter_id=VISIT, required=("report",),
        )


def test_no_chart_remains_provisional_without_fake_patient_or_encounter(setup):
    clinical, durable, binding, store = setup
    before = snapshot(clinical)
    provisional = store.register(
        DOCTOR, visit_ref="synthetic-minimal-visit-without-chart",
        request_ref="synthetic-labs-from-manual-visit",
        required=("labs",), mode="provisional_visit",
    )
    assert provisional["pending_ref"] is None
    assert provisional["mode"] == "provisional_visit"
    assert snapshot(clinical) == before
    assert binding.qualify_refs(provisional["visit_ref"], "synthetic-unlinked-pending") is False
    finished = act(
        store, provisional["case_id"], DOCTOR,
        "received", 20, part="labs",
        source_ref="synthetic-manual-lab-source",
        source_version="revision-01", proof="clinician_verified",
    )
    assert finished["stage"] == "awaiting_doctor_review"
    done = act(store, provisional["case_id"], DOCTOR,
               "review_no_contact", 21, part="labs", source_version="revision-01")
    assert done["complete"]
    assert snapshot(clinical) == before
