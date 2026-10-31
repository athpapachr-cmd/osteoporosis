"""Synthetic source/visit continuity: no real patient, email or communication."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from clinical_followthrough_cycle import (
    Actor, CycleConflict, CycleForbidden, CycleNotFound, SyntheticCycleLedger,
)

DOCTOR = Actor("synthetic-clinician-01", "clinician")
STAFF = Actor("synthetic-secretary-01", "secretary")
SOURCE = Actor("synthetic-qualified-mail-source", "source")
UNKNOWN = Actor("synthetic-anonymous", "unknown")


def registered(required=("report", "images"), mode="signed_pending"):
    store = SyntheticCycleLedger()
    record = store.register(
        DOCTOR, visit_ref="synthetic-visit-20261031",
        request_ref="synthetic-mri-right-knee",
        required=required, mode=mode,
        pending_ref="synthetic-signed-pending-01" if mode == "signed_pending" else None,
    )
    return store, record["case_id"]


def do(store, case, actor, action, version=None, serial=1, **payload):
    return store.act(
        case, actor, action=action,
        expected_version=(store.view(case)["version"] if version is None else version),
        idempotency_key=f"test-event-key-{serial:05}", **payload,
    )


def test_signed_visit_one_obligation_full_cycle_and_quiet_closure():
    store, case = registered()
    assert store.view(case)["stage"] == "awaiting_result"
    assert store.view(case)["pending_ref"] == "synthetic-signed-pending-01"
    report = do(store, case, SOURCE, "received", serial=1, part="report",
                source_ref="synthetic-imap-uid-01", source_version="revision-01",
                proof="qualified_order")
    assert report["stage"] == "partially_received"
    assert report["received"]["report"]["source_version"] == "revision-01"
    assert "images" not in report["received"]
    with pytest.raises(CycleForbidden):
        do(store, case, DOCTOR, "review_notify", serial=2,
           part="report", source_version="revision-01")

    do(store, case, DOCTOR, "identity_verified", serial=3,
       evidence="clinician_verified", evidence_ref="synthetic-verified-chart-01")
    reviewed = do(store, case, DOCTOR, "review_notify", serial=4,
                  part="report", source_version="revision-01")
    assert reviewed["reviewed"]["report"] == "revision-01"
    assert not reviewed["complete"]
    secretary = store.secretary_view(case, STAFF)
    assert secretary["doctor_reviewed"]
    assert secretary["status"] == "to_notify"
    assert "source_ref" not in secretary and "report" not in str(secretary)
    with pytest.raises(CycleConflict):
        do(store, case, DOCTOR, "callback_complete", serial=5)
    do(store, case, STAFF, "contact_attempt", serial=6)
    assert store.secretary_view(case, STAFF)["status"] == "attempted"
    informed = do(store, case, STAFF, "patient_notified", serial=7)
    assert informed["contact_status"] == "notified"
    assert not informed["complete"], "Contact is not doctor-patient closure"
    assert informed["stage"] == "partially_received"

    images = do(store, case, SOURCE, "received", serial=8, part="images",
                source_ref="synthetic-imap-uid-02", source_version="revision-01",
                proof="qualified_order")
    assert images["contact_status"] == "notified"
    assert images["stage"] == "awaiting_doctor_review"
    do(store, case, DOCTOR, "review_no_contact", serial=9, part="images",
       source_version="revision-01")
    assert store.view(case)["stage"] == "awaiting_callback"
    closed = do(store, case, DOCTOR, "callback_complete", serial=10)
    assert closed["complete"] and closed["stage"] == "complete"
    events = store.history(case, DOCTOR)
    assert len(events) == 9
    assert [e["action"] for e in events][-3:] == [
        "received", "review_no_contact", "callback_complete"
    ]


def test_no_chart_visit_can_begin_without_fake_patient_or_signed_encounter():
    store, case = registered(required=("labs",), mode="provisional_visit")
    item = store.view(case)
    assert item["pending_ref"] is None and item["mode"] == "provisional_visit"
    do(store, case, DOCTOR, "received", serial=11, part="labs",
       source_ref="synthetic-labs-uid-01", source_version="revision-01",
       proof="clinician_verified")
    reviewed = do(store, case, DOCTOR, "review_no_contact", serial=12,
                  part="labs", source_version="revision-01")
    assert reviewed["complete"], "No unnecessary second close tap"
    assert reviewed["verified_contact"] is False
    with pytest.raises(CycleForbidden):
        store.secretary_view(case, STAFF)


def test_invalid_auto_link_and_name_only_identity_rejected():
    store, case = registered(required=("report",))
    with pytest.raises(CycleForbidden):
        do(store, case, SOURCE, "received", serial=20, part="report",
           source_ref="synthetic-name-only", source_version="revision-01",
           proof="filename_match")
    assert store.view(case)["version"] == 0
    with pytest.raises(CycleForbidden):
        do(store, case, DOCTOR, "identity_verified", serial=21,
           evidence="name_match", evidence_ref="synthetic-name-match")
    assert store.view(case)["version"] == 0
    with pytest.raises(CycleForbidden):
        do(store, case, STAFF, "identity_verified", serial=22,
           evidence="clinician_verified", evidence_ref="synthetic-evidence")


def test_role_isolation_and_clinician_summary_cannot_be_updated_by_staff():
    store, case = registered(required=("report",))
    with pytest.raises(CycleForbidden):
        store.register(STAFF, visit_ref="synthetic-visit-other",
                       request_ref="synthetic-mri-other", required=("report",),
                       mode="provisional_visit")
    with pytest.raises(CycleForbidden):
        do(store, case, STAFF, "received", serial=30, part="report",
           source_ref="synthetic-uid-30", source_version="revision-01",
           proof="qualified_order")
    with pytest.raises(CycleForbidden):
        do(store, case, STAFF, "review_no_contact", serial=31,
           part="report", source_version="revision-01")
    with pytest.raises(CycleForbidden):
        store.history(case, STAFF)
    with pytest.raises(CycleForbidden):
        do(store, case, UNKNOWN, "patient_notified", serial=32)


def test_source_id_scope_duplicate_poll_and_cross_visit_conflict():
    store, first = registered(required=("report",))
    second = store.register(DOCTOR, visit_ref="synthetic-visit-02",
                            request_ref="synthetic-mri-other", required=("report",),
                            mode="provisional_visit")["case_id"]
    event = dict(part="report", source_ref="synthetic-mail-unique-01",
                 source_version="revision-01", proof="qualified_order")
    first_result = do(store, first, SOURCE, "received", serial=40, **event)
    repeat = do(store, first, SOURCE, "received", serial=41, version=0, **event)
    assert repeat["version"] == first_result["version"] == 1
    assert len(store.history(first, DOCTOR)) == 1
    with pytest.raises(CycleConflict):
        do(store, second, SOURCE, "received", serial=42, **event)


def test_corrected_report_invalidates_prior_review_and_notification():
    store, case = registered(required=("report",))
    do(store, case, SOURCE, "received", serial=50, part="report",
       source_ref="synthetic-mail-corrected", source_version="revision-01",
       proof="qualified_order")
    do(store, case, DOCTOR, "identity_verified", serial=51,
       evidence="qualified_referral", evidence_ref="synthetic-referral-01")
    do(store, case, DOCTOR, "review_notify", serial=52,
       part="report", source_version="revision-01")
    do(store, case, STAFF, "patient_notified", serial=53)
    assert store.view(case)["stage"] == "awaiting_callback"
    updated = do(store, case, SOURCE, "received", serial=54, part="report",
                 source_ref="synthetic-mail-corrected", source_version="revision-02",
                 proof="qualified_order")
    assert updated["stage"] == "awaiting_doctor_review"
    assert updated["reviewed"] == {}
    assert updated["contact_status"] == "not_requested"
    with pytest.raises(CycleForbidden):
        store.secretary_view(case, STAFF)
    with pytest.raises(CycleConflict):
        do(store, case, DOCTOR, "review_no_contact", serial=55,
           part="report", source_version="revision-01")
    closed = do(store, case, DOCTOR, "review_no_contact", serial=56,
                part="report", source_version="revision-02")
    assert closed["complete"]


def test_stale_cas_and_reused_idempotency_do_not_change_state():
    store, case = registered(required=("report",))
    first = do(store, case, SOURCE, "received", serial=60, version=0,
               part="report", source_ref="synthetic-mail-60",
               source_version="revision-01", proof="qualified_order")
    same_key = do(store, case, SOURCE, "received", serial=60, version=0,
                  part="report", source_ref="synthetic-mail-60",
                  source_version="revision-01", proof="qualified_order")
    assert first == same_key
    with pytest.raises(CycleConflict):
        do(store, case, SOURCE, "received", serial=60, version=1,
           part="report", source_ref="synthetic-mail-60",
           source_version="revision-02", proof="qualified_order")
    with pytest.raises(CycleConflict):
        do(store, case, DOCTOR, "review_no_contact", serial=61, version=0,
           part="report", source_version="revision-01")
    assert store.view(case)["version"] == 1


def test_concurrent_duplicate_arrivals_create_only_one_event():
    store, case = registered(required=("report",))
    def delivered(i):
        return do(store, case, SOURCE, "received", serial=70 + i, version=0,
                  part="report", source_ref="synthetic-mail-concurrent",
                  source_version="revision-01", proof="qualified_order")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(delivered, range(8)))
    assert len({r["version"] for r in results}) == 1
    assert len(store.history(case, DOCTOR)) == 1


def test_unsigned_pending_not_faked_from_provisional_and_request_idempotency():
    store, case = registered(required=("report",), mode="provisional_visit")
    again = store.register(DOCTOR, visit_ref="synthetic-visit-20261031",
                           request_ref="synthetic-mri-right-knee",
                           required=("report",), mode="provisional_visit")
    assert again["case_id"] == case
    with pytest.raises(CycleConflict):
        store.register(DOCTOR, visit_ref="synthetic-visit-20261031",
                       request_ref="synthetic-mri-right-knee",
                       required=("report",), mode="signed_pending",
                       pending_ref="synthetic-forged-signed-01")
    with pytest.raises(CycleNotFound):
        store.view("synthetic-not-existing")


def test_refuse_false_closure_until_report_received_and_reviewed():
    store, case = registered(required=("report",))
    with pytest.raises(CycleConflict):
        do(store, case, DOCTOR, "review_no_contact", serial=80,
           part="report", source_version="revision-01")
    with pytest.raises(CycleConflict):
        do(store, case, DOCTOR, "callback_complete", serial=81)
    assert store.view(case)["stage"] == "awaiting_result"
