"""Synthetic-only HTTP demonstration of a single visit-to-callback cycle.

All actors, mail UIDs and visit identifiers are invented by the server.
NEVER attach this router to the real /clinical API or treat its test-role
transitions as qualified secretary/clinician identity in production.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from clinical_followthrough_cycle import (
    Actor, CycleConflict, CycleForbidden, CycleNotFound, SyntheticCycleLedger,
)

DOCTOR = Actor("synthetic-demo-clinician", "clinician")
SECRETARY = Actor("synthetic-demo-secretary", "secretary")
SOURCE = Actor("synthetic-demo-qualified-source", "source")
RECEIPTS = {
    ("signed_pending", "report"): ("synthetic-mail-report-01", "revision-01"),
    ("signed_pending", "images"): ("synthetic-mail-images-02", "revision-01"),
    ("provisional_visit", "labs"): ("synthetic-mail-labs-03", "revision-01"),
}


class SyntheticAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=1, max_length=64)
    expected_version: int = Field(ge=0)
    idempotency_key: str = Field(min_length=16, max_length=100)
    part: str | None = None
    choice: str | None = None


def build_synthetic_followthrough_router(require_test_access, verified_fixture_pending):
    """Only under /clinical/synthetic-161, with existing Clinical Key dependency."""
    router = APIRouter(prefix="/followthrough", tags=["synthetic-followthrough"])
    ledger = SyntheticCycleLedger()

    def invoke(operation):
        try:
            return operation()
        except CycleForbidden as err:
            raise HTTPException(403, str(err)) from None
        except (CycleConflict, CycleNotFound) as err:
            raise HTTPException(409, str(err)) from None

    def qualified_case(case):
        """The exact signed source must remain eligible after preparation."""
        if case["mode"] != "signed_pending":
            return
        pending_id, encounter_id = verified_fixture_pending()
        if (pending_id, encounter_id) != (case["pending_ref"], case["visit_ref"]):
            raise CycleForbidden("Signed Pending no longer matches this visit")

    def guarded(case_id, operation):
        def check_then_act():
            with ledger.lock:
                qualified_case(ledger.view(case_id))
                return operation()
        return invoke(check_then_act)

    def state():
        cases = []
        for case_id in tuple(ledger.works):
            case = ledger.view(case_id)
            if case["mode"] == "signed_pending":
                try:
                    qualified_case(case)
                except (HTTPException, CycleForbidden):
                    # Read-only invalidation projection. Preserve the event audit.
                    case.update(source_valid=False, stage="source_invalidated",
                                complete=False)
                else:
                    case["source_valid"] = True
            cases.append(case)
        return {"mode": "synthetic_only", "live_provider_connected": False,
                "real_messages_used": False, "cases": cases}

    @router.post("/prepare")
    def prepare(actor=Depends(require_test_access)):
        # Check actual *synthetic* seeded pending + signed encounter via caller.
        # The API NEVER manufactures a signed clinical encounter or patient.
        signed_pending_id, encounter_ref = verified_fixture_pending()
        signed = ledger.register(
            DOCTOR, visit_ref=encounter_ref, request_ref="synthetic-mri-work-01",
            required=("report", "images"), mode="signed_pending",
            pending_ref=signed_pending_id,
        )
        if not signed["verified_contact"]:
            ledger.act(signed["case_id"], DOCTOR, action="identity_verified",
                       expected_version=signed["version"],
                       idempotency_key="synthetic-fixture-verified-01",
                       evidence="qualified_referral",
                       evidence_ref="synthetic-qualified-ref-01")
        ledger.register(
            DOCTOR, visit_ref="synthetic-visit-without-chart",
            request_ref="synthetic-labs-work-01", required=("labs",),
            mode="provisional_visit",
        )
        return state()

    @router.get("")
    def get(actor=Depends(require_test_access)):
        return state()

    @router.get("/secretary/{case_id}")
    def secretary(case_id: str, actor=Depends(require_test_access)):
        # This is a synthetic, explicitly labelled role *simulation*, NOT a
        # production staff endpoint or proof of actual staff authentication.
        return guarded(case_id, lambda: ledger.secretary_view(case_id, SECRETARY))

    @router.post("/simulate-arrival")
    def simulate_arrival(req: SyntheticAction, actor=Depends(require_test_access)):
        def action():
            case = ledger.view(req.case_id)
            evidence = RECEIPTS.get((case["mode"], req.part))
            if evidence is None:
                raise CycleConflict("Unsupported simulated provider receipt")
            return ledger.act(req.case_id, SOURCE, action="received",
                              expected_version=req.expected_version,
                              idempotency_key=req.idempotency_key,
                              part=req.part, source_ref=evidence[0],
                              source_version=evidence[1], proof="qualified_order")
        return guarded(req.case_id, action)

    @router.post("/doctor-review")
    def doctor_review(req: SyntheticAction, actor=Depends(require_test_access)):
        def action():
            current = ledger.view(req.case_id)
            material = current["received"].get(req.part)
            if material is None:
                raise CycleConflict("Nothing received for review")
            if req.choice not in ("notify", "no_contact"):
                raise CycleConflict("Expected an explicit clinical choice")
            return ledger.act(
                req.case_id, DOCTOR,
                action="review_notify" if req.choice == "notify" else "review_no_contact",
                expected_version=req.expected_version,
                idempotency_key=req.idempotency_key, part=req.part,
                source_version=material["source_version"],
            )
        return guarded(req.case_id, action)

    @router.post("/secretary-result")
    def secretary_result(req: SyntheticAction, actor=Depends(require_test_access)):
        def action():
            if req.choice not in ("attempted", "notified"):
                raise CycleConflict("Only observed contact outcomes")
            return ledger.act(
                req.case_id, SECRETARY,
                action="patient_notified" if req.choice == "notified" else "contact_attempt",
                expected_version=req.expected_version,
                idempotency_key=req.idempotency_key,
            )
        return guarded(req.case_id, action)

    @router.post("/doctor-callback")
    def doctor_callback(req: SyntheticAction, actor=Depends(require_test_access)):
        return guarded(req.case_id, lambda: ledger.act(
            req.case_id, DOCTOR, action="callback_complete",
            expected_version=req.expected_version,
            idempotency_key=req.idempotency_key,
        ))

    return router, ledger
