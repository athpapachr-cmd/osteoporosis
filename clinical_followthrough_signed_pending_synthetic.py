"""Qualified *existing* signed-visit Pending bridge — synthetic clinic fixtures ONLY.

This is the next offline adapter for #266. It READS the actual ClinicalPendingORM,
EncounterORM and PatientORM models without altering the clinical patient database.
Unlike name/UID similarity, membership in an immutable signed Visit Capture's
pending_ids is required before the same pending can enter follow-through.

It is NOT mounted from main.py, not a production identity/actor authority, and
not a real mail matching or secretary communication service. PostgreSQL is
permitted only for an explicitly identified, local disposable CI database.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from clinical_data import ClinicalPendingORM, EncounterORM, PatientORM
from clinical_followthrough_cycle import Actor, CycleConflict, CycleForbidden
from clinical_followthrough_durable_synthetic import (
    DurableSyntheticFollowthrough, VisitObligationFixture,
)

SCENARIO = "followthrough-signed-binding-v1"
_ACTIVE = frozenset(("open", "awaiting_patient", "awaiting_result"))
_ALLOWED_TYPE = {
    "imaging": frozenset(("images", "report")),
    "laboratory": frozenset(("labs",)),
}
_NATIVE = re.compile(r"^[A-Za-z0-9_-]{4,100}$")
_PREFIX = "synthetic-"


def _native(value: str) -> str:
    if not isinstance(value, str) or not _NATIVE.fullmatch(value):
        raise CycleForbidden("Unqualified synthetic record identity")
    return value


def _qualified_actor(actor: Actor) -> Actor:
    if (not isinstance(actor, Actor) or actor.role != "clinician"
            or not actor.principal.startswith("synthetic-clinician-")
            or not _NATIVE.fullmatch(actor.principal)):
        raise CycleForbidden("Synthetic clinician authority required")
    return actor


@dataclass(frozen=True)
class QualifiedExistingPending:
    """Opaque identifiers only; do not surface patient name or clinical note."""

    visit_ref: str
    pending_ref: str
    request_ref: str
    item_type: str


class SyntheticSignedPendingBinding:
    """Proof that a Pending row is explicitly present in its signed visit.

    The signed visit may be clinically amended; pending linkage must still be
    present in the *current* signed candidate, or we fail closed for separate
    clinician reconciliation. The binding does not interpret clinical results.
    """

    def __init__(self, clinical_engine: Engine):
        # The only PostgreSQL escape is an isolated, local GitHub Actions DB.
        # Never permit the production database host, name or account.
        if clinical_engine.dialect.name == "postgresql":
            url = clinical_engine.url
            if (os.getenv("FOLLOWTHROUGH_PG_SYNTHETIC") != "ci-postgres17"
                    or url.host != "127.0.0.1"
                    or url.port != 5432
                    or url.database != "synthetic_followthrough_ci"
                    or url.username != "postgres"):
                raise CycleForbidden("Only localhost disposable synthetic PostgreSQL")
            try:
                with clinical_engine.connect() as conn:
                    db = conn.scalar(text("SELECT current_database()"))
                    who = conn.scalar(text("SELECT current_user"))
                    # Docker publishes PostgreSQL to client localhost but
                    # PostgreSQL sees its container-bridge IP, not 127.0.0.1.
                    # The URL guard above verifies the only permitted client
                    # host/port, while these server facts verify fixture DB,
                    # role and PostgreSQL major version independently.
                    server_version = int(conn.scalar(text("SHOW server_version_num")))
                    if (db != "synthetic_followthrough_ci"
                            or who != "postgres"
                            or not (170000 <= server_version < 180000)):
                        raise CycleForbidden("Unqualified synthetic PostgreSQL target")
            except CycleForbidden:
                raise
            except Exception:
                raise CycleForbidden("Synthetic PostgreSQL target unavailable") from None
        elif clinical_engine.dialect.name != "sqlite":
            raise CycleForbidden("Fixture-only SQLite clinical storage")
        self.engine = clinical_engine

    @staticmethod
    def _signed_membership(db: Session, pending: ClinicalPendingORM):
        encounter = db.get(EncounterORM, pending.source_encounter_id)
        patient = db.get(PatientORM, pending.patient_id)
        if not encounter or not patient:
            raise CycleForbidden("Clinical patient or visit not qualified")
        if (not isinstance(patient.demographics_json, dict)
                or patient.demographics_json.get("synthetic_scenario") != SCENARIO
                or encounter.patient_id != pending.patient_id
                or encounter.status not in ("completed", "amended")
                or pending.status not in _ACTIVE):
            raise CycleForbidden("Visit, identity or obligation changed")

        payload = encounter.payload_json
        if not isinstance(payload, dict):
            raise CycleForbidden("Malformed signed visit payload")
        source = payload.get("_visit_capture_v1")
        if not isinstance(source, dict) or source.get("signed") is not True:
            raise CycleForbidden("No signed Visit Capture source")
        signed_at = source.get("signed_at")
        try:
            if not isinstance(signed_at, str):
                raise ValueError()
            datetime.fromisoformat(signed_at)
        except ValueError:
            raise CycleForbidden("Invalid signed visit evidence") from None

        ids = source.get("pending_ids")
        candidate = source.get("candidate")
        if (not isinstance(ids, list) or not isinstance(candidate, dict)
                or not isinstance(candidate.get("pending"), list)
                or any(not isinstance(x, str) or not _NATIVE.fullmatch(x)
                       for x in ids)
                or len(ids) != len(candidate["pending"]) or len(ids) != len(set(ids))
                or pending.id not in ids):
            raise CycleForbidden("Pending not independently bound to signed visit")
        index = ids.index(pending.id)
        signed_entry = candidate["pending"][index]
        if (not isinstance(signed_entry, dict)
                or signed_entry.get("item_type") != pending.item_type
                or pending.item_type not in _ALLOWED_TYPE):
            raise CycleForbidden("Signed requested examination does not match Pending")
        return encounter

    def resolve(self, actor: Actor, *, patient_id: str,
                pending_id: str, encounter_id: str) -> QualifiedExistingPending:
        _qualified_actor(actor)
        patient_id, pending_id, encounter_id = [
            _native(p) for p in (patient_id, pending_id, encounter_id)
        ]
        with Session(self.engine) as db:
            pending = db.get(ClinicalPendingORM, pending_id)
            if not pending or pending.patient_id != patient_id:
                raise CycleForbidden("No exact verified Pending patient binding")
            encounter = self._signed_membership(db, pending)
            if encounter.id != encounter_id:
                raise CycleForbidden("Pending belongs to another visit")
            return QualifiedExistingPending(
                visit_ref=_PREFIX + encounter.id,
                pending_ref=_PREFIX + pending.id,
                request_ref=_PREFIX + "request-" + pending.id,
                item_type=pending.item_type,
            )

    def qualify_refs(self, visit_ref: str, pending_ref: str) -> bool:
        """Read-through eligibility check used by durable register(), not a guess."""
        if (not isinstance(visit_ref, str) or not isinstance(pending_ref, str)
                or not visit_ref.startswith(_PREFIX)
                or not pending_ref.startswith(_PREFIX)):
            return False
        try:
            encounter_id = _native(visit_ref[len(_PREFIX):])
            pending_id = _native(pending_ref[len(_PREFIX):])
            with Session(self.engine) as db:
                pending = db.get(ClinicalPendingORM, pending_id)
                if pending is None:
                    return False
                encounter = self._signed_membership(db, pending)
                return encounter.id == encounter_id
        except (CycleForbidden, ValueError):
            return False

    def qualify_in_transaction(self, db: Session, visit_ref: str,
                               pending_ref: str) -> bool:
        """Same-DB, lock-ordered signed source gate for atomic registration.

        When the clinical and follow-through tables share PostgreSQL, acquire
        Encounter then Pending locks (consistent with visit-amendment order).
        Caller cannot pass an unrelated DB session and bypass source ownership.
        This is synthetic-only and has NO real principal or identity authority.
        """
        if db.get_bind() is not self.engine:
            return False
        if (not isinstance(visit_ref, str) or not isinstance(pending_ref, str)
                or not visit_ref.startswith(_PREFIX)
                or not pending_ref.startswith(_PREFIX)):
            return False
        try:
            encounter_id = _native(visit_ref[len(_PREFIX):])
            pending_id = _native(pending_ref[len(_PREFIX):])
            # Lock visit before Pending to avoid lock inversion with
            # clinical_visit_amendments.py's existing load() ordering.
            encounter = db.scalar(
                select(EncounterORM)
                .where(EncounterORM.id == encounter_id)
                .with_for_update()
            )
            if encounter is None:
                return False
            pending = db.scalar(
                select(ClinicalPendingORM)
                .where(ClinicalPendingORM.id == pending_id)
                .with_for_update()
            )
            if pending is None or pending.source_encounter_id != encounter_id:
                return False
            # Hold the patient ownership/marker row through this same-DB event.
            # Lock order: Encounter -> Pending -> Patient -> obligation.
            db.scalar(
                select(PatientORM)
                .where(PatientORM.patient_id == pending.patient_id)
                .with_for_update()
            )
            return self._signed_membership(db, pending).id == encounter_id
        except (CycleForbidden, ValueError, TypeError):
            return False

    def register_existing(
        self, store: DurableSyntheticFollowthrough, actor: Actor, *,
        patient_id: str, pending_id: str, encounter_id: str,
        required: tuple[str, ...],
    ) -> dict:
        qualified = self.resolve(
            actor, patient_id=patient_id, pending_id=pending_id,
            encounter_id=encounter_id,
        )
        req = tuple(dict.fromkeys(required))
        if not req or set(req) - _ALLOWED_TYPE[qualified.item_type]:
            raise CycleConflict("Requested documents do not match signed examination type")
        # Caller-supplied source verifier must be this binding method;
        # otherwise another caller could pass an arbitrary always-true gate.
        if (getattr(store.qualified_signed_pending, "__self__", None) is not self
                or getattr(store.qualified_signed_pending, "__func__", None)
                is not SyntheticSignedPendingBinding.qualify_refs):
            raise CycleForbidden("Durable store must use this qualified Pending verifier")
        # Atomic source validation only when clinical and event tables share
        # the SAME engine; separate disposable SQLite fixtures retain their
        # independent qualifier but do not claim cross-DB atomicity.
        same_db = store.engine is self.engine
        return store.register(
            actor, visit_ref=qualified.visit_ref,
            request_ref=qualified.request_ref, required=req,
            mode="signed_pending", pending_ref=qualified.pending_ref,
            transactional_qualification=(
                self.qualify_in_transaction if same_db else None
            ),
        )

    def project_pending(self, store: DurableSyntheticFollowthrough,
                        actor: Actor, *, patient_id: str,
                        pending_id: str, encounter_id: str) -> dict | None:
        """One existing Pending projection. No new task and NO clinical DB write."""
        qualified = self.resolve(
            actor, patient_id=patient_id, pending_id=pending_id,
            encounter_id=encounter_id,
        )
        with Session(store.engine) as db:
            row = db.scalar(select(VisitObligationFixture).where(
                VisitObligationFixture.pending_ref == qualified.pending_ref
            ))
            if row is None:
                return None
            if (row.visit_ref != qualified.visit_ref
                    or row.request_ref != qualified.request_ref
                    or row.mode != "signed_pending"):
                raise CycleConflict("Stale or cross-visit durable obligation")
            return store.view(row.id, actor)
