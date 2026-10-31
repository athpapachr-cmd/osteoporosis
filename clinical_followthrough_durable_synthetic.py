"""OFFLINE synthetic-only durable follow-through events; never mounted in main.py.

A future qualified real adapter must retain the signed ClinicalPending owner,
confirm real visit/source/actor authority, apply an approved migration and
replace these synthetic-only database guards. This class does none of that.

No mail content, contact details, patient demographics or free text is stored.
Only fixed event kinds and synthetic opaque fixture identifiers are accepted.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

from sqlalchemy import (
    Boolean, CheckConstraint, Column, DateTime, ForeignKey, Integer, JSON,
    String, UniqueConstraint, select, update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import DeclarativeBase, Session

from clinical_followthrough_cycle import (
    Actor, CycleConflict, CycleForbidden, CycleNotFound,
    SyntheticCycleLedger, Work, PARTS,
)

_SYN = re.compile(r"^synthetic-[A-Za-z0-9:_./-]{2,110}$")
_VERSION = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
_IDEM = re.compile(r"^[A-Za-z0-9_-]{16,100}$")
_ACTION_FIELDS = {
    "received": frozenset({"part", "source_ref", "source_version", "proof"}),
    "identity_verified": frozenset({"evidence", "evidence_ref"}),
    "review_notify": frozenset({"part", "source_version"}),
    "review_no_contact": frozenset({"part", "source_version"}),
    "contact_attempt": frozenset(),
    "patient_notified": frozenset(),
    "callback_complete": frozenset(),
}
_ACTION_ROLES = {
    "received": frozenset({"clinician", "source"}),
    "identity_verified": frozenset({"clinician"}),
    "review_notify": frozenset({"clinician"}),
    "review_no_contact": frozenset({"clinician"}),
    "contact_attempt": frozenset({"secretary"}),
    "patient_notified": frozenset({"secretary"}),
    "callback_complete": frozenset({"clinician"}),
}


class DurableFixtureBase(DeclarativeBase):
    pass


class VisitObligationFixture(DurableFixtureBase):
    __tablename__ = "synthetic_followthrough_obligations"
    __table_args__ = (
        UniqueConstraint("visit_ref", "request_ref", name="uq_sfo_visit_request"),
        UniqueConstraint("pending_ref", name="uq_sfo_existing_pending"),
        CheckConstraint("visit_ref LIKE 'synthetic-%' AND request_ref LIKE 'synthetic-%'",
                        name="ck_sfo_opaque_synthetic_only"),
        CheckConstraint(
            "(mode = 'provisional_visit' AND pending_ref IS NULL) "
            "OR (mode = 'signed_pending' AND pending_ref LIKE 'synthetic-%')",
            name="ck_sfo_no_fake_pending",
        ),
        CheckConstraint("version >= 0", name="ck_sfo_revision_nonnegative"),
    )
    id = Column(String(36), primary_key=True)
    visit_ref = Column(String(120), nullable=False)
    request_ref = Column(String(120), nullable=False)
    mode = Column(String(24), nullable=False)
    pending_ref = Column(String(120), nullable=True)
    required_parts = Column(JSON, nullable=False)
    verified_contact = Column(Boolean, nullable=False, default=False)
    version = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class ObligationEventFixture(DurableFixtureBase):
    __tablename__ = "synthetic_followthrough_events"
    __table_args__ = (
        UniqueConstraint("obligation_id", "version", name="uq_sfe_case_version"),
        UniqueConstraint("obligation_id", "actor_id", "idempotency_key",
                         name="uq_sfe_actor_request"),
        CheckConstraint("actor_id LIKE 'synthetic-%'", name="ck_sfe_synthetic_actor"),
        CheckConstraint("version > 0", name="ck_sfe_positive_revision"),
    )
    id = Column(String(36), primary_key=True)
    obligation_id = Column(
        String(36), ForeignKey("synthetic_followthrough_obligations.id"),
        nullable=False, index=True,
    )
    version = Column(Integer, nullable=False)
    actor_id = Column(String(120), nullable=False)
    actor_role = Column(String(16), nullable=False)
    action = Column(String(32), nullable=False)
    idempotency_key = Column(String(100), nullable=False)
    request_sha256 = Column(String(64), nullable=False)
    event_json = Column(JSON, nullable=False)
    occurred_at = Column(DateTime, nullable=False)


class SourceClaimFixture(DurableFixtureBase):
    __tablename__ = "synthetic_followthrough_source_claims"
    __table_args__ = (
        CheckConstraint("source_ref LIKE 'synthetic-%'",
                        name="ck_sfs_synthetic_source_only"),
    )
    source_ref = Column(String(120), primary_key=True)
    obligation_id = Column(
        String(36), ForeignKey("synthetic_followthrough_obligations.id"),
        nullable=False, index=True,
    )
    part = Column(String(16), nullable=False)


def _synthetic(value: str) -> str:
    if not isinstance(value, str) or not _SYN.fullmatch(value):
        raise CycleForbidden("Only synthetic opaque references qualify")
    return value


def _actor(actor: Actor, *allowed: str) -> Actor:
    if not isinstance(actor, Actor):
        raise CycleForbidden("Actor required")
    _synthetic(actor.principal)
    actor.check(*allowed)
    return actor


def _validated_action(actor: Actor, action: str, attributes: dict) -> dict:
    allowed = _ACTION_FIELDS.get(action)
    if allowed is None or set(attributes) != allowed:
        raise CycleConflict("Unsupported or extraneous event attributes")
    _actor(actor, *_ACTION_ROLES[action])
    for key, value in attributes.items():
        if not isinstance(value, str):
            raise CycleConflict("Invalid event attribute type")
        if key in ("source_ref", "evidence_ref"):
            _synthetic(value)
        elif key == "source_version":
            if not _VERSION.fullmatch(value):
                raise CycleConflict("Invalid source revision")
        elif key == "part":
            if value not in PARTS:
                raise CycleConflict("Unexpected examination part")
        elif key == "proof":
            if value not in ("qualified_order", "clinician_verified"):
                raise CycleConflict("Invalid source evidence")
        elif key == "evidence":
            if value not in ("qualified_referral", "clinician_verified"):
                raise CycleConflict("Invalid identity evidence")
    return attributes


def _digest(action: str, attrs: dict) -> str:
    canonical = SyntheticCycleLedger._fingerprint(action, attrs)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _state(row: VisitObligationFixture, history: list[ObligationEventFixture]) -> Work:
    if row.version != len(history) or any(
        event.version != i + 1 for i, event in enumerate(history)
    ):
        raise CycleConflict("Durable event history is not contiguous")
    return Work(
        id=row.id, visit_ref=row.visit_ref, request_ref=row.request_ref,
        mode=row.mode, pending_ref=row.pending_ref,
        required=tuple(row.required_parts),
        verified_contact=bool(row.verified_contact), version=row.version,
        events=[dict(event.event_json) for event in history],
    )


def _load(session: Session, case_id: str, *, lock: bool = False):
    statement = select(VisitObligationFixture).where(VisitObligationFixture.id == case_id)
    if lock:
        statement = statement.with_for_update()
    row = session.scalar(statement)
    if row is None:
        raise CycleNotFound("Unknown synthetic obligation")
    events = list(session.scalars(
        select(ObligationEventFixture)
        .where(ObligationEventFixture.obligation_id == case_id)
        .order_by(ObligationEventFixture.version.asc())
    ))
    return row, _state(row, events)


class DurableSyntheticFollowthrough:
    """A test-only persistence adapter for the same reducer as PR #267.

    The caller must provide an independent test-fixture signed Pending verifier.
    No feature flag or runtime app import may activate this for real PHI.
    """

    def __init__(self, engine: Engine, *, qualified_signed_pending:
                 Callable[[str, str], bool] | None = None):
        self.engine = engine
        self.qualified_signed_pending = qualified_signed_pending

    def register(self, actor: Actor, *, visit_ref: str, request_ref: str,
                 required: tuple[str, ...], mode: str,
                 pending_ref: str | None = None) -> dict:
        _actor(actor, "clinician")
        _synthetic(visit_ref)
        _synthetic(request_ref)
        if mode == "signed_pending":
            _synthetic(pending_ref)
            if (self.qualified_signed_pending is None
                    or self.qualified_signed_pending(visit_ref, pending_ref) is not True):
                raise CycleForbidden("Verified signed Pending source required")
        elif mode != "provisional_visit" or pending_ref is not None:
            raise CycleConflict("Cannot manufacture a signed encounter")
        req = tuple(dict.fromkeys(required))
        if not req or len(req) > 3 or set(req) - PARTS:
            raise CycleConflict("Invalid examination parts")
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        try:
            with Session(self.engine) as db, db.begin():
                existing = db.scalar(select(VisitObligationFixture).where(
                    VisitObligationFixture.visit_ref == visit_ref,
                    VisitObligationFixture.request_ref == request_ref,
                ))
                if existing:
                    if (existing.mode, existing.pending_ref, tuple(existing.required_parts)) != (
                        mode, pending_ref, req
                    ):
                        raise CycleConflict("Existing visit obligation conflicts with request")
                    row, work = _load(db, existing.id)
                    return SyntheticCycleLedger._current(work)
                item = VisitObligationFixture(
                    id=str(uuid4()), visit_ref=visit_ref, request_ref=request_ref,
                    mode=mode, pending_ref=pending_ref, required_parts=list(req),
                    version=0, verified_contact=False,
                    created_at=now, updated_at=now,
                )
                db.add(item)
                db.flush()  # DB unique guarantees duplicate pending is impossible.
                return SyntheticCycleLedger._current(
                    Work(id=item.id, visit_ref=visit_ref, request_ref=request_ref,
                         mode=mode, pending_ref=pending_ref, required=req)
                )
        except (IntegrityError, OperationalError):
            raise CycleConflict("Concurrent or conflicting obligation registration") from None

    def view(self, case_id: str, actor: Actor) -> dict:
        _actor(actor, "clinician")
        with Session(self.engine) as db:
            row, work = _load(db, case_id)
            return SyntheticCycleLedger._current(work)

    def history(self, case_id: str, actor: Actor) -> list[dict]:
        _actor(actor, "clinician")
        with Session(self.engine) as db:
            row, work = _load(db, case_id)
            return [dict(e) for e in work.events]

    def secretary_view(self, case_id: str, actor: Actor) -> dict:
        _actor(actor, "secretary")
        with Session(self.engine) as db:
            row, work = _load(db, case_id)
            ledger = SyntheticCycleLedger()
            ledger.works[case_id] = work
            return ledger.secretary_view(case_id, actor)

    def act(self, case_id: str, actor: Actor, *, action: str,
            expected_version: int, idempotency_key: str, **attrs) -> dict:
        attrs = _validated_action(actor, action, attrs)
        if not isinstance(idempotency_key, str) or not _IDEM.fullmatch(idempotency_key):
            raise CycleConflict("Invalid idempotency token")
        fingerprint = _digest(action, attrs)
        if type(expected_version) is not int or expected_version < 0:
            raise CycleConflict("Invalid expected version")
        try:
            with Session(self.engine) as db, db.begin():
                row, work = _load(db, case_id, lock=True)
                previous = db.scalar(select(ObligationEventFixture).where(
                    ObligationEventFixture.obligation_id == case_id,
                    ObligationEventFixture.actor_id == actor.principal,
                    ObligationEventFixture.idempotency_key == idempotency_key,
                ))
                if previous is not None:
                    if (previous.actor_role != actor.role
                            or previous.action != action
                            or previous.request_sha256 != fingerprint):
                        raise CycleConflict("Conflicting idempotent replay")
                    return SyntheticCycleLedger._current(work)

                ledger = SyntheticCycleLedger()
                ledger.works[case_id] = work
                if action == "received":
                    prior_claim = db.get(SourceClaimFixture, attrs["source_ref"])
                    if prior_claim is not None:
                        ledger.by_source[attrs["source_ref"]] = (
                            prior_claim.obligation_id, prior_claim.part
                        )

                result = ledger.act(case_id, actor, action=action,
                                    expected_version=expected_version,
                                    idempotency_key=idempotency_key, **attrs)
                if result["version"] == row.version:
                    return result  # Identical repeated receipt, no new event.

                before = row.version
                changed = db.execute(
                    update(VisitObligationFixture)
                    .where(VisitObligationFixture.id == case_id,
                           VisitObligationFixture.version == before)
                    .values(version=result["version"],
                            verified_contact=work.verified_contact,
                            updated_at=datetime.now(timezone.utc).replace(tzinfo=None))
                )
                if changed.rowcount != 1:
                    raise CycleConflict("Concurrent obligation update")
                if action == "received":
                    # Re-check after taking the write/CAS lock. Another
                    # transaction could have claimed the UID after the earlier
                    # read but before this row update. Seeing an existing claim
                    # must NEVER mean silently accepting it for another case.
                    committed_claim = db.get(SourceClaimFixture, attrs["source_ref"])
                    if committed_claim is None:
                        db.add(SourceClaimFixture(source_ref=attrs["source_ref"],
                                                  obligation_id=case_id, part=attrs["part"]))
                    elif (committed_claim.obligation_id != case_id
                          or committed_claim.part != attrs["part"]):
                        raise CycleConflict("Source claimed by another visit")
                event = dict(work.events[-1])
                db.add(ObligationEventFixture(
                    id=str(uuid4()), obligation_id=case_id,
                    version=event["version"],
                    actor_id=actor.principal, actor_role=actor.role,
                    action=action, idempotency_key=idempotency_key,
                    request_sha256=fingerprint, event_json=event,
                    occurred_at=datetime.now(timezone.utc).replace(tzinfo=None),
                ))
                db.flush()
                return result
        except (IntegrityError, OperationalError):
            # A rolled-back loser cannot create phantom review/contact events.
            raise CycleConflict("Concurrent source, version or idempotency collision") from None
