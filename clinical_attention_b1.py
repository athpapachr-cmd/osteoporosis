"""B1 synthetic PostgreSQL attention owner: isolated, feature-OFF in production.

No import or router registration from main.py. No create_all. No real patient identity,
message/PDF/ADT, provider, chart, Visit Brief, ClinicalPending or outbound writer.
A separately approved B2 must replace the synthetic-only DB constraints, actor gate
and deployment model before any identifiable processing.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import Column, DateTime, Integer, String, UniqueConstraint, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Session

from clinical_staff_auth import StaffActor


class B1Disabled(RuntimeError):
    pass


class B1Forbidden(PermissionError):
    pass


class B1Conflict(ValueError):
    pass


class B1NotFound(LookupError):
    pass


class B1Base(DeclarativeBase):
    pass


class ClinicalAttentionItemB1(B1Base):
    __tablename__ = "clinical_attention_items"
    item_id = Column(String(64), primary_key=True)
    logical_source_item_key = Column(String(100), nullable=False, unique=True)
    source_type = Column(String(32), nullable=False)
    active_source_revision = Column(String(16), nullable=False)
    source_subject_scope = Column(String(24), nullable=False)
    origin_kind = Column(String(16), nullable=False)
    patient_id = Column(String(120), nullable=True)  # DB CHECK (patient_id IS NULL) in B1
    identity_state = Column(String(24), nullable=False)
    review_state = Column(String(24), nullable=False)
    disposition = Column(String(24), nullable=False)
    version = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class ClinicalAttentionEventB1(B1Base):
    __tablename__ = "clinical_attention_events"
    __table_args__ = (
        UniqueConstraint("item_id", "actor_id", "idempotency_key",
                         name="uq_b1_actor_item_idempotency"),
    )
    event_id = Column(String(36), primary_key=True)
    item_id = Column(String(64), nullable=False)
    actor_id = Column(String(80), nullable=False)
    actor_role = Column(String(16), nullable=False)
    event_type = Column(String(32), nullable=False)
    outcome = Column(String(24), nullable=False)
    item_version = Column(Integer, nullable=False)
    source_revision = Column(String(16), nullable=False)
    idempotency_key = Column(String(36), nullable=True)
    request_hash = Column(String(64), nullable=True)
    occurred_at = Column(DateTime, nullable=False)


_KEY = re.compile(r"SYN-B1-[A-Z0-9-]{1,88}\Z")
_PERSON = re.compile(r"b1-synthetic-person-[A-Za-z0-9_-]{1,48}\Z")
_REV = re.compile(r"r[1-9][0-9]{0,3}\Z")
_TYPES = {"lab_notice", "imaging_notice", "gesy_availability"}
_SCOPES = {"single_subject", "unknown", "mixed_subjects"}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _snapshot(row: ClinicalAttentionItemB1) -> dict:
    # No raw content, subject identifiers, free text or evidence is stored in B1.
    return {
        "item_id": row.item_id,
        "logical_source_item_key": row.logical_source_item_key,
        "source_type": row.source_type,
        "active_source_revision": row.active_source_revision,
        "source_subject_scope": row.source_subject_scope,
        "identity_state": row.identity_state,
        "review_state": row.review_state,
        "disposition": row.disposition,
        "version": row.version,
        "patient_id": None,
    }


def _hash(action: str, expected: int, revision: str | None, scope: str | None) -> str:
    body = json.dumps([action, expected, revision, scope], separators=(",", ":"))
    return hashlib.sha256(body.encode("ascii")).hexdigest()


class SyntheticAttentionPostgresB1:
    """Strictly offline test harness, never mounted in normal Cockpit.

    The env/test-host/DB-role/schema preflight plus B1 table CHECK constraints
    intentionally make B1 impossible to use as a real-clinic identity store.
    """

    def __init__(self, engine: Engine):
        self.engine = engine
        url = engine.url
        if (os.getenv("B1_SYNTHETIC_TEST_ONLY") != "yes"
                or url.get_backend_name() != "postgresql"
                or url.host not in ("127.0.0.1", "localhost")
                or url.database != "osteoporosis"
                or url.username != "cockpit_app"):
            raise B1Disabled("B1 is disabled outside isolated synthetic PostgreSQL")
        with engine.connect() as connection:
            row = connection.execute(text(
                "SELECT current_database(), current_user, "
                "to_regclass('public.clinical_attention_items') IS NOT NULL, "
                "to_regclass('public.clinical_attention_events') IS NOT NULL"
            )).one()
            if row != ("osteoporosis", "cockpit_app", True, True):
                raise B1Disabled("B1 explicit migration and least-role preflight missing")

    def _actor(self, actor: StaffActor, item_id: str, event: str) -> None:
        if (not isinstance(actor, StaffActor)
                or not _PERSON.fullmatch(actor.principal_id)):
            # A shared/static/legacy principal is not an attributable person.
            raise B1Forbidden("Individual clinician identity required")
        if actor.role != "clinician":
            self._denial(actor, item_id, event, "denied_403")
            raise B1Forbidden("Clinician-only B1 action")

    def _denial(self, actor: StaffActor, item_id: str, event: str, outcome: str) -> None:
        # Record attributable failed access in its own transaction. No clinical
        # mutation should be committed if validation/optimistic locking fails.
        with Session(self.engine) as session:
            row = session.get(ClinicalAttentionItemB1, item_id)
            if row is None:
                return
            self._event(session, actor, row, event, outcome)
            session.commit()

    @staticmethod
    def _event(session: Session, actor: StaffActor, item: ClinicalAttentionItemB1,
               event_type: str, outcome: str, key: str | None = None,
               request_hash: str | None = None) -> None:
        session.add(ClinicalAttentionEventB1(
            event_id=str(uuid4()), item_id=item.item_id,
            actor_id=actor.principal_id, actor_role=actor.role,
            event_type=event_type, outcome=outcome, item_version=item.version,
            source_revision=item.active_source_revision,
            idempotency_key=key, request_hash=request_hash, occurred_at=_utcnow(),
        ))

    def create_fixture(self, actor: StaffActor, *, item_id: str, source_key: str,
                       source_type: str, source_scope: str = "single_subject") -> dict:
        self._actor(actor, item_id, "create_fixture")
        if (not _KEY.fullmatch(item_id) or len(item_id) > 64
                or not _KEY.fullmatch(source_key) or len(source_key) > 100
                or source_type not in _TYPES or source_scope not in _SCOPES):
            raise B1Forbidden("Only constrained synthetic fixtures are allowed")
        with Session(self.engine) as session:
            row = session.get(ClinicalAttentionItemB1, item_id, with_for_update=True)
            if row is not None:
                if (row.logical_source_item_key == source_key
                        and row.source_type == source_type
                        and row.source_subject_scope == source_scope):
                    return _snapshot(row)
                raise B1Conflict("Conflicting synthetic source")
            stamp = _utcnow()
            row = ClinicalAttentionItemB1(
                item_id=item_id, logical_source_item_key=source_key,
                source_type=source_type, active_source_revision="r1",
                source_subject_scope=source_scope, origin_kind="synthetic",
                patient_id=None, identity_state="unlinked",
                review_state="unreviewed", disposition="undecided",
                version=0, created_at=stamp, updated_at=stamp,
            )
            session.add(row)
            try:
                session.flush()  # parent before append-only FK audit, same transaction
                self._event(session, actor, row, "fixture_created", "allowed")
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                # Concurrency or a duplicate source scope: never merge items.
                existing = session.get(ClinicalAttentionItemB1, item_id)
                if (existing and existing.logical_source_item_key == source_key
                        and existing.source_type == source_type
                        and existing.source_subject_scope == source_scope):
                    return _snapshot(existing)
                raise B1Conflict("Source/identity duplicate") from exc
            return _snapshot(row)

    def read(self, actor: StaffActor, item_id: str) -> dict:
        self._actor(actor, item_id, "view")
        with Session(self.engine) as session:
            row = session.get(ClinicalAttentionItemB1, item_id)
            if row is None:
                raise B1NotFound("Item not found")
            result = _snapshot(row)
            self._event(session, actor, row, "view", "allowed")
            session.commit()
            return result

    def mutate(self, actor: StaffActor, *, item_id: str, action: str,
               expected_version: int, idempotency_key: str,
               new_revision: str | None = None, source_scope: str | None = None) -> dict:
        self._actor(actor, item_id, action)
        if (action not in ("inspect_source", "correct_source")
                or not isinstance(expected_version, int) or expected_version < 0):
            raise B1Forbidden("Unsupported B1 mutation")
        try:
            key = str(UUID(idempotency_key))
            if key != idempotency_key:
                raise ValueError
        except (ValueError, TypeError, AttributeError) as exc:
            raise B1Forbidden("Invalid idempotency UUID") from exc
        if action == "correct_source":
            if (not isinstance(new_revision, str) or not _REV.fullmatch(new_revision)
                    or source_scope not in _SCOPES):
                raise B1Forbidden("Invalid synthetic source correction")
        elif new_revision is not None or source_scope is not None:
            raise B1Forbidden("Review cannot change source or identity")

        digest = _hash(action, expected_version, new_revision, source_scope)
        try:
            with Session(self.engine) as session:
                row = session.get(ClinicalAttentionItemB1, item_id,
                                  with_for_update=True)
                if row is None:
                    raise B1NotFound("Item not found")
                previous = session.scalar(select(ClinicalAttentionEventB1).where(
                    ClinicalAttentionEventB1.item_id == item_id,
                    ClinicalAttentionEventB1.actor_id == actor.principal_id,
                    ClinicalAttentionEventB1.idempotency_key == key,
                ))
                if previous is not None:
                    if previous.request_hash != digest or previous.item_version != row.version:
                        raise B1Conflict("Stale or conflicting idempotency replay")
                    return _snapshot(row)
                if row.version != expected_version:
                    raise B1Conflict("Source version changed")
                if action == "inspect_source":
                    if row.review_state == "source_inspected":
                        raise B1Conflict("Source already inspected")
                    row.review_state = "source_inspected"
                else:
                    if row.active_source_revision == new_revision:
                        raise B1Conflict("Revision already active")
                    row.active_source_revision = new_revision
                    row.source_subject_scope = source_scope
                    row.patient_id = None
                    row.identity_state = "unlinked"
                    row.review_state = "needs_re_review"
                    row.disposition = "undecided"
                row.version += 1
                row.updated_at = _utcnow()
                self._event(session, actor, row, action, "allowed", key, digest)
                result = _snapshot(row)
                session.commit()
                return result
        except B1Conflict:
            self._denial(actor, item_id, action, "denied_409")
            raise
