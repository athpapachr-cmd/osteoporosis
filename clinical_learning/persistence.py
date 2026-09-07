from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    Date,
    DateTime,
    Integer,
    String,
    UniqueConstraint,
    delete,
    or_,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session


class ClinicalLearningBase(DeclarativeBase):
    pass


class ChallengeRevisionORM(ClinicalLearningBase):
    __tablename__ = "clinical_learning_challenge_revisions"

    challenge_id = Column(String, primary_key=True)
    revision = Column(Integer, primary_key=True)
    schema_version = Column(String, nullable=False)
    module = Column(String, nullable=False, index=True)
    payload_json = Column(JSON, nullable=False)
    content_hash = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, index=True)
    imported_at = Column(DateTime, nullable=False)


class ReferenceVerificationORM(ClinicalLearningBase):
    __tablename__ = "clinical_learning_reference_verification"

    artifact_type = Column(String, primary_key=True)
    artifact_id = Column(String, primary_key=True)
    artifact_revision = Column(Integer, primary_key=True)
    reference_id = Column(String, primary_key=True)
    verification_state = Column(String, nullable=False)
    verification_note = Column(String, nullable=True)
    verified_at = Column(DateTime, nullable=True)


class ChallengeTombstoneORM(ClinicalLearningBase):
    __tablename__ = "clinical_learning_challenge_tombstones"

    challenge_id = Column(String, primary_key=True)
    deleted_at = Column(DateTime, nullable=False)
    max_deleted_revision = Column(Integer, nullable=False)


class FoundationAttemptORM(ClinicalLearningBase):
    __tablename__ = "clinical_learning_foundation_attempts"

    attempt_id = Column(String, primary_key=True)
    foundation_node_id = Column(String, nullable=False, index=True)
    module = Column(String, nullable=False)
    payload_json = Column(JSON, nullable=False)
    assessed_at = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False)


class FoundationStateORM(ClinicalLearningBase):
    __tablename__ = "clinical_learning_foundation_state"

    foundation_node_id = Column(String, primary_key=True)
    module = Column(String, nullable=False)
    state = Column(String, nullable=False)
    retention_state = Column(String, nullable=False)
    last_assessed_at = Column(DateTime, nullable=True)
    next_review_due = Column(Date, nullable=True)
    evidence_attempt_ids_json = Column(JSON, nullable=False, default=list)
    updated_at = Column(DateTime, nullable=False)


class DueItemORM(ClinicalLearningBase):
    __tablename__ = "clinical_learning_due_items"
    __table_args__ = (
        UniqueConstraint("item_type", "target_id", "occurrence", name="uq_cl_learning_due_occurrence"),
    )

    due_item_id = Column(String, primary_key=True)
    item_type = Column(String, nullable=False, index=True)
    target_id = Column(String, nullable=False, index=True)
    occurrence = Column(Integer, nullable=False)
    due_on = Column(Date, nullable=True, index=True)
    due_status = Column(String, nullable=False, index=True)
    delivery_mode = Column(String, nullable=False)
    reason_code = Column(String, nullable=False)
    source_artifact_type = Column(String, nullable=False, index=True)
    source_artifact_id = Column(String, nullable=False, index=True)
    source_revision = Column(Integer, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    deferred_until = Column(Date, nullable=True)
    updated_at = Column(DateTime, nullable=False)


def init_learning_storage(engine: Engine) -> None:
    ClinicalLearningBase.metadata.create_all(bind=engine)


def latest_challenge_revision(session: Session, challenge_id: str) -> ChallengeRevisionORM | None:
    # Mutation callers rely on this row lock to serialize revision creation with
    # delete/reference-overlay mutations. SQLite safely ignores FOR UPDATE; the
    # production Postgres engine enforces it within the surrounding transaction.
    return session.execute(
        select(ChallengeRevisionORM)
        .where(ChallengeRevisionORM.challenge_id == challenge_id)
        .order_by(ChallengeRevisionORM.revision.desc())
        .limit(1)
        .with_for_update()
    ).scalar_one_or_none()


def challenge_revision(
    session: Session,
    challenge_id: str,
    revision: int,
) -> ChallengeRevisionORM | None:
    return session.execute(
        select(ChallengeRevisionORM)
        .where(
            ChallengeRevisionORM.challenge_id == challenge_id,
            ChallengeRevisionORM.revision == revision,
        )
        .with_for_update()
    ).scalar_one_or_none()


def challenge_revisions(session: Session, challenge_id: str) -> list[ChallengeRevisionORM]:
    return list(
        session.execute(
            select(ChallengeRevisionORM)
            .where(ChallengeRevisionORM.challenge_id == challenge_id)
            .order_by(ChallengeRevisionORM.revision.asc())
        ).scalars()
    )


def challenge_is_tombstoned(session: Session, challenge_id: str) -> ChallengeTombstoneORM | None:
    return session.get(ChallengeTombstoneORM, challenge_id)


def reference_overlays(
    session: Session,
    *,
    artifact_id: str,
    artifact_revision: int | None = None,
) -> list[ReferenceVerificationORM]:
    stmt = select(ReferenceVerificationORM).where(
        ReferenceVerificationORM.artifact_type == "challenge",
        ReferenceVerificationORM.artifact_id == artifact_id,
    )
    if artifact_revision is not None:
        stmt = stmt.where(ReferenceVerificationORM.artifact_revision == artifact_revision)
    return list(session.execute(stmt).scalars())


def due_rows_for_source_or_target_challenge(
    session: Session,
    challenge_id: str,
) -> list[DueItemORM]:
    return list(
        session.execute(
            select(DueItemORM).where(
                or_(
                    DueItemORM.target_id == challenge_id,
                    (DueItemORM.source_artifact_type == "challenge")
                    & (DueItemORM.source_artifact_id == challenge_id),
                )
            )
        ).scalars()
    )


def purge_challenge_content(session: Session, challenge_id: str) -> int:
    # Lock the immutable revision rows before any overlay/due/content purge. This
    # prevents a concurrent revise/reference write from crossing the tombstone
    # transaction and leaving resurrected content or an orphan mutable overlay.
    locked_revisions = list(
        session.execute(
            select(ChallengeRevisionORM)
            .where(ChallengeRevisionORM.challenge_id == challenge_id)
            .order_by(ChallengeRevisionORM.revision.asc())
            .with_for_update()
        ).scalars()
    )
    if not locked_revisions:
        return 0
    max_revision = max(int(row.revision) for row in locked_revisions)

    session.execute(
        delete(ReferenceVerificationORM).where(
            ReferenceVerificationORM.artifact_type == "challenge",
            ReferenceVerificationORM.artifact_id == challenge_id,
        )
    )
    session.execute(
        delete(DueItemORM).where(
            or_(
                DueItemORM.target_id == challenge_id,
                (DueItemORM.source_artifact_type == "challenge")
                & (DueItemORM.source_artifact_id == challenge_id),
            )
        )
    )
    session.execute(
        delete(ChallengeRevisionORM).where(ChallengeRevisionORM.challenge_id == challenge_id)
    )
    return max_revision


def foundation_attempts_for_node(session: Session, foundation_node_id: str) -> list[FoundationAttemptORM]:
    return list(
        session.execute(
            select(FoundationAttemptORM)
            .where(FoundationAttemptORM.foundation_node_id == foundation_node_id)
            .order_by(FoundationAttemptORM.assessed_at.asc(), FoundationAttemptORM.created_at.asc())
        ).scalars()
    )


def due_rows(session: Session) -> list[DueItemORM]:
    return list(
        session.execute(
            select(DueItemORM).order_by(
                DueItemORM.due_on.asc().nullslast(),
                DueItemORM.updated_at.desc(),
            )
        ).scalars()
    )


def serialize_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def serialize_date(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def row_payload(row: ChallengeRevisionORM) -> dict[str, Any]:
    return dict(row.payload_json or {})
