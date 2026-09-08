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


class PendingImportORM(ClinicalLearningBase):
    __tablename__ = "clinical_learning_pending_imports"

    import_id = Column(String, primary_key=True)
    source_event_id = Column(String, nullable=False, unique=True, index=True)
    source_format = Column(String, nullable=False)
    state = Column(String, nullable=False, index=True)
    normalized_challenge_json = Column(JSON, nullable=False)
    loop_plan_json = Column(JSON, nullable=False)
    resources_json = Column(JSON, nullable=False, default=list)
    warnings_json = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime, nullable=False, index=True)
    resolved_at = Column(DateTime, nullable=True)
    accepted_challenge_id = Column(String, nullable=True, index=True)
    accepted_revision = Column(Integer, nullable=True)


class LearningLoopPlanORM(ClinicalLearningBase):
    __tablename__ = "clinical_learning_loop_plans"

    cycle_id = Column(String, primary_key=True)
    challenge_id = Column(String, nullable=False, index=True)
    source_revision = Column(Integer, nullable=False)
    payload_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class ConsolidationAttemptORM(ClinicalLearningBase):
    __tablename__ = "clinical_learning_consolidation_attempts"

    attempt_id = Column(String, primary_key=True)
    cycle_id = Column(String, nullable=False, index=True)
    occurrence_id = Column(String, nullable=False, index=True)
    payload_json = Column(JSON, nullable=False)
    answered_at = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False)


class LearningResourceRecommendationORM(ClinicalLearningBase):
    __tablename__ = "clinical_learning_resource_recommendations"

    recommendation_id = Column(String, primary_key=True)
    challenge_id = Column(String, nullable=False, index=True)
    source_revision = Column(Integer, nullable=False)
    payload_json = Column(JSON, nullable=False)
    status = Column(String, nullable=False, index=True)
    checked_at = Column(DateTime, nullable=False, index=True)
    updated_at = Column(DateTime, nullable=False)


def init_learning_storage(engine: Engine) -> None:
    ClinicalLearningBase.metadata.create_all(bind=engine)


def latest_challenge_revision(session: Session, challenge_id: str) -> ChallengeRevisionORM | None:
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


def pending_import_by_source(session: Session, source_event_id: str) -> PendingImportORM | None:
    return session.execute(
        select(PendingImportORM).where(PendingImportORM.source_event_id == source_event_id).limit(1)
    ).scalar_one_or_none()


def pending_imports(session: Session) -> list[PendingImportORM]:
    return list(
        session.execute(
            select(PendingImportORM).order_by(PendingImportORM.created_at.desc())
        ).scalars()
    )


def loop_plans(session: Session) -> list[LearningLoopPlanORM]:
    return list(
        session.execute(
            select(LearningLoopPlanORM).order_by(LearningLoopPlanORM.created_at.desc())
        ).scalars()
    )


def loop_plan_for_challenge(
    session: Session,
    challenge_id: str,
    revision: int | None = None,
) -> LearningLoopPlanORM | None:
    stmt = select(LearningLoopPlanORM).where(LearningLoopPlanORM.challenge_id == challenge_id)
    if revision is not None:
        stmt = stmt.where(LearningLoopPlanORM.source_revision == revision)
    return session.execute(
        stmt.order_by(LearningLoopPlanORM.source_revision.desc()).limit(1)
    ).scalar_one_or_none()


def consolidation_attempts_for_cycle(session: Session, cycle_id: str) -> list[ConsolidationAttemptORM]:
    return list(
        session.execute(
            select(ConsolidationAttemptORM)
            .where(ConsolidationAttemptORM.cycle_id == cycle_id)
            .order_by(ConsolidationAttemptORM.answered_at.asc(), ConsolidationAttemptORM.created_at.asc())
        ).scalars()
    )


def resource_recommendations_for_challenge(
    session: Session,
    challenge_id: str,
    revision: int | None = None,
) -> list[LearningResourceRecommendationORM]:
    stmt = select(LearningResourceRecommendationORM).where(
        LearningResourceRecommendationORM.challenge_id == challenge_id
    )
    if revision is not None:
        stmt = stmt.where(LearningResourceRecommendationORM.source_revision == revision)
    return list(session.execute(stmt.order_by(LearningResourceRecommendationORM.checked_at.desc())).scalars())


def purge_challenge_content(session: Session, challenge_id: str) -> int:
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

    loop_rows = list(
        session.execute(
            select(LearningLoopPlanORM).where(LearningLoopPlanORM.challenge_id == challenge_id)
        ).scalars()
    )
    cycle_ids = [row.cycle_id for row in loop_rows]

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
                (DueItemORM.source_artifact_type == "learning_loop")
                & (DueItemORM.source_artifact_id.in_(cycle_ids) if cycle_ids else False),
            )
        )
    )
    if cycle_ids:
        session.execute(
            delete(ConsolidationAttemptORM).where(ConsolidationAttemptORM.cycle_id.in_(cycle_ids))
        )
    session.execute(
        delete(LearningResourceRecommendationORM).where(
            LearningResourceRecommendationORM.challenge_id == challenge_id
        )
    )
    session.execute(
        delete(LearningLoopPlanORM).where(LearningLoopPlanORM.challenge_id == challenge_id)
    )
    session.execute(
        delete(PendingImportORM).where(PendingImportORM.accepted_challenge_id == challenge_id)
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
