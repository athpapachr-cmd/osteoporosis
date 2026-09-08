from __future__ import annotations

import copy
from datetime import date, datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from .contracts import (
    ContractIssue,
    LearningContractError,
    canonical_content_hash,
    get_foundation_registry,
    normalize_for_save,
    normalize_import_preview,
    validate_challenge_payload,
    validate_foundation_attempt_payload,
)
from .models import ClinicalLearningChallengeV1
from .persistence import (
    ChallengeRevisionORM,
    ChallengeTombstoneORM,
    DueItemORM,
    FoundationAttemptORM,
    FoundationStateORM,
    ReferenceVerificationORM,
    challenge_is_tombstoned,
    challenge_revision,
    challenge_revisions,
    due_rows,
    foundation_attempts_for_node,
    init_learning_storage,
    latest_challenge_revision,
    purge_challenge_content,
    reference_overlays,
    serialize_date,
    serialize_datetime,
)
from .privacy import scan_text


class LearningServiceError(ValueError):
    def __init__(self, code: str, path: str = "", *, status_code: int = 422):
        self.code = code
        self.path = path
        self.status_code = status_code
        super().__init__(code)


def utcnow() -> datetime:
    """Naive UTC for SQLAlchemy DateTime columns used by this repository."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def utcnow_iso() -> str:
    """Offset-aware UTC timestamp for JSON learning artifacts."""
    return datetime.now(timezone.utc).isoformat()


def _naive_utc(value: datetime) -> datetime:
    """Normalize aware datetimes to UTC before removing tzinfo."""
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _today() -> date:
    return datetime.now(timezone.utc).date()


def sanitized_issues(exc: LearningContractError) -> list[dict[str, str]]:
    return [{"code": issue.code, "path": issue.path} for issue in exc.issues]


def _due_status(
    due_on: date | None,
    *,
    completed_at: datetime | None = None,
    deferred_until: date | None = None,
) -> str:
    if completed_at is not None:
        return "completed"
    today = _today()
    if deferred_until is not None and deferred_until > today:
        return "deferred"
    effective = deferred_until if deferred_until is not None else due_on
    if effective is None:
        return "not_scheduled"
    if effective > today:
        return "scheduled"
    if effective == today:
        return "due"
    return "overdue"


def _retention_state(next_review_due: date | None) -> str:
    if next_review_due is None:
        return "not_scheduled"
    return "scheduled" if next_review_due > _today() else "due"


def _as_revision(payload: dict[str, Any], revision: int, supersedes: int | None) -> dict[str, Any]:
    material = copy.deepcopy(payload)
    material["revision"] = revision
    material["supersedes_revision"] = supersedes
    return material


def _find_reference(payload: dict[str, Any], reference_id: str) -> dict[str, Any] | None:
    for reference in payload.get("references") or []:
        if str(reference.get("reference_id")) == reference_id:
            return reference
    return None


def _serialize_due(row: DueItemORM) -> dict[str, Any]:
    return {
        "due_item_id": row.due_item_id,
        "schema_version": "learning_due_state_v1",
        "item_type": row.item_type,
        "target_id": row.target_id,
        "occurrence": row.occurrence,
        "due_on": serialize_date(row.due_on),
        "due_status": _due_status(
            row.due_on,
            completed_at=row.completed_at,
            deferred_until=row.deferred_until,
        ),
        "delivery_mode": row.delivery_mode,
        "reason_code": row.reason_code,
        "source_artifact_type": row.source_artifact_type,
        "source_artifact_id": row.source_artifact_id,
        "source_revision": row.source_revision,
        "completed_at": serialize_datetime(row.completed_at),
        "deferred_until": serialize_date(row.deferred_until),
    }


def _latest_due(session: Session, item_type: str, target_id: str) -> DueItemORM | None:
    return session.execute(
        select(DueItemORM)
        .where(DueItemORM.item_type == item_type, DueItemORM.target_id == target_id)
        .order_by(DueItemORM.occurrence.desc())
        .limit(1)
    ).scalar_one_or_none()


def _materialize_due(
    session: Session,
    *,
    item_type: str,
    target_id: str,
    due_on: date | None,
    reason_code: str,
    source_artifact_type: str,
    source_artifact_id: str,
    source_revision: int | None,
    delivery_mode: str = "visible",
) -> DueItemORM:
    now = utcnow()
    latest = _latest_due(session, item_type, target_id)
    if latest is not None and latest.completed_at is None and latest.due_status != "completed":
        latest.due_on = due_on
        latest.deferred_until = None
        latest.due_status = _due_status(due_on)
        latest.delivery_mode = delivery_mode
        latest.reason_code = reason_code
        latest.source_artifact_type = source_artifact_type
        latest.source_artifact_id = source_artifact_id
        latest.source_revision = source_revision
        latest.updated_at = now
        return latest

    occurrence = 1 if latest is None else latest.occurrence + 1
    row = DueItemORM(
        due_item_id=str(uuid4()),
        item_type=item_type,
        target_id=target_id,
        occurrence=occurrence,
        due_on=due_on,
        due_status=_due_status(due_on),
        delivery_mode=delivery_mode,
        reason_code=reason_code,
        source_artifact_type=source_artifact_type,
        source_artifact_id=source_artifact_id,
        source_revision=source_revision,
        completed_at=None,
        deferred_until=None,
        updated_at=now,
    )
    session.add(row)
    return row


def _deactivate_removed_challenge_due(
    session: Session,
    *,
    challenge_id: str,
    revision: int,
    active_keys: set[tuple[str, str]],
) -> None:
    rows = session.execute(
        select(DueItemORM).where(
            DueItemORM.source_artifact_type == "challenge",
            DueItemORM.source_artifact_id == challenge_id,
        )
    ).scalars().all()
    now = utcnow()
    for row in rows:
        if row.completed_at is not None or (row.item_type, row.target_id) in active_keys:
            continue
        row.due_on = None
        row.deferred_until = None
        row.due_status = "not_applicable"
        row.source_revision = revision
        row.updated_at = now


def _materialize_challenge_due(session: Session, payload: dict[str, Any]) -> None:
    challenge_id = str(payload["challenge_id"])
    revision = int(payload["revision"])
    active: set[tuple[str, str]] = set()

    repetition_due = payload.get("spaced_repetition_due")
    if repetition_due:
        active.add(("challenge_repetition", challenge_id))
        _materialize_due(
            session,
            item_type="challenge_repetition",
            target_id=challenge_id,
            due_on=date.fromisoformat(str(repetition_due)),
            reason_code="challenge_spaced_repetition",
            source_artifact_type="challenge",
            source_artifact_id=challenge_id,
            source_revision=revision,
        )

    for action in payload.get("learning_actions") or []:
        if action.get("status") != "planned" or not action.get("due_on"):
            continue
        action_id = str(action["action_id"])
        active.add(("learning_action", action_id))
        _materialize_due(
            session,
            item_type="learning_action",
            target_id=action_id,
            due_on=date.fromisoformat(str(action["due_on"])),
            reason_code="challenge_learning_action_due",
            source_artifact_type="challenge",
            source_artifact_id=challenge_id,
            source_revision=revision,
        )

    _deactivate_removed_challenge_due(
        session,
        challenge_id=challenge_id,
        revision=revision,
        active_keys=active,
    )


class ClinicalLearningService:
    def __init__(self, engine: Engine):
        self.engine = engine
        init_learning_storage(engine)

    def preview_challenge(self, raw_challenge: Any) -> dict[str, Any]:
        try:
            challenge = validate_challenge_payload(raw_challenge)
        except LearningContractError as exc:
            return {"valid": False, "issues": sanitized_issues(exc)}

        normalized = normalize_import_preview(challenge)
        challenge_id = str(challenge.challenge_id)
        with Session(self.engine) as session:
            tombstone = challenge_is_tombstoned(session, challenge_id)
            latest = latest_challenge_revision(session, challenge_id)
            duplicate_state = "new_identity"
            if tombstone is not None:
                duplicate_state = "tombstoned_identity_rejected"
            elif latest is None:
                if int(normalized["revision"]) != 1:
                    duplicate_state = "stale_revision_rejected"
            else:
                requested_revision = int(normalized["revision"])
                requested_hash = canonical_content_hash(normalized)
                stored_challenge = validate_challenge_payload(latest.payload_json or {})
                stored_import_view = normalize_import_preview(stored_challenge)
                stored_import_hash = canonical_content_hash(stored_import_view)
                if requested_revision == latest.revision:
                    duplicate_state = (
                        "exact_idempotent_duplicate"
                        if requested_hash == stored_import_hash
                        else "same_revision_conflict"
                    )
                elif (
                    requested_revision == latest.revision + 1
                    and normalized.get("supersedes_revision") == latest.revision
                ):
                    duplicate_state = "next_revision_candidate"
                else:
                    duplicate_state = "stale_revision_rejected"

        imported = challenge.model_dump(mode="json")
        warnings: list[dict[str, str]] = []
        if any(ref.get("verification_state") != "unverified" for ref in imported.get("references") or []):
            warnings.append({"code": "imported_reference_verification_reset", "path": "references"})
        if challenge.record_review_state != "imported_pending_review" or challenge.reviewed_at is not None:
            warnings.append({"code": "imported_review_authority_reset", "path": "record_review_state"})
        if challenge.linked_signal_ids:
            warnings.append({"code": "imported_signal_links_removed", "path": "linked_signal_ids"})

        blocked = {
            "tombstoned_identity_rejected",
            "stale_revision_rejected",
            "same_revision_conflict",
        }
        return {
            "valid": duplicate_state not in blocked,
            "issues": [],
            "duplicate_state": duplicate_state,
            "warnings": warnings,
            "normalized_summary": normalized,
        }

    def _prepare_reviewed_payload(
        self,
        raw_challenge: Any,
        *,
        revision: int,
        supersedes_revision: int | None,
    ) -> tuple[ClinicalLearningChallengeV1, dict[str, Any], str]:
        initial = validate_challenge_payload(raw_challenge)
        material = _as_revision(initial.model_dump(mode="json"), revision, supersedes_revision)
        reviewed = validate_challenge_payload(material)
        payload = normalize_for_save(reviewed, reviewed_at_iso=utcnow_iso())
        checked = validate_challenge_payload(payload)
        final_payload = checked.model_dump(mode="json")
        return checked, final_payload, canonical_content_hash(final_payload)

    def create_challenge(self, raw_challenge: Any, *, confirm_save: bool) -> dict[str, Any]:
        if confirm_save is not True:
            raise LearningServiceError("confirm_save_required", "confirm_save")
        first = validate_challenge_payload(raw_challenge)
        challenge_id = str(first.challenge_id)
        with Session(self.engine) as session:
            if challenge_is_tombstoned(session, challenge_id) is not None:
                raise LearningServiceError(
                    "tombstoned_identity_rejected", "challenge.challenge_id", status_code=409
                )
            latest = latest_challenge_revision(session, challenge_id)
            if latest is not None:
                _, _, candidate_hash = self._prepare_reviewed_payload(
                    raw_challenge,
                    revision=latest.revision,
                    supersedes_revision=latest.revision - 1 if latest.revision > 1 else None,
                )
                if candidate_hash == latest.content_hash:
                    return self._challenge_record(session, latest, idempotent=True)
                raise LearningServiceError(
                    "challenge_exists_use_put_for_new_revision",
                    "challenge.challenge_id",
                    status_code=409,
                )

            checked, payload, content_hash = self._prepare_reviewed_payload(
                raw_challenge, revision=1, supersedes_revision=None
            )
            row = ChallengeRevisionORM(
                challenge_id=challenge_id,
                revision=1,
                schema_version=str(payload["schema_version"]),
                module=str(payload["module"]),
                payload_json=payload,
                content_hash=content_hash,
                created_at=_naive_utc(checked.created_at),
                imported_at=utcnow(),
            )
            session.add(row)
            _materialize_challenge_due(session, payload)
            session.commit()
            session.refresh(row)
            return self._challenge_record(session, row, idempotent=False)

    def revise_challenge(
        self, challenge_id: str, raw_challenge: Any, *, confirm_save: bool
    ) -> dict[str, Any]:
        if confirm_save is not True:
            raise LearningServiceError("confirm_save_required", "confirm_save")
        first = validate_challenge_payload(raw_challenge)
        if str(first.challenge_id) != challenge_id:
            raise LearningServiceError("challenge_id_path_mismatch", "challenge.challenge_id")
        with Session(self.engine) as session:
            if challenge_is_tombstoned(session, challenge_id) is not None:
                raise LearningServiceError(
                    "tombstoned_identity_rejected", "challenge.challenge_id", status_code=409
                )
            latest = latest_challenge_revision(session, challenge_id)
            if latest is None:
                raise LearningServiceError("challenge_not_found", "challenge_id", status_code=404)

            _, _, same_hash = self._prepare_reviewed_payload(
                raw_challenge,
                revision=latest.revision,
                supersedes_revision=latest.revision - 1 if latest.revision > 1 else None,
            )
            if same_hash == latest.content_hash:
                return self._challenge_record(session, latest, idempotent=True)

            next_revision = latest.revision + 1
            checked, payload, content_hash = self._prepare_reviewed_payload(
                raw_challenge,
                revision=next_revision,
                supersedes_revision=latest.revision,
            )
            row = ChallengeRevisionORM(
                challenge_id=challenge_id,
                revision=next_revision,
                schema_version=str(payload["schema_version"]),
                module=str(payload["module"]),
                payload_json=payload,
                content_hash=content_hash,
                created_at=_naive_utc(checked.created_at),
                imported_at=utcnow(),
            )
            session.add(row)
            _materialize_challenge_due(session, payload)
            session.commit()
            session.refresh(row)
            return self._challenge_record(session, row, idempotent=False)

    def _challenge_record(
        self, session: Session, row: ChallengeRevisionORM, *, idempotent: bool
    ) -> dict[str, Any]:
        overlays = reference_overlays(
            session, artifact_id=row.challenge_id, artifact_revision=row.revision
        )
        return {
            "challenge_id": row.challenge_id,
            "revision": row.revision,
            "content_hash": row.content_hash,
            "payload": copy.deepcopy(row.payload_json or {}),
            "reference_verification": [
                {
                    "reference_id": item.reference_id,
                    "verification_state": item.verification_state,
                    "verification_note": item.verification_note,
                    "verified_at": serialize_datetime(item.verified_at),
                }
                for item in overlays
            ],
            "idempotent": idempotent,
        }

    def list_challenges(
        self,
        *,
        topic: str | None = None,
        foundation_node: str | None = None,
        challenge_date: date | None = None,
        challenge_mode: str | None = None,
        review_state: str | None = None,
    ) -> list[dict[str, Any]]:
        with Session(self.engine) as session:
            rows = list(
                session.execute(
                    select(ChallengeRevisionORM).order_by(
                        ChallengeRevisionORM.challenge_id.asc(),
                        ChallengeRevisionORM.revision.desc(),
                    )
                ).scalars()
            )
            latest_by_id: dict[str, ChallengeRevisionORM] = {}
            for row in rows:
                latest_by_id.setdefault(row.challenge_id, row)
            out: list[dict[str, Any]] = []
            topic_key = topic.casefold().strip() if topic else None
            for row in latest_by_id.values():
                payload = row.payload_json or {}
                if topic_key and topic_key not in {
                    str(v).casefold().strip() for v in payload.get("topics") or []
                }:
                    continue
                if foundation_node and foundation_node not in (
                    payload.get("foundation_node_ids") or []
                ):
                    continue
                if challenge_date and str(payload.get("created_at", ""))[:10] != challenge_date.isoformat():
                    continue
                if challenge_mode and payload.get("challenge_mode") != challenge_mode:
                    continue
                if review_state and payload.get("record_review_state") != review_state:
                    continue
                due = _latest_due(session, "challenge_repetition", row.challenge_id)
                out.append(
                    {
                        "challenge_id": row.challenge_id,
                        "revision": row.revision,
                        "title": payload.get("title"),
                        "created_at": payload.get("created_at"),
                        "challenge_mode": payload.get("challenge_mode"),
                        "topics": payload.get("topics") or [],
                        "foundation_node_ids": payload.get("foundation_node_ids") or [],
                        "record_review_state": payload.get("record_review_state"),
                        "due": _serialize_due(due) if due is not None else None,
                    }
                )
            out.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
            return out

    def get_challenge(self, challenge_id: str) -> dict[str, Any]:
        with Session(self.engine) as session:
            revisions = challenge_revisions(session, challenge_id)
            if not revisions:
                tombstone = challenge_is_tombstoned(session, challenge_id)
                if tombstone is not None:
                    return {
                        "challenge_id": challenge_id,
                        "deleted": True,
                        "deleted_at": serialize_datetime(tombstone.deleted_at),
                        "max_deleted_revision": tombstone.max_deleted_revision,
                    }
                raise LearningServiceError("challenge_not_found", "challenge_id", status_code=404)
            return {
                "challenge_id": challenge_id,
                "deleted": False,
                "revisions": [
                    self._challenge_record(session, row, idempotent=False) for row in revisions
                ],
                "due": [
                    _serialize_due(row)
                    for row in session.execute(
                        select(DueItemORM).where(
                            DueItemORM.source_artifact_type == "challenge",
                            DueItemORM.source_artifact_id == challenge_id,
                        )
                    ).scalars()
                ],
            }

    def delete_challenge(self, challenge_id: str, *, confirm_delete: bool) -> dict[str, Any]:
        if confirm_delete is not True:
            raise LearningServiceError("confirm_delete_required", "confirm_delete")
        with Session(self.engine) as session:
            tombstone = challenge_is_tombstoned(session, challenge_id)
            if tombstone is not None:
                return {
                    "challenge_id": challenge_id,
                    "deleted": True,
                    "max_deleted_revision": tombstone.max_deleted_revision,
                    "idempotent": True,
                }
            max_revision = purge_challenge_content(session, challenge_id)
            if max_revision < 1:
                raise LearningServiceError("challenge_not_found", "challenge_id", status_code=404)
            session.add(
                ChallengeTombstoneORM(
                    challenge_id=challenge_id,
                    deleted_at=utcnow(),
                    max_deleted_revision=max_revision,
                )
            )
            session.commit()
            return {
                "challenge_id": challenge_id,
                "deleted": True,
                "max_deleted_revision": max_revision,
                "idempotent": False,
            }

    def set_reference_verification(
        self,
        *,
        challenge_id: str,
        revision: int,
        reference_id: str,
        verification_state: str,
        verification_note: str | None,
    ) -> dict[str, Any]:
        allowed = {
            "unverified",
            "verified_locator",
            "verified_content",
            "invalid_or_unresolved",
        }
        if verification_state not in allowed:
            raise LearningServiceError(
                "invalid_reference_verification_state", "verification_state"
            )
        if verification_note:
            findings = scan_text(verification_note, path="verification_note")
            if findings:
                raise LearningContractError(
                    [ContractIssue(item.code, item.path) for item in findings]
                )
        with Session(self.engine) as session:
            row = challenge_revision(session, challenge_id, revision)
            if row is None:
                raise LearningServiceError(
                    "challenge_revision_not_found", "revision", status_code=404
                )
            reference = _find_reference(row.payload_json or {}, reference_id)
            if reference is None:
                raise LearningServiceError("reference_not_found", "reference_id", status_code=404)
            if verification_state in {"verified_locator", "verified_content"} and not any(
                reference.get(key) for key in ("pmid", "doi", "url")
            ):
                raise LearningServiceError(
                    "verification_requires_reference_locator", "verification_state"
                )

            key = ("challenge", challenge_id, revision, reference_id)
            overlay = session.get(ReferenceVerificationORM, key)
            if overlay is None:
                overlay = ReferenceVerificationORM(
                    artifact_type="challenge",
                    artifact_id=challenge_id,
                    artifact_revision=revision,
                    reference_id=reference_id,
                    verification_state=verification_state,
                    verification_note=verification_note,
                    verified_at=None,
                )
                session.add(overlay)
            overlay.verification_state = verification_state
            overlay.verification_note = verification_note
            overlay.verified_at = utcnow() if verification_state != "unverified" else None
            session.commit()
            return {
                "challenge_id": challenge_id,
                "revision": revision,
                "reference_id": reference_id,
                "verification_state": overlay.verification_state,
                "verification_note": overlay.verification_note,
                "verified_at": serialize_datetime(overlay.verified_at),
            }

    def preview_foundation_assessment(
        self,
        foundation_node_id: str,
        raw_attempt: Any,
        *,
        next_review_due: date | None,
    ) -> dict[str, Any]:
        try:
            attempt = validate_foundation_attempt_payload(
                raw_attempt, expected_node_id=foundation_node_id
            )
        except LearningContractError as exc:
            return {"valid": False, "issues": sanitized_issues(exc)}
        return {
            "valid": True,
            "issues": [],
            "foundation_node_id": foundation_node_id,
            "clinician_final_state": attempt.clinician_final_state,
            "retention_state": _retention_state(next_review_due),
            "next_review_due": serialize_date(next_review_due),
            "normalized_attempt": attempt.model_dump(mode="json"),
        }

    def save_foundation_assessment(
        self,
        foundation_node_id: str,
        raw_attempt: Any,
        *,
        next_review_due: date | None,
        confirm_save: bool,
    ) -> dict[str, Any]:
        if confirm_save is not True:
            raise LearningServiceError("confirm_save_required", "confirm_save")
        attempt = validate_foundation_attempt_payload(
            raw_attempt, expected_node_id=foundation_node_id
        )
        payload = attempt.model_dump(mode="json")
        attempt = validate_foundation_attempt_payload(
            payload, expected_node_id=foundation_node_id
        )
        attempt_id = str(attempt.attempt_id)
        assessed_at = _naive_utc(attempt.assessed_at)
        with Session(self.engine) as session:
            existing = session.get(FoundationAttemptORM, attempt_id)
            if existing is not None:
                if (existing.payload_json or {}) == payload:
                    return self._foundation_state_response(
                        session, foundation_node_id, idempotent=True
                    )
                raise LearningServiceError(
                    "foundation_attempt_id_conflict",
                    "attempt.attempt_id",
                    status_code=409,
                )

            # Serialize materialization for an existing Foundation node. SQLite
            # ignores FOR UPDATE in tests; production Postgres holds the row lock
            # until commit, so an older concurrent assessment cannot commit over
            # a newer state that was waiting on the same authoritative row.
            state = session.execute(
                select(FoundationStateORM)
                .where(FoundationStateORM.foundation_node_id == foundation_node_id)
                .with_for_update()
            ).scalar_one_or_none()
            if (
                state is not None
                and state.last_assessed_at is not None
                and assessed_at <= state.last_assessed_at
            ):
                # Exact same attempt_id returned above as idempotent. A different
                # assessment at the same instant has no deterministic ordering,
                # so it fails closed instead of replacing the materialized state.
                raise LearningServiceError(
                    "foundation_assessment_older_than_current_state",
                    "attempt.assessed_at",
                    status_code=409,
                )

            now = utcnow()
            session.add(
                FoundationAttemptORM(
                    attempt_id=attempt_id,
                    foundation_node_id=foundation_node_id,
                    module=attempt.module,
                    payload_json=payload,
                    assessed_at=assessed_at,
                    created_at=now,
                )
            )

            ids: list[str] = [] if state is None else list(state.evidence_attempt_ids_json or [])
            ids.append(attempt_id)
            if state is None:
                state = FoundationStateORM(
                    foundation_node_id=foundation_node_id,
                    module="osteoporosis",
                    state=attempt.clinician_final_state,
                    retention_state=_retention_state(next_review_due),
                    last_assessed_at=assessed_at,
                    next_review_due=next_review_due,
                    evidence_attempt_ids_json=ids,
                    updated_at=now,
                )
                session.add(state)
            else:
                state.state = attempt.clinician_final_state
                state.retention_state = _retention_state(next_review_due)
                state.last_assessed_at = assessed_at
                state.next_review_due = next_review_due
                state.evidence_attempt_ids_json = ids
                state.updated_at = now

            if next_review_due is not None:
                _materialize_due(
                    session,
                    item_type="foundation_reassessment",
                    target_id=foundation_node_id,
                    due_on=next_review_due,
                    reason_code="explicit_foundation_reassessment_due",
                    source_artifact_type="foundation_assessment",
                    source_artifact_id=attempt_id,
                    source_revision=None,
                )
            else:
                due = _latest_due(session, "foundation_reassessment", foundation_node_id)
                if due is not None and due.completed_at is None:
                    due.due_on = None
                    due.deferred_until = None
                    due.due_status = "not_scheduled"
                    due.source_artifact_type = "foundation_assessment"
                    due.source_artifact_id = attempt_id
                    due.source_revision = None
                    due.updated_at = now

            session.commit()
            return self._foundation_state_response(
                session, foundation_node_id, idempotent=False
            )

    def _foundation_state_response(
        self,
        session: Session,
        foundation_node_id: str,
        *,
        idempotent: bool,
    ) -> dict[str, Any]:
        state = session.get(FoundationStateORM, foundation_node_id)
        attempts = foundation_attempts_for_node(session, foundation_node_id)
        due = _latest_due(session, "foundation_reassessment", foundation_node_id)
        if state is None:
            state_payload = {
                "foundation_node_id": foundation_node_id,
                "module": "osteoporosis",
                "state": "UNKNOWN_UNTESTED",
                "retention_state": "not_scheduled",
                "evidence_attempt_ids": [],
                "last_assessed_at": None,
                "next_review_due": None,
                "linked_signal_ids": [],
                "clinician_note": None,
            }
        else:
            state_payload = {
                "foundation_node_id": state.foundation_node_id,
                "module": state.module,
                "state": state.state,
                "retention_state": state.retention_state,
                "evidence_attempt_ids": list(state.evidence_attempt_ids_json or []),
                "last_assessed_at": serialize_datetime(state.last_assessed_at),
                "next_review_due": serialize_date(state.next_review_due),
                "linked_signal_ids": [],
                "clinician_note": (
                    (attempts[-1].payload_json or {}).get("clinician_note")
                    if attempts
                    else None
                ),
            }
        return {
            "state": state_payload,
            "attempt_count": len(attempts),
            "due": _serialize_due(due) if due is not None else None,
            "idempotent": idempotent,
        }

    def foundation_registry(self) -> list[dict[str, Any]]:
        registry = get_foundation_registry()
        with Session(self.engine) as session:
            out: list[dict[str, Any]] = []
            for node in registry.public_nodes():
                node_id = str(node["node_id"])
                state = self._foundation_state_response(
                    session, node_id, idempotent=False
                )
                out.append({"node": node, **state})
            return out

    def list_due(self) -> list[dict[str, Any]]:
        with Session(self.engine) as session:
            return [_serialize_due(row) for row in due_rows(session)]
