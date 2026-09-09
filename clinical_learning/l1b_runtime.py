from __future__ import annotations

import copy
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from .contracts import (
    LearningContractError,
    canonical_content_hash,
    get_foundation_registry,
    normalize_import_preview,
    sanitized_schema_issues,
    validate_challenge_payload,
)
from .ingress import adapt_learning_episode
from .learning_loop import stable_learning_uuid
from .models import (
    ConsolidationAttemptCreateV1,
    LearningLoopPlanV1,
    LearningResourceRecommendationV1,
    LearningResourceStatusUpdateV1,
)
from .persistence import (
    ChallengeRevisionORM,
    ConsolidationAttemptORM,
    DueItemORM,
    LearningLoopPlanORM,
    LearningResourceRecommendationORM,
    PendingImportORM,
    challenge_revision,
    consolidation_attempts_for_cycle,
    due_rows,
    init_learning_storage,
    loop_plans,
    pending_import_by_source,
    pending_imports,
    resource_recommendations_for_challenge,
    serialize_date,
    serialize_datetime,
)
from .privacy import scan_persistable_strings


class LearningLoopRuntimeError(ValueError):
    def __init__(self, code: str, path: str = "", *, status_code: int = 422):
        self.code = code
        self.path = path
        self.status_code = status_code
        super().__init__(code)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _candidate_hash(
    normalized_challenge: dict[str, Any],
    loop_plan: dict[str, Any],
    resources: list[dict[str, Any]],
    warnings: list[str],
) -> str:
    encoded = json.dumps(
        {
            "normalized_challenge": normalized_challenge,
            "loop_plan": loop_plan,
            "resources": resources,
            "warnings": warnings,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _due_status(due_on: date) -> str:
    today = datetime.now(timezone.utc).date()
    if due_on > today:
        return "scheduled"
    if due_on == today:
        return "due"
    return "overdue"


def _require_unique(values: list[Any], *, code: str, path: str) -> None:
    normalized = [str(value) for value in values]
    if len(normalized) != len(set(normalized)):
        raise LearningLoopRuntimeError(code, path)


def _validate_foundation_nodes(values: list[str], *, path: str) -> None:
    registry = get_foundation_registry()
    for index, node_id in enumerate(values):
        if not registry.contains(node_id):
            raise LearningLoopRuntimeError(
                "unknown_foundation_node",
                f"{path}[{index}]",
            )


def _validate_loop_plan(raw: Any) -> dict[str, Any]:
    try:
        model = LearningLoopPlanV1.model_validate(raw)
    except ValidationError as exc:
        issues = sanitized_schema_issues(exc)
        first = issues[0] if issues else None
        raise LearningLoopRuntimeError(
            "learning_loop_contract_invalid",
            first.path if first else "loop_plan",
        ) from None
    payload = model.model_dump(mode="json")

    objective_ids = [str(item["objective_id"]) for item in payload.get("objectives") or []]
    _require_unique(
        objective_ids,
        code="duplicate_learning_objective_id",
        path="loop_plan.objectives",
    )
    for index, objective in enumerate(payload.get("objectives") or []):
        _require_unique(
            list(objective.get("foundation_node_ids") or []),
            code="duplicate_learning_objective_foundation_node",
            path=f"loop_plan.objectives[{index}].foundation_node_ids",
        )
        _validate_foundation_nodes(
            list(objective.get("foundation_node_ids") or []),
            path=f"loop_plan.objectives[{index}].foundation_node_ids",
        )
        _require_unique(
            list(objective.get("source_observation_ids") or []),
            code="duplicate_learning_objective_observation_id",
            path=f"loop_plan.objectives[{index}].source_observation_ids",
        )

    bridge_ids = [str(item["bridge_id"]) for item in payload.get("bridge_targets") or []]
    _require_unique(
        bridge_ids,
        code="duplicate_bridge_id",
        path="loop_plan.bridge_targets",
    )
    for index, bridge in enumerate(payload.get("bridge_targets") or []):
        nodes = list(bridge.get("foundation_node_ids") or [])
        _require_unique(
            nodes,
            code="duplicate_bridge_foundation_node",
            path=f"loop_plan.bridge_targets[{index}].foundation_node_ids",
        )
        _validate_foundation_nodes(
            nodes,
            path=f"loop_plan.bridge_targets[{index}].foundation_node_ids",
        )
        _require_unique(
            list(bridge.get("source_observation_ids") or []),
            code="duplicate_bridge_observation_id",
            path=f"loop_plan.bridge_targets[{index}].source_observation_ids",
        )
        _require_unique(
            list(bridge.get("source_action_ids") or []),
            code="duplicate_bridge_action_id",
            path=f"loop_plan.bridge_targets[{index}].source_action_ids",
        )

    occurrence_ids = [str(item["occurrence_id"]) for item in payload.get("consolidation_occurrences") or []]
    _require_unique(
        occurrence_ids,
        code="duplicate_consolidation_occurrence_id",
        path="loop_plan.consolidation_occurrences",
    )
    sequences = [int(item["sequence"]) for item in payload.get("consolidation_occurrences") or []]
    if sequences != list(range(1, len(sequences) + 1)):
        raise LearningLoopRuntimeError(
            "invalid_consolidation_sequence",
            "loop_plan.consolidation_occurrences",
        )

    objective_set = set(objective_ids)
    bridge_set = set(bridge_ids)
    for index, item in enumerate(payload.get("consolidation_occurrences") or []):
        objective_targets = list(item.get("target_objective_ids") or [])
        bridge_targets = list(item.get("target_bridge_ids") or [])
        foundation_targets = list(item.get("target_foundation_node_ids") or [])
        _require_unique(
            objective_targets,
            code="duplicate_consolidation_objective_id",
            path=f"loop_plan.consolidation_occurrences[{index}].target_objective_ids",
        )
        _require_unique(
            bridge_targets,
            code="duplicate_consolidation_bridge_id",
            path=f"loop_plan.consolidation_occurrences[{index}].target_bridge_ids",
        )
        _require_unique(
            foundation_targets,
            code="duplicate_consolidation_foundation_node",
            path=f"loop_plan.consolidation_occurrences[{index}].target_foundation_node_ids",
        )
        _validate_foundation_nodes(
            foundation_targets,
            path=f"loop_plan.consolidation_occurrences[{index}].target_foundation_node_ids",
        )
        if any(str(value) not in objective_set for value in objective_targets):
            raise LearningLoopRuntimeError(
                "unresolved_consolidation_objective_id",
                f"loop_plan.consolidation_occurrences[{index}].target_objective_ids",
            )
        if any(str(value) not in bridge_set for value in bridge_targets):
            raise LearningLoopRuntimeError(
                "unresolved_consolidation_bridge_id",
                f"loop_plan.consolidation_occurrences[{index}].target_bridge_ids",
            )

    if payload.get("default_spacing_days") != [3, 7, 14, 30]:
        raise LearningLoopRuntimeError(
            "unsupported_initial_spacing_profile",
            "loop_plan.default_spacing_days",
        )
    findings = scan_persistable_strings({"loop_plan": payload})
    if findings:
        raise LearningLoopRuntimeError(findings[0].code, findings[0].path)
    return payload


def _validate_resources(raw_resources: Any) -> list[dict[str, Any]]:
    if raw_resources is None:
        return []
    if not isinstance(raw_resources, list):
        raise LearningLoopRuntimeError("resources_array_required", "resources")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_resources):
        try:
            model = LearningResourceRecommendationV1.model_validate(raw)
        except ValidationError as exc:
            issues = sanitized_schema_issues(exc)
            first = issues[0] if issues else None
            raise LearningLoopRuntimeError(
                "learning_resource_contract_invalid",
                first.path if first else f"resources[{index}]",
            ) from None
        payload = model.model_dump(mode="json")
        recommendation_id = str(payload["recommendation_id"])
        if recommendation_id in seen:
            raise LearningLoopRuntimeError("duplicate_resource_recommendation_id", "resources")
        seen.add(recommendation_id)
        _require_unique(
            list(payload.get("foundation_node_ids") or []),
            code="duplicate_resource_foundation_node",
            path=f"resources[{index}].foundation_node_ids",
        )
        _validate_foundation_nodes(
            list(payload.get("foundation_node_ids") or []),
            path=f"resources[{index}].foundation_node_ids",
        )
        out.append(payload)
    findings = scan_persistable_strings({"resources": out})
    if findings:
        raise LearningLoopRuntimeError(findings[0].code, findings[0].path)
    return out


def _validate_rich_episode_profile(raw_episode: Any) -> None:
    if not isinstance(raw_episode, dict):
        return
    if raw_episode.get("schema_type") != "ClinicalLearningChallengeV1":
        return
    if str(raw_episode.get("schema_version")) not in {"1.0", "1"}:
        return
    if str(raw_episode.get("challenge_mode") or "") != "synthetic":
        raise LearningLoopRuntimeError(
            "rich_learning_episode_synthetic_only",
            "episode.challenge_mode",
        )
    try:
        revision = int(raw_episode.get("revision") or 1)
    except (TypeError, ValueError):
        raise LearningLoopRuntimeError(
            "rich_learning_episode_revision_invalid",
            "episode.revision",
        ) from None
    if revision != 1:
        raise LearningLoopRuntimeError(
            "rich_learning_episode_revision_1_only",
            "episode.revision",
        )


def _challenge_import_view(payload: dict[str, Any]) -> dict[str, Any]:
    challenge = validate_challenge_payload(payload)
    normalized = normalize_import_preview(challenge)
    normalized["privacy"]["deidentification_attested"] = False
    return normalized


def _anchor_loop_at_acceptance(loop_payload: dict[str, Any]) -> dict[str, Any]:
    anchored = copy.deepcopy(loop_payload)
    anchor = datetime.now(timezone.utc).date()
    spacing = list(anchored.get("default_spacing_days") or [])
    occurrences = anchored.get("consolidation_occurrences") or []
    if len(spacing) != len(occurrences):
        raise LearningLoopRuntimeError(
            "spacing_occurrence_count_mismatch",
            "loop_plan.consolidation_occurrences",
        )
    for index, occurrence in enumerate(occurrences):
        occurrence["due_on"] = (anchor + timedelta(days=int(spacing[index]))).isoformat()
    return anchored


class LearningLoopRuntimeService:
    def __init__(self, engine: Engine):
        self.engine = engine
        init_learning_storage(engine)

    def _pending_record(self, row: PendingImportORM, *, idempotent: bool = False) -> dict[str, Any]:
        return {
            "import_id": row.import_id,
            "source_event_id": row.source_event_id,
            "source_format": row.source_format,
            "state": row.state,
            "normalized_challenge": copy.deepcopy(row.normalized_challenge_json or {}),
            "loop_plan": copy.deepcopy(row.loop_plan_json or {}),
            "resources": copy.deepcopy(row.resources_json or []),
            "adapter_warnings": list(row.warnings_json or []),
            "created_at": serialize_datetime(row.created_at),
            "resolved_at": serialize_datetime(row.resolved_at),
            "accepted_challenge_id": row.accepted_challenge_id,
            "accepted_revision": row.accepted_revision,
            "idempotent": idempotent,
        }

    def ingest_episode(
        self,
        raw_episode: Any,
        *,
        source_event_id: str | None = None,
        loop_plan: dict[str, Any] | None = None,
        resources: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        _validate_rich_episode_profile(raw_episode)
        adapted = adapt_learning_episode(
            raw_episode,
            source_event_id=source_event_id,
            supplied_loop_plan=loop_plan,
            supplied_resources=resources,
        )
        checked_loop = _validate_loop_plan(adapted.loop_plan)
        checked_resources = _validate_resources(adapted.resources)
        normalized_hash = _candidate_hash(
            adapted.normalized_challenge,
            checked_loop,
            checked_resources,
            adapted.warnings,
        )
        with Session(self.engine) as session:
            existing = pending_import_by_source(session, adapted.source_event_id)
            if existing is not None:
                if existing.normalized_hash != normalized_hash:
                    raise LearningLoopRuntimeError(
                        "source_event_id_content_conflict",
                        "source_event_id",
                        status_code=409,
                    )
                return self._pending_record(existing, idempotent=True)
            now = _utcnow()
            row = PendingImportORM(
                import_id=str(uuid4()),
                source_event_id=adapted.source_event_id,
                source_format=adapted.source_format,
                state="pending_review",
                normalized_hash=normalized_hash,
                normalized_challenge_json=copy.deepcopy(adapted.normalized_challenge),
                loop_plan_json=copy.deepcopy(checked_loop),
                resources_json=copy.deepcopy(checked_resources),
                warnings_json=list(adapted.warnings),
                created_at=now,
                resolved_at=None,
                accepted_challenge_id=None,
                accepted_revision=None,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return self._pending_record(row)

    def list_pending_imports(self) -> list[dict[str, Any]]:
        with Session(self.engine) as session:
            return [self._pending_record(row) for row in pending_imports(session)]

    def get_pending_import(self, import_id: str) -> dict[str, Any]:
        with Session(self.engine) as session:
            row = session.get(PendingImportORM, import_id)
            if row is None:
                raise LearningLoopRuntimeError("pending_import_not_found", "import_id", status_code=404)
            return self._pending_record(row)

    def reject_pending_import(self, import_id: str) -> dict[str, Any]:
        with Session(self.engine) as session:
            row = session.execute(
                select(PendingImportORM)
                .where(PendingImportORM.import_id == import_id)
                .with_for_update()
            ).scalar_one_or_none()
            if row is None:
                raise LearningLoopRuntimeError("pending_import_not_found", "import_id", status_code=404)
            if row.state == "accepted":
                raise LearningLoopRuntimeError("accepted_import_cannot_be_rejected", "state", status_code=409)
            if row.state == "rejected":
                return self._pending_record(row, idempotent=True)
            row.state = "rejected"
            row.resolved_at = _utcnow()
            session.commit()
            return self._pending_record(row)

    def accept_pending_import(
        self,
        import_id: str,
        *,
        challenge_id: str,
        revision: int,
    ) -> dict[str, Any]:
        with Session(self.engine) as session:
            pending = session.execute(
                select(PendingImportORM)
                .where(PendingImportORM.import_id == import_id)
                .with_for_update()
            ).scalar_one_or_none()
            if pending is None:
                raise LearningLoopRuntimeError("pending_import_not_found", "import_id", status_code=404)
            if pending.state == "rejected":
                raise LearningLoopRuntimeError("rejected_import_cannot_be_accepted", "state", status_code=409)
            if pending.state == "accepted":
                if pending.accepted_challenge_id == challenge_id and pending.accepted_revision == revision:
                    return self._pending_record(pending, idempotent=True)
                raise LearningLoopRuntimeError("pending_import_already_accepted_elsewhere", "state", status_code=409)

            challenge_row = challenge_revision(session, challenge_id, revision)
            if challenge_row is None:
                raise LearningLoopRuntimeError("accepted_challenge_revision_not_found", "challenge_id", status_code=404)
            accepted_view = _challenge_import_view(challenge_row.payload_json or {})
            pending_view = copy.deepcopy(pending.normalized_challenge_json or {})
            pending_view.setdefault("privacy", {})["deidentification_attested"] = False
            if canonical_content_hash(accepted_view) != canonical_content_hash(pending_view):
                raise LearningLoopRuntimeError(
                    "accepted_challenge_does_not_match_pending_candidate",
                    "challenge_id",
                    status_code=409,
                )

            loop_payload = _anchor_loop_at_acceptance(
                _validate_loop_plan(pending.loop_plan_json or {})
            )
            cycle_id = str(loop_payload["cycle_id"])
            existing_loop = session.get(LearningLoopPlanORM, cycle_id)
            now = _utcnow()
            if existing_loop is None:
                session.add(
                    LearningLoopPlanORM(
                        cycle_id=cycle_id,
                        challenge_id=challenge_id,
                        source_revision=revision,
                        payload_json=copy.deepcopy(loop_payload),
                        created_at=now,
                        updated_at=now,
                    )
                )
            elif (
                existing_loop.challenge_id != challenge_id
                or existing_loop.source_revision != revision
                or (existing_loop.payload_json or {}) != loop_payload
            ):
                raise LearningLoopRuntimeError("learning_loop_identity_conflict", "cycle_id", status_code=409)

            checked_resources = _validate_resources(pending.resources_json or [])
            for resource in checked_resources:
                recommendation_id = str(resource["recommendation_id"])
                existing_resource = session.get(LearningResourceRecommendationORM, recommendation_id)
                if existing_resource is None:
                    session.add(
                        LearningResourceRecommendationORM(
                            recommendation_id=recommendation_id,
                            challenge_id=challenge_id,
                            source_revision=revision,
                            payload_json=copy.deepcopy(resource),
                            status=str(resource["status"]),
                            checked_at=_naive_utc(datetime.fromisoformat(str(resource["checked_at"]).replace("Z", "+00:00"))),
                            updated_at=now,
                        )
                    )
                elif (
                    existing_resource.challenge_id != challenge_id
                    or existing_resource.source_revision != revision
                    or (existing_resource.payload_json or {}) != resource
                ):
                    raise LearningLoopRuntimeError("resource_identity_conflict", "recommendation_id", status_code=409)

            for occurrence in loop_payload.get("consolidation_occurrences") or []:
                occurrence_id = str(occurrence["occurrence_id"])
                existing_due = session.execute(
                    select(DueItemORM).where(
                        DueItemORM.item_type == "consolidation_test",
                        DueItemORM.target_id == occurrence_id,
                        DueItemORM.occurrence == 1,
                    )
                ).scalar_one_or_none()
                due_on = date.fromisoformat(str(occurrence["due_on"]))
                if existing_due is None:
                    session.add(
                        DueItemORM(
                            due_item_id=str(uuid4()),
                            item_type="consolidation_test",
                            target_id=occurrence_id,
                            occurrence=1,
                            due_on=due_on,
                            due_status=_due_status(due_on),
                            delivery_mode="visible",
                            reason_code=f"learning_loop_{occurrence['kind']}",
                            source_artifact_type="learning_loop",
                            source_artifact_id=cycle_id,
                            source_revision=revision,
                            completed_at=None,
                            deferred_until=None,
                            updated_at=now,
                        )
                    )
                elif existing_due.source_artifact_id != cycle_id:
                    raise LearningLoopRuntimeError("consolidation_due_identity_conflict", "occurrence_id", status_code=409)

            pending.state = "accepted"
            pending.resolved_at = now
            pending.accepted_challenge_id = challenge_id
            pending.accepted_revision = revision
            session.commit()
            return self._pending_record(pending)

    def _loop_record(self, session: Session, row: LearningLoopPlanORM) -> dict[str, Any]:
        challenge = session.get(ChallengeRevisionORM, (row.challenge_id, row.source_revision))
        attempts = consolidation_attempts_for_cycle(session, row.cycle_id)
        resources = resource_recommendations_for_challenge(
            session, row.challenge_id, row.source_revision
        )
        due_map = {
            item.target_id: item
            for item in session.execute(
                select(DueItemORM).where(
                    DueItemORM.item_type == "consolidation_test",
                    DueItemORM.source_artifact_type == "learning_loop",
                    DueItemORM.source_artifact_id == row.cycle_id,
                )
            ).scalars()
        }
        payload = copy.deepcopy(row.payload_json or {})
        for occurrence in payload.get("consolidation_occurrences") or []:
            due = due_map.get(str(occurrence.get("occurrence_id")))
            occurrence["due_state"] = (
                {
                    "due_item_id": due.due_item_id,
                    "due_on": serialize_date(due.due_on),
                    "due_status": "completed" if due.completed_at is not None else _due_status(due.due_on),
                    "completed_at": serialize_datetime(due.completed_at),
                }
                if due is not None and due.due_on is not None
                else None
            )
        return {
            "cycle_id": row.cycle_id,
            "challenge_id": row.challenge_id,
            "source_revision": row.source_revision,
            "challenge_title": (challenge.payload_json or {}).get("title") if challenge is not None else None,
            "plan": payload,
            "attempts": [copy.deepcopy(item.payload_json or {}) for item in attempts],
            "resources": [copy.deepcopy(item.payload_json or {}) for item in resources],
            "created_at": serialize_datetime(row.created_at),
            "updated_at": serialize_datetime(row.updated_at),
        }

    def list_learning_loops(self) -> list[dict[str, Any]]:
        with Session(self.engine) as session:
            return [self._loop_record(session, row) for row in loop_plans(session)]

    def get_learning_loop(self, cycle_id: str) -> dict[str, Any]:
        with Session(self.engine) as session:
            row = session.get(LearningLoopPlanORM, cycle_id)
            if row is None:
                raise LearningLoopRuntimeError("learning_loop_not_found", "cycle_id", status_code=404)
            return self._loop_record(session, row)

    def add_consolidation_attempt(
        self,
        cycle_id: str,
        occurrence_id: str,
        raw_attempt: Any,
    ) -> dict[str, Any]:
        try:
            request = ConsolidationAttemptCreateV1.model_validate(raw_attempt)
        except ValidationError as exc:
            issues = sanitized_schema_issues(exc)
            first = issues[0] if issues else None
            raise LearningLoopRuntimeError(
                "consolidation_attempt_invalid",
                first.path if first else "attempt",
            ) from None
        if request.result != "not_assessed" and not request.clinician_reviewed:
            raise LearningLoopRuntimeError(
                "consolidation_result_requires_clinician_review",
                "clinician_reviewed",
            )
        persistable = request.model_dump(mode="json")
        findings = scan_persistable_strings({"attempt": persistable})
        if findings:
            raise LearningLoopRuntimeError(findings[0].code, findings[0].path)

        response_text = request.response_text.strip()
        evaluator_note = (request.evaluator_note or "").strip()
        attempt_id = stable_learning_uuid(
            "consolidation-attempt",
            cycle_id,
            occurrence_id,
            response_text,
            request.result,
            evaluator_note,
            request.clinician_reviewed,
        )

        with Session(self.engine) as session:
            loop = session.execute(
                select(LearningLoopPlanORM)
                .where(LearningLoopPlanORM.cycle_id == cycle_id)
                .with_for_update()
            ).scalar_one_or_none()
            if loop is None:
                raise LearningLoopRuntimeError("learning_loop_not_found", "cycle_id", status_code=404)
            payload = copy.deepcopy(loop.payload_json or {})
            occurrences = payload.get("consolidation_occurrences") or []
            occurrence = next(
                (item for item in occurrences if str(item.get("occurrence_id")) == occurrence_id),
                None,
            )
            if occurrence is None:
                raise LearningLoopRuntimeError("consolidation_occurrence_not_found", "occurrence_id", status_code=404)

            existing_attempt = session.get(ConsolidationAttemptORM, attempt_id)
            if existing_attempt is not None:
                return {**copy.deepcopy(existing_attempt.payload_json or {}), "idempotent": True}
            if occurrence.get("status") == "completed":
                raise LearningLoopRuntimeError(
                    "consolidation_occurrence_already_completed",
                    "occurrence_id",
                    status_code=409,
                )

            now = _utcnow()
            attempt_payload = {
                "attempt_id": attempt_id,
                "occurrence_id": occurrence_id,
                "answered_at": _utcnow_iso(),
                "response_text": response_text,
                "result": request.result,
                "evaluator_note": evaluator_note or None,
                "clinician_reviewed": request.clinician_reviewed,
            }
            session.add(
                ConsolidationAttemptORM(
                    attempt_id=attempt_id,
                    cycle_id=cycle_id,
                    occurrence_id=occurrence_id,
                    payload_json=copy.deepcopy(attempt_payload),
                    answered_at=now,
                    created_at=now,
                )
            )
            occurrence["status"] = "completed"
            loop.payload_json = payload
            loop.updated_at = now

            due = session.execute(
                select(DueItemORM)
                .where(
                    DueItemORM.item_type == "consolidation_test",
                    DueItemORM.target_id == occurrence_id,
                    DueItemORM.source_artifact_type == "learning_loop",
                    DueItemORM.source_artifact_id == cycle_id,
                )
                .with_for_update()
            ).scalar_one_or_none()
            if due is not None and due.completed_at is None:
                due.completed_at = now
                due.due_status = "completed"
                due.updated_at = now

            if occurrence.get("kind") == "bridge_transfer" and request.clinician_reviewed:
                bridge_ids = {str(value) for value in occurrence.get("target_bridge_ids") or []}
                for bridge in payload.get("bridge_targets") or []:
                    if str(bridge.get("bridge_id")) not in bridge_ids:
                        continue
                    if request.result in {"retained", "improved_beyond_original"}:
                        bridge["state"] = "demonstrated"
                    elif request.result in {"partially_retained", "not_retained"}:
                        bridge["state"] = "needs_reinforcement"
                loop.payload_json = payload

            session.commit()
            return {**attempt_payload, "idempotent": False}

    def update_resource_status(
        self,
        recommendation_id: str,
        raw_update: Any,
    ) -> dict[str, Any]:
        try:
            request = LearningResourceStatusUpdateV1.model_validate(raw_update)
        except ValidationError as exc:
            issues = sanitized_schema_issues(exc)
            first = issues[0] if issues else None
            raise LearningLoopRuntimeError(
                "resource_status_update_invalid",
                first.path if first else "status",
            ) from None
        with Session(self.engine) as session:
            row = session.execute(
                select(LearningResourceRecommendationORM)
                .where(LearningResourceRecommendationORM.recommendation_id == recommendation_id)
                .with_for_update()
            ).scalar_one_or_none()
            if row is None:
                raise LearningLoopRuntimeError("resource_recommendation_not_found", "recommendation_id", status_code=404)
            payload = copy.deepcopy(row.payload_json or {})
            payload["status"] = request.status
            row.payload_json = payload
            row.status = request.status
            row.updated_at = _utcnow()
            session.commit()
            return payload

    def list_learning_due(self) -> list[dict[str, Any]]:
        with Session(self.engine) as session:
            return [
                {
                    "due_item_id": row.due_item_id,
                    "item_type": row.item_type,
                    "target_id": row.target_id,
                    "occurrence": row.occurrence,
                    "due_on": serialize_date(row.due_on),
                    "due_status": "completed" if row.completed_at is not None else (
                        _due_status(row.due_on) if row.due_on is not None else row.due_status
                    ),
                    "reason_code": row.reason_code,
                    "source_artifact_type": row.source_artifact_type,
                    "source_artifact_id": row.source_artifact_id,
                    "source_revision": row.source_revision,
                    "completed_at": serialize_datetime(row.completed_at),
                }
                for row in due_rows(session)
            ]
