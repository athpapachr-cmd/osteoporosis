from __future__ import annotations

import os
import secrets
from datetime import date
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from .contracts import LearningContractError
from .l1b_runtime import LearningLoopRuntimeError, LearningLoopRuntimeService
from .service import ClinicalLearningService, LearningServiceError, sanitized_issues


def _ui_index() -> Path:
    return Path(__file__).resolve().parents[1] / "static" / "clinical-learning" / "index.html"


def build_learning_router(engine: Engine) -> APIRouter:
    service = ClinicalLearningService(engine)
    loop_service = LearningLoopRuntimeService(engine)
    router = APIRouter(prefix="/clinical/learning", tags=["clinical-learning"])

    def require_learning_key(
        x_clinical_key: Optional[str] = Header(default=None, alias="X-Clinical-Key"),
    ) -> None:
        expected = os.environ.get("CLINICAL_DATA_KEY", "")
        if not expected:
            raise HTTPException(
                status_code=503,
                detail={"code": "clinical_access_not_configured"},
            )
        if not x_clinical_key or not secrets.compare_digest(x_clinical_key, expected):
            raise HTTPException(status_code=401, detail={"code": "invalid_clinical_access"})

    def require_learning_ingest_key(
        x_learning_ingest_key: Optional[str] = Header(default=None, alias="X-Learning-Ingest-Key"),
    ) -> None:
        expected = os.environ.get("CLINICAL_LEARNING_INGEST_KEY", "")
        if not expected:
            raise HTTPException(
                status_code=503,
                detail={"code": "learning_ingress_not_configured"},
            )
        if not x_learning_ingest_key or not secrets.compare_digest(x_learning_ingest_key, expected):
            raise HTTPException(status_code=401, detail={"code": "invalid_learning_ingress"})

    protected = [Depends(require_learning_key)]
    ingress_protected = [Depends(require_learning_ingest_key)]

    def service_error(exc: LearningServiceError) -> HTTPException:
        return HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "path": exc.path},
        )

    def loop_error(exc: LearningLoopRuntimeError) -> HTTPException:
        return HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "path": exc.path},
        )

    def contract_error(exc: LearningContractError) -> HTTPException:
        return HTTPException(
            status_code=422,
            detail={"code": "learning_contract_invalid", "issues": sanitized_issues(exc)},
        )

    def integrity_conflict() -> HTTPException:
        return HTTPException(
            status_code=409,
            detail={"code": "learning_write_conflict_retry"},
        )

    async def json_object(request: Request) -> dict[str, Any]:
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail={"code": "invalid_json"}) from None
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail={"code": "object_body_required"})
        return payload

    def envelope(
        payload: dict[str, Any],
        *,
        required: set[str],
        optional: set[str] | None = None,
    ) -> dict[str, Any]:
        optional = optional or set()
        allowed = required | optional
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise HTTPException(
                status_code=422,
                detail={"code": "unknown_envelope_field", "path": str(unknown[0])},
            )
        missing = sorted(required - set(payload))
        if missing:
            raise HTTPException(
                status_code=422,
                detail={"code": "required_envelope_field_missing", "path": str(missing[0])},
            )
        return payload

    @router.get("", dependencies=protected, include_in_schema=False)
    def learning_page() -> FileResponse:
        return FileResponse(_ui_index(), media_type="text/html")

    # L-1B external ingress is deliberately separate from the broad clinical key.
    # It can create only a synthetic pending learning import, never an accepted
    # Challenge, patient write, Foundation state, Signal or reference verification.
    @router.post("/api/ingress/episodes", dependencies=ingress_protected)
    async def external_episode_ingress(request: Request) -> dict[str, Any]:
        payload = envelope(
            await json_object(request),
            required={"episode"},
            optional={"source_event_id", "loop_plan", "resources"},
        )
        episode = payload["episode"]
        if not isinstance(episode, dict):
            raise HTTPException(status_code=422, detail={"code": "learning_episode_object_required"})
        mode = episode.get("challenge_mode")
        if mode is None and isinstance(episode.get("session"), dict):
            mode = episode.get("challenge_mode", "synthetic")
        if mode not in {None, "synthetic"}:
            raise HTTPException(
                status_code=422,
                detail={"code": "external_ingress_synthetic_only", "path": "episode.challenge_mode"},
            )
        try:
            return loop_service.ingest_episode(
                episode,
                source_event_id=(str(payload["source_event_id"]) if payload.get("source_event_id") else None),
                loop_plan=(payload.get("loop_plan") if isinstance(payload.get("loop_plan"), dict) else None),
                resources=(payload.get("resources") if isinstance(payload.get("resources"), list) else None),
            )
        except LearningContractError as exc:
            raise contract_error(exc) from None
        except LearningLoopRuntimeError as exc:
            raise loop_error(exc) from None
        except IntegrityError:
            raise integrity_conflict() from None

    @router.post("/api/imports", dependencies=protected)
    async def pending_import_create(request: Request) -> dict[str, Any]:
        payload = envelope(
            await json_object(request),
            required={"episode"},
            optional={"source_event_id", "loop_plan", "resources"},
        )
        try:
            return loop_service.ingest_episode(
                payload["episode"],
                source_event_id=(str(payload["source_event_id"]) if payload.get("source_event_id") else None),
                loop_plan=(payload.get("loop_plan") if isinstance(payload.get("loop_plan"), dict) else None),
                resources=(payload.get("resources") if isinstance(payload.get("resources"), list) else None),
            )
        except LearningContractError as exc:
            raise contract_error(exc) from None
        except LearningLoopRuntimeError as exc:
            raise loop_error(exc) from None
        except IntegrityError:
            raise integrity_conflict() from None

    @router.get("/api/imports", dependencies=protected)
    def pending_import_list() -> dict[str, Any]:
        return {"items": loop_service.list_pending_imports()}

    @router.get("/api/imports/{import_id}", dependencies=protected)
    def pending_import_detail(import_id: str) -> dict[str, Any]:
        try:
            return loop_service.get_pending_import(import_id)
        except LearningLoopRuntimeError as exc:
            raise loop_error(exc) from None

    @router.post("/api/imports/{import_id}/reject", dependencies=protected)
    async def pending_import_reject(import_id: str, request: Request) -> dict[str, Any]:
        payload = envelope(await json_object(request), required={"confirm_reject"})
        if payload.get("confirm_reject") is not True:
            raise HTTPException(status_code=422, detail={"code": "confirm_reject_required"})
        try:
            return loop_service.reject_pending_import(import_id)
        except LearningLoopRuntimeError as exc:
            raise loop_error(exc) from None

    @router.post("/api/imports/{import_id}/accepted", dependencies=protected)
    async def pending_import_accepted(import_id: str, request: Request) -> dict[str, Any]:
        payload = envelope(
            await json_object(request),
            required={"challenge_id", "revision", "confirm_link"},
        )
        if payload.get("confirm_link") is not True:
            raise HTTPException(status_code=422, detail={"code": "confirm_link_required"})
        try:
            return loop_service.accept_pending_import(
                import_id,
                challenge_id=str(payload["challenge_id"]),
                revision=int(payload["revision"]),
            )
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_revision", "path": "revision"},
            ) from None
        except LearningLoopRuntimeError as exc:
            raise loop_error(exc) from None
        except IntegrityError:
            raise integrity_conflict() from None

    @router.get("/api/learning-loops", dependencies=protected)
    def learning_loop_list() -> dict[str, Any]:
        return {"items": loop_service.list_learning_loops()}

    @router.get("/api/learning-loops/{cycle_id}", dependencies=protected)
    def learning_loop_detail(cycle_id: str) -> dict[str, Any]:
        try:
            return loop_service.get_learning_loop(cycle_id)
        except LearningLoopRuntimeError as exc:
            raise loop_error(exc) from None

    @router.post(
        "/api/learning-loops/{cycle_id}/occurrences/{occurrence_id}/attempts",
        dependencies=protected,
    )
    async def consolidation_attempt(cycle_id: str, occurrence_id: str, request: Request) -> dict[str, Any]:
        payload = envelope(
            await json_object(request),
            required={"response_text", "result"},
            optional={"evaluator_note", "clinician_reviewed"},
        )
        try:
            return loop_service.add_consolidation_attempt(cycle_id, occurrence_id, payload)
        except LearningContractError as exc:
            raise contract_error(exc) from None
        except LearningLoopRuntimeError as exc:
            raise loop_error(exc) from None
        except IntegrityError:
            raise integrity_conflict() from None

    @router.post(
        "/api/learning-resources/{recommendation_id}/status",
        dependencies=protected,
    )
    async def learning_resource_status(recommendation_id: str, request: Request) -> dict[str, Any]:
        payload = envelope(await json_object(request), required={"status"})
        try:
            return loop_service.update_resource_status(recommendation_id, payload)
        except LearningLoopRuntimeError as exc:
            raise loop_error(exc) from None

    @router.post("/api/challenges/preview", dependencies=protected)
    async def challenge_preview(request: Request) -> dict[str, Any]:
        payload = envelope(await json_object(request), required={"challenge"})
        return service.preview_challenge(payload["challenge"])

    @router.post("/api/challenges", dependencies=protected)
    async def challenge_create(request: Request) -> dict[str, Any]:
        payload = envelope(await json_object(request), required={"challenge", "confirm_save"})
        try:
            return service.create_challenge(
                payload["challenge"],
                confirm_save=payload["confirm_save"],
            )
        except LearningContractError as exc:
            raise contract_error(exc) from None
        except LearningServiceError as exc:
            raise service_error(exc) from None
        except IntegrityError:
            raise integrity_conflict() from None

    @router.put("/api/challenges/{challenge_id}", dependencies=protected)
    async def challenge_revise(challenge_id: str, request: Request) -> dict[str, Any]:
        payload = envelope(await json_object(request), required={"challenge", "confirm_save"})
        try:
            return service.revise_challenge(
                challenge_id,
                payload["challenge"],
                confirm_save=payload["confirm_save"],
            )
        except LearningContractError as exc:
            raise contract_error(exc) from None
        except LearningServiceError as exc:
            raise service_error(exc) from None
        except IntegrityError:
            raise integrity_conflict() from None

    @router.get("/api/challenges", dependencies=protected)
    def challenge_history(
        topic: str | None = Query(default=None, max_length=120),
        foundation_node: str | None = Query(default=None, max_length=160),
        challenge_date: date | None = Query(default=None),
        challenge_mode: str | None = Query(default=None, max_length=60),
        review_state: str | None = Query(default=None, max_length=60),
    ) -> dict[str, Any]:
        return {
            "items": service.list_challenges(
                topic=topic,
                foundation_node=foundation_node,
                challenge_date=challenge_date,
                challenge_mode=challenge_mode,
                review_state=review_state,
            )
        }

    @router.get("/api/challenges/{challenge_id}", dependencies=protected)
    def challenge_detail(challenge_id: str) -> dict[str, Any]:
        try:
            return service.get_challenge(challenge_id)
        except LearningServiceError as exc:
            raise service_error(exc) from None

    @router.delete("/api/challenges/{challenge_id}", dependencies=protected)
    async def challenge_delete(challenge_id: str, request: Request) -> dict[str, Any]:
        payload = envelope(await json_object(request), required={"confirm_delete"})
        try:
            return service.delete_challenge(
                challenge_id,
                confirm_delete=payload["confirm_delete"],
            )
        except LearningServiceError as exc:
            raise service_error(exc) from None
        except IntegrityError:
            raise integrity_conflict() from None

    @router.post(
        "/api/challenges/{challenge_id}/revisions/{revision}/references/{reference_id}/verification",
        dependencies=protected,
    )
    async def reference_verification(
        challenge_id: str,
        revision: int,
        reference_id: str,
        request: Request,
    ) -> dict[str, Any]:
        payload = envelope(
            await json_object(request),
            required={"verification_state"},
            optional={"verification_note"},
        )
        try:
            return service.set_reference_verification(
                challenge_id=challenge_id,
                revision=revision,
                reference_id=reference_id,
                verification_state=str(payload["verification_state"]),
                verification_note=(
                    str(payload["verification_note"])
                    if payload.get("verification_note") is not None
                    else None
                ),
            )
        except LearningContractError as exc:
            raise contract_error(exc) from None
        except LearningServiceError as exc:
            raise service_error(exc) from None
        except IntegrityError:
            raise integrity_conflict() from None

    @router.get("/api/foundation", dependencies=protected)
    def foundation_registry() -> dict[str, Any]:
        return {"items": service.foundation_registry()}

    @router.post(
        "/api/foundation/{foundation_node_id}/assessments/preview",
        dependencies=protected,
    )
    async def foundation_preview(foundation_node_id: str, request: Request) -> dict[str, Any]:
        payload = envelope(
            await json_object(request),
            required={"attempt"},
            optional={"next_review_due", "confirm_save"},
        )
        next_due: date | None = None
        if payload.get("next_review_due"):
            try:
                next_due = date.fromisoformat(str(payload["next_review_due"]))
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail={"code": "invalid_date", "path": "next_review_due"},
                ) from None
        return service.preview_foundation_assessment(
            foundation_node_id,
            payload["attempt"],
            next_review_due=next_due,
        )

    @router.post(
        "/api/foundation/{foundation_node_id}/assessments",
        dependencies=protected,
    )
    async def foundation_save(foundation_node_id: str, request: Request) -> dict[str, Any]:
        payload = envelope(
            await json_object(request),
            required={"attempt", "confirm_save"},
            optional={"next_review_due"},
        )
        next_due: date | None = None
        if payload.get("next_review_due"):
            try:
                next_due = date.fromisoformat(str(payload["next_review_due"]))
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail={"code": "invalid_date", "path": "next_review_due"},
                ) from None
        try:
            return service.save_foundation_assessment(
                foundation_node_id,
                payload["attempt"],
                next_review_due=next_due,
                confirm_save=payload["confirm_save"],
            )
        except LearningContractError as exc:
            raise contract_error(exc) from None
        except LearningServiceError as exc:
            raise service_error(exc) from None
        except IntegrityError:
            raise integrity_conflict() from None

    @router.get("/api/due", dependencies=protected)
    def learning_due() -> dict[str, Any]:
        return {"items": service.list_due()}

    return router
