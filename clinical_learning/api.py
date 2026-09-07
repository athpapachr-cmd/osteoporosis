from __future__ import annotations

import os
import secrets
from datetime import date
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse
from sqlalchemy.engine import Engine

from .contracts import LearningContractError
from .service import ClinicalLearningService, LearningServiceError, sanitized_issues


def _ui_index() -> Path:
    return Path(__file__).resolve().parents[1] / "static" / "clinical-learning" / "index.html"


def build_learning_router(engine: Engine) -> APIRouter:
    service = ClinicalLearningService(engine)
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

    protected = [Depends(require_learning_key)]

    def service_error(exc: LearningServiceError) -> HTTPException:
        return HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "path": exc.path},
        )

    def contract_error(exc: LearningContractError) -> HTTPException:
        return HTTPException(
            status_code=422,
            detail={"code": "learning_contract_invalid", "issues": sanitized_issues(exc)},
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

    @router.get("/api/due", dependencies=protected)
    def learning_due() -> dict[str, Any]:
        return {"items": service.list_due()}

    return router
