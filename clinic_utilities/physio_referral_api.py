from __future__ import annotations

import copy
from typing import Any, Dict, Mapping

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from clinic_utilities.physio_referral_runtime import (
    CU1ContractError,
    CU1GenerateRequest,
    CU1GenerateResponse,
    CU1ValidationError,
    CU1ValidationResponse,
    CU1ValidateRequest,
    _repo_root,
    _require_clinical_key,
    get_cu1_bundle,
    get_cu1_engine,
)


def _gateway_target_is_canonical(draft: Mapping[str, Any]) -> bool:
    """Accept shared_target_optional only when it exactly matches a frozen gateway.

    Direct shared-profile selection does not need shared_target_optional. When a
    regional/shared gateway is used, client input is treated as untrusted and
    must match one registry gateway exactly before ownership can be resolved.
    """

    problem = draft.get("primary_problem")
    if not isinstance(problem, Mapping):
        return True
    target = problem.get("shared_target_optional")
    if target is None:
        return True
    if not isinstance(target, Mapping):
        return False

    source_profile = problem.get("profile_id")
    source_route = problem.get("route_id")
    target_profile = target.get("profile_id")
    target_route = target.get("route_id")
    target_subtype_or_site = target.get("subtype_or_site_id_optional")

    gateways = get_cu1_bundle().registry.get("gateways", {})
    if not isinstance(gateways, Mapping):
        return False

    for gateway in gateways.values():
        if not isinstance(gateway, Mapping):
            continue
        if gateway.get("source_profile") != source_profile:
            continue
        if gateway.get("source_route_alias") != source_route:
            continue
        if gateway.get("target_profile") != target_profile:
            continue
        if gateway.get("target_route") != target_route:
            continue
        expected_detail = gateway.get("target_subtype_or_site")
        if expected_detail != target_subtype_or_site:
            continue
        return True
    return False


def _safety_state_is_canonical(draft: Mapping[str, Any]) -> bool:
    """Validate client-controlled SafetyState identifiers against frozen catalogs."""

    safety = draft.get("safety")
    if safety is None:
        return True
    if not isinstance(safety, Mapping):
        return False

    bundle = get_cu1_bundle()
    allowed_rule_ids = set((bundle.rules.get("rules") or {}).keys())
    acknowledged = safety.get("acknowledged_rule_ids", [])
    if not isinstance(acknowledged, list):
        return False
    if any(not isinstance(rule_id, str) or rule_id not in allowed_rule_ids for rule_id in acknowledged):
        return False

    typed_supplement = bundle.artifacts.get("typed_supplement", {})
    disposition_values = (
        typed_supplement.get("safety_result_completion", {})
        .get("fields", {})
        .get("clinician_disposition", [])
        if isinstance(typed_supplement, Mapping)
        else []
    )
    if not isinstance(disposition_values, list) or not disposition_values:
        return False
    disposition = safety.get("clinician_disposition", "none_recorded")
    if not isinstance(disposition, str) or disposition not in disposition_values:
        return False

    return True


def _blocked_validation(draft: Mapping[str, Any], *, path: str, reason: str) -> CU1ValidationResponse:
    return CU1ValidationResponse(
        normalized_draft=copy.deepcopy(dict(draft)),
        validation_errors=[
            CU1ValidationError(
                error_id="invalid_route_or_subtype",
                error_class="validation_error",
                metadata={"path": path, "reason": reason},
            )
        ],
        safety_results=[],
        highest_severity=None,
        formatter_blocked=True,
    )


def _invalid_gateway_validation(draft: Mapping[str, Any]) -> CU1ValidationResponse:
    return _blocked_validation(
        draft,
        path="primary_problem.shared_target_optional",
        reason="not_a_frozen_registry_gateway",
    )


def _invalid_safety_validation(draft: Mapping[str, Any]) -> CU1ValidationResponse:
    return _blocked_validation(
        draft,
        path="safety",
        reason="unknown_acknowledged_rule_id_or_clinician_disposition",
    )


def build_cu1_physio_referral_router() -> APIRouter:
    router = APIRouter(prefix="/clinical/clinic-utilities/physio-referral", tags=["cu1-physio-referral"])
    protected = [Depends(_require_clinical_key)]
    page_path = _repo_root() / "static/clinic-utilities/physio-referral/index.html"

    @router.get("", include_in_schema=False, dependencies=protected)
    def physio_referral_page() -> FileResponse:
        if not page_path.exists():
            raise HTTPException(status_code=500, detail="CU-1 utility page is unavailable")
        return FileResponse(page_path)

    @router.get("/api/contract", dependencies=protected)
    def cu1_contract() -> Dict[str, Any]:
        return get_cu1_bundle().contract_payload()

    @router.post("/api/validate", response_model=CU1ValidationResponse, dependencies=protected)
    def cu1_validate(req: CU1ValidateRequest) -> CU1ValidationResponse:
        try:
            if not _gateway_target_is_canonical(req.draft):
                return _invalid_gateway_validation(req.draft)
            if not _safety_state_is_canonical(req.draft):
                return _invalid_safety_validation(req.draft)
            return get_cu1_engine().validate(req.draft)
        except CU1ContractError as exc:
            raise HTTPException(status_code=500, detail=f"CU-1 contract error: {exc}") from exc

    @router.post("/api/generate", response_model=CU1GenerateResponse, dependencies=protected)
    def cu1_generate(req: CU1GenerateRequest) -> CU1GenerateResponse:
        try:
            if not _gateway_target_is_canonical(req.draft):
                blocked = _invalid_gateway_validation(req.draft)
                return CU1GenerateResponse(**blocked.model_dump(), mode=req.mode, text=None)
            if not _safety_state_is_canonical(req.draft):
                blocked = _invalid_safety_validation(req.draft)
                return CU1GenerateResponse(**blocked.model_dump(), mode=req.mode, text=None)
            return get_cu1_engine().generate(req.draft, req.mode)
        except CU1ContractError as exc:
            raise HTTPException(status_code=500, detail=f"CU-1 contract error: {exc}") from exc

    return router


__all__ = [
    "build_cu1_physio_referral_router",
    "_gateway_target_is_canonical",
    "_safety_state_is_canonical",
]
