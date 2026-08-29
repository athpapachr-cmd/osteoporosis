from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from clinic_utilities.physio_evidence_runtime import get_cu1_evidence_resolver
from clinic_utilities.physio_referral_runtime import CU1ContractError, _require_clinical_key, get_cu1_bundle


class CU1EvidenceRequest(BaseModel):
    profile_id: str
    route_id: str
    subtype_id_optional: Optional[str] = None


def _selection_exists(profile_id: str, route_id: str, subtype_id: Optional[str]) -> bool:
    bundle = get_cu1_bundle()
    profile = (bundle.registry.get("profiles") or {}).get(profile_id)
    if not isinstance(profile, Mapping):
        return False
    route = (profile.get("routes") or {}).get(route_id)
    if not isinstance(route, Mapping):
        return False
    if subtype_id:
        allowed = (bundle.registry.get("subtypes") or {}).get(route_id, [])
        if subtype_id not in (allowed or []):
            return False
    return True


def build_cu1_physio_evidence_router() -> APIRouter:
    router = APIRouter(prefix="/clinical/clinic-utilities/physio-referral", tags=["cu1-physio-evidence"])
    protected = [Depends(_require_clinical_key)]

    @router.post("/api/evidence", dependencies=protected)
    def cu1_evidence(req: CU1EvidenceRequest) -> Dict[str, Any]:
        if not _selection_exists(req.profile_id, req.route_id, req.subtype_id_optional):
            raise HTTPException(status_code=422, detail="Invalid CU-1 route/subtype selection")
        try:
            return get_cu1_evidence_resolver().route_summary(
                profile_id=req.profile_id,
                route_id=req.route_id,
                subtype_id=req.subtype_id_optional,
            )
        except CU1ContractError as exc:
            raise HTTPException(status_code=500, detail=f"CU-1 evidence contract error: {exc}") from exc

    return router


__all__ = ["build_cu1_physio_evidence_router", "CU1EvidenceRequest"]
