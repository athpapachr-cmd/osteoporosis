from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from clinic_utilities.physio_referral_runtime import _require_clinical_key
from clinic_utilities.physio_referral_product import knee_oa_projection


def build_knee_oa_product_router() -> APIRouter:
    """Protected Cockpit transport for the reviewed Knee-OA product projection.

    Clinical meaning is owned by the shared deterministic projection module.
    The local prototype is only an alternate loopback transport over that same
    projection and is never mounted in production.
    """

    router = APIRouter(
        prefix="/clinical/clinic-utilities/physio-referral/api/product",
        tags=["physio-referral-knee-oa-product"],
    )
    protected = [Depends(_require_clinical_key)]

    @router.get("/bootstrap", dependencies=protected)
    def product_bootstrap() -> Dict[str, Any]:
        payload = dict(knee_oa_projection.bootstrap())
        payload["deployment_context"] = "clinical_excellence_cockpit"
        return payload

    @router.post("/project", dependencies=protected)
    def product_project(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return knee_oa_projection.project(payload)
        except (ValueError, TypeError, KeyError, AssertionError, UnicodeError) as exc:
            raise HTTPException(status_code=400, detail="invalid_or_stale_physio_product_request") from exc

    return router


__all__ = ["build_knee_oa_product_router"]
