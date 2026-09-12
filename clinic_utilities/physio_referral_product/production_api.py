from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from clinic_utilities.physio_referral_runtime import _require_clinical_key
from clinic_utilities.physio_referral_product.prototype import server as knee_oa


def build_knee_oa_product_router() -> APIRouter:
    """Protected Cockpit transport for the reviewed Knee-OA product projection.

    The loopback prototype HTTP server is never mounted. Only its deterministic,
    real-CU1-backed bootstrap/project functions are reused so the production UI
    and the reviewed synthetic candidate cannot silently diverge in clinical
    meaning during this first integration slice.
    """

    router = APIRouter(
        prefix="/clinical/clinic-utilities/physio-referral/api/product",
        tags=["physio-referral-knee-oa-product"],
    )
    protected = [Depends(_require_clinical_key)]

    @router.get("/bootstrap", dependencies=protected)
    def product_bootstrap() -> Dict[str, Any]:
        payload = dict(knee_oa.bootstrap())
        payload["deployment_context"] = "clinical_excellence_cockpit"
        return payload

    @router.post("/project", dependencies=protected)
    def product_project(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return knee_oa.project(payload)
        except (ValueError, TypeError, KeyError, AssertionError, UnicodeError) as exc:
            raise HTTPException(status_code=400, detail="invalid_or_stale_physio_product_request") from exc

    return router


__all__ = ["build_knee_oa_product_router"]
