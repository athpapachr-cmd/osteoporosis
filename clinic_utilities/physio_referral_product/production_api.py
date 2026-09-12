from __future__ import annotations

import copy
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from clinic_utilities.physio_referral_runtime import _require_clinical_key
from clinic_utilities.physio_referral_product import knee_oa_projection
from clinic_utilities.physio_referral_product.knee_oa_presentation_v4 import present_project_result


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
        payload["synthetic_only"] = False
        payload["deployment_context"] = "clinical_excellence_cockpit"
        return payload

    @router.post("/project", dependencies=protected)
    def product_project(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Production requests must identify themselves truthfully at the
            # protected transport boundary. The shared projection retains its
            # frozen synthetic compatibility field only as an internal adapter
            # contract used by the already-reviewed local prototype/tests.
            if payload.get("synthetic_only") is not False:
                raise ValueError("production_usage_context_required")
            internal = copy.deepcopy(payload)
            internal["synthetic_only"] = True
            projected = knee_oa_projection.project(internal)
            return present_project_result(projected, payload)
        except (ValueError, TypeError, KeyError, AssertionError, UnicodeError) as exc:
            raise HTTPException(status_code=400, detail="invalid_or_stale_physio_product_request") from exc

    return router


__all__ = ["build_knee_oa_product_router"]
