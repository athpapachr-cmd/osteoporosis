"""Application entrypoint.

The legacy Cockpit/API implementation remains in ``legacy_main.py`` unchanged.
This thin wrapper composes the existing FastAPI app with the protected,
patient-centric clinical-data routers used by the Clinical Excellence baseline UI.
"""

from fastapi import Request
from fastapi.responses import RedirectResponse

from legacy_main import app, engine
from clinical_auth import ClinicalCookieMiddleware, build_auth_router
from clinical_calendar import build_clinical_calendar_router
from clinical_data import build_clinical_router
from clinical_data_ext import build_clinical_ext_router
from clinical_status import build_clinical_status_router
from clinic_utilities.physio_referral_api import build_cu1_physio_referral_router
from clinic_utilities.rf.api import build_rf_router

# The legacy app still owns a historical GET / route. Keep the old Cockpit
# available at /static/index.html, but make the public service root enter the
# current Clinical Excellence workspace instead of the legacy page.
app.router.routes = [
    route
    for route in app.router.routes
    if not (
        getattr(route, "path", None) == "/"
        and "GET" in (getattr(route, "methods", None) or set())
    )
]


@app.middleware("http")
async def prevent_stale_clinical_workspace_assets(request: Request, call_next):
    """Keep actively deployed Clinical Excellence assets coherent across releases."""

    response = await call_next(request)
    if request.url.path.startswith("/static/baseline-audit/") or request.url.path.startswith(
        "/static/clinic-utilities/"
    ):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.get("/", include_in_schema=False)
def clinical_workspace_root() -> RedirectResponse:
    response = RedirectResponse(url="/static/baseline-audit/", status_code=307)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


app.add_middleware(ClinicalCookieMiddleware)
app.include_router(build_auth_router())
app.include_router(build_clinical_status_router(engine))
app.include_router(build_clinical_router(engine))
app.include_router(build_clinical_ext_router(engine))
app.include_router(build_clinical_calendar_router(engine))
app.include_router(build_cu1_physio_referral_router())
# RF v2 is now a native protected Clinic Utility. The old rf_gateway module is
# deliberately left in the repository as rollback-only code but is not mounted.
app.include_router(build_rf_router(engine))

__all__ = ["app", "engine"]
