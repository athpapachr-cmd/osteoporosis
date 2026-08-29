"""Application entrypoint.

The legacy Cockpit/API implementation remains in ``legacy_main.py`` unchanged.
This thin wrapper composes the existing FastAPI app with the protected,
patient-centric clinical-data routers used by the Clinical Excellence baseline UI.
"""

from fastapi.responses import RedirectResponse

from legacy_main import app, engine
from clinical_auth import ClinicalCookieMiddleware, build_auth_router
from clinical_calendar import build_clinical_calendar_router
from clinical_data import build_clinical_router
from clinical_data_ext import build_clinical_ext_router
from clinical_status import build_clinical_status_router
from clinic_utilities.physio_evidence_api import build_cu1_physio_evidence_router
from clinic_utilities.physio_referral_api import build_cu1_physio_referral_router

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


@app.get("/", include_in_schema=False)
def clinical_workspace_root() -> RedirectResponse:
    return RedirectResponse(url="/static/baseline-audit/", status_code=307)


app.add_middleware(ClinicalCookieMiddleware)
app.include_router(build_auth_router())
app.include_router(build_clinical_status_router(engine))
app.include_router(build_clinical_router(engine))
app.include_router(build_clinical_ext_router(engine))
app.include_router(build_clinical_calendar_router(engine))
app.include_router(build_cu1_physio_referral_router())
app.include_router(build_cu1_physio_evidence_router())

__all__ = ["app", "engine"]
