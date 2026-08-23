"""Application entrypoint.

The legacy Cockpit/API implementation remains in ``legacy_main.py`` unchanged.
This thin wrapper composes the existing FastAPI app with the protected,
patient-centric clinical-data routers used by the Clinical Excellence baseline UI.
"""

from legacy_main import app, engine
from clinical_data import build_clinical_router
from clinical_data_ext import build_clinical_ext_router

app.include_router(build_clinical_router(engine))
app.include_router(build_clinical_ext_router(engine))

__all__ = ["app", "engine"]
