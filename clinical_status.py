from __future__ import annotations

import os

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy.engine import Engine


class ClinicalStorageStatus(BaseModel):
    database_dialect: str
    database_url_configured: bool
    storage_mode: str
    clinical_key_configured: bool


def build_clinical_status_router(engine: Engine) -> APIRouter:
    router = APIRouter(prefix="/clinical", tags=["clinical-status"])
    dialect = engine.dialect.name
    database_url_configured = bool(os.environ.get("DATABASE_URL", "").strip())
    storage_mode = "online_database" if dialect != "sqlite" else "sqlite_fallback"
    clinical_key_configured = bool(os.environ.get("CLINICAL_DATA_KEY", "").strip())

    print(
        "[CLINICAL_STORAGE] "
        f"dialect={dialect} "
        f"database_url_configured={database_url_configured} "
        f"storage_mode={storage_mode} "
        f"clinical_key_configured={clinical_key_configured}"
    )

    @router.get("/storage-status", response_model=ClinicalStorageStatus)
    def storage_status() -> ClinicalStorageStatus:
        return ClinicalStorageStatus(
            database_dialect=dialect,
            database_url_configured=database_url_configured,
            storage_mode=storage_mode,
            clinical_key_configured=clinical_key_configured,
        )

    return router
