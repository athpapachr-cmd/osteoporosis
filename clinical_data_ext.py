from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional
import os
import secrets

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from clinical_data import LabSnapshotORM, PatientORM


class LabSnapshotUpdate(BaseModel):
    lab_date: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    source_encounter_id: Optional[str] = None
    values: Optional[Dict[str, Any]] = None


class LabSnapshotRecord(BaseModel):
    lab_snapshot_id: str
    patient_id: str
    lab_date: str
    source_encounter_id: Optional[str]
    values: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def build_clinical_ext_router(engine: Engine) -> APIRouter:
    router = APIRouter(prefix="/clinical", tags=["clinical-data"])

    def require_clinical_key(
        x_clinical_key: Optional[str] = Header(default=None, alias="X-Clinical-Key"),
    ) -> None:
        expected = os.environ.get("CLINICAL_DATA_KEY", "")
        if not expected:
            raise HTTPException(status_code=503, detail="Clinical data access is disabled")
        if not x_clinical_key or not secrets.compare_digest(x_clinical_key, expected):
            raise HTTPException(status_code=401, detail="Invalid clinical data key")

    protected = [Depends(require_clinical_key)]

    @router.put("/lab/{lab_snapshot_id}", response_model=LabSnapshotRecord, dependencies=protected)
    def update_lab_snapshot(lab_snapshot_id: str, req: LabSnapshotUpdate) -> LabSnapshotRecord:
        with Session(engine) as session:
            row = session.get(LabSnapshotORM, lab_snapshot_id)
            if row is None:
                raise HTTPException(status_code=404, detail="Lab snapshot not found")
            if req.lab_date is not None:
                row.lab_date = req.lab_date
            if req.source_encounter_id is not None:
                row.source_encounter_id = req.source_encounter_id
            if req.values is not None:
                row.values_json = req.values
            row.updated_at = utcnow()
            patient = session.get(PatientORM, row.patient_id)
            if patient is not None:
                patient.updated_at = row.updated_at
            session.add(row)
            session.commit()
            session.refresh(row)
            return LabSnapshotRecord(
                lab_snapshot_id=row.id,
                patient_id=row.patient_id,
                lab_date=row.lab_date,
                source_encounter_id=row.source_encounter_id,
                values=row.values_json or {},
                created_at=row.created_at,
                updated_at=row.updated_at,
            )

    return router
