from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4
import os
import secrets

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, JSON, String, select, func
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session


class ClinicalBase(DeclarativeBase):
    pass


class PatientORM(ClinicalBase):
    __tablename__ = "clinical_patients"

    patient_id = Column(String, primary_key=True)
    demographics_json = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, index=True)
    updated_at = Column(DateTime, nullable=False, index=True)


class EncounterORM(ClinicalBase):
    __tablename__ = "clinical_encounters"

    id = Column(String, primary_key=True)
    patient_id = Column(String, nullable=False, index=True)
    encounter_date = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False, index=True, default="draft")
    payload_json = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, index=True)
    updated_at = Column(DateTime, nullable=False, index=True)


class LabSnapshotORM(ClinicalBase):
    __tablename__ = "clinical_lab_snapshots"

    id = Column(String, primary_key=True)
    patient_id = Column(String, nullable=False, index=True)
    lab_date = Column(String, nullable=False, index=True)
    source_encounter_id = Column(String, nullable=True, index=True)
    values_json = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, index=True)
    updated_at = Column(DateTime, nullable=False, index=True)


class PatientUpsert(BaseModel):
    patient_id: str = Field(min_length=1, max_length=120)
    demographics: Dict[str, Any] = Field(default_factory=dict)


class PatientSummary(BaseModel):
    patient_id: str
    demographics: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    encounter_count: int = 0
    lab_snapshot_count: int = 0


class EncounterCreate(BaseModel):
    encounter_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    status: str = Field(default="draft", pattern=r"^(draft|completed|amended)$")
    payload: Dict[str, Any] = Field(default_factory=dict)


class EncounterUpdate(BaseModel):
    encounter_date: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    status: Optional[str] = Field(default=None, pattern=r"^(draft|completed|amended)$")
    payload: Optional[Dict[str, Any]] = None


class EncounterRecord(BaseModel):
    encounter_id: str
    patient_id: str
    encounter_date: str
    status: str
    payload: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class LabSnapshotCreate(BaseModel):
    lab_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    source_encounter_id: Optional[str] = None
    values: Dict[str, Any] = Field(default_factory=dict)


class LabSnapshotRecord(BaseModel):
    lab_snapshot_id: str
    patient_id: str
    lab_date: str
    source_encounter_id: Optional[str]
    values: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class ClinicalStatus(BaseModel):
    database_dialect: str
    protected: bool


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def build_clinical_router(engine: Engine) -> APIRouter:
    ClinicalBase.metadata.create_all(bind=engine)

    router = APIRouter(prefix="/clinical", tags=["clinical-data"])

    def require_clinical_key(
        x_clinical_key: Optional[str] = Header(default=None, alias="X-Clinical-Key"),
    ) -> None:
        expected = os.environ.get("CLINICAL_DATA_KEY", "")
        if not expected:
            raise HTTPException(
                status_code=503,
                detail="Clinical data access is disabled until CLINICAL_DATA_KEY is configured.",
            )
        if not x_clinical_key or not secrets.compare_digest(x_clinical_key, expected):
            raise HTTPException(status_code=401, detail="Invalid clinical data key")

    protected = [Depends(require_clinical_key)]

    def ensure_patient(session: Session, patient_id: str) -> PatientORM:
        patient = session.get(PatientORM, patient_id)
        if patient is None:
            raise HTTPException(status_code=404, detail="Patient not found")
        return patient

    def patient_summary(session: Session, patient: PatientORM) -> PatientSummary:
        encounters = session.scalar(
            select(func.count()).select_from(EncounterORM).where(EncounterORM.patient_id == patient.patient_id)
        ) or 0
        labs = session.scalar(
            select(func.count()).select_from(LabSnapshotORM).where(LabSnapshotORM.patient_id == patient.patient_id)
        ) or 0
        return PatientSummary(
            patient_id=patient.patient_id,
            demographics=patient.demographics_json or {},
            created_at=patient.created_at,
            updated_at=patient.updated_at,
            encounter_count=int(encounters),
            lab_snapshot_count=int(labs),
        )

    @router.get("/status", response_model=ClinicalStatus, dependencies=protected)
    def clinical_status() -> ClinicalStatus:
        return ClinicalStatus(database_dialect=engine.dialect.name, protected=True)

    @router.post("/patients", response_model=PatientSummary, dependencies=protected)
    def upsert_patient(req: PatientUpsert) -> PatientSummary:
        patient_id = req.patient_id.strip()
        if not patient_id:
            raise HTTPException(status_code=422, detail="patient_id is required")
        now = utcnow()
        with Session(engine) as session:
            patient = session.get(PatientORM, patient_id)
            if patient is None:
                patient = PatientORM(
                    patient_id=patient_id,
                    demographics_json=req.demographics,
                    created_at=now,
                    updated_at=now,
                )
                session.add(patient)
            else:
                patient.demographics_json = req.demographics
                patient.updated_at = now
            session.commit()
            session.refresh(patient)
            return patient_summary(session, patient)

    @router.get("/patients", response_model=List[PatientSummary], dependencies=protected)
    def search_patients(
        query: str = Query(default="", max_length=120),
        limit: int = Query(default=20, ge=1, le=100),
    ) -> List[PatientSummary]:
        q = query.strip()
        with Session(engine) as session:
            stmt = select(PatientORM)
            if q:
                stmt = stmt.where(PatientORM.patient_id.ilike(f"%{q}%"))
            stmt = stmt.order_by(PatientORM.updated_at.desc()).limit(limit)
            patients = session.execute(stmt).scalars().all()
            return [patient_summary(session, patient) for patient in patients]

    @router.get("/patient/{patient_id}", response_model=PatientSummary, dependencies=protected)
    def get_patient(patient_id: str) -> PatientSummary:
        with Session(engine) as session:
            patient = ensure_patient(session, patient_id)
            return patient_summary(session, patient)

    @router.post("/patient/{patient_id}/encounters", response_model=EncounterRecord, dependencies=protected)
    def create_encounter(patient_id: str, req: EncounterCreate) -> EncounterRecord:
        now = utcnow()
        with Session(engine) as session:
            ensure_patient(session, patient_id)
            row = EncounterORM(
                id=str(uuid4()),
                patient_id=patient_id,
                encounter_date=req.encounter_date,
                status=req.status,
                payload_json=req.payload,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            patient = session.get(PatientORM, patient_id)
            if patient is not None:
                patient.updated_at = now
            session.commit()
            session.refresh(row)
            return EncounterRecord(
                encounter_id=row.id,
                patient_id=row.patient_id,
                encounter_date=row.encounter_date,
                status=row.status,
                payload=row.payload_json or {},
                created_at=row.created_at,
                updated_at=row.updated_at,
            )

    @router.get("/patient/{patient_id}/encounters", response_model=List[EncounterRecord], dependencies=protected)
    def list_encounters(patient_id: str) -> List[EncounterRecord]:
        with Session(engine) as session:
            ensure_patient(session, patient_id)
            rows = session.execute(
                select(EncounterORM)
                .where(EncounterORM.patient_id == patient_id)
                .order_by(EncounterORM.encounter_date.desc(), EncounterORM.created_at.desc())
            ).scalars().all()
            return [
                EncounterRecord(
                    encounter_id=row.id,
                    patient_id=row.patient_id,
                    encounter_date=row.encounter_date,
                    status=row.status,
                    payload=row.payload_json or {},
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
                for row in rows
            ]

    @router.get("/encounter/{encounter_id}", response_model=EncounterRecord, dependencies=protected)
    def get_encounter(encounter_id: str) -> EncounterRecord:
        with Session(engine) as session:
            row = session.get(EncounterORM, encounter_id)
            if row is None:
                raise HTTPException(status_code=404, detail="Encounter not found")
            return EncounterRecord(
                encounter_id=row.id,
                patient_id=row.patient_id,
                encounter_date=row.encounter_date,
                status=row.status,
                payload=row.payload_json or {},
                created_at=row.created_at,
                updated_at=row.updated_at,
            )

    @router.put("/encounter/{encounter_id}", response_model=EncounterRecord, dependencies=protected)
    def update_encounter(encounter_id: str, req: EncounterUpdate) -> EncounterRecord:
        with Session(engine) as session:
            row = session.get(EncounterORM, encounter_id)
            if row is None:
                raise HTTPException(status_code=404, detail="Encounter not found")
            if req.encounter_date is not None:
                row.encounter_date = req.encounter_date
            if req.status is not None:
                row.status = req.status
            if req.payload is not None:
                row.payload_json = req.payload
            row.updated_at = utcnow()
            patient = session.get(PatientORM, row.patient_id)
            if patient is not None:
                patient.updated_at = row.updated_at
            session.add(row)
            session.commit()
            session.refresh(row)
            return EncounterRecord(
                encounter_id=row.id,
                patient_id=row.patient_id,
                encounter_date=row.encounter_date,
                status=row.status,
                payload=row.payload_json or {},
                created_at=row.created_at,
                updated_at=row.updated_at,
            )

    @router.post("/patient/{patient_id}/labs", response_model=LabSnapshotRecord, dependencies=protected)
    def create_lab_snapshot(patient_id: str, req: LabSnapshotCreate) -> LabSnapshotRecord:
        now = utcnow()
        with Session(engine) as session:
            ensure_patient(session, patient_id)
            if req.source_encounter_id:
                encounter = session.get(EncounterORM, req.source_encounter_id)
                if encounter is None or encounter.patient_id != patient_id:
                    raise HTTPException(status_code=422, detail="source_encounter_id does not belong to patient")
            row = LabSnapshotORM(
                id=str(uuid4()),
                patient_id=patient_id,
                lab_date=req.lab_date,
                source_encounter_id=req.source_encounter_id,
                values_json=req.values,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            patient = session.get(PatientORM, patient_id)
            if patient is not None:
                patient.updated_at = now
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

    @router.get("/patient/{patient_id}/labs", response_model=List[LabSnapshotRecord], dependencies=protected)
    def list_lab_snapshots(patient_id: str) -> List[LabSnapshotRecord]:
        with Session(engine) as session:
            ensure_patient(session, patient_id)
            rows = session.execute(
                select(LabSnapshotORM)
                .where(LabSnapshotORM.patient_id == patient_id)
                .order_by(LabSnapshotORM.lab_date.asc(), LabSnapshotORM.created_at.asc())
            ).scalars().all()
            return [
                LabSnapshotRecord(
                    lab_snapshot_id=row.id,
                    patient_id=row.patient_id,
                    lab_date=row.lab_date,
                    source_encounter_id=row.source_encounter_id,
                    values=row.values_json or {},
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
                for row in rows
            ]

    return router
