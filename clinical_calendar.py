from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
import os
import re
import secrets
import unicodedata

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Integer, String, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session


class CalendarBase(DeclarativeBase):
    pass


class ClinicalAppointmentORM(CalendarBase):
    __tablename__ = "clinical_appointments"

    id = Column(String, primary_key=True)
    source = Column(String, nullable=False, index=True, default="setmore")
    source_appointment_id = Column(String, nullable=False, index=True)
    start_at = Column(DateTime, nullable=False, index=True)
    end_at = Column(DateTime, nullable=False, index=True)
    duration_minutes = Column(Integer, nullable=False, default=0)
    clinic = Column(String, nullable=False, default="")
    category = Column(String, nullable=False, index=True, default="other")
    patient_display_name = Column(String, nullable=False, default="")
    phone_e164 = Column(String, nullable=False, default="")
    linked_patient_id = Column(String, nullable=True, index=True)
    label = Column(String, nullable=False, default="")
    comment = Column(String, nullable=False, default="")
    status = Column(String, nullable=False, default="scheduled")
    updated_at = Column(DateTime, nullable=False, index=True)


CATEGORY_VALUES = {
    "osteoporosis_first",
    "osteoporosis_review",
    "osteoporosis_unspecified",
    "prolia",
    "aclasta",
    "other",
}


class AppointmentImport(BaseModel):
    source: str = Field(default="setmore", max_length=40)
    source_appointment_id: str = Field(min_length=1, max_length=160)
    start_at: datetime
    end_at: datetime
    clinic: str = Field(default="", max_length=80)
    category: Optional[str] = Field(default=None, max_length=80)
    patient_display_name: str = Field(default="", max_length=200)
    phone_e164: str = Field(default="", max_length=40)
    linked_patient_id: Optional[str] = Field(default=None, max_length=120)
    label: str = Field(default="", max_length=240)
    comment: str = Field(default="", max_length=1000)
    status: str = Field(default="scheduled", max_length=40)


class AppointmentRecord(BaseModel):
    appointment_id: str
    source: str
    source_appointment_id: str
    start_at: datetime
    end_at: datetime
    duration_minutes: int
    clinic: str
    category: str
    patient_display_name: str
    phone_e164: str
    linked_patient_id: Optional[str]
    label: str
    comment: str
    status: str
    updated_at: datetime


class AppointmentImportResult(BaseModel):
    imported: int
    inserted: int
    updated: int


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _normalize(value: str) -> str:
    text = unicodedata.normalize("NFD", (value or "").lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9α-ω\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def classify_appointment(label: str, comment: str, duration_minutes: int) -> str:
    token = _normalize(f"{label} {comment}")
    if any(x in token for x in ("prolia", "denosumab", "δενοσουμαμπ")):
        return "prolia"
    if any(x in token for x in ("aclasta", "zoledronic", "zoledron", "ζολεδρον", "ζολεδρονικο")):
        return "aclasta"

    osteo = any(
        x in token
        for x in (
            "οστεοπορωση",
            "οστεοπενια",
            "osteoporosis",
            "osteopenia",
            "dxa",
            "dexa",
            "οστικη πυκνοτητα",
        )
    )
    if not osteo:
        return "other"

    if any(x in token for x in ("πρωτη επισκεψη", "πρωτη φορα", "first visit", "new patient", "initial")):
        return "osteoporosis_first"
    if any(x in token for x in ("review", "follow up", "followup", "reassessment", "επαναξιολογηση", "επανελεγχ", "παρακολουθηση")):
        return "osteoporosis_review"

    # Duration alone must not invent first-vs-review semantics. Both may be 60'.
    return "osteoporosis_unspecified"


def _record(row: ClinicalAppointmentORM) -> AppointmentRecord:
    return AppointmentRecord(
        appointment_id=row.id,
        source=row.source,
        source_appointment_id=row.source_appointment_id,
        start_at=row.start_at,
        end_at=row.end_at,
        duration_minutes=int(row.duration_minutes or 0),
        clinic=row.clinic or "",
        category=row.category or "other",
        patient_display_name=row.patient_display_name or "",
        phone_e164=row.phone_e164 or "",
        linked_patient_id=row.linked_patient_id,
        label=row.label or "",
        comment=row.comment or "",
        status=row.status or "scheduled",
        updated_at=row.updated_at,
    )


def build_clinical_calendar_router(engine: Engine) -> APIRouter:
    CalendarBase.metadata.create_all(bind=engine)
    router = APIRouter(prefix="/clinical/calendar", tags=["clinical-calendar"])

    def require_clinical_key(
        x_clinical_key: Optional[str] = Header(default=None, alias="X-Clinical-Key"),
    ) -> None:
        expected = os.environ.get("CLINICAL_DATA_KEY", "")
        if not expected:
            raise HTTPException(status_code=503, detail="Clinical data access is disabled")
        if not x_clinical_key or not secrets.compare_digest(x_clinical_key, expected):
            raise HTTPException(status_code=401, detail="Invalid clinical data key")

    def require_ingest_key(
        x_clinical_ingest_key: Optional[str] = Header(default=None, alias="X-Clinical-Ingest-Key"),
    ) -> None:
        expected = os.environ.get("CLINICAL_INGEST_KEY", "")
        if not expected:
            raise HTTPException(status_code=503, detail="CLINICAL_INGEST_KEY is not configured")
        if not x_clinical_ingest_key or not secrets.compare_digest(x_clinical_ingest_key, expected):
            raise HTTPException(status_code=401, detail="Invalid clinical ingest key")

    protected = [Depends(require_clinical_key)]
    ingest_protected = [Depends(require_ingest_key)]

    @router.get("/appointments", response_model=List[AppointmentRecord], dependencies=protected)
    def list_appointments(
        start: datetime = Query(...),
        end: datetime = Query(...),
        include_other: bool = Query(default=False),
    ) -> List[AppointmentRecord]:
        start_utc = _naive_utc(start)
        end_utc = _naive_utc(end)
        if end_utc <= start_utc:
            raise HTTPException(status_code=422, detail="end must be after start")
        if (end_utc - start_utc).days > 31:
            raise HTTPException(status_code=422, detail="calendar range is limited to 31 days")

        with Session(engine) as session:
            stmt = (
                select(ClinicalAppointmentORM)
                .where(ClinicalAppointmentORM.start_at >= start_utc)
                .where(ClinicalAppointmentORM.start_at < end_utc)
                .order_by(ClinicalAppointmentORM.start_at.asc())
            )
            if not include_other:
                stmt = stmt.where(ClinicalAppointmentORM.category != "other")
            rows = session.execute(stmt).scalars().all()
            return [_record(row) for row in rows]

    @router.post("/appointments/import", response_model=AppointmentImportResult, dependencies=ingest_protected)
    def import_appointments(rows: List[AppointmentImport]) -> AppointmentImportResult:
        if len(rows) > 500:
            raise HTTPException(status_code=422, detail="maximum 500 appointments per import")
        inserted = 0
        updated = 0
        now = utcnow()

        with Session(engine) as session:
            for item in rows:
                start_at = _naive_utc(item.start_at)
                end_at = _naive_utc(item.end_at)
                if end_at <= start_at:
                    continue
                duration_minutes = max(int((end_at - start_at).total_seconds() // 60), 0)
                category = item.category if item.category in CATEGORY_VALUES else classify_appointment(
                    item.label,
                    item.comment,
                    duration_minutes,
                )
                record_id = f"{item.source}:{item.source_appointment_id}"
                row = session.get(ClinicalAppointmentORM, record_id)
                if row is None:
                    row = ClinicalAppointmentORM(id=record_id, source=item.source, source_appointment_id=item.source_appointment_id)
                    session.add(row)
                    inserted += 1
                else:
                    updated += 1

                row.start_at = start_at
                row.end_at = end_at
                row.duration_minutes = duration_minutes
                row.clinic = item.clinic.strip()
                row.category = category
                row.patient_display_name = item.patient_display_name.strip()
                row.phone_e164 = item.phone_e164.strip()
                row.linked_patient_id = item.linked_patient_id.strip() if item.linked_patient_id else None
                row.label = item.label.strip()
                row.comment = item.comment.strip()
                row.status = item.status.strip() or "scheduled"
                row.updated_at = now

            session.commit()

        return AppointmentImportResult(imported=inserted + updated, inserted=inserted, updated=updated)

    return router
