from __future__ import annotations

import os
import secrets
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Integer, String, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session


class SurgeryQueueBase(DeclarativeBase):
    pass


class SurgeryQueueORM(SurgeryQueueBase):
    __tablename__ = "clinical_surgery_queue"

    id = Column(String, primary_key=True)
    identity_number = Column(String, nullable=False, index=True)
    full_name = Column(String, nullable=False, index=True)
    date_of_birth = Column(String, nullable=False)
    phone = Column(String, nullable=False)
    procedure_type = Column(String, nullable=False)
    laterality = Column(String, nullable=False, index=True)
    surgery_date = Column(String, nullable=True, index=True)
    queue_position = Column(Integer, nullable=False, index=True)
    status = Column(String, nullable=False, index=True, default="pending")
    completed_at = Column(DateTime, nullable=True, index=True)
    created_at = Column(DateTime, nullable=False, index=True)
    updated_at = Column(DateTime, nullable=False, index=True)


Laterality = Literal["left", "right", "bilateral", "not_applicable", "unspecified"]
MoveDirection = Literal["up", "down"]


class SurgeryCreate(BaseModel):
    identity_number: str = Field(min_length=1, max_length=80)
    full_name: str = Field(min_length=1, max_length=200)
    date_of_birth: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    phone: str = Field(min_length=1, max_length=40)
    procedure_type: str = Field(min_length=1, max_length=240)
    laterality: Laterality
    surgery_date: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")


class SurgeryUpdate(BaseModel):
    identity_number: Optional[str] = Field(default=None, min_length=1, max_length=80)
    full_name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    date_of_birth: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    phone: Optional[str] = Field(default=None, min_length=1, max_length=40)
    procedure_type: Optional[str] = Field(default=None, min_length=1, max_length=240)
    laterality: Optional[Laterality] = None
    surgery_date: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")


class SurgeryMove(BaseModel):
    direction: MoveDirection


class SurgeryRecord(BaseModel):
    surgery_id: str
    identity_number: str
    full_name: str
    date_of_birth: str
    phone: str
    procedure_type: str
    laterality: Laterality
    surgery_date: Optional[str]
    queue_position: int
    status: str
    created_at: datetime
    updated_at: datetime


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _clean_text(value: str) -> str:
    return " ".join((value or "").strip().split())


def build_surgery_queue_router(engine: Engine) -> APIRouter:
    SurgeryQueueBase.metadata.create_all(bind=engine)
    router = APIRouter(prefix="/clinical/surgeries", tags=["clinical-surgery-queue"])

    def require_clinical_key(
        x_clinical_key: Optional[str] = Header(default=None, alias="X-Clinical-Key"),
    ) -> None:
        expected = os.environ.get("CLINICAL_DATA_KEY", "")
        if not expected:
            raise HTTPException(status_code=503, detail="Clinical data access is disabled")
        if not x_clinical_key or not secrets.compare_digest(x_clinical_key, expected):
            raise HTTPException(status_code=401, detail="Invalid clinical data key")

    protected = [Depends(require_clinical_key)]

    def get_row(session: Session, surgery_id: str) -> SurgeryQueueORM:
        row = session.get(SurgeryQueueORM, surgery_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Surgery queue item not found")
        return row

    def normalize_pending_positions(session: Session) -> list[SurgeryQueueORM]:
        rows = session.execute(
            select(SurgeryQueueORM)
            .where(SurgeryQueueORM.status == "pending")
            .order_by(SurgeryQueueORM.queue_position.asc(), SurgeryQueueORM.created_at.asc())
        ).scalars().all()
        for index, row in enumerate(rows, start=1):
            row.queue_position = index
            session.add(row)
        return rows

    def record(row: SurgeryQueueORM) -> SurgeryRecord:
        return SurgeryRecord(
            surgery_id=row.id,
            identity_number=row.identity_number,
            full_name=row.full_name,
            date_of_birth=row.date_of_birth,
            phone=row.phone,
            procedure_type=row.procedure_type,
            laterality=row.laterality,
            surgery_date=row.surgery_date,
            queue_position=int(row.queue_position or 0),
            status=row.status,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    def list_records(session: Session, status: str) -> list[SurgeryRecord]:
        stmt = select(SurgeryQueueORM)
        if status != "all":
            stmt = stmt.where(SurgeryQueueORM.status == status)
        if status == "pending":
            stmt = stmt.order_by(
                SurgeryQueueORM.queue_position.asc(),
                SurgeryQueueORM.created_at.asc(),
            )
        else:
            stmt = stmt.order_by(SurgeryQueueORM.updated_at.desc())
        return [record(row) for row in session.execute(stmt).scalars().all()]

    @router.get("", response_model=list[SurgeryRecord], dependencies=protected)
    def list_surgeries(status: str = Query(default="pending")) -> list[SurgeryRecord]:
        if status not in {"pending", "completed", "all"}:
            raise HTTPException(status_code=422, detail="status must be pending, completed or all")
        with Session(engine) as session:
            if status == "pending":
                normalize_pending_positions(session)
                session.commit()
            return list_records(session, status)

    @router.post("", response_model=SurgeryRecord, dependencies=protected)
    def create_surgery(req: SurgeryCreate) -> SurgeryRecord:
        now = utcnow()
        with Session(engine) as session:
            pending = normalize_pending_positions(session)
            row = SurgeryQueueORM(
                id=str(uuid.uuid4()),
                identity_number=_clean_text(req.identity_number),
                full_name=_clean_text(req.full_name),
                date_of_birth=req.date_of_birth,
                phone=_clean_text(req.phone),
                procedure_type=_clean_text(req.procedure_type),
                laterality=req.laterality,
                surgery_date=req.surgery_date,
                queue_position=len(pending) + 1,
                status="pending",
                completed_at=None,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return record(row)

    @router.put("/{surgery_id}", response_model=SurgeryRecord, dependencies=protected)
    def update_surgery(surgery_id: str, req: SurgeryUpdate) -> SurgeryRecord:
        now = utcnow()
        with Session(engine) as session:
            row = get_row(session, surgery_id)
            if req.identity_number is not None:
                row.identity_number = _clean_text(req.identity_number)
            if req.full_name is not None:
                row.full_name = _clean_text(req.full_name)
            if req.date_of_birth is not None:
                row.date_of_birth = req.date_of_birth
            if req.phone is not None:
                row.phone = _clean_text(req.phone)
            if req.procedure_type is not None:
                row.procedure_type = _clean_text(req.procedure_type)
            if req.laterality is not None:
                row.laterality = req.laterality
            if "surgery_date" in req.model_fields_set:
                row.surgery_date = req.surgery_date
            row.updated_at = now
            session.add(row)
            session.commit()
            session.refresh(row)
            return record(row)

    @router.post("/{surgery_id}/move", response_model=list[SurgeryRecord], dependencies=protected)
    def move_surgery(surgery_id: str, req: SurgeryMove) -> list[SurgeryRecord]:
        with Session(engine) as session:
            row = get_row(session, surgery_id)
            if row.status != "pending":
                raise HTTPException(status_code=409, detail="Only pending surgeries can be reordered")

            rows = normalize_pending_positions(session)
            ids = [item.id for item in rows]
            current_index = ids.index(row.id)
            target_index = current_index - 1 if req.direction == "up" else current_index + 1
            if target_index < 0 or target_index >= len(rows):
                session.commit()
                return list_records(session, "pending")

            rows[current_index], rows[target_index] = rows[target_index], rows[current_index]
            now = utcnow()
            moved_ids = {rows[current_index].id, rows[target_index].id}
            for index, item in enumerate(rows, start=1):
                item.queue_position = index
                if item.id in moved_ids:
                    item.updated_at = now
                session.add(item)
            session.commit()
            return list_records(session, "pending")

    @router.post("/{surgery_id}/complete", response_model=SurgeryRecord, dependencies=protected)
    def complete_surgery(surgery_id: str) -> SurgeryRecord:
        now = utcnow()
        with Session(engine) as session:
            row = get_row(session, surgery_id)
            if row.status != "pending":
                raise HTTPException(status_code=409, detail="Surgery is not pending")
            row.status = "completed"
            row.completed_at = now
            row.updated_at = now
            row.queue_position = 0
            session.add(row)
            normalize_pending_positions(session)
            session.commit()
            session.refresh(row)
            return record(row)

    @router.delete("/{surgery_id}", response_model=SurgeryRecord, dependencies=protected)
    def delete_pending_surgery(surgery_id: str) -> SurgeryRecord:
        now = utcnow()
        with Session(engine) as session:
            row = get_row(session, surgery_id)
            if row.status != "pending":
                raise HTTPException(status_code=409, detail="Only pending surgeries can be deleted")
            # Soft-delete: remove from the active queue while preserving an audit record.
            row.status = "deleted"
            row.updated_at = now
            row.queue_position = 0
            session.add(row)
            normalize_pending_positions(session)
            session.commit()
            session.refresh(row)
            return record(row)

    return router
