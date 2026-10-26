"""Protected Office Cases register and exact RF-to-surgery bridge."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from clinical_office_cases_models import (
    AccidentCaseAuditORM, AccidentCaseORM, AccidentPaymentORM, SurgeryDetailORM,
)
from clinical_surgery_queue import (
    SurgeryQueueORM, _clean_text, _valid_room_date, normalize_pending_positions, utcnow,
)
from clinic_utilities.physio_referral_runtime import _require_clinical_key
from clinic_utilities.rf.persistence import (
    RFApplicationORM, RFApplicationProgressORM, _rf_progress_card,
    track_rf_applications,
)


def _date(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    try:
        if len(value) != 10 or date.fromisoformat(value).isoformat() != value:
            raise ValueError
    except ValueError as exc:
        raise ValueError("Use an ISO calendar date") from exc
    return value


def _text(value: str, label: str) -> str:
    cleaned = _clean_text(value)
    if not cleaned or any(ord(ch) < 32 for ch in cleaned):
        raise HTTPException(status_code=422, detail=f"Invalid {label}")
    return cleaned


def _rf_surgery_prefill(app: RFApplicationORM) -> dict[str, str | None]:
    payload = app.payload_json if isinstance(app.payload_json, dict) else {}
    birth = payload.get("date_of_birth")
    if not isinstance(birth, str):
        birth = None
    else:
        try:
            birth = _date(birth)
        except ValueError:
            birth = None
    phone = payload.get("patient_phone")
    if not isinstance(phone, str) or not phone.strip() or len(phone) > 40 or not phone.isprintable():
        phone = None
    location = _clean_text(app.exact_location)
    procedure = f"Ραδιοσυχνότητες — {location}" if location else ""
    if len(procedure) > 240 or any(ord(char) < 32 for char in procedure):
        procedure = ""
    return {"date_of_birth": birth, "phone": phone, "procedure_type": procedure}


class AccidentCreate(BaseModel):
    patient_name: str = Field(min_length=1, max_length=200)
    phone: str = Field(min_length=1, max_length=40)
    accident_date: str
    visit_date: Optional[str] = None
    writing_started_date: Optional[str] = None
    report_sent_date: Optional[str] = None
    amount_due_cents: Optional[int] = Field(default=None, ge=0, le=2147483647)

    @field_validator("accident_date", "visit_date", "writing_started_date", "report_sent_date")
    @classmethod
    def validate_date(cls, value):
        return _date(value)


class AccidentEdit(BaseModel):
    expected_revision: int = Field(ge=1)
    patient_name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    phone: Optional[str] = Field(default=None, min_length=1, max_length=40)
    accident_date: Optional[str] = None
    visit_date: Optional[str] = None
    writing_started_date: Optional[str] = None
    report_sent_date: Optional[str] = None
    amount_due_cents: Optional[int] = Field(default=None, ge=0, le=2147483647)

    @field_validator("accident_date", "visit_date", "writing_started_date", "report_sent_date")
    @classmethod
    def validate_date(cls, value):
        return _date(value)


class PaymentCreate(BaseModel):
    payment_request_id: uuid.UUID
    amount_cents: int = Field(gt=0, le=2147483647)
    paid_on: str
    payer: str = Field(min_length=1, max_length=200)

    @field_validator("paid_on")
    @classmethod
    def validate_date(cls, value):
        return _date(value)


class PaymentVoid(BaseModel):
    expected_revision: int = Field(ge=1)
    reason: str = Field(min_length=3, max_length=500)


class RFToSurgery(BaseModel):
    expected_rf_revision: int = Field(ge=0)
    date_of_birth: str
    phone: str = Field(min_length=1, max_length=40)
    procedure_type: str = Field(min_length=1, max_length=240)
    surgery_date: Optional[str] = None
    room_entry_time: Optional[str] = Field(default=None, pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")

    @field_validator("date_of_birth", "surgery_date")
    @classmethod
    def validate_date(cls, value):
        return _date(value)


CASE_FIELDS = (
    "patient_name", "phone", "accident_date", "visit_date", "writing_started_date",
    "report_sent_date", "amount_due_cents",
)


def _snapshot(row: AccidentCaseORM) -> dict:
    return {key: getattr(row, key) for key in CASE_FIELDS}


def _payments(session: Session, case_id: str) -> list[AccidentPaymentORM]:
    return list(session.scalars(select(AccidentPaymentORM).where(
        AccidentPaymentORM.case_id == case_id,
    ).order_by(AccidentPaymentORM.created_at, AccidentPaymentORM.id)).all())


def _payment_json(row: AccidentPaymentORM) -> dict:
    return {
        "payment_id": row.id, "amount_cents": row.amount_cents,
        "paid_on": row.paid_on, "payer": row.payer,
        "created_at": row.created_at, "voided_at": row.voided_at,
        "void_reason": row.void_reason,
    }


def _case_json(session: Session, row: AccidentCaseORM, *, detail: bool = False) -> dict:
    events = _payments(session, row.id)
    received = sum(event.amount_cents for event in events if event.voided_at is None)
    due = row.amount_due_cents
    result = {
        "case_id": row.id, **_snapshot(row), "revision": row.revision,
        "received_cents": received, "balance_cents": None if due is None else due - received,
        "created_at": row.created_at, "updated_at": row.updated_at,
    }
    if detail:
        result["payments"] = [_payment_json(event) for event in events]
    return result


def _case(session: Session, case_id: str, *, lock: bool = False) -> AccidentCaseORM:
    stmt = select(AccidentCaseORM).where(AccidentCaseORM.id == case_id)
    row = session.scalar(stmt.with_for_update() if lock else stmt)
    if row is None:
        raise HTTPException(status_code=404, detail="Accident case not found")
    return row


def _audit(session: Session, row: AccidentCaseORM, action: str, before: dict, after: dict, now: datetime) -> None:
    session.add(AccidentCaseAuditORM(
        id=str(uuid.uuid4()), case_id=row.id, revision=row.revision,
        action=action, before_json=before, after_json=after, created_at=now,
    ))


def build_office_cases_router(engine: Engine) -> APIRouter:
    # Admin migration owns production DDL. Do not call create_all here.
    def no_store(response: Response) -> None:
        response.headers["Cache-Control"] = "no-store"

    router = APIRouter(prefix="/clinical/office-cases", tags=["clinical-office-cases"],
                       dependencies=[Depends(_require_clinical_key), Depends(no_store)])

    @router.get("/accidents")
    def list_accidents(limit: int = Query(default=50, ge=1, le=100),
                       offset: int = Query(default=0, ge=0)) -> dict:
        with Session(engine) as session:
            rows = session.scalars(select(AccidentCaseORM).order_by(
                AccidentCaseORM.updated_at.desc(), AccidentCaseORM.id.desc(),
            ).offset(offset).limit(limit + 1)).all()
            return {"cases": [_case_json(session, row) for row in rows[:limit]],
                    "has_more": len(rows) > limit, "next_offset": offset + limit if len(rows) > limit else None}

    @router.post("/accidents")
    def create_accident(req: AccidentCreate) -> dict:
        now = utcnow()
        values = req.model_dump()
        values["patient_name"] = _text(values["patient_name"], "patient name")
        values["phone"] = _text(values["phone"], "phone")
        with Session(engine) as session:
            row = AccidentCaseORM(id=str(uuid.uuid4()), **values, revision=1,
                                  created_at=now, updated_at=now)
            session.add(row)
            session.flush()  # Parent first for the audit FK in production DDL.
            _audit(session, row, "create", {}, _snapshot(row), now)
            session.commit()
            return _case_json(session, row, detail=True)

    @router.get("/accidents/{case_id}")
    def get_accident(case_id: str) -> dict:
        with Session(engine) as session:
            return _case_json(session, _case(session, case_id), detail=True)

    @router.put("/accidents/{case_id}")
    def edit_accident(case_id: str, req: AccidentEdit) -> dict:
        with Session(engine) as session:
            row = _case(session, case_id, lock=True)
            if row.revision != req.expected_revision:
                raise HTTPException(status_code=409, detail="Accident case changed; refresh")
            before = _snapshot(row)
            for field in CASE_FIELDS:
                if field in req.model_fields_set:
                    value = getattr(req, field)
                    if field in {"patient_name", "phone"}:
                        value = _text(value, field) if value is not None else None
                    if field in {"patient_name", "phone", "accident_date"} and value is None:
                        raise HTTPException(status_code=422, detail=f"{field} is required")
                    setattr(row, field, value)
            received = sum(event.amount_cents for event in _payments(session, case_id)
                           if event.voided_at is None)
            if row.amount_due_cents is not None and received > row.amount_due_cents:
                raise HTTPException(status_code=409, detail="Amount due is below payments received")
            if _snapshot(row) != before:
                row.revision += 1
                row.updated_at = utcnow()
                _audit(session, row, "edit", before, _snapshot(row), row.updated_at)
                session.commit()
            return _case_json(session, row, detail=True)

    @router.post("/accidents/{case_id}/payments")
    def add_payment(case_id: str, req: PaymentCreate) -> dict:
        request_id = str(req.payment_request_id)
        payer = _text(req.payer, "payer")
        with Session(engine) as session:
            row = _case(session, case_id, lock=True)
            existing = session.scalar(select(AccidentPaymentORM).where(
                AccidentPaymentORM.case_id == case_id,
                AccidentPaymentORM.payment_request_id == request_id,
            ))
            if existing is not None:
                if (existing.amount_cents, existing.paid_on, existing.payer) != (
                    req.amount_cents, req.paid_on, payer,
                ):
                    raise HTTPException(status_code=409, detail="Payment request ID reused with different details")
                return _case_json(session, row, detail=True)
            received = sum(event.amount_cents for event in _payments(session, case_id)
                           if event.voided_at is None)
            if row.amount_due_cents is not None and received + req.amount_cents > row.amount_due_cents:
                raise HTTPException(status_code=409, detail="Payment exceeds amount due")
            now = utcnow()
            payment = AccidentPaymentORM(
                id=str(uuid.uuid4()), case_id=case_id, payment_request_id=request_id,
                amount_cents=req.amount_cents, paid_on=req.paid_on, payer=payer,
                created_at=now, voided_at=None, void_reason=None,
            )
            session.add(payment)
            row.revision += 1
            row.updated_at = now
            _audit(session, row, "payment_add", {}, {"payment_id": payment.id}, now)
            try:
                session.commit()
            except IntegrityError as exc:
                raise HTTPException(status_code=409, detail="Concurrent payment request; refresh") from exc
            return _case_json(session, row, detail=True)

    @router.post("/accidents/{case_id}/payments/{payment_id}/void")
    def void_payment(case_id: str, payment_id: str, req: PaymentVoid) -> dict:
        reason = _text(req.reason, "void reason")
        with Session(engine) as session:
            row = _case(session, case_id, lock=True)
            if row.revision != req.expected_revision:
                raise HTTPException(status_code=409, detail="Accident case changed; refresh")
            payment = session.scalar(select(AccidentPaymentORM).where(
                AccidentPaymentORM.case_id == case_id,
                AccidentPaymentORM.id == payment_id,
            ).with_for_update())
            if payment is None:
                raise HTTPException(status_code=404, detail="Payment not found")
            if payment.voided_at is not None:
                raise HTTPException(status_code=409, detail="Payment already voided")
            now = utcnow()
            payment.voided_at = now
            payment.void_reason = reason
            row.revision += 1
            row.updated_at = now
            _audit(session, row, "payment_void", {"payment_id": payment_id},
                   {"payment_id": payment_id, "reason": reason}, now)
            session.commit()
            return _case_json(session, row, detail=True)

    @router.get("/rf")
    def rf_cases(query: str = Query(default="", max_length=200),
                 stage: str = Query(default="", max_length=16),
                 cursor: str = Query(default="", max_length=2400)) -> dict:
        try:
            result = track_rf_applications(engine, query=query, stage=stage, cursor=cursor)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        ids = [card["id"] for card in result["applications"]]
        with Session(engine) as session:
            details = session.scalars(select(SurgeryDetailORM).where(
                SurgeryDetailORM.rf_application_id.in_(ids),
            )).all() if ids else []
            surgeries = session.scalars(select(SurgeryQueueORM).where(
                SurgeryQueueORM.id.in_([detail.surgery_id for detail in details]),
            )).all() if details else []
            by_surgery_id = {row.id: row for row in surgeries}
            linked = {}
            for detail in details:
                surgery = by_surgery_id.get(detail.surgery_id)
                if surgery is not None:
                    linked[detail.rf_application_id] = surgery_card(surgery, detail)
        for card in result["applications"]:
            card["linked_surgery"] = linked.get(card["id"])
            card["surgery_id"] = card["linked_surgery"]["surgery_id"] if card["linked_surgery"] else None
        return result

    def surgery_card(surgery: SurgeryQueueORM, detail: SurgeryDetailORM) -> dict:
        return {
            "surgery_id": surgery.id, "full_name": surgery.full_name,
            "procedure_type": surgery.procedure_type,
            "surgery_date": surgery.surgery_date,
            "room_entry_time": detail.room_entry_time,
            "status": surgery.status,
        }

    @router.get("/rf/{application_id}")
    def rf_case(application_id: str) -> dict:
        with Session(engine) as session:
            app = session.get(RFApplicationORM, application_id)
            if app is None:
                raise HTTPException(status_code=404, detail="RF application not found")
            progress = session.scalar(select(RFApplicationProgressORM).where(
                RFApplicationProgressORM.application_id == application_id,
            ))
            card = _rf_progress_card(app, progress)
            detail = session.scalar(select(SurgeryDetailORM).where(
                SurgeryDetailORM.rf_application_id == application_id,
            ))
            surgery = session.get(SurgeryQueueORM, detail.surgery_id) if detail else None
            card["linked_surgery"] = surgery_card(surgery, detail) if surgery and detail else None
            card["surgery_id"] = surgery.id if surgery else None
            card["surgery_prefill"] = (
                _rf_surgery_prefill(app)
                if card["stage"] in {"approved", "scheduled"} and surgery is None
                else None
            )
            return card

    @router.post("/surgeries/from-rf/{application_id}")
    def add_rf_to_surgery(application_id: str, req: RFToSurgery) -> dict:
        _valid_room_date(req.surgery_date, req.room_entry_time)
        phone = _text(req.phone, "phone")
        procedure = _text(req.procedure_type, "procedure")
        with Session(engine) as session:
            app = session.scalar(select(RFApplicationORM).where(
                RFApplicationORM.id == application_id,
            ).with_for_update())
            if app is None:
                raise HTTPException(status_code=404, detail="RF application not found")
            progress = session.scalar(select(RFApplicationProgressORM).where(
                RFApplicationProgressORM.application_id == application_id,
            ).with_for_update())
            card = _rf_progress_card(app, progress)
            if card["revision"] != req.expected_rf_revision or card["stage"] not in {"approved", "scheduled"}:
                raise HTTPException(status_code=409, detail="RF state changed or is not eligible; refresh")
            existing = session.scalar(select(SurgeryDetailORM).where(
                SurgeryDetailORM.rf_application_id == application_id,
            ))
            if existing is not None:
                raise HTTPException(status_code=409, detail="RF already linked to surgery")
            identity = _text(card["identity_number"], "RF identity")
            name = _text(card["patient_name"], "RF patient")
            if card["laterality"] not in {"left", "right"}:
                raise HTTPException(status_code=409, detail="RF laterality is not verified; refresh")
            pending = normalize_pending_positions(session)
            now = utcnow()
            surgery = SurgeryQueueORM(
                id=str(uuid.uuid4()), identity_number=identity, full_name=name,
                date_of_birth=req.date_of_birth, phone=phone,
                procedure_type=procedure, laterality=card["laterality"],
                surgery_date=req.surgery_date, queue_position=len(pending) + 1,
                status="pending", completed_at=None, created_at=now, updated_at=now,
            )
            detail = SurgeryDetailORM(
                surgery_id=surgery.id, rf_application_id=application_id,
                room_entry_time=req.room_entry_time, created_at=now, updated_at=now,
            )
            session.add(surgery)
            try:
                session.flush()  # Parent first: SQL FKs span separate ORM metadata.
                session.add(detail)
                session.commit()
            except IntegrityError as exc:
                raise HTTPException(status_code=409, detail="RF link changed; refresh") from exc
            return {"surgery_id": surgery.id, "rf_application_id": application_id}

    return router
