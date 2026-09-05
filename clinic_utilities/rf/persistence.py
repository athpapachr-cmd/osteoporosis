from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import Column, DateTime, Integer, JSON, String, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session

class RFBase(DeclarativeBase):
    pass

class RFApplicationORM(RFBase):
    __tablename__ = "clinic_rf_applications"
    id = Column(String, primary_key=True)
    patient_identity_key = Column(String, nullable=False, index=True)
    patient_name = Column(String, nullable=False)
    gesy_number = Column(String, nullable=False)
    patient_age = Column(Integer, nullable=False)
    pathway = Column(String, nullable=False, index=True)
    indication_code = Column(String, nullable=False, index=True)
    site_key = Column(String, nullable=False, index=True)
    laterality = Column(String, nullable=False, default="none", index=True)
    exact_location = Column(String, nullable=False)
    product_key = Column(String, nullable=False)
    payload_json = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, index=True)

class RFProcedureHistoryORM(RFBase):
    __tablename__ = "clinic_rf_procedure_history"
    id = Column(String, primary_key=True)
    patient_identity_key = Column(String, nullable=False, index=True)
    indication_code = Column(String, nullable=False, index=True)
    site_key = Column(String, nullable=False, index=True)
    laterality = Column(String, nullable=False, default="none", index=True)
    exact_location = Column(String, nullable=False)
    actual_procedure_date = Column(String, nullable=False, index=True)
    vas_before = Column(Integer, nullable=False)
    vas_after = Column(Integer, nullable=False)
    last_followup_date = Column(String, nullable=False)
    last_followup_vas = Column(Integer, nullable=False)
    provenance = Column(String, nullable=False, default="legacy_manual")
    dedupe_key = Column(String, nullable=False, unique=True, index=True)
    created_at = Column(DateTime, nullable=False, index=True)
    updated_at = Column(DateTime, nullable=False, index=True)

def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

def normalize_identity(value: str) -> str:
    normalized = "".join(re.findall(r"[0-9A-Za-z]+", str(value or "").upper()))
    if not normalized: raise ValueError("identity_number is required")
    return normalized

def initialize_rf_tables(engine: Engine) -> None:
    RFBase.metadata.create_all(bind=engine)

def _history_dedupe_key(data: dict[str, Any]) -> str:
    material="|".join(str(data.get(key) or "").strip().casefold() for key in ("patient_identity_key","site_key","laterality","actual_procedure_date","provenance"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()

def record_application(engine: Engine, data: dict[str, Any]) -> str:
    application_id=str(uuid4())
    row=RFApplicationORM(id=application_id,patient_identity_key=normalize_identity(data["identity_number"]),patient_name=str(data["patient_name"]).strip(),gesy_number=str(data["gesy_number"]).strip(),patient_age=int(data["age"]),pathway=str(data["pathway"]),indication_code=str(data["indication_code"]),site_key=str(data["site_key"]),laterality=str(data.get("laterality") or "none"),exact_location=str(data["exact_location"]).strip(),product_key=str(data["product_key"]),payload_json=dict(data),created_at=utcnow())
    with Session(engine) as session:
        session.add(row); session.commit()
    return application_id

def list_procedure_history(engine: Engine, identity_number: str, *, site_key: str="", laterality: str="") -> list[dict[str, Any]]:
    identity_key=normalize_identity(identity_number)
    with Session(engine) as session:
        stmt=select(RFProcedureHistoryORM).where(RFProcedureHistoryORM.patient_identity_key==identity_key)
        if site_key: stmt=stmt.where(RFProcedureHistoryORM.site_key==site_key)
        if laterality: stmt=stmt.where(RFProcedureHistoryORM.laterality==laterality)
        rows=session.execute(stmt.order_by(RFProcedureHistoryORM.actual_procedure_date.desc(),RFProcedureHistoryORM.created_at.desc())).scalars().all()
    return [{"procedure_history_id":r.id,"indication_code":r.indication_code,"site_key":r.site_key,"laterality":r.laterality,"exact_location":r.exact_location,"actual_procedure_date":r.actual_procedure_date,"vas_before":r.vas_before,"vas_after":r.vas_after,"last_followup_date":r.last_followup_date,"last_followup_vas":r.last_followup_vas,"provenance":r.provenance} for r in rows]

def get_procedure_history(engine: Engine, history_id: str, identity_number: str) -> dict[str, Any] | None:
    identity_key=normalize_identity(identity_number)
    with Session(engine) as session:
        row=session.get(RFProcedureHistoryORM,history_id)
        if row is None or row.patient_identity_key!=identity_key: return None
        return {"procedure_history_id":row.id,"indication_code":row.indication_code,"site_key":row.site_key,"laterality":row.laterality,"exact_location":row.exact_location,"actual_procedure_date":row.actual_procedure_date,"vas_before":row.vas_before,"vas_after":row.vas_after,"last_followup_date":row.last_followup_date,"last_followup_vas":row.last_followup_vas,"provenance":row.provenance}

def record_legacy_procedure(engine: Engine, data: dict[str, Any]) -> str:
    normalized={**data,"patient_identity_key":normalize_identity(data["identity_number"]),"provenance":"legacy_manual"}; dedupe_key=_history_dedupe_key(normalized)
    with Session(engine) as session:
        existing=session.scalar(select(RFProcedureHistoryORM).where(RFProcedureHistoryORM.dedupe_key==dedupe_key))
        if existing is not None: return existing.id
        now=utcnow(); row=RFProcedureHistoryORM(id=str(uuid4()),patient_identity_key=normalized["patient_identity_key"],indication_code=str(data["indication_code"]),site_key=str(data["site_key"]),laterality=str(data.get("laterality") or "none"),exact_location=str(data["exact_location"]).strip(),actual_procedure_date=str(data["actual_procedure_date"]),vas_before=int(data["vas_before"]),vas_after=int(data["vas_after"]),last_followup_date=str(data["last_followup_date"]),last_followup_vas=int(data["last_followup_vas"]),provenance="legacy_manual",dedupe_key=dedupe_key,created_at=now,updated_at=now)
        session.add(row); session.commit(); return row.id
