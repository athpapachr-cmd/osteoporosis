from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .parsers import (
    is_medication_identity_alias,
    normalize_medication_alias,
)
from sqlalchemy import Boolean, Column, DateTime, Integer, JSON, String, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session

from .catalog import DEFAULT_DOCTOR_KEY, DoctorProfile

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

# Managed physician directory within the existing protected RF database.
# No irreversible deletion: previously issued documents retain their own snapshot.
class RFSavedDraftORM(RFBase):
    """Incomplete RF fields only: no files, generated PDFs, or encounter ownership."""
    __tablename__ = "clinic_rf_saved_drafts"
    id = Column(String(80), primary_key=True)
    revision = Column(Integer, nullable=False, default=1)
    pathway = Column(String(2), nullable=False)
    patient_name = Column(String(200), nullable=False, default="")
    identity_key = Column(String(128), nullable=False, default="", index=True)
    gesy_number = Column(String(128), nullable=False, default="")
    form_state = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False, index=True)


class RFDoctorProfileORM(RFBase):
    __tablename__ = "clinic_rf_doctor_profiles"
    key = Column(String(40), primary_key=True)
    name = Column(String(200), nullable=False)
    gesy_code = Column(String(80), nullable=False, unique=True)
    specialty = Column(String(160), nullable=False)
    medical_center = Column(String(240), nullable=False)
    phone = Column(String(80), nullable=False)
    email = Column(String(240), nullable=False)
    active = Column(Boolean, nullable=False, default=True)
    revision = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


DOCTOR_FIELDS = ("name", "gesy_code", "specialty", "medical_center", "phone", "email")


def rf_doctor_snapshot(row: RFDoctorProfileORM) -> dict[str, str]:
    return {field: getattr(row, field) for field in DOCTOR_FIELDS}


def rf_doctor_data(row: RFDoctorProfileORM) -> dict[str, Any]:
    return {
        "key": row.key, **rf_doctor_snapshot(row),
        "active": bool(row.active), "revision": int(row.revision),
    }


def list_rf_doctors(engine: Engine) -> list[dict[str, Any]]:
    with Session(engine) as session:
        rows = session.scalars(
            select(RFDoctorProfileORM).order_by(RFDoctorProfileORM.created_at, RFDoctorProfileORM.key)
        ).all()
        return [rf_doctor_data(row) for row in rows]


def create_rf_doctor(engine: Engine, data: dict[str, str]) -> dict[str, Any]:
    now = utcnow()
    with Session(engine) as session:
        # Never recycle an identifier from a removed physician.
        row = RFDoctorProfileORM(key=uuid4().hex, **data, active=True, revision=1,
                                 created_at=now, updated_at=now)
        session.add(row)
        session.commit()
        return rf_doctor_data(row)


def revise_rf_doctor(engine: Engine, key: str, revision: int, *,
                     data: dict[str, str] | None = None,
                     active: bool | None = None) -> dict[str, Any]:
    with Session(engine) as session:
        row = session.scalar(select(RFDoctorProfileORM).where(
            RFDoctorProfileORM.key == key).with_for_update())
        if row is None:
            raise LookupError("doctor_not_found")
        if row.revision != revision:
            raise ValueError("doctor_revision_conflict")
        if data is not None:
            for field in DOCTOR_FIELDS:
                setattr(row, field, data[field])
        if active is not None:
            row.active = active
        row.revision += 1
        row.updated_at = utcnow()
        session.commit()
        return rf_doctor_data(row)


class RFMedicationAliasORM(RFBase):
    __tablename__ = "clinic_rf_medication_aliases"
    id = Column(String, primary_key=True)
    normalized_alias = Column(String, nullable=False, unique=True, index=True)
    alias = Column(String, nullable=False)
    display_name = Column(String, nullable=False)
    active_ingredient = Column(String, nullable=False, default="")
    category = Column(String, nullable=False, index=True)
    provenance = Column(String, nullable=False, default="clinician_confirmed")
    created_at = Column(DateTime, nullable=False, index=True)
    updated_at = Column(DateTime, nullable=False, index=True)

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
    provenance = Column(String, nullable=False, default="clinician_manual")
    dedupe_key = Column(String, nullable=False, unique=True, index=True)
    created_at = Column(DateTime, nullable=False, index=True)
    updated_at = Column(DateTime, nullable=False, index=True)


class RFApplicationProgressORM(RFBase):
    """One mutable lifecycle projection per immutable issued RF application."""
    __tablename__ = "clinic_rf_application_progress"
    application_id = Column(String, primary_key=True)
    revision = Column(Integer, nullable=False, default=1)
    stage = Column(String(16), nullable=False, default="issued")
    submitted_date = Column(String(10), nullable=True)
    decision = Column(String(16), nullable=True)
    decision_date = Column(String(10), nullable=True)
    rejection_reason = Column(String(2400), nullable=False, default="")
    scheduled_date = Column(String(10), nullable=True)
    performed_date = Column(String(10), nullable=True)
    mine_to_follow = Column(Boolean, nullable=False, default=False)
    updated_at = Column(DateTime, nullable=False)


class RFApplicationProgressAuditORM(RFBase):
    """Clinician-entered revisions, separate from the issued medical document."""
    __tablename__ = "clinic_rf_application_progress_audit"
    id = Column(String, primary_key=True)
    application_id = Column(String, nullable=False, index=True)
    revision = Column(Integer, nullable=False)
    action = Column(String(24), nullable=False)
    reported_source = Column(String(24), nullable=False)
    before_json = Column(JSON, nullable=False, default=dict)
    after_json = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False)


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

def normalize_identity(value: str) -> str:
    normalized = "".join(re.findall(r"[0-9A-Za-z]+", str(value or "").upper()))
    if not normalized: raise ValueError("identity_number is required")
    return normalized

def initialize_rf_tables(engine: Engine) -> None:
    RFBase.metadata.create_all(bind=engine)
    # Seed only once from the existing deployment setting; afterwards the managed
    # value wins, including edits and deliberate deactivation across restarts.
    with Session(engine) as session:
        if session.get(RFDoctorProfileORM, DEFAULT_DOCTOR_KEY) is None:
            try:
                default = DoctorProfile.from_environment()
            except ValueError:
                return  # No configured default: explicit later registration/edit required.
            now = utcnow()
            session.add(RFDoctorProfileORM(
                key=DEFAULT_DOCTOR_KEY,
                **{field: getattr(default, field) for field in DOCTOR_FIELDS},
                active=True, revision=1, created_at=now, updated_at=now))
            session.commit()

_ALIAS_DOSE_RE = re.compile(r"(?i)(?<!\w)\d+(?:[.,]\d+)?\s*(?:mg|g|gr|mcg|µg|χάπια?|χαπια?|δισκία|δισκια|tabs?|tablets?)(?!\w)")


def _validate_medication_alias_identity(alias_text: str) -> str:
    normalized = normalize_medication_alias(alias_text)
    if len(normalized) < 2:
        raise ValueError("Medication alias is too short")
    if not is_medication_identity_alias(alias_text):
        raise ValueError("Medication alias must identify a medicine, not regimen or formulation metadata")
    return normalized


def list_medication_aliases(engine: Engine) -> list[dict[str, Any]]:
    with Session(engine) as session:
        rows = session.execute(
            select(RFMedicationAliasORM).order_by(RFMedicationAliasORM.display_name.asc())
        ).scalars().all()
    return [
        {
            "id": row.id,
            "normalized_alias": row.normalized_alias,
            "alias": row.alias,
            "display_name": row.display_name,
            "active_ingredient": row.active_ingredient,
            "category": row.category,
            "provenance": row.provenance,
        }
        for row in rows
    ]


def upsert_medication_alias(
    engine: Engine,
    *,
    alias: str,
    category: str,
    display_name: str = "",
    active_ingredient: str = "",
) -> dict[str, Any]:
    alias_text = str(alias or "").strip()
    if category not in {"nsaid", "other"}:
        raise ValueError("Unsupported medication category")
    if not alias_text or len(alias_text) > 120:
        raise ValueError("Medication alias is required")
    if "\n" in alias_text or ";" in alias_text or _ALIAS_DOSE_RE.search(alias_text):
        raise ValueError("Medication alias must contain the medicine name only")
    normalized = _validate_medication_alias_identity(alias_text)

    display = str(display_name or alias_text).strip()[:160]
    active = str(active_ingredient or "").strip()[:160]
    now = utcnow()
    with Session(engine) as session:
        row = session.scalar(
            select(RFMedicationAliasORM).where(RFMedicationAliasORM.normalized_alias == normalized)
        )
        if row is None:
            row = RFMedicationAliasORM(
                id=str(uuid4()),
                normalized_alias=normalized,
                alias=alias_text,
                display_name=display,
                active_ingredient=active,
                category=category,
                provenance="clinician_confirmed",
                created_at=now,
                updated_at=now,
            )
            session.add(row)
        else:
            row.alias = alias_text
            row.display_name = display
            row.active_ingredient = active
            row.category = category
            row.updated_at = now
        session.commit()
        session.refresh(row)
        return {
            "id": row.id,
            "normalized_alias": row.normalized_alias,
            "alias": row.alias,
            "display_name": row.display_name,
            "active_ingredient": row.active_ingredient,
            "category": row.category,
            "provenance": row.provenance,
        }


def delete_medication_alias(engine: Engine, alias_id: str) -> bool:
    with Session(engine) as session:
        row = session.get(RFMedicationAliasORM, alias_id)
        if row is None:
            return False
        session.delete(row)
        session.commit()
        return True


def _history_dedupe_key(data: dict[str, Any]) -> str:
    material="|".join(str(data.get(key) or "").strip().casefold() for key in ("patient_identity_key","site_key","laterality","actual_procedure_date","provenance"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()

def _minimal_application_payload(data: dict[str, Any]) -> dict[str, Any]:
    persisted = dict(data)
    persisted.pop("full_medication_text", None)
    persisted.pop("physio_dates_text", None)
    for key in ("nsaid_trials", "other_analgesic_trials"):
        trials = persisted.get(key)
        if isinstance(trials, list):
            persisted[key] = [
                {item_key: item_value for item_key, item_value in item.items() if item_key != "source_text"}
                if isinstance(item, dict) else item
                for item in trials
            ]
    return persisted

def record_application(engine: Engine, data: dict[str, Any], *, session: Session | None = None) -> str:
    application_id=str(uuid4())
    row=RFApplicationORM(id=application_id,patient_identity_key=normalize_identity(data["identity_number"]),patient_name=str(data["patient_name"]).strip(),gesy_number=str(data["gesy_number"]).strip(),patient_age=int(data["age"]),pathway=str(data["pathway"]),indication_code=str(data["indication_code"]),site_key=str(data["site_key"]),laterality=str(data.get("laterality") or "none"),exact_location=str(data["exact_location"]).strip(),product_key=str(data["product_key"]),payload_json=_minimal_application_payload(data),created_at=utcnow())
    if session is not None:
        session.add(row)
    else:
        with Session(engine) as own_session:
            own_session.add(row)
            own_session.commit()
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

def record_legacy_procedure(engine: Engine, data: dict[str, Any], *, session: Session | None = None) -> str:
    normalized={**data,"patient_identity_key":normalize_identity(data["identity_number"]),"provenance":"clinician_manual"}; dedupe_key=_history_dedupe_key(normalized)
    def save(target: Session) -> str:
        existing=target.scalar(select(RFProcedureHistoryORM).where(RFProcedureHistoryORM.dedupe_key==dedupe_key))
        if existing is not None: return existing.id
        now=utcnow(); row=RFProcedureHistoryORM(id=str(uuid4()),patient_identity_key=normalized["patient_identity_key"],indication_code=str(data["indication_code"]),site_key=str(data["site_key"]),laterality=str(data.get("laterality") or "none"),exact_location=str(data["exact_location"]).strip(),actual_procedure_date=str(data["actual_procedure_date"]),vas_before=int(data["vas_before"]),vas_after=int(data["vas_after"]),last_followup_date=str(data["last_followup_date"]),last_followup_vas=int(data["last_followup_vas"]),provenance="clinician_manual",dedupe_key=dedupe_key,created_at=now,updated_at=now)
        target.add(row)
        return row.id
    if session is not None:
        return save(session)
    with Session(engine) as own_session:
        result = save(own_session)
        own_session.commit()
        return result


# No new identity authority. These procedures and applications share the existing RF DB.
# Search values are normalized in Python so Greek accents/ς and identifier separators work
# consistently on the SQLite test engine and production PostgreSQL.
def _rf_search_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value or "").casefold().replace("ς", "σ"))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return " ".join(value.split())


def _rf_id_text(value: str) -> str:
    return "".join(ch for ch in str(value or "").casefold() if ch.isalnum())


# A named, bounded scan avoids a fragile DB-specific case/accent collation.
# The RF application table is clinic-local; truncation is reported, never silently denied.
_RF_SEARCH_SCAN_LIMIT = 10000
_RF_PAGE_SIZE = 20


def _rf_application_card(row: RFApplicationORM) -> dict[str, Any]:
    data = row.payload_json if isinstance(row.payload_json, dict) else {}
    return {
        "kind": "application", "id": row.id, "date": row.created_at.date().isoformat(),
        "patient_name": row.patient_name,
        "identity_number": str(data.get("identity_number") or row.patient_identity_key),
        "gesy_number": row.gesy_number, "indication_code": row.indication_code,
        "site_key": row.site_key, "laterality": row.laterality,
        "exact_location": row.exact_location, "product_key": row.product_key,
        "actual_procedure": False, "pathway": row.pathway,
    }


def _rf_procedure_card(row: RFProcedureHistoryORM, patient: dict[str, str] | None) -> dict[str, Any]:
    patient = patient or {}
    return {
        "kind": "procedure", "id": row.id, "date": row.actual_procedure_date,
        "patient_name": patient.get("patient_name", ""),
        "identity_number": patient.get("identity_number", row.patient_identity_key),
        "gesy_number": patient.get("gesy_number", ""),
        "indication_code": row.indication_code, "site_key": row.site_key,
        "laterality": row.laterality, "exact_location": row.exact_location,
        "product_key": "", "actual_procedure": True, "pathway": "",
    }


def search_rf_sources(engine: Engine, query: str, page: int = 1) -> dict[str, Any]:
    q = _rf_search_text(query)
    q_id = _rf_id_text(query)
    if len(q) < 2 or page < 1 or page > 100:
        raise ValueError("Συμπλήρωσε τουλάχιστον δύο χαρακτήρες αναζήτησης")
    with Session(engine) as session:
        applications = session.execute(select(RFApplicationORM).order_by(
            RFApplicationORM.created_at.desc(), RFApplicationORM.id.desc()
        ).limit(_RF_SEARCH_SCAN_LIMIT + 1)).scalars().all()
        truncated = len(applications) > _RF_SEARCH_SCAN_LIMIT
        applications = applications[:_RF_SEARCH_SCAN_LIMIT]
        matched = []
        identities: dict[str, dict[str, str]] = {}
        # A procedure with no application has identity only, never an invented name/ΓεΣΥ.
        for a in applications:
            ident = str((a.payload_json or {}).get("identity_number") or a.patient_identity_key)
            norm_id = _rf_id_text(ident)
            if (
                q in _rf_search_text(a.patient_name)
                or q_id in norm_id
                or q_id in _rf_id_text(a.gesy_number)
            ):
                matched.append(_rf_application_card(a))
                identities[a.patient_identity_key] = {
                    "patient_name": a.patient_name, "identity_number": ident,
                    "gesy_number": a.gesy_number,
                }
        # Normalized identity allows a procedure-only match, but cannot prove a
        # person's name or GESY when no application has ever been issued.
        norm_identity = _rf_id_text(query)
        id_rows = session.execute(select(RFProcedureHistoryORM).order_by(
            RFProcedureHistoryORM.actual_procedure_date.desc(),
            RFProcedureHistoryORM.created_at.desc()
        ).limit(_RF_SEARCH_SCAN_LIMIT + 1)).scalars().all()
        truncated = truncated or len(id_rows) > _RF_SEARCH_SCAN_LIMIT
        for h in id_rows[:_RF_SEARCH_SCAN_LIMIT]:
            if h.patient_identity_key in identities or (
                len(norm_identity) >= 3 and norm_identity in _rf_id_text(h.patient_identity_key)
            ):
                matched.append(_rf_procedure_card(h, identities.get(h.patient_identity_key)))
    matched.sort(key=lambda item: (item["date"], item["kind"], item["id"]), reverse=True)
    start = (page - 1) * _RF_PAGE_SIZE
    return {"results": matched[start:start + _RF_PAGE_SIZE], "page": page,
            "has_more": len(matched) > start + _RF_PAGE_SIZE, "truncated": truncated}


def get_rf_source(engine: Engine, kind: str, source_id: str) -> dict[str, Any] | None:
    with Session(engine) as session:
        if kind == "application":
            row = session.get(RFApplicationORM, source_id)
            if row is None:
                return None
            data = row.payload_json if isinstance(row.payload_json, dict) else {}
            allowed = {
                "patient_name", "identity_number", "gesy_number", "age",
                "product_key", "indication_code", "laterality", "exact_location",
                "other_area", "other_diagnosis", "additional_notes",
                "rf_reason_codes", "rf_reason_other", "pain_onset_date",
                "pain_onset_vas", "last_assessment_date", "last_assessment_vas",
                "nsaid_trials", "other_analgesic_trials", "adverse_effects",
                "intervention", "physio",
            }
            fields = {k: v for k, v in data.items() if k in allowed}
            fields.update({"patient_name": row.patient_name,
                           "identity_number": str(data.get("identity_number") or row.patient_identity_key),
                           "gesy_number": row.gesy_number,
                           "age": row.patient_age, "product_key": row.product_key,
                           "indication_code": row.indication_code, "laterality": row.laterality,
                           "exact_location": row.exact_location})
            return {"source": _rf_application_card(row), "fields": fields}
        if kind == "procedure":
            row = session.get(RFProcedureHistoryORM, source_id)
            if row is None:
                return None
            # Name/ΓεΣΥ can be resolved from an identity-linked application,
            # but product requires the *exact* RF indication as well as site/side.
            patient_app = session.scalar(select(RFApplicationORM).where(
                RFApplicationORM.patient_identity_key == row.patient_identity_key
            ).order_by(RFApplicationORM.created_at.desc(), RFApplicationORM.id.desc()).limit(1))
            product_app = session.scalar(select(RFApplicationORM).where(
                RFApplicationORM.patient_identity_key == row.patient_identity_key,
                RFApplicationORM.site_key == row.site_key,
                RFApplicationORM.laterality == row.laterality,
                RFApplicationORM.indication_code == row.indication_code,
            ).order_by(RFApplicationORM.created_at.desc(), RFApplicationORM.id.desc()).limit(1))
            parent_data = patient_app.payload_json if patient_app and isinstance(patient_app.payload_json, dict) else {}
            person = {"patient_name": patient_app.patient_name, "gesy_number": patient_app.gesy_number,
                      "identity_number": str(parent_data.get("identity_number") or row.patient_identity_key)} if patient_app else None
            card = _rf_procedure_card(row, person)
            return {"source": card,
                    "fields": {"patient_name": card["patient_name"],
                               "identity_number": card["identity_number"],
                               "gesy_number": card["gesy_number"],
                               "indication_code": row.indication_code,
                               "laterality": row.laterality,
                               "exact_location": row.exact_location,
                               "product_key": product_app.product_key if product_app else ""},
                    "actual_procedure": get_procedure_history(engine, row.id, row.patient_identity_key)}
    return None


_RF_DRAFT_FIELDS = frozenset({
    "pathway", "doctor_key", "patient_name", "identity_number", "gesy_number",
    "age", "product_key", "indication_code", "laterality", "exact_location",
    "other_area", "other_diagnosis", "additional_notes",
    "rf_reason_codes", "rf_reason_other", "pain_onset_date", "pain_onset_vas",
    "last_assessment_date", "last_assessment_vas", "full_medication_text",
    "nsaid_trials", "other_analgesic_trials", "adverse_effects",
    "intervention", "physio_dates_text", "legacy_history",
    "procedure_history_id", "source_kind", "source_id", "source_origin",
    "a2_current_followup_date", "a2_current_followup_vas", "legacy_confirmed",
    "draft_nsaids", "draft_others", "source_patient_name",
})
_RF_DRAFT_ARRAY_FIELDS = {
    "rf_reason_codes": 8, "nsaid_trials": 20, "other_analgesic_trials": 20,
    "adverse_effects": 10, "draft_nsaids": 20, "draft_others": 20,
}
_RF_DRAFT_NESTED_FIELDS = {
    "nsaid_trials": {"drug_name", "dose", "duration", "auto_selected", "source_text", "manual", "learned", "warning", "canonical_key"},
    "other_analgesic_trials": {"drug_name", "dose", "duration", "auto_selected", "source_text", "manual", "learned", "warning", "canonical_key"},
    "adverse_effects": {"treatment", "effect"},
    "draft_nsaids": {"drug_name", "dose", "duration", "auto_selected", "source_text", "manual", "learned", "warning", "canonical_key"},
    "draft_others": {"drug_name", "dose", "duration", "auto_selected", "source_text", "manual", "learned", "warning", "canonical_key"},
    "intervention": {"site", "date", "vas_before", "vas_after"},
    "legacy_history": {"actual_procedure_date", "vas_before", "vas_after", "last_followup_date", "last_followup_vas"},
}
_RF_DRAFT_MAX_BYTES = 85000


def validate_rf_draft_fields(state: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(state, dict) or set(state) - _RF_DRAFT_FIELDS:
        raise ValueError("Μη αναγνωρισμένα πεδία πρόχειρης RF αίτησης")
    if state.get("pathway", "A1") not in {"A1", "A2"}:
        raise ValueError("Μη έγκυρος τύπος πρόχειρης αίτησης")
    # No arbitrary nested content or binary attachments: only RF form fields.
    for key, value in state.items():
        if key in _RF_DRAFT_ARRAY_FIELDS:
            if not isinstance(value, list) or len(value) > _RF_DRAFT_ARRAY_FIELDS[key]:
                raise ValueError("Υπέρβαση πεδίων πρόχειρης αίτησης")
            if key == "rf_reason_codes":
                if any(not isinstance(item, str) or len(item) > 100 for item in value):
                    raise ValueError("Μη έγκυροι λόγοι RF")
            else:
                if any(not isinstance(item, dict) or set(item) - _RF_DRAFT_NESTED_FIELDS[key]
                       for item in value):
                    raise ValueError("Μη έγκυρες γραμμές πρόχειρης αίτησης")
        elif key in {"intervention", "legacy_history"}:
            if value is not None and (
                not isinstance(value, dict) or set(value) - _RF_DRAFT_NESTED_FIELDS[key]
            ):
                raise ValueError("Μη έγκυρα στοιχεία προηγούμενης θεραπείας")
        elif not isinstance(value, (str, int, float, bool, type(None))):
            raise ValueError("Μη έγκυρη μορφή πρόχειρης αίτησης")
        if isinstance(value, str) and len(value) > (30000 if key == "full_medication_text" else 12000 if key == "physio_dates_text" else 4000):
            raise ValueError("Υπερβολικά μεγάλο πεδίο πρόχειρης αίτησης")
    encoded = json.dumps(state, ensure_ascii=False, allow_nan=False)
    if len(encoded.encode("utf-8")) > _RF_DRAFT_MAX_BYTES:
        raise ValueError("Το πρόχειρο RF υπερβαίνει το επιτρεπόμενο μέγεθος")
    return json.loads(encoded)


def _rf_draft_card(row: RFSavedDraftORM) -> dict[str, Any]:
    return {"id": row.id, "revision": row.revision, "pathway": row.pathway,
            "patient_name": row.patient_name,
            "identity_number": row.form_state.get("identity_number", ""),
            "gesy_number": row.gesy_number,
            "updated_at": row.updated_at.isoformat() + "Z"}


def save_rf_draft(engine: Engine, form_state: dict[str, Any],
                  draft_id: str = "", revision: int | None = None) -> dict[str, Any]:
    state = validate_rf_draft_fields(form_state)
    now = utcnow()
    with Session(engine) as session:
        if draft_id:
            row = session.scalar(select(RFSavedDraftORM).where(
                RFSavedDraftORM.id == draft_id).with_for_update())
            if row is None:
                raise LookupError("Το πρόχειρο δεν βρέθηκε")
            if revision is None or row.revision != revision:
                raise RuntimeError("Το πρόχειρο άλλαξε σε άλλη καρτέλα. Φόρτωσέ το ξανά")
            row.revision += 1
        else:
            row = RFSavedDraftORM(id=str(uuid4()), revision=1, created_at=now)
            session.add(row)
        row.form_state = state
        row.pathway = state.get("pathway", "A1")
        row.patient_name = str(state.get("patient_name") or "").strip()[:200]
        row.identity_key = _rf_id_text(state.get("identity_number") or "")[:128]
        row.gesy_number = str(state.get("gesy_number") or "").strip()[:128]
        row.updated_at = now
        session.commit()
        return _rf_draft_card(row)


def list_rf_drafts(engine: Engine, query: str = "", page: int = 1) -> dict[str, Any]:
    if page < 1 or page > 100:
        raise ValueError("Μη έγκυρη σελίδα πρόχειρων")
    q = _rf_search_text(query)
    q_id = _rf_id_text(query)
    with Session(engine) as session:
        rows = session.execute(select(RFSavedDraftORM).order_by(
            RFSavedDraftORM.updated_at.desc(), RFSavedDraftORM.id.desc()
        ).limit(_RF_SEARCH_SCAN_LIMIT + 1)).scalars().all()
        truncated = len(rows) > _RF_SEARCH_SCAN_LIMIT
        rows = rows[:_RF_SEARCH_SCAN_LIMIT]
        if q:
            rows = [row for row in rows if q in _rf_search_text(row.patient_name)
                    or q_id in row.identity_key or q_id in _rf_id_text(row.gesy_number)]
        start = (page - 1) * _RF_PAGE_SIZE
        return {"drafts": [_rf_draft_card(row) for row in rows[start:start + _RF_PAGE_SIZE]],
                "page": page, "has_more": len(rows) > start + _RF_PAGE_SIZE,
                "truncated": truncated}


def get_rf_draft(engine: Engine, draft_id: str) -> dict[str, Any] | None:
    with Session(engine) as session:
        row = session.get(RFSavedDraftORM, draft_id)
        if row is None:
            return None
        return {**_rf_draft_card(row), "fields": dict(row.form_state),
                "attachments_saved": False}


def delete_rf_draft(engine: Engine, draft_id: str, revision: int) -> None:
    with Session(engine) as session:
        row = session.scalar(select(RFSavedDraftORM).where(
            RFSavedDraftORM.id == draft_id).with_for_update())
        if row is None:
            raise LookupError("Το πρόχειρο δεν βρέθηκε")
        if row.revision != revision:
            raise RuntimeError("Το πρόχειρο άλλαξε. Φόρτωσέ το ξανά")
        session.delete(row)
        session.commit()


# Protected RF outcomes: issued document content remains immutable. A missing
# progress row is NOT "submitted" or "pending": historical state is unknown.
_RF_PROGRESS_SOURCE = frozenset({"clinician", "colleague", "gesy_notice", "other"})
_RF_PROGRESS_ACTIONS = frozenset({
    "submit", "approve", "reject", "schedule", "perform",
    "claim", "release", "correct_decision",
})


def _rf_progress_values(row: RFApplicationProgressORM | None) -> dict[str, Any]:
    if row is None:
        return {
            "revision": 0, "stage": "issued", "submitted_date": None,
            "decision": None, "decision_date": None, "rejection_reason": "",
            "scheduled_date": None, "performed_date": None, "mine_to_follow": False,
        }
    return {
        "revision": row.revision, "stage": row.stage,
        "submitted_date": row.submitted_date, "decision": row.decision,
        "decision_date": row.decision_date, "rejection_reason": row.rejection_reason,
        "scheduled_date": row.scheduled_date, "performed_date": row.performed_date,
        "mine_to_follow": bool(row.mine_to_follow),
    }


def _rf_progress_card(app: RFApplicationORM, progress: RFApplicationProgressORM | None) -> dict[str, Any]:
    payload = app.payload_json if isinstance(app.payload_json, dict) else {}
    snapshot = payload.get("doctor_snapshot")
    return {
        "id": app.id,
        "issued_date": app.created_at.date().isoformat(),
        "pathway": app.pathway,
        "patient_name": app.patient_name,
        "identity_number": str(payload.get("identity_number") or app.patient_identity_key),
        "gesy_number": app.gesy_number,
        "indication_code": app.indication_code,
        "laterality": app.laterality,
        "site_key": app.site_key,
        "doctor_key": str(payload.get("doctor_key") or ""),
        "issuer_name": str(snapshot.get("name") or "") if isinstance(snapshot, dict) else "",
        "issuer_self_verified_at_issue": payload.get("issuer_self_verified_at_issue") is True
            and payload.get("issuer_selection_origin") == "explicit",
        "correction_of_application_id": str(payload.get("correction_of_application_id") or ""),
        **_rf_progress_values(progress),
    }


def get_rf_application_progress(engine: Engine, application_id: str) -> dict[str, Any] | None:
    with Session(engine) as session:
        app = session.get(RFApplicationORM, application_id)
        if app is None:
            return None
        return _rf_progress_card(app, session.get(RFApplicationProgressORM, application_id))


def list_rf_application_progress(engine: Engine, query: str = "", scope: str = "all",
                                 page: int = 1) -> dict[str, Any]:
    if scope not in {"all", "mine"} or page < 1 or page > 100:
        raise ValueError("Μη έγκυρο φίλτρο RF")
    text = _rf_search_text(query)
    id_text = _rf_id_text(query)
    with Session(engine) as session:
        apps = session.scalars(select(RFApplicationORM).order_by(
            RFApplicationORM.created_at.desc(), RFApplicationORM.id.desc()
        ).limit(_RF_SEARCH_SCAN_LIMIT + 1)).all()
        truncated = len(apps) > _RF_SEARCH_SCAN_LIMIT
        apps = apps[:_RF_SEARCH_SCAN_LIMIT]
        ids = [app.id for app in apps]
        rows = (session.scalars(select(RFApplicationProgressORM).where(
            RFApplicationProgressORM.application_id.in_(ids))).all() if ids else [])
        by_id = {row.application_id: row for row in rows}
        matched = []
        for app in apps:
            if text and not (
                text in _rf_search_text(app.patient_name)
                or id_text in _rf_id_text(app.patient_identity_key)
                or id_text in _rf_id_text(app.gesy_number)
            ):
                continue
            card = _rf_progress_card(app, by_id.get(app.id))
            if scope == "mine" and not card["mine_to_follow"]:
                continue
            matched.append(card)
    # Counts describe the filtered *attempts*, not all patient histories.
    counts = {
        "issued_unknown": sum(c["stage"] == "issued" for c in matched),
        "pending": sum(c["stage"] == "submitted" for c in matched),
        "approved": sum(c["decision"] == "approved" for c in matched),
        "rejected": sum(c["decision"] == "rejected" for c in matched),
        "scheduled": sum(c["stage"] == "scheduled" for c in matched),
        "performed": sum(c["stage"] == "performed" for c in matched),
        "mine_actionable": sum(c["mine_to_follow"] and c["stage"] in {"submitted", "approved", "scheduled"}
                               for c in matched),
    }
    start = (page - 1) * _RF_PAGE_SIZE
    return {
        "applications": matched[start:start + _RF_PAGE_SIZE],
        "counts": counts, "denominator": len(matched),
        "scope": scope, "page": page, "has_more": len(matched) > start + _RF_PAGE_SIZE,
        "truncated": truncated,
    }


def _rf_event_date(value: str, label: str, *, may_be_future: bool = False) -> str:
    from datetime import date
    if not value or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"Απαιτείται ακριβής ημερομηνία {label}")
    try:
        when = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Μη έγκυρη ημερομηνία {label}") from exc
    if not may_be_future and when > date.today():
        raise ValueError(f"Η ημερομηνία {label} δεν μπορεί να είναι μελλοντική")
    return value


def change_rf_application_progress(engine: Engine, application_id: str, *,
                                   action: str, revision: int, event_date: str = "",
                                   rejection_reason: str = "", reported_source: str = "clinician",
                                   confirmed_performed: bool = False) -> dict[str, Any]:
    if action not in _RF_PROGRESS_ACTIONS or reported_source not in _RF_PROGRESS_SOURCE:
        raise ValueError("Μη έγκυρη ενέργεια ή προέλευση RF")
    if revision < 0 or len(rejection_reason) > 2400:
        raise ValueError("Μη έγκυρη έκδοση ή αιτιολογία RF")
    if action in {"claim", "release"}:
        if event_date or rejection_reason:
            raise ValueError("Η ανάληψη παρακολούθησης δεν αλλάζει έκβαση")
    else:
        _rf_event_date(event_date, "RF", may_be_future=action == "schedule")
        if action != "reject" and action != "correct_decision" and rejection_reason:
            raise ValueError("Η αιτιολογία απόρριψης δεν αφορά αυτή την ενέργεια")
    with Session(engine) as session:
        app = session.get(RFApplicationORM, application_id)
        if app is None:
            raise LookupError("Η αίτηση RF δεν βρέθηκε")
        progress = session.scalar(select(RFApplicationProgressORM).where(
            RFApplicationProgressORM.application_id == application_id).with_for_update())
        previous = _rf_progress_values(progress)
        if previous["revision"] != revision:
            raise RuntimeError("Η έκβαση άλλαξε σε άλλη καρτέλα. Ανανέωσε την αίτηση")
        stage = previous["stage"]
        if action == "submit":
            if stage != "issued":
                raise ValueError("Η αίτηση έχει ήδη καταχωρισμένη πορεία")
            progress = progress or RFApplicationProgressORM(application_id=application_id)
            progress.submitted_date = event_date
            progress.stage = "submitted"
            payload = app.payload_json if isinstance(app.payload_json, dict) else {}
            progress.mine_to_follow = (
                payload.get("issuer_selection_origin") == "explicit"
                and payload.get("issuer_self_verified_at_issue") is True
            )
        elif action in {"approve", "reject"}:
            if stage not in {"issued", "submitted"}:
                raise ValueError("Δεν επιτρέπεται νέα απόφαση σε ολοκληρωμένη έκβαση")
            if previous["submitted_date"] and event_date < previous["submitted_date"]:
                raise ValueError("Η απόφαση προηγείται της υποβολής")
            progress = progress or RFApplicationProgressORM(application_id=application_id)
            progress.decision = "approved" if action == "approve" else "rejected"
            progress.decision_date = event_date
            progress.rejection_reason = rejection_reason.strip() if action == "reject" else ""
            progress.stage = progress.decision
        elif action == "correct_decision":
            if stage not in {"approved", "rejected"} or not previous["decision"]:
                raise ValueError("Δεν υπάρχει απόφαση προς διόρθωση")
            if previous["submitted_date"] and event_date < previous["submitted_date"]:
                raise ValueError("Η απόφαση προηγείται της υποβολής")
            if previous["decision"] == "approved" and rejection_reason:
                raise ValueError("Δεν αποθηκεύεται αιτιολογία απόρριψης σε έγκριση")
            progress.decision_date = event_date
            progress.rejection_reason = rejection_reason.strip() if stage == "rejected" else ""
        elif action == "schedule":
            if stage not in {"approved", "scheduled"}:
                raise ValueError("Προγραμματισμός μόνο μετά από καταχωρισμένη έγκριση")
            if event_date < previous["decision_date"]:
                raise ValueError("Ο προγραμματισμός προηγείται της απόφασης")
            progress.scheduled_date = event_date
            progress.stage = "scheduled"
        elif action == "perform":
            if stage not in {"approved", "scheduled"} or not confirmed_performed:
                raise ValueError("Απαιτείται ρητή επιβεβαίωση πραγματικής εφαρμογής μετά από έγκριση")
            if event_date < previous["decision_date"]:
                raise ValueError("Η πραγματική εφαρμογή προηγείται της έγκρισης")
            if previous["scheduled_date"] and event_date < previous["scheduled_date"]:
                raise ValueError("Η πραγματική εφαρμογή προηγείται της προγραμματισμένης ημερομηνίας")
            progress.performed_date = event_date
            progress.stage = "performed"
        elif action in {"claim", "release"}:
            if action == "claim" and previous["mine_to_follow"]:
                raise ValueError("Η παρακολούθηση έχει ήδη αναληφθεί")
            if action == "release" and not previous["mine_to_follow"]:
                raise ValueError("Δεν υπάρχει προσωπική παρακολούθηση")
            progress = progress or RFApplicationProgressORM(application_id=application_id)
            progress.mine_to_follow = action == "claim"
        else:
            raise ValueError("Μη έγκυρη ενέργεια RF")
        # A single transaction commits the versioned projection and audit.
        now = utcnow()
        progress.revision = revision + 1
        progress.updated_at = now
        session.add(progress)
        after = _rf_progress_values(progress)
        session.add(RFApplicationProgressAuditORM(
            id=str(uuid4()), application_id=application_id, revision=revision + 1,
            action=action, reported_source=reported_source,
            before_json=previous, after_json=after, created_at=now,
        ))
        session.commit()
        return _rf_progress_card(app, progress)


def validate_rf_correction_source(session: Session, *, rejected_id: str,
                                  identity_number: str, indication_code: str,
                                  laterality: str, site_key: str) -> None:
    prior = session.get(RFApplicationORM, rejected_id)
    if prior is None:
        raise ValueError("Η απορριφθείσα αίτηση RF δεν βρέθηκε")
    progress = session.get(RFApplicationProgressORM, rejected_id)
    if progress is None or progress.decision != "rejected":
        raise ValueError("Απαιτείται καταχωρισμένη απόρριψη για διορθωτική αίτηση")
    if (prior.patient_identity_key != normalize_identity(identity_number)
        or prior.indication_code != indication_code or prior.laterality != laterality
        or prior.site_key != site_key):
        raise ValueError("Η διορθωτική RF αφορά διαφορετικό ασθενή, ένδειξη ή πλευρά")
