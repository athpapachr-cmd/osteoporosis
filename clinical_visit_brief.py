"""Booking-scoped, protected Visit Brief. No provider writes or inferred person identity.

Reception remains the calendar source; Clinical Data owns patient/encounter/pending facts.
Current Reception payload has no authoritative attendee continuity assertion, so every
clinical read requires an independent explicit confirmation in that response only.
"""
from __future__ import annotations

import json
import hashlib
import hmac
import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import Literal, Optional
from uuid import uuid4
from zoneinfo import ZoneInfo

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, ConfigDict, UUID4, field_validator
from sqlalchemy import Column, DateTime, Integer, String, select, update, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy.exc import IntegrityError

from clinical_data import (
    ClinicalPendingORM, EncounterORM, PatientORM, VisitCaptureCandidateV1,
    SYNTHETIC_161_PATIENT_IDS,
    _fold_patient_lookup, _patient_matches_search,
)
from clinical_staff_auth import (
    StaffActor, require_staff_actor, require_staff_clinician,
)
from clinical_calendar import ActualSchedule


NAMESPACE = "reception-actual"
CYPRUS = ZoneInfo("Asia/Nicosia")
OPEN_PENDING = {"open", "awaiting_patient", "awaiting_result"}
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class VisitBriefBase(DeclarativeBase):
    pass


class BookingPatientLinkORM(VisitBriefBase):
    __tablename__ = "clinical_booking_patient_links"

    namespace = Column(String(48), primary_key=True)
    booking_uid = Column(String(160), primary_key=True)
    patient_id = Column(String(120), nullable=False)
    state = Column(String(32), nullable=False, default="needs_reconfirmation")
    revision = Column(Integer, nullable=False, default=1)
    evidence_type = Column(String(64), nullable=False)
    confirmed_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class BookingPatientLinkAuditORM(VisitBriefBase):
    __tablename__ = "clinical_booking_patient_link_audit"

    id = Column(String(36), primary_key=True)
    namespace = Column(String(48), nullable=False, index=True)
    booking_uid = Column(String(160), nullable=False, index=True)
    patient_id = Column(String(120), nullable=True)
    event = Column(String(24), nullable=False)
    revision = Column(Integer, nullable=False)
    evidence_type = Column(String(64), nullable=True)
    operator_type = Column(String(32), nullable=False, default="shared_clinical_key")
    occurred_at = Column(DateTime, nullable=False)


class VisitBriefStaffAuditORM(VisitBriefBase):
    """Append-only per-principal access history, distinct from link audit."""
    __tablename__ = "clinical_visit_brief_staff_access_audit"
    id = Column(String(36), primary_key=True)
    principal_id = Column(String(64), nullable=False)
    role = Column(String(32), nullable=False)
    booking_uid = Column(String(160), nullable=False)
    patient_id = Column(String(120), nullable=True)
    event = Column(String(40), nullable=False)
    outcome = Column(String(32), nullable=False)
    revision = Column(Integer, nullable=True)
    occurred_at = Column(DateTime, nullable=False)


def _record_access(session: Session, actor: StaffActor, uid: str, event: str,
                   outcome: str, patient_id: Optional[str] = None,
                   revision: Optional[int] = None) -> None:
    session.add(VisitBriefStaffAuditORM(
        id=str(uuid4()), principal_id=actor.principal_id, role=actor.role,
        booking_uid=uid, patient_id=patient_id, event=event, outcome=outcome,
        revision=revision, occurred_at=_utcnow().replace(tzinfo=None),
    ))


def _record_denial(engine: Engine, actor: StaffActor, uid: str,
                   event: str, status_code: int) -> None:
    """Persist an attributable refusal outside the rolled-back clinical transaction."""
    with Session(engine) as audit:
        _record_access(audit, actor, uid, event, f"denied_{status_code}")
        audit.commit()


class BookingIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    booking_uid: str = Field(min_length=1, max_length=160)


class BookingPatientSearch(BookingIntent):
    observed_booking: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    query: str = Field(min_length=2, max_length=120)
    limit: int = Field(default=20, ge=1, le=20)
    offset: int = Field(default=0, ge=0)


class ConfirmBooking(BookingIntent):
    patient_id: str = Field(min_length=1, max_length=120)
    expected_revision: int = Field(ge=0)
    independently_verified: Literal[True]
    evidence_type: Literal["protected_chart_identifiers"]
    replace_existing: bool = False
    observed_booking: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")


class UnlinkBooking(BookingIntent):
    expected_revision: int = Field(ge=1)
    explicit_unlink: Literal[True]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utc(value: object) -> Optional[datetime]:
    """DB and signed Visit Capture instants are naive UTC, NOT Cyprus wall time."""
    try:
        parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
        if not isinstance(parsed, datetime):
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def _day(value: object) -> Optional[date]:
    if not isinstance(value, str) or not ISO_DATE.fullmatch(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _text(value: object, max_len: int = 240) -> str:
    return value.strip()[:max_len] if isinstance(value, str) else ""


def _facts(items: object, count: int = 4, roles: Optional[dict] = None) -> list[dict]:
    """Preserve clinical verification and uncertainty; expose source *roles*, not raw refs."""
    if not isinstance(items, list):
        return []
    result = []
    for item in items[:count]:
        value = _text(getattr(item, "text", ""), 300)
        if not value:
            continue
        provenance = getattr(item, "provenance", [])
        role_names = sorted({(roles or {}).get(p, "unknown")
                             for p in provenance[:4]}) if isinstance(provenance, list) else []
        result.append({
            "text": value,
            "verification": getattr(item, "verification", "source_stated"),
            "certainty": getattr(item, "certainty", "uncertain"),
            "source_roles": role_names,
        })
    return result


def _booking_observation(booking) -> str:
    """Only detects visible source drift; NEVER proves attendee/person continuity."""
    secret = os.environ.get("CLINICAL_DATA_KEY", "")
    details = "\x1f".join([
        booking.uid, booking.start_at.isoformat(), booking.end_at.isoformat(),
        booking.clinic, booking.patient_display_name, booking.reason,
    ])
    return hmac.new(secret.encode(), details.encode(), hashlib.sha256).hexdigest()


def _booking_card(booking) -> dict:
    return {
        "observation": _booking_observation(booking),
        "booking_uid": booking.uid,
        "start_at": booking.start_at.isoformat(),
        "clinic": _text(booking.clinic, 80),
        "reason": _text(booking.reason, 240),
        "patient_display_name": _text(booking.patient_display_name, 200),
    }


def _enabled() -> bool:
    # OFF by default until clinical retention, identifiable-data and release gates.
    return all(os.environ.get(setting, "").strip() == "1" for setting in (
        "CLINICAL_VISIT_BRIEF_LINKS_ENABLED",
        "COCKPIT_STAFF_AUTH_ENABLED",
        "COCKPIT_VISIT_BRIEF_PRIVATE_STORAGE_QUALIFIED",
        "COCKPIT_VISIT_BRIEF_LINK_RETENTION_APPROVED",
        "COCKPIT_VISIT_BRIEF_AUDIT_RETENTION_APPROVED",
    ))


async def _fresh_booking(uid: str, target_date: date | None = None):
    """Same authenticated Reception schedule source/model as clinical_calendar."""
    target = target_date or _utcnow().astimezone(CYPRUS).date()
    url = os.environ.get("RECEPTION_SCHEDULE_CONTEXT_URL", "").strip()
    key = os.environ.get("CLINICAL_INGEST_KEY", "").strip()
    if not url or not key:
        raise HTTPException(503, "Reception schedule unavailable")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                url, params={"target_date": target.isoformat()},
                headers={"X-Clinical-Ingest-Key": key},
            )
        if response.status_code != 200:
            raise ValueError("Reception schedule unavailable")
        schedule = ActualSchedule.model_validate(response.json())
        fetched = _utc(schedule.fetched_at)
        age = _utcnow() - fetched if fetched is not None else None
        if (fetched is None or schedule.fetched_at.tzinfo is None
                or age is None or not timedelta(0) <= age <= timedelta(minutes=5)
                or schedule.coverage_start is None or schedule.coverage_end is None
                or not schedule.coverage_start <= target <= schedule.coverage_end
                or len(schedule.appointments) > 2000):
            raise ValueError("Reception schedule not fresh/complete")
        matches = [item for item in schedule.appointments if (item.uid if item.uid.startswith('cal.com:') else 'cal.com:'+item.uid) == (uid if uid.startswith('cal.com:') else 'cal.com:'+uid)]
        if len(matches) != 1:
            raise HTTPException(409, "Booking missing or UID ambiguous")
        booking = matches[0]
        start, end = _utc(booking.start_at), _utc(booking.end_at)
        booking_day = start.astimezone(CYPRUS).date() if start else None
        if (booking.start_at.tzinfo is None or booking.end_at.tzinfo is None
                or start is None or end is None or end <= start
                or booking_day is None
                or not schedule.coverage_start <= booking_day <= schedule.coverage_end):
            raise ValueError("Booking interval/date invalid or uncovered")
        # An ambiguity anywhere among the returned UID set is not permitted.
        return booking
    except HTTPException:
        raise
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        raise HTTPException(503, "Reception schedule unavailable or unqualified") from exc


async def _fresh_staff_booking(uid: str, engine: Engine,
                               actor: StaffActor, event: str):
    try:
        return await _fresh_booking(uid)
    except HTTPException as exc:
        _record_denial(engine, actor, uid, event, exc.status_code)
        raise


def _minimal_brief(session: Session, patient_id: str, booking) -> dict:
    """Minimal, signed-as-of-booking projection. Never expose raw encounter JSON."""
    booking_start = _utc(booking.start_at)
    if booking_start is None:
        raise HTTPException(503, "Unqualified booking time")
    local_day = booking_start.astimezone(CYPRUS).date()
    patient = session.get(PatientORM, patient_id)
    if patient is None:
        raise HTTPException(409, "Selected protected patient no longer exists")

    eligible = []
    rows = session.scalars(
        select(EncounterORM).where(
            EncounterORM.patient_id == patient_id,
            EncounterORM.status.in_(("completed", "amended")),
        ).order_by(EncounterORM.encounter_date.desc(), EncounterORM.id.asc())
        .limit(100)
    ).all()
    for row in rows:
        clinical_day = _day(row.encounter_date)
        if clinical_day is None or clinical_day > local_day:
            continue
        # Signed content must have existed before the SELECTED appointment.
        # A later amendment overwrites provenance; no pre-amendment version exists.
        created, updated = _utc(row.created_at), _utc(row.updated_at)
        if (created is None or updated is None
                or created >= booking_start or updated >= booking_start):
            continue
        payload = row.payload_json
        signed = payload.get("_visit_capture_v1") if isinstance(payload, dict) else None
        if not isinstance(signed, dict) or signed.get("signed") is not True:
            continue
        signed_at = _utc(signed.get("signed_at"))
        if signed_at is None or signed_at >= booking_start:
            continue
        if clinical_day == local_day and not signed_at < booking_start:
            continue
        try:
            candidate = VisitCaptureCandidateV1.model_validate(signed.get("candidate"))
        except (ValueError, TypeError):
            continue
        # Signed date and encounter ownership must not contradict each other.
        if candidate.encounter_date != row.encounter_date:
            continue
        eligible.append((clinical_day, signed_at, row.id, candidate))

    eligible.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    previous = None
    prior_plan = None
    historical_plan = None
    checks = []
    if eligible:
        encounter_day, signed_at, encounter_id, candidate = eligible[0]
        roles = {s.source_ref: s.role for s in candidate.source_bindings}
        findings = (_facts(candidate.what_changed, 3, roles)
                    + _facts(candidate.findings, 3, roles)
                    + _facts(candidate.clinical_impression, 3, roles))[:6]
        previous = {
            "date": encounter_day.isoformat(),
            "signed_at": signed_at.isoformat(),
            "source_encounter_id": encounter_id,
            "reason": _text(candidate.reason_for_visit.text, 240),
            "reason_evidence": _facts([candidate.reason_for_visit], 1, roles),
            "findings": findings,
        }
        decisions = _facts(candidate.decisions, 5, roles)
        medications = [
            {"name": _text(item.name, 120), "action": item.action,
             "dose": _text(item.dose, 80),
             "verification": item.verification, "certainty": item.certainty,
             "source_roles": sorted({roles.get(p, "unknown")
                                     for p in item.provenance[:4]})}
            for item in candidate.medications[:5]
        ]
        contact = candidate.next_contact
        prior_plan = {
            "decisions": decisions,
            "medications": medications,
            "next_contact": {
                "when": contact.when,
                "purpose": _text(contact.purpose, 240),
                "source_roles": sorted({roles.get(p, "unknown")
                                        for p in contact.provenance[:4]}),
            } if contact else None,
        }
        if contact:
            if _day(contact.when) == local_day:
                checks.extend({"text": v, "kind": "previous_next_contact",
                               "source_encounter_id": encounter_id,
                               "source_roles": sorted({roles.get(p, "unknown")
                                                       for p in contact.provenance[:4]})}
                              for v in [_text(contact.purpose, 240)]
                              + [_text(v, 240) for v in contact.assess[:6]] if v)
            else:
                historical_plan = {
                    "when": contact.when,
                    "purpose": _text(contact.purpose, 240),
                    "assess": [_text(v, 240) for v in contact.assess[:6]],
                    "date_relevance": "unverified",
                    "source_roles": sorted({roles.get(p, "unknown")
                                            for p in contact.provenance[:4]}),
                }

    pending = []
    pending_rows = session.scalars(
        select(ClinicalPendingORM).where(
            ClinicalPendingORM.patient_id == patient_id,
            ClinicalPendingORM.status.in_(tuple(OPEN_PENDING)),
        ).order_by(ClinicalPendingORM.updated_at.desc(), ClinicalPendingORM.id.asc())
        .limit(10)
    ).all()
    for item in pending_rows:
        line = {
            "description": _text(item.description, 240),
            "status": item.status,
            "source_encounter_id": item.source_encounter_id,
            "review_date": item.review_date,
            "recorded_at": _utc(item.updated_at).isoformat()
                if _utc(item.updated_at) else None,
            "truth": "last_recorded_at_read",
        }
        pending.append(line)
        if _day(item.review_date) == local_day:
            checks.append({
                "text": "Επιβεβαίωση καταγεγραμμένης εκκρεμότητας: "
                        + line["description"][:180],
                "kind": "pending_status_check",
                "source_encounter_id": item.source_encounter_id,
            })

    current_local = _utcnow().astimezone(CYPRUS).date()
    return {
        "patient_id": patient_id,
        "selected_booking_date": local_day.isoformat(),
        "check_heading": "Σήμερα" if local_day == current_local
            else "Για το ραντεβού " + local_day.isoformat(),
        "previous_visit": previous,
        "prior_plan": prior_plan,
        "historical_plan": historical_plan,
        "pending": pending,
        "appointment_checks": checks[:12],
        "pending_truth": "last_recorded_at_read",
        "partial": previous is None,
    }


def attach_visit_brief_routes(
    router: APIRouter, engine: Engine, protected: list,
    *, enabled=_enabled, actor_dependency=require_staff_actor,
    clinician_dependency=require_staff_clinician, booking_loader=_fresh_booking,
    namespace=NAMESPACE, patient_ids=None, clinician_role="clinician",
) -> None:
    """Attach staff-only routes. DDL is applied by a reviewed migration, never by a read/boot."""
    if namespace == NAMESPACE:
        attach_chart_booking_routes(router,engine,protected)

    async def fresh_booking(uid, actor, event):
        try:
            return await booking_loader(uid)
        except HTTPException as exc:
            _record_denial(engine, actor, uid, event, exc.status_code)
            raise

    def allowed_patient(session, patient_id):
        if ((patient_ids is not None and patient_id not in patient_ids)
                or (patient_ids is None and patient_id in SYNTHETIC_161_PATIENT_IDS)):
            raise HTTPException(404, "Patient outside booking scope")
        return session.get(PatientORM, patient_id)
    @router.post("/visit-brief/read")
    async def read_visit_brief(
        req: BookingIntent, actor: StaffActor = Depends(actor_dependency),
    ) -> dict:
        # A qualified, durable audit sink is mandatory before any booking read.
        # Never expose an unaudited booking-only projection while the reviewed
        # migration and retention gates are OFF (and tables may not exist).
        if not enabled():
            raise HTTPException(503, "Visit Brief not enabled")
        booking = await fresh_booking(req.booking_uid, actor, "booking_read")
        card = _booking_card(booking)
        with Session(engine) as session:
            row = session.get(BookingPatientLinkORM, (namespace, req.booking_uid))
            if row is None or row.state == "revoked":
                _record_access(session, actor, req.booking_uid, "booking_read", "allowed")
                session.commit()
                return {"mode": "booking_only", "reason": "unlinked",
                        "booking": card, "revision": row.revision if row else 0,
                        "linked": False}
            # Reception has no qualified attendee continuity signal.
            # NEVER promote equal UID/name/fingerprint into patient GET authority.
            _record_access(session, actor, req.booking_uid, "booking_read", "allowed")
            session.commit()
            return {"mode": "booking_only", "reason": "needs_reconfirmation",
                    "booking": card, "revision": row.revision, "linked": True}

    @router.post("/visit-brief/patient-search")
    async def booking_patient_search(
        req: BookingPatientSearch, actor: StaffActor = Depends(actor_dependency),
    ) -> list[dict]:
        if not enabled():
            raise HTTPException(503, "Booking-patient linkage not enabled")
        booking = await fresh_booking(req.booking_uid, actor, "registry_search")
        if not hmac.compare_digest(_booking_observation(booking), req.observed_booking):
            _record_denial(engine, actor, req.booking_uid, "registry_search", 409)
            raise HTTPException(409, "Booking changed since preview")
        terms = _fold_patient_lookup(req.query.strip()).split()
        if not terms:
            _record_denial(engine, actor, req.booking_uid, "registry_search", 422)
            raise HTTPException(422, "Patient query must not be blank")
        rows = []
        matched = 0
        with Session(engine) as session:
            stmt = select(PatientORM).where(
                PatientORM.patient_id.in_(patient_ids) if patient_ids is not None
                else PatientORM.patient_id.not_in(SYNTHETIC_161_PATIENT_IDS)
            ).order_by(
                PatientORM.updated_at.desc(), PatientORM.patient_id.asc())
            for patient in session.scalars(stmt).yield_per(200):
                if not _patient_matches_search(patient, terms):
                    continue
                if matched >= req.offset:
                    demographics = patient.demographics_json or {}
                    if not isinstance(demographics, dict):
                        demographics = {}
                    safe_keys = ("full_name", "fullName", "name", "ονοματεπώνυμο",
                                 "first_name", "firstName", "last_name", "lastName",
                                 "όνομα", "επώνυμο", "dob", "date_of_birth", "birth_date")
                    rows.append({"patient_id": patient.patient_id,
                                 "demographics": {key: demographics[key]
                                                  for key in safe_keys
                                                  if isinstance(demographics.get(key), str)}})
                    if len(rows) >= req.limit:
                        break
                matched += 1
            _record_access(session, actor, req.booking_uid, "registry_search", "allowed")
            session.commit()
        return rows

    @router.post("/visit-brief/confirm")
    async def confirm_visit_brief(
        req: ConfirmBooking, actor: StaffActor = Depends(actor_dependency),
    ) -> dict:
        if not enabled():
            raise HTTPException(503, "Booking-patient linkage not enabled")
        booking = await fresh_booking(req.booking_uid, actor, "brief_read_confirm")
        if not hmac.compare_digest(_booking_observation(booking), req.observed_booking):
            _record_denial(engine, actor, req.booking_uid, "brief_read_confirm", 409)
            raise HTTPException(409, "Booking changed since clinician preview")
        stamp = _utcnow().replace(tzinfo=None)
        with Session(engine) as session:
            try:
                if allowed_patient(session, req.patient_id) is None:
                    raise HTTPException(404, "Protected patient not found")
                link = session.get(BookingPatientLinkORM, (namespace, req.booking_uid))
                if link is None:
                    if req.expected_revision != 0:
                        raise HTTPException(409, "Link revision changed")
                    link = BookingPatientLinkORM(
                        namespace=namespace, booking_uid=req.booking_uid,
                        patient_id=req.patient_id, state="needs_reconfirmation",
                        revision=1, evidence_type=req.evidence_type,
                        confirmed_at=stamp, updated_at=stamp,
                    )
                    session.add(link)
                    event = "confirmed"
                    revision = 1
                else:
                    if link.revision != req.expected_revision:
                        raise HTTPException(409, "Link revision changed")
                    if link.patient_id != req.patient_id and not req.replace_existing:
                        raise HTTPException(409, "Explicit patient correction required")
                    if link.patient_id != req.patient_id and actor.role != clinician_role:
                        raise HTTPException(403, "Only clinician may correct a link")
                    event = ("corrected" if link.patient_id != req.patient_id else
                             "reconfirmed")
                    revision = link.revision + 1
                    # Atomic CAS; no blind overwrite across concurrent confirmations.
                    changed = session.execute(
                        update(BookingPatientLinkORM)
                        .where(
                            BookingPatientLinkORM.namespace == namespace,
                            BookingPatientLinkORM.booking_uid == req.booking_uid,
                            BookingPatientLinkORM.revision == req.expected_revision,
                        )
                        .values(
                            patient_id=req.patient_id,
                            revision=revision, state="needs_reconfirmation",
                            evidence_type=req.evidence_type,
                            confirmed_at=stamp, updated_at=stamp,
                        )
                    )
                    if changed.rowcount != 1:
                        raise HTTPException(409, "Link revision changed")
                session.add(BookingPatientLinkAuditORM(
                    id=str(uuid4()), namespace=namespace, booking_uid=req.booking_uid,
                    patient_id=req.patient_id, event=event, revision=revision,
                    evidence_type=req.evidence_type,
                    operator_type=actor.role, occurred_at=stamp,
                ))
                # Link changes and the one-time clinical read are attributable.
                access_event = {"confirmed": "link_confirm",
                                "reconfirmed": "link_reconfirm",
                                "corrected": "link_correct"}[event]
                _record_access(session, actor, req.booking_uid, access_event,
                               "allowed", req.patient_id, revision)
                # Build projection in this response only; audit commits atomically.
                brief = _minimal_brief(session, req.patient_id, booking)
                _record_access(session, actor, req.booking_uid, "brief_read_confirm",
                               "allowed", req.patient_id, revision)
                session.commit()
            except HTTPException as exc:
                session.rollback()
                event = ("link_correct" if exc.status_code == 403 else
                         "brief_read_confirm")
                _record_denial(engine, actor, req.booking_uid, event, exc.status_code)
                raise
            except IntegrityError as exc:
                session.rollback()
                _record_denial(engine, actor, req.booking_uid,
                               "brief_read_confirm", 409)
                raise HTTPException(409, "Concurrent link insertion/revision conflict") from exc
        return {"mode": "confirmed_once", "booking": _booking_card(booking),
                "revision": revision, "brief": brief}

    @router.post("/visit-brief/unlink")
    async def unlink_visit_brief(
        req: UnlinkBooking, actor: StaffActor = Depends(actor_dependency),
    ) -> dict:
        if not enabled():
            raise HTTPException(503, "Booking-patient linkage not enabled")
        if actor.role != clinician_role:
            # Denied action is attributable; no patient identity is disclosed.
            with Session(engine) as audit:
                _record_access(audit, actor, req.booking_uid, "link_revoke", "denied")
                audit.commit()
            raise HTTPException(403, "Clinician role required")
        # Revocation works even if Reception has since cancelled the booking.
        stamp = _utcnow().replace(tzinfo=None)
        with Session(engine) as session:
            row = session.get(BookingPatientLinkORM, (namespace, req.booking_uid))
            if row is None or row.revision != req.expected_revision:
                session.rollback()
                _record_denial(engine, actor, req.booking_uid, "link_revoke", 409)
                raise HTTPException(409, "Link revision changed")
            revision = row.revision + 1
            changed = session.execute(
                update(BookingPatientLinkORM)
                .where(
                    BookingPatientLinkORM.namespace == namespace,
                    BookingPatientLinkORM.booking_uid == req.booking_uid,
                    BookingPatientLinkORM.revision == req.expected_revision,
                )
                .values(state="revoked", revision=revision, updated_at=stamp)
            )
            if changed.rowcount != 1:
                session.rollback()
                _record_denial(engine, actor, req.booking_uid, "link_revoke", 409)
                raise HTTPException(409, "Link revision changed")
            session.add(BookingPatientLinkAuditORM(
                id=str(uuid4()), namespace=namespace, booking_uid=req.booking_uid,
                patient_id=row.patient_id, event="revoked", revision=revision,
                evidence_type=None, operator_type=actor.role,
                occurred_at=stamp,
            ))
            _record_access(session, actor, req.booking_uid, "link_revoke",
                           "allowed", row.patient_id, revision)
            session.commit()
        return {"mode": "booking_only", "reason": "unlinked",
                "revision": revision}

    @router.get("/visit-brief/staff-audit")
    def visit_brief_staff_audit(
        actor: StaffActor = Depends(clinician_dependency),
    ) -> list[dict]:
        if not enabled():
            raise HTTPException(503, "Visit Brief not enabled")
        with Session(engine) as session:
            rows = session.scalars(
                select(VisitBriefStaffAuditORM)
                .where(VisitBriefStaffAuditORM.principal_id == "synthetic-key-test"
                       if patient_ids is not None else
                       VisitBriefStaffAuditORM.principal_id != "synthetic-key-test")
                .order_by(VisitBriefStaffAuditORM.occurred_at.desc(),
                          VisitBriefStaffAuditORM.id.desc()).limit(100)
            ).all()
            return [{"principal_id": item.principal_id, "role": item.role,
                     "event": item.event, "outcome": item.outcome,
                     "occurred_at": item.occurred_at.isoformat(),
                     "booking_uid": item.booking_uid, "revision": item.revision}
                    for item in rows]

class ChartBookingIntent(BookingIntent):
    target_date: date | None = None
    @field_validator('booking_uid')
    @classmethod
    def canonical_uid(cls,value):
        uid=value if value.startswith('cal.com:') else 'cal.com:'+value
        if len(uid)>160:raise ValueError('Booking identifier too long')
        return uid

class ChartBookingSelection(ChartBookingIntent):
    patient_id: str = Field(min_length=1, max_length=120)
    expected_revision: int = Field(ge=0)
    observed_booking: str = Field(pattern=r"^[0-9a-f]{64}$")
    replace_existing: bool = False
    request_id: UUID4

class ChartBookingUnlink(ChartBookingIntent):
    expected_revision: int = Field(ge=1)
    request_id: UUID4


def _chart_reference(booking):
    # An explicit doctor's chart choice, not an upstream identity assertion.
    raw=json.dumps([NAMESPACE,booking.uid,booking.patient_display_name],ensure_ascii=False,separators=(',',':'))
    return 'chart_selection:'+hmac.new(os.environ.get('CLINICAL_DATA_KEY','').encode(),raw.encode(),hashlib.sha256).hexdigest()[:48]


def _latest_chart_summary(db,patient_id):
    """Current stored chart summary; deliberately distinct from appointment-as-of Brief."""
    now=_utcnow();today=now.astimezone(CYPRUS).date()
    rows=db.scalars(select(EncounterORM).where(EncounterORM.patient_id==patient_id,
        EncounterORM.status.in_(['completed','amended'])).order_by(EncounterORM.encounter_date.desc(),EncounterORM.updated_at.desc()).limit(100)).all()
    for row in rows:
        day=_day(row.encounter_date);stamp=(row.payload_json or {}).get('_visit_capture_v1',{})
        signed=_utc(stamp.get('signed_at'))
        if not day or day>today or not stamp.get('signed') or not signed or signed>now:continue
        try:candidate=VisitCaptureCandidateV1.model_validate(stamp.get('candidate'))
        except (ValueError,TypeError):continue
        if candidate.encounter_date!=row.encounter_date:continue
        roles={s.source_ref:s.role for s in candidate.source_bindings}
        return {'source_encounter_id':row.id,'date':row.encounter_date,'reason':_text(candidate.reason_for_visit.text),
            'findings':_facts(candidate.findings,3,roles)+_facts(candidate.clinical_impression,2,roles),
            'decisions':_facts(candidate.decisions,4,roles)}
    return None


def attach_chart_booking_routes(router,engine,protected):
    """Normal physician chart association over the existing link/audit owner."""
    from clinical_pending_one_tap import actor_for
    def clinician(request:Request):return actor_for(request,None)
    def no_cache(response:Response):response.headers['Cache-Control']='no-store'
    secured=protected+[Depends(no_cache)]
    def ready():
        inspector=inspect(engine)
        if not all(inspector.has_table(t.__tablename__) for t in [BookingPatientLinkORM,BookingPatientLinkAuditORM,VisitBriefStaffAuditORM]):
            raise HTTPException(503,'Chart linkage storage is not installed')
    def reply(db,row,booking):
        patient=db.get(PatientORM,row.patient_id)
        if not patient:raise HTTPException(409,'Linked chart no longer exists')
        return {'mode':'chart_linked','booking':_booking_card(booking),'revision':row.revision,'linked':True,
            'patient_id':row.patient_id,'patient_name':(patient.demographics_json or {}).get('full_name') or 'Κλινικός φάκελος',
            'latest_visit':_latest_chart_summary(db,row.patient_id),
            'pending':[{'pending_id':p.id,'description':p.description,'status':p.status} for p in db.scalars(select(ClinicalPendingORM).where(ClinicalPendingORM.patient_id==row.patient_id,ClinicalPendingORM.status.in_(tuple(OPEN_PENDING))).order_by(ClinicalPendingORM.updated_at.desc()).limit(20))]}
    def commit(db):
        try:db.commit()
        except IntegrityError:
            db.rollback();raise HTTPException(409,'Booking link changed; reload') from None
    async def booking(req):
        ready();source=await _fresh_booking(req.booking_uid,req.target_date)
        if not source.patient_display_name.strip():raise HTTPException(409,'Booking attendee is unavailable')
        return source.model_copy(update={'uid':req.booking_uid})

    @router.post('/visit-brief/chart-read',dependencies=secured)
    async def chart_read(req:ChartBookingIntent,actor:StaffActor=Depends(clinician)):
        source=await booking(req)
        with Session(engine) as db:
            row=db.get(BookingPatientLinkORM,(NAMESPACE,req.booking_uid))
            _record_access(db,actor,req.booking_uid,'chart_link_read','allowed')
            if row and row.state=='chart_linked' and hmac.compare_digest(row.evidence_type,_chart_reference(source)):
                result=reply(db,row,source)
            else:result={'mode':'booking_only','booking':_booking_card(source),'revision':row.revision if row else 0,
                'linked':bool(row and row.state!='revoked'),'replace_required':bool(row),'reason':'choose_chart'}
            commit(db);return result

    @router.post('/visit-brief/chart-select',dependencies=secured)
    async def chart_select(req:ChartBookingSelection,actor:StaffActor=Depends(clinician)):
        source=await booking(req)
        if not hmac.compare_digest(req.observed_booking,_booking_observation(source)):raise HTTPException(409,'Booking changed since preview')
        reference=_chart_reference(source);stamp=_utcnow().replace(tzinfo=None)
        with Session(engine) as db:
            row=db.scalar(select(BookingPatientLinkORM).where(BookingPatientLinkORM.namespace==NAMESPACE,
                BookingPatientLinkORM.booking_uid==req.booking_uid).with_for_update())
            previous=db.get(BookingPatientLinkAuditORM,str(req.request_id))
            if previous:
                if (previous.namespace!=NAMESPACE or previous.booking_uid!=req.booking_uid or previous.patient_id!=req.patient_id
                    or previous.event!='chart_selected' or previous.operator_type!=actor.role or previous.evidence_type!=reference
                    or previous.revision!=req.expected_revision+1 or not row or row.revision!=previous.revision
                    or row.patient_id!=req.patient_id or row.state!='chart_linked' or row.evidence_type!=reference):
                    raise HTTPException(409,'Selection replay conflict')
                return reply(db,row,source)
            if not db.get(PatientORM,req.patient_id):raise HTTPException(404,'Patient not found')
            if req.patient_id in SYNTHETIC_161_PATIENT_IDS:raise HTTPException(404,'Patient outside normal chart scope')
            if (row.revision if row else 0)!=req.expected_revision:raise HTTPException(409,'Booking link revision changed')
            if row and (row.patient_id!=req.patient_id or row.state=='revoked' or row.evidence_type!=reference) and not req.replace_existing:
                raise HTTPException(409,'Explicit replacement required')
            revision=req.expected_revision+1
            values=dict(patient_id=req.patient_id,state='chart_linked',revision=revision,evidence_type=reference,confirmed_at=stamp,updated_at=stamp)
            if row:
                changed=db.execute(update(BookingPatientLinkORM).where(BookingPatientLinkORM.namespace==NAMESPACE,
                    BookingPatientLinkORM.booking_uid==req.booking_uid,BookingPatientLinkORM.revision==req.expected_revision).values(**values))
                if changed.rowcount!=1:raise HTTPException(409,'Booking link changed')
                db.expire(row)
            else:
                row=BookingPatientLinkORM(namespace=NAMESPACE,booking_uid=req.booking_uid,**values);db.add(row)
            db.add(BookingPatientLinkAuditORM(id=str(req.request_id),namespace=NAMESPACE,booking_uid=req.booking_uid,patient_id=req.patient_id,
                event='chart_selected',revision=revision,evidence_type=reference,operator_type=actor.role,occurred_at=stamp))
            _record_access(db,actor,req.booking_uid,'chart_link_select','allowed',req.patient_id,revision)
            try:db.flush()
            except IntegrityError:
                db.rollback();raise HTTPException(409,'Booking link changed; reload') from None
            result=reply(db,row,source);commit(db);return result

    @router.post('/visit-brief/chart-unlink',dependencies=secured)
    async def chart_unlink(req:ChartBookingUnlink,actor:StaffActor=Depends(clinician)):
        # Revocation must remain possible during Reception downtime.
        ready()
        with Session(engine) as db:
            row=db.scalar(select(BookingPatientLinkORM).where(BookingPatientLinkORM.namespace==NAMESPACE,
                BookingPatientLinkORM.booking_uid==req.booking_uid).with_for_update())
            prior=db.get(BookingPatientLinkAuditORM,str(req.request_id))
            if prior:
                if prior.namespace!=NAMESPACE or prior.booking_uid!=req.booking_uid or prior.operator_type!=actor.role or prior.event!='chart_revoked' or prior.revision!=req.expected_revision+1 or not row or row.state!='revoked' or row.revision!=prior.revision or row.patient_id!=prior.patient_id:
                    raise HTTPException(409,'Unlink replay conflict')
                return {'unlinked':True}
            if not row or row.revision!=req.expected_revision:raise HTTPException(409,'Booking link revision changed')
            row.state='revoked';row.revision+=1;row.updated_at=_utcnow().replace(tzinfo=None)
            db.add(BookingPatientLinkAuditORM(id=str(req.request_id),namespace=NAMESPACE,booking_uid=req.booking_uid,patient_id=row.patient_id,
                event='chart_revoked',revision=row.revision,evidence_type=row.evidence_type,operator_type=actor.role,occurred_at=row.updated_at))
            _record_access(db,actor,req.booking_uid,'chart_link_revoke','allowed',row.patient_id,row.revision)
            commit(db);return {'unlinked':True}
