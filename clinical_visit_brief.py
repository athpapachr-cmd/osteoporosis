"""Booking-scoped, protected Visit Brief. No provider writes or inferred person identity.

Reception remains the calendar source; Clinical Data owns patient/encounter/pending facts.
Current Reception payload has no authoritative attendee continuity assertion, so every
clinical read requires an independent explicit confirmation in that response only.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import Literal, Optional
from uuid import uuid4
from zoneinfo import ZoneInfo

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy import Column, DateTime, Integer, String, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy.exc import IntegrityError

from clinical_data import (
    ClinicalPendingORM, EncounterORM, PatientORM, VisitCaptureCandidateV1,
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


class BookingIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    booking_uid: str = Field(min_length=1, max_length=160)


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
    return os.environ.get("CLINICAL_VISIT_BRIEF_LINKS_ENABLED", "").strip() == "1"


async def _fresh_booking(uid: str):
    """Same authenticated Reception schedule source/model as clinical_calendar."""
    target = _utcnow().astimezone(CYPRUS).date()
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
        matches = [item for item in schedule.appointments if item.uid == uid]
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
) -> None:
    """Extend the existing Calendar router, with existing Clinical Key dependency."""
    VisitBriefBase.metadata.create_all(bind=engine)

    @router.post("/visit-brief/read", dependencies=protected)
    async def read_visit_brief(req: BookingIntent) -> dict:
        booking = await _fresh_booking(req.booking_uid)
        card = _booking_card(booking)
        if not _enabled():
            return {"mode": "booking_only", "reason": "disabled",
                    "booking": card, "revision": 0, "linked": False}
        with Session(engine) as session:
            row = session.get(BookingPatientLinkORM, (NAMESPACE, req.booking_uid))
            if row is None or row.state == "revoked":
                return {"mode": "booking_only", "reason": "unlinked",
                        "booking": card, "revision": row.revision if row else 0,
                        "linked": False}
            # Reception has no qualified attendee continuity signal.
            # NEVER promote equal UID/name/fingerprint into patient GET authority.
            return {"mode": "booking_only", "reason": "needs_reconfirmation",
                    "booking": card, "revision": row.revision, "linked": True}

    @router.post("/visit-brief/confirm", dependencies=protected)
    async def confirm_visit_brief(req: ConfirmBooking) -> dict:
        if not _enabled():
            raise HTTPException(503, "Booking-patient linkage not enabled")
        booking = await _fresh_booking(req.booking_uid)
        if not hmac.compare_digest(_booking_observation(booking), req.observed_booking):
            raise HTTPException(409, "Booking changed since clinician preview")
        stamp = _utcnow().replace(tzinfo=None)
        with Session(engine) as session:
            try:
                if session.get(PatientORM, req.patient_id) is None:
                    raise HTTPException(404, "Protected patient not found")
                link = session.get(BookingPatientLinkORM, (NAMESPACE, req.booking_uid))
                if link is None:
                    if req.expected_revision != 0:
                        raise HTTPException(409, "Link revision changed")
                    link = BookingPatientLinkORM(
                        namespace=NAMESPACE, booking_uid=req.booking_uid,
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
                    event = ("corrected" if link.patient_id != req.patient_id else
                             "reconfirmed")
                    revision = link.revision + 1
                    # Atomic CAS; no blind overwrite across concurrent confirmations.
                    changed = session.execute(
                        update(BookingPatientLinkORM)
                        .where(
                            BookingPatientLinkORM.namespace == NAMESPACE,
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
                    id=str(uuid4()), namespace=NAMESPACE, booking_uid=req.booking_uid,
                    patient_id=req.patient_id, event=event, revision=revision,
                    evidence_type=req.evidence_type,
                    operator_type="shared_clinical_key", occurred_at=stamp,
                ))
                # Build projection in the exact successful confirmation request only.
                brief = _minimal_brief(session, req.patient_id, booking)
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                raise HTTPException(409, "Concurrent link insertion/revision conflict") from exc
        return {"mode": "confirmed_once", "booking": _booking_card(booking),
                "revision": revision, "brief": brief}

    @router.post("/visit-brief/unlink", dependencies=protected)
    async def unlink_visit_brief(req: UnlinkBooking) -> dict:
        if not _enabled():
            raise HTTPException(503, "Booking-patient linkage not enabled")
        # Revocation works even if Reception has since cancelled the booking.
        stamp = _utcnow().replace(tzinfo=None)
        with Session(engine) as session:
            row = session.get(BookingPatientLinkORM, (NAMESPACE, req.booking_uid))
            if row is None or row.revision != req.expected_revision:
                raise HTTPException(409, "Link revision changed")
            revision = row.revision + 1
            changed = session.execute(
                update(BookingPatientLinkORM)
                .where(
                    BookingPatientLinkORM.namespace == NAMESPACE,
                    BookingPatientLinkORM.booking_uid == req.booking_uid,
                    BookingPatientLinkORM.revision == req.expected_revision,
                )
                .values(state="revoked", revision=revision, updated_at=stamp)
            )
            if changed.rowcount != 1:
                raise HTTPException(409, "Link revision changed")
            session.add(BookingPatientLinkAuditORM(
                id=str(uuid4()), namespace=NAMESPACE, booking_uid=req.booking_uid,
                patient_id=row.patient_id, event="revoked", revision=revision,
                evidence_type=None, operator_type="shared_clinical_key",
                occurred_at=stamp,
            ))
            session.commit()
        return {"mode": "booking_only", "reason": "unlinked",
                "revision": revision}
