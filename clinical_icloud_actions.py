"""One-doctor iCloud message decisions, separate from immutable clinical visit.

A message's email header/UID is checked against the existing alias-scoped IMAP
reader before any action. The database stores only an opaque account+UID key,
an optional patient/visit/pending reference, event type and timestamps — never
the message subject, sender, raw body, downloaded files or patient phone.

This feature uses the *existing* authenticated clinician Clinical Key. No
staff account or GESY referral number is required for the pilot.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import re
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, UniqueConstraint, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session

from clinical_data import ClinicalPendingORM, EncounterORM, PatientORM
from clinical_calendar import ClinicalAppointmentORM
from clinical_icloud_mail import _ITEM_RE, _fetch_headers, _summary, _read, _settings, _message, _body
from clinical_icloud_triage_rules import fold
from clinical_reception_calls import _require_clinical_key

_LOG = logging.getLogger(__name__)
_CATEGORIES = frozenset(("clinical", "administrative", "other", "review"))
_ACTIONS = frozenset((
    "classify", "link", "unlink", "review_no_contact", "review_notify",
    "contact_attempt", "patient_notified", "callback_complete",
))
_ACTIVE = frozenset(("open", "awaiting_patient", "awaiting_result"))
_DIGITS = re.compile(r"\D")
_NAME_KEYS = ("full_name", "fullName", "name", "ονοματεπώνυμο", "ονοματεπωνυμο")


class MailDecisionBase(DeclarativeBase):
    pass


class MailDecisionORM(MailDecisionBase):
    __tablename__ = "clinical_icloud_message_decisions"
    source_key = Column(String(64), primary_key=True)
    message_ref = Column(String(40), nullable=False)
    category = Column(String(24), nullable=False, default="review")
    patient_id = Column(String(120), nullable=True, index=True)
    pending_id = Column(String(120), nullable=True, index=True)
    booking_id = Column(String(160), nullable=True)
    version = Column(Integer, nullable=False, default=0)
    review_status = Column(String(32), nullable=False, default="not_reviewed")
    contact_status = Column(String(32), nullable=False, default="not_requested")
    callback_complete = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime, nullable=False)


class MailDecisionEventORM(MailDecisionBase):
    __tablename__ = "clinical_icloud_message_action_events"
    __table_args__ = (
        UniqueConstraint("source_key", "version", name="uq_icloud_message_action_version"),
        UniqueConstraint("source_key", "request_id", name="uq_icloud_message_action_request"),
    )
    id = Column(String(36), primary_key=True)
    source_key = Column(String(64), ForeignKey(MailDecisionORM.source_key), nullable=False)
    version = Column(Integer, nullable=False)
    request_id = Column(String(36), nullable=False)
    action = Column(String(32), nullable=False)
    actor = Column(String(48), nullable=False)
    value_hash = Column(String(64), nullable=False)
    created_at = Column(DateTime, nullable=False)


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: str
    expected_version: int = Field(ge=0)
    request_id: str = Field(pattern=r"^[a-f0-9-]{36}$")
    category: str | None = None
    patient_id: str | None = Field(default=None, max_length=120)
    pending_id: str | None = Field(default=None, max_length=120)
    booking_id: str | None = Field(default=None, max_length=160)


def _stamp() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _key(item_id: str) -> str:
    matched = _ITEM_RE.fullmatch(item_id)
    if not matched:
        raise HTTPException(404, "Unknown message identity")
    username, _password, alias = _settings()
    canonical = "\x00".join((username.casefold(), alias, "INBOX", *matched.groups()))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _qualify(item_id: str, *, include_body: bool = False) -> dict:
    parsed = _ITEM_RE.fullmatch(item_id)
    if not parsed:
        raise HTTPException(404, "Unknown message")
    uidvalidity, uid = parsed.groups()

    def verify(mail, validity, alias):
        if validity != uidvalidity:
            raise HTTPException(409, "Mailbox UIDVALIDITY changed")
        if include_body:
            message = _message(mail, validity, alias, item_id)
        else:
            message, _size = _fetch_headers(mail, uid)
        result = _summary(uid, validity, message, alias, include_nonclinical=True)
        if result is None:
            raise HTTPException(404, "Message outside practice mailbox")
        if include_body:
            result["text"] = _body(message)[:7000]
        return result

    return _read(verify)


def _origin(request: Request) -> None:
    origin = request.headers.get("origin", "")
    expected = f"{request.url.scheme}://{request.url.netloc}"
    if not hmac.compare_digest(origin, expected):
        raise HTTPException(403, "Same-origin clinical action required")


def _projection(row: MailDecisionORM | None, item_id: str, suggested: str) -> dict:
    if row is None:
        return {
            "item_id": item_id, "version": 0, "category": suggested,
            "patient_id": None, "pending_id": None, "booking_id": None,
            "review_status": "not_reviewed", "contact_status": "not_requested",
            "callback_complete": False,
        }
    return {
        "item_id": item_id, "version": row.version, "category": row.category,
        "patient_id": row.patient_id, "pending_id": row.pending_id,
        "booking_id": row.booking_id,
        "review_status": row.review_status, "contact_status": row.contact_status,
        "callback_complete": row.callback_complete == 1,
    }


def _source_valid(db: Session, row: MailDecisionORM) -> bool:
    if row.pending_id:
        pending = db.get(ClinicalPendingORM, row.pending_id)
        if not pending or pending.patient_id != row.patient_id or pending.status not in _ACTIVE:
            return False
        encounter = db.get(EncounterORM, pending.source_encounter_id)
        if (not encounter or encounter.patient_id != row.patient_id
                or encounter.status not in ("completed", "amended")):
            return False
        signed = (encounter.payload_json or {}).get("_visit_capture_v1", {})
        if not isinstance(signed, dict) or not signed.get("signed"):
            return False
    if row.patient_id and db.get(PatientORM, row.patient_id) is None:
        return False
    if row.booking_id:
        booking = db.get(ClinicalAppointmentORM, row.booking_id)
        if not booking or not booking.patient_display_name.strip():
            return False
        if booking.linked_patient_id and row.patient_id and booking.linked_patient_id != row.patient_id:
            return False
    return True


def _validate_link(db: Session, patient_id: str | None,
                   pending_id: str | None, booking_id: str | None) -> None:
    if not patient_id and not booking_id:
        raise HTTPException(422, "Choose a chart or an existing appointment")
    if pending_id and not patient_id:
        raise HTTPException(422, "Pending must belong to a selected chart")
    if patient_id:
        patient = db.get(PatientORM, patient_id)
        if patient is None or patient.patient_id.startswith("cockpit-synthetic-161-"):
            raise HTTPException(404, "Patient not found")
    if booking_id:
        booking = db.get(ClinicalAppointmentORM, booking_id)
        if not booking or not booking.patient_display_name.strip():
            raise HTTPException(404, "Appointment not found")
        if patient_id and booking.linked_patient_id and booking.linked_patient_id != patient_id:
            raise HTTPException(409, "Existing appointment belongs to another chart")
    if pending_id:
        pending = db.get(ClinicalPendingORM, pending_id)
        if pending is None or pending.patient_id != patient_id or pending.status not in _ACTIVE:
            raise HTTPException(409, "Pending is not a current obligation for this chart")
        source = db.get(EncounterORM, pending.source_encounter_id)
        signed = (source.payload_json or {}).get("_visit_capture_v1", {}) if source else {}
        if (not source or source.patient_id != patient_id
                or source.status not in ("completed", "amended")
                or not isinstance(signed, dict) or not signed.get("signed")):
            raise HTTPException(409, "No current signed Pending source")


def _digest(data: Action) -> str:
    return hashlib.sha256(data.model_dump_json(exclude_none=False).encode("utf-8")).hexdigest()


def _matches(name: str, search: str, source: str) -> bool:
    candidate = fold(name)
    words = candidate.split()
    if len(words) < 2 or len(candidate) < 6:
        return False
    if search:
        terms = fold(search).split()
        return bool(terms) and all(term in candidate for term in terms)
    return candidate in fold(source)


def _phone(value: str) -> str:
    return _DIGITS.sub("", value or "")[-10:]


def _candidate_rows(engine: Engine, source: dict, q: str) -> list[dict]:
    if not q and not (source.get("subject") or source.get("from")):
        return []
    matches = []
    haystack = str(source.get("subject") or "") + " " + str(source.get("from") or "")
    with Session(engine) as db:
        for patient in db.scalars(select(PatientORM).order_by(PatientORM.patient_id).limit(5000)):
            data = patient.demographics_json if isinstance(patient.demographics_json, dict) else {}
            name = next((data[k] for k in _NAME_KEYS if isinstance(data.get(k), str)
                         and data[k].strip()), "")
            if not name or not _matches(name, q, haystack):
                continue
            pendings = db.scalars(select(ClinicalPendingORM)
                .where(ClinicalPendingORM.patient_id == patient.patient_id,
                       ClinicalPendingORM.status.in_(tuple(_ACTIVE)))
                .order_by(ClinicalPendingORM.updated_at.desc()).limit(8)).all()
            matches.append({
                "kind": "patient", "id": patient.patient_id, "name": name[:140],
                "phone": str(data.get("phone") or "")[:40],
                "pending": [{"id": p.id, "label": p.description[:150]}
                            for p in pendings],
            })
            if len(matches) >= 10:
                break
        if len(matches) < 10:
            for booking in db.scalars(select(ClinicalAppointmentORM)
                    .where(ClinicalAppointmentORM.status != "cancelled")
                    .order_by(ClinicalAppointmentORM.start_at.desc()).limit(1500)):
                if not _matches(booking.patient_display_name, q, haystack):
                    continue
                matches.append({
                    "kind": "appointment", "id": booking.id,
                    "name": booking.patient_display_name[:140],
                    "phone": (booking.phone_e164 or "")[:40],
                    "pending": [],
                })
                if len(matches) >= 10:
                    break
    return matches


def build_icloud_actions_router(engine: Engine) -> APIRouter:
    """Uses existing Clinical Key — there is currently ONE cockpit clinician."""
    router = APIRouter(
        prefix="/clinical/icloud-actions", tags=["clinical-icloud-actions"],
        dependencies=[Depends(_require_clinical_key)],
    )
    installed = False
    try:
        # Additive, idempotent metadata-only table creation. Failure must NEVER
        # prevent the existing live Cockpit or read-only iCloud viewer boot.
        MailDecisionBase.metadata.create_all(engine, checkfirst=True)
        installed = True
    except SQLAlchemyError:
        _LOG.warning("icloud_action_storage_unavailable")  # no PHI/secrets

    def ready():
        if not installed:
            raise HTTPException(503, "Mail action storage unavailable")

    @router.get("/items/{item_id}")
    def read_action(item_id: str, response: Response):
        response.headers["Cache-Control"] = "private, no-store"
        ready()
        source = _qualify(item_id)
        key = _key(item_id)
        with Session(engine) as db:
            row = db.get(MailDecisionORM, key)
            result = _projection(row, item_id, source["category"])
            result["source_valid"] = row is None or _source_valid(db, row)
            if not result["source_valid"]:
                result["review_status"] = "source_invalidated"
                result["callback_complete"] = False
            return result

    @router.get("/items/{item_id}/candidates")
    def candidates(item_id: str, response: Response,
                   q: str = Query(default="", max_length=140)):
        response.headers["Cache-Control"] = "private, no-store"
        ready()
        source = _qualify(item_id)
        if q and len(q.strip()) < 3:
            raise HTTPException(422, "Enter at least three name characters")
        return {"item_id": item_id, "items": _candidate_rows(engine, source, q)}

    @router.post("/items/{item_id}")
    def action(item_id: str, change: Action, request: Request, response: Response):
        response.headers["Cache-Control"] = "private, no-store"
        ready()
        _origin(request)
        if change.action not in _ACTIONS:
            raise HTTPException(422, "Unknown clinical mail action")
        source = _qualify(item_id)
        key = _key(item_id)
        digest = _digest(change)
        now = _stamp()
        try:
            with Session(engine) as db, db.begin():
                row = db.scalar(
                    select(MailDecisionORM).where(MailDecisionORM.source_key == key)
                    .with_for_update()
                )
                previous = db.scalar(select(MailDecisionEventORM).where(
                    MailDecisionEventORM.source_key == key,
                    MailDecisionEventORM.request_id == change.request_id,
                ))
                if previous is not None:
                    if previous.value_hash != digest:
                        raise HTTPException(409, "Conflicting mail action replay")
                    return _projection(row, item_id, source["category"])
                if row is None:
                    if change.expected_version != 0:
                        raise HTTPException(409, "Mail status changed; refresh")
                    row = MailDecisionORM(
                        source_key=key, message_ref=item_id,
                        category=source["category"], version=0, updated_at=now,
                        review_status="not_reviewed", contact_status="not_requested",
                        callback_complete=0,
                    )
                    db.add(row)
                    db.flush()
                if row.version != change.expected_version:
                    raise HTTPException(409, "Mail status changed; refresh")
                if not _source_valid(db, row):
                    raise HTTPException(409, "Previous patient/visit binding is no longer valid")
                choice = change.action
                if choice == "classify":
                    if change.category not in _CATEGORIES:
                        raise HTTPException(422, "Choose a known mail category")
                    row.category = change.category
                elif choice == "link":
                    if row.review_status != "not_reviewed":
                        raise HTTPException(409, "Reopen email before changing its clinical linkage")
                    _validate_link(db, change.patient_id, change.pending_id, change.booking_id)
                    row.patient_id, row.pending_id, row.booking_id = (
                        change.patient_id, change.pending_id, change.booking_id
                    )
                elif choice == "unlink":
                    if row.review_status != "not_reviewed":
                        raise HTTPException(409, "Reviewed email cannot lose its visit identity")
                    row.patient_id = row.pending_id = row.booking_id = None
                elif choice in ("review_no_contact", "review_notify"):
                    if not (row.patient_id or row.booking_id) or row.review_status != "not_reviewed":
                        raise HTTPException(409, "Select and validate the actual visit first")
                    if row.category != "clinical":
                        raise HTTPException(409, "Only clinical mail may be medically reviewed")
                    row.review_status = "reviewed_no_contact" if choice == "review_no_contact" else "reviewed_notify"
                    row.contact_status = "not_requested" if choice == "review_no_contact" else "to_contact"
                elif choice in ("contact_attempt", "patient_notified"):
                    if (row.review_status != "reviewed_notify"
                            or row.contact_status not in ("to_contact", "attempted")):
                        raise HTTPException(409, "No outstanding doctor-authorized contact task")
                    row.contact_status = "attempted" if choice == "contact_attempt" else "notified"
                elif choice == "callback_complete":
                    if (row.review_status != "reviewed_notify"
                            or row.contact_status != "notified"
                            or row.callback_complete):
                        raise HTTPException(409, "Patient notification or callback is not complete")
                    row.callback_complete = 1
                row.version += 1
                row.updated_at = now
                db.add(MailDecisionEventORM(
                    id=str(uuid4()), source_key=key, request_id=change.request_id,
                    version=row.version, action=choice, actor="solo_clinician",
                    value_hash=digest, created_at=now,
                ))
                db.flush()
                return _projection(row, item_id, source["category"])
        except IntegrityError:
            raise HTTPException(409, "Concurrent mail action; refresh") from None

    return router
