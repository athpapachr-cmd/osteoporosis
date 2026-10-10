from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4
import os
import secrets
import unicodedata

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, JSON, String, delete, select, func
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


class VisitCaptureContextORM(ClinicalBase):
    __tablename__ = "clinical_visit_capture_contexts"

    id = Column(String, primary_key=True)
    client_session_id = Column(String, nullable=False, index=True)
    patient_id = Column(String, nullable=False, index=True)
    patient_updated_at = Column(DateTime, nullable=False)
    issued_at = Column(DateTime, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False, index=True)


class ClinicalPendingORM(ClinicalBase):
    __tablename__ = "clinical_pending_items"

    id = Column(String, primary_key=True)
    patient_id = Column(String, nullable=False, index=True)
    source_encounter_id = Column(String, nullable=False, index=True)
    item_type = Column(String, nullable=False, index=True)
    description = Column(String, nullable=False)
    trigger_text = Column(String, nullable=True)
    status = Column(String, nullable=False, index=True)
    responsible_role = Column(String, nullable=False)
    external_dependency_json = Column(JSON, nullable=True)
    review_date = Column(String, nullable=True, index=True)
    provenance_json = Column(JSON, nullable=False, default=list)
    resolution_event_id = Column(String, nullable=True, index=True)
    created_at = Column(DateTime, nullable=False, index=True)
    updated_at = Column(DateTime, nullable=False, index=True)


# Search identity fields only, never arbitrary clinical/demographic free text.
# No search result is an authoritative patient link: Visit Capture still requires
# an explicit choice of a stored patient_id and the existing protected context.
_PATIENT_SEARCH_NAME_KEYS = (
    "full_name", "fullName", "name", "first_name", "firstName",
    "last_name", "lastName", "given_name", "family_name",
    "surname", "firstname", "lastname",
    "ονομα", "όνομα", "επωνυμο", "επώνυμο", "ονοματεπωνυμο", "ονοματεπώνυμο",
)


def _fold_patient_lookup(value: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKD", value).casefold()
        if not unicodedata.combining(char)
    )


def _patient_matches_search(patient: PatientORM, search_terms: List[str]) -> bool:
    demographics = patient.demographics_json
    names = []
    if isinstance(demographics, dict):
        names = [
            str(demographics[key])
            for key in _PATIENT_SEARCH_NAME_KEYS
            if isinstance(demographics.get(key), str) and demographics[key].strip()
        ]
    searchable = _fold_patient_lookup(" ".join([patient.patient_id, *names]))
    return all(term in searchable for term in search_terms)


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


class RecentEncounterSummary(BaseModel):
    """Minimal protected Home projection; no encounter content or demographics."""
    patient_id: str
    patient_display_name: str
    encounter_date: str
    visit_type: str


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


SourceRole = Literal["heidi_today", "gesy_today", "gesy_previous", "clinician_edit"]
VerificationState = Literal["source_stated", "clinician_confirmed"]
CertaintyState = Literal["certain", "uncertain", "conflicting"]
SourceIdentityState = Literal["consistent", "uncertain", "conflict"]
ChangeDirection = Literal["improved", "worsened", "new", "resolved", "unchanged"]
ComparisonType = Literal["previous_structured_encounter", "previous_gesy_visit", "unavailable"]
DecisionType = Literal["therapy", "referral", "medication", "advice", "sick_leave", "admin"]
MedicationAction = Literal["prescribed", "continued", "stopped"]
PendingStatus = Literal[
    "not_yet_indicated",
    "open",
    "awaiting_patient",
    "awaiting_result",
    "resolved",
    "cancelled",
]


class VisitCaptureSourceBinding(BaseModel):
    model_config = {"extra": "forbid"}

    source_ref: str = Field(min_length=1, max_length=120)
    role: SourceRole
    captured_at: Optional[datetime] = None
    source_date: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")


class VisitCaptureFact(BaseModel):
    model_config = {"extra": "forbid"}

    text: str = Field(min_length=1, max_length=4000)
    provenance: List[str] = Field(min_length=1, max_length=8)
    verification: VerificationState = "source_stated"
    certainty: CertaintyState = "certain"


class VisitCaptureChange(VisitCaptureFact):
    direction: ChangeDirection


class VisitCaptureFinding(VisitCaptureFact):
    region: Optional[str] = Field(default=None, max_length=200)


class VisitCaptureCodingItem(BaseModel):
    model_config = {"extra": "forbid"}

    system: str = Field(default="ICD10", min_length=1, max_length=40)
    code: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=400)
    provenance: List[str] = Field(min_length=1, max_length=8)


class VisitCaptureCoding(BaseModel):
    model_config = {"extra": "forbid"}

    gesy_coded: List[VisitCaptureCodingItem] = Field(default_factory=list, max_length=50)
    coding_complete: bool = False


class VisitCaptureDecision(VisitCaptureFact):
    type: DecisionType
    responsible_role: Optional[str] = Field(default=None, max_length=120)


class VisitCaptureDuration(BaseModel):
    model_config = {"extra": "forbid"}

    value: float
    unit: str = Field(min_length=1, max_length=40)
    approximate: bool = False


class VisitCaptureMedication(BaseModel):
    model_config = {"extra": "forbid"}

    name: str = Field(min_length=1, max_length=240)
    action: MedicationAction
    dose: Optional[str] = Field(default=None, max_length=120)
    frequency: Optional[str] = Field(default=None, max_length=120)
    route: Optional[str] = Field(default=None, max_length=120)
    duration: Optional[VisitCaptureDuration] = None
    provenance: List[str] = Field(min_length=1, max_length=8)
    verification: VerificationState = "source_stated"
    certainty: CertaintyState = "certain"


class VisitCaptureExternalDependency(BaseModel):
    """Only the declared external actor and condition may be retained."""

    model_config = {"extra": "forbid"}

    actor: str = Field(min_length=1, max_length=80)
    condition: str = Field(min_length=1, max_length=240)


class VisitCapturePendingCandidate(BaseModel):
    model_config = {"extra": "forbid"}

    item_type: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=1000)
    trigger: Optional[str] = Field(default=None, max_length=1000)
    status: PendingStatus
    responsible_role: str = Field(min_length=1, max_length=120)
    external_dependency: Optional[VisitCaptureExternalDependency] = None
    review_date: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    provenance: List[str] = Field(min_length=1, max_length=8)


class VisitCaptureNextContact(BaseModel):
    model_config = {"extra": "forbid"}

    when: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    time: Optional[str] = Field(default=None, pattern=r"^\d{2}:\d{2}$")
    purpose: str = Field(min_length=1, max_length=1000)
    assess: List[str] = Field(default_factory=list, max_length=20)
    provenance: List[str] = Field(min_length=1, max_length=8)


class VisitCaptureComparisonBasis(BaseModel):
    model_config = {"extra": "forbid"}

    type: ComparisonType
    source_ref: Optional[str] = Field(default=None, max_length=120)


class VisitCaptureCandidateV1(BaseModel):
    model_config = {"extra": "forbid"}

    schema_version: Literal["visit_capture_candidate_v1"]
    source_identity_state: SourceIdentityState
    encounter_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    visit_type: Optional[str] = Field(default=None, max_length=240)
    specialty: Optional[str] = Field(default=None, max_length=240)
    reason_for_visit: VisitCaptureFact
    source_bindings: List[VisitCaptureSourceBinding] = Field(min_length=1, max_length=12)
    comparison_basis: VisitCaptureComparisonBasis
    what_changed: List[VisitCaptureChange] = Field(default_factory=list, max_length=40)
    findings: List[VisitCaptureFinding] = Field(default_factory=list, max_length=80)
    clinical_impression: List[VisitCaptureFact] = Field(default_factory=list, max_length=40)
    coding: VisitCaptureCoding = Field(default_factory=VisitCaptureCoding)
    decisions: List[VisitCaptureDecision] = Field(default_factory=list, max_length=60)
    medications: List[VisitCaptureMedication] = Field(default_factory=list, max_length=60)
    pending: List[VisitCapturePendingCandidate] = Field(default_factory=list, max_length=60)
    next_contact: Optional[VisitCaptureNextContact] = None
    safety_net: List[VisitCaptureFact] = Field(default_factory=list, max_length=20)
    uncertainties: List[VisitCaptureFact] = Field(default_factory=list, max_length=40)


class VisitCaptureContextCreate(BaseModel):
    model_config = {"extra": "forbid"}

    patient_id: str = Field(min_length=1, max_length=120)
    client_session_id: str = Field(min_length=16, max_length=120)


class VisitCaptureContextRecord(BaseModel):
    context_id: str
    patient_id: str
    expires_at: datetime


class VisitCaptureRequest(BaseModel):
    model_config = {"extra": "forbid"}

    context_id: str = Field(min_length=1, max_length=120)
    candidate: VisitCaptureCandidateV1


class VisitCapturePreview(BaseModel):
    patient_id: str
    context_id: str
    can_save: bool
    blocking_reason: Optional[str] = None
    normalized_candidate: Dict[str, Any]
    snapshot: str
    brief: str
    detail: str


class ClinicalPendingRecord(BaseModel):
    pending_id: str
    patient_id: str
    source_encounter_id: str
    item_type: str
    description: str
    trigger: Optional[str]
    status: str
    responsible_role: str
    external_dependency: Optional[VisitCaptureExternalDependency]
    review_date: Optional[str]
    provenance: List[str]
    resolution_event_id: Optional[str]
    created_at: datetime
    updated_at: datetime


class VisitCaptureSaveRecord(BaseModel):
    encounter: EncounterRecord
    pending: List[ClinicalPendingRecord]
    snapshot: str
    brief: str
    detail: str


class ClinicalStatus(BaseModel):
    database_dialect: str
    protected: bool


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def resolve_encounter_status(
    current_status: str,
    requested_status: Optional[str],
    *,
    content_changed: bool,
) -> str:
    """Preserve finalization semantics for server-persisted encounters.

    Draft encounters may move freely to draft/completed/amended. Once an
    encounter has been completed, later content changes are amendments and the
    record must never silently regress to draft. An already-amended encounter
    remains amended on subsequent saves so that the history is not made to look
    like an untouched original completion.
    """
    requested = requested_status or current_status

    if current_status == "draft":
        return requested

    if current_status == "amended":
        return "amended"

    # current_status == "completed"
    if content_changed or requested == "amended":
        return "amended"
    return "completed"


def _visit_capture_fact_text(items: List[Any]) -> str:
    return "; ".join(item.text for item in items if getattr(item, "text", "").strip())


def _visit_capture_medication_text(items: List[VisitCaptureMedication]) -> str:
    rendered: List[str] = []
    for item in items:
        parts = [item.name]
        if item.dose:
            parts.append(item.dose)
        if item.frequency:
            parts.append(item.frequency)
        if item.route:
            parts.append(item.route)
        if item.duration:
            approx = "~" if item.duration.approximate else ""
            parts.append(f"{approx}{item.duration.value:g} {item.duration.unit}")
        rendered.append(" · ".join(parts))
    return "; ".join(rendered)


def _visit_capture_pending_text(items: List[VisitCapturePendingCandidate]) -> str:
    rendered: List[str] = []
    for item in items:
        text = item.description
        if item.trigger:
            text += f" — {item.trigger}"
        if item.review_date:
            text += f" · review {item.review_date}"
        rendered.append(text)
    return "; ".join(rendered)


def _visit_capture_next_text(item: Optional[VisitCaptureNextContact]) -> str:
    if item is None:
        return ""
    when = " ".join(part for part in [item.when or "", item.time or ""] if part).strip()
    assess = "; ".join(item.assess)
    suffix = f" — {assess}" if assess else ""
    return f"{when + ' · ' if when else ''}{item.purpose}{suffix}"


def _render_visit_capture(candidate: VisitCaptureCandidateV1) -> tuple[str, str, str]:
    changed = _visit_capture_fact_text(candidate.what_changed)
    decisions = _visit_capture_fact_text(candidate.decisions)
    medications = _visit_capture_medication_text(candidate.medications)
    pending = _visit_capture_pending_text(candidate.pending)
    next_text = _visit_capture_next_text(candidate.next_contact)
    findings = _visit_capture_fact_text(candidate.findings)
    impressions = _visit_capture_fact_text(candidate.clinical_impression)
    safety = _visit_capture_fact_text(candidate.safety_net)
    uncertainties = _visit_capture_fact_text(candidate.uncertainties)

    snapshot_lines = [f"ΣΗΜΕΡΑ: {candidate.reason_for_visit.text}"]
    if changed:
        snapshot_lines.append(f"ΤΙ ΑΛΛΑΞΕ: {changed}")
    decision_parts = [part for part in [decisions, medications] if part]
    if decision_parts:
        snapshot_lines.append(f"ΑΠΟΦΑΣΗ: {'; '.join(decision_parts)}")
    if pending:
        snapshot_lines.append(f"ΕΚΚΡΕΜΕΙ: {pending}")
    if next_text:
        snapshot_lines.append(f"ΕΠΟΜΕΝΗ: {next_text}")
    snapshot = "\n".join(snapshot_lines)

    today_parts = [candidate.reason_for_visit.text]
    if changed:
        today_parts.append(changed)
    if findings:
        today_parts.append(f"Ευρήματα: {findings}")
    if impressions:
        today_parts.append(f"Κλινική εκτίμηση: {impressions}")
    brief_lines = [f"ΣΗΜΕΡΑ: {'; '.join(today_parts)}"]
    if decision_parts:
        brief_lines.append(f"ΑΠΟΦΑΣΗ: {'; '.join(decision_parts)}")
    if pending:
        brief_lines.append(f"ΕΚΚΡΕΜΕΙ: {pending}")
    if next_text:
        brief_lines.append(f"ΕΠΟΜΕΝΗ ΕΠΑΦΗ: {next_text}")
    if safety:
        brief_lines.append(f"SAFETY-NET: {safety}")
    brief = "\n\n".join(brief_lines)

    source_text = "; ".join(f"{item.role}: {item.source_ref}" for item in candidate.source_bindings)
    detail_lines = [
        f"ΗΜΕΡΟΜΗΝΙΑ: {candidate.encounter_date}",
        f"ΛΟΓΟΣ ΕΠΙΣΚΕΨΗΣ: {candidate.reason_for_visit.text}",
        f"ΠΗΓΕΣ: {source_text}",
        f"ΒΑΣΗ ΣΥΓΚΡΙΣΗΣ: {candidate.comparison_basis.type}",
    ]
    for label, text in [
        ("ΤΙ ΑΛΛΑΞΕ", changed),
        ("ΕΥΡΗΜΑΤΑ", findings),
        ("ΚΛΙΝΙΚΗ ΕΚΤΙΜΗΣΗ", impressions),
        ("ΑΠΟΦΑΣΕΙΣ", decisions),
        ("ΦΑΡΜΑΚΑ", medications),
        ("ΕΚΚΡΕΜΟΤΗΤΕΣ", pending),
        ("ΕΠΟΜΕΝΗ ΕΠΑΦΗ", next_text),
        ("SAFETY-NET", safety),
        ("ΑΒΕΒΑΙΟΤΗΤΕΣ", uncertainties),
    ]:
        if text:
            detail_lines.append(f"{label}: {text}")
    if candidate.coding.gesy_coded:
        coded = "; ".join(
            f"{item.system} {item.code} — {item.title}" for item in candidate.coding.gesy_coded
        )
        detail_lines.append(
            f"ΚΩΔΙΚΟΠΟΙΗΣΗ ΓΕΣΥ ({'πλήρης' if candidate.coding.coding_complete else 'μη πλήρης'}): {coded}"
        )
    return snapshot, brief, "\n".join(detail_lines)


def _validate_visit_capture_candidate(candidate: VisitCaptureCandidateV1) -> None:
    source_refs = {item.source_ref for item in candidate.source_bindings}
    if candidate.comparison_basis.type == "unavailable" and candidate.what_changed:
        raise HTTPException(
            status_code=422,
            detail="what_changed requires a safe comparison basis",
        )
    if candidate.comparison_basis.source_ref and candidate.comparison_basis.source_ref not in source_refs:
        raise HTTPException(status_code=422, detail="comparison source_ref is not bound")

    fact_groups: List[List[Any]] = [
        [candidate.reason_for_visit],
        candidate.what_changed,
        candidate.findings,
        candidate.clinical_impression,
        candidate.decisions,
        candidate.medications,
        candidate.pending,
        candidate.safety_net,
        candidate.uncertainties,
    ]
    if candidate.next_contact is not None:
        fact_groups.append([candidate.next_contact])
    fact_groups.append(candidate.coding.gesy_coded)
    for group in fact_groups:
        for item in group:
            provenance = getattr(item, "provenance", [])
            unknown = [ref for ref in provenance if ref not in source_refs]
            if unknown:
                raise HTTPException(
                    status_code=422,
                    detail=f"unbound provenance reference: {unknown[0]}",
                )


def _pending_record(row: ClinicalPendingORM) -> ClinicalPendingRecord:
    return ClinicalPendingRecord(
        pending_id=row.id,
        patient_id=row.patient_id,
        source_encounter_id=row.source_encounter_id,
        item_type=row.item_type,
        description=row.description,
        trigger=row.trigger_text,
        status=row.status,
        responsible_role=row.responsible_role,
        external_dependency=row.external_dependency_json,
        review_date=row.review_date,
        provenance=list(row.provenance_json or []),
        resolution_event_id=row.resolution_event_id,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


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
        offset: int = Query(default=0, ge=0),
    ) -> List[PatientSummary]:
        q = query.strip()
        with Session(engine) as session:
            stmt = select(PatientORM).order_by(
                PatientORM.updated_at.desc(), PatientORM.patient_id.asc()
            )
            if not q:
                # Preserve existing bounded legacy listing for other consumers.
                # Visit Capture does not request a listing until text is entered.
                patients = session.execute(stmt.offset(offset).limit(limit)).scalars().all()
            else:
                # The match runs against the WHOLE registry, not a preloaded slice.
                # Python Unicode folding makes accented Greek name searches work
                # consistently on both SQLite and PostgreSQL JSON demographics.
                # Yield rows in batches; keep only the requested results page.
                terms = _fold_patient_lookup(q).split()
                matched = 0
                patients = []
                for patient in session.scalars(stmt).yield_per(200):
                    if not _patient_matches_search(patient, terms):
                        continue
                    if matched >= offset:
                        patients.append(patient)
                        if len(patients) >= limit:
                            break
                    matched += 1
            return [patient_summary(session, patient) for patient in patients]

    @router.get(
        "/recent-encounters",
        response_model=List[RecentEncounterSummary],
        dependencies=protected,
    )
    def recent_encounters(limit: int = Query(default=3, ge=1, le=3)) -> List[RecentEncounterSummary]:
        # Read only three real completed/amended encounters with an existing
        # registered patient. No unsigned drafts or clinical payload in response.
        with Session(engine) as session:
            rows = session.execute(
                select(EncounterORM, PatientORM)
                .join(PatientORM, PatientORM.patient_id == EncounterORM.patient_id)
                .where(EncounterORM.status.in_(("completed", "amended")))
                .order_by(
                    EncounterORM.encounter_date.desc(),
                    EncounterORM.created_at.desc(),
                    EncounterORM.id.asc(),
                )
                .limit(limit)
            ).all()
            results = []
            for encounter, patient in rows:
                demographics = patient.demographics_json
                if not isinstance(demographics, dict):
                    demographics = {}
                display_name = ""
                for key in ("full_name", "fullName", "name", "ονοματεπώνυμο", "ονοματεπωνυμο"):
                    value = demographics.get(key)
                    if isinstance(value, str) and value.strip():
                        display_name = value.strip()
                        break
                if not display_name:
                    first = next((
                        demographics[key].strip()
                        for key in ("first_name", "firstName", "όνομα", "ονομα")
                        if isinstance(demographics.get(key), str) and demographics[key].strip()
                    ), "")
                    last = next((
                        demographics[key].strip()
                        for key in ("last_name", "lastName", "επώνυμο", "επωνυμο")
                        if isinstance(demographics.get(key), str) and demographics[key].strip()
                    ), "")
                    display_name = " ".join(part for part in (first, last) if part)
                if not display_name:
                    display_name = "Ασθενής χωρίς καταχωρισμένο όνομα"

                visit_type = "Κλινική επίσκεψη"
                payload = encounter.payload_json
                if isinstance(payload, dict):
                    signed = payload.get("_visit_capture_v1")
                    if isinstance(signed, dict) and signed.get("signed") is True:
                        candidate = signed.get("candidate")
                        if isinstance(candidate, dict):
                            label = candidate.get("visit_type")
                            if isinstance(label, str) and label.strip():
                                visit_type = label.strip()[:240]
                results.append(RecentEncounterSummary(
                    patient_id=encounter.patient_id,
                    patient_display_name=display_name[:200],
                    encounter_date=encounter.encounter_date,
                    visit_type=visit_type,
                ))
            return results

    @router.get("/patient/{patient_id}", response_model=PatientSummary, dependencies=protected)
    def get_patient(patient_id: str) -> PatientSummary:
        with Session(engine) as session:
            patient = ensure_patient(session, patient_id)
            return patient_summary(session, patient)

    def visit_capture_context(
        session: Session,
        context_id: str,
    ) -> VisitCaptureContextORM:
        context = session.get(VisitCaptureContextORM, context_id)
        if context is None:
            raise HTTPException(status_code=409, detail="Visit Capture context is stale or invalid")
        now = utcnow()
        if context.expires_at <= now:
            session.delete(context)
            session.commit()
            raise HTTPException(status_code=409, detail="Visit Capture context expired")
        patient = ensure_patient(session, context.patient_id)
        if patient.updated_at != context.patient_updated_at:
            raise HTTPException(status_code=409, detail="Patient context changed; confirm patient again")
        return context

    @router.post(
        "/visit-capture/context",
        response_model=VisitCaptureContextRecord,
        dependencies=protected,
    )
    def create_visit_capture_context(req: VisitCaptureContextCreate) -> VisitCaptureContextRecord:
        now = utcnow()
        with Session(engine) as session:
            patient = ensure_patient(session, req.patient_id.strip())
            session.execute(
                delete(VisitCaptureContextORM).where(
                    VisitCaptureContextORM.client_session_id == req.client_session_id
                )
            )
            context = VisitCaptureContextORM(
                id=str(uuid4()),
                client_session_id=req.client_session_id,
                patient_id=patient.patient_id,
                patient_updated_at=patient.updated_at,
                issued_at=now,
                expires_at=now + timedelta(minutes=30),
            )
            session.add(context)
            session.commit()
            session.refresh(context)
            return VisitCaptureContextRecord(
                context_id=context.id,
                patient_id=context.patient_id,
                expires_at=context.expires_at,
            )

    @router.post(
        "/visit-capture/preview",
        response_model=VisitCapturePreview,
        dependencies=protected,
    )
    def preview_visit_capture(req: VisitCaptureRequest) -> VisitCapturePreview:
        _validate_visit_capture_candidate(req.candidate)
        with Session(engine) as session:
            context = visit_capture_context(session, req.context_id)
            snapshot, brief, detail = _render_visit_capture(req.candidate)
            blocked = req.candidate.source_identity_state != "consistent"
            return VisitCapturePreview(
                patient_id=context.patient_id,
                context_id=context.id,
                can_save=not blocked,
                blocking_reason=(
                    "Source identity must be resolved before Save" if blocked else None
                ),
                normalized_candidate=req.candidate.model_dump(mode="json", exclude_none=False),
                snapshot=snapshot,
                brief=brief,
                detail=detail,
            )

    @router.post(
        "/visit-capture/save",
        response_model=VisitCaptureSaveRecord,
        dependencies=protected,
    )
    def save_visit_capture(req: VisitCaptureRequest) -> VisitCaptureSaveRecord:
        _validate_visit_capture_candidate(req.candidate)
        if req.candidate.source_identity_state != "consistent":
            raise HTTPException(
                status_code=409,
                detail="Source identity conflict/uncertainty must be resolved before Save",
            )
        now = utcnow()
        with Session(engine) as session:
            context = visit_capture_context(session, req.context_id)
            consumed = session.execute(
                delete(VisitCaptureContextORM).where(
                    VisitCaptureContextORM.id == context.id
                )
            )
            if consumed.rowcount != 1:
                session.rollback()
                raise HTTPException(status_code=409, detail="Visit Capture context already consumed")

            normalized = req.candidate.model_dump(mode="json", exclude_none=False)
            row = EncounterORM(
                id=str(uuid4()),
                patient_id=context.patient_id,
                encounter_date=req.candidate.encounter_date,
                status="completed",
                payload_json={
                    "_visit_capture_v1": {
                        "schema_version": "1",
                        "signed": True,
                        "signed_at": now.isoformat(),
                        "candidate": normalized,
                    }
                },
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            pending_rows: List[ClinicalPendingORM] = []
            for item in req.candidate.pending:
                pending_row = ClinicalPendingORM(
                    id=str(uuid4()),
                    patient_id=context.patient_id,
                    source_encounter_id=row.id,
                    item_type=item.item_type,
                    description=item.description,
                    trigger_text=item.trigger,
                    status=item.status,
                    responsible_role=item.responsible_role,
                    external_dependency_json=(item.external_dependency.model_dump(mode="json") if item.external_dependency else None),
                    review_date=item.review_date,
                    provenance_json=item.provenance,
                    resolution_event_id=None,
                    created_at=now,
                    updated_at=now,
                )
                session.add(pending_row)
                pending_rows.append(pending_row)

            patient = ensure_patient(session, context.patient_id)
            patient.updated_at = now
            session.add(patient)
            session.commit()
            session.refresh(row)
            for pending_row in pending_rows:
                session.refresh(pending_row)

            snapshot, brief, detail = _render_visit_capture(req.candidate)
            encounter = EncounterRecord(
                encounter_id=row.id,
                patient_id=row.patient_id,
                encounter_date=row.encounter_date,
                status=row.status,
                payload=row.payload_json or {},
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
            return VisitCaptureSaveRecord(
                encounter=encounter,
                pending=[_pending_record(item) for item in pending_rows],
                snapshot=snapshot,
                brief=brief,
                detail=detail,
            )

    @router.get(
        "/patient/{patient_id}/pending",
        response_model=List[ClinicalPendingRecord],
        dependencies=protected,
    )
    def list_pending(patient_id: str) -> List[ClinicalPendingRecord]:
        with Session(engine) as session:
            ensure_patient(session, patient_id)
            rows = session.execute(
                select(ClinicalPendingORM)
                .where(ClinicalPendingORM.patient_id == patient_id)
                .order_by(ClinicalPendingORM.created_at.desc())
            ).scalars().all()
            return [_pending_record(row) for row in rows]

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

            current_status = row.status or "draft"
            existing_capture = (row.payload_json or {}).get("_visit_capture_v1")
            if isinstance(existing_capture, dict) and existing_capture.get("signed") is True:
                changed = (
                    (req.encounter_date is not None and req.encounter_date != row.encounter_date)
                    or (req.status is not None and req.status != current_status)
                    or (req.payload is not None and req.payload != (row.payload_json or {}))
                )
                if changed:
                    raise HTTPException(
                        status_code=409,
                        detail="Signed Visit Capture encounters are immutable; amendment history is not enabled yet",
                    )
                return EncounterRecord(
                    encounter_id=row.id,
                    patient_id=row.patient_id,
                    encounter_date=row.encounter_date,
                    status=row.status,
                    payload=row.payload_json or {},
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )

            content_changed = False

            if req.encounter_date is not None and req.encounter_date != row.encounter_date:
                content_changed = True
                row.encounter_date = req.encounter_date
            if req.payload is not None and req.payload != (row.payload_json or {}):
                content_changed = True
                row.payload_json = req.payload

            row.status = resolve_encounter_status(
                current_status,
                req.status,
                content_changed=content_changed,
            )
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
