"""#161 controlled, persistent test scope in the existing Cockpit and clinical store.

No provider traffic, user-supplied seed content, new credentials, DDL or clinical
attestation. Existing Clinical Key grants test access, never an individual identity.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import secrets
from typing import Literal
from pydantic import BaseModel, ConfigDict

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from clinical_calendar import ActualBooking, ClinicalAppointmentORM
from clinical_data import (
    SYNTHETIC_161_PATIENT_IDS, PatientORM, EncounterORM, ClinicalPendingORM,
    VisitCaptureCandidateV1, _render_visit_capture, _patient_matches_search,
    _fold_patient_lookup,
)
from clinical_staff_auth import StaffActor
from clinical_attention_phase_a import SyntheticAttentionStore, build_synthetic_attention_router
from clinical_followthrough_demo import build_synthetic_followthrough_router
from clinical_visit_brief import (attach_visit_brief_routes, BookingIntent,
                                  _minimal_brief, _booking_card, _record_access, VisitBriefStaffAuditORM)

PREFIX = "/clinical/synthetic-161"
SCENARIO = "cockpit-161-v1"
BOOKINGS = tuple(f"synthetic-161:{i}" for i in range(1, 4))
ROOT = Path(__file__).resolve().parent


def require_test_access(
    x_clinical_key: str | None = Header(default=None, alias="X-Clinical-Key"),
) -> StaffActor:
    expected = os.environ.get("CLINICAL_DATA_KEY", "")
    if not expected:
        raise HTTPException(503, "Clinical access is not configured")
    if not x_clinical_key or not secrets.compare_digest(x_clinical_key, expected):
        raise HTTPException(401, "Existing Clinical Key required")
    return StaffActor("synthetic-key-test", "synthetic_tester")


def _test_patient(db, patient_id):
    if patient_id not in SYNTHETIC_161_PATIENT_IDS:
        raise HTTPException(404, "Patient outside test scenario")
    row = db.get(PatientORM, patient_id)
    if (not row or not isinstance(row.demographics_json, dict)
            or row.demographics_json.get("synthetic_scenario") != SCENARIO):
        raise HTTPException(409, "Prepare the synthetic scenario first")
    return row


def _candidate(day, first=False):
    fact = lambda text: {"text": text, "provenance": ["synthetic-seed"]}
    return VisitCaptureCandidateV1.model_validate({
        "schema_version": "visit_capture_candidate_v1",
        "source_identity_state": "consistent", "encounter_date": day,
        "visit_type": "Συνθετική προηγούμενη επίσκεψη",
        "reason_for_visit": fact("Δοκιμαστική αξιολόγηση πόνου γόνατος"),
        "source_bindings": [{"source_ref": "synthetic-seed", "role": "clinician_edit"}],
        "comparison_basis": {"type": "unavailable"},
        "findings": [fact("Συνθετικό εύρημα: περιορισμός κάμψης γόνατος")],
        "clinical_impression": [fact("Συνθετική κλινική εκτίμηση για δοκιμή συνέχειας")],
        "decisions": [{**fact("Συμφωνήθηκε επανέλεγχος και προσκόμιση προηγούμενης μαγνητικής"),
                       "type": "advice"}],
        "medications": [] if first else [{
            "name": "Δοκιμαστική αγωγή", "action": "continued",
            "provenance": ["synthetic-seed"]}],
        "pending": [] if first else [{
            "item_type": "imaging", "description": "Να προσκομιστεί το CD της μαγνητικής",
            "status": "awaiting_patient", "responsible_role": "patient",
            "provenance": ["synthetic-seed"]}],
        "safety_net": [fact("Συνθετικές οδηγίες επανεπικοινωνίας καταγεγραμμένες στην επίσκεψη")],
        "uncertainties": [fact("Η παραλαβή και η εξέταση των εικόνων δεν έχουν επιβεβαιωθεί")],
    })


def _prepare(engine):
    # Fixed keys + a single transaction: repeat/restart/concurrent requests cannot
    # duplicate or reset records, confirmations, dates or audit history.
    with Session(engine, autoflush=False) as db:
        for i, patient_id in enumerate(SYNTHETIC_161_PATIENT_IDS, 1):
            existing = db.get(PatientORM, patient_id)
            if existing:
                _test_patient(db, patient_id)
                continue
            stamp = datetime(2026, 10, 17, 8)
            name = f"Δοκιμαστικός Ασθενής 161 {i}"
            db.add(PatientORM(patient_id=patient_id, demographics_json={
                "full_name": name, "dob": f"197{i}-01-01",
                "synthetic_scenario": SCENARIO,
            }, created_at=stamp, updated_at=stamp))
            start = stamp + timedelta(hours=i)
            if db.get(ClinicalAppointmentORM, BOOKINGS[i-1]):
                raise HTTPException(409, "Synthetic booking identifier collision")
            db.add(ClinicalAppointmentORM(
                id=BOOKINGS[i-1], source=SCENARIO, source_appointment_id=BOOKINGS[i-1],
                start_at=start, end_at=start + timedelta(minutes=30),
                duration_minutes=30, clinic="Συνθετικό ιατρείο", category="other",
                patient_display_name=name, phone_e164="", linked_patient_id=None,
                label="Δοκιμή #161", comment="Επανέλεγχος γόνατος και μαγνητικής",
                status="scheduled", updated_at=stamp,
            ))
            for days in (30, 10):
                signed_at = stamp - timedelta(days=days)
                candidate = _candidate(signed_at.date().isoformat(), first=days == 30)
                encounter_id = f"{patient_id}-visit-{days}"
                if db.get(EncounterORM, encounter_id):
                    raise HTTPException(409, "Synthetic encounter identifier collision")
                snapshot, brief, detail = _render_visit_capture(candidate)
                db.add(EncounterORM(
                    id=encounter_id, patient_id=patient_id,
                    encounter_date=candidate.encounter_date, status="completed",
                    payload_json={"_visit_capture_v1": {
                        "signed": True, "signed_at": signed_at.isoformat(),
                        "synthetic_scenario": SCENARIO,
                        "candidate": candidate.model_dump(mode="json"),
                        "snapshot": snapshot, "brief": brief, "detail": detail,
                    }}, created_at=signed_at, updated_at=signed_at,
                ))
            pending_id = f"{patient_id}-pending"
            if db.get(ClinicalPendingORM, pending_id):
                raise HTTPException(409, "Synthetic pending identifier collision")
            db.add(ClinicalPendingORM(
                id=pending_id, patient_id=patient_id,
                source_encounter_id=f"{patient_id}-visit-10", item_type="imaging",
                description="Να προσκομιστεί το CD της μαγνητικής", status="awaiting_patient",
                responsible_role="patient", provenance_json=["synthetic-seed"],
                created_at=stamp - timedelta(days=10), updated_at=stamp - timedelta(days=10),
            ))
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            # Another browser may have won the entire transaction. Only accept
            # the existing marked subset; never overwrite a conflicting record.
            for patient_id in SYNTHETIC_161_PATIENT_IDS:
                _test_patient(db, patient_id)


class PendingChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["awaiting_patient", "resolved"]
    expected_updated_at: datetime


def _pending_view(p, db):
    source = db.get(EncounterORM, p.source_encounter_id)
    if (p.id != p.patient_id + "-pending" or not source or source.patient_id != p.patient_id
            or source.payload_json.get("_visit_capture_v1", {}).get("synthetic_scenario") != SCENARIO):
        raise HTTPException(409, "Unexpected pending source")
    # The fixture has exactly one pending row per patient. Existing synthetic
    # status audit therefore identifies this obligation without another store.
    events = [{"id": p.id + "-created", "kind": "created", "at": p.created_at.isoformat()}]
    for event in db.scalars(select(VisitBriefStaffAuditORM).where(
        VisitBriefStaffAuditORM.patient_id == p.patient_id,
        VisitBriefStaffAuditORM.principal_id == "synthetic-key-test",
        VisitBriefStaffAuditORM.event.in_(("synthetic_pending_resolved", "synthetic_pending_awaiting_patient")),
        VisitBriefStaffAuditORM.outcome == "allowed",
    ).order_by(VisitBriefStaffAuditORM.occurred_at, VisitBriefStaffAuditORM.id)):
        events.append({"id": event.id, "kind": "received" if event.event.endswith("_resolved") else "reopened",
                       "at": event.occurred_at.isoformat()})
    return {"pending_id": p.id, "description": p.description, "status": p.status,
            "created_at": p.created_at.isoformat(), "updated_at": p.updated_at.isoformat(),
            "source_encounter_id": p.source_encounter_id, "source_date": source.encounter_date,
            "history": events}


def build_synthetic_161_router(engine):
    router = APIRouter(prefix=PREFIX, tags=["cockpit-synthetic-161"])
    protected = [Depends(require_test_access)]
    attention = SyntheticAttentionStore()

    def verify_synthetic_patient(patient_id, dob=None):
        with Session(engine) as db:
            p = _test_patient(db, patient_id)
            if dob is not None and p.demographics_json.get("dob") != dob:
                raise HTTPException(409, "Synthetic identifiers do not match")
            return p

    def synthetic_patients():
        with Session(engine) as db:
            return [{"patient_id":p.patient_id,"name":p.demographics_json["full_name"]}
                    for p in (_test_patient(db,pid) for pid in SYNTHETIC_161_PATIENT_IDS)]

    router.include_router(build_synthetic_attention_router(
        attention, require_test_access, verify_synthetic_patient, synthetic_patients))

    def verified_fixture_pending():
        with Session(engine) as db:
            patient_id = SYNTHETIC_161_PATIENT_IDS[0]
            _test_patient(db, patient_id)
            pending = db.get(ClinicalPendingORM, patient_id + "-pending")
            source = db.get(EncounterORM, pending.source_encounter_id) if pending else None
            signed = (source.payload_json or {}).get("_visit_capture_v1", {}) if source else {}
            if (not pending or pending.patient_id != patient_id
                    or not source or source.patient_id != patient_id
                    or source.status != "completed" or not signed.get("signed")
                    or signed.get("synthetic_scenario") != SCENARIO):
                raise HTTPException(409, "Signed synthetic Pending source unavailable")
            return pending.id, source.id

    followthrough_router, followthrough = build_synthetic_followthrough_router(
        require_test_access, verified_fixture_pending)
    router.include_router(followthrough_router)


    def booking(db, uid):
        if uid not in BOOKINGS:
            raise HTTPException(404, "Booking outside test scenario")
        for patient_id in SYNTHETIC_161_PATIENT_IDS:
            _test_patient(db, patient_id)
        row = db.get(ClinicalAppointmentORM, uid)
        if not row or row.source != SCENARIO:
            raise HTTPException(409, "Prepare the synthetic scenario first")
        return ActualBooking(
            uid=row.id, start_at=row.start_at.replace(tzinfo=timezone.utc),
            end_at=row.end_at.replace(tzinfo=timezone.utc), clinic=row.clinic,
            patient_display_name=row.patient_display_name, reason=row.comment, category="other",
        )

    async def fresh_booking(uid):
        with Session(engine) as db:
            return booking(db, uid)

    # Public shell contains no clinical content. All data and seed requests
    # require the existing session/key; no extra login or secret is introduced.
    @router.get("/", response_class=HTMLResponse, include_in_schema=False)
    def shell():
        text = (ROOT / "static/cockpit/index.html").read_text()
        text = text.replace("<head>", '<head><link rel="stylesheet" href="/static/cockpit/followthrough-synthetic.css?v=20261031-one-cycle" /><script>window.CockpitClinicalBase="' + PREFIX +
                            '";window.CockpitSynthetic=true;</script>')
        text = text.replace('<div id="doctorMainView">', '''
        <section class="clinical-identity-note" aria-label="Δοκιμή #161">
          <strong>Δοκιμή #161 · Μόνο συνθετικοί ασθενείς</strong>
          <p>Σενάριο 17/10/2026 · Επίλεξε ραντεβού για την προετοιμασία της επίσκεψης.</p>
          <button id="synthetic161Prepare" type="button">Προετοιμασία δοκιμής #161</button>
          <span id="synthetic161Status" role="status"></span>
          <section id="syntheticFollowthroughDemo" aria-label="Πλήρης κύκλος εξέτασης · συνθετική δοκιμή" hidden></section>
          <a href="/static/cockpit/">Επιστροφή στο Cockpit</a>
        </section><div id="doctorMainView">''')
        text = text.replace("Έλεγξα ανεξάρτητα τα στοιχεία αυτού του προσώπου στον προστατευμένο φάκελο. Το όνομα ή το UID του ραντεβού δεν αρκούν.",
                            "Έλεγξα το όνομα και την ημερομηνία γέννησης του επιλεγμένου δοκιμαστικού ασθενούς. Αυτή είναι μόνο δοκιμαστική επιβεβαίωση.")
        text = text.replace('<p class="clinical-muted">Πλήρης κλινική σύνοψη και προηγούμενο σχέδιο εμφανίζονται μόνο μετά από επιβεβαιωμένη σύνδεση με τον προστατευμένο φάκελο.</p>', "")
        text = text.replace("</body>", '<script src="/static/cockpit/synthetic-161.js"></script><script src="/static/cockpit/followthrough-synthetic.js"></script></body>')
        return HTMLResponse(text, headers={"Cache-Control": "no-store"})

    @router.post("/prepare", dependencies=protected)
    def prepare():
        _prepare(engine)
        attention.establish_previously_verified_referral(verify_synthetic_patient)
        return {"ready": True, "storage": engine.dialect.name, "scenario": SCENARIO}

    @router.post("/open-booking")
    def open_booking(req: BookingIntent, actor=Depends(require_test_access)):
        with Session(engine) as db:
            actual = booking(db, req.booking_uid)
            # Fixture identity is declared by exact scenario keys, never by name
            # matching or unqualified external-calendar UID continuity.
            patient_id = SYNTHETIC_161_PATIENT_IDS[BOOKINGS.index(req.booking_uid)]
            patient = _test_patient(db, patient_id)
            brief = _minimal_brief(db, patient_id, actual)
            brief["clinical_attention"] = attention.brief(patient_id)
            # Presentation only: cycle events do NOT mutate signed visit narrative.
            brief["followthrough"] = [
                followthrough.view(case_id)
                for case_id, work in followthrough.works.items()
                if work.pending_ref == patient_id + "-pending"
                and work.visit_ref in {visit_id for visit_id in
                                       [patient_id + "-visit-10", patient_id + "-visit-30"]}
            ]
            brief["pending"] = [_pending_view(p, db) for p in db.scalars(
                select(ClinicalPendingORM).where(ClinicalPendingORM.patient_id == patient_id, ClinicalPendingORM.status != "deleted"))]
            _record_access(db, actor, req.booking_uid, "synthetic_brief_read", "allowed", patient_id)
            db.commit()
            return {"mode": "synthetic_fixture", "booking": _booking_card(actual),
                    "patient_name": patient.demographics_json["full_name"], "brief": brief}

    @router.post("/patient/{patient_id}/pending/{pending_id}")
    def update_pending(patient_id: str, pending_id: str, req: PendingChange,
                       actor=Depends(require_test_access)):
        with Session(engine) as db:
            _test_patient(db, patient_id)
            row = db.get(ClinicalPendingORM, pending_id)
            if (pending_id != patient_id + "-pending" or not row or row.patient_id != patient_id
                    or row.source_encounter_id != patient_id + "-visit-10"):
                raise HTTPException(404, "Pending outside test scenario")
            expected = req.expected_updated_at
            if expected.tzinfo is not None:
                expected = expected.astimezone(timezone.utc).replace(tzinfo=None)
            stamp = datetime.now(timezone.utc).replace(tzinfo=None)
            changed = db.execute(update(ClinicalPendingORM).where(
                ClinicalPendingORM.id == pending_id, ClinicalPendingORM.updated_at == expected,
                ClinicalPendingORM.status != "deleted"
            ).values(status=req.status, updated_at=stamp))
            if changed.rowcount != 1:
                raise HTTPException(409, "Pending changed; reload its current state")
            _record_access(db, actor, BOOKINGS[SYNTHETIC_161_PATIENT_IDS.index(patient_id)],
                           "synthetic_pending_" + req.status, "allowed", patient_id)
            db.commit()
            db.refresh(row)
            return _pending_view(row, db)

    @router.get("/staff-session", dependencies=protected)
    def test_session():
        return {"staff_role": "synthetic_tester", "principal_id": "synthetic-key-test"}

    @router.get("/calendar/cockpit-context", dependencies=protected)
    def context():
        with Session(engine) as db:
            rows = [booking(db, uid) for uid in BOOKINGS]
        def card(row):
            return {**row.model_dump(), "appointment_id": row.uid}
        return {"generated_at": rows[1].start_at + timedelta(minutes=15),
                "source_updated_at": datetime.now(timezone.utc), "today_total": 3,
                "synthetic_scenario": SCENARIO, "current_conflict_count": 1, "previous": card(rows[0]),
                "current": card(rows[1]), "next": card(rows[2]), "upcoming_today": [card(rows[2])]}

    @router.get("/patients", dependencies=protected)
    def patients(query: str = "", limit: int = 20, offset: int = 0):
        with Session(engine) as db:
            rows = [_test_patient(db, pid) for pid in SYNTHETIC_161_PATIENT_IDS]
            terms = _fold_patient_lookup(query).split()
            return [{"patient_id": p.patient_id, "demographics": p.demographics_json}
                    for p in rows if terms and _patient_matches_search(p, terms)][max(0, offset):][:min(20, max(1, limit))]

    @router.get("/recent-encounters", dependencies=protected)
    def recent():
        with Session(engine) as db:
            results = []
            for pid in SYNTHETIC_161_PATIENT_IDS:
                patient = _test_patient(db, pid)
                visit = db.scalar(select(EncounterORM).where(EncounterORM.patient_id == pid)
                                  .order_by(EncounterORM.encounter_date.desc()).limit(1))
                if visit:
                    results.append({"patient_id": pid, "patient_display_name": patient.demographics_json["full_name"],
                                    "encounter_date": visit.encounter_date, "visit_type": "Συνθετική προηγούμενη επίσκεψη"})
            return results

    @router.get("/clinical-updates", dependencies=protected)
    def clinical_updates():
        with Session(engine) as db:
            items = []
            for patient_id in SYNTHETIC_161_PATIENT_IDS:
                patient = _test_patient(db, patient_id)
                for pending in db.scalars(select(ClinicalPendingORM).where(
                        ClinicalPendingORM.patient_id == patient_id, ClinicalPendingORM.status != "deleted")):
                    items.append({**_pending_view(pending, db), "patient_id": patient_id,
                                  "patient_name": patient.demographics_json["full_name"]})
            items.sort(key=lambda p: (p["updated_at"], p["pending_id"]), reverse=True)
            return {"items": items, "open_count": sum(p["status"] in
                    ("open", "awaiting_patient", "awaiting_result") for p in items)}

    @router.get("/patient/{patient_id}/chart", dependencies=protected)
    def chart(patient_id: str):
        with Session(engine) as db:
            patient = _test_patient(db, patient_id)
            visits = db.scalars(select(EncounterORM).where(EncounterORM.patient_id == patient_id)
                                .order_by(EncounterORM.encounter_date.desc())).all()
            pending = db.scalars(select(ClinicalPendingORM).where(ClinicalPendingORM.patient_id == patient_id, ClinicalPendingORM.status != "deleted")).all()
            history = []
            for row in visits:
                signed = row.payload_json.get("_visit_capture_v1", {})
                if signed.get("synthetic_scenario") != SCENARIO:
                    raise HTTPException(409, "Unexpected non-synthetic chart content")
                candidate = VisitCaptureCandidateV1.model_validate(signed["candidate"])
                snapshot, brief, detail = _render_visit_capture(candidate)
                history.append({"encounter_id": row.id, "date": row.encounter_date,
                                "snapshot": snapshot, "brief": brief, "detail": detail,
                                "clinical": candidate.model_dump(mode="json")})
            pending_ids = {p.id for p in pending}
            return {"patient_id": patient_id, "demographics": patient.demographics_json,
                    "visits": history, "pending": [
                        _pending_view(p, db) for p in pending],
                    "followthrough": [
                        followthrough.view(case_id)
                        for case_id, work in followthrough.works.items()
                        if work.pending_ref in pending_ids
                    ]}

    scoped = APIRouter(prefix="/calendar")
    attach_visit_brief_routes(
        scoped, engine, protected, enabled=lambda: True,
        actor_dependency=require_test_access, clinician_dependency=require_test_access,
        booking_loader=fresh_booking, namespace=SCENARIO,
        patient_ids=SYNTHETIC_161_PATIENT_IDS, clinician_role="synthetic_tester",
    )
    router.include_router(scoped)
    return router
