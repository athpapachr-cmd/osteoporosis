"""Clinical Pending checkpoint owner, extended from #166; explicit schema installation.
Normal chart writes reuse protected clinical access; staff sessions retain strict role checks.
"""
from __future__ import annotations

import os
import secrets
from typing import Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Integer, JSON, String, UniqueConstraint, select, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Session

from clinical_staff_auth import StaffActor, require_staff_clinician, STAFF_COOKIE
from clinical_data import (
    ClinicalPendingORM, EncounterORM, PatientORM, utcnow,
)


class PendingEventBase(DeclarativeBase):
    pass


class PendingCheckpointEventORM(PendingEventBase):
    __tablename__ = "clinical_pending_checkpoint_events"
    __table_args__ = (UniqueConstraint("pending_id", "revision", name="uq_clinical_pending_checkpoint_event_revision"),)

    id = Column(String(36), primary_key=True)
    idempotency_key = Column(String(100), nullable=False, unique=True)
    pending_id = Column(String(36), nullable=False, index=True)
    patient_id = Column(String(120), nullable=False, index=True)
    checkpoint_key = Column(String(64), nullable=False)
    plan_version = Column(Integer, nullable=False)
    kind = Column(String(32), nullable=False)
    retracts_event_id = Column(String(36), nullable=True)
    revision = Column(Integer, nullable=False)
    recorded_at = Column(DateTime, nullable=False)
    actor_principal_id = Column(String(64), nullable=False)
    actor_role = Column(String(32), nullable=False)
    plan_json = Column(JSON, nullable=True)


class CheckpointAction(BaseModel):
    model_config = {"extra": "forbid"}
    checkpoint_key: str = Field(min_length=1, max_length=64)
    observation: Literal["yes_confirmed", "not_yet_observed", "retracted"]
    expected_revision: int = Field(ge=0)
    idempotency_key: str = Field(min_length=16, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    retracts_event_id: Optional[str] = Field(default=None, max_length=36)



class VisitCaptureCheckpointStep(BaseModel):
    key: str
    question: str
    meaning: str
    required: Literal[True] = True
    attesting_role: Literal["clinician"] = "clinician"

class VisitCaptureCheckpointPlan(BaseModel):
    version: Literal[1] = 1
    steps: list[VisitCaptureCheckpointStep]

PLANS = {
    "material_images": [("received", "Παραλήφθηκε το CD;", "Παραλήφθηκε το CD"),
                        ("images_reviewed", "Εξετάστηκαν οι εικόνες;", "Εξετάστηκαν οι εικόνες")],
    "document": [("received", "Παραλήφθηκε το έγγραφο;", "Παραλήφθηκε το έγγραφο"),
                 ("report_read", "Διαβάστηκε το πόρισμα;", "Διαβάστηκε το πόρισμα")],
    "completion": [("completed", "Ολοκληρώθηκε η ενέργεια;", "Ολοκληρώθηκε η ενέργεια")],
}

class PendingLifecycleAction(BaseModel):
    model_config = {"extra": "forbid"}
    action: Literal["cancel", "restore"]
    expected_revision: int = Field(ge=0)
    cancellation_event_id: str | None = Field(default=None, max_length=36)
    idempotency_key: str = Field(min_length=16, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")

class AdoptPlan(BaseModel):
    model_config = {"extra": "forbid"}
    expected_revision: int = Field(default=0, ge=0)
    template: Literal["material_images", "document", "completion"]
    first_checkpoint_key: str | None = Field(default=None,max_length=64)
    first_observation: Literal["yes_confirmed", "not_yet_observed"] | None = None
    idempotency_key: str = Field(min_length=16, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")

def installed(engine):
    return inspect(engine).has_table(PendingCheckpointEventORM.__tablename__)

def actor_for(request, patient):
    # Presence, not truthiness: an invalid/empty staff cookie never falls back.
    if STAFF_COOKIE in request.cookies:
        return require_staff_clinician(request)
    expected = os.environ.get("CLINICAL_DATA_KEY", "")
    supplied = request.headers.get("x-clinical-key", "")
    if expected and secrets.compare_digest(supplied, expected):
        return StaffActor("shared-clinical-key", "shared_clinical_key")
    raise HTTPException(401, "Clinical sign-in required")

def can_write(request, patient):
    try:
        actor_for(request, patient)
        return True
    except HTTPException:
        return False

def _signed_source(session, row):
    source=session.get(EncounterORM,row.source_encounter_id)
    signed=(source.payload_json or {}).get("_visit_capture_v1",{}) if source else {}
    if not source or source.patient_id!=row.patient_id or source.status not in ("completed","amended") or not signed.get("signed") or not signed.get("signed_at"):
        raise HTTPException(409,"Pending source is not signed")

def _events(session,row):
    return session.scalars(select(PendingCheckpointEventORM).where(
        PendingCheckpointEventORM.pending_id==row.id,PendingCheckpointEventORM.patient_id==row.patient_id
    ).order_by(PendingCheckpointEventORM.revision)).all()

def _approved_plan(events):
    plans=[e for e in events if e.kind=="plan_approved"]
    if not plans:return None
    if len(plans)!=1:raise HTTPException(409,"Ambiguous plan")
    return VisitCaptureCheckpointPlan.model_validate(plans[0].plan_json)

def _revision(events):return max((e.revision for e in events),default=0)

def _effective(events):
    retracted={e.retracts_event_id for e in events if e.kind=="retracted"}
    confirmed={e.checkpoint_key:e for e in events if e.kind=="yes_confirmed" and e.id not in retracted}
    not_yet={e.checkpoint_key:e.recorded_at for e in events if e.kind=="not_yet_observed"}
    return confirmed,not_yet

def project_lifecycle(session,row):
    events=_events(session,row)
    active=next((e for e in reversed(events) if e.kind in ("cancelled","reopened")),None)
    return {"revision":_revision(events), "cancellation_event_id":active.id if active and active.kind=="cancelled" and row.status=="cancelled" else None}

def project_checkpoint(session,row):
    events=_events(session,row);plan=_approved_plan(events)
    if plan is None:return None
    confirmed,not_yet=_effective(events)
    steps=[dict(key=step.key,question=step.question,meaning=step.meaning,
        state="confirmed" if step.key in confirmed else "not_yet" if step.key in not_yet else "unknown",
        event_id=confirmed[step.key].id if step.key in confirmed else None,
        recorded_at=(confirmed[step.key].recorded_at.isoformat()+"Z") if step.key in confirmed else None,
        observed_not_yet_at=not_yet[step.key].isoformat()+"Z" if step.key in not_yet else None)
        for step in plan.steps]
    labels={step.key:step.meaning for step in plan.steps}
    return {"pending_id":row.id,"patient_id":row.patient_id,"source_encounter_id":row.source_encounter_id,
        "revision":_revision(events),"plan_version":plan.version,"status":row.status,"steps":steps,
        "next_checkpoint":next((step for step in steps if step["state"]!="confirmed"),None),
        "confirmed_keys":list(confirmed),"completed":all(step.key in confirmed for step in plan.steps),
        "history":[{"id":e.id,"checkpoint_key":e.checkpoint_key,"kind":e.kind,"recorded_at":e.recorded_at.isoformat()+"Z",
            "label":("Αναίρεση · " if e.kind=="retracted" else "Όχι ακόμη · " if e.kind=="not_yet_observed" else "")+labels.get(e.checkpoint_key,"Παρακολούθηση υποχρέωσης"),
            "retracts_event_id":e.retracts_event_id} for e in events if e.kind not in ("plan_approved","cancelled","reopened")]}

def attach_pending_one_tap_routes(router,engine,protected):
    def load(db,patient_id,pending_id,request):
        if not installed(engine):raise HTTPException(503,"Checkpoint storage is not installed")
        patient=db.get(PatientORM,patient_id)
        row=db.scalar(select(ClinicalPendingORM).where(ClinicalPendingORM.id==pending_id).with_for_update())
        if not patient or not row or row.patient_id!=patient_id:raise HTTPException(404,"Pending not found")
        actor=actor_for(request,patient);_signed_source(db,row)
        return row,actor

    def replay(db,row,actor,key,kind,checkpoint,retract=None,plan=None,check_plan=True):
        e=db.scalar(select(PendingCheckpointEventORM).where(PendingCheckpointEventORM.idempotency_key==key))
        if e is None:return None
        if (e.patient_id!=row.patient_id or e.pending_id!=row.id or e.actor_principal_id!=actor.principal_id or e.actor_role!=actor.role
            or e.kind!=kind or e.checkpoint_key!=checkpoint or e.retracts_event_id!=retract or (check_plan and e.plan_json!=plan)):
            raise HTTPException(409,"Idempotency conflict")
        return {"event_id":e.id,"replayed":True,"current":project_checkpoint(db,row)}

    def commit(db):
        try:db.commit()
        except IntegrityError:
            db.rollback();raise HTTPException(409,"Checkpoint changed; reload") from None

    @router.post("/patient/{patient_id}/pending/{pending_id}/checkpoint-plan",dependencies=protected)
    def adopt(patient_id:str,pending_id:str,action:AdoptPlan,request:Request):
        with Session(engine) as db:
            row,actor=load(db,patient_id,pending_id,request)
            plan=VisitCaptureCheckpointPlan(steps=[VisitCaptureCheckpointStep(key=k,question=q,meaning=m) for k,q,m in PLANS[action.template]]).model_dump(mode="json")
            # The initial explicit observation is part of the replay identity.
            initial_key=action.first_checkpoint_key or plan["steps"][0]["key"]
            if initial_key not in {step["key"] for step in plan["steps"]} or (action.first_checkpoint_key and not action.first_observation):
                raise HTTPException(422,"Invalid initial checkpoint")
            plan["initial_checkpoint_key"] = initial_key
            plan["initial_observation"] = action.first_observation
            prior=replay(db,row,actor,action.idempotency_key,"plan_approved","__approved_plan__",plan=plan)
            if prior:return prior
            if row.status not in ("open","awaiting_patient","awaiting_result"):raise HTTPException(409,"Plan cannot be adopted")
            events=_events(db,row)
            if _approved_plan(events) or _revision(events)!=action.expected_revision:
                raise HTTPException(409,"Plan cannot be adopted")
            plan_revision=_revision(events)+1 if events else 0
            event=PendingCheckpointEventORM(id=str(uuid4()),idempotency_key=action.idempotency_key,pending_id=row.id,
                patient_id=patient_id,checkpoint_key="__approved_plan__",plan_version=1,kind="plan_approved",revision=plan_revision,
                recorded_at=utcnow(),actor_principal_id=actor.principal_id,actor_role=actor.role,plan_json=plan)
            db.add(event)
            if action.first_observation:
                observed=PendingCheckpointEventORM(id=str(uuid4()),idempotency_key="first-"+event.id,pending_id=row.id,
                    patient_id=patient_id,checkpoint_key=initial_key,plan_version=1,kind=action.first_observation,revision=plan_revision+1,
                    recorded_at=utcnow(),actor_principal_id=actor.principal_id,actor_role=actor.role,plan_json=None)
                db.add(observed)
                complete=action.first_observation=="yes_confirmed" and len(plan["steps"])==1
                row.status="resolved" if complete else "open" if action.first_observation=="yes_confirmed" else row.status
                row.resolution_event_id=observed.id if complete else None;row.updated_at=observed.recorded_at
            commit(db)
            return {"event_id":event.id,"replayed":False,"current":project_checkpoint(db,row)}

    @router.post("/patient/{patient_id}/pending/{pending_id}/checkpoint",dependencies=protected)
    def confirm(patient_id:str,pending_id:str,action:CheckpointAction,request:Request):
        with Session(engine) as db:
            row,actor=load(db,patient_id,pending_id,request)
            prior=replay(db,row,actor,action.idempotency_key,action.observation,action.checkpoint_key,action.retracts_event_id)
            if prior:return prior
            events=_events(db,row);plan=_approved_plan(events)
            if not plan or _revision(events)!=action.expected_revision:raise HTTPException(409,"Checkpoint revision changed")
            if action.checkpoint_key not in {s.key for s in plan.steps}:raise HTTPException(422,"Unknown checkpoint")
            if row.status not in ("open","awaiting_patient","awaiting_result","resolved"):raise HTTPException(409,"Pending is not eligible")
            confirmed,_=_effective(events)
            if action.observation=="retracted":
                previous=confirmed.get(action.checkpoint_key)
                if not previous or previous.id!=action.retracts_event_id:raise HTTPException(409,"Confirmation is no longer active")
                del confirmed[action.checkpoint_key]
            else:
                if action.retracts_event_id or action.checkpoint_key in confirmed or row.status=="resolved":raise HTTPException(409,"Checkpoint already confirmed")
                if action.observation=="yes_confirmed":confirmed[action.checkpoint_key]=True
            event=PendingCheckpointEventORM(id=str(uuid4()),idempotency_key=action.idempotency_key,pending_id=row.id,
                patient_id=patient_id,checkpoint_key=action.checkpoint_key,plan_version=plan.version,kind=action.observation,
                retracts_event_id=action.retracts_event_id,revision=action.expected_revision+1,recorded_at=utcnow(),
                actor_principal_id=actor.principal_id,actor_role=actor.role,plan_json=None)
            complete=all(step.key in confirmed for step in plan.steps)
            row.status="resolved" if complete else "open" if confirmed or action.observation=="retracted" else row.status
            row.resolution_event_id=event.id if complete else None;row.updated_at=event.recorded_at
            db.add(event);commit(db)
            return {"event_id":event.id,"replayed":False,"current":project_checkpoint(db,row)}

    @router.post("/patient/{patient_id}/pending/{pending_id}/lifecycle",dependencies=protected)
    def lifecycle(patient_id:str,pending_id:str,action:PendingLifecycleAction,request:Request):
        with Session(engine) as db:
            row,actor=load(db,patient_id,pending_id,request)
            kind="cancelled" if action.action=="cancel" else "reopened"
            prior=replay(db,row,actor,action.idempotency_key,kind,"__lifecycle__",action.cancellation_event_id,check_plan=False)
            if prior:return prior
            events=_events(db,row)
            if _revision(events)!=action.expected_revision:raise HTTPException(409,"Pending revision changed")
            if action.action=="cancel":
                if action.cancellation_event_id or row.status not in ("open","awaiting_patient","awaiting_result"):
                    raise HTTPException(409,"Pending cannot be cancelled")
                metadata={"previous_status":row.status,"previous_resolution_event_id":row.resolution_event_id}
                row.status="cancelled"
            else:
                current=next((e for e in reversed(events) if e.kind in ("cancelled","reopened")),None)
                if row.status!="cancelled" or not current or current.kind!="cancelled" or current.id!=action.cancellation_event_id:
                    raise HTTPException(409,"Cancellation is no longer active")
                metadata=None
                row.status=current.plan_json["previous_status"]
                row.resolution_event_id=current.plan_json.get("previous_resolution_event_id")
            event=PendingCheckpointEventORM(id=str(uuid4()),idempotency_key=action.idempotency_key,pending_id=row.id,
                patient_id=patient_id,checkpoint_key="__lifecycle__",plan_version=1,kind=kind,
                retracts_event_id=action.cancellation_event_id,revision=action.expected_revision+1,recorded_at=utcnow(),
                actor_principal_id=actor.principal_id,actor_role=actor.role,plan_json=metadata)
            row.updated_at=event.recorded_at;db.add(event);commit(db)
            return {"event_id":event.id,"replayed":False,"current":project_checkpoint(db,row),"lifecycle":project_lifecycle(db,row)}
