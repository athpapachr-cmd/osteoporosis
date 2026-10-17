"""Synthetic-only, event-sourced Clinical Pending one-tap vertical (#162).

Never enables real clinical writes, creates production schema, changes signed
encounters, or confuses a Clinical Key with individual clinician attestation.
The event table must be installed explicitly in an isolated test database.
"""
from __future__ import annotations

import os
from typing import Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Integer, JSON, String, UniqueConstraint, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Session

from clinical_staff_auth import StaffActor, require_staff_clinician, staff_enabled
from clinical_data import (
    ClinicalPendingORM, EncounterORM, PatientORM, VisitCaptureCheckpointPlan, utcnow,
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


def require_synthetic_clinician(request: Request, engine: Engine) -> StaffActor:
    if os.environ.get("CLINICAL_PENDING_ONE_TAP_SYNTHETIC_ENABLED") != "1" or engine.dialect.name != "sqlite":
        raise HTTPException(503, "Checkpoint writer is synthetic-only and disabled")
    if not staff_enabled():
        raise HTTPException(503, "Qualified test clinician session is not enabled")
    return require_staff_clinician(request)


def _signed_source(session: Session, row: ClinicalPendingORM) -> None:
    source = session.get(EncounterORM, row.source_encounter_id)
    if (source is None or source.patient_id != row.patient_id
            or source.status not in ("completed", "amended")):
        raise HTTPException(409, "Pending has no qualified signed encounter")
    signed = (source.payload_json or {}).get("_visit_capture_v1")
    if not isinstance(signed, dict) or signed.get("signed") is not True or not signed.get("signed_at"):
        raise HTTPException(409, "Pending source is not signed")


def approve_synthetic_checkpoint_plan(session: Session, row: ClinicalPendingORM,
                                      plan: VisitCaptureCheckpointPlan, actor: StaffActor) -> None:
    """Plan comes only from deliberate signed Visit Capture Save in this transaction."""
    session.add(PendingCheckpointEventORM(
        id=str(uuid4()), idempotency_key="synthetic-plan-" + uuid4().hex,
        pending_id=row.id, patient_id=row.patient_id, checkpoint_key="__approved_plan__",
        plan_version=plan.version, kind="plan_approved", retracts_event_id=None,
        revision=0, recorded_at=utcnow(), actor_principal_id=actor.principal_id,
        actor_role=actor.role, plan_json=plan.model_dump(mode="json"),
    ))


def _events(session: Session, row: ClinicalPendingORM) -> list[PendingCheckpointEventORM]:
    return session.scalars(select(PendingCheckpointEventORM)
                           .where(PendingCheckpointEventORM.pending_id == row.id,
                                  PendingCheckpointEventORM.patient_id == row.patient_id)
                           .order_by(PendingCheckpointEventORM.revision.asc())).all()


def _approved_plan(events: list[PendingCheckpointEventORM]
                   ) -> VisitCaptureCheckpointPlan | None:
    approvals = [ev for ev in events if ev.kind == "plan_approved" and ev.revision == 0]
    if not approvals:
        return None
    if len(approvals) != 1 or not isinstance(approvals[0].plan_json, dict):
        raise HTTPException(409, "Ambiguous approved checkpoint plan")
    try:
        plan = VisitCaptureCheckpointPlan.model_validate(approvals[0].plan_json)
    except ValueError:
        raise HTTPException(409, "Invalid approved checkpoint plan") from None
    if len({step.key for step in plan.steps}) != len(plan.steps):
        raise HTTPException(409, "Ambiguous approved checkpoint plan")
    return plan


def _revision(events: list[PendingCheckpointEventORM]) -> int:
    return max((ev.revision for ev in events), default=0)


def _effective(events: list[PendingCheckpointEventORM]):
    retracted = {ev.retracts_event_id for ev in events if ev.kind == "retracted"}
    confirmed = {ev.checkpoint_key: ev for ev in events
                 if ev.kind == "yes_confirmed" and ev.id not in retracted}
    not_yet = {ev.checkpoint_key: ev.recorded_at for ev in events
               if ev.kind == "not_yet_observed"}
    return confirmed, not_yet


def project_checkpoint(session: Session, row: ClinicalPendingORM) -> dict:
    events = _events(session, row)
    plan = _approved_plan(events)
    if plan is None:
        return None
    confirmed, not_yet = _effective(events)
    next_step = next((step for step in plan.steps if step.key not in confirmed), None)
    return {
        "pending_id": row.id,
        "patient_id": row.patient_id,
        "source_encounter_id": row.source_encounter_id,
        "revision": _revision(events),
        "plan_version": plan.version,
        "status": row.status,
        "next_checkpoint": (
            {"key": next_step.key, "question": next_step.question,
             "meaning": next_step.meaning, "state": "not_yet" if next_step.key in not_yet else "unknown",
             "observed_not_yet_at": not_yet[next_step.key].isoformat()
                 if next_step.key in not_yet else None}
            if next_step is not None else None
        ),
        "confirmed_keys": [step.key for step in plan.steps if step.key in confirmed],
        "completed": next_step is None,
    }


def attach_pending_one_tap_routes(router: APIRouter, engine: Engine, protected: list) -> None:
    """Opt-in isolated demo; never creates DDL or grants a shared Clinical Key write."""
    @router.post(
        "/patient/{patient_id}/pending/{pending_id}/checkpoint",
        dependencies=protected,
    )
    def confirm_checkpoint(patient_id: str, pending_id: str, action: CheckpointAction,
                           request: Request):
        actor = require_synthetic_clinician(request, engine)
        with Session(engine) as session:
            row = session.get(ClinicalPendingORM, pending_id)
            if row is None or row.patient_id != patient_id:
                raise HTTPException(404, "Pending item not found")
            if session.get(PatientORM, patient_id) is None:
                raise HTTPException(404, "Patient not found")
            _signed_source(session, row)
            events = _events(session, row)
            plan = _approved_plan(events)
            if plan is None:
                raise HTTPException(409, "Pending has no approved checkpoint plan")
            existing = session.scalar(select(PendingCheckpointEventORM).where(
                PendingCheckpointEventORM.idempotency_key == action.idempotency_key))
            if existing is not None:
                if (existing.patient_id != patient_id or existing.pending_id != pending_id
                        or existing.checkpoint_key != action.checkpoint_key
                        or existing.kind != action.observation
                        or existing.retracts_event_id != action.retracts_event_id
                        or existing.actor_principal_id != actor.principal_id
                        or existing.actor_role != actor.role):
                    raise HTTPException(409, "Idempotency key conflict")
                return {"event_id": existing.id, "event_revision": existing.revision,
                        "replayed": True, "current": project_checkpoint(session, row)}
            if _revision(events) != action.expected_revision:
                raise HTTPException(409, "Checkpoint revision changed")
            confirmed, _ = _effective(events)
            next_step = next((step for step in plan.steps if step.key not in confirmed), None)

            if action.observation == "retracted":
                if not action.retracts_event_id:
                    raise HTTPException(422, "Retraction requires the original event")
                active = [ev for ev in events if ev.kind == "yes_confirmed"
                          and confirmed.get(ev.checkpoint_key) is ev]
                last = active[-1] if active else None
                if (last is None or last.id != action.retracts_event_id
                        or last.checkpoint_key != action.checkpoint_key):
                    raise HTTPException(409, "Only the last active confirmation can be retracted")
                complete_after = False
            else:
                if action.retracts_event_id is not None:
                    raise HTTPException(422, "Retraction reference is not permitted")
                if next_step is None or next_step.key != action.checkpoint_key:
                    raise HTTPException(409, "Checkpoint is not the current approved step")
                if next_step.attesting_role != actor.role:
                    raise HTTPException(403, "Checkpoint clinician role required")
                complete_after = (action.observation == "yes_confirmed"
                                  and len(confirmed) + 1 == len(plan.steps))

            if row.status not in ("open", "awaiting_patient", "awaiting_result", "resolved"):
                raise HTTPException(409, "Pending is not eligible")
            if row.status == "resolved" and action.observation != "retracted":
                raise HTTPException(409, "Resolved pending cannot advance")

            event_id = str(uuid4())
            revision = action.expected_revision + 1
            now = utcnow()
            new_status = ("resolved" if complete_after else
                          "open" if action.observation == "retracted" else
                          row.status)
            new_resolution = (event_id if complete_after else
                              None if action.observation == "retracted" else
                              row.resolution_event_id)
            # Synthetic SQLite only: revision uniqueness is the CAS oracle.
            # The new event and parent status commit atomically; production
            # ClinicalPendingORM remains binary/schema compatible and unchanged.
            session.add(PendingCheckpointEventORM(
                id=event_id, idempotency_key=action.idempotency_key,
                patient_id=patient_id, pending_id=pending_id,
                checkpoint_key=action.checkpoint_key, plan_version=plan.version,
                kind=action.observation, retracts_event_id=action.retracts_event_id,
                revision=revision, recorded_at=now,
                actor_principal_id=actor.principal_id, actor_role=actor.role,
                plan_json=None))
            row.status = new_status
            row.resolution_event_id = new_resolution
            row.updated_at = now
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                raise HTTPException(409, "Checkpoint event conflict") from None
            session.refresh(row)
            return {"event_id": event_id, "event_revision": revision,
                    "replayed": False, "current": project_checkpoint(session, row)}

    @router.get("/daily-brief/pending", dependencies=protected)
    def synthetic_daily_brief(request: Request):
        require_synthetic_clinician(request, engine)
        with Session(engine) as session:
            rows = session.scalars(
                select(ClinicalPendingORM)
                .where(ClinicalPendingORM.status.in_(("open", "awaiting_patient", "awaiting_result")))
                .order_by(ClinicalPendingORM.updated_at.desc(), ClinicalPendingORM.id.asc())
                .limit(100)
            ).all()
            result = []
            for row in rows:
                item = project_checkpoint(session, row)
                if item is None:
                    continue
                _signed_source(session, row)
                result.append(item)
                if len(result) >= 4:
                    break
            return {"items": result, "synthetic_only": True}
