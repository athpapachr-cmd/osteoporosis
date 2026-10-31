"""Clinical follow-through event semantics, independent of transport and UI.

Synthetic-first, OFF production. No persistence, patient linking, IMAP, SMS or
clinical chart writes live here. A future reviewed owner must persist events
transactionally and verify source identities before calling this reducer.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from uuid import uuid4

PARTS = frozenset({"images", "report", "labs"})
EVIDENCE = frozenset({"qualified_order", "clinician_verified"})
IDENTITY_EVIDENCE = frozenset({"qualified_referral", "clinician_verified"})
_KEY = re.compile(r"^[a-zA-Z0-9:_./-]{4,120}$")
_IDEM = re.compile(r"^[a-zA-Z0-9_-]{16,100}$")


class CycleConflict(ValueError):
    pass


class CycleForbidden(PermissionError):
    pass


class CycleNotFound(LookupError):
    pass


@dataclass(frozen=True)
class Actor:
    principal: str
    role: str

    def check(self, *roles: str) -> None:
        if not self.principal or self.role not in roles:
            raise CycleForbidden("Actor not authorized for this event")


@dataclass
class Work:
    id: str
    visit_ref: str
    request_ref: str
    mode: str
    pending_ref: str | None
    required: tuple[str, ...]
    verified_contact: bool = False
    version: int = 0
    events: list[dict] = field(default_factory=list)


class SyntheticCycleLedger:
    """In-memory synthetic oracle, NOT a clinical datastore or role authority.

    The existing source/visit/actor checks must be applied by future adapters.
    User-facing operations should be at most one decision per responsible actor.
    """

    def __init__(self):
        self.lock = RLock()
        self.works: dict[str, Work] = {}
        self.by_request: dict[tuple[str, str], str] = {}
        self.by_source: dict[str, tuple[str, str]] = {}

    @staticmethod
    def _ref(value: str) -> str:
        if not isinstance(value, str) or not _KEY.fullmatch(value):
            raise CycleConflict("Invalid opaque source/visit reference")
        return value

    @staticmethod
    def _idem(value: str) -> str:
        if not isinstance(value, str) or not _IDEM.fullmatch(value):
            raise CycleConflict("Invalid idempotency key")
        return value

    @staticmethod
    def _current(work: Work) -> dict:
        arrivals: dict[str, dict] = {}
        reviewed: dict[str, str] = {}
        instruction: dict | None = None
        contact = "not_requested"
        callback = False
        for event in work.events:
            action = event["action"]
            part = event.get("part")
            if action == "received":
                arrivals[part] = {"source_ref": event["source_ref"],
                                  "source_version": event["source_version"]}
                reviewed.pop(part, None)
                # A corrected report invalidates previous review/communication
                # for the current obligation. The old event remains in audit.
                if instruction:
                    instruction = None
                    contact = "not_requested"
                    callback = False
            elif action in ("review_no_contact", "review_notify"):
                reviewed[part] = event["source_version"]
                if action == "review_notify":
                    instruction = {"part": part, "source_version": event["source_version"]}
                    contact = "to_notify"
                    callback = False
            elif action == "contact_attempt":
                contact = "attempted"
            elif action == "patient_notified":
                contact = "notified"
            elif action == "callback_complete":
                callback = True
        complete_parts = all(
            part in arrivals and reviewed.get(part) == arrivals[part]["source_version"]
            for part in work.required
        )
        all_arrived = all(part in arrivals for part in work.required)
        if not all_arrived:
            stage = "partially_received" if arrivals else "awaiting_result"
        elif not complete_parts:
            stage = "awaiting_doctor_review"
        elif instruction and contact in ("to_notify", "attempted"):
            stage = "awaiting_secretary"
        elif instruction and contact == "notified" and not callback:
            stage = "awaiting_callback"
        else:
            stage = "complete"
        return dict(
            case_id=work.id, visit_ref=work.visit_ref, request_ref=work.request_ref,
            mode=work.mode, pending_ref=work.pending_ref, required=list(work.required),
            version=work.version, verified_contact=work.verified_contact,
            received={p: dict(value) for p, value in arrivals.items()},
            reviewed=dict(reviewed), instruction=instruction,
            contact_status=contact, callback_complete=callback, stage=stage,
            complete=stage == "complete",
        )

    @staticmethod
    def _fingerprint(action: str, attrs: dict) -> str:
        return json.dumps({"action": action, **attrs}, sort_keys=True,
                          separators=(",", ":"), ensure_ascii=True)

    def register(self, actor: Actor, *, visit_ref: str, request_ref: str,
                 required: tuple[str, ...], mode: str,
                 pending_ref: str | None = None) -> dict:
        actor.check("clinician")
        visit_ref = self._ref(visit_ref)
        request_ref = self._ref(request_ref)
        req = tuple(dict.fromkeys(required))
        if not req or len(req) > 3 or set(req) - PARTS:
            raise CycleConflict("Invalid requested examinations")
        if mode not in ("signed_pending", "provisional_visit"):
            raise CycleConflict("Unknown visit mode")
        if (mode == "signed_pending") != (pending_ref is not None):
            raise CycleConflict("Signed pending and provisional visit must not be confused")
        if pending_ref is not None:
            self._ref(pending_ref)
        identity = (visit_ref, request_ref)
        with self.lock:
            if identity in self.by_request:
                old = self.works[self.by_request[identity]]
                if (old.mode, old.pending_ref, old.required) != (mode, pending_ref, req):
                    raise CycleConflict("Existing visit request has different evidence")
                return self._current(old)
            work = Work(id=str(uuid4()), visit_ref=visit_ref, request_ref=request_ref,
                        mode=mode, pending_ref=pending_ref, required=req)
            self.works[work.id] = work
            self.by_request[identity] = work.id
            return self._current(work)

    def view(self, case_id: str) -> dict:
        with self.lock:
            work = self.works.get(case_id)
            if work is None:
                raise CycleNotFound("Unknown obligation")
            return self._current(work)

    def secretary_view(self, case_id: str, actor: Actor) -> dict:
        actor.check("secretary")
        with self.lock:
            p = self.view(case_id)
            if not p["instruction"] or not p["verified_contact"]:
                raise CycleForbidden("No qualified secretary instruction")
            return {"case_id": p["case_id"], "version": p["version"],
                    "task": "Ενημέρωσε τον επιβεβαιωμένο ασθενή να καλέσει τον ιατρό",
                    "status": p["contact_status"], "doctor_reviewed": True}
            # Never expose source refs, report text, medical interpretation,
            # unverified contact details or the original clinical record.

    def history(self, case_id: str, actor: Actor) -> list[dict]:
        actor.check("clinician")
        with self.lock:
            work = self.works.get(case_id)
            if work is None:
                raise CycleNotFound("Unknown obligation")
            return [dict(event) for event in work.events]

    def act(self, case_id: str, actor: Actor, *, action: str,
            expected_version: int, idempotency_key: str, **attrs) -> dict:
        if not isinstance(actor, Actor):
            raise CycleForbidden("No actor")
        key = self._idem(idempotency_key)
        payload = self._fingerprint(action, attrs)
        with self.lock:
            work = self.works.get(case_id)
            if work is None:
                raise CycleNotFound("Unknown obligation")
            if action in ("received",):
                actor.check("source", "clinician")
            elif action in ("identity_verified", "review_notify", "review_no_contact",
                            "callback_complete"):
                actor.check("clinician")
            elif action in ("contact_attempt", "patient_notified"):
                actor.check("secretary")
            else:
                raise CycleConflict("Unsupported action")
            for event in work.events:
                if event["principal"] == actor.principal and event["idempotency_key"] == key:
                    if event["fingerprint"] != payload:
                        raise CycleConflict("Idempotency key reused with different action")
                    return self._current(work)
            if type(expected_version) is not int or work.version != expected_version:
                raise CycleConflict("Stale state; refresh before acting")

            state = self._current(work)
            part = attrs.get("part")
            source_ref = attrs.get("source_ref")
            source_version = attrs.get("source_version")
            if action == "received":
                if part not in work.required:
                    raise CycleConflict("Unexpected examination type")
                self._ref(source_ref)
                self._ref(source_version)
                proof = attrs.get("proof")
                if not ((actor.role == "source" and proof == "qualified_order")
                        or (actor.role == "clinician" and proof == "clinician_verified")):
                    raise CycleForbidden("Unqualified source cannot be linked by name")
                owner = self.by_source.get(source_ref)
                if owner and owner != (case_id, part):
                    raise CycleConflict("Source already belongs to another obligation")
                old = state["received"].get(part)
                if old and old["source_ref"] == source_ref and old["source_version"] == source_version:
                    return state  # Repeated provider poll is no new clinical event.
                self.by_source[source_ref] = (case_id, part)
            elif action == "identity_verified":
                if attrs.get("evidence") not in IDENTITY_EVIDENCE:
                    raise CycleForbidden("Name, email or phone alone is not evidence")
                self._ref(attrs.get("evidence_ref"))
                if work.verified_contact:
                    return state
                work.verified_contact = True
            elif action in ("review_no_contact", "review_notify"):
                if part not in work.required or part not in state["received"]:
                    raise CycleConflict("No received material for clinician to review")
                if state["received"][part]["source_version"] != source_version:
                    raise CycleConflict("Review references stale result version")
                if action == "review_notify":
                    if not work.verified_contact:
                        raise CycleForbidden("Patient/contact not independently verified")
                    if state["instruction"] is not None and state["contact_status"] != "not_requested":
                        raise CycleConflict("Communication already assigned")
            elif action in ("contact_attempt", "patient_notified"):
                if state["instruction"] is None or not work.verified_contact:
                    raise CycleForbidden("No clinician-authorized verified patient task")
                if state["contact_status"] not in ("to_notify", "attempted"):
                    raise CycleConflict("Communication already completed")
            elif action == "callback_complete":
                if (state["contact_status"] != "notified" or
                    not all(part in state["reviewed"] and part in state["received"] and
                            state["reviewed"][part] == state["received"][part]["source_version"]
                            for part in work.required)):
                    raise CycleConflict("No confirmed notification/current full review")
            work.version += 1
            event = {"version": work.version, "action": action, "actor_role": actor.role,
                     "principal": actor.principal, "at": datetime.now(timezone.utc).isoformat(),
                     "idempotency_key": key, "fingerprint": payload, **attrs}
            work.events.append(event)
            return self._current(work)
