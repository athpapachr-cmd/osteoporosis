"""Single-clinician operational mail sorting; no clinical writer or boot DDL."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import hmac
import json
import re
from typing import Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import Column, DateTime, BigInteger, JSON, String, select, update
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session

from clinical_reception_calls import _require_clinical_key
from clinical_icloud_triage_rules import fold

Category = Literal["clinical", "administrative", "other", "review"]
MAX_OVERRIDES, MAX_RULES, MAX_RECEIPTS, MAX_BYTES = 5000, 100, 64, 1024 * 1024
_ID = re.compile(r"^([1-9][0-9]{0,18})-([1-9][0-9]{0,18})$")
_DOMAIN = re.compile(r"[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?\.[a-z]{2,63}")
_MIXED = ("wetransfer.com", "swisstransfer.com", "wesendit.com")


class PreferenceBase(DeclarativeBase):
    pass


class PreferenceORM(PreferenceBase):
    __tablename__ = "icloud_triage_preferences"
    mailbox_key = Column(String(64), primary_key=True)
    revision = Column(BigInteger, nullable=False)
    state_json = Column(JSON, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class RuleInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rule_id: UUID | None = None
    sender: str = Field(min_length=3, max_length=254)
    domain: bool = False
    terms: list[str] = Field(default_factory=list, max_length=8)
    category: Category

    @model_validator(mode="after")
    def criteria(self):
        self.sender = self.sender.strip().casefold()
        host = self.sender if self.domain else self.sender.rsplit("@", 1)[-1]
        if not _DOMAIN.fullmatch(host) or (not self.domain and (
            self.sender.count("@") != 1 or not re.fullmatch(r"[^\s<>@]+@[^\s<>@]+", self.sender)
        )):
            raise ValueError("Choose an exact sender address or domain")
        self.terms = list(dict.fromkeys(fold(term) for term in self.terms))
        if any(not term or len(term) > 80 or not re.fullmatch(r"[\w .-]+", term) for term in self.terms):
            raise ValueError("Use short literal subject terms")
        if any(host == d or host.endswith("." + d) for d in _MIXED) and not self.terms:
            raise ValueError("Transfer services require a subject term")
        return self


class Change(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation_id: UUID
    expected_revision: int = Field(ge=0, le=2**53 - 2, strict=True)
    kind: Literal["category", "rule", "disable", "undo"]
    item_id: str | None = Field(default=None, pattern=_ID.pattern)
    category: Category | None = None  # null resets automatic sorting
    rule: RuleInput | None = None
    rule_id: UUID | None = None
    undo_id: UUID | None = None

    @model_validator(mode="after")
    def shape(self):
        if self.kind == "category":
            valid = self.item_id is not None and self.rule_id is None and self.undo_id is None
        elif self.kind == "rule":
            valid = self.rule is not None and self.item_id is None and self.category is None and self.rule_id is None and self.undo_id is None
        elif self.kind == "disable":
            valid = self.rule_id is not None and self.item_id is None and self.category is None and self.rule is None and self.undo_id is None
        else:
            valid = self.undo_id is not None and self.item_id is None and self.category is None and self.rule is None and self.rule_id is None
        if not valid:
            raise ValueError("Invalid triage operation")
        return self


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def scope_key(username: str, alias: str) -> str:
    return _hash([username.strip().casefold(), alias.casefold(), "INBOX"])


def message_key(scope: str, item_id: str) -> str:
    return _hash([scope, *_ID.fullmatch(item_id).groups()])


def unavailable():
    return HTTPException(503, {"code": "TRIAGE_STORAGE_UNAVAILABLE"})


def read_preferences(engine, scope):
    try:
        with Session(engine) as db:
            row = db.get(PreferenceORM, scope)
            if row is None:
                return 0, {"schema": 1, "overrides": {}, "rules": [], "receipts": []}
            state = deepcopy(row.state_json)
            if (not isinstance(state, dict) or state.get("schema") != 1
                    or not isinstance(state.get("overrides"), dict)
                    or not isinstance(state.get("rules"), list)
                    or not isinstance(state.get("receipts"), list)
                    or not isinstance(row.revision, int) or row.revision < 1
                    or len(state["overrides"]) > MAX_OVERRIDES or len(state["rules"]) > MAX_RULES
                    or len(state["receipts"]) > MAX_RECEIPTS):
                raise unavailable()
            try:
                for value in state["overrides"].values():
                    if value["category"] not in ("clinical", "administrative", "other", "review"):
                        raise ValueError()
                for rule in state["rules"]:
                    UUID(rule["rule_id"])
                    RuleInput(**{k: rule[k] for k in ("rule_id", "sender", "domain", "terms", "category")})
                    if (type(rule["enabled"]) is not bool or type(rule["after_uid"]) is not int
                            or rule["after_uid"] < 0 or not re.fullmatch(r"[1-9][0-9]{0,18}", rule["uidvalidity"])):
                        raise ValueError()
            except (KeyError, TypeError, ValueError):
                raise unavailable() from None
            return row.revision, state
    except SQLAlchemyError:
        raise unavailable() from None


def metadata(scope, validity, revision):
    return {"mailbox_key": scope, "uidvalidity": validity, "revision": revision, "available": True}


def projection(scope, validity, revision, state):
    return {"triage": metadata(scope, validity, revision), "rules": [
        {**rule, "active": rule["enabled"] and rule["uidvalidity"] == validity}
        for rule in state["rules"]]}


def classify(state, scope, validity, uid, message, suggested):
    from email.utils import getaddresses
    saved = state["overrides"].get(message_key(scope, f"{validity}-{uid}"))
    if saved:
        return saved["category"], "manual"
    sender = (getaddresses([str(message.get("From", ""))]) or [("", "")])[0][1].casefold()
    host = sender.rsplit("@", 1)[-1]
    subject = fold(message.get("Subject", ""))
    matched = []
    for rule in state["rules"]:
        if not rule["enabled"] or rule["uidvalidity"] != validity or int(uid) <= rule["after_uid"]:
            continue
        sender_match = (host == rule["sender"] or host.endswith("." + rule["sender"])) if rule["domain"] else sender == rule["sender"]
        if sender_match and all(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", subject) for term in rule["terms"]):
            matched.append(rule["category"])
    if len(set(matched)) > 1:
        return "review", "conflict"
    return (matched[0], "rule") if matched else (suggested, "suggestion")


def _origin(request):
    origin, host = request.headers.get("origin", ""), request.headers.get("host", "")
    if (not host or any(c in host for c in "\r\n\t /") or not origin
            or not any(hmac.compare_digest(origin, value) for value in (f"https://{host}", f"{request.url.scheme}://{host}"))):
        raise HTTPException(403, "Same-origin triage action required")


def mutate(engine, scope, validity, change, qualify):
    """CAS is the write lock on both dialects. Provider reads precede the short DB transaction."""
    revision, state = read_preferences(engine, scope)
    payload_hash = _hash(change.model_dump(mode="json"))
    op_id = str(change.operation_id)
    receipt = next((r for r in state["receipts"] if r["id"] == op_id), None)
    if receipt:
        if receipt["hash"] != payload_hash:
            raise HTTPException(409, "Operation identity conflict")
        return {**projection(scope, validity, revision, state), "operation_id": op_id,
                "committed_revision": receipt["revision"], "replayed": True}
    if revision != change.expected_revision:
        raise HTTPException(409, "Preferences changed; reload before saving")
    now = datetime.now(timezone.utc)
    inverse = {"override": None, "rule": None}
    if change.kind == "undo":
        previous = next((r for r in state["receipts"] if r["id"] == str(change.undo_id)), None)
        if not previous or previous["revision"] != revision or previous["kind"] == "undo":
            raise HTTPException(409, "Undo superseded by another change")
        for field, collection in (("override", "overrides"), ("rule", "rules")):
            value = previous["inverse"][field]
            if value is not None:
                key, before = value
                if collection == "overrides":
                    if before is None:
                        state[collection].pop(key, None)
                    else:
                        state[collection][key] = before
                else:
                    state[collection] = [r for r in state[collection] if r["rule_id"] != key]
                    if before is not None:
                        state[collection].append(before)
    else:
        # Qualify only new operations, never replay an old write or trust client scope/watermark.
        high_water = qualify(change.item_id, change.rule is not None)
        if change.kind == "category":
            if _ID.fullmatch(change.item_id).group(1) != validity:
                raise HTTPException(409, "Mailbox epoch changed")
            key = message_key(scope, change.item_id)
            inverse["override"] = [key, deepcopy(state["overrides"].get(key))]
            if change.category is None:
                state["overrides"].pop(key, None)
            else:
                state["overrides"][key] = {"category": change.category, "updated_at": now.isoformat()}
        if change.rule is not None:
            rule_id = str(change.rule.rule_id or uuid4())
            before = next((r for r in state["rules"] if r["rule_id"] == rule_id), None)
            if change.rule.rule_id and before is None:
                raise HTTPException(404, "Rule not found")
            inverse["rule"] = [rule_id, deepcopy(before)]
            rule = {**change.rule.model_dump(mode="json"), "rule_id": rule_id,
                    "enabled": True, "uidvalidity": validity, "after_uid": high_water,
                    "revision": (before["revision"] + 1) if before else 1,
                    "created_at": before["created_at"] if before else now.isoformat()}
            state["rules"] = [r for r in state["rules"] if r["rule_id"] != rule_id] + [rule]
        if change.kind == "disable":
            before = next((r for r in state["rules"] if r["rule_id"] == str(change.rule_id)), None)
            if before is None:
                raise HTTPException(404, "Rule not found")
            inverse["rule"] = [before["rule_id"], deepcopy(before)]
            before["enabled"] = False
            before["revision"] += 1
    new_revision = revision + 1
    state["receipts"] = (state["receipts"] + [{"id": op_id, "hash": payload_hash,
        "revision": new_revision, "kind": change.kind, "inverse": inverse}])[-MAX_RECEIPTS:]
    if (len(state["overrides"]) > MAX_OVERRIDES or len(state["rules"]) > MAX_RULES
            or len(json.dumps(state, ensure_ascii=False).encode()) > MAX_BYTES):
        raise HTTPException(409, "Preference capacity reached; nothing saved")
    try:
        with Session(engine) as db:
            try:
                if revision == 0:
                    db.add(PreferenceORM(mailbox_key=scope, revision=new_revision, state_json=state, updated_at=now))
                else:
                    result = db.execute(update(PreferenceORM).where(
                        PreferenceORM.mailbox_key == scope, PreferenceORM.revision == revision
                    ).values(revision=new_revision, state_json=state, updated_at=now))
                    if result.rowcount != 1:
                        raise HTTPException(409, "Preferences changed; reload before saving")
                db.commit()
            except Exception:
                db.rollback()
                raise
    except (IntegrityError, OperationalError):
        raise HTTPException(409, "Save not confirmed; reload before retrying") from None
    except SQLAlchemyError:
        raise unavailable() from None
    return {**projection(scope, validity, new_revision, state), "operation_id": op_id,
            "committed_revision": new_revision, "replayed": False}


def build_preferences_router(engine):
    from clinical_icloud_mail import _read, _settings, _uid_next, _fetch_headers, _summary
    router = APIRouter(prefix="/clinical/icloud-mail/preferences",
                       dependencies=[Depends(_require_clinical_key)])

    @router.get("")
    def read(response: Response):
        response.headers["Cache-Control"] = "private, no-store"
        def result(mail, validity, alias):
            scope = scope_key(_settings()[0], alias)
            revision, state = read_preferences(engine, scope)
            return projection(scope, validity, revision, state)
        return _read(result)

    @router.post("")
    def write(change: Change, request: Request, response: Response):
        _origin(request)
        response.headers["Cache-Control"] = "private, no-store"
        def result(mail, validity, alias):
            scope = scope_key(_settings()[0], alias)
            def qualify(item_id, watermark):
                if item_id:
                    match = _ID.fullmatch(item_id)
                    if match.group(1) != validity:
                        raise HTTPException(409, "Mailbox epoch changed")
                    header, _ = _fetch_headers(mail, match.group(2))
                    if _summary(match.group(2), validity, header, alias, include_nonclinical=True) is None:
                        raise HTTPException(404, "Message outside practice mailbox")
                return _uid_next(mail) - 1 if watermark else 0
            return mutate(engine, scope, validity, change, qualify)
        return _read(result)
    return router
