"""Bounded, protected RF tracking reads. No second RF registry or outcome writer.

All cursor identifiers travel inside authenticated POST bodies, never document URLs.
Mutable progress is not a cross-request historical snapshot.
"""
from __future__ import annotations

import base64
from datetime import date, datetime, timedelta, timezone
import hashlib
import hmac
import json
import os
import re
import secrets
from typing import Any

from sqlalchemy import and_, or_, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from .persistence import (
    RFApplicationORM, RFApplicationProgressORM,
    _rf_application_source, _rf_id_text, _rf_progress_card, _rf_search_text,
)

_SCAN_BATCH = 200
_MATCH_BATCH = 20
_WEEK_BATCH = 100
_CURSOR_SECONDS = 600
_PROCESS_KEY = secrets.token_bytes(32)
_STAGES = frozenset({"", "issued", "submitted", "approved", "rejected", "scheduled", "performed"})


def _key() -> bytes:
    configured = os.environ.get("CLINICAL_DATA_KEY", "")
    return (configured.encode("utf-8") + b":native-rf-tracking:v1") if configured else _PROCESS_KEY


def _seal(data: dict[str, Any]) -> str:
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    mac = hmac.new(_key(), raw, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(raw + mac).decode().rstrip("=")


def _unseal(cursor: str, purpose: str, fingerprint: str) -> dict[str, Any]:
    try:
        packed = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        raw, mac = packed[:-32], packed[-32:]
        if len(mac) != 32 or not hmac.compare_digest(hmac.new(_key(), raw, hashlib.sha256).digest(), mac):
            raise ValueError("signature")
        data = json.loads(raw)
        if (not isinstance(data, dict) or data.get("purpose") != purpose
                or data.get("fingerprint") != fingerprint or data.get("v") != 1
                or type(data.get("until")) not in {int, float}
                or data["until"] < datetime.now(timezone.utc).timestamp()):
            raise ValueError("context/expired")
        return data
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("Η συνέχεια RF δεν ισχύει. Ξεκίνησε νέα αναζήτηση.") from exc


def _fingerprint(data: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def _valid_date(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("Μη έγκυρη ημερομηνία RF")
    if date.fromisoformat(value).isoformat() != value:
        raise ValueError("Μη έγκυρη ημερομηνία RF")
    return value


def _pair(value: Any) -> tuple[datetime, str]:
    if (not isinstance(value, list) or len(value) != 2
            or not isinstance(value[0], str) or not isinstance(value[1], str)):
        raise ValueError("Μη έγκυρη συνέχεια RF")
    result = datetime.fromisoformat(value[0])
    if result.tzinfo is not None or not value[1]:
        raise ValueError("Μη έγκυρη συνέχεια RF")
    return result, value[1]


def _after_created(created, id_column, pair: tuple[datetime, str]):
    when, row_id = pair
    return or_(created < when, and_(created == when, id_column < row_id))


def _up_to(created, id_column, pair: tuple[datetime, str]):
    when, row_id = pair
    return or_(created < when, and_(created == when, id_column <= row_id))


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def tracking_search(engine: Engine, *, query: str = "", scope: str = "all",
                    stage: str = "", issued_from: str = "", issued_to: str = "",
                    cursor: str = "") -> dict[str, Any]:
    if scope not in {"all", "mine"} or stage not in _STAGES:
        raise ValueError("Μη έγκυρο φίλτρο RF")
    if issued_from: _valid_date(issued_from)
    if issued_to: _valid_date(issued_to)
    if issued_from and issued_to and issued_from > issued_to:
        raise ValueError("Μη έγκυρο εύρος ημερομηνιών RF")
    f = {"query": query, "scope": scope, "stage": stage,
         "issued_from": issued_from, "issued_to": issued_to}
    fingerprint = _fingerprint(f)
    data = _unseal(cursor, "search", fingerprint) if cursor else None
    high = _pair(data["high"]) if data else None
    after = _pair(data["after"]) if data else None
    norm = _rf_search_text(query)
    norm_id = _rf_id_text(query)
    seen: list[dict[str, Any]] = []
    read_count = 0
    with Session(engine) as session:
        if data is None:
            newest = session.scalar(select(RFApplicationORM).order_by(
                RFApplicationORM.created_at.desc(), RFApplicationORM.id.desc()).limit(1))
            if newest is None:
                return {"applications": [], "next_cursor": None, "has_more": False,
                        "scan_complete": True, "truncated": False, "source_as_of": _stamp(),
                        "counts_exact": False, "snapshot_consistency": "mutable_progress"}
            high = (newest.created_at, newest.id)
        q = select(RFApplicationORM).where(_up_to(
            RFApplicationORM.created_at, RFApplicationORM.id, high))
        if after:
            q = q.where(_after_created(RFApplicationORM.created_at, RFApplicationORM.id, after))
        if issued_from:
            q = q.where(RFApplicationORM.created_at >= datetime.fromisoformat(issued_from))
        if issued_to:
            q = q.where(RFApplicationORM.created_at < datetime.combine(
                date.fromisoformat(issued_to) + timedelta(days=1), datetime.min.time()))
        q = q.order_by(RFApplicationORM.created_at.desc(), RFApplicationORM.id.desc())
        raw = session.scalars(q.limit(_SCAN_BATCH + 1)).all()
        scan = raw[:_SCAN_BATCH]
        progress = (session.scalars(select(RFApplicationProgressORM).where(
            RFApplicationProgressORM.application_id.in_([x.id for x in scan]))).all() if scan else [])
        by_id = {x.application_id: x for x in progress}
        last: tuple[datetime, str] | None = None
        for app in scan:
            read_count += 1
            last = (app.created_at, app.id)
            if norm and not (norm in _rf_search_text(app.patient_name)
                             or norm_id in _rf_id_text(app.patient_identity_key)
                             or norm_id in _rf_id_text(app.gesy_number)):
                continue
            card = _rf_progress_card(app, by_id.get(app.id))
            if scope == "mine" and not card["mine_to_follow"]:
                continue
            if stage and card["stage"] != stage:
                continue
            seen.append(card)
            if len(seen) == _MATCH_BATCH:
                break
        has_more = read_count < len(raw) or len(raw) > _SCAN_BATCH
        next_cursor = (_seal({"v": 1, "purpose": "search", "fingerprint": fingerprint,
                              "high": [high[0].isoformat(), high[1]],
                              "after": [last[0].isoformat(), last[1]],
                              "until": datetime.now(timezone.utc).timestamp() + _CURSOR_SECONDS})
                       if has_more and last else None)
        return {"applications": seen, "next_cursor": next_cursor, "has_more": has_more,
                "scan_complete": not has_more, "truncated": has_more,
                "source_as_of": _stamp(), "counts_exact": False,
                "snapshot_consistency": "mutable_progress", "scanned": read_count}


def tracking_week(engine: Engine, *, week_start: str, cursor: str = "") -> dict[str, Any]:
    day = date.fromisoformat(_valid_date(week_start))
    if day.weekday() != 0:
        raise ValueError("Η εβδομάδα RF αρχίζει Δευτέρα")
    end = (day + timedelta(days=6)).isoformat()
    fingerprint = _fingerprint({"week_start": week_start})
    data = _unseal(cursor, "week", fingerprint) if cursor else None
    after = data.get("after") if data else None
    if after is not None and (not isinstance(after, str) or not after):
        raise ValueError("Μη έγκυρη συνέχεια εβδομάδας RF")
    progress = RFApplicationProgressORM
    with Session(engine) as session:
        q = (select(RFApplicationORM, progress)
             .join(progress, progress.application_id == RFApplicationORM.id)
             .where(or_(progress.scheduled_date.between(week_start, end),
                        progress.performed_date.between(week_start, end))))
        if after:
            q = q.where(progress.application_id > after)
        rows = session.execute(q.order_by(progress.application_id.asc()).limit(_WEEK_BATCH + 1)).all()
        partial = len(rows) > _WEEK_BATCH
        chosen = rows[:_WEEK_BATCH]
        next_cursor = (_seal({"v": 1, "purpose": "week", "fingerprint": fingerprint,
                              "after": chosen[-1][0].id,
                              "until": datetime.now(timezone.utc).timestamp() + _CURSOR_SECONDS})
                       if partial and chosen else None)
        return {"week_start": week_start, "week_end": end,
                "applications": [_rf_progress_card(app, row) for app, row in chosen],
                "next_cursor": next_cursor, "has_more": partial,
                "complete": not partial, "truncated": partial,
                "source_as_of": _stamp(), "snapshot_consistency": "mutable_progress",
                "booking_time_status": "off_unverified_link"}


def guarded_rejected_correction(engine: Engine, *, application_id: str,
                                expected_revision: int) -> dict[str, Any]:
    """Exact immutable attempt + locked mutable progress, one read-only transaction."""
    with Session(engine) as session:
        app = session.get(RFApplicationORM, application_id)
        if app is None:
            raise LookupError("Δεν βρέθηκε η επιλεγμένη αίτηση RF")
        progress = session.scalar(select(RFApplicationProgressORM).where(
            RFApplicationProgressORM.application_id == application_id).with_for_update())
        if (progress is None or progress.stage != "rejected"
                or progress.decision != "rejected"
                or progress.revision != expected_revision):
            raise RuntimeError("Η απόρριψη ή έκδοση της RF άλλαξε. Ανανέωσε την αίτηση")
        reusable = _rf_application_source(app)
        return {"application_id": app.id, "revision": progress.revision,
                "source": reusable["source"], "fields": reusable["fields"]}
