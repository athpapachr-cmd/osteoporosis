"""Clinician-key-protected Reception call peek and explicit handled action.

The Reception dashboard remains the call and handling-state owner. This
projection neither stores patient/call data nor creates clinical tasks. It is
OFF unless an operator separately enables the private server-to-server source.
"""
from __future__ import annotations

import os
import secrets
from datetime import datetime, timezone
from typing import Any, Literal

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Response
from pydantic import BaseModel, Field

_SOURCE_URL = "https://ortho-reception-backend-v2.onrender.com/private/cockpit/reception-calls"
_HANDLED_URL = _SOURCE_URL + "/handled"
_MAX_BYTES = 2_000_000
_MAX_ROWS = 200


def _require_clinical_key(x_clinical_key: str | None = Header(default=None, alias="X-Clinical-Key")) -> None:
    """The already-established single-clinician key/session, no new actor owner."""
    expected = os.getenv("CLINICAL_DATA_KEY", "")
    if not expected:
        raise HTTPException(503, "Clinical access is disabled")
    if not x_clinical_key or not secrets.compare_digest(x_clinical_key, expected):
        raise HTTPException(401, "Clinical sign-in required")


def _settings() -> str:
    if os.getenv("RECEPTION_CALLS_READ_ENABLED") != "1":
        raise HTTPException(503, "Reception call peek is not activated")
    key = os.getenv("RECEPTION_CALLS_SOURCE_KEY", "")
    if not key:
        raise HTTPException(503, "Reception call source not configured")
    return key


async def _fetch_calls(source_key: str) -> dict[str, Any]:
    """Fetch only from the fixed, HTTPS Reception-owned scoped endpoint."""
    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
            response = await client.get(
                _SOURCE_URL,
                headers={"X-Cockpit-Calls-Key": source_key, "Accept": "application/json"},
            )
        if response.status_code != 200 or len(response.content) > _MAX_BYTES:
            raise ValueError("Reception source not qualified")
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Invalid Reception source")
        return data
    except (httpx.HTTPError, ValueError, TypeError):
        # Do not put source responses or credentials into logs/errors.
        raise HTTPException(503, "Reception call source unavailable") from None


async def _set_handled(source_key: str, todo_id: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
            result = await client.post(_HANDLED_URL, json={"todo_id": todo_id, "handled": True},
                headers={"X-Cockpit-Calls-Key": source_key, "Accept": "application/json"})
        if result.status_code != 200 or len(result.content) > 4096:
            raise ValueError("Reception handling not confirmed")
        body = result.json()
        if not isinstance(body, dict) or body.get("handled") is not True or body.get("todo_id") != todo_id:
            raise ValueError("Reception handling not confirmed")
    except (httpx.HTTPError, ValueError, TypeError):
        raise HTTPException(503, "Reception handling unavailable") from None


def _value(obj: dict, key: str, limit: int) -> str:
    v = obj.get(key)
    return v.strip()[:limit] if isinstance(v, str) else ""


def _timestamp(raw: str) -> datetime | None:
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return value.astimezone(timezone.utc) if value.tzinfo else None
    except (TypeError, ValueError):
        return None


def _project(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = data.get("calls")
    if not isinstance(rows, list) or len(rows) > _MAX_ROWS:
        raise HTTPException(503, "Reception call coverage unavailable")
    projected = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        when = _timestamp(_value(row, "logged_at", 64))
        if when is None:
            raise HTTPException(503, "Reception call coverage unavailable")
        # The Reception dashboard can legitimately contain calls without a
        # call ID. Keep these actual rows visible without minting an identity.
        call_id = _value(row, "call_id", 120)
        first = _value(row, "first_name", 100)
        last = _value(row, "last_name", 100)
        name = " ".join(p for p in (first, last) if p)
        if not name:
            name = _value(row, "caller_name", 160) or "Άγνωστος καλών"
        narrative_status = _value(row, "call_narrative_status", 24)
        narrative = _value(row, "call_narrative", 700) if narrative_status == "complete" else ""
        fallback = (_value(row, "display_message", 700)
                    or _value(row, "message_clean", 700)
                    or _value(row, "reason_for_call", 300))
        summary = narrative or fallback or "Δεν υπάρχει ακόμη διαθέσιμη σύνοψη."
        kind = _value(row, "type", 32)
        handled = row.get("handled")
        todo_id = row.get("todo_id")
        if "handled" not in row or "todo_id" not in row or handled not in (True, False, None) or (todo_id is not None and
                (not isinstance(todo_id, str) or not todo_id or len(todo_id) > 220)):
            raise HTTPException(503, "Reception handled state unavailable")
        # Reception tracks completion in its own separate todo/handled state.
        # A message or callback request is not proof of an open obligation.
        projected.append({
            "call_id": call_id or None, "logged_at": when.isoformat(),
            "name": name, "phone": _value(row, "phone", 40),
            "summary": summary, "summary_pending": narrative_status in ("pending", "processing", "retrying"),
            "kind": kind, "handled": handled, "todo_id": todo_id,
        })
    projected.sort(key=lambda x: x["logged_at"], reverse=True)
    # Only a small, transient protected peek. Complete state stays in Reception.
    return projected[:4]


class SetHandledRequest(BaseModel):
    todo_id: str = Field(min_length=1, max_length=220)
    handled: Literal[True]


def build_reception_calls_router() -> APIRouter:
    router = APIRouter(prefix="/clinical/reception-calls", tags=["clinical-reception-calls"])

    @router.get("/peek", dependencies=[Depends(_require_clinical_key)])
    async def peek(response: Response) -> dict[str, Any]:
        response.headers["Cache-Control"] = "private, no-store"
        key = _settings()
        data = await _fetch_calls(key)
        calls = _project(data)
        count = data.get("unhandled_count")
        coverage = data.get("coverage")
        total = data.get("total")
        if type(total) is not int or total != len(data["calls"]) or coverage not in (
            "available_call_log_max_200", "ambiguous_call_identity"
        ) or (
            count is not None and (type(count) is not int or not 0 <= count <= total)
        ) or (coverage == "ambiguous_call_identity" and count is not None) or (
            coverage == "available_call_log_max_200" and count is None
        ):
            raise HTTPException(503, "Reception count coverage unavailable")
        return {
            "source": "reception_dashboard",
            "coverage": coverage,
            "is_unread_count": False,
            "unhandled_count": count,
            "calls": calls,
        }

    @router.post("/handled", dependencies=[Depends(_require_clinical_key)])
    async def set_handled(body: SetHandledRequest, response: Response) -> dict[str, Any]:
        response.headers["Cache-Control"] = "private, no-store"
        key = _settings()
        await _set_handled(key, body.todo_id)
        return {"handled": True}

    return router
