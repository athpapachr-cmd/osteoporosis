from __future__ import annotations

import os
import re
import secrets
from typing import Mapping
from urllib.parse import parse_qs

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response


RF_UPSTREAM_ORIGIN = "https://ortho-reception-backend-v2.onrender.com"
RF_GATEWAY_PREFIX = "/clinical/clinic-utilities/rf"
RF_GATEWAY_KEY_ENV = "RF_GATEWAY_ACCESS_KEY"
RF_APPLICATION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,80}$")
RF_GATEWAY_MAX_BODY_BYTES = 24 * 1024 * 1024
RF_HISTORY_MAX_BODY_BYTES = 4096
RF_GATEWAY_TIMEOUT_SECONDS = 45.0

_RF_FORM_ACTION = 'action="/rf/create"'
_RF_HISTORY_FETCH = "fetch('/rf/history?' + query.toString(), { credentials: 'same-origin' })"


def _require_clinical_key(x_clinical_key: str = Header(default="")) -> None:
    """Reuse the existing protected-clinical header contract.

    Browser sessions receive this header only after ``ClinicalCookieMiddleware``
    validates the HttpOnly ``clinical_session`` cookie. Direct API callers may
    still use the existing explicit clinical key contract.
    """

    expected = os.getenv("CLINICAL_DATA_KEY", "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="Clinical data access is not configured")
    if not x_clinical_key or not secrets.compare_digest(x_clinical_key, expected):
        raise HTTPException(status_code=401, detail="Unauthorized clinical data access")


def _gateway_access_key() -> str:
    key = os.getenv(RF_GATEWAY_KEY_ENV, "").strip()
    if not key:
        raise HTTPException(
            status_code=503,
            detail="Η ασφαλής σύνδεση με τη λειτουργία ραδιοκυμάτων δεν έχει ρυθμιστεί.",
        )
    return key


def _upstream_headers(access_key: str, content_type: str | None = None) -> dict[str, str]:
    """Build the only headers allowed to cross the service boundary."""

    headers = {
        "X-RF-Key": access_key,
        "Accept": "*/*",
    }
    if content_type:
        headers["Content-Type"] = content_type
    return headers


async def _send_upstream(
    method: str,
    path: str,
    *,
    access_key: str,
    params: Mapping[str, str] | None = None,
    content: bytes | None = None,
    content_type: str | None = None,
) -> httpx.Response:
    try:
        async with httpx.AsyncClient(
            timeout=RF_GATEWAY_TIMEOUT_SECONDS,
            follow_redirects=False,
        ) as client:
            return await client.request(
                method,
                RF_UPSTREAM_ORIGIN + path,
                params=params,
                content=content,
                headers=_upstream_headers(access_key, content_type),
            )
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="Η λειτουργία ραδιοκυμάτων δεν ανταποκρίθηκε εγκαίρως.",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail="Η λειτουργία ραδιοκυμάτων δεν είναι προσωρινά διαθέσιμη.",
        ) from exc


async def _request_upstream(
    method: str,
    path: str,
    *,
    params: Mapping[str, str] | None = None,
    content: bytes | None = None,
    content_type: str | None = None,
) -> httpx.Response:
    return await _send_upstream(
        method,
        path,
        access_key=_gateway_access_key(),
        params=params,
        content=content,
        content_type=content_type,
    )


def _raise_for_upstream_service_failure(response: httpx.Response) -> None:
    # Do not expose upstream credential/config diagnostics to the browser.
    if response.status_code in {401, 403}:
        raise HTTPException(
            status_code=502,
            detail="Η ασφαλής πρόσβαση στη λειτουργία ραδιοκυμάτων απέτυχε.",
        )
    if response.status_code >= 500:
        raise HTTPException(
            status_code=502,
            detail="Η λειτουργία ραδιοκυμάτων δεν είναι προσωρινά διαθέσιμη.",
        )


def _rewrite_form_routes(html: str) -> str:
    """Adapt only the RF transport seams needed by this gateway.

    History identifiers are intentionally moved from a browser GET query string
    into a same-origin POST body so they do not create an additional identifier-
    bearing URL/access-log surface in the Osteoporosis service.
    """

    if _RF_FORM_ACTION not in html or _RF_HISTORY_FETCH not in html:
        raise HTTPException(
            status_code=502,
            detail="Η φόρμα ραδιοκυμάτων δεν είναι συμβατή με την ασφαλή πύλη.",
        )

    history_post = (
        f"fetch('{RF_GATEWAY_PREFIX}/history', {{ method: 'POST', "
        "credentials: 'same-origin', "
        "headers: { 'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8' }, "
        "body: query.toString() })"
    )
    return html.replace(
        _RF_FORM_ACTION,
        f'action="{RF_GATEWAY_PREFIX}/create"',
        1,
    ).replace(
        _RF_HISTORY_FETCH,
        history_post,
        1,
    )


def _relay_response(
    upstream: httpx.Response,
    *,
    include_content_disposition: bool = False,
) -> Response:
    headers = {"Cache-Control": "no-store"}
    content_type = upstream.headers.get("content-type", "application/octet-stream")
    headers["Content-Type"] = content_type
    if include_content_disposition:
        disposition = upstream.headers.get("content-disposition", "").strip()
        if disposition:
            headers["Content-Disposition"] = disposition
    return Response(content=upstream.content, status_code=upstream.status_code, headers=headers)


def _bounded_application_id(application_id: str) -> str:
    if not RF_APPLICATION_ID_RE.fullmatch(application_id):
        raise HTTPException(status_code=404, detail="Η αίτηση δεν βρέθηκε.")
    return application_id


def _history_value(values: dict[str, list[str]], name: str, max_length: int) -> str:
    items = values.get(name, [""])
    if len(items) != 1:
        raise HTTPException(status_code=400, detail="Μη έγκυρο αίτημα ιστορικού RF.")
    value = items[0].strip()
    if len(value) > max_length:
        raise HTTPException(status_code=422, detail="Μη έγκυρο αίτημα ιστορικού RF.")
    return value


async def _parse_history_body(request: Request) -> dict[str, str]:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "application/x-www-form-urlencoded":
        raise HTTPException(status_code=415, detail="Απαιτείται form-urlencoded αίτημα ιστορικού RF.")

    declared_length = request.headers.get("content-length", "").strip()
    if declared_length:
        try:
            if int(declared_length) > RF_HISTORY_MAX_BODY_BYTES:
                raise HTTPException(status_code=413, detail="Το αίτημα ιστορικού RF είναι πολύ μεγάλο.")
        except ValueError:
            raise HTTPException(status_code=400, detail="Μη έγκυρο Content-Length.")

    body = await request.body()
    if len(body) > RF_HISTORY_MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Το αίτημα ιστορικού RF είναι πολύ μεγάλο.")
    try:
        values = parse_qs(body.decode("utf-8"), keep_blank_values=True, max_num_fields=8)
    except (UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Μη έγκυρο αίτημα ιστορικού RF.") from exc

    return {
        "gesy_number": _history_value(values, "gesy_number", 128),
        "identity_number": _history_value(values, "identity_number", 128),
        "application_location": _history_value(values, "application_location", 256),
    }


def build_rf_gateway_router() -> APIRouter:
    router = APIRouter(
        prefix=RF_GATEWAY_PREFIX,
        tags=["clinic-utilities-rf"],
        dependencies=[Depends(_require_clinical_key)],
    )

    @router.get("", response_class=HTMLResponse)
    async def rf_form_gateway() -> HTMLResponse:
        upstream = await _request_upstream("GET", "/rf")
        _raise_for_upstream_service_failure(upstream)
        if upstream.status_code >= 400:
            return HTMLResponse(
                content=upstream.text,
                status_code=upstream.status_code,
                headers={"Cache-Control": "no-store"},
            )
        html = _rewrite_form_routes(upstream.text)
        return HTMLResponse(
            content=html,
            status_code=upstream.status_code,
            headers={"Cache-Control": "no-store"},
        )

    @router.post("/history")
    async def rf_history_gateway(request: Request) -> Response:
        params = await _parse_history_body(request)
        upstream = await _request_upstream(
            "GET",
            "/rf/history",
            params=params,
        )
        _raise_for_upstream_service_failure(upstream)
        return _relay_response(upstream)

    @router.post("/create")
    async def rf_create_gateway(request: Request) -> Response:
        declared_length = request.headers.get("content-length", "").strip()
        if declared_length:
            try:
                if int(declared_length) > RF_GATEWAY_MAX_BODY_BYTES:
                    raise HTTPException(status_code=413, detail="Το αίτημα RF είναι πολύ μεγάλο.")
            except ValueError:
                raise HTTPException(status_code=400, detail="Μη έγκυρο Content-Length.")

        body = await request.body()
        if len(body) > RF_GATEWAY_MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="Το αίτημα RF είναι πολύ μεγάλο.")

        content_type = request.headers.get("content-type", "").strip()
        if not content_type.lower().startswith("multipart/form-data"):
            raise HTTPException(status_code=415, detail="Απαιτείται multipart/form-data.")

        upstream = await _request_upstream(
            "POST",
            "/rf/create",
            content=body,
            content_type=content_type,
        )
        _raise_for_upstream_service_failure(upstream)

        if upstream.status_code == 303:
            location = upstream.headers.get("location", "")
            prefix = "/rf/pdf/"
            if not location.startswith(prefix):
                raise HTTPException(
                    status_code=502,
                    detail="Μη έγκυρη απόκριση από τη λειτουργία ραδιοκυμάτων.",
                )
            application_id = _bounded_application_id(location[len(prefix) :])
            response = RedirectResponse(
                url=f"{RF_GATEWAY_PREFIX}/pdf/{application_id}",
                status_code=303,
            )
            response.headers["Cache-Control"] = "no-store"
            return response

        if 300 <= upstream.status_code < 400:
            raise HTTPException(
                status_code=502,
                detail="Μη αναμενόμενη ανακατεύθυνση από τη λειτουργία ραδιοκυμάτων.",
            )
        return _relay_response(upstream)

    @router.get("/pdf/{application_id}")
    async def rf_pdf_gateway(application_id: str) -> Response:
        bounded_id = _bounded_application_id(application_id)
        upstream = await _request_upstream("GET", f"/rf/pdf/{bounded_id}")
        _raise_for_upstream_service_failure(upstream)
        return _relay_response(upstream, include_content_disposition=True)

    return router


__all__ = [
    "RF_GATEWAY_KEY_ENV",
    "RF_GATEWAY_PREFIX",
    "RF_UPSTREAM_ORIGIN",
    "build_rf_gateway_router",
]
