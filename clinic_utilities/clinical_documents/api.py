from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import ValidationError

from clinic_utilities.physio_referral_runtime import _repo_root, _require_clinical_key

from .models import ClinicianProfile, SickLeaveDraftV1
from .sick_leave import (
    MAX_PREVIOUS_PDF_BYTES,
    MAX_SIGNATURE_BYTES,
    PDF_SCHEMA,
    PDF_VERSION,
    build_sick_leave_pdf,
    content_disposition,
    read_previous_sick_leave_pdf,
    reuse_options,
    sick_leave_filename,
    validate_signature_image,
)


MAX_DRAFT_JSON_BYTES = 16 * 1024


async def _read_signature(upload: UploadFile | None) -> bytes | None:
    if upload is None:
        return None
    content = await upload.read(MAX_SIGNATURE_BYTES + 1)
    try:
        validate_signature_image(content, upload.content_type or "", upload.filename or "")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return content


def _parse_draft(draft_json: str) -> SickLeaveDraftV1:
    if len(str(draft_json or "").encode("utf-8")) > MAX_DRAFT_JSON_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Το draft αναρρωτικής άδειας υπερβαίνει το επιτρεπτό μέγεθος",
        )
    try:
        raw = json.loads(draft_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="Μη έγκυρο JSON αναρρωτικής άδειας") from exc
    try:
        return SickLeaveDraftV1.model_validate(raw)
    except ValidationError as exc:
        detail = [
            {
                "loc": [str(part) for part in item.get("loc", ())],
                "msg": item.get("msg", "Μη έγκυρη τιμή"),
                "type": item.get("type", "validation_error"),
            }
            for item in exc.errors(include_input=False)
        ]
        raise HTTPException(status_code=422, detail=detail) from exc


def _clinician_profile() -> ClinicianProfile:
    try:
        return ClinicianProfile.from_environment()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _pdf_response(
    draft: SickLeaveDraftV1,
    *,
    clinician: ClinicianProfile,
    signature_bytes: bytes | None,
    inline: bool,
) -> Response:
    try:
        pdf_bytes, _ = build_sick_leave_pdf(draft, clinician=clinician, signature_bytes=signature_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    filename = sick_leave_filename(draft)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": content_disposition(filename, inline=inline),
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


def build_clinical_documents_router() -> APIRouter:
    router = APIRouter(
        prefix="/clinical/clinic-utilities/sick-leave",
        tags=["clinical-documents-sick-leave-v1"],
        dependencies=[Depends(_require_clinical_key)],
    )
    page_path = _repo_root() / "static" / "clinic-utilities" / "sick-leave" / "index.html"

    @router.get("", include_in_schema=False)
    def sick_leave_page() -> FileResponse:
        if not page_path.is_file():
            raise HTTPException(status_code=500, detail="Η σελίδα αναρρωτικής άδειας δεν είναι διαθέσιμη")
        return FileResponse(
            page_path,
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    @router.get("/api/contract")
    def sick_leave_contract():
        try:
            clinician = ClinicianProfile.from_environment()
            clinician_payload = {
                "name": clinician.name,
                "specialty": clinician.specialty,
                "phone": clinician.phone,
                "email": clinician.email,
                "clinic": clinician.clinic,
                "address": clinician.address,
            }
            configured = True
        except ValueError:
            clinician_payload = None
            configured = False
        return {
            "schema": PDF_SCHEMA,
            "version": PDF_VERSION,
            "identity_types": ["ADT", "ARC"],
            "clinician_configured": configured,
            "clinician": clinician_payload,
            "persistence": {
                "patient_database": False,
                "browser_patient_storage": False,
                "autosave": False,
                "signature_persisted": False,
            },
            "limits": {
                "draft_json_bytes": MAX_DRAFT_JSON_BYTES,
                "signature_bytes": MAX_SIGNATURE_BYTES,
                "previous_pdf_bytes": MAX_PREVIOUS_PDF_BYTES,
            },
        }

    @router.post("/api/preview")
    async def sick_leave_preview(
        draft_json: str = Form(...),
        signature: UploadFile | None = File(default=None),
    ):
        draft = _parse_draft(draft_json)
        signature_bytes = await _read_signature(signature)
        return _pdf_response(
            draft,
            clinician=_clinician_profile(),
            signature_bytes=signature_bytes,
            inline=True,
        )

    @router.post("/api/pdf")
    async def sick_leave_pdf(
        draft_json: str = Form(...),
        signature: UploadFile | None = File(default=None),
    ):
        draft = _parse_draft(draft_json)
        signature_bytes = await _read_signature(signature)
        return _pdf_response(
            draft,
            clinician=_clinician_profile(),
            signature_bytes=signature_bytes,
            inline=False,
        )

    @router.post("/api/import-previous")
    async def sick_leave_import_previous(previous_pdf: UploadFile = File(...)):
        filename = str(previous_pdf.filename or "").lower()
        content_type = str(previous_pdf.content_type or "").lower()
        if not filename.endswith(".pdf") or content_type not in {"application/pdf", "application/octet-stream"}:
            raise HTTPException(status_code=415, detail="Η προηγούμενη άδεια πρέπει να είναι PDF")
        content = await previous_pdf.read(MAX_PREVIOUS_PDF_BYTES + 1)
        if len(content) > MAX_PREVIOUS_PDF_BYTES:
            raise HTTPException(status_code=413, detail="Το προηγούμενο PDF υπερβαίνει τα 8 MB")
        try:
            metadata = read_previous_sick_leave_pdf(content)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return reuse_options(metadata)

    return router


__all__ = ["MAX_DRAFT_JSON_BYTES", "build_clinical_documents_router"]
