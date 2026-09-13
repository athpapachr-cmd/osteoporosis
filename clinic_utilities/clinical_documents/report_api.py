from __future__ import annotations

import json

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import ValidationError

from clinic_utilities.physio_referral_runtime import _repo_root, _require_clinical_key

from .models import ClinicianProfile
from .report_ai import OpenAIReportProvider, provider_status, require_provider
from .report_models import (
    FinalMedicalReportV1,
    MedicalReportCaseV1,
    MedicalReportResearchRequestV1,
)
from .report_pdf import build_medical_report_pdf, medical_report_filename
from .report_sources import (
    MAX_EXTRACTED_CHARS,
    MAX_REPORT_FILES,
    MAX_REPORT_FILE_BYTES,
    MAX_REPORT_TOTAL_BYTES,
    build_clinician_context_source,
    deterministic_warnings,
    enforce_source_totals,
    extract_source_bytes,
    validate_analysis_references,
)
from .sick_leave import MAX_SIGNATURE_BYTES, content_disposition, validate_signature_image

MAX_CASE_JSON_BYTES = 32 * 1024
MAX_FINAL_JSON_BYTES = 256 * 1024


def _parse_json_model(raw_text: str, model, *, max_bytes: int, label: str):
    raw_bytes = raw_text.encode("utf-8")
    if len(raw_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Το {label} υπερβαίνει το επιτρεπτό μέγεθος")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Μη έγκυρο JSON για {label}") from exc
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        detail = [
            {"loc": [str(part) for part in item.get("loc", ())], "msg": item.get("msg", "Μη έγκυρη τιμή")}
            for item in exc.errors(include_input=False)
        ]
        raise HTTPException(status_code=422, detail=detail) from exc


def _clinician_profile() -> ClinicianProfile:
    try:
        return ClinicianProfile.from_environment()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


async def _signature_bytes(upload: UploadFile | None) -> bytes | None:
    if upload is None:
        return None
    content = await upload.read(MAX_SIGNATURE_BYTES + 1)
    try:
        validate_signature_image(content, upload.content_type or "", upload.filename or "")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return content


def _report_provider() -> OpenAIReportProvider:
    return OpenAIReportProvider()


def _provider_error(exc: RuntimeError) -> HTTPException:
    return HTTPException(status_code=503, detail=str(exc))


def build_medical_report_router() -> APIRouter:
    router = APIRouter(
        prefix="/clinical/clinic-utilities/medical-report",
        tags=["clinical-documents-medical-report-v1"],
        dependencies=[Depends(_require_clinical_key)],
    )
    page_path = _repo_root() / "static" / "clinic-utilities" / "medical-report" / "index.html"

    @router.get("", include_in_schema=False)
    def report_page() -> FileResponse:
        if not page_path.is_file():
            raise HTTPException(status_code=500, detail="Η σελίδα ιατρικών εκθέσεων δεν είναι διαθέσιμη")
        return FileResponse(page_path, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})

    @router.get("/api/contract")
    def report_contract():
        status = provider_status()
        return {
            "version": "medical_report_v1",
            "report_types": ["accident_medical_report", "medico_legal_expert_report"],
            "accepted_extensions": [".pdf", ".txt", ".md", ".docx"],
            "limits": {
                "max_files": MAX_REPORT_FILES,
                "max_file_bytes": MAX_REPORT_FILE_BYTES,
                "max_total_bytes": MAX_REPORT_TOTAL_BYTES,
                "max_extracted_chars": MAX_EXTRACTED_CHARS,
                "max_case_json_bytes": MAX_CASE_JSON_BYTES,
                "max_final_json_bytes": MAX_FINAL_JSON_BYTES,
                "max_signature_bytes": MAX_SIGNATURE_BYTES,
            },
            "persistence": {
                "patient_case_database": False,
                "browser_case_storage": False,
                "autosave": False,
                "source_files_persisted": False,
                "signature_persisted": False,
            },
            "ai": status,
            "clinician_configured": _contract_clinician_configured(),
        }

    @router.post("/api/analyze")
    async def analyze_report(
        case_json: str = Form(...),
        files: list[UploadFile] = File(default=[]),
    ):
        case = _parse_json_model(case_json, MedicalReportCaseV1, max_bytes=MAX_CASE_JSON_BYTES, label="την υπόθεση")
        if len(files) > MAX_REPORT_FILES:
            raise HTTPException(status_code=413, detail=f"Επιτρέπονται έως {MAX_REPORT_FILES} αρχεία")

        sources = []
        context_source = build_clinician_context_source(case.clinician_context)
        if context_source is not None:
            sources.append(context_source)

        total_bytes = 0
        for index, upload in enumerate(files, start=1):
            content = await upload.read(MAX_REPORT_FILE_BYTES + 1)
            if len(content) > MAX_REPORT_FILE_BYTES:
                raise HTTPException(status_code=413, detail=f"Το αρχείο {upload.filename or index} υπερβαίνει τα 12 MiB")
            total_bytes += len(content)
            if total_bytes > MAX_REPORT_TOTAL_BYTES:
                raise HTTPException(status_code=413, detail="Τα αρχεία υπερβαίνουν συνολικά τα 40 MiB")
            try:
                source = extract_source_bytes(content, f"src-upload-{index:03d}", upload.filename or f"source-{index}")
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=f"{upload.filename or index}: {exc}") from exc
            sources.append(source)

        try:
            enforce_source_totals(sources)
        except ValueError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        if not any(source.status == "extracted" for source in sources):
            raise HTTPException(status_code=422, detail="Δεν υπάρχει εξαγώγιμο κείμενο για δημιουργία έκθεσης")

        try:
            require_provider(for_identifiable_records=True)
            analysis, usage = _report_provider().analyze(case, sources)
            validate_analysis_references(analysis, sources)
        except RuntimeError as exc:
            raise _provider_error(exc) from exc
        except ValueError as exc:
            raise HTTPException(status_code=502, detail=f"Το AI επέστρεψε μη έγκυρη τεκμηρίωση: {exc}") from exc

        manifests = [
            {
                "source_id": source.source_id,
                "filename": source.filename,
                "source_type": source.source_type,
                "status": source.status,
                "page_count": source.page_count,
                "character_count": source.character_count,
            }
            for source in sources
        ]
        return {
            "case": case.model_dump(mode="json"),
            "sources": manifests,
            "analysis": analysis.model_dump(mode="json"),
            "deterministic_warnings": deterministic_warnings(analysis, sources),
            "usage": usage.model_dump(mode="json"),
        }

    @router.post("/api/research")
    def research_report(request: MedicalReportResearchRequestV1 = Body(...)):
        try:
            require_provider(for_identifiable_records=False)
            result = _report_provider().research(request.case, request.analysis)
        except RuntimeError as exc:
            raise _provider_error(exc) from exc
        return result.model_dump(mode="json")

    @router.post("/api/preview")
    async def preview_report(
        report_json: str = Form(...),
        signature: UploadFile | None = File(default=None),
    ):
        report = _parse_json_model(report_json, FinalMedicalReportV1, max_bytes=MAX_FINAL_JSON_BYTES, label="την τελική έκθεση")
        return _pdf_response(report, await _signature_bytes(signature), inline=True)

    @router.post("/api/pdf")
    async def final_report_pdf(
        report_json: str = Form(...),
        signature: UploadFile | None = File(default=None),
    ):
        report = _parse_json_model(report_json, FinalMedicalReportV1, max_bytes=MAX_FINAL_JSON_BYTES, label="την τελική έκθεση")
        return _pdf_response(report, await _signature_bytes(signature), inline=False)

    return router


def _contract_clinician_configured() -> bool:
    try:
        ClinicianProfile.from_environment()
        return True
    except ValueError:
        return False


def _pdf_response(report: FinalMedicalReportV1, signature_bytes: bytes | None, *, inline: bool) -> Response:
    try:
        content = build_medical_report_pdf(report, clinician=_clinician_profile(), signature_bytes=signature_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    filename = medical_report_filename(report)
    return Response(
        content=content,
        media_type="application/pdf",
        headers={
            "Content-Disposition": content_disposition(filename, inline=inline),
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


__all__ = ["build_medical_report_router"]
