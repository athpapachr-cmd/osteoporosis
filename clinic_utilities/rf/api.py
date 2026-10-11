from __future__ import annotations

from calendar import monthrange
from datetime import date
import re
from typing import Any, Literal
import unicodedata

import fitz
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from clinic_utilities.physio_referral_runtime import _repo_root, _require_clinical_key
from .catalog import (
    INDICATIONS,
    DEFAULT_DOCTOR_KEY,
    LATERALITY_LABELS,
    PRODUCT_LABELS,
    RF_REASON_LABELS,
    DoctorProfile,
    product_catalog_from_environment,
)
from .parsers import parse_medications, parse_physio_dates
from .prior_pdf import extract_prior_rf_pdf, MAX_PRIOR_RF_PDF_BYTES
from .pdf import (
    MAX_HOROS_JPEG_BYTES,
    MAX_HOROS_JPEG_COUNT,
    MAX_HOROS_JPEG_TOTAL_BYTES,
    build_official_rf_pdf,
)
from .persistence import (
    get_procedure_history,
    initialize_rf_tables,
    RFDoctorProfileORM,
    DOCTOR_FIELDS,
    create_rf_doctor,
    list_rf_doctors,
    revise_rf_doctor,
    rf_doctor_snapshot,
    list_procedure_history,
    list_medication_aliases,
    record_application,
    record_legacy_procedure,
    upsert_medication_alias,
    delete_medication_alias,
    RFSavedDraftORM,
    search_rf_sources, get_rf_source,
    save_rf_draft, get_rf_draft, list_rf_drafts, delete_rf_draft,
    normalize_identity,
)

MAX_IMAGING_BYTES = 20 * 1024 * 1024


class RFDoctorInput(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    gesy_code: str = Field(min_length=1, max_length=80)
    specialty: str = Field(min_length=1, max_length=160)
    medical_center: str = Field(min_length=1, max_length=240)
    phone: str = Field(min_length=1, max_length=80)
    email: str = Field(min_length=3, max_length=240)

    @field_validator(*DOCTOR_FIELDS)
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip() or any(ord(ch) < 32 for ch in value):
            raise ValueError("Απαιτούνται μη κενά στοιχεία ιατρού")
        return value.strip()

    @field_validator("email")
    @classmethod
    def email_format(cls, value: str) -> str:
        if "@" not in value or "." not in value.rsplit("@", 1)[-1]:
            raise ValueError("Μη έγκυρο email ιατρού")
        return value


class RFDoctorRevision(BaseModel):
    revision: int = Field(ge=1)


class RFDoctorEdit(RFDoctorInput):
    revision: int = Field(ge=1)


class RFDoctorStatus(RFDoctorRevision):
    active: bool



class RFMedicationTrial(BaseModel):
    source_text: str = ""
    drug_name: str = Field(min_length=1, max_length=160)
    dose: str = Field(default="", max_length=80)
    duration: str = Field(default="", max_length=120)


class RFAdverseEffect(BaseModel):
    treatment: str = Field(min_length=1, max_length=160)
    effect: str = Field(min_length=1, max_length=240)


class RFIntervention(BaseModel):
    site: str = Field(min_length=1, max_length=160)
    date: str
    vas_before: int = Field(ge=0, le=10)
    vas_after: int = Field(ge=0, le=10)


class RFLegacyHistory(BaseModel):
    actual_procedure_date: str
    vas_before: int = Field(ge=0, le=10)
    vas_after: int = Field(ge=0, le=10)
    last_followup_date: str
    last_followup_vas: int = Field(ge=0, le=10)


class RFApplicationDraft(BaseModel):
    pathway: Literal["A1", "A2"]
    # Missing in older clients => legacy default; explicit null/blank is NOT defaulted.
    doctor_key: str = Field(default=DEFAULT_DOCTOR_KEY, min_length=1, max_length=40)
    patient_name: str = Field(min_length=1, max_length=200)
    identity_number: str = Field(min_length=1, max_length=128)
    gesy_number: str = Field(min_length=1, max_length=128)
    # Age zero was previously possible when an empty browser field was coerced
    # with Number(''). Reject it server-side as well as fixing the browser draft.
    age: int = Field(ge=1, le=130)
    product_key: str
    indication_code: str
    laterality: Literal["left", "right"]
    exact_location: str = Field(min_length=1, max_length=300)
    other_area: str = Field(default="", max_length=160)
    other_diagnosis: str = Field(default="", max_length=240)
    rf_reason_codes: list[str] = Field(default_factory=list, max_length=8)
    rf_reason_other: str = Field(default="", max_length=600)
    pain_onset_date: str = ""
    pain_onset_vas: int | None = Field(default=None, ge=0, le=10)
    last_assessment_date: str = ""
    last_assessment_vas: int | None = Field(default=None, ge=0, le=10)
    full_medication_text: str = Field(default="", max_length=30000)
    nsaid_trials: list[RFMedicationTrial] = Field(default_factory=list, max_length=3)
    other_analgesic_trials: list[RFMedicationTrial] = Field(default_factory=list, max_length=3)
    adverse_effects: list[RFAdverseEffect] = Field(default_factory=list, max_length=3)
    intervention: RFIntervention | None = None
    physio_dates_text: str = Field(default="", max_length=12000)
    additional_notes: str = Field(default="", max_length=4000)
    imaging_review_confirmed: bool = False
    procedure_history_id: str = Field(default="", max_length=80)
    legacy_history: RFLegacyHistory | None = None
    a2_current_followup_date: str = Field(default="", max_length=20)
    a2_current_followup_vas: int | None = Field(default=None, ge=0, le=10)
    source_origin: str = Field(default="", max_length=20)
    legacy_confirmed: bool = False
    source_kind: str = Field(default="", max_length=20)
    source_id: str = Field(default="", max_length=80)
    draft_id: str = Field(default="", max_length=80)
    draft_revision: int | None = Field(default=None, ge=1)


class RFHistoryLookup(BaseModel):
    identity_number: str = Field(min_length=1, max_length=128)
    site_key: str = Field(default="", max_length=80)
    laterality: Literal["left", "right"] | None = None



class RFPartialDraftBody(BaseModel):
    fields: dict[str, Any]


class RFPartialDraftUpdate(RFPartialDraftBody):
    revision: int = Field(ge=1)


class RFPartialDraftRevision(BaseModel):
    revision: int = Field(ge=1)


class RFTextRequest(BaseModel):
    text: str = Field(default="", max_length=30000)


class RFMedicationDictionaryEntry(BaseModel):
    alias: str = Field(min_length=1, max_length=120)
    category: Literal["nsaid", "other"]
    display_name: str = Field(default="", max_length=160)
    active_ingredient: str = Field(default="", max_length=160)


def _parse_iso_date(value: str, label: str) -> date:
    try:
        return date.fromisoformat(str(value or "").strip())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Μη έγκυρη ημερομηνία: {label}") from exc


def _add_calendar_months(value: date, months: int) -> date:
    idx = value.month - 1 + months
    year = value.year + idx // 12
    month = idx % 12 + 1
    day = min(value.day, monthrange(year, month)[1])
    return date(year, month, day)


def _reason_text(draft: RFApplicationDraft) -> str:
    if any(code not in RF_REASON_LABELS for code in draft.rf_reason_codes):
        raise HTTPException(status_code=422, detail="Μη έγκυρη αιτιολόγηση RF")
    parts = [RF_REASON_LABELS[code] for code in draft.rf_reason_codes]
    if draft.rf_reason_other.strip():
        parts.append(draft.rf_reason_other.strip())
    if not parts:
        raise HTTPException(status_code=422, detail="Απαιτείται λόγος επιλογής RF")
    return "; ".join(parts) + "."


def _resolve_other(draft: RFApplicationDraft, indication: dict):
    if not indication.get("official_other"):
        return "", ""
    area = str(indication.get("other_area") or draft.other_area).strip()
    diagnosis = str(indication.get("other_diagnosis") or draft.other_diagnosis).strip()
    if not area or not diagnosis:
        raise HTTPException(status_code=422, detail="Για το Άλλο απαιτούνται περιοχή και διάγνωση")
    return area, diagnosis


def _location_mode(indication_code: str) -> str:
    if indication_code == "OTHER_CUSTOM":
        return "custom"
    if indication_code == "MORTON_NEUROMA":
        return "refine"
    return "derived"


def _validate_exact_location(draft: RFApplicationDraft, indication: dict) -> str:
    actual = draft.exact_location.strip()
    expected = str((indication.get("location_labels") or {}).get(draft.laterality) or "").strip()
    mode = _location_mode(draft.indication_code)
    if mode == "derived":
        if not expected or actual != expected:
            raise HTTPException(
                status_code=422,
                detail="Η ακριβής εντόπιση προκύπτει αυτόματα από την ένδειξη και την πλευρά και δεν μπορεί να αλλάξει.",
            )
        return expected
    if mode == "refine":
        if not expected or not actual.startswith(expected):
            raise HTTPException(
                status_code=422,
                detail="Η εξειδίκευση της εντόπισης πρέπει να παραμένει στην επιλεγμένη πλευρά/περιοχή.",
            )
    return actual


def _resolve_medications(draft: RFApplicationDraft, learned_aliases: list[dict] | None = None):
    # The official table provides capacity for up to three NSAIDs and up to
    # three other analgesics. It is not a minimum-treatment requirement.
    parsed = parse_medications(draft.full_medication_text, learned_aliases or []) if draft.full_medication_text.strip() else {
        "auto_selected_nsaids": [],
        "auto_selected_others": [],
    }
    nsaid = ([x.model_dump() for x in draft.nsaid_trials] or parsed["auto_selected_nsaids"])[:3]
    other = ([x.model_dump() for x in draft.other_analgesic_trials] or parsed["auto_selected_others"])[:3]
    return nsaid, other


def _validate_a1(draft: RFApplicationDraft, indication: dict, learned_aliases: list[dict] | None = None):
    onset = _parse_iso_date(draft.pain_onset_date, "έναρξη πόνου")
    assessment = _parse_iso_date(draft.last_assessment_date, "τελευταία αξιολόγηση")
    if draft.pain_onset_vas is None or draft.last_assessment_vas is None:
        raise HTTPException(status_code=422, detail="Απαιτούνται οι δύο τιμές VAS")
    if assessment < _add_calendar_months(onset, 3):
        raise HTTPException(
            status_code=422,
            detail="Το νέο επίσημο A.1 απαιτεί πόνο που επιμένει για τουλάχιστον τρεις μήνες.",
        )

    intervention = draft.intervention.model_dump() if draft.intervention else None
    if indication.get("requires_intervention") and intervention is None:
        detail = (
            "Για αίτημα RF στο ισχίο απαιτείται η τεκμηρίωση του διαγνωστικού block του σημείου 8."
            if indication.get("intervention_kind") == "hip_diagnostic_block"
            else "Για ιερολαγόνια απαιτούνται σημείο, ημερομηνία και VAS πριν/μετά την έγχυση του σημείου 8."
        )
        raise HTTPException(status_code=422, detail=detail)
    if intervention:
        _parse_iso_date(intervention["date"], "παρέμβαση σημείου 8")

    physio = parse_physio_dates(draft.physio_dates_text)
    if draft.physio_dates_text.strip() and physio["invalid_or_ambiguous_tokens"]:
        raise HTTPException(
            status_code=422,
            detail="Υπάρχουν ασαφείς/μη έγκυρες ημερομηνίες φυσιοθεραπείας. Χρειάζεται πλήρες έτος.",
        )
    nsaid, other = _resolve_medications(draft, learned_aliases)
    return {
        "pain_onset_date": onset.isoformat(),
        "pain_onset_vas": draft.pain_onset_vas,
        "last_assessment_date": assessment.isoformat(),
        "last_assessment_vas": draft.last_assessment_vas,
        "nsaid_trials": nsaid,
        "other_analgesic_trials": other,
        "adverse_effects": [x.model_dump() for x in draft.adverse_effects],
        "intervention": intervention or {},
        "physio": physio,
        "rf_reason_text": _reason_text(draft),
    }


def _validate_legacy(history: RFLegacyHistory):
    procedure = _parse_iso_date(history.actual_procedure_date, "προηγούμενη εφαρμογή")
    followup = _parse_iso_date(history.last_followup_date, "τελευταία αξιολόγηση")
    if followup < procedure:
        raise HTTPException(status_code=422, detail="Η τελευταία αξιολόγηση δεν μπορεί να προηγείται της εφαρμογής")
    return {
        "actual_procedure_date": procedure.isoformat(),
        "vas_before": history.vas_before,
        "vas_after": history.vas_after,
        "last_followup_date": followup.isoformat(),
        "last_followup_vas": history.last_followup_vas,
    }


_IMAGING_TERMS = (
    "radiolog", "ακτινολογ", "ακτινογραφ", "x-ray", "x ray",
    "magnetic resonance", "μαγνητικ", "computed tomography", "αξονικ",
    "ultrasound", "υπερηχο", "densitometr", "οστικη πυκνοτητα",
)
_IMAGING_WORD_TERMS = ("mri", "ct", "dxa")
_LAB_TERMS = (
    "biochemistry", "haematology", "hematology", "serum", "plasma",
    "reference range", "τιμες αναφορας", "laboratory", "εργαστηρ",
    "validator", "mg/dl", "mmol/l", "calcium", "magnesium", "phosphate",
)


def _normalize_document_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", without_marks.casefold()).strip()


def _contains_word(text: str, word: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text) is not None


def _extract_pdf_text(content: bytes) -> str:
    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Το αρχείο απεικόνισης δεν είναι έγκυρο PDF") from exc
    try:
        if doc.page_count < 1:
            raise HTTPException(status_code=422, detail="Το αρχείο απεικόνισης δεν περιέχει σελίδες")
        parts = []
        total = 0
        for page in doc:
            try:
                text = page.get_text("text") or ""
            except Exception:
                text = ""
            if text:
                parts.append(text)
                total += len(text)
            if total >= 50000:
                break
        return "\n".join(parts)[:50000]
    finally:
        doc.close()


def _assess_imaging_pdf(content: bytes) -> dict:
    text = _normalize_document_text(_extract_pdf_text(content))
    imaging_hits = sum(term in text for term in _IMAGING_TERMS)
    imaging_hits += sum(_contains_word(text, term) for term in _IMAGING_WORD_TERMS)
    lab_hits = sum(term in text for term in _LAB_TERMS)

    if imaging_hits:
        return {
            "status": "imaging_supported",
            "requires_confirmation": False,
            "message": "Αναγνωρίστηκε απεικονιστική έκθεση. Ελέγξτε ότι αφορά το συγκεκριμένο αίτημα.",
        }
    if lab_hits >= 3:
        return {
            "status": "clearly_non_imaging",
            "requires_confirmation": False,
            "message": "Το PDF φαίνεται να είναι εργαστηριακή ή άλλη μη απεικονιστική εξέταση και δεν μπορεί να χρησιμοποιηθεί για το σημείο 3.",
        }
    return {
        "status": "ambiguous_or_unreadable",
        "requires_confirmation": True,
        "message": "Δεν μπορεί να επιβεβαιωθεί αυτόματα ο τύπος του PDF. Απαιτείται ρητή επιβεβαίωση ιατρού ότι είναι η απεικονιστική έκθεση του σημείου 3.",
    }


async def _read_imaging_pdf(upload: UploadFile) -> bytes:
    filename = str(upload.filename or "").lower()
    content_type = str(upload.content_type or "").lower()
    if not filename.endswith(".pdf") or content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="Η απεικονιστική έκθεση πρέπει να είναι PDF")
    content = await upload.read(MAX_IMAGING_BYTES + 1)
    if len(content) > MAX_IMAGING_BYTES:
        raise HTTPException(status_code=413, detail="Το PDF της απεικόνισης υπερβαίνει τα 20 MB")
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=422, detail="Το αρχείο απεικόνισης δεν είναι έγκυρο PDF")
    # Do not trust the magic bytes alone: require a parseable PDF with pages.
    _extract_pdf_text(content)
    return content


async def _read_horos_jpegs(uploads: list[UploadFile] | None) -> list[bytes]:
    """Strict byte/count ceilings before JPEG metadata inspection or decoding."""
    if not uploads:
        return []
    if len(uploads) > MAX_HOROS_JPEG_COUNT:
        raise HTTPException(status_code=413, detail="Υπερβολικά πολλές εικόνες Horos (μέγιστο 10)")
    contents: list[bytes] = []
    total = 0
    for number, upload in enumerate(uploads, start=1):
        filename = str(upload.filename or "").lower()
        content_type = str(upload.content_type or "").lower()
        if not filename.endswith((".jpg", ".jpeg")) or content_type not in {
            "image/jpeg", "image/pjpeg", "application/octet-stream"
        }:
            raise HTTPException(status_code=415, detail=f"Η εικόνα {number} πρέπει να είναι JPEG")
        content = await upload.read(MAX_HOROS_JPEG_BYTES + 1)
        if len(content) > MAX_HOROS_JPEG_BYTES:
            raise HTTPException(status_code=413, detail=f"Η εικόνα {number} υπερβαίνει τα 8 MB")
        total += len(content)
        if total > MAX_HOROS_JPEG_TOTAL_BYTES:
            raise HTTPException(status_code=413, detail="Οι εικόνες Horos υπερβαίνουν συνολικά τα 40 MB")
        if not content.startswith(b"\xff\xd8\xff") or not content.endswith(b"\xff\xd9"):
            raise HTTPException(status_code=422, detail=f"Η εικόνα {number} δεν είναι πλήρες JPEG")
        contents.append(content)
    return contents


def build_rf_router(engine: Engine) -> APIRouter:
    initialize_rf_tables(engine)
    router = APIRouter(
        prefix="/clinical/clinic-utilities/rf",
        tags=["clinic-utilities-rf-v2"],
        dependencies=[Depends(_require_clinical_key)],
    )
    page_path = _repo_root() / "static" / "clinic-utilities" / "rf" / "index.html"

    @router.get("", include_in_schema=False)
    def rf_page():
        if not page_path.is_file():
            raise HTTPException(status_code=500, detail="RF utility page is unavailable")
        return FileResponse(page_path, headers={"Cache-Control": "no-store"})

    @router.get("/api/contract")
    def rf_contract():
        doctors = {
            row["key"]: {"label": row["name"], "configured": row["active"]}
            for row in list_rf_doctors(engine)
        }
        doctor_ok = doctors.get(DEFAULT_DOCTOR_KEY, {}).get("configured", False)
        try:
            configured = set(product_catalog_from_environment())
        except ValueError:
            configured = set()
        return {
            "version": "rf-v2-category-a-2026-09",
            "category": "A",
            "application_target_rule": "single_unilateral",
            "pathways": {"A1": "Νέα θεραπεία", "A2": "Συνέχιση θεραπείας"},
            "indications": {
                code: {
                    "label": item["label"],
                    "site_key": item["site_key"],
                    "requires_intervention": bool(item.get("requires_intervention")),
                    "location_labels": dict(item.get("location_labels") or {}),
                    "location_mode": _location_mode(code),
                }
                for code, item in INDICATIONS.items()
            },
            "laterality": LATERALITY_LABELS,
            "rf_reasons": RF_REASON_LABELS,
            "products": {
                key: {"label": label, "configured": key in configured}
                for key, label in PRODUCT_LABELS.items()
            },
            "doctor_configured": doctor_ok,  # retained for older RF clients
            "doctors": doctors,  # labels/configuration only; no private profile fields
        }

    # RF directory is part of the existing protected clinical utility.
    @router.get("/api/doctors")
    def rf_doctors():
        return {"doctors": list_rf_doctors(engine)}

    @router.post("/api/doctors", status_code=201)
    def rf_add_doctor(req: RFDoctorInput):
        try:
            return create_rf_doctor(engine, req.model_dump())
        except IntegrityError as exc:
            raise HTTPException(status_code=409, detail="Υπάρχει ήδη ιατρός με αυτόν τον κωδικό ΓεΣΥ") from exc

    @router.put("/api/doctors/{doctor_key}")
    def rf_edit_doctor(doctor_key: str, req: RFDoctorEdit):
        try:
            return revise_rf_doctor(engine, doctor_key, req.revision, data=req.model_dump(exclude={"revision"}))
        except LookupError as exc:
            raise HTTPException(status_code=404, detail="Ο ιατρός δεν βρέθηκε") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail="Τα στοιχεία ιατρού άλλαξαν — ανανέωσε τη λίστα") from exc
        except IntegrityError as exc:
            raise HTTPException(status_code=409, detail="Υπάρχει ήδη ιατρός με αυτόν τον κωδικό ΓεΣΥ") from exc

    @router.post("/api/doctors/{doctor_key}/status")
    def rf_set_doctor_status(doctor_key: str, req: RFDoctorStatus):
        try:
            return revise_rf_doctor(engine, doctor_key, req.revision, active=req.active)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail="Ο ιατρός δεν βρέθηκε") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail="Τα στοιχεία ιατρού άλλαξαν — ανανέωσε τη λίστα") from exc


    # Previous RF sources are NOT prior procedures unless explicitly identified
    # as a real clinician-recorded RF application event.
    @router.get("/api/reuse/search")
    def rf_reuse_search(query: str = Query(min_length=2, max_length=150),
                        page: int = Query(default=1, ge=1, le=100)):
        try:
            return search_rf_sources(engine, query, page)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get("/api/reuse/{kind}/{source_id}")
    def rf_reuse_source(kind: Literal["application", "procedure"], source_id: str):
        result = get_rf_source(engine, kind, source_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Δεν βρέθηκε η παλιά RF εγγραφή")
        return result

    @router.post("/api/reuse/pdf")
    async def rf_reuse_pdf(previous_rf_pdf: UploadFile = File(...)):
        if not str(previous_rf_pdf.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=415, detail="Χρειάζεται παλιό RF PDF")
        content = await previous_rf_pdf.read(MAX_PRIOR_RF_PDF_BYTES + 1)
        if len(content) > MAX_PRIOR_RF_PDF_BYTES:
            raise HTTPException(status_code=413, detail="Η παλιά RF αίτηση είναι μεγαλύτερη από 20 MB")
        try:
            return extract_prior_rf_pdf(content)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get("/api/drafts")
    def rf_list_drafts(query: str = Query(default="", max_length=150),
                       page: int = Query(default=1, ge=1, le=100)):
        try:
            return list_rf_drafts(engine, query, page)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/api/drafts", status_code=201)
    def rf_save_draft(req: RFPartialDraftBody):
        try:
            return save_rf_draft(engine, req.fields)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get("/api/drafts/{draft_id}")
    def rf_open_draft(draft_id: str):
        row = get_rf_draft(engine, draft_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Το πρόχειρο δεν βρέθηκε")
        return row

    @router.put("/api/drafts/{draft_id}")
    def rf_update_draft(draft_id: str, req: RFPartialDraftUpdate):
        try:
            return save_rf_draft(engine, req.fields, draft_id, req.revision)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.delete("/api/drafts/{draft_id}")
    def rf_delete_draft(draft_id: str, req: RFPartialDraftRevision):
        try:
            delete_rf_draft(engine, draft_id, req.revision)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"deleted": True}

    @router.post("/api/history")
    def rf_history(req: RFHistoryLookup):
        if req.site_key and req.site_key not in {x["site_key"] for x in INDICATIONS.values()}:
            raise HTTPException(status_code=422, detail="Μη έγκυρη περιοχή RF")
        if req.laterality and req.laterality not in LATERALITY_LABELS:
            raise HTTPException(status_code=422, detail="Μη έγκυρη πλευρά")
        rows = list_procedure_history(
            engine,
            req.identity_number,
            site_key=req.site_key,
            laterality=req.laterality or "",
        )
        return {"found": bool(rows), "procedures": rows}

    @router.get("/api/medication-dictionary")
    def rf_medication_dictionary():
        return {"entries": list_medication_aliases(engine)}

    @router.post("/api/medication-dictionary")
    def rf_upsert_medication_dictionary(req: RFMedicationDictionaryEntry):
        try:
            entry = upsert_medication_alias(
                engine,
                alias=req.alias,
                category=req.category,
                display_name=req.display_name,
                active_ingredient=req.active_ingredient,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"entry": entry}

    @router.delete("/api/medication-dictionary/{alias_id}")
    def rf_delete_medication_dictionary(alias_id: str):
        if not delete_medication_alias(engine, alias_id):
            raise HTTPException(status_code=404, detail="Το εκμαθημένο φάρμακο δεν βρέθηκε")
        return {"deleted": True}

    @router.post("/api/parse-medications")
    def rf_parse_medications(req: RFTextRequest):
        return parse_medications(req.text, list_medication_aliases(engine))

    @router.post("/api/parse-physio")
    def rf_parse_physio(req: RFTextRequest):
        return parse_physio_dates(req.text)

    @router.post("/api/validate-imaging")
    async def rf_validate_imaging(imaging_report: UploadFile = File(...)):
        imaging_bytes = await _read_imaging_pdf(imaging_report)
        assessment = _assess_imaging_pdf(imaging_bytes)
        return {
            "status": assessment["status"],
            "requires_confirmation": assessment["requires_confirmation"],
            "message": assessment["message"],
        }

    @router.post("/api/create")
    async def rf_create(
        draft_json: str = Form(...),
        imaging_report: UploadFile = File(...),
        horos_jpegs: list[UploadFile] | None = File(None),
    ):
        try:
            draft = RFApplicationDraft.model_validate_json(draft_json)
        except Exception as exc:
            raise HTTPException(status_code=422, detail="Μη έγκυρο RF draft") from exc

        indication = INDICATIONS.get(draft.indication_code)
        if not indication:
            raise HTTPException(status_code=422, detail="Μη έγκυρη ένδειξη RF")
        if draft.product_key not in PRODUCT_LABELS:
            raise HTTPException(status_code=422, detail="Μη έγκυρο προϊόν RF")
        if draft.laterality not in LATERALITY_LABELS:
            raise HTTPException(status_code=422, detail="Μη έγκυρη πλευρά")

        other_area, other_diagnosis = _resolve_other(draft, indication)
        exact_location = _validate_exact_location(draft, indication)
        imaging_bytes = await _read_imaging_pdf(imaging_report)
        imaging_assessment = _assess_imaging_pdf(imaging_bytes)
        if imaging_assessment["status"] == "clearly_non_imaging":
            raise HTTPException(status_code=422, detail=imaging_assessment["message"])
        if imaging_assessment["requires_confirmation"] and not draft.imaging_review_confirmed:
            raise HTTPException(status_code=422, detail=imaging_assessment["message"])
        jpeg_bytes = await _read_horos_jpegs(horos_jpegs)
        imaging_review = (
            "auto_supported"
            if imaging_assessment["status"] == "imaging_supported"
            else "clinician_confirmed"
        )
        payload = {
            **draft.model_dump(exclude={"legacy_history", "imaging_review_confirmed", "draft_id", "draft_revision"}),
            "site_key": indication["site_key"],
            "imaging_attachment_review": imaging_review,
            "other_area": other_area,
            "other_diagnosis": other_diagnosis,
            "exact_location": exact_location,
            "application_date": date.today().isoformat(),
        }

        # A selected application is a source of reusable fields, NEVER proof of
        # treatment. Explicitly reject forged cross-patient/source links.
        if bool(draft.source_kind) != bool(draft.source_id):
            raise HTTPException(status_code=422, detail="Ατελής αναφορά παλιάς RF αίτησης")
        if draft.source_kind:
            if draft.source_kind not in {"application", "procedure"}:
                raise HTTPException(status_code=422, detail="Μη έγκυρη πηγή παλιάς RF")
            source = get_rf_source(engine, draft.source_kind, draft.source_id)
            if source is None:
                raise HTTPException(status_code=404, detail="Δεν βρέθηκε η προηγούμενη RF πηγή")
            identity = source["fields"].get("identity_number") or ""
            if not identity or normalize_identity(identity) != normalize_identity(draft.identity_number):
                raise HTTPException(status_code=422, detail="Η παλιά RF πηγή αφορά άλλον ασθενή")
            previous_site = source["source"]["site_key"]
            previous_side = source["source"]["laterality"]
            if previous_site != indication["site_key"] or previous_side != draft.laterality:
                raise HTTPException(status_code=422, detail="Η προηγούμενη RF πηγή αφορά άλλη περιοχή/πλευρά")

        prior = None
        legacy = None

        if draft.pathway == "A1":
            payload.update(_validate_a1(draft, indication, list_medication_aliases(engine)))
        else:
            if draft.procedure_history_id and draft.legacy_history is not None:
                raise HTTPException(
                    status_code=422,
                    detail="Επιλέξτε υπάρχον ιστορικό ή legacy καταχώρηση, όχι και τα δύο",
                )
            if draft.procedure_history_id:
                prior = get_procedure_history(engine, draft.procedure_history_id, draft.identity_number)
                if prior is None:
                    raise HTTPException(status_code=404, detail="Η προηγούμενη εφαρμογή δεν βρέθηκε")
                if prior["site_key"] != indication["site_key"]:
                    raise HTTPException(status_code=422, detail="Η προηγούμενη εφαρμογή αφορά διαφορετική περιοχή")
                if draft.laterality != prior.get("laterality"):
                    raise HTTPException(status_code=422, detail="Η προηγούμενη εφαρμογή αφορά διαφορετική πλευρά")
            elif draft.legacy_history is not None:
                if draft.source_origin == "pdf" and not draft.legacy_confirmed:
                    raise HTTPException(status_code=422,
                        detail="Επιβεβαίωσε ρητά ότι έγινε η πραγματική προηγούμενη εφαρμογή")
                prior = _validate_legacy(draft.legacy_history)
                legacy = {
                    **prior,
                    "identity_number": draft.identity_number,
                    "indication_code": draft.indication_code,
                    "site_key": indication["site_key"],
                    "laterality": draft.laterality,
                    "exact_location": exact_location,
                }
            else:
                raise HTTPException(status_code=422, detail="Το A.2 απαιτεί προηγούμενη πραγματική εφαρμογή RF")
            if bool(draft.a2_current_followup_date) != (draft.a2_current_followup_vas is not None):
                raise HTTPException(status_code=422,
                    detail="Συμπλήρωσε μαζί ημερομηνία και VAS της νέας αξιολόγησης")
            if draft.a2_current_followup_date:
                current_date = _parse_iso_date(draft.a2_current_followup_date, "τρέχουσα αξιολόγηση")
                if current_date < _parse_iso_date(prior["actual_procedure_date"], "πραγματική εφαρμογή"):
                    raise HTTPException(status_code=422,
                        detail="Η νέα αξιολόγηση προηγείται της πραγματικής εφαρμογής")
                prior = {**prior, "last_followup_date": current_date.isoformat(),
                         "last_followup_vas": draft.a2_current_followup_vas}

        try:
            product = product_catalog_from_environment()[draft.product_key]
        except ValueError as exc:
            raise HTTPException(status_code=503, detail="Η σταθερή ρύθμιση RF δεν είναι πλήρης") from exc

        # A row lock on PostgreSQL keeps the selected six-field issuer
        # snapshot consistent with concurrent protected edit/deactivation.
        # Commit both application and optional A.2 legacy row atomically.
        with Session(engine) as session:
            linked_draft = None
            if draft.draft_id:
                linked_draft = session.scalar(select(RFSavedDraftORM).where(
                    RFSavedDraftORM.id == draft.draft_id).with_for_update())
                if linked_draft is None or linked_draft.revision != draft.draft_revision:
                    raise HTTPException(status_code=409, detail="Το πρόχειρο άλλαξε ή έχει ήδη ολοκληρωθεί")
                saved_identity = str(linked_draft.form_state.get("identity_number") or "")
                if saved_identity and normalize_identity(saved_identity) != normalize_identity(draft.identity_number):
                    raise HTTPException(status_code=422, detail="Το πρόχειρο ανήκει σε άλλη ταυτότητα")
                if linked_draft.pathway != draft.pathway:
                    raise HTTPException(status_code=409, detail="Αποθήκευσε ξανά το πρόχειρο μετά την αλλαγή τύπου")
            row = session.scalar(select(RFDoctorProfileORM).where(
                RFDoctorProfileORM.key == draft.doctor_key).with_for_update())
            if row is None or not row.active:
                raise HTTPException(status_code=422, detail="Ο ιατρός δεν είναι ενεργός ή δεν υπάρχει")
            snapshot = rf_doctor_snapshot(row)
            doctor = DoctorProfile(**snapshot)
            payload["doctor_snapshot"] = snapshot
            try:
                pdf_bytes = build_official_rf_pdf(
                    payload,
                    doctor=doctor,
                    product=product,
                    radiology_pdf_bytes=imaging_bytes,
                    prior_history=prior,
                    horos_jpeg_bytes=jpeg_bytes,
                )
            except FileNotFoundError as exc:
                raise HTTPException(status_code=503, detail="Λείπει το επίσημο RF PDF template") from exc
            except (RuntimeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            if legacy is not None:
                record_legacy_procedure(engine, legacy, session=session)
            application_id = record_application(engine, payload, session=session)
            if linked_draft is not None:
                session.delete(linked_draft)
            session.commit()
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'inline; filename="RF_{application_id}.pdf"',
                "Cache-Control": "no-store",
                "X-RF-Application-Id": application_id,
            },
        )

    return router
