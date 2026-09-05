from __future__ import annotations

from calendar import monthrange
from datetime import date
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.engine import Engine

from clinic_utilities.physio_referral_runtime import _repo_root, _require_clinical_key
from .catalog import INDICATIONS, LATERALITY_LABELS, PRODUCT_LABELS, RF_REASON_LABELS, DoctorProfile, product_catalog_from_environment
from .parsers import parse_medications, parse_physio_dates
from .pdf import build_official_rf_pdf
from .persistence import get_procedure_history, initialize_rf_tables, list_procedure_history, record_application, record_legacy_procedure

MAX_IMAGING_BYTES = 20 * 1024 * 1024

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
    patient_name: str = Field(min_length=1, max_length=200)
    identity_number: str = Field(min_length=1, max_length=128)
    gesy_number: str = Field(min_length=1, max_length=128)
    age: int = Field(ge=0, le=130)
    product_key: str
    indication_code: str
    laterality: str = "none"
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
    procedure_history_id: str = Field(default="", max_length=80)
    legacy_history: RFLegacyHistory | None = None

class RFHistoryLookup(BaseModel):
    identity_number: str = Field(min_length=1, max_length=128)
    site_key: str = Field(default="", max_length=80)
    laterality: str = Field(default="", max_length=40)

class RFTextRequest(BaseModel):
    text: str = Field(default="", max_length=30000)

def _parse_iso_date(value: str, label: str) -> date:
    try: return date.fromisoformat(str(value or "").strip())
    except ValueError as exc: raise HTTPException(status_code=422, detail=f"Μη έγκυρη ημερομηνία: {label}") from exc

def _add_calendar_months(value: date, months: int) -> date:
    idx=value.month-1+months; year=value.year+idx//12; month=idx%12+1; day=min(value.day,monthrange(year,month)[1]); return date(year,month,day)

def _reason_text(draft: RFApplicationDraft) -> str:
    if any(code not in RF_REASON_LABELS for code in draft.rf_reason_codes): raise HTTPException(status_code=422,detail="Μη έγκυρη αιτιολόγηση RF")
    parts=[RF_REASON_LABELS[code] for code in draft.rf_reason_codes]
    if draft.rf_reason_other.strip(): parts.append(draft.rf_reason_other.strip())
    if not parts: raise HTTPException(status_code=422,detail="Απαιτείται λόγος επιλογής RF")
    return "; ".join(parts)+"."

def _resolve_other(draft, indication):
    if not indication.get("official_other"): return "",""
    area=str(indication.get("other_area") or draft.other_area).strip(); diagnosis=str(indication.get("other_diagnosis") or draft.other_diagnosis).strip()
    if not area or not diagnosis: raise HTTPException(status_code=422,detail="Για το Άλλο απαιτούνται περιοχή και διάγνωση")
    return area,diagnosis

def _resolve_medications(draft):
    if not draft.full_medication_text.strip(): raise HTTPException(status_code=422,detail="Απαιτείται η πλήρης φαρμακευτική αγωγή")
    parsed=parse_medications(draft.full_medication_text); nsaid=[x.model_dump() for x in draft.nsaid_trials]; other=[x.model_dump() for x in draft.other_analgesic_trials]
    return (nsaid or parsed["auto_selected_nsaids"])[:3], (other or parsed["auto_selected_others"])[:3]

def _validate_a1(draft, indication):
    onset=_parse_iso_date(draft.pain_onset_date,"έναρξη πόνου"); assessment=_parse_iso_date(draft.last_assessment_date,"τελευταία αξιολόγηση")
    if draft.pain_onset_vas is None or draft.last_assessment_vas is None: raise HTTPException(status_code=422,detail="Απαιτούνται οι δύο τιμές VAS")
    if assessment < _add_calendar_months(onset,3): raise HTTPException(status_code=422,detail="Το νέο επίσημο A.1 απαιτεί πόνο που επιμένει για τουλάχιστον τρεις μήνες.")
    intervention=draft.intervention.model_dump() if draft.intervention else None
    if indication.get("requires_intervention") and intervention is None:
        detail="Για αίτημα RF στο ισχίο απαιτείται η τεκμηρίωση του διαγνωστικού block του σημείου 8." if indication.get("intervention_kind")=="hip_diagnostic_block" else "Για ιερολαγόνια απαιτούνται σημείο, ημερομηνία και VAS πριν/μετά την έγχυση του σημείου 8."
        raise HTTPException(status_code=422,detail=detail)
    if intervention: _parse_iso_date(intervention["date"],"παρέμβαση σημείου 8")
    physio=parse_physio_dates(draft.physio_dates_text)
    if draft.physio_dates_text.strip() and physio["invalid_or_ambiguous_tokens"]: raise HTTPException(status_code=422,detail="Υπάρχουν ασαφείς/μη έγκυρες ημερομηνίες φυσιοθεραπείας. Χρειάζεται πλήρες έτος.")
    nsaid,other=_resolve_medications(draft)
    return {"pain_onset_date":onset.isoformat(),"pain_onset_vas":draft.pain_onset_vas,"last_assessment_date":assessment.isoformat(),"last_assessment_vas":draft.last_assessment_vas,"nsaid_trials":nsaid,"other_analgesic_trials":other,"adverse_effects":[x.model_dump() for x in draft.adverse_effects],"intervention":intervention or {},"physio":physio,"rf_reason_text":_reason_text(draft)}

def _validate_legacy(history):
    procedure=_parse_iso_date(history.actual_procedure_date,"προηγούμενη εφαρμογή"); followup=_parse_iso_date(history.last_followup_date,"τελευταία αξιολόγηση")
    if followup<procedure: raise HTTPException(status_code=422,detail="Η τελευταία αξιολόγηση δεν μπορεί να προηγείται της εφαρμογής")
    return {"actual_procedure_date":procedure.isoformat(),"vas_before":history.vas_before,"vas_after":history.vas_after,"last_followup_date":followup.isoformat(),"last_followup_vas":history.last_followup_vas}

async def _read_imaging_pdf(upload: UploadFile) -> bytes:
    filename=str(upload.filename or "").lower(); content_type=str(upload.content_type or "").lower()
    if not filename.endswith(".pdf") or content_type not in {"application/pdf","application/octet-stream"}: raise HTTPException(status_code=415,detail="Η απεικονιστική έκθεση πρέπει να είναι PDF")
    content=await upload.read(MAX_IMAGING_BYTES+1)
    if len(content)>MAX_IMAGING_BYTES: raise HTTPException(status_code=413,detail="Το PDF της απεικόνισης υπερβαίνει τα 20 MB")
    if not content.startswith(b"%PDF"): raise HTTPException(status_code=422,detail="Το αρχείο απεικόνισης δεν είναι έγκυρο PDF")
    return content

def build_rf_router(engine: Engine) -> APIRouter:
    initialize_rf_tables(engine)
    router=APIRouter(prefix="/clinical/clinic-utilities/rf",tags=["clinic-utilities-rf-v2"],dependencies=[Depends(_require_clinical_key)])
    page_path=_repo_root()/"static"/"clinic-utilities"/"rf"/"index.html"

    @router.get("",include_in_schema=False)
    def rf_page():
        if not page_path.is_file(): raise HTTPException(status_code=500,detail="RF utility page is unavailable")
        return FileResponse(page_path,headers={"Cache-Control":"no-store"})

    @router.get("/api/contract")
    def rf_contract():
        try: DoctorProfile.from_environment(); doctor_ok=True
        except ValueError: doctor_ok=False
        try: configured=set(product_catalog_from_environment())
        except ValueError: configured=set()
        return {"version":"rf-v2-category-a-2026-09","category":"A","pathways":{"A1":"Νέα θεραπεία","A2":"Συνέχιση θεραπείας"},"indications":{code:{"label":item["label"],"site_key":item["site_key"],"requires_intervention":bool(item.get("requires_intervention"))} for code,item in INDICATIONS.items()},"laterality":LATERALITY_LABELS,"rf_reasons":RF_REASON_LABELS,"products":{key:{"label":label,"configured":key in configured} for key,label in PRODUCT_LABELS.items()},"doctor_configured":doctor_ok}

    @router.post("/api/history")
    def rf_history(req: RFHistoryLookup):
        if req.site_key and req.site_key not in {x["site_key"] for x in INDICATIONS.values()}: raise HTTPException(status_code=422,detail="Μη έγκυρη περιοχή RF")
        if req.laterality and req.laterality not in LATERALITY_LABELS: raise HTTPException(status_code=422,detail="Μη έγκυρη πλευρά")
        rows=list_procedure_history(engine,req.identity_number,site_key=req.site_key,laterality=req.laterality); return {"found":bool(rows),"procedures":rows}

    @router.post("/api/parse-medications")
    def rf_parse_medications(req: RFTextRequest): return parse_medications(req.text)

    @router.post("/api/parse-physio")
    def rf_parse_physio(req: RFTextRequest): return parse_physio_dates(req.text)

    @router.post("/api/create")
    async def rf_create(draft_json: str=Form(...), imaging_report: UploadFile=File(...)):
        try: draft=RFApplicationDraft.model_validate_json(draft_json)
        except Exception as exc: raise HTTPException(status_code=422,detail="Μη έγκυρο RF draft") from exc
        indication=INDICATIONS.get(draft.indication_code)
        if not indication: raise HTTPException(status_code=422,detail="Μη έγκυρη ένδειξη RF")
        if draft.product_key not in PRODUCT_LABELS: raise HTTPException(status_code=422,detail="Μη έγκυρο προϊόν RF")
        if draft.laterality not in LATERALITY_LABELS: raise HTTPException(status_code=422,detail="Μη έγκυρη πλευρά")
        other_area,other_diagnosis=_resolve_other(draft,indication); imaging_bytes=await _read_imaging_pdf(imaging_report)
        payload={**draft.model_dump(exclude={"legacy_history"}),"site_key":indication["site_key"],"other_area":other_area,"other_diagnosis":other_diagnosis,"application_date":date.today().isoformat()}
        prior=None; legacy=None
        if draft.pathway=="A1": payload.update(_validate_a1(draft,indication))
        else:
            if draft.procedure_history_id and draft.legacy_history is not None: raise HTTPException(status_code=422,detail="Επιλέξτε υπάρχον ιστορικό ή legacy καταχώρηση, όχι και τα δύο")
            if draft.procedure_history_id:
                prior=get_procedure_history(engine,draft.procedure_history_id,draft.identity_number)
                if prior is None: raise HTTPException(status_code=404,detail="Η προηγούμενη εφαρμογή δεν βρέθηκε")
                if prior["site_key"]!=indication["site_key"]: raise HTTPException(status_code=422,detail="Η προηγούμενη εφαρμογή αφορά διαφορετική περιοχή")
                if draft.laterality not in {"none",prior.get("laterality") or "none"}: raise HTTPException(status_code=422,detail="Η προηγούμενη εφαρμογή αφορά διαφορετική πλευρά")
            elif draft.legacy_history is not None:
                prior=_validate_legacy(draft.legacy_history); legacy={**prior,"identity_number":draft.identity_number,"indication_code":draft.indication_code,"site_key":indication["site_key"],"laterality":draft.laterality,"exact_location":draft.exact_location}
            else: raise HTTPException(status_code=422,detail="Το A.2 απαιτεί προηγούμενη πραγματική εφαρμογή RF")
        try: doctor=DoctorProfile.from_environment(); product=product_catalog_from_environment()[draft.product_key]
        except ValueError as exc: raise HTTPException(status_code=503,detail="Η σταθερή ρύθμιση RF δεν είναι πλήρης") from exc
        try: pdf_bytes=build_official_rf_pdf(payload,doctor=doctor,product=product,radiology_pdf_bytes=imaging_bytes,prior_history=prior)
        except FileNotFoundError as exc: raise HTTPException(status_code=503,detail="Λείπει το επίσημο RF PDF template") from exc
        except (RuntimeError,ValueError) as exc: raise HTTPException(status_code=422,detail=str(exc)) from exc
        if legacy is not None: record_legacy_procedure(engine,legacy)
        application_id=record_application(engine,payload)
        return Response(content=pdf_bytes,media_type="application/pdf",headers={"Content-Disposition":f'inline; filename="RF_{application_id}.pdf"',"Cache-Control":"no-store","X-RF-Application-Id":application_id})

    return router
