from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import fitz
from pypdf import PdfReader, PdfWriter

from .catalog import DoctorProfile, ProductProfile

ROOT = Path(__file__).resolve().parent
OFFICIAL_TEMPLATE_PATH = ROOT / "templates" / "rf_official_form_v2.pdf"
UNICODE_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

_PAGE1_TEXT = {
    "doctor_name": fitz.Rect(286,208,521,225), "doctor_gesy": fitz.Rect(286,227,521,244),
    "doctor_specialty": fitz.Rect(286,246,521,263), "doctor_center": fitz.Rect(286,265,521,282),
    "doctor_phone": fitz.Rect(286,284,521,301), "doctor_email": fitz.Rect(286,303,521,320),
    "application_date": fitz.Rect(286,323,521,340), "patient_name": fitz.Rect(203,390,368,407),
    "patient_age": fitz.Rect(448,390,522,407), "identity_number": fitz.Rect(203,409,326,426),
    "gesy_number": fitz.Rect(448,409,522,426), "product_seq": fitz.Rect(73,515,102,533),
    "product_code": fitz.Rect(103,515,180,533), "product_description": fitz.Rect(181,515,447,533),
    "product_quantity": fitz.Rect(448,515,522,533),
}
_PAGE1_CATEGORY_A_CHECK = fitz.Rect(490,612,520,643)
_INDICATION_CHECKS = {
    "KNEE_OA_KL34":fitz.Rect(501,159,522,174), "SI_DEGENERATIVE":fitz.Rect(501,201,522,216),
    "HIP_OA_KL34":fitz.Rect(499,242,522,257), "MORTON_NEUROMA":fitz.Rect(499,269,522,284),
    "SHOULDER_OA_KL34":fitz.Rect(499,379,522,394), "SHOULDER_IRREPARABLE_CUFF":fitz.Rect(499,394,522,421),
    "OTHER_LATERAL_EPICONDYLITIS":fitz.Rect(499,433,522,448), "OTHER_DEQUERVAIN":fitz.Rect(499,433,522,448), "OTHER_CUSTOM":fitz.Rect(499,433,522,448),
}
_OTHER_AREA=fitz.Rect(188,447,418,462); _OTHER_DIAGNOSIS=fitz.Rect(197,461,423,476); _A1_IMAGING_CHECK=fitz.Rect(499,665,523,693)
_A1_REASON=fitz.Rect(145,141,478,183); _A1_LOCATION=fitz.Rect(145,229,484,273)
_A1_VAS={"pain_onset_date":fitz.Rect(300,322,383,340),"pain_onset_vas":fitz.Rect(300,341,383,358),"last_assessment_date":fitz.Rect(300,359,383,377),"last_assessment_vas":fitz.Rect(300,378,383,395)}
_MED_ROWS={
 "nsaid":[(fitz.Rect(228,459,351,477),fitz.Rect(352,459,437,477),fitz.Rect(438,459,523,477)),(fitz.Rect(228,477,351,495),fitz.Rect(352,477,437,495),fitz.Rect(438,477,523,495)),(fitz.Rect(228,495,351,513),fitz.Rect(352,495,437,513),fitz.Rect(438,495,523,513))],
 "other":[(fitz.Rect(228,514,351,532),fitz.Rect(352,514,437,532),fitz.Rect(438,514,523,532)),(fitz.Rect(228,532,351,550),fitz.Rect(352,532,437,550),fitz.Rect(438,532,523,550)),(fitz.Rect(228,550,351,568),fitz.Rect(352,550,437,568),fitz.Rect(438,550,523,568))],
}
_ADVERSE_ROWS=[(fitz.Rect(94,689,192,707),fitz.Rect(193,689,523,707)),(fitz.Rect(94,707,192,725),fitz.Rect(193,707,523,725)),(fitz.Rect(94,725,192,744),fitz.Rect(193,725,523,744))]
_SI_ROW={"site":fitz.Rect(286,215,340,244),"date":fitz.Rect(341,215,408,244),"vas_before":fitz.Rect(409,215,466,244),"vas_after":fitz.Rect(466,215,523,244)}
_HIP_ROW={"site":fitz.Rect(286,244,340,263),"date":fitz.Rect(341,244,408,263),"vas_before":fitz.Rect(409,244,466,263),"vas_after":fitz.Rect(466,244,523,263)}
_PHYSIO_ROW={"start":fitz.Rect(94,359,237,377),"end":fitz.Rect(237,359,380,377),"count":fitz.Rect(380,359,523,377)}; _A1_NOTES=fitz.Rect(98,468,522,598)
_A2_HISTORY={"actual_procedure_date":fitz.Rect(322,525,428,541),"vas_before":fitz.Rect(419,549,493,565),"vas_after":fitz.Rect(419,573,505,589),"last_followup_date":fitz.Rect(308,597,413,613),"last_followup_vas":fitz.Rect(410,621,498,637)}
_A2_IMAGING_CHECK=fitz.Rect(499,645,523,675); _A2_NOTES=fitz.Rect(98,105,522,151)

def _font_path() -> Path:
    if not UNICODE_FONT_PATH.is_file(): raise RuntimeError("DejaVu Sans is required for RF PDF generation")
    return UNICODE_FONT_PATH

def _write(page: fitz.Page, rect: fitz.Rect, value: Any, *, size: float=8.0, align: int=0) -> None:
    text=str(value or "").strip()
    if not text: return
    font_path=_font_path(); name="RFDejaVuSans"; page.insert_font(fontname=name,fontfile=str(font_path)); current=size
    while current>=5.5:
        spare=page.insert_textbox(rect,text,fontname=name,fontfile=str(font_path),fontsize=current,lineheight=1.05,align=align,color=(0,0,0),overlay=True)
        if spare>=0: return
        page.clean_contents(); current-=0.5
    raise ValueError(f"RF PDF text does not fit: {text[:80]}")

def _check(page: fitz.Page, rect: fitz.Rect) -> None: _write(page,rect,"✓",size=12,align=1)

def _write_common(page: fitz.Page, data: dict[str,Any], doctor: DoctorProfile, product: ProductProfile) -> None:
    values={"doctor_name":doctor.name,"doctor_gesy":doctor.gesy_code,"doctor_specialty":doctor.specialty,"doctor_center":doctor.medical_center,"doctor_phone":doctor.phone,"doctor_email":doctor.email,"application_date":data["application_date"],"patient_name":data["patient_name"],"patient_age":data["age"],"identity_number":data["identity_number"],"gesy_number":data["gesy_number"],"product_seq":"1","product_code":product.code,"product_description":product.description,"product_quantity":product.quantity}
    for field,value in values.items(): _write(page,_PAGE1_TEXT[field],value,size=7.5,align=1 if field in {"product_seq","product_code","product_quantity"} else 0)
    _check(page,_PAGE1_CATEGORY_A_CHECK)

def _write_indication(page: fitz.Page, data: dict[str,Any]) -> None:
    code=data["indication_code"]; _check(page,_INDICATION_CHECKS[code])
    if code.startswith("OTHER_"):
        _write(page,_OTHER_AREA,data.get("other_area"),size=7); _write(page,_OTHER_DIAGNOSIS,data.get("other_diagnosis"),size=7)

def _meds(data,key): return [x for x in (data.get(key) or []) if isinstance(x,dict)][:3]
def _write_med_rows(page,category,rows):
    for boxes,med in zip(_MED_ROWS[category],rows):
        drug,dose,duration=boxes; _write(page,drug,med.get("drug_name"),size=6.5); _write(page,dose,med.get("dose"),size=6.5,align=1); _write(page,duration,med.get("duration"),size=6.2,align=1)

def _write_a1(document: fitz.Document, data: dict[str,Any]) -> None:
    page2,page3,page4=document[1],document[2],document[3]; _write_indication(page2,data); _check(page2,_A1_IMAGING_CHECK)
    _write(page3,_A1_REASON,data.get("rf_reason_text"),size=7.5); _write(page3,_A1_LOCATION,data.get("exact_location"),size=8)
    for key,rect in _A1_VAS.items(): _write(page3,rect,data.get(key),size=7,align=1)
    _write_med_rows(page3,"nsaid",_meds(data,"nsaid_trials")); _write_med_rows(page3,"other",_meds(data,"other_analgesic_trials"))
    for boxes,item in zip(_ADVERSE_ROWS,data.get("adverse_effects") or []):
        if isinstance(item,dict): _write(page3,boxes[0],item.get("treatment"),size=6.2); _write(page3,boxes[1],item.get("effect"),size=6.2)
    intervention=data.get("intervention") or {}; target=_SI_ROW if data.get("site_key")=="si" else (_HIP_ROW if data.get("site_key")=="hip" else None)
    if target:
        for key,rect in target.items(): _write(page4,rect,intervention.get(key),size=6.2,align=1)
    physio=data.get("physio") or {}; _write(page4,_PHYSIO_ROW["start"],physio.get("start_date"),size=6.5,align=1); _write(page4,_PHYSIO_ROW["end"],physio.get("end_date"),size=6.5,align=1); _write(page4,_PHYSIO_ROW["count"],physio.get("treatment_count"),size=7,align=1); _write(page4,_A1_NOTES,data.get("additional_notes"),size=7)

def _write_a2(document: fitz.Document, data: dict[str,Any], history: dict[str,Any]) -> None:
    page5,page6=document[4],document[5]; _write_indication(page5,data)
    for key,rect in _A2_HISTORY.items(): _write(page5,rect,history.get(key),size=7,align=1)
    _check(page5,_A2_IMAGING_CHECK); _write(page6,_A2_NOTES,data.get("additional_notes"),size=7)

def build_official_rf_pdf(data: dict[str,Any], *, doctor: DoctorProfile, product: ProductProfile, radiology_pdf_bytes: bytes, template_path: Path=OFFICIAL_TEMPLATE_PATH, prior_history: dict[str,Any]|None=None) -> bytes:
    if not template_path.is_file(): raise FileNotFoundError(f"Missing official RF template: {template_path}")
    if not radiology_pdf_bytes.startswith(b"%PDF"): raise ValueError("Imaging report is not a PDF")
    try:
        radiology_reader=PdfReader(io.BytesIO(radiology_pdf_bytes))
        if not radiology_reader.pages: raise ValueError("Imaging report PDF is empty")
    except Exception as exc: raise ValueError("Imaging report PDF is unreadable") from exc
    source=fitz.open(template_path)
    try:
        if source.page_count<6: raise RuntimeError("Official RF template does not contain A.1/A.2 pages")
        _write_common(source[0],data,doctor,product)
        if data["pathway"]=="A1": _write_a1(source,data); selected=[0,1,2,3]
        elif data["pathway"]=="A2":
            if not prior_history: raise ValueError("A.2 requires prior procedure history")
            _write_a2(source,data,prior_history); selected=[0,4,5]
        else: raise ValueError("Unsupported RF pathway")
        stamped=fitz.open()
        for idx in selected: stamped.insert_pdf(source,from_page=idx,to_page=idx)
        stamped_bytes=stamped.tobytes(garbage=4,deflate=True,clean=True); stamped.close()
    finally: source.close()
    writer=PdfWriter()
    for page in PdfReader(io.BytesIO(stamped_bytes)).pages: writer.add_page(page)
    for page in radiology_reader.pages: writer.add_page(page)
    output=io.BytesIO(); writer.write(output); return output.getvalue()
