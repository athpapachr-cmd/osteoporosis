from __future__ import annotations

import base64
from datetime import timedelta
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import quote
from uuid import uuid4

import fitz

from .models import ClinicianProfile, SickLeaveDocumentMetadataV1, SickLeaveDraftV1


PDF_SCHEMA = "sick_leave_certificate_v1"
PDF_VERSION = "1.0"
_METADATA_MARKER = "ClinicalDocuments.SickLeaveV1:"
MAX_PREVIOUS_PDF_BYTES = 8 * 1024 * 1024
MAX_SIGNATURE_BYTES = 2 * 1024 * 1024
UNICODE_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
UNICODE_BOLD_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def _font_path(*, bold: bool = False) -> Path:
    path = UNICODE_BOLD_FONT_PATH if bold else UNICODE_FONT_PATH
    if not path.is_file():
        raise RuntimeError("DejaVu Sans is required for Clinical Documents PDF generation")
    return path


def _format_date(value) -> str:
    return value.strftime("%d/%m/%Y")


def _safe_filename_component(value: str) -> str:
    value = unicodedata.normalize("NFC", str(value or "").strip())
    value = re.sub(r"[\\/:*?\"<>|\x00-\x1f]", "", value)
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._ ")
    return value[:100] or "Ασθενής"


def sick_leave_filename(draft: SickLeaveDraftV1) -> str:
    return f"Αναρρωτική_{_safe_filename_component(draft.patient_name)}_{draft.issued_on.strftime('%d-%m-%Y')}.pdf"


def content_disposition(filename: str, *, inline: bool = False) -> str:
    disposition = "inline" if inline else "attachment"
    encoded = quote(filename, safe="")
    return f"{disposition}; filename=\"sick-leave.pdf\"; filename*=UTF-8''{encoded}"


def _metadata_token(metadata: SickLeaveDocumentMetadataV1) -> str:
    payload = metadata.model_dump(mode="json", by_alias=True)
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    token = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return _METADATA_MARKER + token


def _decode_metadata_token(keywords: str) -> SickLeaveDocumentMetadataV1:
    text = str(keywords or "")
    index = text.find(_METADATA_MARKER)
    if index < 0:
        raise ValueError("Το PDF δεν περιέχει αναγνωρισμένα metadata αναρρωτικής άδειας V1")
    token = text[index + len(_METADATA_MARKER):].split()[0].strip(";, ")
    if not token:
        raise ValueError("Τα metadata της προηγούμενης άδειας είναι κενά")
    token += "=" * (-len(token) % 4)
    try:
        raw = base64.urlsafe_b64decode(token.encode("ascii"))
        data = json.loads(raw.decode("utf-8"))
        return SickLeaveDocumentMetadataV1.model_validate(data)
    except Exception as exc:
        raise ValueError("Τα metadata της προηγούμενης άδειας δεν είναι έγκυρα") from exc


def read_previous_sick_leave_pdf(content: bytes) -> SickLeaveDocumentMetadataV1:
    if not content or len(content) > MAX_PREVIOUS_PDF_BYTES:
        raise ValueError("Το προηγούμενο PDF είναι κενό ή υπερβαίνει το επιτρεπτό μέγεθος")
    if not content.startswith(b"%PDF"):
        raise ValueError("Το προηγούμενο αρχείο δεν είναι έγκυρο PDF")
    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ValueError("Το προηγούμενο PDF δεν μπορεί να αναγνωστεί") from exc
    try:
        if doc.page_count < 1:
            raise ValueError("Το προηγούμενο PDF δεν περιέχει σελίδες")
        return _decode_metadata_token((doc.metadata or {}).get("keywords", ""))
    finally:
        doc.close()


def reuse_options(metadata: SickLeaveDocumentMetadataV1) -> dict:
    next_start = metadata.leave_to + timedelta(days=1)
    identity = {
        "patient_name": metadata.patient_name,
        "id_type": metadata.id_type,
        "id_number": metadata.id_number,
    }
    return {
        "previous": metadata.model_dump(mode="json", by_alias=True),
        "extension": {
            **identity,
            "diagnosis": metadata.diagnosis,
            "leave_from": next_start.isoformat(),
            "leave_to": "",
            "derived_from_document_id": metadata.document_id,
            "relation": "extension",
        },
        "new_leave_same_patient": {
            **identity,
            "diagnosis": "",
            "leave_from": "",
            "leave_to": "",
            "derived_from_document_id": metadata.document_id,
            "relation": "new_leave_same_patient",
        },
    }


def validate_signature_image(content: bytes, content_type: str = "", filename: str = "") -> tuple[float, float]:
    if not content:
        raise ValueError("Το αρχείο υπογραφής είναι κενό")
    if len(content) > MAX_SIGNATURE_BYTES:
        raise ValueError("Η υπογραφή υπερβαίνει τα 2 MB")
    lower_name = str(filename or "").lower()
    lower_type = str(content_type or "").lower()
    is_png = content.startswith(b"\x89PNG\r\n\x1a\n")
    is_jpeg = content.startswith(b"\xff\xd8\xff")
    if not (is_png or is_jpeg):
        raise ValueError("Η υπογραφή πρέπει να είναι PNG ή JPEG")
    if lower_type and lower_type not in {"image/png", "image/jpeg", "image/jpg", "application/octet-stream"}:
        raise ValueError("Μη αποδεκτός τύπος αρχείου υπογραφής")
    if lower_name and not lower_name.endswith((".png", ".jpg", ".jpeg")):
        raise ValueError("Η υπογραφή πρέπει να έχει κατάληξη PNG/JPG/JPEG")
    try:
        pix = fitz.Pixmap(content)
        width, height = float(pix.width), float(pix.height)
        pix = None
    except Exception as exc:
        raise ValueError("Η εικόνα υπογραφής δεν μπορεί να αναγνωστεί") from exc
    if width <= 0 or height <= 0:
        raise ValueError("Η εικόνα υπογραφής δεν έχει έγκυρες διαστάσεις")
    if width * height > 20_000_000:
        raise ValueError("Η εικόνα υπογραφής έχει υπερβολικά μεγάλες διαστάσεις")
    return width, height


def _insert_textbox(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    *,
    size: float,
    bold: bool = False,
    align: int = fitz.TEXT_ALIGN_LEFT,
    min_size: float = 7.0,
    color=(0, 0, 0),
) -> None:
    value = str(text or "").strip()
    if not value:
        return
    font_path = _font_path(bold=bold)
    font_name = "CDDejaVuBold" if bold else "CDDejaVu"
    page.insert_font(fontname=font_name, fontfile=str(font_path))
    current = size
    while current >= min_size:
        result = page.insert_textbox(
            rect,
            value,
            fontname=font_name,
            fontfile=str(font_path),
            fontsize=current,
            lineheight=1.15,
            align=align,
            color=color,
            overlay=True,
        )
        if result >= 0:
            return
        current -= 0.5
    raise ValueError(f"Το κείμενο δεν χωρά στο PDF: {value[:80]}")


def _draw_label_value(page: fitz.Page, label: str, value: str, rect: fitz.Rect) -> None:
    label_rect = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + 15)
    value_rect = fitz.Rect(rect.x0, rect.y0 + 16, rect.x1, rect.y1)
    _insert_textbox(page, label_rect, label.upper(), size=7.5, bold=True, color=(0.20, 0.31, 0.45))
    _insert_textbox(page, value_rect, value, size=10.5, min_size=8.0)


def build_sick_leave_pdf(
    draft: SickLeaveDraftV1,
    *,
    clinician: ClinicianProfile,
    signature_bytes: bytes | None = None,
) -> tuple[bytes, SickLeaveDocumentMetadataV1]:
    if signature_bytes:
        validate_signature_image(signature_bytes)

    metadata = SickLeaveDocumentMetadataV1(
        document_id=str(uuid4()),
        patient_name=draft.patient_name,
        id_type=draft.id_type,
        id_number=draft.id_number,
        diagnosis=draft.diagnosis,
        leave_from=draft.leave_from,
        leave_to=draft.leave_to,
        issued_on=draft.issued_on,
        derived_from_document_id=draft.derived_from_document_id,
        relation=draft.relation,
    )

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 points
    navy = (0.08, 0.20, 0.34)
    soft = (0.95, 0.97, 0.99)
    border = (0.78, 0.84, 0.90)

    # Clinician header.
    _insert_textbox(page, fitz.Rect(52, 48, 355, 72), clinician.name, size=15, bold=True, color=navy)
    _insert_textbox(page, fitz.Rect(52, 72, 355, 94), clinician.specialty, size=10.5, color=(0.25, 0.31, 0.38))
    contact = "\n".join(part for part in [clinician.phone, clinician.email, clinician.clinic, clinician.address] if part)
    _insert_textbox(page, fitz.Rect(370, 48, 543, 106), contact, size=8.3, align=fitz.TEXT_ALIGN_RIGHT, color=(0.28, 0.33, 0.39))
    page.draw_line(fitz.Point(52, 115), fitz.Point(543, 115), color=navy, width=1.1)

    _insert_textbox(
        page,
        fitz.Rect(52, 142, 543, 180),
        "ΒΕΒΑΙΩΣΗ ΑΣΘΕΝΕΙΑΣ",
        size=17,
        bold=True,
        align=fitz.TEXT_ALIGN_CENTER,
        color=navy,
    )

    # Patient identity card.
    card = fitz.Rect(52, 205, 543, 286)
    page.draw_rect(card, color=border, fill=soft, width=0.8)
    _draw_label_value(page, "Όνομα ασθενούς", draft.patient_name, fitz.Rect(70, 221, 350, 273))
    _draw_label_value(page, draft.id_type, draft.id_number, fitz.Rect(375, 221, 525, 273))

    # Diagnosis.
    _insert_textbox(page, fitz.Rect(52, 316, 543, 333), "ΔΙΑΓΝΩΣΗ", size=8, bold=True, color=(0.20, 0.31, 0.45))
    diagnosis_rect = fitz.Rect(52, 339, 543, 437)
    page.draw_rect(diagnosis_rect, color=border, fill=(1, 1, 1), width=0.8)
    _insert_textbox(page, fitz.Rect(68, 354, 527, 422), draft.diagnosis, size=11, min_size=8.0)

    # Leave range card.
    leave_rect = fitz.Rect(52, 472, 543, 563)
    page.draw_rect(leave_rect, color=(0.60, 0.74, 0.88), fill=(0.93, 0.97, 1.0), width=1.0)
    _insert_textbox(page, fitz.Rect(70, 490, 525, 510), "ΑΝΑΡΡΩΤΙΚΗ ΑΔΕΙΑ", size=8.3, bold=True, align=fitz.TEXT_ALIGN_CENTER, color=navy)
    leave_text = f"Από {_format_date(draft.leave_from)} έως και {_format_date(draft.leave_to)}"
    _insert_textbox(page, fitz.Rect(70, 517, 525, 548), leave_text, size=13.5, bold=True, align=fitz.TEXT_ALIGN_CENTER, color=navy)

    # Issue date + signature block.
    _insert_textbox(page, fitz.Rect(52, 628, 275, 650), f"Ημερομηνία έκδοσης: {_format_date(draft.issued_on)}", size=9.5)
    _insert_textbox(page, fitz.Rect(350, 605, 535, 625), "Με εκτίμηση,", size=9, align=fitz.TEXT_ALIGN_CENTER)
    if signature_bytes:
        image_rect = fitz.Rect(385, 628, 500, 675)
        page.insert_image(image_rect, stream=signature_bytes, keep_proportion=True, overlay=True)
    _insert_textbox(page, fitz.Rect(350, 678, 535, 699), clinician.name, size=9.3, bold=True, align=fitz.TEXT_ALIGN_CENTER, color=navy)
    _insert_textbox(page, fitz.Rect(350, 700, 535, 720), clinician.specialty, size=8.4, align=fitz.TEXT_ALIGN_CENTER, color=(0.28, 0.33, 0.39))

    doc.set_metadata(
        {
            "title": "Βεβαίωση Ασθένειας",
            "author": clinician.name,
            "subject": "Sick Leave Certificate V1",
            "keywords": _metadata_token(metadata),
            "creator": "Clinical Documents Engine",
            "producer": "Clinical Documents Engine / PyMuPDF",
        }
    )
    output = doc.tobytes(garbage=4, deflate=True, clean=True)
    doc.close()
    return output, metadata
