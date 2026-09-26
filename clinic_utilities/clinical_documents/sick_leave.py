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


SICK_LEAVE_TEMPLATES = (
    {"id": "classic", "label": "Classic Medical", "description": "Ισορροπημένη ιατρική εμφάνιση"},
    {"id": "modern", "label": "Modern Clinic", "description": "Σύγχρονη και ευρύχωρη"},
    {"id": "minimal", "label": "Minimal", "description": "Λιτή, σαν επαγγελματικό letterhead"},
    {"id": "compact", "label": "Compact", "description": "Πυκνή και πρακτική"},
    {"id": "formal", "label": "Formal", "description": "Πιο θεσμική / executive αισθητική"},
)

SICK_LEAVE_COLOR_THEMES = (
    {"id": "navy", "label": "Navy"},
    {"id": "teal", "label": "Teal"},
    {"id": "graphite", "label": "Graphite"},
    {"id": "burgundy", "label": "Burgundy"},
    {"id": "forest", "label": "Forest"},
    {"id": "monochrome", "label": "Monochrome"},
)

_PALETTES = {
    "navy": {
        "accent": (0.08, 0.20, 0.34),
        "accent2": (0.18, 0.43, 0.66),
        "soft": (0.94, 0.97, 1.00),
        "border": (0.76, 0.83, 0.90),
        "muted": (0.28, 0.33, 0.39),
    },
    "teal": {
        "accent": (0.05, 0.32, 0.34),
        "accent2": (0.08, 0.52, 0.53),
        "soft": (0.93, 0.98, 0.98),
        "border": (0.72, 0.86, 0.85),
        "muted": (0.25, 0.36, 0.37),
    },
    "graphite": {
        "accent": (0.16, 0.18, 0.22),
        "accent2": (0.36, 0.40, 0.45),
        "soft": (0.96, 0.96, 0.97),
        "border": (0.78, 0.80, 0.83),
        "muted": (0.33, 0.35, 0.39),
    },
    "burgundy": {
        "accent": (0.38, 0.08, 0.15),
        "accent2": (0.58, 0.16, 0.25),
        "soft": (0.99, 0.95, 0.96),
        "border": (0.89, 0.77, 0.80),
        "muted": (0.39, 0.29, 0.31),
    },
    "forest": {
        "accent": (0.10, 0.30, 0.20),
        "accent2": (0.20, 0.48, 0.31),
        "soft": (0.94, 0.98, 0.95),
        "border": (0.76, 0.86, 0.79),
        "muted": (0.28, 0.36, 0.31),
    },
    "monochrome": {
        "accent": (0.10, 0.10, 0.10),
        "accent2": (0.32, 0.32, 0.32),
        "soft": (0.96, 0.96, 0.96),
        "border": (0.78, 0.78, 0.78),
        "muted": (0.34, 0.34, 0.34),
    },
}

_TEMPLATE_IDS = {item["id"] for item in SICK_LEAVE_TEMPLATES}
_THEME_IDS = {item["id"] for item in SICK_LEAVE_COLOR_THEMES}


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


def sick_leave_appearance_contract() -> dict:
    return {
        "default_template": "classic",
        "default_color_theme": "navy",
        "templates": [dict(item) for item in SICK_LEAVE_TEMPLATES],
        "color_themes": [dict(item) for item in SICK_LEAVE_COLOR_THEMES],
    }


def validate_sick_leave_appearance(template_id: str = "classic", color_theme: str = "navy") -> tuple[str, str]:
    template = str(template_id or "classic").strip().casefold()
    theme = str(color_theme or "navy").strip().casefold()
    if template not in _TEMPLATE_IDS:
        raise ValueError("Μη υποστηριζόμενο template αναρρωτικής άδειας")
    if theme not in _THEME_IDS:
        raise ValueError("Μη υποστηριζόμενο χρωματικό θέμα αναρρωτικής άδειας")
    return template, theme


def _draw_clinician_header(
    page: fitz.Page,
    clinician: ClinicianProfile,
    palette: dict,
    *,
    template_id: str,
) -> None:
    accent = palette["accent"]
    muted = palette["muted"]
    contact = "\n".join(part for part in [clinician.phone, clinician.email, clinician.clinic, clinician.address] if part)

    if template_id == "formal":
        page.draw_rect(fitz.Rect(0, 0, 595, 122), color=accent, fill=accent, width=0)
        _insert_textbox(page, fitz.Rect(52, 36, 355, 62), clinician.name, size=15.5, bold=True, color=(1, 1, 1))
        _insert_textbox(page, fitz.Rect(52, 64, 355, 86), clinician.specialty, size=10.3, color=(0.94, 0.94, 0.94))
        _insert_textbox(page, fitz.Rect(370, 34, 543, 101), contact, size=8.2, align=fitz.TEXT_ALIGN_RIGHT, color=(0.96, 0.96, 0.96))
        return

    if template_id == "modern":
        page.draw_rect(fitz.Rect(52, 45, 58, 102), color=accent, fill=accent, width=0)
        _insert_textbox(page, fitz.Rect(72, 45, 360, 70), clinician.name, size=16, bold=True, color=accent)
        _insert_textbox(page, fitz.Rect(72, 72, 360, 94), clinician.specialty, size=10.2, color=muted)
        _insert_textbox(page, fitz.Rect(375, 44, 543, 101), contact, size=8.2, align=fitz.TEXT_ALIGN_RIGHT, color=muted)
        return

    if template_id == "minimal":
        _insert_textbox(page, fitz.Rect(52, 48, 360, 72), clinician.name, size=14.5, bold=True, color=accent)
        _insert_textbox(page, fitz.Rect(52, 74, 360, 94), clinician.specialty, size=9.8, color=muted)
        _insert_textbox(page, fitz.Rect(365, 46, 543, 98), contact, size=8.0, align=fitz.TEXT_ALIGN_RIGHT, color=muted)
        page.draw_line(fitz.Point(52, 111), fitz.Point(543, 111), color=palette["border"], width=0.7)
        return

    if template_id == "compact":
        _insert_textbox(page, fitz.Rect(52, 36, 360, 58), clinician.name, size=13.5, bold=True, color=accent)
        _insert_textbox(page, fitz.Rect(52, 58, 360, 77), clinician.specialty, size=9.2, color=muted)
        _insert_textbox(page, fitz.Rect(365, 35, 543, 83), contact, size=7.6, align=fitz.TEXT_ALIGN_RIGHT, color=muted)
        page.draw_line(fitz.Point(52, 93), fitz.Point(543, 93), color=accent, width=0.9)
        return

    # classic
    _insert_textbox(page, fitz.Rect(52, 48, 355, 72), clinician.name, size=15, bold=True, color=accent)
    _insert_textbox(page, fitz.Rect(52, 72, 355, 94), clinician.specialty, size=10.5, color=muted)
    _insert_textbox(page, fitz.Rect(370, 48, 543, 106), contact, size=8.3, align=fitz.TEXT_ALIGN_RIGHT, color=muted)
    page.draw_line(fitz.Point(52, 115), fitz.Point(543, 115), color=accent, width=1.1)


def _draw_patient_identity(
    page: fitz.Page,
    draft: SickLeaveDraftV1,
    palette: dict,
    rect: fitz.Rect,
    *,
    template_id: str,
) -> None:
    accent = palette["accent"]
    border = palette["border"]
    soft = palette["soft"]

    if template_id == "minimal":
        _draw_label_value(page, "Όνομα ασθενούς", draft.patient_name, fitz.Rect(rect.x0, rect.y0, 355, rect.y1))
        _draw_label_value(page, draft.id_type, draft.id_number, fitz.Rect(380, rect.y0, rect.x1, rect.y1))
        page.draw_line(fitz.Point(rect.x0, rect.y1), fitz.Point(355, rect.y1), color=border, width=0.7)
        page.draw_line(fitz.Point(380, rect.y1), fitz.Point(rect.x1, rect.y1), color=border, width=0.7)
        return

    fill = soft
    width = 1.0 if template_id in {"modern", "formal"} else 0.8
    page.draw_rect(rect, color=accent if template_id == "formal" else border, fill=fill, width=width)
    xpad = 18 if template_id != "compact" else 13
    _draw_label_value(page, "Όνομα ασθενούς", draft.patient_name, fitz.Rect(rect.x0 + xpad, rect.y0 + 14, 350, rect.y1 - 10))
    _draw_label_value(page, draft.id_type, draft.id_number, fitz.Rect(375, rect.y0 + 14, rect.x1 - xpad, rect.y1 - 10))


def _draw_diagnosis(
    page: fitz.Page,
    draft: SickLeaveDraftV1,
    palette: dict,
    label_y: float,
    rect: fitz.Rect,
    *,
    template_id: str,
) -> None:
    accent = palette["accent"]
    border = palette["border"]
    _insert_textbox(page, fitz.Rect(rect.x0, label_y, rect.x1, label_y + 17), "ΔΙΑΓΝΩΣΗ", size=8, bold=True, color=accent)

    if template_id == "minimal":
        _insert_textbox(page, fitz.Rect(rect.x0, rect.y0 + 4, rect.x1, rect.y1 - 8), draft.diagnosis, size=11, min_size=8.0)
        page.draw_line(fitz.Point(rect.x0, rect.y1), fitz.Point(rect.x1, rect.y1), color=border, width=0.7)
        return

    fill = palette["soft"] if template_id == "modern" else (1, 1, 1)
    page.draw_rect(rect, color=border, fill=fill, width=0.8)
    _insert_textbox(page, fitz.Rect(rect.x0 + 16, rect.y0 + 14, rect.x1 - 16, rect.y1 - 12), draft.diagnosis, size=11, min_size=8.0)


def _draw_leave_range(
    page: fitz.Page,
    draft: SickLeaveDraftV1,
    palette: dict,
    rect: fitz.Rect,
    *,
    template_id: str,
) -> None:
    accent = palette["accent"]
    accent2 = palette["accent2"]
    leave_text = f"Από {_format_date(draft.leave_from)} έως και {_format_date(draft.leave_to)}"

    if template_id == "minimal":
        page.draw_line(fitz.Point(rect.x0, rect.y0), fitz.Point(rect.x0, rect.y1), color=accent, width=3.0)
        _insert_textbox(page, fitz.Rect(rect.x0 + 18, rect.y0 + 8, rect.x1, rect.y0 + 28), "ΑΝΑΡΡΩΤΙΚΗ ΑΔΕΙΑ", size=8.2, bold=True, color=accent)
        _insert_textbox(page, fitz.Rect(rect.x0 + 18, rect.y0 + 35, rect.x1, rect.y1), leave_text, size=13.5, bold=True, color=accent)
        return

    if template_id == "formal":
        page.draw_rect(rect, color=accent, fill=accent, width=0)
        _insert_textbox(page, fitz.Rect(rect.x0 + 18, rect.y0 + 16, rect.x1 - 18, rect.y0 + 36), "ΑΝΑΡΡΩΤΙΚΗ ΑΔΕΙΑ", size=8.3, bold=True, align=fitz.TEXT_ALIGN_CENTER, color=(1, 1, 1))
        _insert_textbox(page, fitz.Rect(rect.x0 + 18, rect.y0 + 43, rect.x1 - 18, rect.y1 - 10), leave_text, size=13.5, bold=True, align=fitz.TEXT_ALIGN_CENTER, color=(1, 1, 1))
        return

    page.draw_rect(rect, color=accent2, fill=palette["soft"], width=1.0)
    _insert_textbox(page, fitz.Rect(rect.x0 + 18, rect.y0 + 16, rect.x1 - 18, rect.y0 + 36), "ΑΝΑΡΡΩΤΙΚΗ ΑΔΕΙΑ", size=8.3, bold=True, align=fitz.TEXT_ALIGN_CENTER, color=accent)
    _insert_textbox(page, fitz.Rect(rect.x0 + 18, rect.y0 + 43, rect.x1 - 18, rect.y1 - 10), leave_text, size=13.5, bold=True, align=fitz.TEXT_ALIGN_CENTER, color=accent)


def _draw_issue_and_signature(
    page: fitz.Page,
    draft: SickLeaveDraftV1,
    clinician: ClinicianProfile,
    signature_bytes: bytes | None,
    palette: dict,
    *,
    issue_y: float,
    signature_y: float,
) -> None:
    accent = palette["accent"]
    muted = palette["muted"]
    _insert_textbox(page, fitz.Rect(52, issue_y, 285, issue_y + 24), f"Ημερομηνία έκδοσης: {_format_date(draft.issued_on)}", size=9.5)
    _insert_textbox(page, fitz.Rect(350, signature_y, 535, signature_y + 20), "Με εκτίμηση,", size=9, align=fitz.TEXT_ALIGN_CENTER)
    if signature_bytes:
        page.insert_image(
            fitz.Rect(385, signature_y + 23, 500, signature_y + 70),
            stream=signature_bytes,
            keep_proportion=True,
            overlay=True,
        )
    _insert_textbox(page, fitz.Rect(350, signature_y + 73, 535, signature_y + 94), clinician.name, size=9.3, bold=True, align=fitz.TEXT_ALIGN_CENTER, color=accent)
    _insert_textbox(page, fitz.Rect(350, signature_y + 95, 535, signature_y + 115), clinician.specialty, size=8.4, align=fitz.TEXT_ALIGN_CENTER, color=muted)


def build_sick_leave_pdf(
    draft: SickLeaveDraftV1,
    *,
    clinician: ClinicianProfile,
    signature_bytes: bytes | None = None,
    template_id: str = "classic",
    color_theme: str = "navy",
) -> tuple[bytes, SickLeaveDocumentMetadataV1]:
    if signature_bytes:
        validate_signature_image(signature_bytes)

    template_id, color_theme = validate_sick_leave_appearance(template_id, color_theme)
    palette = _PALETTES[color_theme]

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
    _draw_clinician_header(page, clinician, palette, template_id=template_id)

    if template_id == "compact":
        _insert_textbox(page, fitz.Rect(52, 110, 543, 142), "ΒΕΒΑΙΩΣΗ ΑΣΘΕΝΕΙΑΣ", size=15.5, bold=True, align=fitz.TEXT_ALIGN_CENTER, color=palette["accent"])
        _draw_patient_identity(page, draft, palette, fitz.Rect(52, 160, 543, 226), template_id=template_id)
        _draw_diagnosis(page, draft, palette, 250, fitz.Rect(52, 272, 543, 352), template_id=template_id)
        _draw_leave_range(page, draft, palette, fitz.Rect(52, 382, 543, 458), template_id=template_id)
        _draw_issue_and_signature(page, draft, clinician, signature_bytes, palette, issue_y=545, signature_y=522)

    elif template_id == "modern":
        _insert_textbox(page, fitz.Rect(72, 132, 543, 169), "ΒΕΒΑΙΩΣΗ ΑΣΘΕΝΕΙΑΣ", size=18, bold=True, color=palette["accent"])
        _insert_textbox(page, fitz.Rect(72, 169, 543, 190), "Ιατρική βεβαίωση αναρρωτικής άδειας", size=8.8, color=palette["muted"])
        _draw_patient_identity(page, draft, palette, fitz.Rect(52, 212, 543, 294), template_id=template_id)
        _draw_diagnosis(page, draft, palette, 322, fitz.Rect(52, 345, 543, 438), template_id=template_id)
        _draw_leave_range(page, draft, palette, fitz.Rect(52, 476, 543, 566), template_id=template_id)
        _draw_issue_and_signature(page, draft, clinician, signature_bytes, palette, issue_y=634, signature_y=610)

    elif template_id == "minimal":
        _insert_textbox(page, fitz.Rect(52, 142, 543, 178), "ΒΕΒΑΙΩΣΗ ΑΣΘΕΝΕΙΑΣ", size=16.5, bold=True, color=palette["accent"])
        _draw_patient_identity(page, draft, palette, fitz.Rect(52, 212, 543, 265), template_id=template_id)
        _draw_diagnosis(page, draft, palette, 305, fitz.Rect(52, 326, 543, 405), template_id=template_id)
        _draw_leave_range(page, draft, palette, fitz.Rect(52, 462, 543, 542), template_id=template_id)
        _draw_issue_and_signature(page, draft, clinician, signature_bytes, palette, issue_y=628, signature_y=604)

    elif template_id == "formal":
        _insert_textbox(page, fitz.Rect(52, 145, 543, 181), "ΒΕΒΑΙΩΣΗ ΑΣΘΕΝΕΙΑΣ", size=17, bold=True, align=fitz.TEXT_ALIGN_CENTER, color=palette["accent"])
        page.draw_line(fitz.Point(212, 188), fitz.Point(383, 188), color=palette["accent2"], width=1.2)
        _draw_patient_identity(page, draft, palette, fitz.Rect(52, 215, 543, 296), template_id=template_id)
        _draw_diagnosis(page, draft, palette, 326, fitz.Rect(52, 349, 543, 441), template_id=template_id)
        _draw_leave_range(page, draft, palette, fitz.Rect(52, 478, 543, 568), template_id=template_id)
        _draw_issue_and_signature(page, draft, clinician, signature_bytes, palette, issue_y=635, signature_y=610)

    else:  # classic
        _insert_textbox(page, fitz.Rect(52, 142, 543, 180), "ΒΕΒΑΙΩΣΗ ΑΣΘΕΝΕΙΑΣ", size=17, bold=True, align=fitz.TEXT_ALIGN_CENTER, color=palette["accent"])
        _draw_patient_identity(page, draft, palette, fitz.Rect(52, 205, 543, 286), template_id=template_id)
        _draw_diagnosis(page, draft, palette, 316, fitz.Rect(52, 339, 543, 437), template_id=template_id)
        _draw_leave_range(page, draft, palette, fitz.Rect(52, 472, 543, 563), template_id=template_id)
        _draw_issue_and_signature(page, draft, clinician, signature_bytes, palette, issue_y=628, signature_y=605)

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
