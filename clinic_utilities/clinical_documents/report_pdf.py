from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import fitz

from .models import ClinicianProfile
from .report_models import FinalMedicalReportV1
from .sick_leave import validate_signature_image

FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
A4_WIDTH = 595
A4_HEIGHT = 842
LEFT = 52
RIGHT = 543
TOP = 68
BOTTOM = 770
NAVY = (0.08, 0.20, 0.34)
GREY = (0.32, 0.36, 0.42)


def _font_path(bold: bool = False) -> Path:
    path = FONT_BOLD if bold else FONT_REGULAR
    if not path.is_file():
        raise RuntimeError("DejaVu Sans is required for medical-report PDF generation")
    return path


def _safe_component(value: str) -> str:
    text = unicodedata.normalize("NFC", str(value or "").strip())
    text = re.sub(r"[\\/:*?\"<>|\x00-\x1f]", "", text)
    text = re.sub(r"\s+", "_", text).strip("._ ")
    return text[:100] or "Ασθενής"


def medical_report_filename(report: FinalMedicalReportV1) -> str:
    prefix = "Ιατρονομική_Έκθεση" if report.case.report_type == "medico_legal_expert_report" else "Ιατρική_Έκθεση"
    return f"{prefix}_{_safe_component(report.case.patient_name)}_{report.case.report_date.strftime('%d-%m-%Y')}.pdf"


def _word_chunks(word: str, width: int) -> list[str]:
    if len(word) <= width:
        return [word]
    return [word[index:index + width] for index in range(0, len(word), width)]


def _wrap(text: str, *, width: int = 94) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text or "").replace("\r", "").split("\n"):
        raw_words = paragraph.split()
        words = [chunk for word in raw_words for chunk in _word_chunks(word, width)]
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = current + " " + word
            if len(candidate) <= width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


class _Writer:
    def __init__(self, clinician: ClinicianProfile):
        self.doc = fitz.open()
        self.clinician = clinician
        self.page: fitz.Page | None = None
        self.y = TOP
        self.page_number = 0
        self.new_page(first=True)

    def new_page(self, *, first: bool = False):
        self.page = self.doc.new_page(width=A4_WIDTH, height=A4_HEIGHT)
        self.page_number += 1
        self.page.insert_font(fontname="MRRegular", fontfile=str(_font_path(False)))
        self.page.insert_font(fontname="MRBold", fontfile=str(_font_path(True)))
        self.y = TOP
        if first:
            self.text(self.clinician.name, size=14.5, bold=True, color=NAVY)
            self.text(self.clinician.specialty, size=9.5, color=GREY)
            contact = " · ".join(item for item in [self.clinician.phone, self.clinician.email] if item)
            if contact:
                self.text(contact, size=8.2, color=GREY)
            self.page.draw_line(fitz.Point(LEFT, self.y + 4), fitz.Point(RIGHT, self.y + 4), color=NAVY, width=1)
            self.y += 22
        else:
            self.text("Ιατρική Έκθεση", size=8, bold=True, color=GREY)
            self.page.draw_line(fitz.Point(LEFT, self.y + 2), fitz.Point(RIGHT, self.y + 2), color=(0.75, 0.78, 0.82), width=0.5)
            self.y += 14

    def ensure(self, height: float):
        if self.y + height > BOTTOM:
            self.new_page()

    def text(self, text: str, *, size: float = 10.2, bold: bool = False, color=(0, 0, 0), indent: float = 0):
        line_height = size * 1.38
        for line in _wrap(text, width=94 if not indent else 88):
            self.ensure(line_height)
            if line:
                self.page.insert_text(
                    fitz.Point(LEFT + indent, self.y),
                    line,
                    fontname="MRBold" if bold else "MRRegular",
                    fontsize=size,
                    color=color,
                )
            self.y += line_height

    def heading(self, title: str):
        self.ensure(38)
        self.y += 7
        self.text(title, size=11.5, bold=True, color=NAVY)
        self.y += 4

    def paragraph(self, text: str):
        if not str(text or "").strip():
            return
        self.text(text, size=10.1)
        self.y += 7

    def finish(self, signature_bytes: bytes | None = None) -> bytes:
        self.ensure(120)
        self.y += 18
        self.text("Με εκτίμηση,", size=9.5)
        if signature_bytes:
            validate_signature_image(signature_bytes)
            rect = fitz.Rect(LEFT, self.y, LEFT + 125, self.y + 48)
            self.page.insert_image(rect, stream=signature_bytes, keep_proportion=True, overlay=True)
            self.y += 54
        self.text(self.clinician.name, size=9.8, bold=True, color=NAVY)
        self.text(self.clinician.specialty, size=8.8, color=GREY)

        total = self.doc.page_count
        for index in range(total):
            page = self.doc[index]
            page.insert_font(fontname="MRRegular", fontfile=str(_font_path(False)))
            footer = f"Σελίδα {index + 1} από {total}"
            page.insert_text(fitz.Point(470, 812), footer, fontname="MRRegular", fontsize=7.5, color=GREY)

        output = self.doc.tobytes(garbage=4, deflate=True, clean=True)
        self.doc.close()
        return output


def build_medical_report_pdf(
    report: FinalMedicalReportV1,
    *,
    clinician: ClinicianProfile,
    signature_bytes: bytes | None = None,
) -> bytes:
    writer = _Writer(clinician)
    report_title = "ΙΑΤΡΟΝΟΜΙΚΗ / MEDICO-LEGAL ΕΚΘΕΣΗ" if report.case.report_type == "medico_legal_expert_report" else "ΙΑΤΡΙΚΗ ΕΚΘΕΣΗ"
    writer.text(report_title, size=16.5, bold=True, color=NAVY)
    writer.y += 12

    writer.heading("Στοιχεία υπόθεσης")
    details = [
        f"Όνομα: {report.case.patient_name}",
        f"{report.case.id_type}: {report.case.id_number}" if report.case.id_number else "",
        f"Ημερομηνία έκθεσης: {report.case.report_date.strftime('%d/%m/%Y')}",
        f"Ημερομηνία συμβάντος: {report.case.incident_date.strftime('%d/%m/%Y')}" if report.case.incident_date else "",
        f"Επάγγελμα: {report.case.occupation}" if report.case.occupation else "",
        f"Εντολέας: {report.case.instructing_party}" if report.case.instructing_party else "",
        f"Reference: {report.case.instructing_reference}" if report.case.instructing_reference else "",
    ]
    writer.paragraph("\n".join(item for item in details if item))

    for section in report.sections:
        if not section.draft_text.strip():
            continue
        writer.heading(section.title)
        writer.paragraph(section.draft_text)

    if report.research_text.strip():
        writer.heading("Βιβλιογραφική τεκμηρίωση")
        writer.paragraph(report.research_text)
    if report.citations:
        writer.heading("Βιβλιογραφία")
        for index, citation in enumerate(report.citations, start=1):
            label = citation.title.strip() or citation.url
            writer.paragraph(f"{index}. {label}\n{citation.url}")
    if report.declaration_text.strip():
        writer.heading("Δήλωση")
        writer.paragraph(report.declaration_text)

    return writer.finish(signature_bytes=signature_bytes)


__all__ = ["build_medical_report_pdf", "medical_report_filename"]
