from __future__ import annotations

import base64
from datetime import date, timedelta
import io
import re
import unicodedata
import zipfile
from xml.etree import ElementTree as ET

import fitz

from .report_models import (
    EvidenceItemV1,
    MedicalReportAnalysisV1,
    MedicalReportCaseV1,
    ReportSourceV1,
    SourcePageV1,
    WorkAbsenceIntervalV1,
)
from .sick_leave import read_previous_sick_leave_pdf

MAX_REPORT_FILES = 20
MAX_REPORT_FILE_BYTES = 12 * 1024 * 1024
MAX_REPORT_TOTAL_BYTES = 40 * 1024 * 1024
MAX_EXTRACTED_CHARS = 350_000
MAX_PDF_PAGES = 5000
MAX_VISUAL_PDF_PAGES = 12
MAX_DOCX_XML_BYTES = 4 * 1024 * 1024
MIN_PDF_TEXT_CHARS = 20


def safe_display_filename(value: str) -> str:
    name = unicodedata.normalize("NFC", str(value or "").strip())
    name = name.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[\x00-\x1f]", "", name).strip()
    return (name[:240] or "source")


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1253", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Το αρχείο κειμένου δεν μπορεί να αποκωδικοποιηθεί")


def _source(source_id: str, filename: str, pages: list[SourcePageV1], *, source_type: str = "other", min_chars: int = 1) -> ReportSourceV1:
    text_count = sum(len(page.text) for page in pages)
    status = "extracted" if text_count >= min_chars else "no_extractable_text"
    return ReportSourceV1(
        source_id=source_id,
        filename=safe_display_filename(filename),
        source_type=source_type,
        status=status,
        extraction_method="text" if status == "extracted" else "none",
        review_required=False,
        page_count=max(1, len(pages)),
        character_count=text_count,
        pages=pages,
    )


def extract_pdf(content: bytes, source_id: str, filename: str, *, source_type: str = "other") -> ReportSourceV1:
    if not content.startswith(b"%PDF"):
        raise ValueError("Το αρχείο δεν είναι έγκυρο PDF")
    try:
        document = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ValueError("Το PDF δεν μπορεί να αναγνωστεί") from exc
    try:
        if document.page_count < 1:
            raise ValueError("Το PDF δεν περιέχει σελίδες")
        if document.page_count > MAX_PDF_PAGES:
            raise ValueError(f"Το PDF υπερβαίνει τις {MAX_PDF_PAGES} σελίδες")
        pages = [
            SourcePageV1(page_number=index + 1, text=(document[index].get_text("text") or "").strip())
            for index in range(document.page_count)
        ]
        source = _source(source_id, filename, pages, source_type=source_type, min_chars=MIN_PDF_TEXT_CHARS)
    finally:
        document.close()

    if source_type == "sick_leave_certificate":
        try:
            metadata = read_previous_sick_leave_pdf(content)
            source.structured_notes.append(
                f"SICK_LEAVE_INTERVAL|leave_from={metadata.leave_from.isoformat()}|leave_to={metadata.leave_to.isoformat()}|source=embedded_metadata"
            )
        except ValueError:
            pass
    _add_text_sick_leave_note(source)
    return source


def extract_text(content: bytes, source_id: str, filename: str, *, source_type: str = "other") -> ReportSourceV1:
    text = _decode_text(content).strip()
    source = _source(source_id, filename, [SourcePageV1(page_number=1, text=text)], source_type=source_type, min_chars=1)
    _add_text_sick_leave_note(source)
    return source


def extract_docx(content: bytes, source_id: str, filename: str, *, source_type: str = "other") -> ReportSourceV1:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            info = archive.getinfo("word/document.xml")
            if info.file_size > MAX_DOCX_XML_BYTES:
                raise ValueError("Το αποσυμπιεσμένο DOCX είναι υπερβολικά μεγάλο")
            xml = archive.read(info)
    except ValueError:
        raise
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ValueError("Το DOCX δεν μπορεί να αναγνωστεί") from exc
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ValueError("Το DOCX περιέχει μη έγκυρο XML") from exc
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs: list[str] = []
    for paragraph in root.iter(namespace + "p"):
        pieces = [node.text or "" for node in paragraph.iter(namespace + "t")]
        line = "".join(pieces).strip()
        if line:
            paragraphs.append(line)
    source = _source(source_id, filename, [SourcePageV1(page_number=1, text="\n".join(paragraphs))], source_type=source_type, min_chars=1)
    _add_text_sick_leave_note(source)
    return source


def extract_source_bytes(content: bytes, source_id: str, filename: str, *, source_type: str = "other") -> ReportSourceV1:
    if not content:
        raise ValueError("Το αρχείο είναι κενό")
    if len(content) > MAX_REPORT_FILE_BYTES:
        raise ValueError("Το αρχείο υπερβαίνει τα 12 MiB")
    lower = safe_display_filename(filename).lower()
    if lower.endswith(".pdf"):
        return extract_pdf(content, source_id, filename, source_type=source_type)
    if lower.endswith((".txt", ".md")):
        return extract_text(content, source_id, filename, source_type=source_type)
    if lower.endswith(".docx"):
        return extract_docx(content, source_id, filename, source_type=source_type)
    raise ValueError("Υποστηρίζονται μόνο PDF, TXT, MD και DOCX")


def build_clinician_context_source(text: str) -> ReportSourceV1 | None:
    value = str(text or "").strip()
    if not value:
        return None
    if len(value) > 20_000:
        raise ValueError("Το κλινικό ιστορικό/οδηγίες υπερβαίνουν το επιτρεπτό μέγεθος")
    return ReportSourceV1(
        source_id="src-clinician-context",
        filename="Κλινικό ιστορικό και οδηγίες ιατρού",
        source_type="clinician_note_self",
        status="extracted",
        extraction_method="clinician_text",
        review_required=False,
        page_count=1,
        character_count=len(value),
        pages=[SourcePageV1(page_number=1, text=value)],
    )


def render_pdf_pages_for_visual(content: bytes) -> list[tuple[int, str]]:
    try:
        document = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ValueError("Το PDF δεν μπορεί να αποδοθεί για οπτική ανάγνωση") from exc
    try:
        if document.page_count > MAX_VISUAL_PDF_PAGES:
            raise ValueError(f"Η οπτική ανάγνωση περιορίζεται σε {MAX_VISUAL_PDF_PAGES} σελίδες ανά PDF")
        rendered: list[tuple[int, str]] = []
        matrix = fitz.Matrix(1.5, 1.5)
        for index in range(document.page_count):
            pix = document[index].get_pixmap(matrix=matrix, alpha=False)
            raw = pix.tobytes("jpeg", jpg_quality=82)
            data_url = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
            rendered.append((index + 1, data_url))
        return rendered
    finally:
        document.close()


def visual_source_from_pages(source: ReportSourceV1, pages: list[SourcePageV1]) -> ReportSourceV1:
    text_count = sum(len(page.text) for page in pages)
    if text_count < 1:
        return source
    visual = source.model_copy(deep=True)
    visual.status = "visual_extracted"
    visual.extraction_method = "visual_ai"
    visual.review_required = True
    visual.pages = pages
    visual.character_count = text_count
    visual.structured_notes.append("VISUAL_AI_EXTRACTION_REVIEW_REQUIRED")
    _add_text_sick_leave_note(visual)
    return visual


def _parse_short_date(raw: str) -> date:
    day, month, year = [int(piece) for piece in re.split(r"[./-]", raw.strip())]
    if year < 100:
        year += 2000 if year < 70 else 1900
    return date(year, month, day)


_SICK_LEAVE_RE = re.compile(
    r"(?:αναρρωτικ(?:ή|ης)\s+άδεια\s*)?από\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\s*"
    r"(?:έως|εως)\s*(?:και\s*)?(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})",
    flags=re.IGNORECASE,
)


def _add_text_sick_leave_note(source: ReportSourceV1) -> None:
    if source.source_type != "sick_leave_certificate":
        return
    if any(note.startswith("SICK_LEAVE_INTERVAL|") for note in source.structured_notes):
        return
    for page in source.pages:
        match = _SICK_LEAVE_RE.search(page.text)
        if not match:
            continue
        try:
            start = _parse_short_date(match.group(1))
            end = _parse_short_date(match.group(2))
        except ValueError:
            continue
        if end < start:
            continue
        source.structured_notes.append(
            f"SICK_LEAVE_INTERVAL|leave_from={start.isoformat()}|leave_to={end.isoformat()}|page={page.page_number}|source=text"
        )
        break


def _structured_leave_note(source: ReportSourceV1) -> tuple[date, date, int | None] | None:
    for note in source.structured_notes:
        if not note.startswith("SICK_LEAVE_INTERVAL|"):
            continue
        fields = {}
        for part in note.split("|")[1:]:
            if "=" in part:
                key, value = part.split("=", 1)
                fields[key] = value
        try:
            start = date.fromisoformat(fields["leave_from"])
            end = date.fromisoformat(fields["leave_to"])
        except (KeyError, ValueError):
            continue
        if end < start:
            continue
        page = int(fields["page"]) if fields.get("page", "").isdigit() else None
        return start, end, page
    return None


def apply_deterministic_sick_leave_facts(analysis: MedicalReportAnalysisV1, sources: list[ReportSourceV1]) -> None:
    existing = {(item.leave_from, item.leave_to, tuple(item.source_ids)) for item in analysis.work_absence_intervals}
    evidence_ids = {item.evidence_id for item in analysis.evidence_items}
    for source in sources:
        if source.source_type != "sick_leave_certificate":
            continue
        fact = _structured_leave_note(source)
        if fact is None:
            continue
        start, end, page = fact
        key = (start, end, (source.source_id,))
        evidence_id = f"det-leave-evidence-{source.source_id}"
        if evidence_id not in evidence_ids:
            analysis.evidence_items.append(EvidenceItemV1(
                evidence_id=evidence_id,
                source_id=source.source_id,
                page_numbers=[page] if page else [],
                event_date=start,
                date_text=f"{start.strftime('%d/%m/%Y')}–{end.strftime('%d/%m/%Y')}",
                evidence_type="work_absence",
                statement=f"Τεκμηριωμένη αναρρωτική άδεια από {start.strftime('%d/%m/%Y')} έως {end.strftime('%d/%m/%Y')}",
                certainty="documented",
                requires_clinician_review=True,
            ))
            evidence_ids.add(evidence_id)
        if key not in existing:
            analysis.work_absence_intervals.append(WorkAbsenceIntervalV1(
                interval_id=f"det-leave-{source.source_id}",
                leave_from=start,
                leave_to=end,
                source_ids=[source.source_id],
                evidence_ids=[evidence_id],
                note="Deterministic extraction from sick-leave source",
                requires_clinician_review=True,
            ))
            existing.add(key)


def enforce_source_totals(sources: list[ReportSourceV1]) -> None:
    chars = sum(source.character_count for source in sources)
    if chars > MAX_EXTRACTED_CHARS:
        raise ValueError("Το συνολικό εξαγόμενο κείμενο υπερβαίνει τους 350.000 χαρακτήρες")


def source_prompt_text(sources: list[ReportSourceV1]) -> str:
    blocks: list[str] = []
    for source in sources:
        blocks.append(
            f"=== SOURCE {source.source_id} | {source.filename} | declared_type={source.source_type} | extraction={source.extraction_method} ==="
        )
        for note in source.structured_notes:
            blocks.append(f"[STRUCTURED SOURCE NOTE] {note}")
        if source.status == "no_extractable_text":
            blocks.append("[NO_EXTRACTABLE_TEXT]")
            continue
        if source.status == "visual_extracted":
            blocks.append("[VISUAL_AI_EXTRACTION_REVIEW_REQUIRED]")
        for page in source.pages:
            blocks.append(f"--- PAGE {page.page_number} ---\n{page.text}")
    return "\n\n".join(blocks)


def validate_analysis_references(analysis: MedicalReportAnalysisV1, sources: list[ReportSourceV1]) -> None:
    source_map = {source.source_id: source for source in sources}
    if len(source_map) != len(sources):
        raise ValueError("Duplicate source id")
    evidence_map = {}
    for item in analysis.evidence_items:
        if item.evidence_id in evidence_map:
            raise ValueError(f"Duplicate evidence id: {item.evidence_id}")
        evidence_map[item.evidence_id] = item
        source = source_map.get(item.source_id)
        if source is None:
            raise ValueError(f"Unknown source reference: {item.source_id}")
        if any(page < 1 or page > source.page_count for page in item.page_numbers):
            raise ValueError(f"Invalid source page reference for {item.evidence_id}")
    for summary in analysis.source_summaries:
        if summary.source_id not in source_map:
            raise ValueError(f"Unknown source summary reference: {summary.source_id}")
    for event in analysis.timeline:
        for source_id in event.source_ids:
            if source_id not in source_map:
                raise ValueError(f"Unknown timeline source: {source_id}")
        for evidence_id in event.evidence_ids:
            if evidence_id not in evidence_map:
                raise ValueError(f"Unknown timeline evidence: {evidence_id}")
    for interval in analysis.work_absence_intervals:
        for source_id in interval.source_ids:
            if source_id not in source_map:
                raise ValueError(f"Unknown work-absence source: {source_id}")
        for evidence_id in interval.evidence_ids:
            if evidence_id not in evidence_map:
                raise ValueError(f"Unknown work-absence evidence: {evidence_id}")
    for resolution in analysis.clinician_resolutions:
        for evidence_id in resolution.related_evidence_ids:
            if evidence_id not in evidence_map:
                raise ValueError(f"Unknown clinician-resolution evidence: {evidence_id}")
    for diagnosis in analysis.diagnosis_analyses:
        for evidence_id in diagnosis.supporting_evidence_ids:
            if evidence_id not in evidence_map:
                raise ValueError(f"Unknown diagnosis evidence: {evidence_id}")
    for section in analysis.report_sections:
        for evidence_id in section.supporting_evidence_ids:
            if evidence_id not in evidence_map:
                raise ValueError(f"Unknown report-section evidence: {evidence_id}")
    for need in analysis.future_needs:
        for evidence_id in need.basis_evidence_ids:
            if evidence_id not in evidence_map:
                raise ValueError(f"Unknown future-needs evidence: {evidence_id}")


def validate_refinement_integrity(original: MedicalReportAnalysisV1, updated: MedicalReportAnalysisV1) -> None:
    if [item.model_dump(mode="json") for item in original.evidence_items] != [item.model_dump(mode="json") for item in updated.evidence_items]:
        raise ValueError("Refinement attempted to mutate the original Evidence Ledger")
    if [item.model_dump(mode="json") for item in original.source_summaries] != [item.model_dump(mode="json") for item in updated.source_summaries]:
        raise ValueError("Refinement attempted to mutate source summaries")
    evidence_ids = {item.evidence_id for item in original.evidence_items}
    for resolution in updated.clinician_resolutions:
        if any(item not in evidence_ids for item in resolution.related_evidence_ids):
            raise ValueError("Refinement returned an unknown clinician-resolution evidence id")
    for event in updated.timeline:
        if any(item not in evidence_ids for item in event.evidence_ids):
            raise ValueError("Refinement returned an unknown timeline evidence id")
    for section in updated.report_sections:
        if any(item not in evidence_ids for item in section.supporting_evidence_ids):
            raise ValueError("Refinement returned an unknown report-section evidence id")


def filter_contextual_warnings(case: MedicalReportCaseV1, analysis: MedicalReportAnalysisV1) -> None:
    purpose = (case.purpose_and_questions or "").lower()
    work_specific = any(token in purpose for token in ("επάγγελ", "εργασία", "επιστροφή στην εργασία", "work capacity", "return to work", "occupation"))
    if work_specific:
        return
    analysis.warnings = [
        warning for warning in analysis.warnings
        if not re.search(r"\b(επάγγελ|occupation)\w*", warning, flags=re.IGNORECASE)
    ]


def deterministic_warnings(analysis: MedicalReportAnalysisV1, sources: list[ReportSourceV1]) -> list[str]:
    warnings: list[str] = []
    unreadable = [source.filename for source in sources if source.status == "no_extractable_text"]
    if unreadable:
        warnings.append("Χωρίς εξαγώγιμο κείμενο: " + ", ".join(unreadable))
    visual = [source.filename for source in sources if source.status == "visual_extracted"]
    if visual:
        warnings.append("Οπτική ανάγνωση AI — απαιτεί ιατρικό έλεγχο: " + ", ".join(visual))

    conflict_counts: dict[str, int] = {}
    resolved = {item.topic_or_conflict_key for item in analysis.clinician_resolutions if item.topic_or_conflict_key}
    for item in analysis.evidence_items:
        if item.conflict_key and item.conflict_key not in resolved:
            conflict_counts[item.conflict_key] = conflict_counts.get(item.conflict_key, 0) + 1
    for key, count in sorted(conflict_counts.items()):
        if count > 1:
            warnings.append(f"Πιθανή σύγκρουση πηγών: {key} ({count} στοιχεία)")

    intervals = sorted(analysis.work_absence_intervals, key=lambda item: (item.leave_from, item.leave_to, item.interval_id))
    previous = None
    for interval in intervals:
        if previous is not None:
            if interval.leave_from <= previous.leave_to:
                warnings.append(
                    "Επικάλυψη αναρρωτικών περιόδων: "
                    f"{previous.leave_from.strftime('%d/%m/%Y')}–{previous.leave_to.strftime('%d/%m/%Y')} και "
                    f"{interval.leave_from.strftime('%d/%m/%Y')}–{interval.leave_to.strftime('%d/%m/%Y')}"
                )
            elif interval.leave_from > previous.leave_to + timedelta(days=1):
                gap = (interval.leave_from - previous.leave_to).days - 1
                warnings.append(f"Κενό {gap} ημερών μεταξύ αναρρωτικών περιόδων μετά τις {previous.leave_to.strftime('%d/%m/%Y')}")
        if previous is None or interval.leave_to > previous.leave_to:
            previous = interval
    return warnings
