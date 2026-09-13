from __future__ import annotations

import io
import re
import unicodedata
import zipfile
from xml.etree import ElementTree as ET

import fitz

from .report_models import MedicalReportAnalysisV1, ReportSourceV1, SourcePageV1

MAX_REPORT_FILES = 20
MAX_REPORT_FILE_BYTES = 12 * 1024 * 1024
MAX_REPORT_TOTAL_BYTES = 40 * 1024 * 1024
MAX_EXTRACTED_CHARS = 350_000
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


def _source(source_id: str, filename: str, pages: list[SourcePageV1], *, source_type: str = "other") -> ReportSourceV1:
    text_count = sum(len(page.text) for page in pages)
    status = "extracted" if text_count >= MIN_PDF_TEXT_CHARS else "no_extractable_text"
    return ReportSourceV1(
        source_id=source_id,
        filename=safe_display_filename(filename),
        source_type=source_type,
        status=status,
        page_count=max(1, len(pages)),
        character_count=text_count,
        pages=pages,
    )


def extract_pdf(content: bytes, source_id: str, filename: str) -> ReportSourceV1:
    if not content.startswith(b"%PDF"):
        raise ValueError("Το αρχείο δεν είναι έγκυρο PDF")
    try:
        document = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ValueError("Το PDF δεν μπορεί να αναγνωστεί") from exc
    try:
        if document.page_count < 1:
            raise ValueError("Το PDF δεν περιέχει σελίδες")
        pages = [
            SourcePageV1(page_number=index + 1, text=(document[index].get_text("text") or "").strip())
            for index in range(document.page_count)
        ]
        return _source(source_id, filename, pages)
    finally:
        document.close()


def extract_text(content: bytes, source_id: str, filename: str) -> ReportSourceV1:
    text = _decode_text(content).strip()
    return _source(source_id, filename, [SourcePageV1(page_number=1, text=text)])


def extract_docx(content: bytes, source_id: str, filename: str) -> ReportSourceV1:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            xml = archive.read("word/document.xml")
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
    return _source(source_id, filename, [SourcePageV1(page_number=1, text="\n".join(paragraphs))])


def extract_source_bytes(content: bytes, source_id: str, filename: str) -> ReportSourceV1:
    if not content:
        raise ValueError("Το αρχείο είναι κενό")
    if len(content) > MAX_REPORT_FILE_BYTES:
        raise ValueError("Το αρχείο υπερβαίνει τα 12 MiB")
    lower = safe_display_filename(filename).lower()
    if lower.endswith(".pdf"):
        return extract_pdf(content, source_id, filename)
    if lower.endswith((".txt", ".md")):
        return extract_text(content, source_id, filename)
    if lower.endswith(".docx"):
        return extract_docx(content, source_id, filename)
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
        page_count=1,
        character_count=len(value),
        pages=[SourcePageV1(page_number=1, text=value)],
    )


def enforce_source_totals(sources: list[ReportSourceV1]) -> None:
    chars = sum(source.character_count for source in sources)
    if chars > MAX_EXTRACTED_CHARS:
        raise ValueError("Το συνολικό εξαγόμενο κείμενο υπερβαίνει τους 350.000 χαρακτήρες")


def source_prompt_text(sources: list[ReportSourceV1]) -> str:
    blocks: list[str] = []
    for source in sources:
        blocks.append(f"=== SOURCE {source.source_id} | {source.filename} | declared_type={source.source_type} ===")
        if source.status != "extracted":
            blocks.append("[NO_EXTRACTABLE_TEXT]")
            continue
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


def deterministic_warnings(analysis: MedicalReportAnalysisV1, sources: list[ReportSourceV1]) -> list[str]:
    warnings: list[str] = []
    unreadable = [source.filename for source in sources if source.status == "no_extractable_text"]
    if unreadable:
        warnings.append("Χωρίς εξαγώγιμο κείμενο: " + ", ".join(unreadable))
    conflict_counts: dict[str, int] = {}
    for item in analysis.evidence_items:
        if item.conflict_key:
            conflict_counts[item.conflict_key] = conflict_counts.get(item.conflict_key, 0) + 1
    for key, count in sorted(conflict_counts.items()):
        if count > 1:
            warnings.append(f"Πιθανή σύγκρουση πηγών: {key} ({count} στοιχεία)")
    return warnings
