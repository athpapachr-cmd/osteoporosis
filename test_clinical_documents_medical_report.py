from __future__ import annotations

from datetime import date
import io
import json
from pathlib import Path
import zipfile

import fitz
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from clinic_utilities.clinical_documents.report_ai import build_generalized_research_prompt
from clinic_utilities.clinical_documents.report_api import build_medical_report_router
from clinic_utilities.clinical_documents.report_models import (
    DiagnosisAnalysisV1,
    EvidenceItemV1,
    FinalMedicalReportV1,
    FutureNeedDraftV1,
    MedicalReportAnalysisV1,
    MedicalReportCaseV1,
    MedicalReportResearchResultV1,
    PrognosisQuestionV1,
    ProviderUsageV1,
    ReportSectionDraftV1,
    SourceSummaryV1,
    TimelineEventV1,
)
from clinic_utilities.clinical_documents.report_pdf import build_medical_report_pdf, medical_report_filename
from clinic_utilities.clinical_documents.report_sources import (
    MAX_REPORT_FILE_BYTES,
    build_clinician_context_source,
    extract_source_bytes,
    validate_analysis_references,
)
from clinic_utilities.clinical_documents.models import ClinicianProfile

SYNTHETIC_KEY = "synthetic-clinical-key"
SYNTHETIC_PROFILE = {
    "name": "Δρ Δοκιμαστικός Ιατρός",
    "specialty": "Ορθοπαιδικός Χειρουργός",
    "phone": "+357 00 000000",
    "email": "doctor@example.invalid",
}


def _case(report_type="accident_medical_report"):
    return MedicalReportCaseV1(
        report_type=report_type,
        patient_name="Δοκιμαστικός Ασθενής",
        id_type="ADT",
        id_number="SYNTH-123",
        occupation="Δοκιμαστική εργασία",
        incident_date=date(2026, 8, 1),
        report_date=date(2026, 9, 13),
        instructing_party="Δοκιμαστικός εντολέας",
        instructing_reference="REF-SYNTH",
        purpose_and_questions="Σύνθεση της κλινικής πορείας.",
        clinician_context="Μετά το συμβάν αναφέρθηκε αυχεναλγία. Στην εξέταση διαπιστώθηκε περιορισμός κίνησης.",
    )


def _clinician():
    return ClinicianProfile(**SYNTHETIC_PROFILE)


def _analysis(source_id="src-clinician-context"):
    return MedicalReportAnalysisV1(
        source_summaries=[SourceSummaryV1(source_id=source_id, proposed_source_type="clinician_note_self", summary="Συνθετική κλινική πηγή")],
        evidence_items=[EvidenceItemV1(evidence_id="ev-1", source_id=source_id, page_numbers=[1], event_date=date(2026, 8, 1), evidence_type="clinician_observed", statement="Περιορισμός κινητικότητας αυχένα")],
        timeline=[TimelineEventV1(event_id="tl-1", event_date=date(2026, 8, 1), title="Αρχική εκτίμηση", summary="Καταγράφηκε περιορισμός κινητικότητας.", source_ids=[source_id], evidence_ids=["ev-1"])],
        diagnosis_analyses=[DiagnosisAnalysisV1(diagnosis="Κάκωση μαλακών μορίων αυχένα", supporting_evidence_ids=["ev-1"], causation_draft="Συμβατή χρονικά με το συμβάν.")],
        report_sections=[ReportSectionDraftV1(section_id="clinical_course", title="Χρονολογική κλινική πορεία", draft_text="Κατά την αρχική εκτίμηση διαπιστώθηκε περιορισμός κινητικότητας.", supporting_evidence_ids=["ev-1"])],
        prognosis_questions=[PrognosisQuestionV1(diagnosis_or_problem="Κάκωση μαλακών μορίων αυχένα", question="Ποια είναι η συνήθης διάρκεια αποκατάστασης;", rationale="Απαιτείται τεκμηρίωση πρόγνωσης.")],
        future_needs=[FutureNeedDraftV1(problem="Επίμονα συμπτώματα", suggested_need="Επανεκτίμηση εφόσον επιμένουν τα συμπτώματα.", basis_evidence_ids=["ev-1"])],
    )


class FakeProvider:
    def analyze(self, case, sources):
        source_id = next(source.source_id for source in sources if source.status == "extracted")
        return _analysis(source_id), ProviderUsageV1(model="fake-model", input_tokens=100, output_tokens=50, total_tokens=150)

    def research(self, case, analysis):
        return MedicalReportResearchResultV1(
            research_text="Συνθετική βιβλιογραφική σύνθεση.",
            citations=[{"title": "Synthetic source", "url": "https://example.invalid/source"}],
            queries=["synthetic prognosis query"],
            usage=ProviderUsageV1(model="fake-research", total_tokens=20, web_search_calls=1),
        )


def _docx_bytes(text="Συνθετική γνωμάτευση"):
    document_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>'''
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as zf:
        zf.writestr("word/document.xml", document_xml)
    return output.getvalue()


def _pdf_bytes(text="Συνθετική ακτινολογική γνωμάτευση"):
    doc = fitz.open(); page = doc.new_page(); page.insert_text((72, 72), text); raw = doc.tobytes(); doc.close(); return raw


def _router_client(monkeypatch, *, phi=True):
    monkeypatch.setenv("CLINICAL_DATA_KEY", SYNTHETIC_KEY)
    monkeypatch.setenv("RF_DOCTOR_PROFILE_JSON", json.dumps(SYNTHETIC_PROFILE, ensure_ascii=False))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key")
    monkeypatch.setenv("CLINICAL_DOCUMENTS_AI_ENABLED", "true")
    monkeypatch.setenv("CLINICAL_DOCUMENTS_PHI_PROVIDER_APPROVED", "true" if phi else "false")
    import clinic_utilities.clinical_documents.report_api as report_api
    monkeypatch.setattr(report_api, "_report_provider", lambda: FakeProvider())
    app = FastAPI(); app.include_router(build_medical_report_router()); return TestClient(app)


def test_valid_accident_and_medico_legal_cases_accept_unicode():
    assert _case().report_type == "accident_medical_report"
    assert _case("medico_legal_expert_report").report_type == "medico_legal_expert_report"
    assert "Δοκιμαστικός" in _case().patient_name


def test_pdf_source_extraction_preserves_page_provenance():
    source = extract_source_bytes(_pdf_bytes(), "src-1", "report.pdf")
    assert source.status == "extracted" and source.page_count == 1
    assert source.pages[0].page_number == 1
    assert "Συνθετική" in source.pages[0].text


def test_txt_markdown_and_docx_extract_locally():
    txt = extract_source_bytes("Κλινική σημείωση".encode(), "src-txt", "note.txt")
    md = extract_source_bytes("# Ιστορικό\nΣυνθετικό".encode(), "src-md", "note.md")
    docx = extract_source_bytes(_docx_bytes(), "src-docx", "note.docx")
    assert txt.status == md.status == docx.status == "extracted"
    assert "γνωμάτευση" in docx.pages[0].text


def test_unsupported_and_oversize_files_are_rejected():
    with pytest.raises(ValueError, match="PDF, TXT, MD"):
        extract_source_bytes(b"hello", "src", "image.png")
    with pytest.raises(ValueError, match="12 MiB"):
        extract_source_bytes(b"x" * (MAX_REPORT_FILE_BYTES + 1), "src", "huge.txt")


def test_image_only_pdf_is_flagged_without_ocr():
    doc = fitz.open(); doc.new_page(); raw = doc.tobytes(); doc.close()
    source = extract_source_bytes(raw, "src", "scan.pdf")
    assert source.status == "no_extractable_text"
    assert source.character_count == 0


def test_clinician_context_is_distinct_self_note_source():
    source = build_clinician_context_source("Δικό μου συνθετικό ιστορικό")
    assert source is not None
    assert source.source_type == "clinician_note_self"
    assert source.source_id == "src-clinician-context"


def test_analysis_reference_validation_accepts_valid_and_rejects_forged_refs():
    source = build_clinician_context_source("Αρκετό συνθετικό κείμενο για extraction")
    analysis = _analysis(source.source_id)
    validate_analysis_references(analysis, [source])
    forged = analysis.model_copy(deep=True)
    forged.evidence_items[0].source_id = "missing-source"
    with pytest.raises(ValueError, match="Unknown source"):
        validate_analysis_references(forged, [source])


def test_duplicate_evidence_ids_fail_closed():
    source = build_clinician_context_source("Αρκετό συνθετικό κείμενο για extraction")
    analysis = _analysis(source.source_id)
    analysis.evidence_items.append(analysis.evidence_items[0].model_copy())
    with pytest.raises(ValueError, match="Duplicate evidence"):
        validate_analysis_references(analysis, [source])


def test_diagnosis_and_future_needs_are_always_review_required():
    analysis = _analysis()
    assert analysis.diagnosis_analyses[0].requires_clinician_review is True
    assert analysis.future_needs[0].requires_clinician_review is True
    with pytest.raises(ValidationError):
        DiagnosisAnalysisV1(diagnosis="Synthetic", requires_clinician_review=False)


def test_research_prompt_excludes_direct_identifiers():
    case = _case(); prompt = build_generalized_research_prompt(case, _analysis())
    assert case.patient_name not in prompt
    assert case.id_number not in prompt
    assert case.instructing_reference not in prompt
    assert "Ποια είναι η συνήθης διάρκεια" in prompt


def test_final_report_requires_clinician_confirmation():
    with pytest.raises(ValidationError):
        FinalMedicalReportV1(case=_case(), sections=_analysis().report_sections, clinician_confirmed=False)


def test_long_final_report_generates_parseable_multipage_pdf_and_safe_filename():
    sections = [ReportSectionDraftV1(section_id="clinical_course", title="Κλινική πορεία", draft_text=("Συνθετικό ιατρικό κείμενο χωρίς πραγματικά δεδομένα. " * 700))]
    report = FinalMedicalReportV1(case=_case(), sections=sections, clinician_confirmed=True)
    raw = build_medical_report_pdf(report, clinician=_clinician())
    doc = fitz.open(stream=raw, filetype="pdf")
    try:
        assert doc.page_count >= 2
        assert "ΙΑΤΡΙΚΗ ΕΚΘΕΣΗ" in "\n".join(page.get_text() for page in doc)
    finally: doc.close()
    filename = medical_report_filename(report)
    assert "Δοκιμαστικός_Ασθενής" in filename
    assert "SYNTH-123" not in filename


def test_protected_api_analyze_research_and_final_pdf(monkeypatch):
    client = _router_client(monkeypatch)
    headers = {"X-Clinical-Key": SYNTHETIC_KEY}
    assert client.get("/clinical/clinic-utilities/medical-report/api/contract").status_code == 401
    contract = client.get("/clinical/clinic-utilities/medical-report/api/contract", headers=headers)
    assert contract.status_code == 200
    assert contract.json()["persistence"]["patient_case_database"] is False

    case_json = json.dumps(_case().model_dump(mode="json"), ensure_ascii=False)
    analyzed = client.post("/clinical/clinic-utilities/medical-report/api/analyze", headers=headers, data={"case_json": case_json}, files={"files": ("note.txt", "Συνθετική εξωτερική σημείωση".encode(), "text/plain")})
    assert analyzed.status_code == 200, analyzed.text
    body = analyzed.json(); assert body["analysis"]["evidence_items"][0]["source_id"]
    assert body["usage"]["model"] == "fake-model"

    researched = client.post("/clinical/clinic-utilities/medical-report/api/research", headers=headers, json={"case": body["case"], "analysis": body["analysis"]})
    assert researched.status_code == 200
    assert researched.json()["citations"][0]["url"] == "https://example.invalid/source"

    final = FinalMedicalReportV1(case=_case(), sections=_analysis().report_sections, clinician_confirmed=True)
    pdf = client.post("/clinical/clinic-utilities/medical-report/api/pdf", headers=headers, data={"report_json": json.dumps(final.model_dump(mode="json"), ensure_ascii=False)})
    assert pdf.status_code == 200
    assert pdf.headers["content-type"].startswith("application/pdf")


def test_analyze_fails_closed_without_phi_provider_approval(monkeypatch):
    client = _router_client(monkeypatch, phi=False)
    headers = {"X-Clinical-Key": SYNTHETIC_KEY}
    response = client.post("/clinical/clinic-utilities/medical-report/api/analyze", headers=headers, data={"case_json": json.dumps(_case().model_dump(mode="json"), ensure_ascii=False)})
    assert response.status_code == 503
    assert "δεν έχει εγκριθεί" in response.text


def test_browser_has_no_case_persistence_apis_and_navigation_is_wired():
    js = Path("static/clinic-utilities/medical-report/app.js").read_text(encoding="utf-8")
    assert "localStorage" not in js and "sessionStorage" not in js and "indexedDB" not in js
    main = Path("main.py").read_text(encoding="utf-8")
    nav = Path("static/baseline-audit/g4-workspace-ergonomics.js").read_text(encoding="utf-8")
    assert "build_medical_report_router" in main
    assert "/clinical/clinic-utilities/medical-report" in nav
    assert "Ιατρικές εκθέσεις" in nav


def test_existing_sick_leave_package_remains_present():
    assert Path("clinic_utilities/clinical_documents/sick_leave.py").is_file()
    assert Path("static/clinic-utilities/sick-leave/app.js").is_file()
