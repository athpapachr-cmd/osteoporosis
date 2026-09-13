from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import fitz
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from clinic_utilities.clinical_documents.report_api import build_medical_report_router
from clinic_utilities.clinical_documents.report_models import (
    ClinicianResolutionV1,
    EvidenceItemV1,
    MedicalReportAnalysisV1,
    MedicalReportCaseV1,
    MedicalReportRefinementResultV1,
    ProviderUsageV1,
    ReportSectionDraftV1,
    SourcePageV1,
    SourceSummaryV1,
    TimelineEventV1,
)
from clinic_utilities.clinical_documents.report_sources import (
    apply_deterministic_sick_leave_facts,
    deterministic_warnings,
    extract_source_bytes,
    filter_contextual_warnings,
    validate_refinement_integrity,
    visual_source_from_pages,
)

SYNTH_KEY = "synthetic-key"


def _case(*, purpose="Σύνθεση κλινικής πορείας"):
    return MedicalReportCaseV1(
        patient_name="Synthetic Patient",
        id_type="ADT",
        id_number="SYNTH",
        incident_date=date(2026, 7, 4),
        report_date=date(2026, 9, 13),
        purpose_and_questions=purpose,
    )


def _analysis(source_id="src-upload-001"):
    return MedicalReportAnalysisV1(
        source_summaries=[SourceSummaryV1(source_id=source_id, proposed_source_type="other", summary="Synthetic source")],
        evidence_items=[EvidenceItemV1(
            evidence_id="ev-1", source_id=source_id, page_numbers=[1], evidence_type="other",
            statement="Synthetic evidence", conflict_key="incident_date",
        )],
        timeline=[TimelineEventV1(
            event_id="tl-1", event_date=date(2026, 7, 4), title="Synthetic event",
            summary="Synthetic timeline", source_ids=[source_id], evidence_ids=["ev-1"],
        )],
        report_sections=[ReportSectionDraftV1(
            section_id="clinical_course", title="Κλινική πορεία",
            draft_text="Synthetic report text", supporting_evidence_ids=["ev-1"],
        )],
    )


def _image_only_pdf():
    doc = fitz.open()
    page = doc.new_page()
    page.draw_rect(fitz.Rect(50, 50, 250, 120), color=(0, 0, 0), fill=(0.9, 0.9, 0.9))
    raw = doc.tobytes()
    doc.close()
    return raw


def test_new_source_types_are_accepted_and_semantics_are_present():
    for source_type in ("prescription", "imaging_referral", "specialist_referral", "lab_or_service_referral", "heidi_transcript"):
        source = extract_source_bytes(b"Synthetic source text", "src", "source.txt", source_type=source_type)
        assert source.source_type == source_type
    ai = Path("clinic_utilities/clinical_documents/report_ai.py").read_text(encoding="utf-8")
    assert "A referral/request means care was requested" in ai
    assert "A prescription means a medicine was prescribed" in ai
    assert "requested/pending" in ai


def test_sick_leave_text_is_structured_before_general_synthesis():
    text = "Synthetic certificate\nΑναρρωτική άδεια από 11/9/26 έως και 23/10/26"
    source = extract_source_bytes(text.encode("utf-8"), "src-leave", "leave.txt", source_type="sick_leave_certificate")
    assert any(note.startswith("SICK_LEAVE_INTERVAL|") for note in source.structured_notes)
    analysis = MedicalReportAnalysisV1()
    apply_deterministic_sick_leave_facts(analysis, [source])
    assert len(analysis.work_absence_intervals) == 1
    interval = analysis.work_absence_intervals[0]
    assert interval.leave_from == date(2026, 9, 11)
    assert interval.leave_to == date(2026, 10, 23)
    assert analysis.evidence_items[0].evidence_type == "work_absence"


def test_image_only_source_can_be_marked_visual_ai_and_review_required():
    source = extract_source_bytes(_image_only_pdf(), "src-scan", "scan.pdf", source_type="imaging_report")
    assert source.status == "no_extractable_text"
    visual = visual_source_from_pages(source, [SourcePageV1(page_number=1, text="Synthetic visible radiology text")])
    assert visual.status == "visual_extracted"
    assert visual.extraction_method == "visual_ai"
    assert visual.review_required is True
    assert "VISUAL_AI_EXTRACTION_REVIEW_REQUIRED" in visual.structured_notes


def test_clinician_resolution_preserves_original_evidence_and_resolves_conflict_warning():
    source = extract_source_bytes(b"Synthetic note", "src-upload-001", "note.txt")
    original = _analysis()
    original.evidence_items.append(EvidenceItemV1(
        evidence_id="ev-2", source_id="src-upload-001", page_numbers=[1], evidence_type="other",
        statement="Conflicting synthetic date", conflict_key="incident_date",
    ))
    updated = original.model_copy(deep=True)
    updated.clinician_resolutions.append(ClinicianResolutionV1(
        resolution_id="res-1", topic_or_conflict_key="incident_date",
        clinician_statement="Primary record accepted as authoritative.", related_evidence_ids=["ev-1", "ev-2"],
    ))
    validate_refinement_integrity(original, updated)
    assert not any("incident_date" in item for item in deterministic_warnings(updated, [source]))
    forged = updated.model_copy(deep=True)
    forged.evidence_items[0].statement = "Mutated source evidence"
    with pytest.raises(ValueError, match="Evidence Ledger"):
        validate_refinement_integrity(original, forged)


def test_missing_occupation_warning_is_context_sensitive():
    analysis = MedicalReportAnalysisV1(warnings=["Δεν δηλώθηκε επάγγελμα.", "Άλλη προειδοποίηση"])
    filter_contextual_warnings(_case(), analysis)
    assert analysis.warnings == ["Άλλη προειδοποίηση"]
    work_analysis = MedicalReportAnalysisV1(warnings=["Δεν δηλώθηκε επάγγελμα."])
    filter_contextual_warnings(_case(purpose="Εκτίμηση ικανότητας προς εργασία"), work_analysis)
    assert work_analysis.warnings == ["Δεν δηλώθηκε επάγγελμα."]


class FakeV11Provider:
    def visual_extract_pdf(self, content, source):
        return visual_source_from_pages(source, [SourcePageV1(page_number=1, text="Synthetic visually extracted text")]), ProviderUsageV1(model="fake-vision", total_tokens=5)

    def analyze(self, case, sources):
        source = next(item for item in sources if item.source_id.startswith("src-upload"))
        analysis = _analysis(source.source_id)
        analysis.source_summaries[0].proposed_source_type = source.source_type
        return analysis, ProviderUsageV1(model="fake-analysis", total_tokens=10)

    def refine(self, case, analysis, clinician_message):
        updated = analysis.model_copy(deep=True)
        updated.clinician_resolutions.append(ClinicianResolutionV1(
            resolution_id="res-api", topic_or_conflict_key="incident_date",
            clinician_statement=clinician_message, related_evidence_ids=["ev-1"],
        ))
        updated.report_sections[0].draft_text = "Revised synthetic report text"
        return MedicalReportRefinementResultV1(
            assistant_reply="Synthetic clarification accepted for review.",
            updated_analysis=updated,
            proposed_resolutions=[updated.clinician_resolutions[-1]],
            usage=ProviderUsageV1(model="fake-refine", total_tokens=4),
        )


def _client(monkeypatch):
    monkeypatch.setenv("CLINICAL_DATA_KEY", SYNTH_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-not-used")
    monkeypatch.setenv("CLINICAL_DOCUMENTS_AI_ENABLED", "true")
    monkeypatch.setenv("CLINICAL_DOCUMENTS_PHI_PROVIDER_APPROVED", "true")
    monkeypatch.setenv("RF_DOCTOR_PROFILE_JSON", json.dumps({
        "name": "Synthetic Doctor", "specialty": "Orthopaedics",
        "phone": "+35700000000", "email": "doctor@example.invalid",
    }))
    import clinic_utilities.clinical_documents.report_api as report_api
    monkeypatch.setattr(report_api, "_report_provider", lambda: FakeV11Provider())
    app = FastAPI()
    app.include_router(build_medical_report_router())
    return TestClient(app)


def test_api_visual_fallback_and_session_only_refinement(monkeypatch):
    client = _client(monkeypatch)
    headers = {"X-Clinical-Key": SYNTH_KEY}
    analyzed = client.post(
        "/clinical/clinic-utilities/medical-report/api/analyze",
        headers=headers,
        data={
            "case_json": json.dumps(_case().model_dump(mode="json")),
            "file_source_types_json": json.dumps(["imaging_report"]),
        },
        files={"files": ("scan.pdf", _image_only_pdf(), "application/pdf")},
    )
    assert analyzed.status_code == 200, analyzed.text
    body = analyzed.json()
    assert body["sources"][0]["status"] == "visual_extracted"
    assert body["sources"][0]["review_required"] is True
    assert body["usage"]["total_tokens"] == 15

    refined = client.post(
        "/clinical/clinic-utilities/medical-report/api/refine",
        headers=headers,
        json={
            "case": body["case"], "analysis": body["analysis"],
            "clinician_message": "Synthetic clinician resolution.",
        },
    )
    assert refined.status_code == 200, refined.text
    assert refined.json()["updated_analysis"]["evidence_items"] == body["analysis"]["evidence_items"]
    assert refined.json()["updated_analysis"]["clinician_resolutions"][0]["status"] == "confirmed"


def test_browser_v11_controls_exist_without_persistent_case_storage():
    sources_js = Path("static/clinic-utilities/medical-report/v1-1-sources.js").read_text(encoding="utf-8")
    refine_js = Path("static/clinic-utilities/medical-report/v1-1-refine.js").read_text(encoding="utf-8")
    index = Path("static/clinic-utilities/medical-report/index.html").read_text(encoding="utf-8")
    assert "Αφαίρεση" in sources_js
    assert "prescription" in sources_js and "imaging_referral" in sources_js and "heidi_transcript" in sources_js
    assert "DataTransfer" in sources_js
    assert "setInterval" in sources_js and "μην το υποβάλετε ξανά" in sources_js
    assert "/api/refine" in refine_js and "Συζήτηση / διευκρίνιση" in refine_js
    assert "v1-1-sources.js" in index and "v1-1-refine.js" in index
    combined = sources_js + refine_js
    assert "localStorage" not in combined and "sessionStorage" not in combined and "indexedDB" not in combined
