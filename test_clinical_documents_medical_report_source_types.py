from __future__ import annotations

from datetime import date
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from clinic_utilities.clinical_documents.report_api import build_medical_report_router
from clinic_utilities.clinical_documents.report_models import MedicalReportAnalysisV1, ProviderUsageV1, SourceSummaryV1


class FakeProvider:
    def analyze(self, case, sources):
        source = next(item for item in sources if item.source_id.startswith("src-upload-"))
        return (
            MedicalReportAnalysisV1(
                source_summaries=[
                    SourceSummaryV1(
                        source_id=source.source_id,
                        proposed_source_type=source.source_type,
                        summary="Synthetic classified source",
                    )
                ]
            ),
            ProviderUsageV1(model="fake-source-classifier"),
        )


def _client(monkeypatch):
    monkeypatch.setenv("CLINICAL_DATA_KEY", "synthetic-key")
    monkeypatch.setenv("RF_DOCTOR_PROFILE_JSON", json.dumps({
        "name": "Synthetic Doctor",
        "specialty": "Orthopaedics",
        "phone": "+35700000000",
        "email": "synthetic@example.invalid",
    }))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-not-used")
    monkeypatch.setenv("CLINICAL_DOCUMENTS_AI_ENABLED", "true")
    monkeypatch.setenv("CLINICAL_DOCUMENTS_PHI_PROVIDER_APPROVED", "true")
    import clinic_utilities.clinical_documents.report_api as report_api
    monkeypatch.setattr(report_api, "_report_provider", lambda: FakeProvider())
    app = FastAPI()
    app.include_router(build_medical_report_router())
    return TestClient(app)


def _case_json():
    return json.dumps({
        "report_type": "accident_medical_report",
        "patient_name": "Synthetic Patient",
        "id_type": "ADT",
        "id_number": "SYNTH",
        "report_date": date(2026, 9, 13).isoformat(),
        "clinician_context": "",
    })


def test_uploaded_source_type_is_clinician_controlled_and_preserved(monkeypatch):
    client = _client(monkeypatch)
    response = client.post(
        "/clinical/clinic-utilities/medical-report/api/analyze",
        headers={"X-Clinical-Key": "synthetic-key"},
        data={
            "case_json": _case_json(),
            "file_source_types_json": json.dumps(["imaging_report"]),
        },
        files={"files": ("imaging.txt", b"Synthetic imaging report with sufficient text", "text/plain")},
    )
    assert response.status_code == 200, response.text
    assert response.json()["sources"][0]["source_type"] == "imaging_report"


def test_invalid_or_misaligned_source_classification_fails_closed(monkeypatch):
    client = _client(monkeypatch)
    headers = {"X-Clinical-Key": "synthetic-key"}
    invalid = client.post(
        "/clinical/clinic-utilities/medical-report/api/analyze",
        headers=headers,
        data={"case_json": _case_json(), "file_source_types_json": json.dumps(["not-a-source-type"])},
        files={"files": ("note.txt", b"Synthetic medical record", "text/plain")},
    )
    assert invalid.status_code == 422

    misaligned = client.post(
        "/clinical/clinic-utilities/medical-report/api/analyze",
        headers=headers,
        data={"case_json": _case_json(), "file_source_types_json": json.dumps(["imaging_report", "lab_report"])},
        files={"files": ("note.txt", b"Synthetic medical record", "text/plain")},
    )
    assert misaligned.status_code == 422


def test_browser_exposes_editable_source_classification_without_persistence():
    js = Path("static/clinic-utilities/medical-report/app.js").read_text(encoding="utf-8")
    assert "source-type-select" in js
    assert "file_source_types_json" in js
    assert "localStorage" not in js
    assert "sessionStorage" not in js
    assert "indexedDB" not in js
