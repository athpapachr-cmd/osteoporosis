from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import fitz
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from clinic_utilities.clinical_documents import build_clinical_documents_router
from clinic_utilities.clinical_documents.api import MAX_DRAFT_JSON_BYTES
from clinic_utilities.clinical_documents.models import (
    ClinicianProfile,
    SickLeaveDocumentMetadataV1,
    SickLeaveDraftV1,
)
from clinic_utilities.clinical_documents.sick_leave import (
    MAX_SIGNATURE_BYTES,
    build_sick_leave_pdf,
    read_previous_sick_leave_pdf,
    reuse_options,
    sick_leave_filename,
    validate_signature_image,
)


SYNTHETIC_PATIENT = "Δοκιμαστικός Ασθενής"
SYNTHETIC_DIAGNOSIS = "Θλάση μαλακών μορίων αυχενικής μοίρας"
SYNTHETIC_KEY = "synthetic-clinical-key"
SYNTHETIC_PROFILE = {
    "name": "Δρ Δοκιμαστικός Ιατρός",
    "specialty": "Ορθοπαιδικός Χειρουργός",
    "phone": "+357 00 000000",
    "email": "doctor@example.invalid",
    "medical_center": "Δοκιμαστικό Ιατρικό Κέντρο",
}


def _draft(*, id_type: str = "ADT", id_number: str = "TEST-123") -> SickLeaveDraftV1:
    return SickLeaveDraftV1(
        patient_name=SYNTHETIC_PATIENT,
        id_type=id_type,
        id_number=id_number,
        diagnosis=SYNTHETIC_DIAGNOSIS,
        leave_from=date(2026, 9, 12),
        leave_to=date(2026, 9, 14),
        issued_on=date(2026, 9, 12),
    )


def _clinician() -> ClinicianProfile:
    return ClinicianProfile(
        name=SYNTHETIC_PROFILE["name"],
        specialty=SYNTHETIC_PROFILE["specialty"],
        phone=SYNTHETIC_PROFILE["phone"],
        email=SYNTHETIC_PROFILE["email"],
        clinic=SYNTHETIC_PROFILE["medical_center"],
    )


def _png_signature() -> bytes:
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 180, 60), False)
    pix.clear_with(255)
    return pix.tobytes("png")


def _router_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("CLINICAL_DATA_KEY", SYNTHETIC_KEY)
    monkeypatch.delenv("CLINICAL_DOCUMENTS_CLINICIAN_PROFILE_JSON", raising=False)
    monkeypatch.setenv("RF_DOCTOR_PROFILE_JSON", json.dumps(SYNTHETIC_PROFILE, ensure_ascii=False))
    app = FastAPI()
    app.include_router(build_clinical_documents_router())
    return TestClient(app)


def test_valid_adt_and_arc_and_nonnumeric_identifier_are_accepted():
    adt = _draft(id_type="ADT", id_number="A-12/β")
    arc = _draft(id_type="ARC", id_number="ARC-X9-Δ")
    assert adt.id_type == "ADT"
    assert arc.id_type == "ARC"
    assert arc.id_number == "ARC-X9-Δ"
    assert adt.inclusive_duration_days == 3


def test_leave_end_before_start_is_rejected():
    with pytest.raises(ValidationError):
        SickLeaveDraftV1(
            patient_name=SYNTHETIC_PATIENT,
            id_type="ADT",
            id_number="TEST-123",
            diagnosis=SYNTHETIC_DIAGNOSIS,
            leave_from=date(2026, 9, 15),
            leave_to=date(2026, 9, 14),
            issued_on=date(2026, 9, 12),
        )


def test_required_fields_and_unknown_fields_are_rejected():
    payload = {
        "patient_name": "",
        "id_type": "ADT",
        "id_number": "",
        "diagnosis": "",
        "leave_from": "2026-09-12",
        "leave_to": "2026-09-14",
        "issued_on": "2026-09-12",
        "unexpected_patient_state": "must-not-be-accepted",
    }
    with pytest.raises(ValidationError):
        SickLeaveDraftV1.model_validate(payload)


def test_pdf_is_a4_parseable_and_contains_expected_greek_text():
    pdf_bytes, metadata = build_sick_leave_pdf(_draft(), clinician=_clinician())
    assert pdf_bytes.startswith(b"%PDF")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        assert doc.page_count == 1
        page = doc[0]
        assert abs(page.rect.width - 595) < 2
        assert abs(page.rect.height - 842) < 2
        text = page.get_text("text")
        assert "ΒΕΒΑΙΩΣΗ ΑΣΘΕΝΕΙΑΣ" in text
        assert SYNTHETIC_PATIENT in text
        assert SYNTHETIC_DIAGNOSIS in text
        assert "12/09/2026" in text
        assert "14/09/2026" in text
        assert metadata.patient_name == SYNTHETIC_PATIENT
    finally:
        doc.close()


def test_greek_filename_includes_patient_and_date_but_excludes_diagnosis_and_id():
    draft = _draft(id_number="SECRET-ID")
    filename = sick_leave_filename(draft)
    assert filename.startswith("Αναρρωτική_Δοκιμαστικός_Ασθενής_")
    assert "12-09-2026" in filename
    assert "SECRET-ID" not in filename
    assert "Θλάση" not in filename


def test_pdf_metadata_round_trip_and_reuse_modes():
    pdf_bytes, original = build_sick_leave_pdf(_draft(), clinician=_clinician())
    imported = read_previous_sick_leave_pdf(pdf_bytes)
    assert imported.document_id == original.document_id
    assert imported.patient_name == SYNTHETIC_PATIENT
    assert imported.diagnosis == SYNTHETIC_DIAGNOSIS

    reuse = reuse_options(imported)
    extension = reuse["extension"]
    assert extension["patient_name"] == SYNTHETIC_PATIENT
    assert extension["diagnosis"] == SYNTHETIC_DIAGNOSIS
    assert extension["leave_from"] == "2026-09-15"
    assert extension["leave_to"] == ""
    assert extension["relation"] == "extension"
    assert extension["derived_from_document_id"] == original.document_id

    new_reason = reuse["new_leave_same_patient"]
    assert new_reason["patient_name"] == SYNTHETIC_PATIENT
    assert new_reason["id_number"] == "TEST-123"
    assert new_reason["diagnosis"] == ""
    assert new_reason["leave_from"] == ""
    assert new_reason["leave_to"] == ""
    assert new_reason["relation"] == "new_leave_same_patient"


def test_imported_metadata_rejects_inverted_dates_and_orphan_relation():
    common = {
        "schema": "sick_leave_certificate_v1",
        "version": "1.0",
        "document_id": "00000000-0000-4000-8000-000000000000",
        "patient_name": SYNTHETIC_PATIENT,
        "id_type": "ADT",
        "id_number": "TEST-123",
        "diagnosis": SYNTHETIC_DIAGNOSIS,
        "issued_on": "2026-09-12",
    }
    with pytest.raises(ValidationError):
        SickLeaveDocumentMetadataV1.model_validate(
            {**common, "leave_from": "2026-09-15", "leave_to": "2026-09-14"}
        )
    with pytest.raises(ValidationError):
        SickLeaveDocumentMetadataV1.model_validate(
            {
                **common,
                "leave_from": "2026-09-12",
                "leave_to": "2026-09-14",
                "relation": "extension",
                "derived_from_document_id": "",
            }
        )


def test_unknown_pdf_fails_without_visible_text_or_ocr_guessing():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Sick leave looking text without V1 metadata")
    raw = doc.tobytes()
    doc.close()
    with pytest.raises(ValueError, match="metadata"):
        read_previous_sick_leave_pdf(raw)


def test_signature_is_optional_and_valid_png_can_be_embedded():
    plain_pdf, _ = build_sick_leave_pdf(_draft(), clinician=_clinician())
    signature = _png_signature()
    width, height = validate_signature_image(signature, "image/png", "signature.png")
    assert width == 180
    assert height == 60
    signed_pdf, _ = build_sick_leave_pdf(_draft(), clinician=_clinician(), signature_bytes=signature)
    assert signed_pdf.startswith(b"%PDF")
    assert len(signed_pdf) > len(plain_pdf)


def test_invalid_or_oversize_signature_is_rejected():
    with pytest.raises(ValueError):
        validate_signature_image(b"not-an-image", "text/plain", "signature.txt")
    with pytest.raises(ValueError, match="2 MB"):
        validate_signature_image(b"x" * (MAX_SIGNATURE_BYTES + 1), "image/png", "signature.png")


def test_clinician_profile_falls_back_to_existing_rf_profile(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CLINICAL_DOCUMENTS_CLINICIAN_PROFILE_JSON", raising=False)
    monkeypatch.setenv("RF_DOCTOR_PROFILE_JSON", json.dumps(SYNTHETIC_PROFILE, ensure_ascii=False))
    profile = ClinicianProfile.from_environment()
    assert profile.name == SYNTHETIC_PROFILE["name"]
    assert profile.clinic == SYNTHETIC_PROFILE["medical_center"]


def test_protected_contract_and_pdf_api(monkeypatch: pytest.MonkeyPatch):
    client = _router_client(monkeypatch)
    path = "/clinical/clinic-utilities/sick-leave/api/contract"
    assert client.get(path).status_code == 401

    headers = {"X-Clinical-Key": SYNTHETIC_KEY}
    contract = client.get(path, headers=headers)
    assert contract.status_code == 200
    body = contract.json()
    assert body["schema"] == "sick_leave_certificate_v1"
    assert body["clinician_configured"] is True
    assert body["persistence"] == {
        "patient_database": False,
        "browser_patient_storage": False,
        "autosave": False,
        "signature_persisted": False,
    }
    assert body["limits"]["draft_json_bytes"] == MAX_DRAFT_JSON_BYTES

    draft_json = json.dumps(_draft().model_dump(mode="json"), ensure_ascii=False)
    pdf_response = client.post(
        "/clinical/clinic-utilities/sick-leave/api/pdf",
        headers=headers,
        data={"draft_json": draft_json},
    )
    assert pdf_response.status_code == 200
    assert pdf_response.headers["content-type"].startswith("application/pdf")
    disposition = pdf_response.headers["content-disposition"]
    assert "filename*=UTF-8''" in disposition
    assert "TEST-123" not in disposition
    assert "Θλάση" not in disposition

    previous = client.post(
        "/clinical/clinic-utilities/sick-leave/api/import-previous",
        headers=headers,
        files={"previous_pdf": ("previous.pdf", pdf_response.content, "application/pdf")},
    )
    assert previous.status_code == 200
    assert previous.json()["extension"]["leave_from"] == "2026-09-15"


def test_oversized_draft_payload_is_rejected_before_json_parse(monkeypatch: pytest.MonkeyPatch):
    client = _router_client(monkeypatch)
    response = client.post(
        "/clinical/clinic-utilities/sick-leave/api/pdf",
        headers={"X-Clinical-Key": SYNTHETIC_KEY},
        data={"draft_json": "x" * (MAX_DRAFT_JSON_BYTES + 1)},
    )
    assert response.status_code == 413


def test_page_and_contract_fail_closed_without_clinical_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLINICAL_DATA_KEY", SYNTHETIC_KEY)
    app = FastAPI()
    app.include_router(build_clinical_documents_router())
    client = TestClient(app)
    assert client.get("/clinical/clinic-utilities/sick-leave").status_code == 401
    assert client.get("/clinical/clinic-utilities/sick-leave/api/contract").status_code == 401


def test_browser_workspace_has_no_patient_or_signature_storage_api():
    js = Path("static/clinic-utilities/sick-leave/app.js").read_text(encoding="utf-8")
    assert "localStorage" not in js
    assert "sessionStorage" not in js
    assert "indexedDB" not in js
    assert "/api/import-previous" in js
    assert "signatureFile" in js


def test_runtime_is_mounted_and_navigation_exposes_sick_leave():
    main = Path("main.py").read_text(encoding="utf-8")
    helper = Path("static/baseline-audit/g4-workspace-ergonomics.js").read_text(encoding="utf-8")
    assert "build_clinical_documents_router" in main
    assert "app.include_router(build_clinical_documents_router())" in main
    assert "/clinical/clinic-utilities/sick-leave" in helper
    assert "Αναρρωτική άδεια" in helper


def test_phase1_package_has_no_database_or_patient_persistence_owner():
    root = Path("clinic_utilities/clinical_documents")
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    lowered = source.lower()
    assert "sqlalchemy" not in lowered
    assert "create_engine" not in lowered
    assert "sessionmaker" not in lowered
    assert "localstorage" not in lowered
    assert "sessionstorage" not in lowered
