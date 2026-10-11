from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError
from pypdf import PdfReader
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from clinic_utilities.rf.api import RFApplicationDraft, _assess_imaging_pdf, _validate_a1, build_rf_router
from clinic_utilities.rf.catalog import INDICATIONS, DoctorProfile, ProductProfile
from clinic_utilities.rf.parsers import parse_medications, parse_physio_dates
from clinic_utilities.rf.pdf import build_official_rf_pdf
from clinic_utilities.rf.persistence import (
    initialize_rf_tables,
    list_procedure_history,
    record_legacy_procedure,
)


CLINICAL_KEY = "synthetic-clinical-key"
DOCTOR_JSON = json.dumps(
    {
        "name": "Synthetic Doctor",
        "gesy_code": "SYN-DOCTOR",
        "specialty": "Synthetic Specialty",
        "medical_center": "Synthetic Center",
        "phone": "00000000",
        "email": "synthetic@example.invalid",
    }
)
PRODUCT_JSON = json.dumps(
    {
        "medikey": {"code": "SYN-M", "description": "Synthetic Medikey", "quantity": "1"},
        "diros": {"code": "SYN-D", "description": "Synthetic DIROS", "quantity": "1"},
        "thermedico": {"code": "SYN-T", "description": "Synthetic Thermedico", "quantity": "1"},
    }
)


def memory_engine():
    return create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def make_pdf(page_count: int) -> bytes:
    doc = fitz.open()
    try:
        for _ in range(page_count):
            doc.new_page(width=595, height=842)
        return doc.tobytes()
    finally:
        doc.close()


def make_text_pdf(text: str) -> bytes:
    doc = fitz.open()
    try:
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), text, fontsize=10)
        return doc.tobytes()
    finally:
        doc.close()


def make_a1_draft(**overrides):
    data = {
        "pathway": "A1",
        "patient_name": "Synthetic Patient",
        "identity_number": "SYN123",
        "gesy_number": "GESY-SYN",
        "age": 67,
        "product_key": "medikey",
        "indication_code": "KNEE_OA_KL34",
        "laterality": "left",
        "exact_location": "Αριστερό γόνατο",
        "rf_reason_codes": ["FAILED_PHARMACOLOGIC", "FAILED_CONSERVATIVE"],
        "pain_onset_date": "2026-01-01",
        "pain_onset_vas": 8,
        "last_assessment_date": "2026-04-01",
        "last_assessment_vas": 7,
        "full_medication_text": "Arcoxia 90 mg για 3 μήνες\nPanadol 1 g για 3 μήνες",
        "physio_dates_text": "01/02/2026\n08/02/2026\n15/02/2026",
    }
    data.update(overrides)
    return RFApplicationDraft(**data)


class RFMedicationParserTests(unittest.TestCase):
    def test_greek_accented_duration_and_automatic_three_plus_three(self):
        result = parse_medications(
            "\n".join(
                [
                    "Arcoxia 90 mg για 3 μήνες",
                    "Brufen 600 mg για 2 μήνες",
                    "Voltaren 75 mg για 1 μήνα",
                    "Naprosyn 500 mg για 4 εβδομάδες",
                    "Panadol 1 g για 3 μήνες",
                    "Parcoten 1 g για 2 μήνες",
                    "Tramadex 100 mg για 1 μήνα",
                    "Lyrica 75 mg για 6 εβδομάδες",
                ]
            )
        )
        self.assertEqual(len(result["auto_selected_nsaids"]), 3)
        self.assertEqual(len(result["auto_selected_others"]), 3)
        arcoxia = next(item for item in result["nsaid_candidates"] if item["canonical_key"] == "etoricoxib")
        self.assertEqual(arcoxia["dose"], "90 mg")
        self.assertEqual(arcoxia["duration"], "3 μήνες")

    def test_duplicate_brand_same_active_family_prefers_longer_documented_trial(self):
        result = parse_medications("Tramadex 100 mg για 1 μήνα\nMabron 50 mg για 2 μήνες")
        tramadol = [item for item in result["other_candidates"] if item["canonical_key"] == "tramadol"]
        self.assertEqual(len(tramadol), 1)
        self.assertEqual(tramadol[0]["duration"], "2 μήνες")
        self.assertEqual(tramadol[0]["source_text"], "Mabron 50 mg για 2 μήνες")


class RFPhysioParserTests(unittest.TestCase):
    def test_dates_are_deduplicated_sorted_and_counted(self):
        result = parse_physio_dates("15/02/2026\n01/02/2026\n08/02/2026\n01/02/2026")
        self.assertEqual(result["start_date"], "2026-02-01")
        self.assertEqual(result["end_date"], "2026-02-15")
        self.assertEqual(result["treatment_count"], 3)
        self.assertEqual(result["invalid_or_ambiguous_tokens"], [])

    def test_short_date_without_year_is_flagged(self):
        result = parse_physio_dates("01/02\n08/02/2026")
        self.assertIn("01/02", result["invalid_or_ambiguous_tokens"])


class RFValidationTests(unittest.TestCase):
    def test_zero_age_is_rejected(self):
        with self.assertRaises(ValidationError):
            make_a1_draft(age=0)

    def test_a1_requires_at_least_three_calendar_months(self):
        draft = make_a1_draft(last_assessment_date="2026-03-31")
        with self.assertRaises(HTTPException) as caught:
            _validate_a1(draft, INDICATIONS[draft.indication_code])
        self.assertEqual(caught.exception.status_code, 422)
        self.assertIn("τρεις μήνες", caught.exception.detail)

    def test_si_requires_documented_intervention(self):
        draft = make_a1_draft(
            indication_code="SI_DEGENERATIVE",
            exact_location="Αριστερή ιερολαγόνια",
            intervention=None,
        )
        with self.assertRaises(HTTPException) as caught:
            _validate_a1(draft, INDICATIONS[draft.indication_code])
        self.assertEqual(caught.exception.status_code, 422)
        self.assertIn("ιερολαγόνια", caught.exception.detail.lower())


class RFPersistenceTests(unittest.TestCase):
    def test_legacy_procedure_is_deduplicated_and_filterable(self):
        engine = memory_engine()
        initialize_rf_tables(engine)
        payload = {
            "identity_number": "ID-123",
            "indication_code": "KNEE_OA_KL34",
            "site_key": "knee",
            "laterality": "left",
            "exact_location": "Αριστερό γόνατο",
            "actual_procedure_date": "2026-09-04",
            "vas_before": 8,
            "vas_after": 3,
            "last_followup_date": "2026-09-05",
            "last_followup_vas": 4,
        }
        first = record_legacy_procedure(engine, payload)
        second = record_legacy_procedure(engine, payload)
        self.assertEqual(first, second)
        rows = list_procedure_history(engine, "ID 123", site_key="knee", laterality="left")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["provenance"], "clinician_manual")


class RFPdfTests(unittest.TestCase):
    def setUp(self):
        self.doctor = DoctorProfile(
            name="Synthetic Doctor",
            gesy_code="SYN",
            specialty="Synthetic",
            medical_center="Synthetic Center",
            phone="000",
            email="synthetic@example.invalid",
        )
        self.product = ProductProfile(
            key="medikey",
            label="Medikey",
            code="SYN-M",
            description="Synthetic product",
            quantity="1",
        )
        self.imaging = make_pdf(1)

    def _template_path(self, tempdir: str) -> Path:
        path = Path(tempdir) / "official.pdf"
        path.write_bytes(make_pdf(6))
        return path

    def test_a1_selects_only_common_plus_a1_pages_and_appends_imaging(self):
        data = {
            "pathway": "A1",
            "application_date": "2026-09-05",
            "patient_name": "Synthetic Patient",
            "age": 67,
            "identity_number": "SYN123",
            "gesy_number": "GESY-SYN",
            "indication_code": "KNEE_OA_KL34",
            "site_key": "knee",
            "exact_location": "Αριστερό γόνατο",
            "rf_reason_text": "Synthetic reason",
            "pain_onset_date": "2026-01-01",
            "pain_onset_vas": 8,
            "last_assessment_date": "2026-04-01",
            "last_assessment_vas": 7,
            "nsaid_trials": [{"drug_name": "Etoricoxib", "dose": "90 mg", "duration": "3 μήνες"}],
            "other_analgesic_trials": [{"drug_name": "Παρακεταμόλη", "dose": "1 g", "duration": "3 μήνες"}],
            "adverse_effects": [],
            "physio": {"start_date": "2026-02-01", "end_date": "2026-02-15", "treatment_count": 3},
            "additional_notes": "",
        }
        with tempfile.TemporaryDirectory() as tempdir:
            output = build_official_rf_pdf(
                data,
                doctor=self.doctor,
                product=self.product,
                radiology_pdf_bytes=self.imaging,
                template_path=self._template_path(tempdir),
            )
        self.assertTrue(output.startswith(b"%PDF"))
        self.assertEqual(len(PdfReader(io.BytesIO(output)).pages), 5)

    def test_a2_selects_only_common_plus_a2_pages_and_appends_imaging(self):
        data = {
            "pathway": "A2",
            "application_date": "2026-09-05",
            "patient_name": "Synthetic Patient",
            "age": 67,
            "identity_number": "SYN123",
            "gesy_number": "GESY-SYN",
            "indication_code": "KNEE_OA_KL34",
            "site_key": "knee",
            "exact_location": "Αριστερό γόνατο",
            "additional_notes": "",
        }
        prior = {
            "actual_procedure_date": "2026-09-04",
            "vas_before": 8,
            "vas_after": 3,
            "last_followup_date": "2026-09-05",
            "last_followup_vas": 4,
        }
        with tempfile.TemporaryDirectory() as tempdir:
            output = build_official_rf_pdf(
                data,
                doctor=self.doctor,
                product=self.product,
                radiology_pdf_bytes=self.imaging,
                template_path=self._template_path(tempdir),
                prior_history=prior,
            )
        self.assertEqual(len(PdfReader(io.BytesIO(output)).pages), 4)


class RFNativeApiTests(unittest.TestCase):
    def setUp(self):
        self.engine = memory_engine()
        self.env = patch.dict(
            os.environ,
            {
                "CLINICAL_DATA_KEY": CLINICAL_KEY,
                "RF_DOCTOR_PROFILE_JSON": DOCTOR_JSON,
                "RF_PRODUCT_CATALOG_JSON": PRODUCT_JSON,
            },
            clear=False,
        )
        self.env.start()
        app = FastAPI()
        app.include_router(build_rf_router(self.engine))
        self.client = TestClient(app)
        self.headers = {"X-Clinical-Key": CLINICAL_KEY}

    def tearDown(self):
        self.client.close()
        self.env.stop()

    def test_native_routes_require_existing_clinical_auth(self):
        response = self.client.get("/clinical/clinic-utilities/rf/api/contract")
        self.assertEqual(response.status_code, 401)

    def test_contract_exposes_category_a_only_and_not_secret_values(self):
        response = self.client.get("/clinical/clinic-utilities/rf/api/contract", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["category"], "A")
        self.assertEqual(set(body["pathways"]), {"A1", "A2"})
        self.assertNotIn("doctor", body)
        self.assertNotIn("SYN-DOCTOR", response.text)
        self.assertFalse(any(key.startswith("B") or key.startswith("G") for key in body["indications"]))

    def test_a2_legacy_transition_accepts_recent_real_procedure_without_old_ten_week_rule(self):
        draft = {
            "pathway": "A2",
            "patient_name": "Synthetic Patient",
            "identity_number": "TRANSITION-1",
            "gesy_number": "GESY-SYN",
            "age": 67,
            "product_key": "medikey",
            "indication_code": "KNEE_OA_KL34",
            "laterality": "left",
            "exact_location": "Αριστερό γόνατο",
            "imaging_review_confirmed": True,
            "legacy_history": {
                "actual_procedure_date": "2026-09-04",
                "vas_before": 8,
                "vas_after": 3,
                "last_followup_date": "2026-09-05",
                "last_followup_vas": 4,
            },
        }
        with patch("clinic_utilities.rf.api.build_official_rf_pdf", return_value=b"%PDF-synthetic"):
            response = self.client.post(
                "/clinical/clinic-utilities/rf/api/create",
                headers=self.headers,
                data={"draft_json": json.dumps(draft)},
                files={"imaging_report": ("imaging.pdf", make_pdf(1), "application/pdf")},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "application/pdf")
        history = self.client.post(
            "/clinical/clinic-utilities/rf/api/history",
            headers=self.headers,
            json={"identity_number": "TRANSITION-1", "site_key": "knee", "laterality": "left"},
        )
        self.assertEqual(history.status_code, 200)
        self.assertTrue(history.json()["found"])
        self.assertEqual(history.json()["procedures"][0]["actual_procedure_date"], "2026-09-04")


class RFImagingSemanticGuardTests(unittest.TestCase):
    def setUp(self):
        self.engine = memory_engine()
        self.env = patch.dict(
            os.environ,
            {
                "CLINICAL_DATA_KEY": CLINICAL_KEY,
                "RF_DOCTOR_PROFILE_JSON": DOCTOR_JSON,
                "RF_PRODUCT_CATALOG_JSON": PRODUCT_JSON,
            },
            clear=False,
        )
        self.env.start()
        app = FastAPI()
        app.include_router(build_rf_router(self.engine))
        self.client = TestClient(app)
        self.headers = {"X-Clinical-Key": CLINICAL_KEY}

    def tearDown(self):
        self.client.close()
        self.env.stop()

    def _preview(self, content: bytes):
        return self.client.post(
            "/clinical/clinic-utilities/rf/api/validate-imaging",
            headers=self.headers,
            files={"imaging_report": ("report.pdf", content, "application/pdf")},
        )

    def _create(self, content: bytes, confirmed: bool):
        draft = make_a1_draft(imaging_review_confirmed=confirmed).model_dump(mode="json")
        with patch("clinic_utilities.rf.api.build_official_rf_pdf", return_value=b"%PDF-synthetic"):
            return self.client.post(
                "/clinical/clinic-utilities/rf/api/create",
                headers=self.headers,
                data={"draft_json": json.dumps(draft)},
                files={"imaging_report": ("report.pdf", content, "application/pdf")},
            )

    def test_radiology_text_is_supported_without_confirmation(self):
        pdf = make_text_pdf("Radiology report MRI right knee. Findings and conclusion.")
        assessment = _assess_imaging_pdf(pdf)
        self.assertEqual(assessment["status"], "imaging_supported")
        response = self._preview(pdf)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "imaging_supported")
        self.assertFalse(response.json()["requires_confirmation"])
        self.assertNotIn("Findings and conclusion", response.text)
        self.assertEqual(self._create(pdf, confirmed=False).status_code, 200)

    def test_clear_laboratory_report_is_rejected_even_if_confirmed(self):
        pdf = make_text_pdf(
            "BIOCHEMISTRY blood serum plasma reference range laboratory validator "
            "Calcium 9.0 mg/dL Magnesium 2.0 mg/dL Phosphate 4.0 mg/dL"
        )
        assessment = _assess_imaging_pdf(pdf)
        self.assertEqual(assessment["status"], "clearly_non_imaging")
        response = self._preview(pdf)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "clearly_non_imaging")
        created = self._create(pdf, confirmed=True)
        self.assertEqual(created.status_code, 422)
        self.assertIn("μη απεικονιστική", created.json()["detail"])

    def test_textless_pdf_requires_explicit_clinician_confirmation(self):
        pdf = make_pdf(1)
        response = self._preview(pdf)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ambiguous_or_unreadable")
        self.assertTrue(response.json()["requires_confirmation"])
        self.assertEqual(self._create(pdf, confirmed=False).status_code, 422)
        self.assertEqual(self._create(pdf, confirmed=True).status_code, 200)

    def test_fake_pdf_magic_bytes_are_not_accepted(self):
        response = self._preview(b"%PDF-not-a-real-document")
        self.assertEqual(response.status_code, 422)



# Test-only synthetic JPEG generation; requires no Pillow dependency.
def make_horos_jpeg(orientation: int = 1, width: int = 160, height: int = 100) -> bytes:
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, width, height), False)
    quadrants = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for y in range(height):
        for x in range(width):
            pix.set_pixel(x, y, quadrants[2 * int(y >= height // 2) + int(x >= width // 2)])
    jpeg = pix.tobytes("jpg")
    # Little-endian TIFF, one EXIF orientation SHORT tag (0x0112).
    tiff = (
        b"II*\x00\x08\x00\x00\x00" + b"\x01\x00"
        + b"\x12\x01\x03\x00\x01\x00\x00\x00"
        + bytes([orientation, 0, 0, 0]) + b"\x00\x00\x00\x00"
    )
    exif = b"Exif\x00\x00" + tiff
    return jpeg[:2] + b"\xff\xe1" + (len(exif) + 2).to_bytes(2, "big") + exif + jpeg[2:]


def make_two_page_imaging() -> bytes:
    doc = fitz.open()
    try:
        doc.new_page().insert_text((72, 72), "MRI radiology synthetic REPORT-FIRST")
        doc.new_page().insert_text((72, 72), "REPORT-SECOND")
        return doc.tobytes()
    finally:
        doc.close()


class RFDoctorHorosExtensionTests(unittest.TestCase):
    """Protected synthetic API + official template, no production records/provider."""
    def setUp(self):
        self.engine = memory_engine()
        self.secondary = {
            "name": "Synthetic Doctor Two",
            "gesy_code": "SYN-D2",
            "specialty": "Synthetic Orthopaedics",
            "medical_center": "Synthetic Clinic Two",
            "phone": "55500222",
            "email": "doctor2@example.invalid",
        }
        self.env = patch.dict(os.environ, {
            "CLINICAL_DATA_KEY": CLINICAL_KEY,
            "RF_DOCTOR_PROFILE_JSON": DOCTOR_JSON,
            "RF_PRODUCT_CATALOG_JSON": PRODUCT_JSON,
        })
        self.env.start()
        app = FastAPI()
        app.include_router(build_rf_router(self.engine))
        self.client = TestClient(app)
        self.headers = {"X-Clinical-Key": CLINICAL_KEY}
        created = self.client.post(
            "/clinical/clinic-utilities/rf/api/doctors",
            headers=self.headers, json=self.secondary,
        )
        self.assertEqual(created.status_code, 201, created.text)
        self.secondary_key = created.json()["key"]

    def tearDown(self):
        self.client.close()
        self.env.stop()

    def _draft(self, pathway="A1", **changes):
        data = make_a1_draft().model_dump()
        data["pathway"] = pathway
        if pathway == "A2":
            data["legacy_history"] = {
                "actual_procedure_date": "2026-07-02",
                "vas_before": 8, "vas_after": 3,
                "last_followup_date": "2026-09-01",
                "last_followup_vas": 5,
            }
        data.update(changes)
        return data

    def _create(self, draft=None, images=(), imaging=None):
        files = [("imaging_report", ("report.pdf", imaging if imaging is not None else make_two_page_imaging(), "application/pdf"))]
        files += [("horos_jpegs", (f"horos-{i}.jpg", content, "image/jpeg")) for i, content in enumerate(images)]
        return self.client.post(
            "/clinical/clinic-utilities/rf/api/create",
            headers=self.headers,
            data={"draft_json": json.dumps(draft if draft is not None else self._draft())},
            files=files,
        )

    def _rows(self):
        from sqlalchemy import select
        from sqlalchemy.orm import Session
        from clinic_utilities.rf.persistence import RFApplicationORM, RFProcedureHistoryORM
        with Session(self.engine) as session:
            return (
                session.scalars(select(RFApplicationORM)).all(),
                session.scalars(select(RFProcedureHistoryORM)).all(),
            )

    def test_doctor_contract_and_server_side_six_pdf_fields_a1_a2(self):
        contract = self.client.get("/clinical/clinic-utilities/rf/api/contract", headers=self.headers).json()
        self.assertTrue(contract["doctors"]["papachristou"]["configured"])
        self.assertTrue(contract["doctors"][self.secondary_key]["configured"])
        self.assertNotIn("doctor2@example.invalid", json.dumps(contract))
        for pathway in ("A1", "A2"):
            for key, expected in (("papachristou", json.loads(DOCTOR_JSON)),
                                  (self.secondary_key, self.secondary)):
                with self.subTest(pathway=pathway, doctor=key):
                    response = self._create(self._draft(pathway, doctor_key=key))
                    self.assertEqual(response.status_code, 200, response.text[:300])
                    doc = fitz.open(stream=response.content, filetype="pdf")
                    try:
                        self.assertEqual(doc.page_count, (4 if pathway == "A1" else 3) + 2)
                        page_text = doc[0].get_text()
                        for field in ("name", "gesy_code", "specialty", "medical_center", "phone", "email"):
                            self.assertIn(expected[field], page_text)
                        self.assertIn("REPORT-FIRST", doc[-2].get_text())
                        self.assertIn("REPORT-SECOND", doc[-1].get_text())
                    finally:
                        doc.close()
        applications, previous = self._rows()
        self.assertCountEqual([row.payload_json["doctor_key"] for row in applications],
                         ["papachristou", self.secondary_key, "papachristou", self.secondary_key])
        self.assertEqual(len(previous), 1)  # same A.2 procedure deduplicated across request doctors
        self.assertEqual({row.actual_procedure_date for row in previous}, {"2026-07-02"})

    def test_legacy_omission_persisted_default_but_historical_absence_unattributed(self):
        from clinic_utilities.rf.persistence import record_application
        raw = self._draft()
        raw.pop("doctor_key")
        response = self._create(raw)
        self.assertEqual(response.status_code, 200)
        applications, _ = self._rows()
        self.assertEqual(applications[0].payload_json["doctor_key"], "papachristou")
        old = dict(applications[0].payload_json)
        old.pop("doctor_key")
        record_application(self.engine, old)
        applications, _ = self._rows()
        self.assertEqual(sum("doctor_key" not in x.payload_json for x in applications), 1)

    def test_invalid_blank_null_unknown_and_unconfigured_doctor_fail_before_write(self):
        for key in ("", None, "unknown", " " + self.secondary_key):
            with self.subTest(key=key):
                res = self._create(self._draft(doctor_key=key))
                self.assertEqual(res.status_code, 422, res.text[:300])
        self.assertEqual(len(self._rows()[0]), 0)
        doctor = self.client.get(
            "/clinical/clinic-utilities/rf/api/doctors", headers=self.headers
        ).json()["doctors"]
        second = next(row for row in doctor if row["key"] == self.secondary_key)
        status = self.client.post(
            f"/clinical/clinic-utilities/rf/api/doctors/{self.secondary_key}/status",
            headers=self.headers, json={"revision": second["revision"], "active": False},
        )
        self.assertEqual(status.status_code, 200, status.text)
        res = self._create(self._draft(doctor_key=self.secondary_key))
        self.assertEqual(res.status_code, 422, res.text[:300])
        self.assertEqual(len(self._rows()[0]), 0)


    def test_managed_profiles_crud_snapshot_history_and_restart(self):
        base = "/clinical/clinic-utilities/rf/api"
        self.assertEqual(self.client.get(f"{base}/doctors").status_code, 401)
        self.assertEqual(self.client.post(f"{base}/doctors", json=self.secondary).status_code, 401)
        self.assertEqual(self.client.get(f"{base}/doctors", headers=self.headers).status_code, 200)
        before = self._create(self._draft(doctor_key=self.secondary_key))
        self.assertEqual(before.status_code, 200, before.text[:250])
        first = self._rows()[0][0].payload_json
        self.assertEqual(first["doctor_snapshot"], self.secondary)
        self.assertEqual(first["doctor_key"], self.secondary_key)
        self.assertNotIn("approval_status", first)
        self.assertNotIn("followup_assignee", first)

        doctors = self.client.get(f"{base}/doctors", headers=self.headers).json()["doctors"]
        original = next(row for row in doctors if row["key"] == self.secondary_key)
        updated = {**self.secondary, "name": "Synthetic Doctor Two Updated",
                   "medical_center": "Synthetic Updated Clinic"}
        revision = original["revision"]
        changed = self.client.put(
            f"{base}/doctors/{self.secondary_key}", headers=self.headers,
            json={**updated, "revision": revision},
        )
        self.assertEqual(changed.status_code, 200, changed.text)
        self.assertEqual(changed.json()["revision"], revision + 1)
        self.assertEqual(self.client.put(
            f"{base}/doctors/{self.secondary_key}", headers=self.headers,
            json={**self.secondary, "revision": revision},
        ).status_code, 409)
        self.assertEqual(self.client.post(
            f"{base}/doctors", headers=self.headers, json=updated,
        ).status_code, 409 if updated["gesy_code"] == self.secondary["gesy_code"] else 201)
        self.assertEqual(self._rows()[0][0].payload_json["doctor_snapshot"], self.secondary)

        after = self._create(self._draft(doctor_key=self.secondary_key))
        self.assertEqual(after.status_code, 200, after.text[:250])
        document = fitz.open(stream=after.content, filetype="pdf")
        try:
            self.assertIn(updated["name"], document[0].get_text())
            self.assertIn(updated["medical_center"], document[0].get_text())
        finally:
            document.close()
        snapshots = [row.payload_json["doctor_snapshot"] for row in self._rows()[0]]
        self.assertIn(self.secondary, snapshots)
        self.assertIn(updated, snapshots)

        disabled = self.client.post(
            f"{base}/doctors/{self.secondary_key}/status", headers=self.headers,
            json={"revision": changed.json()["revision"], "active": False},
        )
        self.assertEqual(disabled.status_code, 200)
        self.assertEqual(self._create(self._draft(doctor_key=self.secondary_key)).status_code, 422)
        self.assertEqual(len(self._rows()[0]), 2)
        self.assertFalse(self.client.get(f"{base}/contract", headers=self.headers).json()
                         ["doctors"][self.secondary_key]["configured"])
        self.assertEqual(self.client.post(
            f"{base}/doctors/{self.secondary_key}/status", headers=self.headers,
            json={"revision": changed.json()["revision"], "active": True},
        ).status_code, 409)
        enabled = self.client.post(
            f"{base}/doctors/{self.secondary_key}/status", headers=self.headers,
            json={"revision": disabled.json()["revision"], "active": True},
        )
        self.assertEqual(enabled.status_code, 200)
        self.assertTrue(enabled.json()["active"])

        # Initial default is imported from legacy config once; managed edits win on restart.
        default = next(row for row in doctors if row["key"] == "papachristou")
        revised_default = {**{k: default[k] for k in (
            "name", "gesy_code", "specialty", "medical_center", "phone", "email")},
            "medical_center": "Managed Default Clinic", "revision": default["revision"]}
        self.assertEqual(self.client.put(
            f"{base}/doctors/papachristou", headers=self.headers, json=revised_default
        ).status_code, 200)
        restarted = FastAPI()
        restarted.include_router(build_rf_router(self.engine))
        with TestClient(restarted) as client:
            directory = client.get(f"{base}/doctors", headers=self.headers).json()["doctors"]
            self.assertEqual(next(r for r in directory if r["key"] == "papachristou")
                             ["medical_center"], "Managed Default Clinic")
            self.assertTrue(next(r for r in directory if r["key"] == self.secondary_key)["active"])

    def test_more_doctors_without_environment_or_code_change(self):
        base = "/clinical/clinic-utilities/rf/api"
        extra = {**self.secondary, "name": "Third Synthetic Physician", "gesy_code": "SYN-D3",
                 "email": "third@example.invalid"}
        created = self.client.post(f"{base}/doctors", headers=self.headers, json=extra)
        self.assertEqual(created.status_code, 201)
        key = created.json()["key"]
        self.assertNotIn(key, ("papachristou", self.secondary_key))
        output = self._create(self._draft(doctor_key=key))
        self.assertEqual(output.status_code, 200, output.text[:250])
        self.assertEqual(self._rows()[0][0].payload_json["doctor_key"], key)
        self.assertEqual(self._rows()[0][0].payload_json["doctor_snapshot"], extra)


    def test_zero_one_three_images_page_order_and_readable_imaging(self):
        variants = [[], [make_horos_jpeg(1)],
                    [make_horos_jpeg(1), make_horos_jpeg(2), make_horos_jpeg(6)]]
        expected_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for images in variants:
            with self.subTest(count=len(images)):
                response = self._create(images=images)
                self.assertEqual(response.status_code, 200, response.text[:250])
                doc = fitz.open(stream=response.content, filetype="pdf")
                try:
                    self.assertEqual(doc.page_count, 4 + 2 + len(images))
                    self.assertIn("REPORT-FIRST", doc[4].get_text())
                    self.assertIn("REPORT-SECOND", doc[5].get_text())
                    for image_page, color in zip(list(doc)[6:], expected_colors):
                        info = image_page.get_image_info()
                        self.assertEqual(len(info), 1)
                        bbox = fitz.Rect(info[0]["bbox"])
                        pixel = image_page.get_pixmap(
                            clip=fitz.Rect(bbox.x0 + 4, bbox.y0 + 4, bbox.x0 + 6, bbox.y0 + 6)
                        ).samples[:3]
                        self.assertTrue(all(abs(x - y) < 45 for x, y in zip(pixel, color)),
                                        (tuple(pixel), color))
                finally:
                    doc.close()

    def test_all_eight_exif_orientations_including_mirrors(self):
        # Expected upper-left color after EXIF rotation/flip, with quadrant color source R,G,B,Y.
        corners = {
            1: (255, 0, 0), 2: (0, 255, 0),
            3: (255, 255, 0), 4: (0, 0, 255),
            5: (255, 0, 0), 6: (0, 0, 255),
            7: (255, 255, 0), 8: (0, 255, 0),
        }
        for orientation, expected_color in corners.items():
            with self.subTest(orientation=orientation):
                result = self._create(images=[make_horos_jpeg(orientation)])
                self.assertEqual(result.status_code, 200, result.text[:250])
                doc = fitz.open(stream=result.content, filetype="pdf")
                try:
                    image = doc[-1]
                    self.assertEqual(image.rect.width > image.rect.height, orientation <= 4)
                    bbox = fitz.Rect(image.get_image_info()[0]["bbox"])
                    pixel = image.get_pixmap(
                        clip=fitz.Rect(bbox.x0 + 4, bbox.y0 + 4, bbox.x0 + 6, bbox.y0 + 6)
                    ).samples[:3]
                    self.assertTrue(all(abs(x - y) < 45 for x, y in zip(pixel, expected_color)),
                                    (orientation, tuple(pixel), expected_color))
                finally:
                    doc.close()

    def test_corrupt_oversized_dimensional_and_too_many_jpegs_have_zero_writes(self):
        corrupt = b"\xff\xd8\xffBROKEN\xff\xd9"
        oversized = b"\xff\xd8\xff" + (b"\x00" * (8 * 1024 * 1024)) + b"\xff\xd9"
        cases = [
            ([corrupt], 422),
            ([oversized], 413),
            ([make_horos_jpeg(width=7001, height=2)], 422),
            ([make_horos_jpeg()] * 11, 413),
        ]
        for images, expected_status in cases:
            with self.subTest(expected=expected_status, count=len(images)):
                self.assertEqual(self._create(self._draft("A2"), images).status_code, expected_status)
                self.assertEqual([len(x) for x in self._rows()], [0, 0])

    def test_repaired_eoi_cannot_hide_truncated_jpeg_or_write_database(self):
        # P1-01 reproduction: tolerant decoders fill missing scan area with grey.
        original = make_horos_jpeg(6)
        self.assertGreater(len(original), 900)
        damaged = original[:-900] + b"\xff\xd9"
        self.assertTrue(damaged.startswith(b"\xff\xd8\xff"))
        self.assertTrue(damaged.endswith(b"\xff\xd9"))
        for pathway, uploads in (("A1", [damaged]), ("A2", [make_horos_jpeg(), damaged])):
            with self.subTest(pathway=pathway):
                result = self._create(self._draft(pathway), images=uploads)
                self.assertEqual(result.status_code, 422, result.text[:300])
                self.assertEqual([len(rows) for rows in self._rows()], [0, 0])

    def test_horos_does_not_bypass_mandatory_imaging_pdf_semantics(self):
        image = make_horos_jpeg()
        draft = self._draft()
        # The unmodified imaging validator must still reject a clearly non-imaging PDF.
        lab = make_text_pdf("BIOCHEMISTRY serum plasma reference range laboratory validator calcium magnesium")
        self.assertEqual(self._create(draft, [image], lab).status_code, 422)
        self.assertEqual(self._create(draft, [image], make_pdf(1)).status_code, 422)
        self.assertEqual([len(x) for x in self._rows()], [0, 0])
        without_imaging = self.client.post(
            "/clinical/clinic-utilities/rf/api/create",
            headers=self.headers,
            data={"draft_json": json.dumps(draft)},
            files={"horos_jpegs": ("only.jpg", image, "image/jpeg")},
        )
        self.assertEqual(without_imaging.status_code, 422)
        self.assertEqual([len(x) for x in self._rows()], [0, 0])




class RFReuseSavedDraftTests(unittest.TestCase):
    def setUp(self):
        self.engine = memory_engine()
        self.env = patch.dict(os.environ, {
            "CLINICAL_DATA_KEY": CLINICAL_KEY,
            "RF_DOCTOR_PROFILE_JSON": DOCTOR_JSON,
            "RF_PRODUCT_CATALOG_JSON": PRODUCT_JSON,
        }, clear=False)
        self.env.start()
        app = FastAPI()
        app.include_router(build_rf_router(self.engine))
        self.client = TestClient(app)
        self.headers = {"X-Clinical-Key": CLINICAL_KEY}
        self.base = "/clinical/clinic-utilities/rf/api"

    def tearDown(self):
        self.client.close()
        self.env.stop()

    def test_incomplete_a1_a2_drafts_saved_reopened_revisioned_and_no_binary(self):
        from sqlalchemy.orm import Session
        from clinic_utilities.rf.persistence import RFApplicationORM, RFProcedureHistoryORM
        self.assertEqual(self.client.get(self.base + "/drafts").status_code, 401)
        self.assertEqual(self.client.post(self.base + "/drafts", json={"fields":{}}).status_code, 401)
        first = self.client.post(self.base + "/drafts", headers=self.headers, json={
            "fields":{"pathway":"A1","patient_name":"Μαρία Παπαδοπούλου",
                      "identity_number":"","nsaid_trials":[],
                      "additional_notes":"Ελλιπές πρόχειρο"}
        })
        self.assertEqual(first.status_code, 201, first.text)
        obj = first.json()
        self.assertEqual(obj["revision"], 1)
        self.assertTrue(obj["id"])
        recent = self.client.get(self.base + "/drafts", headers=self.headers)
        self.assertEqual(recent.status_code, 200, recent.text)
        self.assertEqual(len(recent.json()["drafts"]), 1)
        self.assertEqual(self.client.get(self.base + "/drafts", headers=self.headers,
                          params={"query":"παπαδοπουλου"}).json()["drafts"][0]["id"], obj["id"])
        reopened = self.client.get(self.base + "/drafts/" + obj["id"], headers=self.headers)
        self.assertEqual(reopened.status_code, 200)
        self.assertEqual(reopened.json()["fields"]["additional_notes"], "Ελλιπές πρόχειρο")
        self.assertFalse(reopened.json()["attachments_saved"])
        newer = {"pathway":"A2", "identity_number":"",
                 "legacy_history":{"actual_procedure_date":"","vas_before":None,
                                   "vas_after":None,"last_followup_date":"","last_followup_vas":None},
                 "doctor_key":""}
        put = self.client.put(self.base + "/drafts/" + obj["id"], headers=self.headers,
                              json={"revision":1,"fields":newer})
        self.assertEqual(put.status_code, 200, put.text)
        self.assertEqual(put.json()["revision"], 2)
        self.assertEqual(self.client.put(self.base + "/drafts/" + obj["id"],
            headers=self.headers, json={"revision":1,"fields":{"pathway":"A1"}}).status_code, 409)
        self.assertEqual(self.client.get(self.base + "/drafts/" + obj["id"],
            headers=self.headers).json()["fields"]["pathway"], "A2")
        self.assertEqual(self.client.post(self.base + "/drafts", headers=self.headers,
            json={"fields":{"pathway":"A1","imaging_report":"%PDF-fake-data"}}).status_code, 422)
        self.assertEqual(self.client.post(self.base + "/drafts", headers=self.headers,
            json={"fields":{"pathway":"A1","additional_notes":"X"*50000}}).status_code, 422)
        restarted = FastAPI()
        restarted.include_router(build_rf_router(self.engine))
        with TestClient(restarted) as new_client:
            revived = new_client.get(self.base + "/drafts/" + obj["id"], headers=self.headers)
            self.assertEqual(revived.status_code, 200)
            self.assertEqual(revived.json()["fields"]["pathway"], "A2")
        with Session(self.engine) as session:
            self.assertEqual(session.query(RFApplicationORM).count(), 0)
            self.assertEqual(session.query(RFProcedureHistoryORM).count(), 0)
        self.assertEqual(self.client.request("DELETE", self.base + "/drafts/" + obj["id"],
            headers=self.headers, json={"revision":1}).status_code, 409)
        deleted = self.client.request("DELETE", self.base + "/drafts/" + obj["id"],
            headers=self.headers, json={"revision":2})
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(self.client.get(self.base + "/drafts/" + obj["id"],
            headers=self.headers).status_code, 404)

    def test_single_field_greek_name_identity_gesy_and_explicit_source(self):
        from clinic_utilities.rf.persistence import record_application, record_legacy_procedure
        orig = make_a1_draft(patient_name="Μαρία Παπαδοπούλου",
                             identity_number="SYN-001", gesy_number="CY-GESY-01").model_dump()
        original = record_application(self.engine, {**orig, "site_key":"knee",
            "application_date":"2026-08-01"})
        other = make_a1_draft(patient_name="Μαρία Παπαδοπούλου",
                             identity_number="SYN-002", gesy_number="CY-GESY-02").model_dump()
        record_application(self.engine, {**other, "site_key":"knee",
            "application_date":"2026-08-01"})
        historic = record_legacy_procedure(self.engine, {
            "identity_number":"SYN-001", "indication_code":"KNEE_OA_KL34",
            "site_key":"knee", "laterality":"left", "exact_location":"Αριστερό γόνατο",
            "actual_procedure_date":"2026-08-05", "vas_before":8, "vas_after":3,
            "last_followup_date":"2026-08-10", "last_followup_vas":4
        })
        for search in ("μαρια παπαδοπουλου", "SYN001", "CY GESY 01"):
            response = self.client.get(self.base + "/reuse/search",
                headers=self.headers, params={"query":search})
            self.assertEqual(response.status_code, 200, (search, response.text))
            self.assertTrue(any(r["id"] == original for r in response.json()["results"]))
        all_named = self.client.get(self.base + "/reuse/search",
             headers=self.headers, params={"query":"ΜΑΡΊΑ"}).json()["results"]
        self.assertEqual(sum(r["kind"] == "application" for r in all_named), 2)
        self.assertTrue(any(r["kind"] == "procedure" and r["id"] == historic for r in all_named))
        selected = self.client.get(self.base + "/reuse/application/" + original,
                                   headers=self.headers)
        self.assertEqual(selected.status_code, 200)
        self.assertEqual(selected.json()["fields"]["identity_number"], "SYN-001")
        self.assertNotIn("doctor_snapshot", selected.json()["fields"])
        self.assertFalse(selected.json()["source"]["actual_procedure"])
        procedure = self.client.get(self.base + "/reuse/procedure/" + historic, headers=self.headers)
        self.assertEqual(procedure.status_code, 200)
        self.assertTrue(procedure.json()["source"]["actual_procedure"])
        self.assertEqual(procedure.json()["actual_procedure"]["vas_after"], 3)
        self.assertEqual(self.client.get(self.base + "/reuse/search",
             params={"query":"Μαρία"}).status_code, 401)

    def test_reuse_cross_patient_rejection_and_a2_actual_procedure_and_draft_completion(self):
        from clinic_utilities.rf.persistence import record_application, record_legacy_procedure
        from sqlalchemy.orm import Session
        from clinic_utilities.rf.persistence import RFApplicationORM, RFSavedDraftORM
        base_draft = make_a1_draft(patient_name="Synthetic A", identity_number="PAT-A").model_dump()
        previous = record_application(self.engine, {**base_draft, "site_key":"knee"})
        real = record_legacy_procedure(self.engine, {
            "identity_number":"PAT-A","indication_code":"KNEE_OA_KL34",
            "site_key":"knee","laterality":"left","exact_location":"Αριστερό γόνατο",
            "actual_procedure_date":"2026-06-01","vas_before":9,"vas_after":3,
            "last_followup_date":"2026-06-10","last_followup_vas":5
        })
        draft = {"pathway":"A2","patient_name":"Synthetic A","identity_number":"PAT-A",
                 "gesy_number":"GESY-SYN", "age":67, "doctor_key":"papachristou",
                 "product_key":"medikey","indication_code":"KNEE_OA_KL34",
                 "laterality":"left","exact_location":"Αριστερό γόνατο",
                 "imaging_review_confirmed":True,"procedure_history_id":real,
                 "source_kind":"application","source_id":previous,
                 "a2_current_followup_date":"2026-08-09","a2_current_followup_vas":7}
        partial = self.client.post(self.base + "/drafts",headers=self.headers,
            json={"fields":{"pathway":"A2","identity_number":"PAT-A",
                           "procedure_history_id":real,"source_kind":"application","source_id":previous}})
        self.assertEqual(partial.status_code, 201, partial.text)
        item = partial.json()
        draft.update(draft_id=item["id"],draft_revision=item["revision"])
        def create(payload):
            with patch("clinic_utilities.rf.api.build_official_rf_pdf", return_value=b"%PDF-synthetic") as build:
                response = self.client.post(self.base + "/create",
                    headers=self.headers, data={"draft_json":json.dumps(payload)},
                    files={"imaging_report":("imaging.pdf",make_pdf(1),"application/pdf")})
                return response, build
        wrong, _ = create({**draft, "identity_number":"WRONG-PATIENT"})
        self.assertEqual(wrong.status_code, 422, wrong.text)
        with Session(self.engine) as session:
            self.assertEqual(session.query(RFSavedDraftORM).count(), 1)
            self.assertEqual(session.query(RFApplicationORM).count(), 1)
        success, build = create(draft)
        self.assertEqual(success.status_code, 200, success.text)
        self.assertEqual(build.call_args.kwargs["prior_history"]["last_followup_vas"], 7)
        with Session(self.engine) as session:
            self.assertEqual(session.query(RFSavedDraftORM).count(), 0)
            self.assertEqual(session.query(RFApplicationORM).count(), 2)
        repeat, _ = create(draft)
        self.assertEqual(repeat.status_code, 409, repeat.text)
        with Session(self.engine) as session:
            self.assertEqual(session.query(RFApplicationORM).count(), 2)


    def test_previous_procedure_never_prefills_product_from_other_shoulder_indication(self):
        from clinic_utilities.rf.persistence import record_application, record_legacy_procedure
        old_application = {**make_a1_draft(
            patient_name="Synthetic Patient A", identity_number="PAT-PRODUCT").model_dump(),
            "site_key":"shoulder","laterality":"left",
            "indication_code":"SHOULDER_OA_KL34", "product_key":"medikey",
            "exact_location":"Αριστερός ώμος"}
        record_application(self.engine, old_application)
        real = record_legacy_procedure(self.engine, {
            "identity_number":"PAT-PRODUCT","indication_code":"SHOULDER_IRREPARABLE_CUFF",
            "site_key":"shoulder","laterality":"left","exact_location":"Αριστερός ώμος",
            "actual_procedure_date":"2026-07-01","vas_before":8,"vas_after":3,
            "last_followup_date":"2026-08-01","last_followup_vas":5,
        })
        path = self.base + "/reuse/procedure/" + real
        no_match = self.client.get(path,headers=self.headers)
        self.assertEqual(no_match.status_code, 200, no_match.text)
        self.assertEqual(no_match.json()["fields"]["patient_name"], "Synthetic Patient A")
        self.assertEqual(no_match.json()["fields"]["product_key"], "",
                         "same body area and side do not imply the same RF product")
        compatible = {**old_application, "indication_code":"SHOULDER_IRREPARABLE_CUFF",
                      "product_key":"diros"}
        record_application(self.engine, compatible)
        valid = self.client.get(path, headers=self.headers)
        self.assertEqual(valid.status_code, 200, valid.text)
        self.assertEqual(valid.json()["fields"]["product_key"], "diros")

    def test_previous_rf_name_conflict_blocks_issue_until_explicit_name_resolution(self):
        from clinic_utilities.rf.persistence import record_application, record_legacy_procedure, RFApplicationORM
        from sqlalchemy.orm import Session
        original = {**make_a1_draft(
            patient_name="Patient A",identity_number="PAT-NAME").model_dump(),
            "site_key":"knee"}
        source_id = record_application(self.engine, original)
        procedure_id = record_legacy_procedure(self.engine, {
            "identity_number":"PAT-NAME","indication_code":"KNEE_OA_KL34",
            "site_key":"knee","laterality":"left","exact_location":"Αριστερό γόνατο",
            "actual_procedure_date":"2026-05-01","vas_before":8,"vas_after":2,
            "last_followup_date":"2026-06-01","last_followup_vas":4,
        })
        draft = {**original,"pathway":"A2","patient_name":"Patient B",
                 "procedure_history_id":procedure_id,
                 "source_kind":"application","source_id":source_id,
                 "imaging_review_confirmed":True}
        def submit(payload):
            with patch("clinic_utilities.rf.api.build_official_rf_pdf",
                       return_value=b"%PDF-synthetic") as builder:
                response = self.client.post(self.base + "/create",headers=self.headers,
                    data={"draft_json":json.dumps(payload)},
                    files={"imaging_report":("imaging.pdf",make_pdf(1),"application/pdf")})
                return response, builder
        denied, builder = submit(draft)
        self.assertEqual(denied.status_code, 422, denied.text)
        self.assertIn("ασυμφωνία ονόματος", denied.text)
        builder.assert_not_called()
        with Session(self.engine) as session:
            self.assertEqual(session.query(RFApplicationORM).count(),1)
        confirmed, builder = submit({**draft,"source_name_confirmed":True})
        self.assertEqual(confirmed.status_code,200,confirmed.text)
        builder.assert_called_once()
        with Session(self.engine) as session:
            rows=session.query(RFApplicationORM).all()
            self.assertEqual(len(rows),2)
            self.assertEqual(next(x for x in rows if x.patient_name=="Patient B").patient_identity_key,
                             "PATNAME")
        # An accent/case-only name variation is not a clinical identity conflict.
        accented = {**original,"pathway":"A2","patient_name":"PATIENT A",
                    "procedure_history_id":procedure_id,"source_kind":"application",
                    "source_id":source_id,"imaging_review_confirmed":True}
        case_only, _ = submit(accented)
        self.assertEqual(case_only.status_code,200,case_only.text)

    def test_same_site_side_but_different_indication_is_not_an_a2_continuation(self):
        from clinic_utilities.rf.persistence import record_application, record_legacy_procedure
        from sqlalchemy.orm import Session
        from clinic_utilities.rf.persistence import RFApplicationORM
        previous = {**make_a1_draft(identity_number="PAT-DIAG").model_dump(),
                    "indication_code":"SHOULDER_OA_KL34", "site_key":"shoulder",
                    "laterality":"left", "exact_location":"Αριστερός ώμος"}
        old_request = record_application(self.engine, previous)
        old_actual = record_legacy_procedure(self.engine, {
            "identity_number":"PAT-DIAG","indication_code":"SHOULDER_OA_KL34",
            "site_key":"shoulder","laterality":"left","exact_location":"Αριστερός ώμος",
            "actual_procedure_date":"2026-05-01","vas_before":8,"vas_after":3,
            "last_followup_date":"2026-06-01","last_followup_vas":4
        })
        request = {**previous, "pathway":"A2",
                   "indication_code":"SHOULDER_IRREPARABLE_CUFF",
                   "procedure_history_id":old_actual, "source_kind":"",
                   "source_id":"", "imaging_review_confirmed":True}
        def make(payload):
            with patch("clinic_utilities.rf.api.build_official_rf_pdf",
                       return_value=b"%PDF-synthetic"):
                return self.client.post(self.base + "/create", headers=self.headers,
                    data={"draft_json":json.dumps(payload)},
                    files={"imaging_report":("imaging.pdf",make_pdf(1),"application/pdf")})
        rejected = make(request)
        self.assertEqual(rejected.status_code, 422, rejected.text)
        self.assertIn("διαφορετική ένδειξη", rejected.text)
        # An application source also cannot be linked to a different diagnosis.
        actual_new = record_legacy_procedure(self.engine, {
            "identity_number":"PAT-DIAG", "indication_code":"SHOULDER_IRREPARABLE_CUFF",
            "site_key":"shoulder", "laterality":"left", "exact_location":"Αριστερός ώμος",
            "actual_procedure_date":"2026-08-01","vas_before":9,"vas_after":2,
            "last_followup_date":"2026-08-10","last_followup_vas":5
        })
        mismatched_source = make({**request, "procedure_history_id":actual_new,
                                  "source_kind":"application", "source_id":old_request})
        self.assertEqual(mismatched_source.status_code, 422, mismatched_source.text)
        self.assertIn("διαφορετική ένδειξη", mismatched_source.text)
        with Session(self.engine) as session:
            self.assertEqual(session.query(RFApplicationORM).count(), 1)

    def test_pdf_old_form_read_is_provisional_and_unrelated_pdf_not_accepted(self):
        from clinic_utilities.rf.catalog import DoctorProfile, ProductProfile
        from clinic_utilities.rf.pdf import build_official_rf_pdf
        doctor = DoctorProfile(name="Synthetic Doctor", gesy_code="SYN",
                 specialty="Synthetic",medical_center="Synthetic Centre",
                 phone="000",email="synthetic@example.invalid")
        product = ProductProfile(key="medikey",label="Medikey",code="SYN-M",
                 description="Synthetic",quantity="1")
        values = {**make_a1_draft().model_dump(), "application_date":"2026-09-11",
                  "site_key":"knee","rf_reason_text":"Synthetic indication"}
        pdf = build_official_rf_pdf(values,doctor=doctor,product=product,
                                     radiology_pdf_bytes=make_pdf(1))
        ok = self.client.post(self.base + "/reuse/pdf",headers=self.headers,
             files={"previous_rf_pdf":("old-rf.pdf",pdf,"application/pdf")})
        self.assertEqual(ok.status_code, 200, ok.text[:500])
        self.assertIn(ok.json()["status"],("recognized","partial"))
        self.assertEqual(ok.json()["fields"]["identity_number"], "SYN123")
        self.assertEqual(ok.json()["fields"]["indication_code"], "KNEE_OA_KL34")
        self.assertEqual(ok.json()["fields"]["laterality"], "left")
        self.assertFalse(ok.json()["actual_procedure_verified"])
        self.assertFalse(ok.json()["attachments_saved"])
        prior = {"actual_procedure_date":"2026-07-01","vas_before":8,
                 "vas_after":3,"last_followup_date":"2026-08-02","last_followup_vas":4}
        a2 = build_official_rf_pdf({**values, "pathway":"A2"},
                 doctor=doctor, product=product, radiology_pdf_bytes=make_pdf(1),
                 prior_history=prior)
        parsed_a2 = self.client.post(self.base + "/reuse/pdf",headers=self.headers,
            files={"previous_rf_pdf":("a2-rf.pdf",a2,"application/pdf")})
        self.assertEqual(parsed_a2.status_code, 200, parsed_a2.text[:300])
        self.assertIn(parsed_a2.json()["status"], ("partial", "recognized"))
        self.assertEqual(parsed_a2.json()["fields"]["identity_number"], "SYN123")
        self.assertIn("vas_before", parsed_a2.json()["fields"].get("previous_procedure_candidate", {}), str(parsed_a2.json()))
        self.assertEqual(parsed_a2.json()["fields"]["previous_procedure_candidate"]["vas_before"], 8)
        self.assertFalse(parsed_a2.json()["actual_procedure_verified"])
        wrong = self.client.post(self.base + "/reuse/pdf",headers=self.headers,
             files={"previous_rf_pdf":("not-rf.pdf",make_pdf(5),"application/pdf")})
        self.assertEqual(wrong.status_code, 200)
        self.assertEqual(wrong.json()["status"],"unrecognized")
        self.assertEqual(self.client.post(self.base + "/reuse/pdf",
             files={"previous_rf_pdf":("old-rf.pdf",pdf,"application/pdf")}).status_code,401)


class RFIntegrationSourceTests(unittest.TestCase):
    def test_main_mounts_native_router_not_gateway(self):
        source = Path("main.py").read_text(encoding="utf-8")
        self.assertIn("from clinic_utilities.rf.api import build_rf_router", source)
        self.assertIn("app.include_router(build_rf_router(engine))", source)
        self.assertNotIn("app.include_router(build_rf_gateway_router())", source)


if __name__ == "__main__":
    unittest.main()
