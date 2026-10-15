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
        self.assertIn("ρητή επίλυση", denied.text)
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


class RFOutcomeFollowupTests(unittest.TestCase):
    """Only synthetic clinical and doctor records. No external GeSY/provider calls."""

    def setUp(self):
        from datetime import date
        self.today = date.today().isoformat()
        self.engine = memory_engine()
        self.env = patch.dict(os.environ, {
            "CLINICAL_DATA_KEY": CLINICAL_KEY,
            "RF_DOCTOR_PROFILE_JSON": DOCTOR_JSON,
            "RF_PRODUCT_CATALOG_JSON": PRODUCT_JSON,
        })
        self.env.start()
        app = FastAPI()
        app.include_router(build_rf_router(self.engine))
        self.client = TestClient(app)
        self.base = "/clinical/clinic-utilities/rf/api"
        self.headers = {"X-Clinical-Key": CLINICAL_KEY}

    def tearDown(self):
        self.client.close()
        self.env.stop()

    def issue(self, **changes):
        draft = make_a1_draft().model_dump()
        draft.update(changes)
        with patch("clinic_utilities.rf.api.build_official_rf_pdf",
                   return_value=b"%PDF-SYNTHETIC"):
            response = self.client.post(
                self.base + "/create", headers=self.headers,
                data={"draft_json": json.dumps(draft)},
                files={"imaging_report": ("imaging.pdf", make_text_pdf("MRI synthetic"), "application/pdf")},
            )
        return response

    def event(self, app_id, action, revision=0, **changes):
        return self.client.post(
            f"{self.base}/outcomes/{app_id}/events", headers=self.headers,
            json={"action": action, "revision": revision,
                  "event_date": self.today if action not in {"claim", "release"} else "",
                  **changes},
        )

    def card(self, app_id):
        return self.client.get(f"{self.base}/outcomes/{app_id}", headers=self.headers).json()

    def search(self, **kwargs):
        return self.client.post(self.base + "/outcomes/search",
                                headers=self.headers,
                                json={"scope": "all", **kwargs})

    def test_issue_is_not_submitted_and_self_verified_submitted_is_mine(self):
        from clinic_utilities.rf.persistence import (
            RFApplicationORM, RFApplicationProgressAuditORM, RFProcedureHistoryORM)
        from sqlalchemy.orm import Session
        issued = self.issue(identity_number="SYN-OWN-1")
        self.assertEqual(issued.status_code, 200, issued.text)
        app_id = issued.headers["x-rf-application-id"]
        initial = self.card(app_id)
        self.assertEqual(initial["stage"], "issued")
        self.assertFalse(initial["mine_to_follow"])
        self.assertEqual(initial["revision"], 0)
        self.assertTrue(initial["issuer_self_verified_at_issue"])
        self.assertEqual(self.search().json()["counts"]["pending"], 0)
        with Session(self.engine) as session:
            payload = session.get(RFApplicationORM, app_id).payload_json
            self.assertEqual(payload["issuer_selection_origin"], "explicit")
            self.assertTrue(payload["issuer_self_verified_at_issue"])
            self.assertEqual(payload["doctor_snapshot"]["gesy_code"], "SYN-DOCTOR")
        self.assertEqual(self.event(app_id, "submit").status_code, 200)
        submitted = self.card(app_id)
        self.assertEqual(submitted["stage"], "submitted")
        self.assertTrue(submitted["mine_to_follow"])
        self.assertEqual(self.search().json()["counts"]["pending"], 1)
        self.assertEqual(self.search(scope="mine").json()["counts"]["mine_actionable"], 1)
        self.assertEqual(self.event(app_id, "submit", 1).status_code, 422)
        self.assertEqual(self.event(app_id, "reject", 0, rejection_reason="wrong revision").status_code, 409)
        approved = self.event(app_id, "approve", 1)
        self.assertEqual(approved.status_code, 200, approved.text)
        self.assertEqual(approved.json()["decision"], "approved")
        self.assertEqual(self.search().json()["counts"]["approved"], 1)
        scheduled = self.event(app_id, "schedule", 2)
        self.assertEqual(scheduled.status_code, 200, scheduled.text)
        self.assertEqual(scheduled.json()["stage"], "scheduled")
        self.assertEqual(self.event(app_id, "perform", 3).status_code, 422)
        performed = self.event(app_id, "perform", 3, confirmed_performed=True)
        self.assertEqual(performed.status_code, 200, performed.text)
        self.assertEqual(performed.json()["stage"], "performed")
        self.assertEqual(self.search().json()["counts"]["mine_actionable"], 0)
        with Session(self.engine) as session:
            self.assertEqual(session.query(RFApplicationProgressAuditORM).count(), 4)
            self.assertEqual(session.query(RFProcedureHistoryORM).count(), 0)
            # State reporting never changes the immutable issued document.
            self.assertIsNone(session.get(RFApplicationORM, app_id).payload_json.get("approval_status"))

    def test_tracking_exceeds_legacy_2000_and_10000_limits(self):
        from datetime import datetime, timedelta
        from sqlalchemy.orm import Session
        from clinic_utilities.rf.persistence import RFApplicationORM, RFApplicationProgressORM
        base_date = datetime(2026, 1, 1)
        with Session(self.engine) as session:
            session.bulk_save_objects([
                RFApplicationORM(
                    id=f"synthetic-bulk-{i:05d}",
                    patient_identity_key=f"SYNBULK{i:05d}",
                    patient_name="Rare Oldest" if i == 0 else "Synthetic Bulk",
                    gesy_number=f"SYNGESY{i:05d}", patient_age=60,
                    pathway="A1", indication_code="KNEE_OA_KL34",
                    site_key="knee", laterality="left",
                    exact_location="Synthetic", product_key="medikey",
                    payload_json={"identity_number": f"SYNBULK{i:05d}"},
                    created_at=base_date + timedelta(seconds=i),
                ) for i in range(10050)
            ])
            session.bulk_save_objects([
                RFApplicationProgressORM(
                    application_id=f"synthetic-bulk-{i:05d}",
                    revision=1, stage="submitted", submitted_date="2026-01-02",
                    updated_at=base_date,
                ) for i in range(8000,10050)
            ])
            session.commit()
        found = 0
        token = ""
        for _ in range(160):
            r = self.search(mode="tracking", stage="submitted", cursor=token)
            self.assertEqual(r.status_code, 200, r.text)
            data = r.json()
            found += len(data["applications"])
            if data["scan_complete"]:
                break
            self.assertTrue(data["has_more"])
            self.assertTrue(data["next_cursor"])
            token = data["next_cursor"]
        else:
            self.fail("Tracked >2000 matching RF did not exhaust bounded cursor")
        self.assertEqual(found, 2050)
        token = ""
        seen_oldest = []
        empty_batches = 0
        for _ in range(40):
            r = self.search(mode="tracking", query="Rare Oldest", cursor=token)
            self.assertEqual(r.status_code, 200, r.text)
            data = r.json()
            seen_oldest.extend(data["applications"])
            if not data["applications"] and data["has_more"]:
                empty_batches += 1
            if data["scan_complete"]:
                break
            token = data["next_cursor"]
        else:
            self.fail("Tracking cannot scan >10000 older RF attempts")
        self.assertGreater(empty_batches, 0)
        self.assertEqual([x["id"] for x in seen_oldest], ["synthetic-bulk-00000"])

    def test_tracking_cursor_and_rf_date_only_week(self):
        ids = []
        for index in range(26):
            created = self.issue(identity_number=f"SYN-CURSOR-{index:03d}",
                                 patient_name=f"Synthetic cursor {index}")
            self.assertEqual(created.status_code, 200, created.text)
            ids.append(created.headers["x-rf-application-id"])
        first = self.search(mode="tracking").json()
        self.assertEqual(len(first["applications"]), 20)
        self.assertTrue(first["has_more"])
        self.assertFalse(first["scan_complete"])
        self.assertIsNone(first["counts"])
        self.assertNotIn("SYN-CURSOR-000", {c["identity_number"] for c in first["applications"]})
        following = self.search(mode="tracking", cursor=first["next_cursor"]).json()
        self.assertEqual(len(following["applications"]), 6)
        self.assertTrue(following["scan_complete"])
        self.assertFalse(following["has_more"])
        combined = [a["id"] for a in first["applications"] + following["applications"]]
        self.assertEqual(len(set(combined)), 26)
        self.assertEqual(self.search(mode="tracking", cursor=first["next_cursor"],
                                     query="different filter").status_code, 422)
        self.assertEqual(self.search(mode="tracking", cursor=first["next_cursor"] + "X").status_code, 422)
        self.assertEqual(self.search(mode="tracking", query="NO-MATCH").json()["applications"], [])
        from datetime import date, timedelta
        scheduled = date.today() + timedelta(days=1)
        monday = (scheduled - timedelta(days=scheduled.weekday())).isoformat()
        scheduled_text = scheduled.isoformat()
        self.assertEqual(self.event(ids[0], "approve").status_code, 200)
        planning = self.event(ids[0], "schedule", 1, event_date=scheduled_text)
        self.assertEqual(planning.status_code, 200, planning.text)
        week = self.client.post(self.base + "/outcomes/calendar", headers=self.headers,
                                json={"week_start": monday})
        self.assertEqual(week.status_code, 200, week.text)
        self.assertTrue(week.json()["complete"])
        self.assertFalse(week.json()["time_verified"])
        self.assertTrue(any(f["application_id"] == ids[0] and f["kind"] == "scheduled"
                            and f["date"] == scheduled_text for f in week.json()["facts"]))
        invalid_week = self.client.post(self.base + "/outcomes/calendar",
            headers=self.headers, json={"week_start": (date.fromisoformat(monday) + timedelta(days=1)).isoformat()})
        self.assertEqual(invalid_week.status_code, 422)

    def test_atomic_rejected_correction_rejects_stale_revision_and_missing_proof(self):
        from clinic_utilities.rf.persistence import RFApplicationORM
        from sqlalchemy.orm import Session
        original = self.issue(identity_number="SYN-ATOMIC-TEST")
        self.assertEqual(original.status_code, 200)
        app_id = original.headers["x-rf-application-id"]
        endpoint = self.base + "/reuse/rejected-correction"
        body = {"application_id": app_id, "expected_revision": 1}
        self.assertEqual(self.client.post(endpoint, json=body).status_code, 401)
        self.assertEqual(self.client.post(endpoint, headers=self.headers, json=body).status_code, 409)
        self.assertEqual(self.client.post(endpoint, headers=self.headers, json={
            "application_id": "no-such-id", "expected_revision": 1}).status_code, 404)
        self.assertEqual(self.client.post(endpoint, headers=self.headers, json={
            "application_id": app_id}).status_code, 422)
        rejection = self.event(app_id, "reject", rejection_reason="synthetic").json()
        rev = rejection["revision"]
        guarded = self.client.post(endpoint, headers=self.headers, json={
            "application_id": app_id, "expected_revision": rev})
        self.assertEqual(guarded.status_code, 200, guarded.text)
        self.assertEqual(guarded.json()["application_id"], app_id)
        self.assertEqual(guarded.json()["revision"], rev)
        self.assertEqual(guarded.json()["fields"]["identity_number"], "SYN-ATOMIC-TEST")
        self.assertEqual(guarded.headers["cache-control"], "no-store")
        self.assertEqual(self.client.post(endpoint, headers=self.headers, json={
            "application_id": app_id, "expected_revision": rev + 1}).status_code, 409)
        self.assertEqual(self.event(app_id, "correct_decision", rev,
            corrected_decision="rejected", rejection_reason="updated").status_code, 200)
        self.assertEqual(self.client.post(endpoint, headers=self.headers, json={
            "application_id": app_id, "expected_revision": rev}).status_code, 409)
        self.assertEqual(self.issue(identity_number="SYN-ATOMIC-TEST",
            correction_of_application_id=app_id,
            correction_source_expected_revision=rev).status_code, 409)
        self.assertEqual(self.issue(identity_number="SYN-ATOMIC-TEST",
            correction_of_application_id=app_id).status_code, 422)
        with Session(self.engine) as session:
            self.assertEqual(session.query(RFApplicationORM).count(), 1)

    def test_colleague_reported_rejection_without_submission_and_linked_correction(self):
        from clinic_utilities.rf.persistence import RFApplicationORM, RFProcedureHistoryORM
        from sqlalchemy.orm import Session
        colleague = {
            "name": "Synthetic Other Clinician", "gesy_code": "SYN-OTHER",
            "specialty": "Synthetic", "medical_center": "Synthetic Facility",
            "phone": "00000002", "email": "colleague@example.invalid",
        }
        created = self.client.post(self.base + "/doctors", headers=self.headers, json=colleague)
        self.assertEqual(created.status_code, 201, created.text)
        key = created.json()["key"]
        original = self.issue(identity_number="SYN-COL-1", doctor_key=key)
        self.assertEqual(original.status_code, 200, original.text)
        app_id = original.headers["x-rf-application-id"]
        self.assertFalse(self.card(app_id)["issuer_self_verified_at_issue"])
        rejected = self.event(app_id, "reject", rejection_reason="Συνθετικός λόγος απόρριψης",
                              reported_source="colleague")
        self.assertEqual(rejected.status_code, 200, rejected.text)
        self.assertIsNone(rejected.json()["submitted_date"])
        self.assertFalse(rejected.json()["mine_to_follow"])
        self.assertEqual(rejected.json()["decision"], "rejected")
        self.assertEqual(self.search().json()["counts"]["pending"], 0)
        self.assertEqual(self.event(app_id, "submit", 1).status_code, 422)
        # Correction is a distinct A1 attempt, never an automatic A2/actual treatment.
        new = self.issue(identity_number="SYN-COL-1", doctor_key=key,
                         correction_of_application_id=app_id,
                          correction_source_expected_revision=rejected.json()['revision'])
        self.assertEqual(new.status_code, 200, new.text)
        new_id = new.headers["x-rf-application-id"]
        self.assertNotEqual(app_id, new_id)
        self.assertEqual(self.card(new_id)["stage"], "issued")
        self.assertFalse(self.card(new_id)["mine_to_follow"])
        self.assertEqual(self.card(new_id)["correction_of_application_id"], app_id)
        self.assertEqual(self.card(app_id)["rejection_reason"], "Συνθετικός λόγος απόρριψης")
        self.assertEqual(self.event(new_id, "submit").status_code, 200)
        self.assertFalse(self.card(new_id)["mine_to_follow"])
        self.assertEqual(self.event(new_id, "claim", 1).status_code, 200)
        self.assertTrue(self.card(new_id)["mine_to_follow"])
        self.assertEqual(self.event(new_id, "release", 2).status_code, 200)
        self.assertFalse(self.card(new_id)["mine_to_follow"])
        with Session(self.engine) as session:
            self.assertEqual(session.get(RFApplicationORM, new_id).pathway, "A1")
            self.assertEqual(session.query(RFProcedureHistoryORM).count(), 0)

    def test_legacy_omitted_default_and_repurposed_managed_default_fail_closed(self):
        from clinic_utilities.rf.persistence import RFApplicationORM
        from sqlalchemy.orm import Session
        raw = make_a1_draft(identity_number="LEGACY-OWN").model_dump()
        raw.pop("doctor_key")
        with patch("clinic_utilities.rf.api.build_official_rf_pdf", return_value=b"%PDF-SYN"):
            old = self.client.post(self.base + "/create", headers=self.headers,
                data={"draft_json": json.dumps(raw)},
                files={"imaging_report": ("img.pdf", make_text_pdf("MRI synthetic"), "application/pdf")})
        self.assertEqual(old.status_code, 200, old.text)
        oid = old.headers["x-rf-application-id"]
        with Session(self.engine) as session:
            payload = session.get(RFApplicationORM, oid).payload_json
            self.assertEqual(payload["doctor_key"], "papachristou")
            self.assertEqual(payload["issuer_selection_origin"], "legacy_default")
            self.assertFalse(payload["issuer_self_verified_at_issue"])
        self.assertEqual(self.event(oid, "submit").status_code, 200)
        self.assertFalse(self.card(oid)["mine_to_follow"])
        directory = self.client.get(self.base + "/doctors", headers=self.headers).json()["doctors"]
        default = next(x for x in directory if x["key"] == "papachristou")
        changed = {k: default[k] for k in ("name", "gesy_code", "specialty",
                                           "medical_center", "phone", "email")}
        changed.update({"name": "Synthetic Repurposed Doctor", "gesy_code": "OTHER-GESY"})
        response = self.client.put(self.base + "/doctors/papachristou",
            headers=self.headers, json={**changed, "revision": default["revision"]})
        self.assertEqual(response.status_code, 200, response.text)
        another = self.issue(identity_number="DEFAULT-REASSIGNED", doctor_key="papachristou")
        self.assertEqual(another.status_code, 200, another.text)
        other_id = another.headers["x-rf-application-id"]
        self.assertFalse(self.card(other_id)["issuer_self_verified_at_issue"])
        self.assertEqual(self.event(other_id, "submit").status_code, 200)
        self.assertFalse(self.card(other_id)["mine_to_follow"])

    def test_historical_application_without_attestation_and_invalid_correction_do_not_write(self):
        from clinic_utilities.rf.persistence import RFApplicationORM, RFApplicationProgressAuditORM
        from sqlalchemy.orm import Session
        old = self.issue(identity_number="SYN-HISTORY")
        self.assertEqual(old.status_code, 200, old.text)
        old_id = old.headers["x-rf-application-id"]
        with Session(self.engine) as session:
            row = session.get(RFApplicationORM, old_id)
            payload = dict(row.payload_json)
            payload.pop("issuer_selection_origin")
            payload.pop("issuer_self_verified_at_issue")
            row.payload_json = payload
            session.commit()
        self.assertEqual(self.event(old_id, "submit").status_code, 200)
        self.assertFalse(self.card(old_id)["mine_to_follow"])
        rejected_not_yet = self.issue(identity_number="SYN-HISTORY",
                                      correction_of_application_id=old_id)
        self.assertEqual(rejected_not_yet.status_code, 422)
        self.assertEqual(self.event(old_id, "reject", 1,
                                    rejection_reason="Synthetic rejection").status_code, 200)
        wrong_patient = self.issue(identity_number="SYN-WRONG",
                                   correction_of_application_id=old_id)
        self.assertEqual(wrong_patient.status_code, 422)
        valid = self.issue(identity_number="SYN-HISTORY",
                           correction_of_application_id=old_id,
                            correction_source_expected_revision=self.card(old_id)["revision"])
        self.assertEqual(valid.status_code, 200, valid.text)
        with Session(self.engine) as session:
            self.assertEqual(session.query(RFApplicationORM).count(), 2)
            self.assertEqual(session.query(RFApplicationProgressAuditORM).count(), 2)

    def test_explicit_followup_override_before_submitted_is_not_overwritten(self):
        owned = self.issue(identity_number="SYN-OVERRIDE")
        self.assertEqual(owned.status_code, 200, owned.text)
        app_id = owned.headers["x-rf-application-id"]
        self.assertEqual(self.event(app_id, "claim", 0).status_code, 200)
        self.assertTrue(self.card(app_id)["mine_to_follow"])
        self.assertEqual(self.event(app_id, "release", 1).status_code, 200)
        self.assertFalse(self.card(app_id)["mine_to_follow"])
        submitted = self.event(app_id, "submit", 2)
        self.assertEqual(submitted.status_code, 200, submitted.text)
        self.assertFalse(submitted.json()["mine_to_follow"],
                         "an explicit release must override self-issuer auto-claim")
        self.assertEqual(self.event(app_id, "claim", 3).status_code, 200)
        self.assertTrue(self.card(app_id)["mine_to_follow"])
        # An unauthorized actor cannot claim or change a clinical outcome.
        self.assertEqual(self.client.post(
            f"{self.base}/outcomes/{app_id}/events",
            json={"action": "release", "revision": 4},
        ).status_code, 401)

    def test_protected_search_unknown_and_date_status_guards(self):
        self.assertEqual(self.client.post(self.base + "/outcomes/search", json={}).status_code, 401)
        a = self.issue(identity_number="SYN-GUARDS")
        self.assertEqual(a.status_code, 200, a.text)
        app_id = a.headers["x-rf-application-id"]
        self.assertEqual(self.search(query="SYN-GUARDS").json()["denominator"], 1)
        self.assertEqual(self.search(scope="mine").json()["denominator"], 0)
        self.assertEqual(self.event(app_id, "schedule").status_code, 422)
        self.assertEqual(self.event(app_id, "perform", confirmed_performed=True).status_code, 422)
        self.assertEqual(self.event(app_id, "reject", event_date="9999-01-01",
                                    rejection_reason="future").status_code, 422)
        self.assertEqual(self.event(app_id, "reject",
                                    rejection_reason="synthetic").status_code, 200)
        self.assertEqual(self.event(app_id, "correct_decision", 1,
                                    rejection_reason="corrected synthetic reason").status_code, 200)
        self.assertEqual(self.card(app_id)["rejection_reason"], "corrected synthetic reason")
        self.assertEqual(self.event(app_id, "approve", 2).status_code, 422)
        corrected = self.event(app_id, "correct_decision", 2,
                               corrected_decision="approved")
        self.assertEqual(corrected.status_code, 200, corrected.text)
        self.assertEqual(corrected.json()["decision"], "approved")
        self.assertEqual(corrected.json()["stage"], "approved")
        self.assertEqual(corrected.json()["rejection_reason"], "")
        self.assertEqual(self.search().json()["counts"]["rejected"], 0)
        self.assertEqual(self.search().json()["counts"]["approved"], 1)


    def test_submitted_date_correction_keeps_manual_release_and_atomic_revision_audit(self):
        from datetime import datetime, timedelta
        from clinic_utilities.rf.persistence import (
            RFApplicationORM, RFApplicationProgressAuditORM)
        from sqlalchemy.orm import Session
        created = self.issue(identity_number="SYN-SUBMITTED-CORRECT")
        self.assertEqual(created.status_code, 200, created.text)
        app_id = created.headers["x-rf-application-id"]
        before = (datetime.utcnow() - timedelta(days=10))
        submitted_date = (before + timedelta(days=2)).date().isoformat()
        correct_date = (before + timedelta(days=3)).date().isoformat()
        with Session(self.engine) as session:
            session.get(RFApplicationORM, app_id).created_at = before
            session.commit()
        submitted = self.event(app_id, "submit", event_date=submitted_date)
        self.assertEqual(submitted.status_code, 200, submitted.text)
        self.assertTrue(submitted.json()["mine_to_follow"])
        self.assertEqual(self.event(app_id, "release", 1).status_code, 200)
        correction = self.event(app_id, "correct_submission", 2, event_date=correct_date,
                                reported_source="colleague")
        self.assertEqual(correction.status_code, 200, correction.text)
        self.assertEqual(correction.json()["revision"], 3)
        self.assertEqual(correction.json()["stage"], "submitted")
        self.assertEqual(correction.json()["submitted_date"], correct_date)
        self.assertFalse(correction.json()["mine_to_follow"],
                         "explicit personal release survives the correction")
        self.assertEqual(self.event(app_id, "correct_submission", 2,
                                    event_date=submitted_date).status_code, 409)
        with Session(self.engine) as session:
            audits = session.query(RFApplicationProgressAuditORM).filter_by(
                application_id=app_id).order_by(RFApplicationProgressAuditORM.revision).all()
            self.assertEqual(len(audits), 3)
            last = audits[-1]
            self.assertEqual(last.action, "correct_submission")
            self.assertEqual(last.reported_source, "colleague")
            self.assertEqual(last.revision, 3)
            self.assertEqual(last.before_json["submitted_date"], submitted_date)
            self.assertEqual(last.after_json["submitted_date"], correct_date)
            self.assertEqual(last.before_json["followup_choice"], "release")
            self.assertEqual(last.after_json["followup_choice"], "release")

    def test_issue_chronology_and_downstream_decision_correction(self):
        from datetime import datetime, timedelta, date
        from clinic_utilities.rf.persistence import (
            RFApplicationORM, RFApplicationProgressAuditORM)
        from sqlalchemy.orm import Session
        created = self.issue(identity_number="SYN-CHRONOLOGY")
        self.assertEqual(created.status_code, 200, created.text)
        app_id = created.headers["x-rf-application-id"]
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        for action in ("submit", "approve", "reject"):
            response = self.event(app_id, action, event_date=yesterday)
            self.assertEqual(response.status_code, 422, (action, response.text))
        self.assertEqual(self.card(app_id)["revision"], 0,
                         "events preceding PDF issuance must not write")
        issued_at = datetime.utcnow() - timedelta(days=12)
        day = lambda offset: (issued_at + timedelta(days=offset)).date().isoformat()
        with Session(self.engine) as session:
            session.get(RFApplicationORM, app_id).created_at = issued_at
            session.commit()
        self.assertEqual(self.event(app_id, "submit", event_date=day(2)).status_code, 200)
        self.assertEqual(self.event(app_id, "approve", 1, event_date=day(4)).status_code, 200)
        self.assertEqual(self.event(app_id, "schedule", 2, event_date=day(7)).status_code, 200)
        while_scheduled = self.event(app_id, "correct_decision", 3,
                                     event_date=day(6), reported_source="colleague")
        self.assertEqual(while_scheduled.status_code, 200, while_scheduled.text)
        self.assertEqual(while_scheduled.json()["stage"], "scheduled")
        self.assertEqual(self.event(app_id, "correct_decision", 4,
                                    event_date=day(6),
                                    corrected_decision="rejected").status_code, 422)
        self.assertEqual(self.event(app_id, "perform", 4, event_date=day(6),
                                    confirmed_performed=True).status_code, 422)
        self.assertEqual(self.event(app_id, "perform", 4, event_date=day(9),
                                    confirmed_performed=True).status_code, 200)
        self.assertEqual(self.event(app_id, "correct_submission", 5,
                                    event_date=day(7)).status_code, 422)
        self.assertEqual(self.event(app_id, "correct_decision", 5,
                                    event_date=day(8)).status_code, 422)
        self.assertEqual(self.event(app_id, "correct_decision", 5,
                                    event_date=day(3),
                                    corrected_decision="rejected").status_code, 422)
        corrected = self.event(app_id, "correct_decision", 5,
                               event_date=day(5), reported_source="gesy_notice")
        self.assertEqual(corrected.status_code, 200, corrected.text)
        self.assertEqual(corrected.json()["stage"], "performed",
                         "decision date correction cannot downgrade performed")
        self.assertEqual(corrected.json()["decision"], "approved")
        self.assertEqual(corrected.json()["performed_date"], day(9))
        self.assertEqual(corrected.json()["scheduled_date"], day(7))
        self.assertEqual(corrected.json()["decision_date"], day(5))
        self.assertEqual(corrected.json()["last_decision_report"]["reported_source"], "gesy_notice")
        with Session(self.engine) as session:
            self.assertEqual(session.query(RFApplicationProgressAuditORM).filter_by(
                application_id=app_id).count(), 6,
                "rejected contradictory corrections must have no audit writes")

    def test_linked_correction_blocks_reversal_and_post_link_rejection_date(self):
        from datetime import datetime, timedelta
        from clinic_utilities.rf.persistence import RFApplicationORM
        from sqlalchemy.orm import Session
        created = self.issue(identity_number="SYN-LINK-SAFETY")
        self.assertEqual(created.status_code, 200, created.text)
        app_id = created.headers["x-rf-application-id"]
        issued_at = datetime.utcnow() - timedelta(days=10)
        day = lambda offset: (issued_at + timedelta(days=offset)).date().isoformat()
        with Session(self.engine) as session:
            session.get(RFApplicationORM, app_id).created_at = issued_at
            session.commit()
        self.assertEqual(self.event(app_id, "reject", event_date=day(3)).status_code, 200)
        followup = self.issue(identity_number="SYN-LINK-SAFETY",
                              correction_of_application_id=app_id,
                               correction_source_expected_revision=self.card(app_id)["revision"])
        self.assertEqual(followup.status_code, 200, followup.text)
        linked_id = followup.headers["x-rf-application-id"]
        with Session(self.engine) as session:
            session.get(RFApplicationORM, linked_id).created_at = issued_at + timedelta(days=6)
            session.commit()
        self.assertEqual(self.event(app_id, "correct_decision", 1,
                                    corrected_decision="approved",
                                    event_date=day(4)).status_code, 422)
        self.assertEqual(self.event(app_id, "correct_decision", 1,
                                    corrected_decision="rejected", rejection_reason="late",
                                    event_date=day(7)).status_code, 422)
        safe = self.event(app_id, "correct_decision", 1,
                          corrected_decision="rejected", rejection_reason="corrected",
                          event_date=day(4))
        self.assertEqual(safe.status_code, 200, safe.text)
        self.assertEqual(safe.json()["decision"], "rejected")
        self.assertEqual(safe.json()["rejection_reason"], "corrected")
        self.assertEqual(self.card(linked_id)["correction_of_application_id"], app_id)

    def test_decision_provenance_read_from_existing_audit_or_unknown(self):
        from clinic_utilities.rf.persistence import RFApplicationProgressAuditORM
        from sqlalchemy.orm import Session
        created = self.issue(identity_number="SYN-SOURCE-VIEW")
        self.assertEqual(created.status_code, 200, created.text)
        app_id = created.headers["x-rf-application-id"]
        self.assertIsNone(self.card(app_id)["last_decision_report"])
        reported = self.event(app_id, "reject", reported_source="colleague",
                              rejection_reason="<img src=x onerror=alert(1)>")
        self.assertEqual(reported.status_code, 200, reported.text)
        provenance = reported.json()["last_decision_report"]
        self.assertEqual(provenance["reported_source"], "colleague")
        self.assertEqual(provenance["event_date"], self.today)
        self.assertTrue(provenance["recorded_at"].endswith("Z"))
        self.assertEqual(self.card(app_id)["last_decision_report"], provenance)
        corrected = self.event(app_id, "correct_decision", 1,
                               reported_source="gesy_notice",
                               rejection_reason="διορθωμένη αιτία")
        self.assertEqual(corrected.status_code, 200, corrected.text)
        self.assertEqual(corrected.json()["last_decision_report"]["reported_source"], "gesy_notice")
        with Session(self.engine) as session:
            session.query(RFApplicationProgressAuditORM).filter_by(
                application_id=app_id).delete()
            session.commit()
        historical = self.card(app_id)
        self.assertEqual(historical["decision"], "rejected")
        self.assertIsNone(historical["last_decision_report"],
                          "a missing historical audit must not infer provenance")


    def test_issued_copy_restores_saved_fields_without_attachments_or_mutation(self):
        from clinic_utilities.rf.persistence import RFApplicationORM
        from sqlalchemy.orm import Session
        original = self.issue(identity_number="SYN-COPY-001", patient_name="Synthetic Copy",
            nsaid_trials=[{"drug_name":"Synthetic Nsaid","dose":"90 mg",
                           "duration":"3 months","source_text":"unretained raw text"}],
            physio_dates_text="01/02/2026\n08/02/2026")
        self.assertEqual(original.status_code, 200, original.text)
        app_id = original.headers["x-rf-application-id"]
        payload = {"application_id":app_id,"expected_revision":0}
        route = self.base + "/reuse/issued-copy"
        self.assertEqual(self.client.post(route,json=payload).status_code, 401)
        response = self.client.post(route,headers=self.headers,json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["application_id"],app_id)
        self.assertFalse(body["files_saved"])
        self.assertEqual(response.headers["cache-control"],"no-store")
        self.assertEqual(body["fields"]["identity_number"],"SYN-COPY-001")
        self.assertEqual(body["fields"]["pathway"],"A1")
        self.assertEqual(body["fields"]["physio_dates_text"],"2026-02-01\n2026-02-08")
        self.assertEqual(body["fields"]["nsaid_trials"][0]["drug_name"],"Synthetic Nsaid")
        self.assertNotIn("source_text",body["fields"]["nsaid_trials"][0])
        self.assertNotIn("full_medication_text",body["fields"])
        self.assertNotIn("correction_source_expected_revision",body["fields"])
        with Session(self.engine) as session:
            self.assertEqual(session.query(RFApplicationORM).count(),1)
        self.assertEqual(self.event(app_id,"reject",rejection_reason="synthetic").status_code,200)
        self.assertEqual(self.client.post(route,headers=self.headers,json=payload).status_code,409)
        self.assertEqual(self.client.post(route,headers=self.headers,json={
            "application_id":"missing","expected_revision":0}).status_code,404)

    def test_exact_application_delete_requires_code_revision_and_clinical_key(self):
        from clinic_utilities.rf.persistence import (
            RFApplicationORM, RFApplicationProgressORM,
            RFApplicationProgressAuditORM, RFApplicationDeletionAuditORM)
        from sqlalchemy.orm import Session
        first = self.issue(identity_number="SYN-DELETE",patient_name="Synthetic Repeat")
        second = self.issue(identity_number="SYN-DELETE",patient_name="Synthetic Repeat")
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        first_id,second_id = (first.headers["x-rf-application-id"],
                              second.headers["x-rf-application-id"])
        route = self.base + "/outcomes/" + first_id + "/delete"
        body = {"expected_revision":0,"confirmation_code":first_id[-8:].upper()}
        self.assertEqual(self.client.post(route,json=body).status_code,401)
        self.assertEqual(self.client.post(route,headers=self.headers,json={
            **body,"confirmation_code":"WRONGID!"}).status_code,422)
        self.assertEqual(self.event(first_id,"submit").status_code,200)
        self.assertEqual(self.client.post(route,headers=self.headers,json=body).status_code,409)
        body["expected_revision"]=1
        deleted=self.client.post(route,headers=self.headers,json=body)
        self.assertEqual(deleted.status_code,200,deleted.text)
        self.assertEqual(deleted.json(),{"deleted":True})
        self.assertEqual(deleted.headers["cache-control"],"no-store")
        self.assertEqual(self.client.get(self.base + "/outcomes/" + first_id,
                                         headers=self.headers).status_code,404)
        self.assertEqual(self.card(second_id)["identity_number"],"SYN-DELETE")
        self.assertEqual(self.search(mode="tracking").json()["applications"][0]["id"],second_id)
        with Session(self.engine) as session:
            self.assertEqual(session.query(RFApplicationORM).count(),1)
            self.assertIsNone(session.get(RFApplicationProgressORM,first_id))
            self.assertEqual(session.query(RFApplicationProgressAuditORM).count(),0)
            receipt=session.query(RFApplicationDeletionAuditORM).one()
            self.assertEqual(receipt.application_id,first_id)
            self.assertEqual(receipt.prior_revision,1)
        self.assertEqual(self.client.post(route,headers=self.headers,json=body).status_code,404)

    def test_delete_rejected_parent_with_linked_correction_is_blocked(self):
        parent = self.issue(identity_number="SYN-LINK-DELETE")
        self.assertEqual(parent.status_code,200)
        parent_id = parent.headers["x-rf-application-id"]
        rejection=self.event(parent_id,"reject",rejection_reason="synthetic")
        self.assertEqual(rejection.status_code,200)
        child = self.issue(identity_number="SYN-LINK-DELETE",
                           correction_of_application_id=parent_id,
                           correction_source_expected_revision=rejection.json()["revision"])
        self.assertEqual(child.status_code,200,child.text)
        child_id=child.headers["x-rf-application-id"]
        blocked=self.client.post(
            self.base + "/outcomes/" + parent_id + "/delete",headers=self.headers,
            json={"expected_revision":1,"confirmation_code":parent_id[-8:].upper()})
        self.assertEqual(blocked.status_code,409)
        self.assertIn("νεότερη",blocked.text)
        self.assertEqual(self.card(parent_id)["stage"],"rejected")
        self.assertEqual(self.card(child_id)["correction_of_application_id"],parent_id)


    def test_a2_copy_preserves_verifiable_actual_procedure(self):
        from clinic_utilities.rf.persistence import record_legacy_procedure
        previous_id = record_legacy_procedure(self.engine, {
            "identity_number":"SYN-A2-REUSE",
            "indication_code":"KNEE_OA_KL34", "site_key":"knee",
            "laterality":"left", "exact_location":"Αριστερό γόνατο",
            "actual_procedure_date":"2026-05-01",
            "vas_before":8, "vas_after":3,
            "last_followup_date":"2026-06-01", "last_followup_vas":5
        })
        issued = self.issue(identity_number="SYN-A2-REUSE",pathway="A2",
            procedure_history_id=previous_id,
            a2_current_followup_date="2026-09-01",
            a2_current_followup_vas=6)
        self.assertEqual(issued.status_code, 200, issued.text)
        app_id=issued.headers["x-rf-application-id"]
        response=self.client.post(self.base + "/reuse/issued-copy",
            headers=self.headers,
            json={"application_id":app_id,"expected_revision":0})
        self.assertEqual(response.status_code,200,response.text)
        fields=response.json()["fields"]
        self.assertEqual(fields["pathway"],"A2")
        self.assertEqual(fields["a2_current_followup_date"],"2026-09-01")
        self.assertEqual(fields["a2_current_followup_vas"],6)
        self.assertEqual(fields["procedure_history_id"],previous_id)
        history=response.json()["previous_procedure"]
        self.assertIsNotNone(history)
        self.assertEqual(history["procedure_history_id"],previous_id)
        self.assertEqual(history["actual_procedure_date"],"2026-05-01")


class RFIntegrationSourceTests(unittest.TestCase):
    def test_main_mounts_native_router_not_gateway(self):
        source = Path("main.py").read_text(encoding="utf-8")
        self.assertIn("from clinic_utilities.rf.api import build_rf_router", source)
        self.assertIn("app.include_router(build_rf_router(engine))", source)
        self.assertNotIn("app.include_router(build_rf_gateway_router())", source)


if __name__ == "__main__":
    unittest.main()
