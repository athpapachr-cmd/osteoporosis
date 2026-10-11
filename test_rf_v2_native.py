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
            "RF_DOCTOR_MAXOUTLOU_PROFILE_JSON": json.dumps(self.secondary),
            "RF_PRODUCT_CATALOG_JSON": PRODUCT_JSON,
        })
        self.env.start()
        app = FastAPI()
        app.include_router(build_rf_router(self.engine))
        self.client = TestClient(app)
        self.headers = {"X-Clinical-Key": CLINICAL_KEY}

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
        self.assertTrue(contract["doctors"]["maxoutlou"]["configured"])
        self.assertNotIn("doctor2@example.invalid", json.dumps(contract))
        for pathway in ("A1", "A2"):
            for key, expected in (("papachristou", json.loads(DOCTOR_JSON)),
                                  ("maxoutlou", self.secondary)):
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
                         ["papachristou", "maxoutlou", "papachristou", "maxoutlou"])
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
        for key in ("", None, "unknown", " maxoutlou"):
            with self.subTest(key=key):
                res = self._create(self._draft(doctor_key=key))
                self.assertEqual(res.status_code, 422, res.text[:300])
        self.assertEqual(len(self._rows()[0]), 0)
        with patch.dict(os.environ, {"RF_DOCTOR_MAXOUTLOU_PROFILE_JSON": ""}):
            res = self._create(self._draft(doctor_key="maxoutlou"))
            self.assertEqual(res.status_code, 503, res.text[:300])
        self.assertEqual(len(self._rows()[0]), 0)

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



class RFIntegrationSourceTests(unittest.TestCase):
    def test_main_mounts_native_router_not_gateway(self):
        source = Path("main.py").read_text(encoding="utf-8")
        self.assertIn("from clinic_utilities.rf.api import build_rf_router", source)
        self.assertIn("app.include_router(build_rf_router(engine))", source)
        self.assertNotIn("app.include_router(build_rf_gateway_router())", source)


if __name__ == "__main__":
    unittest.main()
