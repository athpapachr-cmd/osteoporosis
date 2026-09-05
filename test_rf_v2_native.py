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

from clinic_utilities.rf.api import RFApplicationDraft, _validate_a1, build_rf_router
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
        self.assertEqual(arcoxia["duration"], "3 μηνες")

    def test_duplicate_brand_same_active_family_is_deduplicated(self):
        result = parse_medications("Tramadex 100 mg για 1 μήνα\nMabron 50 mg για 2 μήνες")
        tramadol = [item for item in result["other_candidates"] if item["canonical_key"] == "tramadol"]
        self.assertEqual(len(tramadol), 1)
        self.assertEqual(tramadol[0]["duration"], "2 μηνες")


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
        self.assertEqual(rows[0]["provenance"], "legacy_manual")


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
            "nsaid_trials": [{"drug_name": "Etoricoxib", "dose": "90 mg", "duration": "3 μηνες"}],
            "other_analgesic_trials": [{"drug_name": "Παρακεταμόλη", "dose": "1 g", "duration": "3 μηνες"}],
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


class RFIntegrationSourceTests(unittest.TestCase):
    def test_main_mounts_native_router_not_gateway(self):
        source = Path("main.py").read_text(encoding="utf-8")
        self.assertIn("from clinic_utilities.rf.api import build_rf_router", source)
        self.assertIn("app.include_router(build_rf_router(engine))", source)
        self.assertNotIn("app.include_router(build_rf_gateway_router())", source)


if __name__ == "__main__":
    unittest.main()
