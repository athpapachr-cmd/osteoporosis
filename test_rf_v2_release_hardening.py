from __future__ import annotations

import unittest

from fastapi import HTTPException
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from clinic_utilities.rf.api import RFApplicationDraft, RFMedicationTrial, _resolve_medications
from clinic_utilities.rf.persistence import (
    RFApplicationORM,
    initialize_rf_tables,
    record_application,
)


def memory_engine():
    return create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def draft_with_manual_trials(nsaid_count: int, other_count: int) -> RFApplicationDraft:
    return RFApplicationDraft(
        pathway="A1",
        patient_name="Synthetic Patient",
        identity_number="SYN-3PLUS3",
        gesy_number="GESY-SYN",
        age=67,
        product_key="medikey",
        indication_code="KNEE_OA_KL34",
        laterality="left",
        exact_location="Left knee",
        full_medication_text="synthetic medication source",
        nsaid_trials=[
            RFMedicationTrial(drug_name=f"NSAID-{i}", dose="dose", duration="duration")
            for i in range(nsaid_count)
        ],
        other_analgesic_trials=[
            RFMedicationTrial(drug_name=f"OTHER-{i}", dose="dose", duration="duration")
            for i in range(other_count)
        ],
    )


class RFThreePlusThreeTests(unittest.TestCase):
    def test_a1_rejects_fewer_than_three_plus_three(self):
        draft = draft_with_manual_trials(2, 2)
        with self.assertRaises(HTTPException) as caught:
            _resolve_medications(draft)
        self.assertEqual(caught.exception.status_code, 422)

    def test_a1_accepts_exact_three_plus_three(self):
        nsaid, other = _resolve_medications(draft_with_manual_trials(3, 3))
        self.assertEqual(len(nsaid), 3)
        self.assertEqual(len(other), 3)


class RFApplicationDataMinimizationTests(unittest.TestCase):
    def test_raw_paste_and_source_lines_are_not_persisted(self):
        engine = memory_engine()
        initialize_rf_tables(engine)
        application_id = record_application(
            engine,
            {
                "identity_number": "SYN-RAW-1",
                "patient_name": "Synthetic Patient",
                "gesy_number": "GESY-SYN",
                "age": 67,
                "pathway": "A1",
                "indication_code": "KNEE_OA_KL34",
                "site_key": "knee",
                "laterality": "left",
                "exact_location": "Left knee",
                "product_key": "medikey",
                "full_medication_text": "RAW MEDICATION PASTE MUST NOT PERSIST",
                "physio_dates_text": "2026-02-01\n2026-02-08",
                "nsaid_trials": [
                    {
                        "source_text": "raw NSAID source line",
                        "drug_name": "NSAID-A",
                        "dose": "dose",
                        "duration": "duration",
                    }
                ],
                "other_analgesic_trials": [
                    {
                        "source_text": "raw analgesic source line",
                        "drug_name": "OTHER-A",
                        "dose": "dose",
                        "duration": "duration",
                    }
                ],
                "physio": {
                    "start_date": "2026-02-01",
                    "end_date": "2026-02-08",
                    "treatment_count": 2,
                },
            },
        )
        with Session(engine) as session:
            row = session.scalar(
                select(RFApplicationORM).where(RFApplicationORM.id == application_id)
            )
            self.assertIsNotNone(row)
            payload = row.payload_json

        self.assertNotIn("full_medication_text", payload)
        self.assertNotIn("physio_dates_text", payload)
        self.assertNotIn("source_text", payload["nsaid_trials"][0])
        self.assertNotIn("source_text", payload["other_analgesic_trials"][0])
        self.assertEqual(payload["physio"]["treatment_count"], 2)


if __name__ == "__main__":
    unittest.main()
