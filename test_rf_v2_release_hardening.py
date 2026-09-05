from __future__ import annotations

import unittest

from fastapi import HTTPException
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from clinic_utilities.rf.api import RFApplicationDraft, _resolve_medications
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


def draft_with_medication(text: str) -> RFApplicationDraft:
    return RFApplicationDraft(
        pathway="A1",
        patient_name="Synthetic Patient",
        identity_number="SYN-3PLUS3",
        gesy_number="GESY-SYN",
        age=67,
        product_key="medikey",
        indication_code="KNEE_OA_KL34",
        laterality="left",
        exact_location="Î‘ÏÎ¹ÏƒÏ„ÎµÏÏŒ Î³ÏŒÎ½Î±Ï„Î¿",
        full_medication_text=text,
    )


class RFThreePlusThreeTests(unittest.TestCase):
    def test_a1_rejects_fewer_than_three_plus_three(self):
        draft = draft_with_medication(
            "Arcoxia 90 mg Î³Î¹Î± 3 Î¼Î®Î½ÎµÏ‚\n"
            "Brufen 600 mg Î³Î¹Î± 2 Î¼Î®Î½ÎµÏ‚\n"
            "Panadol 1 g Î³Î¹Î± 3 Î¼Î®Î½ÎµÏ‚\n"
            "Tramadex 100 mg Î³Î¹Î± 1 Î¼Î®Î½Î±"
        )
        with self.assertRaises(HTTPException) as caught:
            _resolve_medications(draft)
        self.assertEqual(caught.exception.status_code, 422)
        self.assertIn("3 ÎœÎ£Î‘Î¦", caught.exception.detail)
        self.assertIn("3 Î¬Î»Î»Î±", caught.exception.detail)

    def test_a1_accepts_exact_three_plus_three(self):
        draft = draft_with_medication(
            "Arcoxia 90 mg Î³Î¹Î± 3 Î¼Î®Î½ÎµÏ‚\n"
            "Brufen 600 mg Î³Î¹Î± 2 Î¼Î®Î½ÎµÏ‚\n"
            "Voltaren 75 mg Î³Î¹Î± 1 Î¼Î®Î½Î±\n"
            "Panadol 1 g Î³Î¹Î± 3 Î¼Î®Î½ÎµÏ‚\n"
            "Parcoten 1 g Î³Î¹Î± 2 Î¼Î®Î½ÎµÏ‚\n"
            "Tramadex 100 mg Î³Î¹Î± 1 Î¼Î®Î½Î±"
        )
        nsaid, other = _resolve_medications(draft)
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
                "exact_location": "Î‘ÏÎ¹ÏƒÏ„ÎµÏÏŒ Î³ÏŒÎ½Î±Ï„Î¿",
                "product_key": "medikey",
                "full_medication_text": "RAW MEDICATION PASTE MUST NOT PERSIST",
                "physio_dates_text": "01/02/2026\n08/02/2026",
                "nsaid_trials": [
                    {"source_text": "Arcoxia raw line", "drug_name": "Etoricoxib", "dose": "90 mg", "duration": "3 Î¼Î®Î½ÎµÏ‚"}
                ],
                "other_analgesic_trials": [
                    {"source_text": "Panadol raw line", "drug_name": "Î Î±ÏÎ±ÎºÎµÏ„Î±Î¼ÏŒÎ»Î·", "dose": "1 g", "duration": "3 Î¼Î®Î½ÎµÏ‚"}
                ],
                "physio": {"start_date": "2026-02-01", "end_date": "2026-02-08", "treatment_count": 2},
            },
        )
        with Session(engine) as session:
            row = session.scalar(select(RFApplicationORM(¤¹İ¡•É”¡IÁÁ±¥…Ñ¥½¹=I4¹¥€ôô…ÁÁ±¥…Ñ¥½¹}¥¤¤(€€€€€€€€€€€Á…å±½…€ôÉ½Ü¹Á…å±½…‘}©Í½¸(€€€€€€€Í•±˜¹…ÍÍ•ÉÑ9½Ñ%¸ ‰™Õ±±}µ•‘¥…Ñ¥½¹}Ñ•áĞˆ°Á…å±½…¤(€€€€€€€Í•±˜¹…ÍÍ•ÉÑ9½Ñ%¸ ‰Á¡åÍ¥½}‘…Ñ•Í}Ñ•áĞˆ°Á…å±½…¤(€€€€€€€Í•±˜¹…ÍÍ•ÉÑ9½Ñ%¸ ‰Í½ÕÉ•}Ñ•áĞˆ°Á…å±½…‘l‰¹Í…¥‘}ÑÉ¥…±Ì‰ulÁt¤(€€€€€€€Í•±˜¹…ÍÍ•ÉÑ9½Ñ%¸ ‰Í½ÕÉ•}Ñ•áĞˆ°Á…å±½…‘l‰½Ñ¡•É}…¹…±•Í¥}ÑÉ¥…±Ì‰ulÁt¤(€€€€€€€Í•±˜¹…ÍÍ•ÉÑÅÕ…°¡Á…å±½…‘l‰Á¡åÍ¥¼‰ul‰ÑÉ•…Ñµ•¹Ñ}½Õ¹Ğ‰t°€È¤(()¥˜}}¹…µ•}|€ôô€‰}}µ…¥¹}|ˆè(€€€Õ¹¥ÑÑ•ÍĞ¹µ…¥¸ ¤(