from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from clinic_utilities.rf.api import build_rf_router
from clinic_utilities.rf.parsers import parse_medications
from clinic_utilities.rf.persistence import (
    RFMedicationAliasORM,
    delete_medication_alias,
    initialize_rf_tables,
    list_medication_aliases,
    upsert_medication_alias,
)


def memory_engine():
    return create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


class RFMedicationLearningParserTests(unittest.TestCase):
    def test_unknown_line_stays_visible_with_nonpersistent_suggestion(self):
        result = parse_medications("Mysteron 50 mg 2 μήνες")
        self.assertEqual(result["nsaid_candidates"], [])
        self.assertEqual(result["other_candidates"], [])
        self.assertEqual(len(result["unrecognized_candidates"]), 1)
        row = result["unrecognized_candidates"][0]
        self.assertEqual(row["source_text"], "Mysteron 50 mg 2 μήνες")
        self.assertEqual(row["suggested_name"], "Mysteron")
        self.assertEqual(row["dose"], "50 mg")
        self.assertEqual(row["duration"], "2 μήνες")

    def test_clinician_learned_alias_is_recognized_on_next_parse(self):
        learned = [{
            "id": "L1",
            "normalized_alias": "mysteron",
            "alias": "Mysteron",
            "display_name": "Mysteron",
            "active_ingredient": "",
            "category": "nsaid",
            "provenance": "clinician_confirmed",
        }]
        result = parse_medications("Mysteron 50 mg 2 μήνες", learned)
        self.assertEqual(len(result["nsaid_candidates"]), 1)
        row = result["nsaid_candidates"][0]
        self.assertTrue(row["learned"])
        self.assertEqual(row["canonical_key"], "learned:mysteron")
        self.assertEqual(row["drug_name"], "Mysteron")
        self.assertEqual(result["unrecognized_candidates"], [])

    def test_built_in_mapping_has_precedence_over_conflicting_learned_alias(self):
        learned = [{
            "id": "L1",
            "normalized_alias": "narox",
            "alias": "Narox",
            "display_name": "Narox",
            "active_ingredient": "",
            "category": "other",
            "provenance": "clinician_confirmed",
        }]
        result = parse_medications("Narox 90 mg 10 ημέρες", learned)
        self.assertEqual(result["nsaid_candidates"][0]["canonical_key"], "etoricoxib")
        self.assertFalse(result["nsaid_candidates"][0]["learned"])
        self.assertEqual(result["other_candidates"], [])

    def test_learned_alias_matches_identity_prefix_not_arbitrary_interior_token(self):
        learned = [{
            "id": "L1",
            "normalized_alias": "mysteron",
            "alias": "Mysteron",
            "display_name": "Mysteron",
            "active_ingredient": "",
            "category": "nsaid",
            "provenance": "clinician_confirmed",
        }]
        correct = parse_medications("Mysteron XR 50 mg", learned)
        self.assertEqual(correct["nsaid_candidates"][0]["canonical_key"], "learned:mysteron")

        interior = parse_medications("OtherDrug Mysteron 50 mg", learned)
        self.assertEqual(interior["nsaid_candidates"], [])
        self.assertEqual(len(interior["unrecognized_candidates"]), 1)

    def test_parser_defensively_ignores_unsafe_stale_metadata_alias(self):
        learned = [{
            "id": "STALE",
            "normalized_alias": "xr",
            "alias": "XR",
            "display_name": "XR",
            "active_ingredient": "",
            "category": "nsaid",
            "provenance": "clinician_confirmed",
        }]
        parsed = parse_medications("Mysteron XR 50 mg", learned)
        self.assertEqual(parsed["nsaid_candidates"], [])
        self.assertEqual(parsed["other_candidates"], [])
        self.assertEqual(len(parsed["unrecognized_candidates"]), 1)

    def test_learned_alias_tolerates_leading_generic_form_marker_but_not_interior_match(self):
        learned = [{
            "id": "L2",
            "normalized_alias": "mysteron",
            "alias": "Mysteron",
            "display_name": "Mysteron",
            "active_ingredient": "",
            "category": "other",
            "provenance": "clinician_confirmed",
        }]
        parsed = parse_medications("Tablet Mysteron 50 mg", learned)
        self.assertEqual(parsed["other_candidates"][0]["canonical_key"], "learned:mysteron")


class RFMedicationLearningPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.engine = memory_engine()
        initialize_rf_tables(self.engine)

    def test_upsert_reclassify_and_delete_alias(self):
        first = upsert_medication_alias(
            self.engine, alias="Mysteron", category="nsaid", display_name="Mysteron"
        )
        self.assertEqual(first["category"], "nsaid")
        second = upsert_medication_alias(
            self.engine, alias="mysteron", category="other", display_name="Mysteron"
        )
        self.assertEqual(second["id"], first["id"])
        self.assertEqual(second["category"], "other")
        self.assertEqual(len(list_medication_aliases(self.engine)), 1)
        self.assertTrue(delete_medication_alias(self.engine, first["id"]))
        self.assertEqual(list_medication_aliases(self.engine), [])

    def test_dictionary_schema_does_not_store_source_dose_or_duration(self):
        upsert_medication_alias(self.engine, alias="Mysteron", category="nsaid")
        with Session(self.engine) as session:
            row = session.scalar(select(RFMedicationAliasORM))
            self.assertIsNotNone(row)
            columns = set(row.__table__.columns.keys())
        self.assertNotIn("source_text", columns)
        self.assertNotIn("dose", columns)
        self.assertNotIn("duration", columns)

    def test_alias_with_dose_is_rejected_for_data_minimization(self):
        with self.assertRaises(ValueError):
            upsert_medication_alias(self.engine, alias="Mysteron 50 mg", category="nsaid")

    def test_pure_numeric_and_generic_form_aliases_are_rejected(self):
        invalid_aliases = [
            "50",
            "50.0",
            "mg",
            "mcg",
            "tablet",
            "tablets",
            "δισκίο",
            "χάπια",
            "capsule",
            "syrup",
            "gel",
            "patch",
            "injection",
        ]
        for alias in invalid_aliases:
            with self.subTest(alias=alias):
                with self.assertRaises(ValueError):
                    upsert_medication_alias(self.engine, alias=alias, category="nsaid")

    def test_generic_token_cannot_poison_future_unknown_medication_matching(self):
        with self.assertRaises(ValueError):
            upsert_medication_alias(self.engine, alias="mg", category="nsaid")

        learned = list_medication_aliases(self.engine)
        self.assertEqual(learned, [])

        parsed = parse_medications("Mysteron 50 mg 2 μήνες", learned)
        self.assertEqual(parsed["nsaid_candidates"], [])
        self.assertEqual(parsed["other_candidates"], [])
        self.assertEqual(len(parsed["unrecognized_candidates"]), 1)
        self.assertEqual(
            parsed["unrecognized_candidates"][0]["suggested_name"],
            "Mysteron",
        )

    def test_valid_short_medication_names_are_not_rejected_by_generic_guard(self):
        entry = upsert_medication_alias(self.engine, alias="Xefo", category="nsaid")
        self.assertEqual(entry["normalized_alias"], "xefo")
        self.assertEqual(entry["category"], "nsaid")

    def test_release_route_frequency_metadata_only_aliases_are_rejected(self):
        invalid_aliases = [
            "XR",
            "SR",
            "MR",
            "PRN",
            "PO",
            "daily",
            "oral",
            "IV",
            "BID",
            "night",
            "forte",
        ]
        for alias in invalid_aliases:
            with self.subTest(alias=alias):
                with self.assertRaises(ValueError):
                    upsert_medication_alias(self.engine, alias=alias, category="nsaid")

    def test_valid_brand_plus_release_modifier_remains_learnable(self):
        entry = upsert_medication_alias(
            self.engine,
            alias="Mysteron XR",
            category="nsaid",
            display_name="Mysteron XR",
        )
        self.assertEqual(entry["normalized_alias"], "mysteron xr")
        parsed = parse_medications("Mysteron XR 50 mg", list_medication_aliases(self.engine))
        self.assertEqual(parsed["nsaid_candidates"][0]["canonical_key"], "learned:mysteron xr")

    def test_xr_poisoning_attempt_cannot_classify_future_unknown_line(self):
        with self.assertRaises(ValueError):
            upsert_medication_alias(self.engine, alias="XR", category="nsaid")
        self.assertEqual(list_medication_aliases(self.engine), [])

        parsed = parse_medications("Mysteron XR 50 mg", list_medication_aliases(self.engine))
        self.assertEqual(parsed["nsaid_candidates"], [])
        self.assertEqual(parsed["other_candidates"], [])
        self.assertEqual(len(parsed["unrecognized_candidates"]), 1)
        self.assertTrue(parsed["unrecognized_candidates"][0]["source_text"].startswith("Mysteron XR"))


class RFMedicationLearningApiTests(unittest.TestCase):
    def setUp(self):
        self.engine = memory_engine()
        self.env = patch.dict(os.environ, {"CLINICAL_DATA_KEY": "rf-learning-key"}, clear=False)
        self.env.start()
        app = FastAPI()
        app.include_router(build_rf_router(self.engine))
        self.client = TestClient(app)
        self.headers = {"X-Clinical-Key": "rf-learning-key"}

    def tearDown(self):
        self.client.close()
        self.env.stop()

    def test_dictionary_routes_are_protected(self):
        response = self.client.get("/clinical/clinic-utilities/rf/api/medication-dictionary")
        self.assertEqual(response.status_code, 401)

    def test_learn_parse_reclassify_delete_round_trip(self):
        learned = self.client.post(
            "/clinical/clinic-utilities/rf/api/medication-dictionary",
            headers=self.headers,
            json={"alias": "Mysteron", "display_name": "Mysteron", "category": "nsaid"},
        )
        self.assertEqual(learned.status_code, 200, learned.text)
        entry = learned.json()["entry"]

        parsed = self.client.post(
            "/clinical/clinic-utilities/rf/api/parse-medications",
            headers=self.headers,
            json={"text": "Mysteron 50 mg 2 μήνες"},
        )
        self.assertEqual(parsed.status_code, 200, parsed.text)
        self.assertEqual(parsed.json()["nsaid_candidates"][0]["canonical_key"], "learned:mysteron")
        self.assertTrue(parsed.json()["nsaid_candidates"][0]["learned"])

        changed = self.client.post(
            "/clinical/clinic-utilities/rf/api/medication-dictionary",
            headers=self.headers,
            json={"alias": "Mysteron", "display_name": "Mysteron", "category": "other"},
        )
        self.assertEqual(changed.status_code, 200)
        parsed_again = self.client.post(
            "/clinical/clinic-utilities/rf/api/parse-medications",
            headers=self.headers,
            json={"text": "Mysteron 50 mg 2 μήνες"},
        ).json()
        self.assertEqual(parsed_again["nsaid_candidates"], [])
        self.assertEqual(parsed_again["other_candidates"][0]["canonical_key"], "learned:mysteron")

        deleted = self.client.delete(
            f"/clinical/clinic-utilities/rf/api/medication-dictionary/{entry['id']}",
            headers=self.headers,
        )
        self.assertEqual(deleted.status_code, 200)
        unknown = self.client.post(
            "/clinical/clinic-utilities/rf/api/parse-medications",
            headers=self.headers,
            json={"text": "Mysteron 50 mg 2 μήνες"},
        ).json()
        self.assertEqual(len(unknown["unrecognized_candidates"]), 1)

    def test_api_rejects_patient_specific_dose_in_dictionary_alias(self):
        response = self.client.post(
            "/clinical/clinic-utilities/rf/api/medication-dictionary",
            headers=self.headers,
            json={"alias": "Mysteron 50 mg", "category": "nsaid"},
        )
        self.assertEqual(response.status_code, 422)

    def test_api_rejects_generic_alias_tokens(self):
        for alias in ("50", "mg", "mcg", "tablet", "δισκίο"):
            with self.subTest(alias=alias):
                response = self.client.post(
                    "/clinical/clinic-utilities/rf/api/medication-dictionary",
                    headers=self.headers,
                    json={"alias": alias, "category": "nsaid"},
                )
                self.assertEqual(response.status_code, 422, response.text)

    def test_api_rejects_release_route_frequency_metadata_aliases(self):
        for alias in ("XR", "SR", "MR", "PRN", "PO", "daily", "oral", "BID"):
            with self.subTest(alias=alias):
                response = self.client.post(
                    "/clinical/clinic-utilities/rf/api/medication-dictionary",
                    headers=self.headers,
                    json={"alias": alias, "category": "nsaid"},
                )
                self.assertEqual(response.status_code, 422, response.text)


if __name__ == "__main__":
    unittest.main()
