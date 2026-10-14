"""Synthetic-only Native RF tracking v1 regressions (no patient/provider access)."""
from __future__ import annotations

from datetime import datetime, timedelta
import json
import os
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, inspect, insert, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from clinic_utilities.rf.api import build_rf_router
from clinic_utilities.rf.persistence import (
    RFApplicationORM, RFApplicationProgressORM, get_rf_draft,
    get_rf_source, initialize_rf_tables, save_rf_draft, validate_rf_correction_source,
)
from clinic_utilities.rf.tracking import (
    guarded_rejected_correction, tracking_search, tracking_week,
)

SYNTHETIC_KEY = "synthetic-rf-tracking-test-key"
API = "/clinical/clinic-utilities/rf"


def engine_in_memory():
    return create_engine("sqlite+pysqlite:///:memory:",
                         connect_args={"check_same_thread": False},
                         poolclass=StaticPool)


def synthetic_app(id_: str, *, at: datetime | None = None, name: str = "Συνθετικός Ασθενής",
                  identity: str = "SYN999") -> RFApplicationORM:
    return RFApplicationORM(
        id=id_, created_at=at or datetime(2026, 10, 14, 12, 0),
        patient_identity_key=identity, patient_name=name,
        gesy_number="SYN-GESY", patient_age=62, pathway="A1",
        indication_code="KNEE_OA_KL34", site_key="knee", laterality="left",
        exact_location="Synthetic site", product_key="medikey",
        payload_json={"patient_name": name, "identity_number": identity,
                      "gesy_number": "SYN-GESY", "age": 62,
                      "indication_code": "KNEE_OA_KL34", "product_key": "medikey",
                      "laterality": "left", "exact_location": "Synthetic site"},
    )


class RFTrackingSyntheticTest(unittest.TestCase):

    def setUp(self):
        self.patcher = patch.dict(os.environ, {"CLINICAL_DATA_KEY": SYNTHETIC_KEY})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.engine = engine_in_memory()
        initialize_rf_tables(self.engine)
        self.addCleanup(self.engine.dispose)

    def _with_rejected(self, *, id_: str = "attempt-a", revision: int = 3):
        with Session(self.engine) as session:
            session.add(synthetic_app(id_))
            session.add(RFApplicationProgressORM(
                application_id=id_, revision=revision,
                stage="rejected", decision="rejected", rejection_reason="Synthetic rejection",
                updated_at=datetime(2026, 10, 14, 12),
            ))
            session.commit()

    def test_atomic_selected_rejection_and_issue_guard(self):
        self._with_rejected()
        found = guarded_rejected_correction(
            self.engine, application_id="attempt-a", expected_revision=3)
        self.assertEqual(found["application_id"], "attempt-a")
        self.assertEqual(found["revision"], 3)
        self.assertEqual(found["fields"]["identity_number"], "SYN999")
        # The generic source remains byte/behavior compatible, but is NOT an eligibility proof.
        self.assertEqual(found["fields"],
                         get_rf_source(self.engine, "application", "attempt-a")["fields"])
        with self.assertRaises(LookupError):
            guarded_rejected_correction(self.engine, application_id="wrong-id", expected_revision=3)
        with self.assertRaises(RuntimeError):
            guarded_rejected_correction(self.engine, application_id="attempt-a", expected_revision=2)
        with Session(self.engine) as session:
            validate_rf_correction_source(
                session, rejected_id="attempt-a", expected_revision=3,
                identity_number="SYN999", indication_code="KNEE_OA_KL34",
                laterality="left", site_key="knee",
            )
            with self.assertRaises(ValueError):
                validate_rf_correction_source(
                    session, rejected_id="attempt-a", expected_revision=3,
                    identity_number="OTHER", indication_code="KNEE_OA_KL34",
                    laterality="left", site_key="knee",
                )
            with self.assertRaises(ValueError):
                validate_rf_correction_source(
                    session, rejected_id="attempt-a", expected_revision=None,
                    identity_number="SYN999", indication_code="KNEE_OA_KL34",
                    laterality="left", site_key="knee",
                )
        with Session(self.engine) as session:
            source = session.get(RFApplicationProgressORM, "attempt-a")
            source.revision = 4  # Concurrent outcome change after correct prefill.
            session.commit()
        with Session(self.engine) as session:
            with self.assertRaises(RuntimeError):
                validate_rf_correction_source(
                    session, rejected_id="attempt-a", expected_revision=3,
                    identity_number="SYN999", indication_code="KNEE_OA_KL34",
                    laterality="left", site_key="knee",
                )
        with Session(self.engine) as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(RFApplicationORM)), 1)

    def test_protected_post_read_and_missing_revision_fail_closed(self):
        self._with_rejected()
        app = FastAPI()
        app.include_router(build_rf_router(self.engine))
        client = TestClient(app)
        url = API + "/api/reuse/rejected-correction"
        self.assertEqual(client.post(url, json={"application_id":"attempt-a",
                                                "expected_revision":3}).status_code,401)
        headers = {"X-Clinical-Key": SYNTHETIC_KEY}
        success = client.post(url, json={"application_id":"attempt-a",
                                         "expected_revision":3}, headers=headers)
        self.assertEqual(success.status_code,200)
        self.assertEqual(success.headers["cache-control"],"no-store")
        self.assertEqual(success.json()["application_id"],"attempt-a")
        self.assertEqual(client.post(url, json={"application_id":"attempt-a",
                                                "expected_revision":2},headers=headers).status_code,409)
        self.assertEqual(client.post(url, json={"application_id":"missing",
                                                "expected_revision":3},headers=headers).status_code,404)
        self.assertEqual(client.post(url, json={"application_id":"attempt-a"},
                                     headers=headers).status_code,422)
        self.assertEqual(client.get(API + "/tracking",headers=headers).status_code,200)
        self.assertEqual(client.get(API + "/tracking").status_code,401)
        missing = {
            "pathway":"A1", "patient_name":"Synthetic Patient", "identity_number":"SYN999",
            "gesy_number":"SYN-GESY", "age":62, "product_key":"medikey",
            "indication_code":"KNEE_OA_KL34", "laterality":"left",
            "exact_location":"Synthetic site", "correction_of_application_id":"attempt-a",
        }
        issued = client.post(API + "/api/create", headers=headers,
                             data={"draft_json":json.dumps(missing)},
                             files={"imaging_report":("synthetic.pdf", b"invalid-but-not-read",
                                                      "application/pdf")})
        self.assertEqual(issued.status_code,422,issued.text)
        with Session(self.engine) as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(RFApplicationORM)),1)

    def test_week_dates_index_and_truthful_absence_of_appointment_time(self):
        self._with_rejected()
        with Session(self.engine) as session:
            progress = session.get(RFApplicationProgressORM,"attempt-a")
            progress.scheduled_date = "2026-10-12"
            progress.performed_date = "2026-10-13"
            session.commit()
        indexes = {x["name"] for x in inspect(self.engine).get_indexes("clinic_rf_application_progress")}
        self.assertIn("ix_rf_progress_scheduled_date", indexes)
        self.assertIn("ix_rf_progress_performed_date", indexes)
        week = tracking_week(self.engine,week_start="2026-10-12")
        self.assertTrue(week["complete"])
        self.assertFalse(week["has_more"])
        self.assertEqual(len(week["applications"]),1)
        self.assertEqual(week["applications"][0]["scheduled_date"],"2026-10-12")
        self.assertEqual(week["applications"][0]["performed_date"],"2026-10-13")
        self.assertEqual(week["booking_time_status"],"off_unverified_link")
        with self.assertRaises(ValueError):
            tracking_week(self.engine,week_start="2026-10-13")
        with self.assertRaises(ValueError):
            tracking_week(self.engine,week_start="2026-02-30")

    def test_signed_cursors_filter_binding_and_legacy_source(self):
        with Session(self.engine) as session:
            for i in range(45):
                session.add(synthetic_app("s-%03d" % i,
                            at=datetime(2026,10,14,12) - timedelta(minutes=i),
                            identity="SYN%03d" % i))
            session.commit()
        base = tracking_search(self.engine,query="συνθετικος",scope="all")
        self.assertEqual(len(base["applications"]),20)
        self.assertTrue(base["has_more"])
        self.assertFalse(base["scan_complete"])
        with self.assertRaises(ValueError):
            tracking_search(self.engine,query="other",cursor=base["next_cursor"])
        cursor = base["next_cursor"]
        with self.assertRaises(ValueError):
            tracking_search(self.engine,query="συνθετικος",
                            cursor=cursor[:-1] + ("A" if cursor[-1] != "A" else "B"))
        seen = {row["id"] for row in base["applications"]}
        while cursor:
            part = tracking_search(self.engine,query="συνθετικος",cursor=cursor)
            seen.update(row["id"] for row in part["applications"])
            cursor = part["next_cursor"]
        self.assertEqual(len(seen),45)
        self.assertTrue(tracking_search(self.engine,query="absent")["scan_complete"])
        # Existing generic reusable application endpoint still selects exactly one attempt.
        self.assertEqual(get_rf_source(self.engine,"application","s-044")["source"]["id"],"s-044")

    def test_saved_draft_never_rearms_legacy_correction(self):
        state = {"pathway":"A1","patient_name":"Synthetic Draft",
                 "identity_number":"SYN999","correction_of_application_id":"attempt-a"}
        draft = save_rf_draft(self.engine,state)
        reread = get_rf_draft(self.engine,draft["id"])
        self.assertNotIn("correction_of_application_id",reread["fields"])

    def test_cursor_progresses_past_both_legacy_limits(self):
        # Bounded batched insert/scan; older matches must survive 2k result and 10k scan caps.
        rows = []
        origin = datetime(2026,10,14,12)
        for i in range(10025):
            name = "Συνθετικός Ασθενής" if i < 2050 else (
                "ΣΥΝΘΕΤΙΚΟΣ ΆΛΦΑ" if i == 10012 else "OTHER SYNTHETIC")
            rows.append({
                "id":"bulk-%05d" % i,
                "created_at":origin - timedelta(minutes=i),
                "patient_identity_key":"SYN%05d" % i,
                "patient_name":name, "gesy_number":"SYN-%05d" % i,
                "patient_age":60, "pathway":"A1", "indication_code":"KNEE_OA_KL34",
                "site_key":"knee", "laterality":"left", "exact_location":"Synthetic",
                "product_key":"medikey", "payload_json":{"identity_number":"SYN%05d" % i},
            })
        with Session(self.engine) as session:
            session.execute(insert(RFApplicationORM), rows)
            session.commit()
        def exhaust(q):
            pages, ids, cursor = 0, [], ""
            while True:
                page = tracking_search(self.engine,query=q,cursor=cursor)
                pages += 1
                ids.extend(x["id"] for x in page["applications"])
                if not page["has_more"]:
                    self.assertTrue(page["scan_complete"])
                    break
                self.assertTrue(page["next_cursor"])
                cursor = page["next_cursor"]
                self.assertLess(pages,200)
            return ids
        common = exhaust("συνθετικοσ ασθενησ")
        self.assertEqual(len(common),2050)  # Beyond legacy 100 pages x 20.
        rare = exhaust("αλφα")
        self.assertEqual(rare,["bulk-10012"])  # Beyond legacy 10k scanned attempts.


if __name__ == "__main__":
    unittest.main()
