import os
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from clinical_data import build_clinical_router


class BaselineFinalizationApiTest(unittest.TestCase):
    def setUp(self):
        self.previous_key = os.environ.get("CLINICAL_DATA_KEY")
        os.environ["CLINICAL_DATA_KEY"] = "test-clinical-key"
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        app = FastAPI()
        app.include_router(build_clinical_router(engine))
        self.client = TestClient(app)
        self.headers = {"X-Clinical-Key": "test-clinical-key"}

    def tearDown(self):
        if self.previous_key is None:
            os.environ.pop("CLINICAL_DATA_KEY", None)
        else:
            os.environ["CLINICAL_DATA_KEY"] = self.previous_key

    def test_completed_payload_reload_noop_and_amendment(self):
        patient_id = "synthetic-pilot-patient"
        create_patient = self.client.post(
            "/clinical/patients",
            headers=self.headers,
            json={"patient_id": patient_id, "demographics": {}},
        )
        self.assertEqual(create_patient.status_code, 200)

        draft_payload = {
            "internal_uuid": "synthetic-case-1",
            "baseline_phase": "pilot",
            "step6": {"capture_quality": {"ready_for_audit": "yes"}},
        }
        create_encounter = self.client.post(
            f"/clinical/patient/{patient_id}/encounters",
            headers=self.headers,
            json={
                "encounter_date": "2026-08-30",
                "status": "draft",
                "payload": draft_payload,
            },
        )
        self.assertEqual(create_encounter.status_code, 200)
        encounter_id = create_encounter.json()["encounter_id"]

        final_payload = {
            **draft_payload,
            "step6": {
                "capture_quality": {"ready_for_audit": "yes"},
                "final_marker": "saved-before-server-finalization",
            },
            "pilot_completion": {
                "status": "complete",
                "completed_at": "2026-08-30T12:00:00Z",
            },
            "implementation_slice": "steps_1_6_pilot_complete",
        }
        finish = self.client.put(
            f"/clinical/encounter/{encounter_id}",
            headers=self.headers,
            json={
                "encounter_date": "2026-08-30",
                "status": "completed",
                "payload": final_payload,
            },
        )
        self.assertEqual(finish.status_code, 200)
        self.assertEqual(finish.json()["status"], "completed")
        self.assertEqual(finish.json()["payload"], final_payload)

        reload_response = self.client.get(
            f"/clinical/encounter/{encounter_id}",
            headers=self.headers,
        )
        self.assertEqual(reload_response.status_code, 200)
        self.assertEqual(reload_response.json()["status"], "completed")
        self.assertEqual(reload_response.json()["payload"], final_payload)

        noop_save = self.client.put(
            f"/clinical/encounter/{encounter_id}",
            headers=self.headers,
            json={
                "encounter_date": "2026-08-30",
                "status": "draft",
                "payload": final_payload,
            },
        )
        self.assertEqual(noop_save.status_code, 200)
        self.assertEqual(noop_save.json()["status"], "completed")

        amended_payload = {
            **final_payload,
            "step6": {
                **final_payload["step6"],
                "post_completion_edit": "material-change",
            },
        }
        amendment = self.client.put(
            f"/clinical/encounter/{encounter_id}",
            headers=self.headers,
            json={
                "encounter_date": "2026-08-30",
                "status": "draft",
                "payload": amended_payload,
            },
        )
        self.assertEqual(amendment.status_code, 200)
        self.assertEqual(amendment.json()["status"], "amended")
        self.assertEqual(amendment.json()["payload"], amended_payload)


if __name__ == "__main__":
    unittest.main()
