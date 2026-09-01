import os
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from clinical_data import build_clinical_router


class C2ServerAuthoritativeWorkspaceTest(unittest.TestCase):
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
        self.patient_id = "synthetic-c2-patient"
        created = self.client.post(
            "/clinical/patients",
            headers=self.headers,
            json={"patient_id": self.patient_id, "demographics": {}},
        )
        self.assertEqual(created.status_code, 200)

    def tearDown(self):
        if self.previous_key is None:
            os.environ.pop("CLINICAL_DATA_KEY", None)
        else:
            os.environ["CLINICAL_DATA_KEY"] = self.previous_key

    def create_draft(self):
        response = self.client.post(
            f"/clinical/patient/{self.patient_id}/encounters",
            headers=self.headers,
            json={
                "encounter_date": "2026-09-01",
                "status": "draft",
                "payload": {"internal_uuid": "synthetic-c2-visit", "value": "v1"},
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["encounter_id"])
        self.assertTrue(body["updated_at"])
        return body

    def test_matching_version_updates_and_stale_device_is_blocked(self):
        initial = self.create_draft()
        encounter_id = initial["encounter_id"]
        version_v1 = initial["updated_at"]

        device_a = self.client.put(
            f"/clinical/encounter/{encounter_id}",
            headers=self.headers,
            json={
                "encounter_date": "2026-09-01",
                "status": "draft",
                "payload": {"internal_uuid": "synthetic-c2-visit", "value": "device-a-v2"},
                "expected_updated_at": version_v1,
            },
        )
        self.assertEqual(device_a.status_code, 200)
        self.assertEqual(device_a.json()["payload"]["value"], "device-a-v2")
        version_v2 = device_a.json()["updated_at"]
        self.assertNotEqual(version_v1, version_v2)

        device_b_stale = self.client.put(
            f"/clinical/encounter/{encounter_id}",
            headers=self.headers,
            json={
                "encounter_date": "2026-09-01",
                "status": "draft",
                "payload": {"internal_uuid": "synthetic-c2-visit", "value": "device-b-stale"},
                "expected_updated_at": version_v1,
            },
        )
        self.assertEqual(device_b_stale.status_code, 409)
        self.assertIn("another device", device_b_stale.json()["detail"])

        latest = self.client.get(
            f"/clinical/encounter/{encounter_id}",
            headers=self.headers,
        )
        self.assertEqual(latest.status_code, 200)
        self.assertEqual(latest.json()["payload"]["value"], "device-a-v2")
        self.assertEqual(latest.json()["updated_at"], version_v2)

    def test_cross_device_reload_reads_latest_server_draft(self):
        initial = self.create_draft()
        encounter_id = initial["encounter_id"]
        updated = self.client.put(
            f"/clinical/encounter/{encounter_id}",
            headers=self.headers,
            json={
                "payload": {"internal_uuid": "synthetic-c2-visit", "value": "latest-server-state"},
                "status": "draft",
                "expected_updated_at": initial["updated_at"],
            },
        )
        self.assertEqual(updated.status_code, 200)

        device_b_reload = self.client.get(
            f"/clinical/encounter/{encounter_id}",
            headers=self.headers,
        )
        self.assertEqual(device_b_reload.status_code, 200)
        self.assertEqual(device_b_reload.json()["payload"]["value"], "latest-server-state")
        self.assertEqual(device_b_reload.json()["updated_at"], updated.json()["updated_at"])

    def test_legacy_no_token_update_remains_backward_compatible(self):
        initial = self.create_draft()
        response = self.client.put(
            f"/clinical/encounter/{initial['encounter_id']}",
            headers=self.headers,
            json={
                "payload": {"internal_uuid": "legacy-caller", "value": "legacy-write"},
                "status": "draft",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["payload"]["value"], "legacy-write")


if __name__ == "__main__":
    unittest.main()
