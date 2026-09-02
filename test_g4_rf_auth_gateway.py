from __future__ import annotations

import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from clinical_auth import ClinicalCookieMiddleware, build_auth_router
from clinic_utilities import rf_gateway


CLINICAL_KEY = "synthetic-clinical-key"
RF_KEY = "synthetic-rf-gateway-key"
VALID_APPLICATION_ID = "ABCDEFGHIJKLMNOP"


def upstream_response(
    status_code: int,
    content: bytes | str = b"",
    *,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    if isinstance(content, str):
        content = content.encode("utf-8")
    return httpx.Response(status_code=status_code, content=content, headers=headers or {})


class G4RFAuthGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env_patch = patch.dict(
            os.environ,
            {
                "CLINICAL_DATA_KEY": CLINICAL_KEY,
                rf_gateway.RF_GATEWAY_KEY_ENV: RF_KEY,
            },
            clear=False,
        )
        self.env_patch.start()

        app = FastAPI()
        app.add_middleware(ClinicalCookieMiddleware)
        app.include_router(build_auth_router())
        app.include_router(rf_gateway.build_rf_gateway_router())
        self.client = TestClient(app, base_url="https://testserver")

    def tearDown(self) -> None:
        self.client.close()
        self.env_patch.stop()

    def login(self) -> None:
        response = self.client.post("/clinical/login", json={"key": CLINICAL_KEY})
        self.assertEqual(response.status_code, 200)

    def test_gateway_requires_existing_clinical_auth(self) -> None:
        with patch.object(rf_gateway, "_send_upstream", new=AsyncMock()) as send:
            response = self.client.get(rf_gateway.RF_GATEWAY_PREFIX)
        self.assertEqual(response.status_code, 401)
        send.assert_not_awaited()

    def test_missing_rf_gateway_secret_fails_closed_before_upstream(self) -> None:
        self.login()
        with patch.dict(os.environ, {rf_gateway.RF_GATEWAY_KEY_ENV: ""}, clear=False):
            with patch.object(rf_gateway, "_send_upstream", new=AsyncMock()) as send:
                response = self.client.get(rf_gateway.RF_GATEWAY_PREFIX)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn(RF_KEY, response.text)
        send.assert_not_awaited()

    def test_form_uses_server_key_and_rewrites_browser_routes(self) -> None:
        self.login()
        form = (
            '<form method="post" action="/rf/create">'
            "<script>fetch('/rf/history?' + 'x')</script>"
            "</form>"
        )
        send = AsyncMock(
            return_value=upstream_response(
                200,
                form,
                headers={"content-type": "text/html; charset=utf-8"},
            )
        )
        with patch.object(rf_gateway, "_send_upstream", new=send):
            response = self.client.get(rf_gateway.RF_GATEWAY_PREFIX)

        self.assertEqual(response.status_code, 200)
        self.assertIn(f'action="{rf_gateway.RF_GATEWAY_PREFIX}/create"', response.text)
        self.assertIn(f"fetch('{rf_gateway.RF_GATEWAY_PREFIX}/history?", response.text)
        self.assertNotIn("https://ortho-reception-backend-v2.onrender.com", response.text)
        self.assertNotIn(RF_KEY, response.text)
        self.assertEqual(response.headers.get("cache-control"), "no-store")

        send.assert_awaited_once()
        _, kwargs = send.await_args
        self.assertEqual(kwargs["access_key"], RF_KEY)
        self.assertIsNone(kwargs["content_type"])
        self.assertIsNone(kwargs["content"])

        upstream_headers = rf_gateway._upstream_headers(RF_KEY)
        self.assertEqual(upstream_headers["X-RF-Key"], RF_KEY)
        self.assertNotIn("Cookie", upstream_headers)
        self.assertNotIn("Authorization", upstream_headers)
        self.assertNotIn("X-Clinical-Key", upstream_headers)

    def test_history_forwards_only_named_query_fields(self) -> None:
        self.login()
        send = AsyncMock(
            return_value=upstream_response(
                200,
                '{"found":false}',
                headers={"content-type": "application/json"},
            )
        )
        with patch.object(rf_gateway, "_send_upstream", new=send):
            response = self.client.get(
                rf_gateway.RF_GATEWAY_PREFIX + "/history",
                params={
                    "gesy_number": "G-1",
                    "identity_number": "I-1",
                    "application_location": "Γόνατο",
                    "unexpected": "must-not-forward",
                },
            )

        self.assertEqual(response.status_code, 200)
        _, kwargs = send.await_args
        self.assertEqual(
            kwargs["params"],
            {
                "gesy_number": "G-1",
                "identity_number": "I-1",
                "application_location": "Γόνατο",
            },
        )

    def test_create_forwards_multipart_and_rewrites_pdf_redirect(self) -> None:
        self.login()
        send = AsyncMock(
            return_value=upstream_response(
                303,
                headers={"location": f"/rf/pdf/{VALID_APPLICATION_ID}"},
            )
        )
        with patch.object(rf_gateway, "_send_upstream", new=send):
            response = self.client.post(
                rf_gateway.RF_GATEWAY_PREFIX + "/create",
                data={"patient_name": "Synthetic"},
                files={"radiology_report": ("report.pdf", b"%PDF-synthetic", "application/pdf")},
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(
            response.headers["location"],
            f"{rf_gateway.RF_GATEWAY_PREFIX}/pdf/{VALID_APPLICATION_ID}",
        )
        _, kwargs = send.await_args
        self.assertEqual(kwargs["access_key"], RF_KEY)
        self.assertTrue(kwargs["content_type"].lower().startswith("multipart/form-data"))
        self.assertIn(b"Synthetic", kwargs["content"])
        self.assertIn(b"%PDF-synthetic", kwargs["content"])

    def test_pdf_relays_binary_content_and_disposition(self) -> None:
        self.login()
        send = AsyncMock(
            return_value=upstream_response(
                200,
                b"%PDF-result",
                headers={
                    "content-type": "application/pdf",
                    "content-disposition": 'attachment; filename="rf.pdf"',
                },
            )
        )
        with patch.object(rf_gateway, "_send_upstream", new=send):
            response = self.client.get(
                f"{rf_gateway.RF_GATEWAY_PREFIX}/pdf/{VALID_APPLICATION_ID}"
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"%PDF-result")
        self.assertEqual(response.headers["content-type"], "application/pdf")
        self.assertEqual(response.headers["content-disposition"], 'attachment; filename="rf.pdf"')
        _, kwargs = send.await_args
        self.assertEqual(kwargs["path"], f"/rf/pdf/{VALID_APPLICATION_ID}")

    def test_invalid_pdf_id_never_reaches_upstream(self) -> None:
        self.login()
        with patch.object(rf_gateway, "_send_upstream", new=AsyncMock()) as send:
            response = self.client.get(rf_gateway.RF_GATEWAY_PREFIX + "/pdf/../../etc")
        self.assertIn(response.status_code, {404, 422})
        send.assert_not_awaited()

    def test_upstream_auth_failure_is_sanitized(self) -> None:
        self.login()
        send = AsyncMock(
            return_value=upstream_response(
                401,
                '{"detail":"RF_ACCESS_KEY mismatch secret-internal-detail"}',
                headers={"content-type": "application/json"},
            )
        )
        with patch.object(rf_gateway, "_send_upstream", new=send):
            response = self.client.get(rf_gateway.RF_GATEWAY_PREFIX)

        self.assertEqual(response.status_code, 502)
        self.assertNotIn("RF_ACCESS_KEY", response.text)
        self.assertNotIn("secret-internal-detail", response.text)

    def test_network_failure_is_sanitized(self) -> None:
        self.login()

        class FailingClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def request(self, *args, **kwargs):
                request = httpx.Request("GET", rf_gateway.RF_UPSTREAM_ORIGIN + "/rf")
                raise httpx.ConnectError("synthetic connection failure", request=request)

        with patch.object(rf_gateway.httpx, "AsyncClient", return_value=FailingClient()):
            response = self.client.get(rf_gateway.RF_GATEWAY_PREFIX)

        self.assertEqual(response.status_code, 502)
        self.assertNotIn("synthetic connection failure", response.text)


if __name__ == "__main__":
    unittest.main()
